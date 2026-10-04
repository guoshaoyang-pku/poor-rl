from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from rlforge.fp8_alignment import _causal_conv, _causal_conv_core, _gated_rms_forward, _rms_forward, install


def test_gated_norm_matches_unrounded_fp32_formula_and_preserves_master_grad():
    torch.manual_seed(3)
    weight = torch.nn.Parameter(torch.randn(128))
    module = SimpleNamespace(weight=weight, variance_epsilon=1e-6)
    x = torch.randn(32, 128, dtype=torch.bfloat16, requires_grad=True)
    gate = torch.randn_like(x, requires_grad=True)
    output = _gated_rms_forward(module, x, gate)
    xf = x.float()
    expected = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
                * weight.bfloat16().float() * F.silu(gate.float())).bfloat16()
    assert torch.equal(output, expected)
    output.float().square().mean().backward()
    assert weight.grad.dtype == torch.float32
    assert torch.isfinite(weight.grad).all()


def test_causal_conv_rounds_products_before_fp32_sum():
    torch.manual_seed(7)
    x = torch.randn(1, 32, 100, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(32, 4))
    output = _causal_conv(x, weight, None, "silu")
    padded = F.pad(x.float(), (3, 0))
    expected = sum((padded[..., tap:tap + 100] * weight.bfloat16().float()[:, tap][None, :, None])
                   .bfloat16().float() for tap in range(4))
    assert torch.equal(output, F.silu(expected).bfloat16())
    output.float().square().mean().backward()
    assert weight.grad.dtype == torch.float32
    assert torch.isfinite(weight.grad).all()


def test_install_preserves_parameter_identity_and_state_dict():
    mq = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(16, 128)
            self.norm = mq.Qwen3_5RMSNorm(128)
            self.gated_norm = mq.Qwen3_5RMSNormGated(128)

        def get_input_embeddings(self):
            return self.embed_tokens

    model = Model()
    params = dict(model.named_parameters())
    state = {name: value.clone() for name, value in model.state_dict().items()}
    assert set(install(model)) == {"embed_tokens", "norm", "gated_norm"}
    assert all(dict(model.named_parameters())[name] is value for name, value in params.items())
    assert all(torch.equal(model.state_dict()[name], value) for name, value in state.items())
    assert model.embed_tokens(torch.tensor([[1, 2]])).dtype == torch.bfloat16
    x = torch.randn(10, 128, dtype=torch.bfloat16)
    assert model.norm(x).dtype == torch.bfloat16


def test_zero_centered_norm_uses_bf16_weight_then_fp32_offset():
    module = SimpleNamespace(weight=torch.nn.Parameter(torch.randn(128)), eps=1e-6)
    x = torch.randn(16, 128, dtype=torch.bfloat16)
    out = _rms_forward(module, x)
    xf = x.float()
    expected = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
                * (1 + module.weight.bfloat16().float())).bfloat16()
    assert torch.equal(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compilation gate")
@pytest.mark.parametrize("activation", [None, "silu"])
def test_compiled_conv_preserves_rounding_gradients_and_fp32_update(monkeypatch, activation):
    torch.manual_seed(17)
    x = torch.randn(2, 128, 193, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(128, 4, device="cuda"))
    inputs = (x, weight)
    dy = torch.randn_like(x)
    expected = _causal_conv_core(x, weight, None)
    if activation is not None:
        expected = F.silu(expected)
    expected = expected.to(x.dtype)
    expected_grads = torch.autograd.grad(expected, inputs, dy)
    monkeypatch.setenv("RLFORGE_FP8_ALIGN_COMPILE", "1")
    actual = _causal_conv(x, weight, None, activation)
    actual_grads = torch.autograd.grad(actual, inputs, dy)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        # Weight-gradient reductions may sum in a different order when fused.
        cosine = F.cosine_similarity(actual_grad.float().flatten(),
                                     expected_grad.float().flatten(), dim=0)
        assert cosine > 0.99999
        assert (actual_grad.float() - expected_grad.float()).norm() < 0.001 * expected_grad.float().norm()
    weight.grad = actual_grads[-1]
    optimizer = torch.optim.AdamW([weight], lr=2e-6)
    before = weight.detach().clone()
    optimizer.step()
    assert weight.dtype == weight.grad.dtype == torch.float32
    assert optimizer.state[weight]["exp_avg"].dtype == torch.float32
    assert optimizer.state[weight]["exp_avg_sq"].dtype == torch.float32
    assert not torch.equal(weight, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compilation gate")
def test_compiled_conv_keeps_each_bf16_tap_product(monkeypatch):
    torch.manual_seed(19)
    x = torch.randn(4, 2048, 769, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2048, 4, device="cuda")
    expected = _causal_conv_core(x, weight, None).to(x.dtype)
    monkeypatch.setenv("RLFORGE_FP8_ALIGN_COMPILE", "1")
    actual = _causal_conv(x, weight, None, None)
    assert torch.equal(actual, expected)
