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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compilation gate")
@pytest.mark.parametrize("gated", [False, True])
def test_compiled_norm_preserves_gradients_and_fp32_master(monkeypatch, gated):
    torch.manual_seed(23)
    weight = torch.nn.Parameter(torch.randn(128, device="cuda"))
    module = SimpleNamespace(weight=weight, eps=1e-6, variance_epsilon=1e-6)
    x = torch.randn(4096, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    gate = torch.randn_like(x, requires_grad=True)
    values = (x, gate, weight) if gated else (x, weight)

    def run():
        return _gated_rms_forward(module, x, gate) if gated else _rms_forward(module, x)

    monkeypatch.setenv("RLFORGE_FP8_ALIGN_POINTWISE", "0")
    expected = run()
    dy = torch.randn_like(expected)
    reference_grad = torch.autograd.grad(expected, values, dy)
    monkeypatch.setenv("RLFORGE_FP8_ALIGN_POINTWISE", "1")
    actual = run()
    actual_grad = torch.autograd.grad(actual, values, dy)
    assert torch.equal(actual, expected)
    for result, reference in zip(actual_grad, reference_grad):
        assert torch.isfinite(result).all()
        assert (result.float() - reference.float()).norm() < 0.001 * reference.float().norm()
        assert F.cosine_similarity(result.double().flatten(), reference.double().flatten(), dim=0) > 0.99999
    weight.grad = actual_grad[-1]
    optimizer = torch.optim.AdamW([weight], lr=2e-6)
    before = weight.detach().clone()
    optimizer.step()
    assert weight.dtype == weight.grad.dtype == torch.float32
    assert optimizer.state[weight]["exp_avg"].dtype == torch.float32
    assert optimizer.state[weight]["exp_avg_sq"].dtype == torch.float32
    assert not torch.equal(before, weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compilation gate")
def test_compiled_gdn_preparation_preserves_precision_boundaries(monkeypatch):
    from rlforge.fp8_alignment import _gdn_prepare

    torch.manual_seed(23)
    q = torch.randn(2, 193, 8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    b = torch.randn(2, 193, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    a = torch.randn_like(b, requires_grad=True)
    a_log = torch.nn.Parameter(torch.randn(16, device="cuda"))
    dt_bias = torch.nn.Parameter(torch.randn(16, device="cuda"))
    values = q, k, b, a, a_log, dt_bias
    monkeypatch.setenv("RLFORGE_FP8_ALIGN_POINTWISE", "0")
    expected = _gdn_prepare(*values)
    dy = tuple(torch.randn_like(value) for value in expected)
    reference_grad = torch.autograd.grad(expected, values, dy)
    monkeypatch.setenv("RLFORGE_FP8_ALIGN_POINTWISE", "1")
    actual = _gdn_prepare(*values)
    actual_grad = torch.autograd.grad(actual, values, dy)
    assert [value.dtype for value in actual] == [torch.bfloat16, torch.bfloat16, torch.float32, torch.float32]
    for result, reference in zip(actual, expected):
        assert torch.equal(result, reference)
    for result, reference in zip(actual_grad, reference_grad):
        assert torch.isfinite(result).all()
        assert (result.float() - reference.float()).norm() < 0.001 * reference.float().norm()
        assert F.cosine_similarity(result.float().flatten(), reference.float().flatten(), dim=0) > 0.99999
    assert actual_grad[-2].dtype == actual_grad[-1].dtype == torch.float32


_DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA compact backward gate")),
]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("shape", [(73, 128), (2, 97, 128), (2, 97, 8, 128)])
@pytest.mark.parametrize("gated", [False, True])
def test_compact_norm_preserves_forward_gradients_and_fp32_update(monkeypatch, device, shape, gated):
    torch.manual_seed(31)
    x = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
    gate = torch.randn_like(x, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(shape[-1], device=device))
    module = SimpleNamespace(weight=weight, eps=1e-6, variance_epsilon=1e-6)
    inputs = (x, gate, weight) if gated else (x, weight)
    monkeypatch.setenv("RLFORGE_FP8_ALIGN_POINTWISE", "1")

    def run():
        return _gated_rms_forward(module, x, gate) if gated else _rms_forward(module, x)

    monkeypatch.setenv("RLFORGE_FP8_ALIGN_BACKWARD", "0")
    expected = run()
    dy = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, inputs, dy)
    monkeypatch.setenv("RLFORGE_FP8_ALIGN_BACKWARD", "1")
    actual = run()
    actual_grads = torch.autograd.grad(actual, inputs, dy)
    assert torch.equal(actual, expected)
    for result, reference in zip(actual_grads, expected_grads):
        assert result.dtype == reference.dtype and torch.isfinite(result).all()
        assert (result.float() - reference.float()).norm() < 0.001 * reference.float().norm()
        assert F.cosine_similarity(result.double().flatten(), reference.double().flatten(), dim=0) > 0.99999
    weight.grad = actual_grads[-1]
    optimizer = torch.optim.AdamW([weight], lr=2e-6)
    before = weight.detach().clone()
    optimizer.step()
    assert weight.dtype == weight.grad.dtype == torch.float32
    assert optimizer.state[weight]["exp_avg"].dtype == torch.float32
    assert optimizer.state[weight]["exp_avg_sq"].dtype == torch.float32
    assert not torch.equal(weight, before)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("requires", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("used", ["both", "one"])
def test_compact_qk_norm_handles_independent_and_unused_gradients(monkeypatch, device, requires, used):
    from rlforge.fp8_alignment import _gdn_prepare

    torch.manual_seed(37)
    q = torch.randn(2, 97, 8, 128, device=device, dtype=torch.bfloat16, requires_grad=requires[0])
    k = torch.randn_like(q, requires_grad=requires[1])
    b = torch.randn(2, 97, 8, device=device, dtype=torch.bfloat16)
    a = torch.randn_like(b)
    a_log = torch.randn(8, device=device)
    bias = torch.randn_like(a_log)
    values = (q, k, b, a, a_log, bias)
    monkeypatch.setenv("RLFORGE_FP8_ALIGN_POINTWISE", "1")
    monkeypatch.setenv("RLFORGE_FP8_ALIGN_BACKWARD", "0")
    expected = _gdn_prepare(*values)
    monkeypatch.setenv("RLFORGE_FP8_ALIGN_BACKWARD", "1")
    actual = _gdn_prepare(*values)
    assert all(torch.equal(result, reference) for result, reference in zip(actual, expected))
    chosen = [i for i, needed in enumerate(requires) if needed]
    if used == "one":
        chosen = chosen[:1]
    dy = tuple(torch.randn_like(expected[i]) for i in chosen)
    inputs = tuple(value for value in (q, k) if value.requires_grad)
    expected_grads = torch.autograd.grad(tuple(expected[i] for i in chosen), inputs, dy, allow_unused=True)
    actual_grads = torch.autograd.grad(tuple(actual[i] for i in chosen), inputs, dy, allow_unused=True)
    for result, reference in zip(actual_grads, expected_grads):
        if reference is None:
            assert result is None
        else:
            assert result.dtype == reference.dtype and torch.isfinite(result).all()
            assert (result.float() - reference.float()).norm() < 0.001 * reference.float().norm()
            assert F.cosine_similarity(result.double().flatten(), reference.double().flatten(), dim=0) > 0.99999


@pytest.mark.parametrize("kind", ["rms", "gated_rms", "qk_norm"])
def test_compact_backward_reuses_graph_across_original_base_shapes(monkeypatch, kind):
    from rlforge import fp8_alignment as align

    graphs = []
    original = getattr(align, "_" + kind + "_backward")

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(original, dynamic=True, backend=backend)

    def checked(*args):
        views = (args[0], *args[2:]) if kind == "rms" else (args[0], args[1], *args[3:])
        if kind == "qk_norm":
            views = args
        assert all(value._base is None and value.is_contiguous() for value in views)
        return compiled(*args)

    monkeypatch.setattr(align, "_" + kind + "_backward", checked)
    torch.manual_seed(41)
    weight = torch.nn.Parameter(torch.randn(128))
    module = SimpleNamespace(weight=weight, eps=1e-6, variance_epsilon=1e-6)
    for shape, spacing in [((1, 17, 128), 1), ((2, 19, 128), 2),
                           ((3, 23, 128), 4), ((1, 17, 8, 128), 2),
                           ((3, 19, 8, 128), 4)]:
        storage_shape = (*shape[:-2], shape[-2] * spacing, shape[-1])
        x = torch.randn(storage_shape, dtype=torch.bfloat16)[..., ::spacing, :].requires_grad_()
        gate = torch.randn(storage_shape, dtype=torch.bfloat16)[..., ::spacing, :].requires_grad_()

        def run():
            if kind == "rms":
                return align._rms_forward(module, x), (x, weight)
            if kind == "gated_rms":
                return align._gated_rms_forward(module, x, gate), (x, gate, weight)
            return align._QKNorm.apply(x, gate), (x, gate)

        monkeypatch.setenv("RLFORGE_FP8_ALIGN_BACKWARD", "1")
        out, values = run()
        outputs = (out,) if isinstance(out, torch.Tensor) else out
        torch.autograd.grad(outputs, values, tuple(torch.randn_like(value) for value in outputs))
        assert len(graphs) == 1
