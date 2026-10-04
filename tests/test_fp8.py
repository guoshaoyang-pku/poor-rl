import pytest
import torch

from rlforge import fp8


def test_row_quantization_does_not_depend_on_batch():
    torch.manual_seed(3)
    x = torch.randn(7, 64)
    q, scale = fp8.quantize_rows(x)
    for i in range(len(x)):
        qi, si = fp8.quantize_rows(x[i:i + 1])
        assert torch.equal(q[i].float(), qi[0].float())
        assert torch.equal(scale[i], si[0])
    zeros, zs = fp8.quantize_rows(torch.zeros(2, 64))
    assert torch.isfinite(zs).all() and (zs > 0).all()
    assert not zeros.float().any()


def test_install_preserves_master_parameters_and_checkpoint_keys():
    model = torch.nn.Sequential(torch.nn.Linear(64, 80), torch.nn.Linear(80, 32))
    params = tuple(model.parameters())
    keys = tuple(model.state_dict())
    assert fp8.install(model, exclude=()) == ("0", "1")
    assert tuple(model.parameters()) == params
    assert tuple(model.state_dict()) == keys
    assert all(p.dtype == torch.float32 for p in model.parameters())
    with pytest.raises(ValueError, match="FP32 master"):
        fp8.install(torch.nn.Linear(64, 80).to(torch.bfloat16), exclude=())


_HOPPER = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


@pytest.mark.skipif(not _HOPPER, reason="Native FP8 requires Hopper or later")
def test_fp8_forward_backward_and_fp32_update():
    torch.manual_seed(11)
    model = torch.nn.Linear(64, 80, bias=False, device="cuda", dtype=torch.float32)
    ref = torch.nn.Linear(64, 80, bias=False, device="cuda", dtype=torch.float32)
    ref.load_state_dict(model.state_dict())
    fp8.install(model, exclude=())
    x = torch.randn(47, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    g = torch.randn(47, 80, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y, yr = model(x), ref(xr)
    y.backward(g)
    yr.backward(g)
    rel = (y.float() - yr.float()).norm() / yr.float().norm()
    cos = torch.nn.functional.cosine_similarity(
        model.weight.grad.flatten(), ref.weight.grad.flatten(), dim=0)
    assert rel < 0.05 and cos > 0.995
    assert model.weight.grad.dtype == torch.float32
    assert torch.isfinite(model.weight.grad).all()
    old_cache = model._rlforge_fp8_cache
    before = model.weight.detach().clone()
    torch.optim.AdamW(model.parameters(), lr=2e-6).step()
    assert (model.weight != before).any()
    assert model.weight.dtype == torch.float32
    with torch.no_grad():
        model(x)
    assert model._rlforge_fp8_cache is not old_cache


@pytest.mark.skipif(not _HOPPER, reason="Native FP8 requires Hopper or later")
def test_fused_quantizer_matches_reference_and_decode_batch():
    torch.manual_seed(8)
    x = torch.randn(47, 64, device="cuda", dtype=torch.bfloat16)
    q, scale = fp8.quantize_rows(x)
    qr, sr = fp8.quantize_rows(x.cpu())
    assert torch.equal(q.float().cpu(), qr.float())
    assert torch.allclose(scale.cpu(), sr, rtol=2e-7, atol=0)
    w = torch.randn(80, 64, device="cuda", dtype=torch.float32)
    wq, ws = fp8.quantize_rows(w)
    full = fp8.scaled_mm(q, wq, scale, ws)
    first = fp8.scaled_mm(q[:1], wq, scale[:1], ws)
    assert torch.allclose(full[:1].float(), first.float(), atol=0.0625, rtol=0.005)


@pytest.mark.skipif(not _HOPPER, reason="Native FP8 requires Hopper or later")
def test_fp8_chunked_head_matches_forward_and_updates_fp32_master():
    from rlforge.fast_logprob import logprob_entropy
    torch.manual_seed(23)
    hidden = torch.randn(73, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(160, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    weight._rlforge_fp8_head = True
    targets = torch.randint(0, 160, (73,), device="cuda")
    hq, hs = fp8.quantize_rows(hidden.detach())
    wq, ws = fp8.quantize_rows(weight.detach())
    logits = fp8.forward_mm(hq, wq, hs, ws).float()
    expected = logits.log_softmax(-1).gather(1, targets[:, None]).squeeze(1)
    lp, entropy = logprob_entropy(hidden, weight, None, targets, 1.0)
    assert torch.allclose(lp, expected, atol=1e-5, rtol=1e-5)
    (-lp.mean() + 0.005 * entropy.mean()).backward()
    assert weight.grad.dtype == torch.float32
    assert torch.isfinite(weight.grad).all() and torch.isfinite(hidden.grad).all()
    cache = weight._rlforge_fp8_head_cache
    torch.optim.AdamW([weight], lr=2e-6).step()
    with torch.no_grad():
        logprob_entropy(hidden, weight, None, targets, 1.0)
    assert weight._rlforge_fp8_head_cache is not cache


@pytest.mark.skipif(not _HOPPER, reason="Native FP8 requires Hopper or later")
def test_graph_replay_refreshes_inputs_preserves_results_and_bounds_cache(monkeypatch):
    fp8.clear_graph_cache()
    monkeypatch.setenv("RLFORGE_FP8_GRAPHS", "1")
    monkeypatch.setenv("RLFORGE_FP8_GRAPH_MAX_SHAPES", "1")
    monkeypatch.setenv("RLFORGE_FP8_GRAPH_MAX_MB", "16")
    try:
        first = torch.randn(512, 128, device="cuda")
        fn = lambda x: (x.square(),)
        output = fp8.graph_replay("square", fn, (first,))[0]
        saved = output.clone()
        second = torch.randn_like(first)
        torch.testing.assert_close(fp8.graph_replay("square", fn, (second,))[0], second.square())
        assert torch.equal(output, saved)
        fp8.graph_replay("square", fn, (torch.randn(513, 128, device="cuda"),))
        assert len(fp8._GRAPH_CACHE) == 1
        assert len(fp8._GRAPH_SKIPPED) == 1
        fp8.graph_replay("square", fn, (first,))
        assert len(fp8._GRAPH_CACHE) == 1
        assert fp8._GRAPH_CACHE_BYTES <= 16 * 2**20
    finally:
        fp8.clear_graph_cache()


@pytest.mark.skipif(not _HOPPER, reason="Native FP8 requires Hopper or later")
def test_fused_gate_up_preserves_fp32_parameters_and_updates(monkeypatch):
    torch.manual_seed(23)
    class Qwen3_5MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(64, 128, bias=False)
            self.up_proj = torch.nn.Linear(64, 128, bias=False)
            self.down_proj = torch.nn.Linear(128, 64, bias=False)
            self.act_fn = torch.nn.functional.silu

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    monkeypatch.setenv("RLFORGE_FP8_FUSE_MLP", "1")
    model = Qwen3_5MLP().cuda()
    parameters, keys = tuple(model.parameters()), tuple(model.state_dict())
    reference = Qwen3_5MLP().cuda()
    reference.load_state_dict(model.state_dict())
    fp8.install(model)
    assert tuple(model.parameters()) == parameters
    assert tuple(model.state_dict()) == keys
    x = torch.randn(73, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x) * 1e-4
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result, expected = model(x), reference(x)
    rel = (result.float() - expected.float()).norm() / expected.float().norm()
    assert rel < 0.1
    x2 = torch.randn_like(x).requires_grad_()
    monkeypatch.setenv("RLFORGE_FP8_GRAPHS", "0")
    eager_grads = torch.autograd.grad(model(x) + model(x2), (x, x2, *model.parameters()), dy)
    monkeypatch.setenv("RLFORGE_FP8_GRAPHS", "1")
    graph_grads = torch.autograd.grad(model(x) + model(x2), (x, x2, *model.parameters()), dy)
    for eager, graphed in zip(eager_grads, graph_grads):
        torch.testing.assert_close(eager, graphed, rtol=1e-5, atol=1e-6)
    expected.backward(dy)
    result.backward(dy)
    first_grads = [p.grad.detach().clone() for p in model.parameters()]
    assert all(torch.nn.functional.cosine_similarity(p.grad.flatten(), r.grad.flatten(), dim=0) > 0.99
               for p, r in zip(model.parameters(), reference.parameters()))
    assert all(p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all() for p in model.parameters())
    cached = model._rlforge_fp8_gate_up_cache
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)
    optimizer.step()
    assert all(torch.any(a != b) for a, b in zip(before, model.parameters()))
    model.zero_grad(set_to_none=True)
    model(x).backward(dy)
    assert model._rlforge_fp8_gate_up_cache is not cached
    assert all(optimizer.state[p]["exp_avg"].dtype == torch.float32 for p in model.parameters())
    assert all(torch.isfinite(grad).all() for grad in first_grads)
    fp8.clear_graph_cache()


@pytest.mark.skipif(not _HOPPER, reason="Native FP8 requires Hopper or later")
def test_scaled_mm_graph_survives_other_graph_eviction(monkeypatch):
    monkeypatch.setenv("RLFORGE_FP8_GRAPHS", "1")
    monkeypatch.setenv("RLFORGE_FP8_GRAPH_MAX_SHAPES", "2")
    monkeypatch.setenv("RLFORGE_FP8_GRAPH_MAX_MB", "512")
    fp8.clear_graph_cache()
    try:
        weight = torch.randn(256, 1024, device="cuda")
        wq, ws = fp8.quantize_rows(weight)
        operands = []
        for rows in (512, 513, 514, 515):
            xq, xs = fp8.quantize_rows(torch.randn(rows, 1024, device="cuda"))
            operands.append((xq, wq, xs, ws))

        def mm(xq, wq, xs, ws):
            return fp8.scaled_mm(xq, wq, xs, ws, out_dtype=torch.float32)

        for inputs in operands:
            torch.testing.assert_close(fp8.graph_replay("workspace", mm, inputs), mm(*inputs))
        for _ in range(4):
            for inputs in reversed(operands):
                torch.testing.assert_close(fp8.graph_replay("workspace", mm, inputs), mm(*inputs))
        torch.cuda.synchronize()
    finally:
        fp8.clear_graph_cache()
