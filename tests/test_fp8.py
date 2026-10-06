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


@pytest.mark.parametrize("offset", [0, 1, 2, 3])
def test_shared_forward_aligns_scale_views_without_changing_values(monkeypatch, offset):
    monkeypatch.setenv("RLFORGE_FP8_FORWARD", "torch")
    a = torch.zeros(1, 16, dtype=torch.float8_e4m3fn)
    b = torch.zeros(32, 16, dtype=a.dtype)
    activation_storage = torch.arange(8, dtype=torch.float32).reshape(-1, 1)
    weight_storage = torch.arange(40, dtype=torch.float32).reshape(-1, 1)
    sa = activation_storage[offset:offset + 1]
    sb = weight_storage[offset:offset + 32]
    output = torch.zeros(1, 32, dtype=torch.bfloat16)

    def scaled_mm(actual_a, actual_b, *, scale_a, scale_b, out_dtype, use_fast_accum):
        assert actual_a is a and actual_b.data_ptr() == b.data_ptr()
        assert actual_b.stride() == (1, 16)
        assert scale_a.data_ptr() % 16 == scale_b.data_ptr() % 16 == 0
        assert torch.equal(scale_a, sa) and torch.equal(scale_b, sb.t())
        assert out_dtype == torch.bfloat16 and use_fast_accum is False
        if offset == 0:
            assert scale_a is sa
        return output

    monkeypatch.setattr(torch, "_scaled_mm", scaled_mm)
    assert fp8.forward_mm(a, b, sa, sb) is output
    assert torch.equal(activation_storage, torch.arange(8).float().reshape(-1, 1))
    assert torch.equal(weight_storage, torch.arange(40).float().reshape(-1, 1))


def test_unknown_forward_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("RLFORGE_FP8_FORWARD", "unknown")
    with pytest.raises(ValueError, match="native or torch"):
        fp8.forward_mm(None, None, None, None)


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


@pytest.mark.parametrize("fused", [False, True])
def test_optimizer_hook_invalidates_master_version(fused):
    model = torch.nn.Linear(64, 80)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6, fused=fused)
    handle = fp8.register_optimizer_cache_hook(model, optimizer)
    versions = [p._version for p in model.parameters()]
    before = [p.detach().clone() for p in model.parameters()]
    model(torch.ones(3, 64)).sum().backward()
    optimizer.step()
    assert all(p._version > version for p, version in zip(model.parameters(), versions))
    assert all(not torch.equal(p, saved) for p, saved in zip(model.parameters(), before))
    assert all(p.dtype == p.grad.dtype == torch.float32 for p in model.parameters())
    assert all(optimizer.state[p]["exp_avg"].dtype == torch.float32 for p in model.parameters())
    handle.remove()


_HOPPER = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


@pytest.mark.skipif(not _HOPPER, reason="Native FP8 requires Hopper or later")
def test_shared_forward_scale_offsets_match_aligned_cuda_reference(monkeypatch):
    monkeypatch.setenv("RLFORGE_FP8_FORWARD", "torch")
    torch.manual_seed(37)
    x = torch.randn(7, 1024, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 1024, device="cuda", dtype=torch.float32)
    q, scales = fp8.quantize_rows(x)
    wq, ws = fp8.quantize_rows(weight)
    for index in range(4):
        row = q[index:index + 1]
        aligned = scales[index:index + 1].clone()
        expected = torch._scaled_mm(row, wq.t(), scale_a=aligned,
            scale_b=ws.t().contiguous(), out_dtype=torch.bfloat16, use_fast_accum=False)
        actual = fp8.forward_mm(row, wq, scales[index:index + 1], ws)
        assert torch.equal(actual, expected)
        assert actual.dtype == torch.bfloat16 and torch.isfinite(actual).all()


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
def test_tensor_quantizers_reuse_compilation_across_token_lengths():
    if fp8.triton is None:
        pytest.skip("Triton quantizers unavailable")
    torch.manual_seed(29)
    kernels = (fp8._amax_kernel, fp8._amax_scale_kernel,
               fp8._quantize_tensor_kernel, fp8._quantize_dual_kernel)
    device = torch.cuda.current_device()
    warm_sizes = None
    for rows in (33, 34, 35, 47, 48, 49, 63):
        x = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            q, scale = fp8._quantize_tensor(x, dtype)
            expected_scale = x.float().abs().amax().clamp_min(1e-12) / torch.finfo(dtype).max
            torch.testing.assert_close(scale, expected_scale, rtol=2e-7, atol=0)
            qt, st = fp8._quantize_tensor(x, dtype, transpose=True)
            qd, qtd, sd = fp8._quantize_dual(x, dtype)
            assert torch.equal(q.float(), qt.t().float())
            assert torch.equal(q.float(), qd.float())
            assert torch.equal(q.float(), qtd[:, :rows].t().float())
            assert not qtd[:, rows:].float().any()
            assert torch.equal(scale, st) and torch.equal(scale, sd)
            assert qd.is_contiguous() and qtd.is_contiguous()
        sizes = tuple(len(kernel.device_caches[device][0]) for kernel in kernels)
        if warm_sizes is None:
            warm_sizes = sizes
        else:
            assert sizes == warm_sizes, (rows, warm_sizes, sizes)


@pytest.mark.skipif(not _HOPPER, reason="Native FP8 requires Hopper or later")
@pytest.mark.parametrize("fused", [False, True])
def test_optimizer_hook_refreshes_linear_gate_up_and_head_caches(fused):
    from types import SimpleNamespace
    from rlforge.fast_logprob import _fp8_head_cache

    torch.manual_seed(29)
    model = torch.nn.ModuleDict({
        name: torch.nn.Linear(64, 80, bias=False, device="cuda")
        for name in ("linear", "gate", "up", "head")
    })
    packed = SimpleNamespace(gate_proj=model["gate"], up_proj=model["up"])
    fp8._weight_cache(model["linear"])
    fp8._gate_up_cache(packed)
    _fp8_head_cache(model["head"].weight)
    old = (model["linear"]._rlforge_fp8_cache, packed._rlforge_fp8_gate_up_cache,
           model["head"].weight._rlforge_fp8_head_cache)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6, fused=fused)
    handle = fp8.register_optimizer_cache_hook(model, optimizer)
    before = [p.detach().clone() for p in model.parameters()]
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()
    assert all(not torch.equal(p, saved) for p, saved in zip(model.parameters(), before))
    assert all(p.dtype == p.grad.dtype == torch.float32 for p in model.parameters())
    assert all(optimizer.state[p]["exp_avg"].dtype == torch.float32 for p in model.parameters())
    actual = (fp8._weight_cache(model["linear"]), fp8._gate_up_cache(packed),
              _fp8_head_cache(model["head"].weight))
    current = (model["linear"]._rlforge_fp8_cache, packed._rlforge_fp8_gate_up_cache,
               model["head"].weight._rlforge_fp8_head_cache)
    assert all(a is not b for a, b in zip(old, current))
    weights = (model["linear"].weight,
               torch.cat((model["gate"].weight, model["up"].weight)),
               model["head"].weight)
    for weight, cached in zip(weights, actual):
        q, scale = fp8.quantize_rows(weight.detach())
        assert torch.equal(cached[0].float(), q.float())
        assert torch.equal(cached[1], scale)
        qt, st = fp8._quantize_tensor(weight.detach(), torch.float8_e4m3fn, transpose=True)
        if isinstance(cached[2], list):
            actual_qt, actual_st = cached[2][0]
        else:
            actual_qt, actual_st = cached[2:]
        assert torch.equal(actual_qt.float(), qt.float())
        assert torch.equal(actual_st, st)
    handle.remove()


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
