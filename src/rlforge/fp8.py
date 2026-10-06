"""Hopper FP8 linear training with FP32 master parameters and gradients.

Forward quantization is stateless E4M3: one scale per activation token and
per output channel of the master weight. Forward uses vLLM's native CUTLASS
kernel and quantization contract. Backward uses E5M2 gradients and E4M3 operands; native FP8
GEMMs accumulate and return FP32. Norms, attention, and reductions are unchanged.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from types import MethodType

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = tl = None


_GRAPH_CACHE = OrderedDict()
_GRAPH_CACHE_BYTES = 0
_GRAPH_SKIPPED = OrderedDict()


class _KernelGraph:
    def __init__(self, fn, inputs):
        self.inputs = tuple(x.detach().clone() for x in inputs)
        self.stream = stream = torch.cuda.Stream(device=inputs[0].device)
        stream.wait_stream(torch.cuda.current_stream(inputs[0].device))
        with torch.cuda.stream(stream), torch.no_grad(), torch.autocast("cuda", enabled=False):
            for _ in range(2):
                fn(*self.inputs)
        torch.cuda.current_stream(inputs[0].device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        # Warm up cuBLAS on the capture stream so its workspace outlives graph eviction.
        with torch.cuda.graph(self.graph, stream=stream), torch.no_grad(), torch.autocast("cuda", enabled=False):
            outputs = fn(*self.inputs)
        self.single_output = isinstance(outputs, torch.Tensor)
        self.outputs = (outputs,) if self.single_output else tuple(outputs)
        if not all(isinstance(x, torch.Tensor) for x in self.outputs):
            raise TypeError("FP8 graph outputs must be tensors")
        self.bytes = sum(x.numel() * x.element_size() for x in (*self.inputs, *self.outputs))

    def __call__(self, inputs):
        for destination, source in zip(self.inputs, inputs):
            destination.copy_(source)
        self.graph.replay()
        outputs = tuple(x.clone() for x in self.outputs)
        return outputs[0] if self.single_output else outputs


def clear_graph_cache():
    """Release captured static tensors; call after changing the computation contract."""
    global _GRAPH_CACHE_BYTES
    _GRAPH_CACHE.clear()
    _GRAPH_SKIPPED.clear()
    _GRAPH_CACHE_BYTES = 0


def _skip_graph(key):
    _GRAPH_SKIPPED[key] = None
    while len(_GRAPH_SKIPPED) > 256:
        _GRAPH_SKIPPED.popitem(last=False)


def graph_replay(key, fn, inputs):
    """Replay stateless CUDA work with fresh outputs and a bounded cache shared across layers.

    ``key`` identifies the operation and any scalar options consumed by ``fn``. Every tensor
    input is refreshed on each call, including weights; callables must not capture tensor state.
    This helper belongs inside custom autograd functions and does not create an autograd graph.
    """
    global _GRAPH_CACHE_BYTES
    if (os.environ.get("RLFORGE_FP8_GRAPHS", "0") == "0" or not inputs
            or not all(x.is_cuda and x.device == inputs[0].device for x in inputs)
            or torch.cuda.is_current_stream_capturing()):
        return fn(*inputs)
    cache_key = (key, tuple((tuple(x.shape), tuple(x.stride()), x.dtype, x.device) for x in inputs))
    if cache_key in _GRAPH_SKIPPED:
        return fn(*inputs)
    max_shapes = int(os.environ.get("RLFORGE_FP8_GRAPH_MAX_SHAPES", "64"))
    max_bytes = int(float(os.environ.get("RLFORGE_FP8_GRAPH_MAX_MB", "1024")) * 2**20)
    if max_shapes <= 0 or max_bytes <= 0:
        return fn(*inputs)
    cached = _GRAPH_CACHE.pop(cache_key, None)
    if cached is None:
        input_bytes = sum(x.numel() * x.element_size() for x in inputs)
        if input_bytes * 2 > max_bytes:
            _skip_graph(cache_key)
            return fn(*inputs)
        with torch.cuda.device(inputs[0].device):
            before_allocated = torch.cuda.memory_allocated()
            before_reserved = torch.cuda.memory_reserved()
            cached = _KernelGraph(fn, inputs)
            cached.bytes = max(cached.bytes, torch.cuda.memory_allocated() - before_allocated,
                               torch.cuda.memory_reserved() - before_reserved)
        if cached.bytes > max_bytes:
            del cached
            _skip_graph(cache_key)
            return fn(*inputs)
        while _GRAPH_CACHE and (len(_GRAPH_CACHE) >= max_shapes
                               or _GRAPH_CACHE_BYTES + cached.bytes > max_bytes):
            evicted_key, oldest = _GRAPH_CACHE.popitem(last=False)
            _GRAPH_CACHE_BYTES -= oldest.bytes
            _skip_graph(evicted_key)
        _GRAPH_CACHE_BYTES += cached.bytes
    _GRAPH_CACHE[cache_key] = cached
    return cached(inputs)


if triton is not None:
    @triton.jit
    def _quantize_rows_kernel(X, Q, S, K: tl.constexpr, STRIDE: tl.constexpr,
                              LIMIT: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        x = tl.load(X + row * STRIDE + cols, cols < K, other=0).to(tl.float32)
        scale = tl.maximum(tl.div_rn(tl.max(tl.abs(x), 0), LIMIT), 1.0 / (LIMIT * 512))
        q = tl.minimum(tl.maximum(tl.div_rn(x, scale), -LIMIT), LIMIT)
        tl.store(Q + row * K + cols, q, cols < K)
        tl.store(S + row, scale)

    @triton.jit(do_not_specialize=["N"])
    def _amax_kernel(X, PARTIAL, N, BLOCK: tl.constexpr):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + idx, idx < N, other=0).to(tl.float32)
        tl.store(PARTIAL + tl.program_id(0), tl.max(tl.abs(x), 0))

    @triton.jit(do_not_specialize=["N"])
    def _amax_scale_kernel(PARTIAL, SCALE, N, LIMIT: tl.constexpr,
                           BLOCK: tl.constexpr):
        idx = tl.arange(0, BLOCK)
        x = tl.load(PARTIAL + idx, idx < N, other=0)
        tl.store(SCALE, tl.maximum(tl.max(x, 0), 1e-12) / LIMIT)

    @triton.jit(do_not_specialize=["M"])
    def _quantize_tensor_kernel(X, Q, S, M, N: tl.constexpr,
                                TRANSPOSE: tl.constexpr, LIMIT: tl.constexpr,
                                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        x = tl.load(X + rows[:, None] * N + cols[None, :], mask, other=0).to(tl.float32)
        scale = tl.load(S)
        q = tl.minimum(tl.maximum(x / scale, -LIMIT), LIMIT)
        if TRANSPOSE:
            tl.store(Q + cols[None, :] * M + rows[:, None], q, mask)
        else:
            tl.store(Q + rows[:, None] * N + cols[None, :], q, mask)

    @triton.jit(do_not_specialize=["M", "MP"])
    def _quantize_dual_kernel(X, Q, QT, S, M, N: tl.constexpr,
                              MP, LIMIT: tl.constexpr,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        MP = tl.multiple_of(MP, 16)
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        x = tl.load(X + rows[:, None] * N + cols[None, :],
                    (rows[:, None] < M) & (cols[None, :] < N), other=0).to(tl.float32)
        q = tl.minimum(tl.maximum(x / tl.load(S), -LIMIT), LIMIT)
        mask = (rows[:, None] < MP) & (cols[None, :] < N)
        tl.store(Q + rows[:, None] * N + cols[None, :], q, mask)
        # Triton 3.7 can corrupt a reused FP8 layout conversion with runtime MP.
        # Transpose in FP32 before the identity barrier and independent FP8 cast.
        qt = tl.inline_asm_elementwise(
            "mov.b32 $0, $1;", constraints="=f,f", args=[tl.trans(q)],
            dtype=tl.float32, is_pure=False, pack=1)
        tl.store(QT + cols[:, None] * MP + rows[None, :],
                 qt, tl.trans(mask))


def quantize_rows(x: torch.Tensor, dtype=torch.float8_e4m3fn):
    """Return row-major FP8 values and contiguous FP32 dequant scales [rows, 1]."""
    if x.ndim != 2:
        raise ValueError("FP8 row quantization requires a matrix")
    limit = torch.finfo(dtype).max
    if x.is_cuda and triton is not None and x.stride(1) == 1:
        q = torch.empty(x.shape, device=x.device, dtype=dtype)
        scale = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
        block = triton.next_power_of_2(x.shape[1])
        _quantize_rows_kernel[(x.shape[0],)](
            x, q, scale, x.shape[1], x.stride(0), limit, block,
            num_warps=4 if block <= 4096 else 8)
        return q, scale
    xf = x.float()
    scale = (xf.abs().amax(-1, keepdim=True) / limit).clamp_min(1.0 / (limit * 512))
    return (xf / scale).clamp(-limit, limit).to(dtype).contiguous(), scale.contiguous()


def _tensor_scale(x, limit):
    if x.is_cuda and triton is not None:
        blocks = triton.cdiv(x.numel(), 4096)
        partial = torch.empty(blocks, device=x.device, dtype=torch.float32)
        scale = torch.empty((), device=x.device, dtype=torch.float32)
        _amax_kernel[(blocks,)](x, partial, x.numel(), 4096)
        _amax_scale_kernel[(1,)](partial, scale, blocks, limit,
                                 triton.next_power_of_2(blocks))
        return scale
    return x.float().abs().amax().clamp_min(1e-12) / limit


def _quantize_tensor(x, dtype, transpose=False, scale=None):
    x = x.contiguous()
    limit = torch.finfo(dtype).max
    scale = _tensor_scale(x, limit) if scale is None else scale
    if x.is_cuda and triton is not None:
        shape = x.shape[::-1] if transpose else x.shape
        q = torch.empty(shape, device=x.device, dtype=dtype)
        _quantize_tensor_kernel[(triton.cdiv(x.shape[0], 32), triton.cdiv(x.shape[1], 64))](
            x, q, scale, x.shape[0], x.shape[1], transpose, limit, 32, 64)
        return q, scale
    q = (x.float() / scale).clamp(-limit, limit).to(dtype)
    return (q.t().contiguous() if transpose else q.contiguous()), scale


def _quantize_dual(x, dtype):
    x = x.contiguous()
    m, n = x.shape
    mp = (m + 15) // 16 * 16
    limit = torch.finfo(dtype).max
    scale = _tensor_scale(x, limit)
    if x.is_cuda and triton is not None:
        q = torch.empty((mp, n), device=x.device, dtype=dtype)
        qt = torch.empty((n, mp), device=x.device, dtype=dtype)
        _quantize_dual_kernel[(triton.cdiv(mp, 32), triton.cdiv(n, 64))](
            x, q, qt, scale, m, n, mp, limit, 32, 64)
        return q[:m], qt, scale
    xp = F.pad(x, (0, 0, 0, mp - m))
    q = (xp.float() / scale).clamp(-limit, limit).to(dtype)
    return q[:m], q.t().contiguous(), scale


def scaled_mm(a, b, scale_a, scale_b, *, out_dtype=torch.bfloat16):
    """Compute A @ B.T. Both FP8 operands are stored row-major."""
    if a.shape[1] != b.shape[1]:
        raise ValueError("FP8 GEMM contraction dimensions differ")
    if a.shape[1] % 16 or b.shape[0] % 16:
        raise ValueError("Native FP8 GEMM requires K and N divisible by 16")
    sb = scale_b.t().contiguous() if scale_b.ndim == 2 else scale_b
    # Torch's rowwise GEMM requires aligned scale pointers, including M=1 views.
    sa = scale_a.clone() if scale_a.data_ptr() % 16 else scale_a
    sb = sb.clone() if sb.data_ptr() % 16 else sb
    with torch.autocast("cuda", enabled=False):
        return torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb,
                                out_dtype=out_dtype, use_fast_accum=False)


def forward_mm(a, b, scale_a, scale_b):
    backend = os.environ.get("RLFORGE_FP8_FORWARD", "native")
    if backend == "torch":
        return scaled_mm(a, b, scale_a, scale_b)
    if backend != "native":
        raise ValueError("RLFORGE_FP8_FORWARD must be native or torch")
    from vllm import _custom_ops as ops
    return ops.cutlass_scaled_mm(a, b.t(), out_dtype=torch.bfloat16,
                                 scale_a=scale_a, scale_b=scale_b.t(), bias=None)


def _weight_cache(module):
    w = module.weight
    key = (w._version, w.data_ptr(), w.device, w.dtype)
    cached = getattr(module, "_rlforge_fp8_cache", None)
    if cached is None or cached[0] != key:
        with torch.no_grad():
            q, scale = quantize_rows(w.detach())
            qt, st = _quantize_tensor(w.detach(), torch.float8_e4m3fn, transpose=True)
        cached = (key, q, scale, qt, st)
        module._rlforge_fp8_cache = cached
    return cached[1:]


def _linear_backward(xc, xs, wtq, wts, g, *, need_x, need_w, x_dtype):
    gq, gtq, gs = _quantize_dual(g, torch.float8_e5m2)
    outputs = []
    if need_x:
        outputs.append(scaled_mm(gq, wtq, gs, wts, out_dtype=torch.float32).to(x_dtype))
    if need_w:
        pad = (-g.shape[0]) % 16
        xp = F.pad(xc, (0, 0, 0, pad)) if pad else xc
        xts = xs.amax()
        xtq, _ = _quantize_tensor(xp, torch.float8_e4m3fn, transpose=True, scale=xts)
        outputs.append(scaled_mm(gtq, xtq, gs, xts, out_dtype=torch.float32))
    return tuple(outputs)


class _FP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, wq, ws, wtq, wts):
        shape = x.shape
        xc = x.reshape(-1, shape[-1]).to(torch.bfloat16).contiguous()
        def forward(xc, wq, ws):
            xq, xs = quantize_rows(xc)
            return forward_mm(xq, wq, xs, ws), xs
        y, xs = graph_replay("linear_forward", forward, (xc, wq, ws))
        if bias is not None:
            y = (y.float() + bias.float()).to(torch.bfloat16)
        ctx.save_for_backward(xc, xs, wtq, wts)
        ctx.shape, ctx.x_dtype = shape, x.dtype
        ctx.bias_dtype = None if bias is None else bias.dtype
        return y.reshape(*shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad):
        xc, xs, wtq, wts = ctx.saved_tensors
        g = grad.reshape(-1, grad.shape[-1]).contiguous()
        need_x, need_w, need_b = ctx.needs_input_grad[:3]
        dx = dw = db = None
        if need_x or need_w:
            def backward(xc, xs, wtq, wts, g):
                return _linear_backward(xc, xs, wtq, wts, g, need_x=need_x,
                                        need_w=need_w, x_dtype=ctx.x_dtype)
            values = iter(graph_replay(("linear_backward", need_x, need_w, ctx.x_dtype),
                                       backward, (xc, xs, wtq, wts, g)))
            dx = next(values).reshape(ctx.shape) if need_x else None
            dw = next(values) if need_w else None
        if need_b:
            db = g.float().sum(0).to(ctx.bias_dtype)
        return dx, dw, db, None, None, None, None


def _gate_up_cache(module):
    gate, up = module.gate_proj.weight, module.up_proj.weight
    key = (gate._version, gate.data_ptr(), up._version, up.data_ptr(), gate.device, gate.dtype)
    cached = getattr(module, "_rlforge_fp8_gate_up_cache", None)
    if cached is None or cached[0] != key:
        with torch.no_grad():
            combined = torch.cat((gate.detach(), up.detach()), dim=0)
            q, scales = quantize_rows(combined)
            qt, scale_t = _quantize_tensor(combined, torch.float8_e4m3fn, transpose=True)
        cached = (key, q, scales, qt, scale_t)
        module._rlforge_fp8_gate_up_cache = cached
    return cached[1:]


class _FP8GateUp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate_weight, up_weight, wq, ws, wtq, wts):
        shape = x.shape
        xc = x.reshape(-1, shape[-1]).to(torch.bfloat16).contiguous()
        def forward(xc, wq, ws):
            xq, xs = quantize_rows(xc)
            return forward_mm(xq, wq, xs, ws), xs
        y, xs = graph_replay("linear_forward", forward, (xc, wq, ws))
        ctx.save_for_backward(xc, xs, wtq, wts)
        ctx.shape, ctx.x_dtype = shape, x.dtype
        ctx.gate_rows = gate_weight.shape[0]
        return y.reshape(*shape[:-1], gate_weight.shape[0] + up_weight.shape[0])

    @staticmethod
    def backward(ctx, grad):
        xc, xs, wtq, wts = ctx.saved_tensors
        g = grad.reshape(-1, grad.shape[-1]).contiguous()
        need_x, need_gate, need_up = ctx.needs_input_grad[:3]
        need_w = need_gate or need_up
        def backward(xc, xs, wtq, wts, g):
            return _linear_backward(xc, xs, wtq, wts, g, need_x=need_x,
                                    need_w=need_w, x_dtype=ctx.x_dtype)
        values = iter(graph_replay(("linear_backward", need_x, need_w, ctx.x_dtype),
                                   backward, (xc, xs, wtq, wts, g)))
        dx = next(values).reshape(ctx.shape) if need_x else None
        dw = next(values) if need_w else None
        dg = dw[:ctx.gate_rows] if need_gate else None
        du = dw[ctx.gate_rows:] if need_up else None
        return dx, dg, du, None, None, None, None


def _mlp_forward(self, x):
    packed = _FP8GateUp.apply(x, self.gate_proj.weight, self.up_proj.weight, *_gate_up_cache(self))
    gate, up = packed.chunk(2, dim=-1)
    return self.down_proj(self.act_fn(gate) * up)


def linear(x, weight, bias=None, *, cache=None):
    """FP8 linear with an FP32 differentiable master weight."""
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)[0] < 9:
        raise RuntimeError("Native FP8 training requires a CUDA GPU with capability >= 9.0")
    if x.shape[-1] != weight.shape[1] or weight.shape[0] % 16 or weight.shape[1] % 16:
        raise ValueError("FP8 linear input/weight dimensions must align and be multiples of 16")
    if cache is None:
        with torch.no_grad():
            wq, ws = quantize_rows(weight.detach())
            wtq, wts = _quantize_tensor(weight.detach(), torch.float8_e4m3fn, transpose=True)
    else:
        wq, ws, wtq, wts = cache
    return _FP8Linear.apply(x, weight, bias, wq, ws, wtq, wts)


def _linear_forward(self, x):
    return linear(x, self.weight, self.bias, cache=_weight_cache(self))


def register_optimizer_cache_hook(model, optimizer):
    """Invalidate quantized master-weight caches after fused or ordinary optimizer steps."""
    parameters = tuple(p for p in model.parameters() if p.requires_grad)

    def after_step(optimizer, args, kwargs):
        torch.autograd.graph.increment_version(parameters)

    return optimizer.register_step_post_hook(after_step)


def install(model, *, exclude=("lm_head", "in_proj_a", "in_proj_b"), require_fp32=True):
    """Patch aligned nn.Linear modules in place, preserving parameters and state_dict."""
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or any(name == n or name.endswith("." + n) for n in exclude):
            continue
        if any(part in name.split(".") for part in ("visual", "vision_model", "vision_tower")):
            continue
        if module.in_features % 16 or module.out_features % 16:
            continue
        if require_fp32 and module.weight.dtype != torch.float32:
            raise ValueError(f"{name}: FP8 training needs an FP32 master weight")
        module.forward = MethodType(_linear_forward, module)
        names.append(name)
    if not names:
        raise ValueError("No aligned nn.Linear modules were found for FP8")
    if os.environ.get("RLFORGE_FP8_FUSE_MLP", "1") != "0":
        patched = set(names)
        for name, module in model.named_modules():
            if type(module).__name__ not in ("Qwen3_5MLP", "Qwen3_5MoeMLP"):
                continue
            prefix = name + "." if name else ""
            if not all(prefix + n in patched for n in ("gate_proj", "up_proj", "down_proj")):
                continue
            if module.gate_proj.bias is not None or module.up_proj.bias is not None:
                continue
            if module.gate_proj.weight.shape != module.up_proj.weight.shape:
                continue
            module.forward = MethodType(_mlp_forward, module)
    return tuple(names)
