"""[v3_2] Faster chunked per-token log-prob / entropy for a large-vocab LM head (drop-in for TRL's
``trl.trainer.utils._ChunkedLogProbFunction`` as used by ``rlforge.prefix_share._logprob_fn``).

Why: Qwen3.5-0.8B has a 248k x 1024 head. The profile of one v3_1 micro-batch row (H200, 81k shared tokens,
policy fwd + bwd + KL-ref fwd) puts ~47% of the CUDA time in TRL's chunked log-prob: the backward's two
``grad_logits @ W`` / ``grad_logits.T @ h`` GEMMs run in plain FP32 (SIMT/xmma sgemm, no tensor cores) and every
[2048 x 8192] logits tile goes through ~8 separate fp32 elementwise passes, in the forward of the policy, the
forward of the KL reference and the backward recompute.

What (same math, same operand rounding):
  * logits tile = bf16 GEMM of bf16 hidden x bf16 head (identical to TRL: mm into a bf16 buffer), the head cast to
    bf16 ONCE per call instead of once per (token chunk, vocab chunk);
  * the online logsumexp / entropy / target-logit update and the backward's ``g * (onehot - p)`` tile are single
    fused kernels (torch.compile / Inductor) reading the bf16 tile once;
  * backward GEMMs: operands are the same values as TRL's (bf16-exact hidden and head upcast, fp32 grad tile);
    ``RLFORGE_LOGPROB_BWD`` = ``tf32`` (default: TF32 tensor cores, grad tile rounded to 10 mantissa bits, fp32
    accumulate/output), ``fp32`` (TRL's exact kernel class) or ``bf16`` (grad tile rounded to bf16).
Values: the forward differs from TRL only by fp32 summation order inside a tile (~1e-7 relative).
Set ``RLFORGE_FAST_LOGPROB=0`` to fall back to TRL's function.
"""
from __future__ import annotations

import os

import torch

TOK_CHUNK = int(os.environ.get("RLFORGE_LOGPROB_TOK_CHUNK", "4096"))
VOCAB_CHUNK = int(os.environ.get("RLFORGE_LOGPROB_VOCAB_CHUNK", "16384"))
BWD_MODE = os.environ.get("RLFORGE_LOGPROB_BWD", "tf32")
_COMPILE = os.environ.get("RLFORGE_LOGPROB_COMPILE", "1") != "0"


def _fwd_tile(l, m, se, xse, tl, tgt_local, in_chunk, logit_scale: float, inv_t: float):
    x = l.float() * logit_scale * inv_t  # same two multiplies as TRL (mul_(logit_scale); mul_(inv_t))
    cmax = x.amax(dim=-1)
    mn = torch.maximum(m, cmax)
    r = torch.exp(m - mn)
    e = torch.exp(x - mn.unsqueeze(-1))
    se2 = se * r + e.sum(dim=-1)
    xse2 = xse * r + (e * x).sum(dim=-1)
    t = torch.gather(x, 1, tgt_local.unsqueeze(1)).squeeze(1)
    tl2 = tl + torch.where(in_chunk, t, torch.zeros_like(t))
    return mn, se2, xse2, tl2


def _bwd_tile(l, log_z, g, g_ent, ent, tgt_local, in_chunk, logit_scale: float, inv_t: float, out_dtype: torch.dtype):
    x = l.float() * logit_scale * inv_t
    lp = x - log_z.unsqueeze(-1)
    p = torch.exp(lp)
    gl = (-g).unsqueeze(-1) * p
    if g_ent is not None:
        gl = gl + (-g_ent).unsqueeze(-1) * p * (lp + ent.unsqueeze(-1))
    cols = torch.arange(l.shape[1], device=l.device).unsqueeze(0)
    onehot = (cols == tgt_local.unsqueeze(1)) & in_chunk.unsqueeze(1)
    gl = gl + torch.where(onehot, g.unsqueeze(-1), torch.zeros_like(gl))
    gl = gl * inv_t * logit_scale
    return gl.to(out_dtype)


if _COMPILE:
    _fwd_tile_c = torch.compile(_fwd_tile, dynamic=True)
    _bwd_tile_c = torch.compile(_bwd_tile, dynamic=True)
else:
    _fwd_tile_c, _bwd_tile_c = _fwd_tile, _bwd_tile


def _compute_dtype(h):
    return torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else h.dtype


def _fp8_head_cache(weight):
    from rlforge.fp8 import _quantize_tensor, quantize_rows
    key = (weight._version, weight.data_ptr(), VOCAB_CHUNK)
    cached = getattr(weight, "_rlforge_fp8_head_cache", None)
    if cached is None or cached[0] != key:
        quantized, scales = quantize_rows(weight.detach())
        backward = []
        for v0 in range(0, weight.shape[0], VOCAB_CHUNK):
            backward.append(_quantize_tensor(weight.detach()[v0:v0 + VOCAB_CHUNK],
                                              torch.float8_e4m3fn, transpose=True))
        cached = key, quantized, scales, backward
        weight._rlforge_fp8_head_cache = cached
    return cached[1:]


def _stats(h, wc, targets, logit_scale, inv_t, fp8_cache=None, *, graph=True):
    """(log_z, logprob, entropy) fp32 [N] for bf16 h [N,H] and bf16 head wc [V,H]."""
    if fp8_cache is not None and graph and os.environ.get("RLFORGE_FP8_HEAD_GRAPH", "0") == "1":
        from rlforge.fp8 import graph_replay
        wq, ws, _ = fp8_cache

        def forward(h, targets, weight, scales):
            return _stats(h, weight, targets, logit_scale, inv_t, (weight, scales, ()), graph=False)

        return graph_replay(("head_forward", logit_scale, inv_t, TOK_CHUNK, VOCAB_CHUNK),
                            forward, (h, targets, wq, ws))
    N = h.shape[0]
    V = wc.shape[0]
    dev = h.device
    log_z = torch.empty(N, device=dev, dtype=torch.float32)
    logp = torch.empty(N, device=dev, dtype=torch.float32)
    ent = torch.empty(N, device=dev, dtype=torch.float32)
    for t0 in range(0, N, TOK_CHUNK):
        t1 = min(N, t0 + TOK_CHUNK)
        hc = h[t0:t1]
        tg = targets[t0:t1]
        n = t1 - t0
        m = torch.full((n,), float("-inf"), device=dev, dtype=torch.float32)
        se = torch.zeros(n, device=dev, dtype=torch.float32)
        xse = torch.zeros(n, device=dev, dtype=torch.float32)
        tl = torch.zeros(n, device=dev, dtype=torch.float32)
        if fp8_cache is not None:
            from rlforge.fp8 import forward_mm, quantize_rows
            hq, hs = quantize_rows(hc)
            wq, ws, _ = fp8_cache
        for v0 in range(0, V, VOCAB_CHUNK):
            v1 = min(V, v0 + VOCAB_CHUNK)
            l = (hc @ wc[v0:v1].t() if fp8_cache is None else
                 forward_mm(hq, wq[v0:v1], hs, ws[v0:v1]))
            in_chunk = (tg >= v0) & (tg < v1)
            loc = torch.clamp(tg - v0, 0, v1 - v0 - 1)
            m, se, xse, tl = _fwd_tile_c(l, m, se, xse, tl, loc, in_chunk, logit_scale, inv_t)
        lz = m + torch.log(se)
        log_z[t0:t1] = lz
        logp[t0:t1] = tl - lz
        ent[t0:t1] = lz - xse / se
    return log_z, logp, ent


def _fp8_backward(h, targets, log_z, ent, g, ge, fp8_cache, need_h, need_w, ls, inv_t, *, graph=True):
    from rlforge.fp8 import _quantize_dual, _quantize_tensor, forward_mm, quantize_rows, scaled_mm
    wq, ws, backward_weights = fp8_cache
    if graph and os.environ.get("RLFORGE_FP8_HEAD_GRAPH", "0") == "1":
        from rlforge.fp8 import graph_replay
        has_entropy = ge is not None
        inputs = (h, targets, log_z, ent, g, *(() if ge is None else (ge,)), wq, ws,
                  *(value for pair in backward_weights for value in pair))

        def backward(*values):
            hv, tv, zv, ev, gv = values[:5]
            offset = 6 if has_entropy else 5
            gev = values[5] if has_entropy else None
            qv, sv = values[offset:offset + 2]
            bw = tuple(zip(values[offset + 2::2], values[offset + 3::2]))
            dh, dw = _fp8_backward(hv, tv, zv, ev, gv, gev, (qv, sv, bw),
                                   need_h, need_w, ls, inv_t, graph=False)
            return tuple(v for v in (dh, dw) if v is not None)

        outputs = iter(graph_replay(("head_backward", has_entropy, need_h, need_w, ls, inv_t,
                                    TOK_CHUNK, VOCAB_CHUNK), backward, inputs))
        return next(outputs) if need_h else None, next(outputs) if need_w else None
    n, hidden = h.shape
    vocab = wq.shape[0]
    grad_h = torch.zeros(n, hidden, device=h.device, dtype=torch.float32) if need_h else None
    grad_w = torch.zeros(vocab, hidden, device=h.device, dtype=torch.float32) if need_w else None
    for t0 in range(0, n, TOK_CHUNK):
        t1 = min(n, t0 + TOK_CHUNK)
        hc, tg = h[t0:t1], targets[t0:t1]
        hq, hs = quantize_rows(hc)
        if need_w:
            pad = (-(t1 - t0)) % 16
            hp = torch.nn.functional.pad(hc, (0, 0, 0, pad)) if pad else hc
            htq, hts = _quantize_tensor(hp, torch.float8_e4m3fn, transpose=True)
        for vi, v0 in enumerate(range(0, vocab, VOCAB_CHUNK)):
            v1 = min(vocab, v0 + VOCAB_CHUNK)
            l = forward_mm(hq, wq[v0:v1], hs, ws[v0:v1])
            in_chunk = (tg >= v0) & (tg < v1)
            loc = torch.clamp(tg - v0, 0, v1 - v0 - 1)
            gl = _bwd_tile_c(l, log_z[t0:t1], g[t0:t1], None if ge is None else ge[t0:t1],
                             ent[t0:t1], loc, in_chunk, ls, inv_t, torch.float32)
            gq, gtq, gs = _quantize_dual(gl, torch.float8_e5m2)
            if need_h:
                wtq, wts = backward_weights[vi]
                grad_h[t0:t1] += scaled_mm(gq, wtq, gs, wts, out_dtype=torch.float32)
            if need_w:
                grad_w[v0:v1] += scaled_mm(gtq, htq, gs, hts, out_dtype=torch.float32)
    return grad_h, grad_w


class FastChunkedLogProb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, temperature, logit_scale):
        ctx.set_materialize_grads(False)
        cd = _compute_dtype(hidden)
        h = hidden.to(cd)
        fp8_cache = _fp8_head_cache(weight) if getattr(weight, "_rlforge_fp8_head", False) else None
        wc = weight if fp8_cache is not None else weight.to(cd)
        inv_t = 1.0 / temperature
        log_z, logp, ent = _stats(h, wc, targets, float(logit_scale), inv_t, fp8_cache)
        ctx.save_for_backward(hidden, weight, targets, log_z, ent)
        ctx.cd, ctx.inv_t, ctx.logit_scale = cd, inv_t, float(logit_scale)
        ctx.fp8_cache = fp8_cache
        return logp, ent

    @staticmethod
    def backward(ctx, g_lp, g_ent):
        hidden, weight, targets, log_z, ent = ctx.saved_tensors
        cd, inv_t, ls = ctx.cd, ctx.inv_t, ctx.logit_scale
        need_h, need_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        if g_lp is None and g_ent is None:
            return None, None, None, None, None
        N, H = hidden.shape
        V = weight.shape[0]
        dev = hidden.device
        h = hidden.to(cd)
        wc = weight if ctx.fp8_cache is not None else weight.to(cd)
        g = g_lp.float() if g_lp is not None else torch.zeros(N, device=dev)
        ge = g_ent.float() if g_ent is not None else None
        if ctx.fp8_cache is not None:
            grad_h, grad_w = _fp8_backward(h, targets, log_z, ent, g, ge, ctx.fp8_cache,
                                          need_h, need_w, ls, inv_t)
            return (grad_h.to(hidden.dtype) if need_h else None,
                    grad_w.to(weight.dtype) if need_w else None, None, None, None)
        mode = BWD_MODE
        gemm_dtype = torch.bfloat16 if mode == "bf16" else torch.float32
        hf = h if mode == "bf16" else None
        wf = wc if mode == "bf16" else None
        grad_h = torch.zeros(N, H, device=dev, dtype=torch.float32) if need_h else None
        grad_w = torch.zeros(V, H, device=dev, dtype=torch.float32) if need_w else None
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = mode == "tf32"
        try:
            for t0 in range(0, N, TOK_CHUNK):
                t1 = min(N, t0 + TOK_CHUNK)
                hc = h[t0:t1]
                hcf = hc if mode == "bf16" else hc.float()
                tg = targets[t0:t1]
                for v0 in range(0, V, VOCAB_CHUNK):
                    v1 = min(V, v0 + VOCAB_CHUNK)
                    wv = wc[v0:v1]
                    l = hc @ wv.t()
                    in_chunk = (tg >= v0) & (tg < v1)
                    loc = torch.clamp(tg - v0, 0, v1 - v0 - 1)
                    gl = _bwd_tile_c(l, log_z[t0:t1], g[t0:t1], None if ge is None else ge[t0:t1],
                                     ent[t0:t1], loc, in_chunk, ls, inv_t, gemm_dtype)
                    wvf = wv if mode == "bf16" else wv.float()
                    if need_h:
                        if mode == "bf16":
                            grad_h[t0:t1] += (gl @ wvf).float()
                        else:
                            grad_h[t0:t1].addmm_(gl, wvf)
                    if need_w:
                        if mode == "bf16":
                            grad_w[v0:v1] += (gl.t() @ hcf).float()
                        else:
                            grad_w[v0:v1].addmm_(gl.t(), hcf)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        return (grad_h.to(hidden.dtype) if need_h else None,
                grad_w.to(weight.dtype) if need_w else None, None, None, None)


def logprob_entropy(hidden, weight, bias, targets, temperature, logit_scale=1.0, softcap=None):
    """(logprob, entropy) fp32 [N]. Same contract as TRL's _ChunkedLogProbFunction.apply for bias=None and no
    softcapping (Qwen3.5); other heads must use TRL's function."""
    if bias is not None or softcap is not None:
        raise NotImplementedError("fast_logprob: bias / final_logit_softcapping not supported")
    if not torch.is_grad_enabled() or not (hidden.requires_grad or weight.requires_grad):
        with torch.no_grad():
            cd = _compute_dtype(hidden)
            fp8_cache = _fp8_head_cache(weight) if getattr(weight, "_rlforge_fp8_head", False) else None
            wc = weight if fp8_cache is not None else weight.to(cd)
            _, lp, ent = _stats(hidden.to(cd), wc, targets, float(logit_scale),
                               1.0 / temperature, fp8_cache)
        return lp, ent
    return FastChunkedLogProb.apply(hidden, weight, targets, temperature, logit_scale)
