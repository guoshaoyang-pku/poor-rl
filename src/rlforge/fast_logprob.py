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


def _stats(h, wc, targets, logit_scale, inv_t):
    """(log_z, logprob, entropy) fp32 [N] for bf16 h [N,H] and bf16 head wc [V,H]."""
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
        for v0 in range(0, V, VOCAB_CHUNK):
            v1 = min(V, v0 + VOCAB_CHUNK)
            l = hc @ wc[v0:v1].t()  # bf16 out, fp32 accumulate (TRL: torch.mm(..., out=bf16 buf))
            in_chunk = (tg >= v0) & (tg < v1)
            loc = torch.clamp(tg - v0, 0, v1 - v0 - 1)
            m, se, xse, tl = _fwd_tile_c(l, m, se, xse, tl, loc, in_chunk, logit_scale, inv_t)
        lz = m + torch.log(se)
        log_z[t0:t1] = lz
        logp[t0:t1] = tl - lz
        ent[t0:t1] = lz - xse / se
    return log_z, logp, ent


class FastChunkedLogProb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, temperature, logit_scale):
        ctx.set_materialize_grads(False)
        cd = _compute_dtype(hidden)
        h = hidden.to(cd)
        wc = weight.to(cd)
        inv_t = 1.0 / temperature
        log_z, logp, ent = _stats(h, wc, targets, float(logit_scale), inv_t)
        ctx.save_for_backward(hidden, weight, targets, log_z, ent)
        ctx.cd, ctx.inv_t, ctx.logit_scale = cd, inv_t, float(logit_scale)
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
        wc = weight.to(cd)
        g = g_lp.float() if g_lp is not None else torch.zeros(N, device=dev)
        ge = g_ent.float() if g_ent is not None else None
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
            _, lp, ent = _stats(hidden.to(cd), weight.to(cd), targets, float(logit_scale), 1.0 / temperature)
        return lp, ent
    return FastChunkedLogProb.apply(hidden, weight, targets, temperature, logit_scale)
