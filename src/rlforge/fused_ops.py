"""[v3_2] Fused elementwise ops for the Qwen3.5 trainer forward/backward (trainer side only; vLLM untouched).

Profile of a v3_2 micro-batch row after fast_logprob (H200): the GatedDeltaNet causal depthwise conv runs through
transformers' torch fallback (``causal_conv1d`` is not installed) as an fp32 ``conv_depthwise2d`` (~10% of CUDA
time fwd+bwd), and RMSNorm / gated RMSNorm / SwiGLU are chains of separate fp32 elementwise kernels (copy_/mul/...).
Each replacement below is the SAME expression as the transformers code, compiled by Inductor into one kernel
(fp32 where the original is fp32, identical casts); only the fp32 rounding of the fused arithmetic can differ.

``install()`` patches (env RLFORGE_FUSED_OPS, comma list or "all"; default off):
  conv     modeling_qwen3_5.causal_conv1d_fn (fallback path): x.to(w.dtype) -> 4-tap causal conv -> silu -> x.dtype
  norm     Qwen3_5RMSNorm.forward and Qwen3_5RMSNormGated.forward
  mlp      Qwen3_5MLP.forward: act(gate) * up fused (the three Linear GEMMs are unchanged)
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

_INSTALLED: set = set()
_ORIG: dict = {}
_PATCHED: dict = {}


def _conv_core(x, w, activation: bool):
    # x [B, C, L] any dtype, w [C, K] (fp32 master under autocast); identical to the fallback:
    # F.conv1d(x.to(w.dtype), w[:, None], padding=K-1, groups=C)[..., :L] -> silu -> .to(x.dtype)
    K = w.shape[1]
    L = x.shape[2]
    xf = F.pad(x.to(w.dtype), (K - 1, 0))
    out = xf[:, :, 0:L] * w[:, 0:1]
    for k in range(1, K):
        out = out + xf[:, :, k:k + L] * w[:, k:k + 1]
    if activation:
        out = F.silu(out)
    return out.to(x.dtype)


_conv_c = None


def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
    global _conv_c
    if bias is not None or activation not in (None, "silu", "swish"):
        raise NotImplementedError("fused conv: bias / activation not supported")
    if _conv_c is None:
        _conv_c = torch.compile(_conv_core, dynamic=True)
    return _conv_c(hidden_states, weight, activation is not None)


def _rms_fwd(x, weight, eps: float):
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    out = out * (1.0 + weight.float())
    return out.type_as(x)


def _rmsg_fwd(h, gate, weight, eps: float):
    dt = h.dtype
    hf = h.to(torch.float32)
    var = hf.pow(2).mean(-1, keepdim=True)
    hf = hf * torch.rsqrt(var + eps)
    hf = weight * hf.to(dt)
    hf = hf * F.silu(gate.to(torch.float32))
    return hf.to(dt)


def _swiglu(g, u):
    return F.silu(g) * u


_c = {}


def _compiled(name, fn):
    if name not in _c:
        _c[name] = torch.compile(fn, dynamic=True)
    return _c[name]


def install(spec: str | None = None):
    spec = os.environ.get("RLFORGE_FUSED_OPS", "") if spec is None else spec
    parts = {"conv", "norm", "mlp"} if spec.strip() == "all" else {p.strip() for p in spec.split(",") if p.strip()}
    if not parts:
        return set()
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    _ORIG.setdefault("conv", (mq, "causal_conv1d_fn", mq.causal_conv1d_fn))
    _ORIG.setdefault("rms", (mq.Qwen3_5RMSNorm, "forward", mq.Qwen3_5RMSNorm.forward))
    _ORIG.setdefault("rmsg", (mq.Qwen3_5RMSNormGated, "forward", mq.Qwen3_5RMSNormGated.forward))
    _ORIG.setdefault("mlp", (mq.Qwen3_5MLP, "forward", mq.Qwen3_5MLP.forward))
    if "conv" in parts and "conv" not in _INSTALLED:
        mq.causal_conv1d_fn = causal_conv1d_fn
        _INSTALLED.add("conv")
    if "norm" in parts and "norm" not in _INSTALLED:
        def rms_forward(self, x):
            return _compiled("rms", _rms_fwd)(x, self.weight, self.eps)

        def rmsg_forward(self, hidden_states, gate):
            return _compiled("rmsg", _rmsg_fwd)(hidden_states, gate, self.weight, self.variance_epsilon)

        mq.Qwen3_5RMSNorm.forward = rms_forward
        mq.Qwen3_5RMSNormGated.forward = rmsg_forward
        _INSTALLED.add("norm")
    if "mlp" in parts and "mlp" not in _INSTALLED:
        def mlp_forward(self, x):
            if self.config.hidden_act != "silu":
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            return self.down_proj(_compiled("swiglu", _swiglu)(self.gate_proj(x), self.up_proj(x)))

        mq.Qwen3_5MLP.forward = mlp_forward
        _INSTALLED.add("mlp")
    for k, (obj, attr, _) in _ORIG.items():
        _PATCHED[k] = getattr(obj, attr)
    print(f"[rlforge] fused_ops installed: {sorted(_INSTALLED)}", flush=True)
    return set(_INSTALLED)


def set_enabled(flag: bool):
    """Test helper (v32_gate): swap the original transformers functions back in (False) or the patched ones (True)."""
    for k, (obj, attr, orig) in _ORIG.items():
        setattr(obj, attr, _PATCHED.get(k, orig) if flag else orig)
