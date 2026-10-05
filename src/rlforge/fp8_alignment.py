"""Qwen3.5 BF16 activation boundaries shared with vLLM FP8 serving.

Parameter objects and state_dict entries remain FP32. Optional compact norm
backward recomputes FP32 intermediates and returns gradients to the FP32 masters.
Attention and recurrent kernels still differ between the two backends.
"""
from __future__ import annotations

import os
from types import MethodType

import torch
import torch.nn.functional as F

_COMPILED_CONV = None
_COMPILED_RMS = None
_COMPILED_GATED_RMS = None
_COMPILED_GDN_PREP = None
_COMPILED_RMS_BACKWARD = None
_COMPILED_GATED_RMS_BACKWARD = None
_COMPILED_QK_BACKWARD = None


def _embedding_forward(self, input):
    return F.embedding(input, self.weight, self.padding_idx, self.max_norm,
                       self.norm_type, self.scale_grad_by_freq, self.sparse).to(torch.bfloat16)


def _rms_core(xf, inv_rms, weight, dtype):
    out = xf * inv_rms
    return (out * (1.0 + weight.to(torch.bfloat16).float())).to(dtype)


def _rms_value(x, weight, eps):
    global _COMPILED_RMS
    xf = x.float()
    inv_rms = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    if x.is_cuda and os.environ.get("RLFORGE_FP8_ALIGN_POINTWISE", "0") == "1":
        if _COMPILED_RMS is None:
            _COMPILED_RMS = torch.compile(
                _rms_core, dynamic=True, options={"emulate_precision_casts": True})
        out = _COMPILED_RMS(xf.reshape(-1, x.shape[-1]), inv_rms.reshape(-1, 1),
                            weight, x.dtype)
        return out.reshape(x.shape), inv_rms
    return _rms_core(xf, inv_rms, weight, x.dtype), inv_rms


def _rms_backward(x, weight, inv_rms, grad):
    xf, gf = x.float(), grad.float()
    gn = gf * (1.0 + weight.to(torch.bfloat16).float())
    gr = (gn * xf).sum(-1, keepdim=True)
    gv = gr * -0.5 * inv_rms.pow(3)
    dx = (gn * inv_rms + gv.expand_as(xf) / xf.shape[-1] * 2.0 * xf).to(x.dtype)
    dw = (gf * (xf * inv_rms)).sum(tuple(range(x.ndim - 1)))
    return dx, dw.to(torch.bfloat16).to(weight.dtype)


class _RMS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        ctx.set_materialize_grads(False)
        out, inv_rms = _rms_value(x, weight, eps)
        ctx.save_for_backward(x, weight, inv_rms)
        return out

    @staticmethod
    def backward(ctx, grad):
        global _COMPILED_RMS_BACKWARD
        if grad is None:
            return None, None, None
        x, weight, inv_rms = ctx.saved_tensors
        backward = _rms_backward
        if x.is_cuda:
            if _COMPILED_RMS_BACKWARD is None:
                _COMPILED_RMS_BACKWARD = torch.compile(
                    backward, dynamic=True, options={"emulate_precision_casts": True})
            backward = _COMPILED_RMS_BACKWARD
        # Canonical inputs avoid guards on original view bases and Q/K slice strides.
        dx, dw = backward(x.reshape(-1, x.shape[-1]).detach().contiguous(), weight,
                          inv_rms.reshape(-1, 1).detach().contiguous(), grad.reshape(-1, x.shape[-1]).detach().contiguous())
        return dx.reshape(x.shape) if ctx.needs_input_grad[0] else None, dw if ctx.needs_input_grad[1] else None, None


def _rms_forward(self, x):
    if os.environ.get("RLFORGE_FP8_ALIGN_BACKWARD", "0") == "1":
        return _RMS.apply(x, self.weight, self.eps)
    return _rms_value(x, self.weight, self.eps)[0]


def _gated_rms_core(xf, inv_rms, gate_silu, weight, dtype):
    out = xf * inv_rms
    out = out * weight.to(torch.bfloat16).float()
    return (out * gate_silu).to(dtype)


def _gated_rms_value(x, gate, weight, eps):
    global _COMPILED_GATED_RMS
    xf = x.float()
    inv_rms = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    gate_silu = F.silu(gate.float())
    if x.is_cuda and os.environ.get("RLFORGE_FP8_ALIGN_POINTWISE", "0") == "1":
        if _COMPILED_GATED_RMS is None:
            _COMPILED_GATED_RMS = torch.compile(
                _gated_rms_core, dynamic=True, options={"emulate_precision_casts": True})
        return _COMPILED_GATED_RMS(xf, inv_rms, gate_silu, weight, x.dtype), inv_rms
    return _gated_rms_core(xf, inv_rms, gate_silu, weight, x.dtype), inv_rms


def _gated_rms_backward(x, gate, weight, inv_rms, grad):
    xf, zf, gf = x.float(), gate.float(), grad.float()
    wf = weight.to(torch.bfloat16).float()
    norm = xf * inv_rms
    gw = gf * F.silu(zf)
    gn = gw * wf
    gr = (gn * xf).sum(-1, keepdim=True)
    gv = gr * -0.5 * inv_rms.pow(3)
    dx = (gn * inv_rms + gv.expand_as(xf) / xf.shape[-1] * 2.0 * xf).to(x.dtype)
    dw = (gw * norm).sum(tuple(range(x.ndim - 1))).to(torch.bfloat16).to(weight.dtype)
    dz = torch.ops.aten.silu_backward(gf * (norm * wf), zf).to(gate.dtype)
    return dx, dz, dw


class _GatedRMS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, weight, eps):
        ctx.set_materialize_grads(False)
        out, inv_rms = _gated_rms_value(x, gate, weight, eps)
        ctx.save_for_backward(x, gate, weight, inv_rms)
        return out

    @staticmethod
    def backward(ctx, grad):
        global _COMPILED_GATED_RMS_BACKWARD
        if grad is None:
            return None, None, None, None
        x, gate, weight, inv_rms = ctx.saved_tensors
        backward = _gated_rms_backward
        if x.is_cuda:
            if _COMPILED_GATED_RMS_BACKWARD is None:
                _COMPILED_GATED_RMS_BACKWARD = torch.compile(
                    backward, dynamic=True, options={"emulate_precision_casts": True})
            backward = _COMPILED_GATED_RMS_BACKWARD
        dx, dz, dw = backward(x.reshape(-1, x.shape[-1]).detach().contiguous(),
                              gate.reshape(-1, gate.shape[-1]).detach().contiguous(), weight,
                              inv_rms.reshape(-1, 1).detach().contiguous(), grad.reshape(-1, x.shape[-1]).detach().contiguous())
        return (dx.reshape(x.shape) if ctx.needs_input_grad[0] else None,
                dz.reshape(gate.shape) if ctx.needs_input_grad[1] else None,
                dw if ctx.needs_input_grad[2] else None, None)


def _gated_rms_forward(self, x, gate):
    if os.environ.get("RLFORGE_FP8_ALIGN_BACKWARD", "0") == "1":
        return _GatedRMS.apply(x, gate, self.weight, self.variance_epsilon)
    return _gated_rms_value(x, gate, self.weight, self.variance_epsilon)[0]


def _gdn_prepare_core(qf, kf, q_norm, k_norm, dtype):
    return (qf / q_norm).to(dtype), (kf / k_norm).to(dtype)


def _qk_norm_value(q, k):
    global _COMPILED_GDN_PREP
    qf, kf = q.float(), k.float()
    q_norm = torch.sqrt(qf.square().sum(-1, keepdim=True) + 1e-6)
    k_norm = torch.sqrt(kf.square().sum(-1, keepdim=True) + 1e-6)
    if q.is_cuda and os.environ.get("RLFORGE_FP8_ALIGN_POINTWISE", "0") == "1":
        if _COMPILED_GDN_PREP is None:
            _COMPILED_GDN_PREP = torch.compile(
                _gdn_prepare_core, dynamic=True, options={"emulate_precision_casts": True,
                                                         "eager_numerics.division_rounding": True})
        q, k = _COMPILED_GDN_PREP(qf, kf, q_norm, k_norm, q.dtype)
    else:
        q, k = _gdn_prepare_core(qf, kf, q_norm, k_norm, q.dtype)
    return q, k, q_norm, k_norm


def _qk_norm_backward(x, norm, grad):
    xf, gf = x.float(), grad.float()
    direct = gf / norm
    norm_grad = -(gf * xf / norm.square()).sum(-1, keepdim=True)
    return (direct + norm_grad / (2.0 * norm) * (2.0 * xf)).to(x.dtype)


class _QKNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        ctx.set_materialize_grads(False)
        out_q, out_k, q_norm, k_norm = _qk_norm_value(q, k)
        ctx.save_for_backward(q, k, q_norm, k_norm)
        return out_q, out_k

    @staticmethod
    def backward(ctx, grad_q, grad_k):
        global _COMPILED_QK_BACKWARD
        q, k, q_norm, k_norm = ctx.saved_tensors
        backward = _qk_norm_backward
        if q.is_cuda:
            if _COMPILED_QK_BACKWARD is None:
                _COMPILED_QK_BACKWARD = torch.compile(
                    backward, dynamic=True, options={"emulate_precision_casts": True})
            backward = _COMPILED_QK_BACKWARD
        dq = dk = None
        if grad_q is not None and ctx.needs_input_grad[0]:
            dq = backward(q.reshape(-1, q.shape[-1]).detach().contiguous(), q_norm.reshape(-1, 1).detach().contiguous(),
                          grad_q.reshape(-1, q.shape[-1]).detach().contiguous()).reshape(q.shape)
        if grad_k is not None and ctx.needs_input_grad[1]:
            dk = backward(k.reshape(-1, k.shape[-1]).detach().contiguous(), k_norm.reshape(-1, 1).detach().contiguous(),
                          grad_k.reshape(-1, k.shape[-1]).detach().contiguous()).reshape(k.shape)
        return dq, dk


def _gdn_prepare(q, k, b, a, a_log, dt_bias):
    if os.environ.get("RLFORGE_FP8_ALIGN_BACKWARD", "0") == "1":
        q, k = _QKNorm.apply(q, k)
    else:
        q, k, _, _ = _qk_norm_value(q, k)
    beta = b.float().sigmoid()
    g = -a_log.float().exp() * F.softplus(a.float() + dt_bias.to(torch.bfloat16).float())
    return q, k, beta, g


def _causal_conv_core(x, weight, bias):
    xp = F.pad(x.float(), (weight.shape[-1] - 1, 0))
    weight = weight.to(x.dtype).float()
    out = torch.zeros_like(x, dtype=torch.float32)
    for tap in range(weight.shape[-1]):
        product = xp[..., tap:tap + x.shape[-1]] * weight[:, tap][None, :, None]
        out = out + product.to(x.dtype).float()
    if bias is not None:
        out = out + bias.to(x.dtype).float()[None, :, None]
    return out


def _causal_conv(x, weight, bias, activation):
    global _COMPILED_CONV
    if activation not in (None, "silu", "swish"):
        raise ValueError("FP8 aligned GDN convolution requires SiLU")
    if x.is_cuda and os.environ.get("RLFORGE_FP8_ALIGN_COMPILE", "0") == "1":
        if _COMPILED_CONV is None:
            # Inductor otherwise removes the intermediate BF16 tap-product casts.
            _COMPILED_CONV = torch.compile(
                _causal_conv_core, dynamic=True, options={"emulate_precision_casts": True})
        out = _COMPILED_CONV(x, weight, bias)
    else:
        out = _causal_conv_core(x, weight, bias)
    # Fusing SiLU changes rare BF16 values enough to amplify through the model.
    if activation is not None:
        out = F.silu(out)
    return out.to(x.dtype)


def _gdn_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    batch, length, _ = hidden_states.shape
    previous = cache_params is not None and cache_params.has_previous_state(self.layer_idx, state_idx=0)
    if previous and length == 1:
        return self._rlforge_original_gdn_forward(
            hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)
    hidden_states = mq.apply_mask_to_padding_states(hidden_states, attention_mask)
    qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    z = self.in_proj_z(hidden_states).reshape(batch, length, -1, self.head_v_dim)
    b, a = self.in_proj_b(hidden_states), self.in_proj_a(hidden_states)
    if cache_params is not None:
        qkv = cache_params.update_conv_state(qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size)
    qkv = _causal_conv(qkv, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation)
    qkv = qkv[..., -length:].transpose(1, 2)
    q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
    q = q.reshape(batch, length, -1, self.head_k_dim)
    k = k.reshape(batch, length, -1, self.head_k_dim)
    v = v.reshape(batch, length, -1, self.head_v_dim)
    q, k, beta, g = _gdn_prepare(q, k, b, a, self.A_log, self.dt_bias)
    repeats = self.num_v_heads // self.num_k_heads
    if repeats > 1:
        q, k = q.repeat_interleave(repeats, dim=2), k.repeat_interleave(repeats, dim=2)
    state = cache_params.layers[self.layer_idx].recurrent_states[0] if previous else None
    out, final = mq.torch_chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, initial_state=state, output_final_state=cache_params is not None,
        use_qk_l2norm_in_kernel=False, cu_seqlens=kwargs.pop("cu_seq_lens_q", None), **kwargs)
    if cache_params is not None:
        cache_params.update_recurrent_state(final, self.layer_idx)
    out = self.norm(out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
    return self.out_proj(out.reshape(batch, length, -1))


def install(model, *, embedding=True, norm=True, gated_norm=True, gdn=False):
    """Align Qwen3.5 elementwise forward rules without replacing parameters."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    names = []
    input_embedding = model.get_input_embeddings() if embedding else None
    for name, module in model.named_modules():
        if any(part in name.split(".") for part in ("visual", "vision_model", "vision_tower")):
            continue
        forward = None
        if module is input_embedding:
            forward = _embedding_forward
        elif norm and isinstance(module, mq.Qwen3_5RMSNorm):
            forward = _rms_forward
        elif gated_norm and isinstance(module, mq.Qwen3_5RMSNormGated):
            if module.activation != "silu":
                raise ValueError("FP8 gated RMS alignment requires SiLU")
            forward = _gated_rms_forward
        elif gdn and isinstance(module, mq.Qwen3_5GatedDeltaNet):
            if not hasattr(module, "_rlforge_original_gdn_forward"):
                module._rlforge_original_gdn_forward = module.forward
            forward = _gdn_forward
        if forward is not None:
            module.forward = MethodType(forward, module)
            names.append(name)
    return tuple(names)
