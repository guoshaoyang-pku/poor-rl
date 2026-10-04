"""Qwen3.5 BF16 activation boundaries shared with vLLM FP8 serving.

Parameter objects and state_dict entries remain FP32. The casts are ordinary
autograd operations, so gradients accumulate into the original FP32 masters.
Attention and recurrent kernels still differ between the two backends.
"""
from __future__ import annotations

import os
from types import MethodType

import torch
import torch.nn.functional as F

_COMPILED_CONV = None


def _embedding_forward(self, input):
    return F.embedding(input, self.weight, self.padding_idx, self.max_norm,
                       self.norm_type, self.scale_grad_by_freq, self.sparse).to(torch.bfloat16)


def _rms_forward(self, x):
    xf = x.float()
    out = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
    return (out * (1.0 + self.weight.to(torch.bfloat16).float())).to(x.dtype)


def _gated_rms_forward(self, x, gate):
    xf = x.float()
    out = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.variance_epsilon)
    out = out * self.weight.to(torch.bfloat16).float()
    return (out * F.silu(gate.float())).to(x.dtype)


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
    qf, kf = q.float(), k.float()
    q = (qf / torch.sqrt(qf.square().sum(-1, keepdim=True) + 1e-6)).to(q.dtype)
    k = (kf / torch.sqrt(kf.square().sum(-1, keepdim=True) + 1e-6)).to(k.dtype)
    beta = b.float().sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.to(torch.bfloat16).float())
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
