# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional vLLM QK/RoPE with the trainer's BF16 product rounding.

Enable RLFORGE_FP8_ROPE_BF16=1 with poor_rl_fp8. FP8 projections and scales
stay unchanged; each BF16 rotary product rounds before the sum. Derived from
the vLLM fused Qwen3.5 kernel, with FP fusion disabled to keep those boundaries.
"""
import functools
import hashlib
import os
from pathlib import Path

import torch
from vllm.triton_utils import tl, triton

NATIVE_SOURCE_SHA256 = "182a13b20ac5857674a368d030268161ae0d4f358a38114cded804c1791abcaa"
_INSTALLED = None


@triton.jit
def _fused_qk_rmsnorm_rope_gate_kernel(
    q_gate_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    gate_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_gate_stride_t,
    k_stride_t,
    q_out_stride_t,
    k_out_stride_t,
    gate_out_stride_t,
    cache_stride_p,
    positions_stride_m,
    positions_stride_t,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rotary: tl.constexpr,
    eps: tl.constexpr,
    norm_beta: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ROT_HALF_BLOCK: tl.constexpr,
    HAS_PASS: tl.constexpr,
    HAS_MROPE: tl.constexpr,
    MROPE_SECTION_H: tl.constexpr,
    MROPE_SECTION_W: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    is_k = head >= num_q_heads
    local_head = tl.where(is_k, head - num_q_heads, head)

    if is_k:
        in_base = k_ptr + token * k_stride_t + local_head * head_dim
        w_ptr = k_weight_ptr
        out_base = k_out_ptr + token * k_out_stride_t + local_head * head_dim
    else:
        in_base = q_gate_ptr + token * q_gate_stride_t + local_head * 2 * head_dim
        w_ptr = q_weight_ptr
        out_base = q_out_ptr + token * q_out_stride_t + local_head * head_dim

    # --- RMSNorm: variance over the full head_dim ---
    head_offs = tl.arange(0, HEAD_BLOCK)
    head_mask = head_offs < head_dim
    x = tl.load(in_base + head_offs, mask=head_mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / head_dim
    inv_rms = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + head_offs, mask=head_mask, other=0.0).to(tl.float32) + norm_beta
    # Round-trip through INPUT_DTYPE so the RoPE input matches the bf16-storage
    # behavior of the unfused (qk_rmsnorm -> memory -> apply_rope) reference path.
    x_norm = (x * inv_rms * w).to(INPUT_DTYPE).to(tl.float32)

    # --- Pass-through tail [rotary_dim, head_dim): RMSNorm-only, no rotation ---
    # The rotary head [0, rotary_dim) will be overwritten by the RoPE store below.
    if HAS_PASS:
        pass_mask = head_mask & (head_offs >= rotary_dim)
        tl.store(out_base + head_offs, x_norm, mask=pass_mask)

    # --- Partial RoPE on the first rotary_dim elements ---
    # Triton lacks easy sub-vector slicing of x_norm, so we recompute the
    # normalized rotary halves on a smaller block (next_pow2(half_rotary)).
    # The extra ~rotary_dim element reload hits L1, so the cost is negligible.
    rot_offs = tl.arange(0, ROT_HALF_BLOCK)
    rot_mask = rot_offs < half_rotary
    x_rot1 = tl.load(in_base + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    x_rot2 = tl.load(in_base + half_rotary + rot_offs, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    w_rot1 = (
        tl.load(w_ptr + rot_offs, mask=rot_mask, other=0.0).to(tl.float32) + norm_beta
    )
    w_rot2 = (
        tl.load(w_ptr + half_rotary + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
        + norm_beta
    )
    x_rot1 = (x_rot1 * inv_rms * w_rot1).to(INPUT_DTYPE).to(tl.float32)
    x_rot2 = (x_rot2 * inv_rms * w_rot2).to(INPUT_DTYPE).to(tl.float32)

    # Always use int64 for positions to avoid overflow in address computation.
    pos_t = tl.load(positions_ptr + token * positions_stride_t).to(tl.int64)
    if HAS_MROPE:
        pos_h = tl.load(
            positions_ptr + positions_stride_m + token * positions_stride_t
        ).to(tl.int64)
        pos_w = tl.load(
            positions_ptr + 2 * positions_stride_m + token * positions_stride_t
        ).to(tl.int64)
        is_h = (rot_offs % 3 == 1) & (rot_offs < 3 * MROPE_SECTION_H)
        is_w = (rot_offs % 3 == 2) & (rot_offs < 3 * MROPE_SECTION_W)
        pos = tl.where(is_h, pos_h, tl.where(is_w, pos_w, pos_t))
    else:
        pos = pos_t
    cache_offset = pos * cache_stride_p
    cos = tl.load(
        cos_sin_cache_ptr + cache_offset + rot_offs, mask=rot_mask, other=0.0
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr + cache_offset + half_rotary + rot_offs,
        mask=rot_mask,
        other=0.0,
    ).to(tl.float32)

    qc = (x_rot1 * cos).to(INPUT_DTYPE).to(tl.float32)
    ks = (x_rot2 * sin).to(INPUT_DTYPE).to(tl.float32)
    kc = (x_rot2 * cos).to(INPUT_DTYPE).to(tl.float32)
    qs = (x_rot1 * sin).to(INPUT_DTYPE).to(tl.float32)
    # The trainer's unfused BF16 reference rounds both products and the
    # resulting sum/difference. Keep the final elementwise op in BF16 too;
    # otherwise Triton emits an FP32 FMA and only rounds at the store.
    o1 = (qc - ks).to(INPUT_DTYPE)
    o2 = (kc + qs).to(INPUT_DTYPE)
    tl.store(out_base + rot_offs, o1, mask=rot_mask)
    tl.store(out_base + half_rotary + rot_offs, o2, mask=rot_mask)

    # --- Gate copy (q heads only, verbatim) ---
    if not is_k:
        gate_in_base = in_base + head_dim
        gate_out_base = gate_out_ptr + token * gate_out_stride_t + local_head * head_dim
        g = tl.load(gate_in_base + head_offs, mask=head_mask, other=0.0)
        tl.store(gate_out_base + head_offs, g, mask=head_mask)


def fused_qk_rmsnorm_rope_gate(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    mrope_section: list[int] | tuple[int, int, int] | None = None,
    norm_beta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused split + QK-RMSNorm + (partial) RoPE + gate copy for Qwen attn.

    Args:
        q_gate: (n_tokens, num_q_heads * 2 * head_dim) -- per head: [q|gate]
        k: (n_tokens, num_kv_heads * head_dim)
        q_weight: (head_dim,) RMSNorm weight
        k_weight: (head_dim,) RMSNorm weight
        cos_sin_cache: (max_pos, rotary_dim) packed [cos|sin]
        positions: (n_tokens,) or (3, n_tokens) int32 or int64
        eps: RMSNorm epsilon
        num_q_heads: number of Q heads (after TP split)
        num_kv_heads: number of KV heads (after TP split)
        head_dim: per-head dimension
        rotary_dim: rotary dimension; must be even and <= head_dim
        mrope_section: interleaved T/H/W frequency counts for 2D positions
        norm_beta: scalar added to the RMSNorm weight

    Returns:
        (q_out, k_out, gate_out) -- all contiguous (n_tokens, heads * head_dim).
        ``gate_out`` is the raw (pre-sigmoid) gate.
    """
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError(
            f"rotary_dim must be a positive even integer <= head_dim, "
            f"got rotary_dim={rotary_dim}, head_dim={head_dim}"
        )
    if q_gate.dtype != torch.bfloat16 or k.dtype != q_gate.dtype:
        raise ValueError(
            "Shared RoPE requires BF16 q_gate and k, "
            f"got {q_gate.dtype} and {k.dtype}"
        )
    for name, tensor in (
        ("q_gate", q_gate),
        ("k", k),
        ("q_weight", q_weight),
        ("k_weight", k_weight),
        ("cos_sin_cache", cos_sin_cache),
    ):
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} must be contiguous in its last dimension")

    if positions.ndim not in (1, 2):
        raise ValueError(f"positions must be 1D or 2D, got shape={positions.shape}")
    if positions.shape[-1] != q_gate.shape[0]:
        raise ValueError(
            "positions token dimension must match q_gate, "
            f"got {positions.shape[-1]} and {q_gate.shape[0]}"
        )

    has_mrope = positions.ndim == 2
    if has_mrope:
        if positions.shape[0] != 3:
            raise ValueError(
                f"MRoPE positions must have shape (3, n_tokens), got {positions.shape}"
            )
        if mrope_section is None or len(mrope_section) != 3:
            raise ValueError("mrope_section must contain the T/H/W frequency counts")
        if sum(mrope_section) != rotary_dim // 2:
            raise ValueError(
                "mrope_section must sum to rotary_dim // 2, "
                f"got {mrope_section} and rotary_dim={rotary_dim}"
            )
        mrope_section_h = mrope_section[1]
        mrope_section_w = mrope_section[2]
        positions_stride_m, positions_stride_t = positions.stride()
    else:
        if mrope_section is not None:
            raise ValueError("mrope_section requires 2D MRoPE positions")
        mrope_section_h = 0
        mrope_section_w = 0
        positions_stride_m = 0
        positions_stride_t = positions.stride(0)

    n_tokens = q_gate.shape[0]
    q_out = torch.empty(
        (n_tokens, num_q_heads * head_dim), dtype=q_gate.dtype, device=q_gate.device
    )
    k_out = torch.empty(
        (n_tokens, num_kv_heads * head_dim), dtype=k.dtype, device=k.device
    )
    gate_out = torch.empty_like(q_out)
    if n_tokens == 0:
        return q_out, k_out, gate_out

    half_rotary = rotary_dim // 2
    head_block = triton.next_power_of_2(head_dim)
    rot_half_block = triton.next_power_of_2(half_rotary)
    num_warps = max(1, head_block // 64)

    grid = (n_tokens, num_q_heads + num_kv_heads)
    _fused_qk_rmsnorm_rope_gate_kernel[grid](
        q_gate,
        k,
        q_out,
        k_out,
        gate_out,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        q_gate.stride(0),
        k.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        gate_out.stride(0),
        cos_sin_cache.stride(0),
        positions_stride_m,
        positions_stride_t,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        half_rotary,
        eps,
        norm_beta=norm_beta,
        INPUT_DTYPE=tl.bfloat16,
        HEAD_BLOCK=head_block,
        ROT_HALF_BLOCK=rot_half_block,
        HAS_PASS=rotary_dim < head_dim,
        HAS_MROPE=has_mrope,
        MROPE_SECTION_H=mrope_section_h,
        MROPE_SECTION_W=mrope_section_w,
        num_warps=num_warps,
        num_stages=2,
        enable_fp_fusion=False,
    )
    return q_out, k_out, gate_out



def install():
    """Install the source-pinned BF16 route only for an opted-in FP8 config."""
    global _INSTALLED
    if os.environ.get("RLFORGE_FP8_ROPE_BF16") != "1":
        return False
    from vllm.model_executor.layers import fused_qk_norm_rope as native_module
    from vllm.model_executor.models import qwen3_next

    if _INSTALLED is not None:
        if qwen3_next.fused_qk_rmsnorm_rope_gate is not _INSTALLED:
            raise RuntimeError("Shared BF16 RoPE route was replaced after installation")
        return True
    digest = hashlib.sha256(Path(native_module.__file__).read_bytes()).hexdigest()
    if digest != NATIVE_SOURCE_SHA256:
        raise RuntimeError("Native QK/RoPE source changed; shared BF16 RoPE refused")
    native = native_module.fused_qk_rmsnorm_rope_gate
    if qwen3_next.fused_qk_rmsnorm_rope_gate is not native:
        raise RuntimeError("Native QK/RoPE route already replaced; shared BF16 RoPE refused")

    @functools.wraps(native)
    def shared(*args, **kwargs):
        q_gate = args[0] if args else kwargs["q_gate"]
        if q_gate.dtype != torch.bfloat16:
            return native(*args, **kwargs)
        return fused_qk_rmsnorm_rope_gate(*args, **kwargs)

    qwen3_next.fused_qk_rmsnorm_rope_gate = shared
    _INSTALLED = shared
    return True
