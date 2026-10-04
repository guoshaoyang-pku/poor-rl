#!/usr/bin/env python3
"""Reference math for the GSPO loss in `async_grpo_train.py`.

The trainer computes the per-sequence ratio and both normalizations inside one fused,
packed-row compute_loss. Here the same quantities are computed the slow way, loop by loop,
so the unit test can compare the two implementations without depending on the packing
layout. Pure function: given the per-token log-ratios, advantages and masks of a batch of
sequences, return (rho, token_loss, seq_mean_loss).
"""
from __future__ import annotations


def adaptive_clip_eps(
    rho,
    eps_low: float,
    eps_high: float,
    max_low_frac: float,
    max_high_frac: float,
    max_eps: float,
):
    import math

    if rho.ndim != 1:
        raise ValueError("rho must be a 1D tensor")
    if not 0 < max_low_frac < 1 or not 0 < max_high_frac < 1:
        raise ValueError("clip-fraction caps must be strictly between 0 and 1")
    if max_eps < max(eps_low, eps_high):
        raise ValueError("max_eps must be at least both base eps values")
    if rho.numel() == 0:
        return eps_low, eps_high

    count = rho.numel()
    low_eps = eps_low
    high_eps = eps_high
    allowed_low = math.floor(max_low_frac * count)
    allowed_high = math.floor(max_high_frac * count)

    ascending = rho.sort().values
    if torch_count(rho < 1 - low_eps) > allowed_low:
        low_eps = max(low_eps, min(max_eps, float((1 - ascending[allowed_low]).item())))
    if torch_count(rho < 1 - low_eps) > allowed_low:
        raise ValueError("max_eps cannot satisfy the requested low-side clip cap")

    descending = rho.sort(descending=True).values
    if torch_count(rho > 1 + high_eps) > allowed_high:
        high_eps = max(high_eps, min(max_eps, float((descending[allowed_high] - 1).item())))
    if torch_count(rho > 1 + high_eps) > allowed_high:
        raise ValueError("max_eps cannot satisfy the requested high-side clip cap")

    return low_eps, high_eps


def torch_count(mask) -> int:
    return int(mask.sum().item())


def reference_gspo(log_ratios, advantages, masks, eps_low: float, eps_high: float):
    """Naive per-sequence GSPO. Returns (rho, loss_token, loss_seq_mean).

    log_ratios/advantages: lists per sequence of per-token floats (completion tokens only).
    masks: parallel lists of 0/1 floats selecting scored tokens.
    """
    import math

    rho = []
    per_token_losses = []
    for lr_seq, adv_seq, mask_seq in zip(log_ratios, advantages, masks):
        n = max(sum(mask_seq), 1e-9)
        mean_lr = sum(l * m for l, m in zip(lr_seq, mask_seq)) / n
        r = math.exp(mean_lr)
        rho.append(r)
        r_clip = min(max(r, 1 - eps_low), 1 + eps_high)
        per_token_losses.append(
            [-min(r * a, r_clip * a) * m for a, m in zip(adv_seq, mask_seq)]
        )

    flat = [v for seq in per_token_losses for v in seq]
    n_tok = max(sum(m for seq in masks for m in seq), 1e-9)
    loss_token = sum(flat) / n_tok
    loss_seq_mean = sum(
        sum(seq) / max(sum(mask), 1e-9) for seq, mask in zip(per_token_losses, masks)
    ) / len(masks)
    return rho, loss_token, loss_seq_mean
