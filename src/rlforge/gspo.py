#!/usr/bin/env python3
"""Reference math for the GSPO loss in `async_grpo_train.py`.

The trainer computes the per-sequence ratio and both normalizations inside one fused,
packed-row compute_loss. Here the same quantities are computed the slow way, loop by loop,
so the unit test can compare the two implementations without depending on the packing
layout. Pure function: given the per-token log-ratios, advantages and masks of a batch of
sequences, return (rho, token_loss, seq_mean_loss).
"""
from __future__ import annotations


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
