#!/usr/bin/env python3
"""CPU tests for the GSPO loss core in rlforge.trainer.GSPOAsyncGRPOTrainer.compute_loss.

`packed_gspo_core` re-implements the packed-row math with the *same tensor ops* the
trainer uses (index_add segmentation over the packed row), then the tests compare it
against the naive per-sequence reference in rlforge.gspo. What is under test is the
math (ratio, clip, both normalizations, gradient direction), not the model forward -- the
forward is identical to the parent's and covered by TRL.

Packed layout, verified against TRL 1.14's rollout worker (`_SampleBuilder`) and the
parent's compute_loss:
  * each sequence is [prompt tokens (mask 0), completion tokens (mask 1)], packed end to
    end, `position_ids` restarting at 0 per sequence;
  * after the trainer's `[:, 1:]` shift, shifted index k scores unshifted token k+1. A
    sequence's prompt sits at unshifted positions p..q; its shifted slot covers the FIRST
    completion token (unshifted q+1). Therefore a prompt position's shifted slot is
    scored exactly when the next token is a completion token -- so in the shifted tensors
    the mask is 1 at every completion token and at each prompt position that directly
    precedes one. Every completion token is scored (TRL's `has_trained_token` /
    `sum(completion_mask)` counts confirm none is dropped at boundaries).
"""
import math

import torch

from rlforge.gspo import reference_gspo


def packed_gspo_core(log_ratio, advantages, position_ids, completion_mask, eps_low, eps_high, norm):
    """Same tensor ops as GSPOAsyncGRPOTrainer.compute_loss, without the model.

    All four tensors are the shifted (T-1)-length versions.
    """
    seq_ids = (position_ids[0] == 0).cumsum(0)[1:] - 1
    num_seq = int((position_ids == 0).sum())
    valid = completion_mask[0] > 0

    zeros = torch.zeros(num_seq, dtype=log_ratio.dtype)
    seq_lr_sum = zeros.index_add(0, seq_ids, log_ratio[0] * valid)
    seq_n_tok = zeros.index_add(0, seq_ids, valid.to(log_ratio.dtype))
    seq_mean_lr = seq_lr_sum / seq_n_tok.clamp(min=1.0)
    rho = torch.exp(seq_mean_lr)
    rho_clipped = torch.clamp(rho, 1 - eps_low, 1 + eps_high)

    rho_tok, rho_clip_tok = rho[seq_ids], rho_clipped[seq_ids]
    per_token_loss = -torch.min(rho_tok * advantages[0], rho_clip_tok * advantages[0])

    if norm == "seq_mean":
        seq_loss_sum = torch.zeros(num_seq, dtype=log_ratio.dtype).index_add_(
            0, seq_ids, per_token_loss * valid
        )
        return rho, (seq_loss_sum / seq_n_tok.clamp(min=1.0)).mean()
    loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
    return rho, loss


def make_batch(seq_lens, seed=0):
    """Shifted packed batch + per-sequence python lists for the reference.

    Shifted mask = 1 wherever the *next* unshifted token is a completion token (see
    module docstring): every completion token scores, and each prompt position that
    directly precedes the first completion token also scores (it owns that first
    completion token's prediction).
    """
    g = torch.Generator().manual_seed(seed)
    pos_parts, mask_parts, lr_parts, adv_parts = [], [], [], []
    for n in seq_lens:
        pos_parts.append(torch.arange(1 + n))                       # prompt pos 0, comp 1..n
        m = torch.ones(1 + n)
        m[0] = 0.0                                                  # prompt token itself not a target
        mask_parts.append(m)
        lr_parts.append(torch.randn(n, generator=g) * 1e-3)
        adv_parts.append(torch.randn(n, generator=g))

    position_ids_u = torch.cat(pos_parts).unsqueeze(0)              # (1, T) unshifted
    completion_mask_u = torch.cat(mask_parts).unsqueeze(0)

    # compute_loss takes the *unshifted* position_ids and the `[:, 1:]`-shifted mask /
    # old / adv / log_probs. seq_ids is derived from the unshifted position_ids by the
    # same cumsum trick as the parent, which is why it has length T-1 too.
    position_ids = position_ids_u
    completion_mask = completion_mask_u[:, 1:]

    padded_lr = torch.cat([torch.cat([torch.zeros(1), lrp]) for lrp in lr_parts]).unsqueeze(0)
    padded_ad = torch.cat([torch.cat([torch.zeros(1), adp]) for adp in adv_parts]).unsqueeze(0)
    log_ratio = padded_lr[:, 1:]
    advantages = padded_ad[:, 1:]
    assert log_ratio.shape == completion_mask.shape and position_ids.shape[1] == completion_mask.shape[1] + 1, (
        log_ratio.shape, completion_mask.shape, position_ids.shape,
    )

    lr_ref = [t.tolist() for t in lr_parts]
    adv_ref = [t.tolist() for t in adv_parts]
    mask_ref = [[1.0] * n for n in seq_lens]
    return log_ratio, advantages, position_ids, completion_mask, lr_ref, adv_ref, mask_ref


def test_ratio_and_losses_match_reference():
    seq_lens = [5, 37, 3, 120, 16, 64, 8, 200]
    eps_low, eps_high = 3e-4, 4e-4
    for seed in (0, 1, 7):
        lr, ad, pos, cm, lr_ref, ad_ref, mask_ref = make_batch(seq_lens, seed)
        rho_ref, tok_ref, seq_ref = reference_gspo(lr_ref, ad_ref, mask_ref, eps_low, eps_high)

        rho_tok, loss_tok = packed_gspo_core(lr, ad, pos, cm, eps_low, eps_high, "token")
        rho_seq, loss_seq = packed_gspo_core(lr, ad, pos, cm, eps_low, eps_high, "seq_mean")

        assert torch.allclose(rho_tok, rho_seq)
        assert torch.allclose(rho_tok, torch.tensor(rho_ref), atol=1e-9)
        assert math.isclose(loss_tok.item(), tok_ref, rel_tol=1e-6)
        assert math.isclose(loss_seq.item(), seq_ref, rel_tol=1e-6)


def test_seq_mean_reweights_long_sequences():
    # seq A: 1 completion token (advantage -1); seq B: 200 completion tokens (+0.001).
    # rho == 1 so per-token loss = -advantage on scored slots.
    # unshifted: A = [prompt, comp] (2 tokens), B = [prompt, 200 comp] (201) -> T = 203.
    # shifted slot k scores unshifted token k+1, so A's completion lands on shifted 0 and
    # B's 200 completions land on shifted 2..201 (shifted 1 covers B's prompt: mask 0).
    lr = torch.zeros(1, 202)
    ad = torch.tensor([[-1.0, 0.0] + [0.001] * 200])
    pos = torch.tensor([[0, 1] + [0] + list(range(1, 201))])      # unshifted, length 203
    cm = torch.tensor([[1.0, 0.0] + [1.0] * 200])
    rho, loss_tok = packed_gspo_core(lr, ad, pos, cm, 3e-4, 4e-4, "token")
    _, loss_seq = packed_gspo_core(lr, ad, pos, cm, 3e-4, 4e-4, "seq_mean")
    assert torch.allclose(rho, torch.ones(2), atol=1e-6)
    # token norm: (1*1 + 200*(-0.001)) / 201 -- dominated by the long sequence
    assert math.isclose(loss_tok.item(), (1.0 - 0.2) / 201, rel_tol=1e-6)
    # seq_mean: (1/1 + (-0.2)/200) / 2 -- both sequences count equally
    assert math.isclose(loss_seq.item(), (1.0 - 0.001) / 2, rel_tol=1e-6)


def test_seq_clip_engages_on_drift():
    eps_low, eps_high = 3e-4, 4e-4
    # two sequences, 10 completions each; seq 0 drifts up beyond eps_high, seq 1 down
    # beyond eps_low. unshifted T = 22, shifted length 21.
    lr = torch.tensor([[0.001] * 10 + [0.0] + [-0.002] * 10])
    ad = torch.ones(1, 21)
    pos = torch.tensor([[0] + list(range(1, 11)) + [0] + list(range(1, 11))])
    cm = torch.tensor([[1.0] * 10 + [0.0] + [1.0] * 10])
    rho, _ = packed_gspo_core(lr, ad, pos, cm, eps_low, eps_high, "seq_mean")
    assert rho[0].item() > 1 + eps_high
    assert rho[1].item() < 1 - eps_low


def test_gradient_direction_positive_advantage():
    # With advantage > 0 and rho inside the clip range, the loss is -rho * advantage, so
    # d(loss)/d(log_ratio) = -advantage * rho / n_tokens < 0: increasing the sampled
    # tokens' log-prob decreases the loss. (Outside the clip range the min() picks the
    # clipped branch, which is constant in rho -- zero gradient is the intended PPO
    # semantics there.)
    lr = torch.full((1, 4), 1e-4, requires_grad=True)   # rho = exp(1e-4) < 1 + eps_high
    ad = torch.ones(1, 4)
    pos = torch.tensor([[0, 1, 2, 3, 4]])
    cm = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    rho, loss = packed_gspo_core(lr, ad, pos, cm, 3e-4, 4e-4, "seq_mean")
    assert rho.item() < 1 + 4e-4
    loss.backward()
    assert (lr.grad < 0).all()
    # and the magnitude matches -advantage * rho / n_tokens
    assert torch.allclose(lr.grad, torch.full((1, 4), -rho.item() / 4), rtol=1e-5)


def test_gradient_zero_when_clipped():
    # advantage > 0 with rho above the clip: min() selects the clipped branch, which no
    # longer depends on rho, so the gradient is exactly zero (PPO semantics).
    lr = torch.full((1, 4), 0.01, requires_grad=True)   # rho = exp(0.01) >> 1 + eps_high
    ad = torch.ones(1, 4)
    pos = torch.tensor([[0, 1, 2, 3, 4]])
    cm = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    _, loss = packed_gspo_core(lr, ad, pos, cm, 3e-4, 4e-4, "seq_mean")
    loss.backward()
    assert (lr.grad == 0).all()


if __name__ == "__main__":
    test_ratio_and_losses_match_reference()
    test_seq_mean_reweights_long_sequences()
    test_seq_clip_engages_on_drift()
    test_gradient_direction_positive_advantage()
    test_gradient_zero_when_clipped()
    print("5 GSPO core tests passed")
