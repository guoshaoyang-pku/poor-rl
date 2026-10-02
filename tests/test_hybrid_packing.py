"""Regression: padding-free packing must not leak across sample boundaries.

The bug this pins: a packed row concatenates several rollout samples and resets `position_ids` at
each boundary. The attention layers honour that through a block-diagonal mask, but the
GatedDeltaNet layers never received the boundaries at all, so the conv window and the recurrent
state of sample i leaked into sample i+1. Measured on Qwen3.5-0.8B before the fix: +0.489 and
+1.142 nats on segments 1 and 2 of a 3-way packed row -- inflating the trainer's logprobs and
pinning `gspo/seq_clip_low_frac` near 1.0.

These tests are CPU/GPU only in the sense that they need torch; the ones that need a real hybrid
model skip when no local Qwen3.5 checkpoint is available.
"""

import os

import pytest
import torch

from rlforge.hybrid_packing import (
    assert_packing_contract,
    boundary_aware_packing,
    install,
    packed_segments,
    segments,
)

MODEL_PATH = os.environ.get("RLFORGE_HYBRID_MODEL", "/data/home/guoshaoyang/models/Qwen3.5-0.8B-ms")


def _needs_model():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    if not os.path.isdir(MODEL_PATH):
        pytest.skip(f"no local hybrid checkpoint at {MODEL_PATH}")


# ---------------------------------------------------------------- boundary derivation


def test_single_sequence_is_not_packed():
    assert packed_segments(torch.arange(8).unsqueeze(0)) is None


def test_two_segments_are_found():
    pos = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]])
    assert packed_segments(pos) == [(0, 3), (3, 8)]


def test_batch_of_two_is_refused_rather_than_guessed():
    # A packed row is always batch 1; anything else is not the layout this supports.
    pos = torch.tensor([[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]])
    assert packed_segments(pos) is None


def test_row_not_starting_at_zero_is_refused():
    assert packed_segments(torch.tensor([[3, 4, 0, 1, 2]])) is None


def test_none_position_ids_is_not_packed():
    assert packed_segments(None) is None


# ---------------------------------------------------------------- context manager


def test_context_manager_sets_and_restores():
    pos = torch.tensor([[0, 1, 0, 1, 2]])
    assert segments() is None
    with boundary_aware_packing(pos) as found:
        assert found == [(0, 2), (2, 5)]
        assert segments() == [(0, 2), (2, 5)]
    assert segments() is None


def test_context_manager_is_reentrant():
    outer = torch.tensor([[0, 1, 0, 1]])
    inner = torch.tensor([[0, 1, 2]])
    with boundary_aware_packing(outer):
        with boundary_aware_packing(inner) as found:
            assert found is None  # inner row is a single sequence
            assert segments() is None
        assert segments() == [(0, 2), (2, 4)]
    assert segments() is None


def test_context_manager_restores_on_exception():
    with pytest.raises(RuntimeError):
        with boundary_aware_packing(torch.tensor([[0, 1, 0, 1]])):
            raise RuntimeError("boom")
    assert segments() is None


# ---------------------------------------------------------------- contract assertion


def test_contract_rejects_use_cache_on_a_packed_row():
    pos = torch.tensor([[0, 1, 0, 1]])
    with pytest.raises(ValueError, match="use_cache"):
        assert_packing_contract(pos, use_cache=True)


def test_contract_rejects_attention_mask_on_a_packed_row():
    pos = torch.tensor([[0, 1, 0, 1]])
    with pytest.raises(ValueError, match="attention_mask"):
        assert_packing_contract(pos, attention_mask=torch.ones(1, 4))


def test_contract_accepts_the_trainer_forward():
    # What rlforge's compute_loss does: packed row, use_cache=False, no mask.
    assert_packing_contract(torch.tensor([[0, 1, 0, 1]]), use_cache=False, attention_mask=None)


def test_contract_is_free_for_un_packed_rows():
    # A single-sequence row is never checked, so an otherwise-illegal flag is not an error here.
    assert_packing_contract(torch.arange(4).unsqueeze(0), use_cache=True)


# ---------------------------------------------------------------- end-to-end on a real model


def _load():
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16
    ).to("cuda").eval()
    return model


def _logprobs(model, ids, pos=None):
    kwargs = {"use_cache": False}
    if pos is not None:
        kwargs["position_ids"] = torch.tensor([pos], device="cuda")
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids], device="cuda"), **kwargs).logits.float()[0]
    lp = torch.log_softmax(logits, dim=-1)
    return torch.stack([lp[i - 1, torch.tensor(ids, device="cuda")[i]]
                        for i in range(1, len(ids))])


def test_patched_forward_matches_unpacked_forward():
    """The whole point: a packed row must score exactly like the samples scored alone."""
    _needs_model()
    model = _load()
    assert install(model), "expected the hybrid GatedDeltaNet ops to be patched"

    torch.manual_seed(0)
    length = 32
    seqs = [torch.randint(1000, 50000, (length,)).tolist() for _ in range(3)]
    clean = [_logprobs(model, s) for s in seqs]

    ids, pos = [], []
    for s in seqs:
        pos.extend(range(length))
        ids.extend(s)
    packed = _logprobs(model, ids, pos)

    for i, reference in enumerate(clean):
        base = i * length
        got = packed[base:base + length - 1]
        deviation = (got - reference).abs().max().item()
        assert deviation < 1e-4, f"segment {i} leaked: max|dev| = {deviation:.4e}"


def test_unpatched_forward_does_leak():
    """Guards the test above from silently passing on a build where packing is already correct."""
    _needs_model()
    model = _load()  # deliberately no install()

    torch.manual_seed(0)
    length = 32
    seqs = [torch.randint(1000, 50000, (length,)).tolist() for _ in range(3)]
    clean = [_logprobs(model, s) for s in seqs]

    ids, pos = [], []
    for s in seqs:
        pos.extend(range(length))
        ids.extend(s)
    packed = _logprobs(model, ids, pos)

    # The trailing segment is the one that accumulates leakage; if upstream ever fixes this,
    # delete this test together with the wrapper.
    base = 2 * length
    deviation = (packed[base:base + length - 1] - clean[2]).abs().max().item()
    assert deviation > 1e-3, (
        "expected the un-patched forward to leak across the boundary; if this now passes, "
        "transformers/TRL fixed packed-sequence handling and rlforge.hybrid_packing may be "
        "retired"
    )
