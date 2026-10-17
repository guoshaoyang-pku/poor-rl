"""Budgeted look-ahead: beyond the K=1 limit, at most RLFORGE_AHEAD_EXTRA_FRAC of admitted groups (default 0.15)."""
import types

import pytest

from rlforge import score_loop as SL


class _V:
    def __init__(self, v=0):
        self.value = v


def _gate(k, B=64, G=8):
    g = types.SimpleNamespace(model_version=0, _ahead_origin=0, _ahead_batch_samples=B, _ahead_batches=k,
                              _ahead_dropped=_V(), _ahead_admitted=0, _ahead_next_group=0, _ahead_groups={},
                              _ahead_waiters={}, _ahead_extra_groups=0, num_generations=G)
    g.admit = lambda: SL.ConcurrentScoreMixin._ahead_try_admit(g, g._ahead_next_group)
    return g


def _fill(g):
    n = 0
    while g.admit():
        n += 1
    return n


def test_k1_unchanged():
    g = _gate(1)
    assert _fill(g) == 2 * 64 // 8 and g._ahead_extra_groups == 0
    g.model_version = 1
    assert _fill(g) == 64 // 8


@pytest.mark.parametrize('frac', [0.15, 0.3])
def test_k2_extra_groups_stay_within_budget(monkeypatch, frac):
    monkeypatch.setattr(SL, '_AHEAD_EXTRA_FRAC', frac)
    g = _gate(2)
    for version in range(40):  # the trainer consumes one batch per version; the gate refills each time
        g.model_version = version
        _fill(g)
        groups = g._ahead_admitted // g.num_generations
        assert g._ahead_extra_groups <= frac * groups + 1e-9
    assert g._ahead_extra_groups > 0  # the extra room is actually used
    # never beyond K future batches
    assert g._ahead_admitted <= (39 + 3) * 64
