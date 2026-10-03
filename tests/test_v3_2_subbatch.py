"""CPU tests for rlforge_v3_2 sub-batch planner and balanced batcher (no model)."""
import random

import torch

import rlforge.prefix_share as ps


def _row(groups):
    """groups: list of (prompt_ids, [completion_lens]) -> packed (1,T) tensors"""
    ids, pos, cm = [], [], []
    for p, cls in groups:
        for n in cls:
            c = [random.randrange(5, 1000) for _ in range(n)]
            ids += p + c
            pos += list(range(len(p) + n))
            cm += [0] * len(p) + [1] * n
    t = lambda x: torch.tensor([x])  # noqa: E731
    return t(ids), t(pos), t(cm)


def test_plan_covers_every_completion_slot_once():
    random.seed(0)
    for budget in (2048, 8192, 65536, 0):
        groups = [([7] * 300, [random.randint(1, 3000) for _ in range(32)]),
                  ([9] * 129, [random.randint(1, 500) for _ in range(16)]),
                  ([3] * 1, [5, 6])]
        ids, pos, cm = _row(groups)
        plan = ps.plan_subbatches(ids, pos, cm, budget)
        seen = []
        for sb in plan:
            P = sb["P"]
            assert P % ps.ALIGN == 0
            assert budget == 0 or sb["tokens"] <= budget or sum(len(b) for b in sb["buckets"]) == 1
            for b in sb["buckets"]:
                L = b[0][1] - (b[0][0] + P)
                for a, e in b:
                    assert e - (a + P) <= L
                    seen.append((a, e))
        starts = (pos[0] == 0).nonzero().flatten().tolist()
        T = ids.shape[1]
        assert sorted(seen) == list(zip(starts, starts[1:] + [T]))


def test_dp_buckets_optimal_small():
    import itertools
    random.seed(1)
    for _ in range(50):
        ls = sorted([random.randint(1, 100) for _ in range(7)], reverse=True)
        lam = random.choice([0, 10, 50])
        runs = ps._dp_buckets(ls, lam)
        cost = sum((j - i) * ls[i] + lam for i, j in runs)
        best = float("inf")
        for mask in itertools.product([0, 1], repeat=len(ls) - 1):
            cuts = [0] + [k + 1 for k, m in enumerate(mask) if m] + [len(ls)]
            c = sum((cuts[q + 1] - cuts[q]) * ls[cuts[q]] + lam for q in range(len(cuts) - 1))
            best = min(best, c)
        assert abs(cost - best) < 1e-9


def _samples(gid, P, lens):
    return [{"group_id": gid, "input_ids": [1] * (P + n), "completion_mask": [0] * P + [1] * n} for n in lens]


def test_balanced_batcher_keeps_samples_and_balances():
    random.seed(2)
    R = 4
    b = ps.BalancedGroupRowBatcher([], R, 128)
    imb_old, imb_new = [], []
    for trial in range(200):
        batch = []
        for g in range(4):
            mu = random.choice([700, 1500, 2500, 4000])
            lens = [min(16384, max(10, int(random.lognormvariate(__import__("math").log(mu), 0.6)))) for _ in range(32)]
            batch += _samples(trial * 10 + g, random.randint(2000, 6000), lens)
        rows = b._partition(batch)
        assert sum(len(r) for r in rows) == 128 and all(rows)
        assert sorted(id(s) for r in rows for s in r) == sorted(id(s) for s in batch)
        for r in rows:  # samples of one group stay contiguous and in order inside a row
            gids = [s["group_id"] for s in r]
            assert gids == sorted(gids, key=lambda g: [x["group_id"] for x in batch].index(g))
        loads = b.last_loads
        imb_new.append(max(loads) / (sum(loads) / R))
        old = ps.GroupRowBatcher([], R, 128)._partition(batch)
        lo = [ps.GroupRowBatcher._cost(_split_units(r)[0]) if r else 0 for r in old]
        lo = [sum(ps.GroupRowBatcher._cost(u) for u in _split_units(r)) for r in old]
        imb_old.append(max(lo) / (sum(lo) / R))
    mo, mn = sum(imb_old) / len(imb_old), sum(imb_new) / len(imb_new)
    print(f"row max/mean: GroupRowBatcher {mo:.3f} -> Balanced {mn:.3f}")
    assert mn < 1.05 and mn < mo


def _split_units(row):
    units = []
    for s in row:
        if units and units[-1][0]["group_id"] == s["group_id"]:
            units[-1].append(s)
        else:
            units.append([s])
    return units


if __name__ == "__main__":
    test_plan_covers_every_completion_slot_once()
    test_dp_buckets_optimal_small()
    test_balanced_batcher_keeps_samples_and_balances()
    print("ok")
