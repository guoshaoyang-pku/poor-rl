#!/usr/bin/env python3
"""CPU tests for rlforge.drop_audit (plain asserts; run: PYTHONPATH=src python tests/test_drop_audit.py).

1. hand-checked scenario through TRL's real RolloutQueueDataset + DataCollatorForRollout (audited subclasses):
   exact per-step counts, buckets (longest / median / n-truncated), partial groups, length stats, reward means,
   metric-sink entries (reduced with TRL's _reduce_metric), jsonl header + offline report aggregate;
2. short / interleaved groups are flagged (short, late_samples);
3. fence: an audit bug disables the audit, TRL's stream is unchanged;
4. equivalence: TRL's own AsyncGRPOTrainer.get_train_dataloader (on a stand-in trainer, plain DataLoader) with and
   without drop_audit.install, for TRL's FixedCountBatcher and rlforge's GroupRowBatcher: identical collated batches,
   identical TRL metrics, audit totals == an independent replay of the staleness rule;
5. dispatcher: the same through accelerate's real DataLoaderDispatcher (split_batches + dispatch_batches, one
   micro-batch prefetch) at production geometry (NGEN=32, 4 rows, GAS=8, STALE=3, GroupRowBatcher) and HF's loop
   order: identical batches with the audit on/off, and every step record equals exactly what that step trained
   (sample ids recovered from the batch tensors) plus the drops an independent replay attributes to it;
6. wait_s compensation: the audit's hook time is taken back out of TRL's queue-wait accumulator;
7. install robustness: short cap (edges dropped, not an error), patch failure falls back to TRL's dataloader,
   broken trainer -> install returns None;
8. jsonl robustness: failed writes are retried with the lines kept, non-finite floats become null, a header line per
   segment, and the report keeps the last record per step across a relaunch;
9. overhead micro-benchmark (wrapped vs unwrapped queue dataset, collate hook, flush cost).
"""

import importlib.util
import json
import math
import os
import queue
import random
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from types import SimpleNamespace

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402
from accelerate import PartialState  # noqa: E402

PartialState()  # TRL logs through accelerate's logger, which needs an initialized state (CPU, single process)
import trl.experimental.async_grpo.async_grpo_trainer as agt  # noqa: E402
from trl.experimental.async_grpo.async_rollout_worker import RolloutSample  # noqa: E402

from rlforge import drop_audit as da  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "..", "scripts", "drop_audit_report.py")


def load_report():
    spec = importlib.util.spec_from_file_location("drop_audit_report", REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mk(gid, ver, length, reward, prompt_len=10, tag=None):
    n = prompt_len + length
    ids = list(range(n)) if tag is None else [tag] * n
    return RolloutSample(prompt=[], completion=[], input_ids=ids,
                         completion_mask=[0] * prompt_len + [1] * length, old_log_probs=[0.0] * n,
                         advantage=0.0, model_version=ver, group_id=gid, metrics={"reward": float(reward)})


class StopFeed(Exception):
    pass


def _raise_stop(_s):
    raise StopFeed()


def approx(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol


def zero_anomalies(rec):
    return sum(rec["anomalies"].values()) == 0


# ---------------------------------------------------------------------------------------------------- test 1
def test_hand_scenario():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "drop_audit.jsonl")
    audit = da.DropAudit(num_generations=4, max_completion=16384, path=path)
    audit.configure(token_budget=0, gas=1)
    DS, COL = da.audited_classes(agt, audit)
    groups = [  # gid, version, lengths, rewards
        (0, 1, [100, 200, 300, 1900], [1, 1, 0, 0]),       # lt2k   stale 1 -> kept
        (1, 0, [2100, 50, 50, 50], [0, 0, 0, 1]),         # 2k_4k  stale 2 -> dropped
        (2, 1, [5000, 16384, 10, 10], [0.5, -2, 1, 1]),    # trunc  2 kept, sync, 2 dropped -> partial
        (3, 2, [9000, 8192, 100, 100], [1, 0, 0, 0]),      # 8k_16k kept
        (4, 3, [4096, 4000, 1, 1], [1, 1, 1, 0]),          # 4k_8k  2 kept | sync | 2 kept
    ]
    q = queue.Queue()
    for gid, ver, lens, rws in groups:
        for L, r in zip(lens, rws):
            q.put(mk(gid, ver, L, r))
    state = {"v": 2}
    trl_metrics = defaultdict(list)
    ds = DS(rollout_queue=q, model_version_fn=lambda: state["v"], check_health_fn=_raise_stop,
            stale_after_s=1e9, metrics=trl_metrics, max_staleness=1, poll_interval_s=0.01)
    col = COL(0, 1, metrics=trl_metrics)
    assert audit.enabled and isinstance(ds.queue, da._AuditedQueue)
    it = iter(ds)
    sink = defaultdict(list)
    recs = []
    for step, k in enumerate([6, 6, 2], 1):
        items = [next(it) for _ in range(k)]
        col.torch_call([[items]])  # one micro-batch per step (gas=1, no prefetch), one row
        state["v"] += 1  # TRL's weight-sync callback runs before the audit's
        recs.append(audit.flush(step, state["v"], sink, n_trained=1))
    r1, r2, r3 = recs
    # --- step 1
    assert r1["samples"] == {"arrived": 10, "kept": 6, "dropped": 4, "trained": 6, "drop_frac": 0.4}, r1["samples"]
    assert r1["decision_versions"] == [2, 2] and r1["microbatches"] == 1 and r1["prefetched"] == 0
    b = r1["buckets"]
    assert (b["lt2k"]["groups"], b["lt2k"]["kept_full"], b["lt2k"]["samples"], b["lt2k"]["samples_dropped"]) == (1, 1, 4, 0)
    assert approx(b["lt2k"]["reward_trained"], 0.5) and b["lt2k"]["lag_mean"] == 1
    assert (b["2k_4k"]["groups"], b["2k_4k"]["dropped_full"], b["2k_4k"]["samples_dropped"]) == (1, 1, 4)
    assert b["2k_4k"]["drop_frac"] == 1.0 and b["2k_4k"]["group_drop_frac"] == 1.0
    assert approx(b["2k_4k"]["reward_dropped"], 0.25) and b["2k_4k"]["lag_mean"] == 2
    assert all(b[n]["groups"] == 0 for n in ("4k_8k", "8k_16k", "trunc"))
    # median bucketing: group 1 (2100, 50, 50, 50) has median 50 -> lt2k; both groups have no truncated member
    bm, bt = r1["buckets_median"], r1["buckets_ntrunc"]
    assert (bm["lt2k"]["groups"], bm["lt2k"]["dropped_full"], bm["2k_4k"]["groups"]) == (2, 1, 0), bm
    assert (bt["t0"]["groups"], bt["t0"]["dropped_full"], bt["t1"]["groups"], bt["t2p"]["groups"]) == (2, 1, 0, 0)
    assert r1["open_groups"] == 1 and r1["open_samples"] == 2 and r1["open_mb_samples"] == 0
    g = r1["len"]["generated"]
    assert (g["n"], g["sum"], g["p50"], g["p90"], g["max"]) == (10, 26134, 200, 5000, 16384), g
    assert g["hist"] == {"lt2k": 7, "2k_4k": 1, "4k_8k": 1, "8k_16k": 0, "trunc": 1}
    t = r1["len"]["trained"]
    assert (t["n"], t["sum"], t["p50"], t["p90"], t["max"]) == (6, 23884, 300, 16384, 16384), t
    assert t["hist"] == {"lt2k": 4, "2k_4k": 0, "4k_8k": 1, "8k_16k": 0, "trunc": 1}
    assert r1["len"]["dropped"]["sum"] == 2250
    assert approx(r1["reward"]["trained_samples"], 0.5 / 6) and approx(r1["reward"]["dropped_samples"], 0.25)
    assert approx(r1["reward"]["kept_samples"], 0.5 / 6)
    assert approx(r1["reward"]["trained_groups"], 0.5) and approx(r1["reward"]["dropped_groups"], 0.25)
    assert r1["reward"]["partial_groups"] is None
    assert r1["staleness_at_arrival"] == {"kept": {"1": 6}, "dropped": {"2": 4}}
    assert zero_anomalies(r1), r1["anomalies"]
    assert {"kept_ne_trained", "backlog_nonzero", "fifo_underrun", "fifo_overrun"} <= set(r1["anomalies"])
    # --- step 2
    assert r2["samples"]["arrived"] == 8 and r2["samples"]["kept"] == 6 and r2["samples"]["dropped"] == 2
    assert r2["samples"]["trained"] == 6
    b = r2["buckets"]
    tr = b["trunc"]
    assert (tr["groups"], tr["partial"], tr["samples"], tr["samples_dropped"], tr["samples_dropped_partial"]) == (1, 1, 4, 2, 2)
    assert tr["kept_full"] == 0 and tr["dropped_full"] == 0 and tr["drop_frac"] == 0.5 and tr["group_drop_frac"] == 0.0
    assert approx(tr["reward_partial"], 0.125) and tr["lag_mean"] == 1
    assert approx(tr["r_partial_kept_sum"], -1.5) and approx(tr["r_partial_dropped_sum"], 2.0)
    assert (b["8k_16k"]["groups"], b["8k_16k"]["kept_full"]) == (1, 1) and approx(b["8k_16k"]["reward_trained"], 0.25)
    # group 2 (10, 10, 5000, 16384): median 10 -> lt2k, one truncated member -> t1; group 3 median 100 -> lt2k
    bm, bt = r2["buckets_median"], r2["buckets_ntrunc"]
    assert (bm["lt2k"]["groups"], bm["lt2k"]["partial"], bm["trunc"]["groups"]) == (2, 1, 0), bm
    assert (bt["t0"]["groups"], bt["t1"]["groups"], bt["t1"]["partial"]) == (1, 1, 1), bt
    g = r2["len"]["generated"]
    assert (g["n"], g["sum"], g["p50"], g["p90"], g["max"]) == (8, 25508, 100, 9000, 9000), g
    assert g["hist"] == {"lt2k": 4, "2k_4k": 1, "4k_8k": 1, "8k_16k": 2, "trunc": 0}
    t = r2["len"]["trained"]
    assert (t["n"], t["sum"], t["p50"], t["p90"]) == (6, 25488, 4000, 9000), t
    assert approx(r2["reward"]["trained_samples"], 0.5) and approx(r2["reward"]["dropped_samples"], 1.0)
    assert approx(r2["reward"]["partial_groups"], 0.125)
    assert r2["staleness_at_arrival"] == {"kept": {"0": 2, "1": 4}, "dropped": {"2": 2}}
    assert r2["open_groups"] == 1 and r2["decision_versions"] == [3, 3]
    # --- step 3
    assert r3["samples"]["arrived"] == 2 and r3["samples"]["dropped"] == 0 and r3["samples"]["trained"] == 2
    b = r3["buckets"]
    assert (b["4k_8k"]["groups"], b["4k_8k"]["kept_full"]) == (1, 1) and approx(b["4k_8k"]["reward_trained"], 0.75)
    assert b["4k_8k"]["lag_mean"] == 0 and r3["open_groups"] == 0
    assert r3["cum"] == {"arrived": 20, "kept": 14, "dropped": 6, "collated": 14, "trained": 14, "backlog": 0}
    assert all(zero_anomalies(r) for r in recs)
    assert trl_metrics["sample/dropped_stale_total"] == [1.0] * 6  # TRL's own drop counter agrees
    # --- metric sink: one entry per step; TRL's own reduction gives the per-step value
    R = agt._reduce_metric
    assert sink["drop_audit/2k_4k/drop_frac"] == [(4.0, 4.0)]
    assert sink["drop_audit/lt2k/groups_total"] == [1.0, 0.0, 0.0]
    assert sink["drop_audit/trunc/groups_partial_total"] == [0.0, 1.0, 0.0]
    assert sink["drop_audit/by_median/lt2k/groups_total"] == [2.0, 2.0, 1.0]
    assert sink["drop_audit/by_ntrunc/t1/groups_total"] == [0.0, 1.0, 0.0]
    assert approx(R("drop_audit/drop_frac", sink["drop_audit/drop_frac"]), 6 / 20)
    assert approx(R("drop_audit/len_generated_mean", sink["drop_audit/len_generated_mean"]), (26134 + 25508 + 2) / 20)
    assert R("drop_audit/len_generated_max", sink["drop_audit/len_generated_max"]) == 16384
    assert R("drop_audit/samples_dropped_total", sink["drop_audit/samples_dropped_total"]) == 6
    assert R("drop_audit/samples_trained_total", sink["drop_audit/samples_trained_total"]) == 14
    assert sink["drop_audit/len_hist/trunc/generated_total"] == [1.0, 0.0, 0.0]
    assert approx(R("drop_audit/reward_dropped_groups", sink["drop_audit/reward_dropped_groups"]), 0.25)
    assert R("drop_audit/anomalies_total", sink["drop_audit/anomalies_total"]) == 0
    assert R("drop_audit/jsonl_fail_total", sink["drop_audit/jsonl_fail_total"]) == 0
    assert sink["drop_audit/backlog"] == [0.0, 0.0, 0.0]
    assert all(k.startswith("drop_audit/") for k in sink)
    kinds = {k: isinstance(v[0], tuple) for k, v in sink.items()}
    assert all(all(isinstance(x, tuple) == kinds[k] for x in v) for k, v in sink.items())  # one shape per key
    # --- jsonl + offline report
    lines = [json.loads(x) for x in open(path)]
    assert lines[0]["type"] == "header" and lines[0]["resume_step"] == 0 and lines[0]["config"]["exact_batcher"]
    assert [x["type"] for x in lines] == ["header", "step", "step", "step"]
    rep = load_report()
    recs_disk = rep.load(path, None, None, False)
    assert [r["step"] for r in recs_disk] == [1, 2, 3]
    agg = rep.aggregate(recs_disk)
    tot = agg["total"]
    assert (tot["groups"], tot["kept_full"], tot["dropped_full"], tot["partial"]) == (5, 3, 1, 1)
    assert (tot["samples"], tot["samples_dropped"]) == (20, 6)
    assert agg["len"]["generated"]["n"] == 20 and agg["len"]["generated"]["max"] == 16384
    assert agg["len"]["trained"]["hist"] == {"lt2k": 8, "2k_4k": 1, "4k_8k": 2, "8k_16k": 2, "trunc": 1}
    assert agg["samples"] == {"arrived": 20, "kept": 14, "dropped": 6, "trained": 14}
    assert agg["bucketings"]["buckets_ntrunc"]["t1"]["groups"] == 1
    agg23 = rep.aggregate(rep.load(path, 2, 3, False))
    assert agg23["total"]["groups"] == 3 and agg23["buckets"]["trunc"]["partial"] == 1
    text = rep.report(agg)
    assert "ALL" in text and "trunc" in text and "MEDIAN" in text and "TRUNCATED" in text
    assert rep.main([path, "--per-step"]) == 0
    print("[test_hand_scenario] OK")
    print(text)
    print()
    print(rep.per_step(recs_disk))
    return path


# ---------------------------------------------------------------------------------------------------- test 2
def test_short_and_interleaved():
    audit = da.DropAudit(num_generations=4, max_completion=16384, path=None)
    DS, COL = da.audited_classes(agt, audit)
    q = queue.Queue()
    order = [(0, 100), (0, 100), (0, 100),          # group 0 has only 3 rows (short)
             (1, 100), (1, 100), (2, 3000), (1, 100), (1, 100),  # group 2 interleaves group 1
             (2, 3000), (2, 3000), (2, 3000)]
    for gid, L in order:
        q.put(mk(gid, 0, L, 1.0))
    ds = DS(rollout_queue=q, model_version_fn=lambda: 0, check_health_fn=_raise_stop, stale_after_s=1e9,
            metrics=defaultdict(list), max_staleness=3, poll_interval_s=0.01)
    it = iter(ds)
    items = [next(it) for _ in range(len(order))]
    COL(0, 1, metrics=defaultdict(list)).torch_call([[items]])
    rec = audit.flush(1, 1, None, n_trained=1)
    # group 0 (3 rows) is closed by group 1's first row -> short. Group 1 is closed at 2 rows when group 2's first
    # row interleaves, group 2 is closed at 1 row when group 1 resumes; the 5 rows arriving for closed groups are
    # counted as late_samples (sample-level stats still include them). Interleaving is therefore loud, not silent.
    lt, b2 = rec["buckets"]["lt2k"], rec["buckets"]["2k_4k"]
    assert (lt["groups"], lt["short"], lt["samples"]) == (2, 2, 5), lt
    assert (b2["groups"], b2["short"], b2["samples"]) == (1, 1, 1), b2
    assert rec["buckets_median"]["lt2k"]["short"] == 2 and rec["buckets_ntrunc"]["t0"]["short"] == 3
    assert rec["anomalies"]["late_samples"] == 5, rec["anomalies"]
    assert rec["open_groups"] == 0
    assert rec["samples"]["arrived"] == len(order) == rec["samples"]["trained"] and rec["samples"]["dropped"] == 0
    print("[test_short_and_interleaved] OK", {k: v for k, v in rec["anomalies"].items() if v})


def test_fence():
    """An audit bug (here: a sample whose metrics is not a dict) disables the audit; TRL's stream is unchanged."""
    def feed():
        q = queue.Queue()
        for i in range(12):
            s = mk(i // 4, 0 if i < 8 else -5, 100 + i, 1.0)
            if i == 5:
                s.metrics = ["not", "a", "dict"]
            q.put(s)
        return q

    def pull(cls, audit=None):
        ds = cls(rollout_queue=feed(), model_version_fn=lambda: 0, check_health_fn=_raise_stop, stale_after_s=1e9,
                 metrics=defaultdict(list), max_staleness=3, poll_interval_s=0.001)
        out = []
        try:
            for item in ds:
                out.append((item["group_id"], tuple(item["input_ids"]), item["advantage"]))
        except StopFeed:
            pass
        return out, ds.metrics

    audit = da.DropAudit(num_generations=4, max_completion=16384, path=None)
    plain, m_plain = pull(agt.RolloutQueueDataset)
    wrapped, m_wrapped = pull(da.audited_classes(agt, audit)[0], audit)
    assert plain == wrapped and len(plain) == 8, (len(plain), len(wrapped))
    assert m_plain["sample/dropped_stale_total"] == m_wrapped["sample/dropped_stale_total"] == [1.0] * 4
    assert not audit.enabled and audit.flush(1, 1, defaultdict(list)) is None
    print("[test_fence] OK (audit disabled itself, stream identical)")


# ---------------------------------------------------------------------------------------------------- test 4
class _Accel:
    num_processes = 4
    is_main_process = True

    def __init__(self, real=None):
        self._real = real

    def prepare(self, dl):
        return dl if self._real is None else self._real.prepare(dl)


class StandInTrainer:
    """Just the attributes TRL's AsyncGRPOTrainer.get_train_dataloader reads; the method itself is TRL's."""
    get_train_dataloader = agt.AsyncGRPOTrainer.get_train_dataloader

    def __init__(self, q, ngen, max_staleness, out, gas, real_accel=None, cap=16384):
        self.accelerator = _Accel(real_accel)
        self.rollout_queue = q
        self.model_version = 0
        self.rollout_worker = SimpleNamespace(check_health=_raise_stop)
        self.args = SimpleNamespace(heartbeat_stale_after_s=1e9, max_staleness=max_staleness, report_to=[],
                                    token_budget=0, per_device_train_batch_size=ngen, output_dir=out,
                                    num_generations=ngen, max_completion_length=cap,
                                    gradient_accumulation_steps=gas)
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.processing_class = SimpleNamespace(pad_token_id=0)
        self._trained_groups = set()
        self._rollout_dataset = None
        self.callbacks = []

    def add_callback(self, cb):
        self.callbacks.append(cb)


def make_stream(n_groups, ngen, seed):
    rng = random.Random(seed)
    samples = []
    for gid in range(n_groups):
        approx_step = gid // 8
        ver = max(0, approx_step - rng.choice([0, 1, 1, 2, 2, 3, 3, 4, 5]))
        base = rng.choice([300, 800, 1500, 3000, 6000, 12000])
        # ~15% short groups (an empty completion yields no training row), so version bumps land mid-group
        # and partial groups occur.
        rows = ngen if rng.random() > 0.15 else ngen - rng.randint(1, 3)
        for _ in range(rows):
            L = 16384 if rng.random() < 0.03 else max(1, min(16383, int(rng.lognormvariate(math.log(base), 0.6))))
            samples.append(mk(gid, ver, L, rng.choice([-2.0, 0.0, 0.5, 1.0]), prompt_len=rng.randint(5, 40)))
    return samples


def run_trainer_path(samples, ngen, gas, stale, audited, out, real_accel=None, max_steps=None, cap=16384):
    """HF's loop order: fetch gas micro-batches, (optimizer), weight sync (version bump), step-end callbacks."""
    q = queue.Queue()
    for s in samples:
        q.put(s)
    tr = StandInTrainer(q, ngen, stale, out, gas, real_accel, cap)
    audit = da.install(tr, path=os.path.join(out, "drop_audit.jsonl")) if audited else None
    assert (audit is not None) == audited
    dl = tr.get_train_dataloader()
    tr._rollout_dataset.poll_interval_s = 0.001  # the end of the feed is detected quickly
    assert (type(tr._rollout_dataset) is agt.RolloutQueueDataset) == (not audited)
    assert agt.RolloutQueueDataset.__name__ == "RolloutQueueDataset"  # module patch was scoped
    assert agt.DataCollatorForRollout.__name__ == "DataCollatorForRollout"
    batches, per_step, step = [], [], 0
    it = iter(dl)
    try:
        while max_steps is None or step < max_steps:
            bs = []
            for _ in range(gas):
                bs.append(next(it))
                batches.append(bs[-1])  # kept even if the feed ends mid-step (collated, never trained)
            per_step.append(bs)
            tr.current_gradient_accumulation_steps = len(bs)
            step += 1
            tr.model_version += 1  # weight sync, then the step-end callbacks
            for cb in tr.callbacks:
                cb.on_step_end(tr.args, SimpleNamespace(global_step=step), None)
    except StopFeed:
        pass
    for cb in tr.callbacks:
        cb.on_train_end(tr.args, SimpleNamespace(global_step=step), None)
    return tr, audit, batches, step, per_step


def replay(samples, per_step_kept, stale, ngen):
    """Independent re-statement of TRL's rule + the plain DataLoader's pull pattern (no prefetch)."""
    v, kept_in_step, out = 0, 0, []
    for s in samples:
        drop = v - s.model_version > stale
        out.append((s, drop, v))
        if not drop:
            kept_in_step += 1
            if kept_in_step == per_step_kept:
                v += 1
                kept_in_step = 0
    groups = {}
    for s, drop, vv in out:
        g = groups.setdefault(s.group_id, {"n": 0, "nd": 0, "mx": 0, "tr": False})
        L = s.completion_mask.count(1)
        g["n"] += 1
        g["nd"] += drop
        g["mx"] = max(g["mx"], L)
        g["tr"] |= L >= 16384
    return out, groups


def compare_runs(ta, ba, tb, bb):
    assert len(ba) == len(bb) > 0, (len(ba), len(bb))
    for x, y in zip(ba, bb):
        assert x.keys() == y.keys()
        for k in x:
            assert x[k].dtype == y[k].dtype and torch.equal(x[k], y[k]), k
    assert ta._trained_groups == tb._trained_groups
    ma = dict(ta._metrics["train"])
    mb = {k: v for k, v in tb._metrics["train"].items() if not k.startswith("drop_audit/")}
    assert ma.keys() == mb.keys(), set(ma) ^ set(mb)
    for k in ma:
        if k == "sample/time_in_queue_s":
            continue  # wall-clock
        assert ma[k] == mb[k], k


def test_equivalence(batcher_name):
    ngen, gas, stale, nproc = 8, 2, 3, 4
    per_step = ngen * nproc * gas
    samples = make_stream(150, ngen, seed=7)
    orig_fcb = agt.FixedCountBatcher
    if batcher_name == "GroupRowBatcher":
        from rlforge.prefix_share import GroupRowBatcher
        agt.FixedCountBatcher = GroupRowBatcher
    try:
        tmp = tempfile.mkdtemp()
        ta, _, ba, steps_a, _ = run_trainer_path(samples, ngen, gas, stale, False, os.path.join(tmp, "a"))
        tb, audit, bb, steps_b, _ = run_trainer_path(samples, ngen, gas, stale, True, os.path.join(tmp, "b"))
    finally:
        agt.FixedCountBatcher = orig_fcb
    assert steps_a == steps_b
    compare_runs(ta, ba, tb, bb)
    # audit totals == independent replay
    rep = load_report()
    recs = rep.load(os.path.join(tmp, "b", "drop_audit.jsonl"), None, None, True)
    assert len(recs) == steps_b + 1 and recs[-1]["final"]
    out, groups = replay(samples, per_step, stale, ngen)
    n_drop = sum(d for _, d, _ in out)
    agg = rep.aggregate(recs)
    assert agg["samples"]["arrived"] == len(samples) == len(out)
    assert agg["samples"]["dropped"] == n_drop == len(ta._metrics["train"]["sample/dropped_stale_total"])
    n_collated = sum(int(((b["position_ids"] == 0) & (b["attention_mask"] == 1)).sum()) for b in bb)
    assert recs[-1]["cum"]["collated"] == n_collated, (recs[-1]["cum"], n_collated)
    assert agg["samples"]["trained"] == steps_b * per_step == recs[-1]["cum"]["trained"]  # exact
    assert recs[-1]["untrained_collated"] == n_collated - steps_b * per_step
    for r in recs[:-1]:  # every step end: nothing kept and uncollated, the step trained what it kept
        assert r["cum"]["backlog"] == 0 and r["samples"]["kept"] == r["samples"]["trained"] == per_step, r["samples"]
        assert r["prefetched"] == 0 and zero_anomalies(r), r["anomalies"]
    assert 0 < recs[-1]["cum"]["backlog"] < ngen * nproc  # only the unfinished micro-batch is outstanding
    assert agg["anomalies"]["decision_mismatch"] == 0 and sum(agg["anomalies"].values()) == 0, agg["anomalies"]
    exp = {n: {"groups": 0, "kept_full": 0, "dropped_full": 0, "partial": 0, "short": 0, "samples": 0,
               "samples_dropped": 0} for n in agg["names"]}
    for g in groups.values():
        name = agg["names"][audit.bucket(g["mx"], g["tr"])]
        e = exp[name]
        e["groups"] += 1
        e["short"] += g["n"] < ngen
        e["samples"] += g["n"]
        e["samples_dropped"] += g["nd"]
        e["kept_full" if g["nd"] == 0 else "dropped_full" if g["nd"] == g["n"] else "partial"] += 1
    for n in agg["names"]:
        got = {k: agg["buckets"][n][k] for k in exp[n]}
        assert got == exp[n], (n, got, exp[n])
    # per-step: without a prefetch, a step's micro-batches are exactly the pulls between version bumps
    win = defaultdict(lambda: [0, 0])
    for _, d, vv in out:
        win[vv][0] += 1
        win[vv][1] += d
    for r in recs[:-1]:
        assert [r["samples"]["arrived"], r["samples"]["dropped"]] == win[r["step"] - 1], (r["step"], r["samples"])
    tot = agg["total"]
    assert tot["partial"] > 0 and tot["dropped_full"] > 0 and tot["short"] > 0, tot  # all paths exercised
    assert recs[-1]["open_groups"] == 0
    print(f"[test_equivalence:{batcher_name}] OK steps={steps_b} batches={len(bb)} samples={len(samples)} "
          f"dropped={n_drop} groups={tot['groups']} kept_full={tot['kept_full']} dropped_full={tot['dropped_full']} "
          f"partial={tot['partial']} short={tot['short']} final_backlog={recs[-1]['cum']['backlog']}")


# ---------------------------------------------------------------------------------------------------- test 5
_ACC = []


def real_dispatcher():
    if not _ACC:
        from accelerate import Accelerator
        from accelerate.utils import DataLoaderConfiguration
        _ACC.append(Accelerator(cpu=True, dataloader_config=DataLoaderConfiguration(split_batches=True,
                                                                                     dispatch_batches=True)))
    return _ACC[0]


def dispatch_stream(n_groups, seed, short_p, ngen):
    """Group version lags its length class (long groups arrive staler), as in production. Sample i's tokens are
    all i + 1, so a trained batch tells exactly which samples it holds."""
    rng = random.Random(seed)
    out = []
    for gid in range(n_groups):
        base = rng.choice([400, 900, 1800, 3500, 7000, 12000])
        lens = [16384 if rng.random() < (0.01 if base < 7000 else 0.15) else
                max(1, min(16383, int(rng.lognormvariate(math.log(base), 0.5)))) for _ in range(ngen)]
        mx = max(lens)
        lag = (0 if mx < 2048 else 1 if mx < 4096 else 2 if mx < 8192 else 3 if mx < 16384 else 4)
        lag += rng.choice([0, 0, 1])
        ver = max(0, gid // 32 - lag)
        rows = ngen - (rng.randint(1, 2) if rng.random() < short_p else 0)
        P = rng.randint(50, 300)
        for L in lens[:rows]:
            out.append(mk(gid, ver, L, rng.choice([-2.0, 0.0, 0.5, 1.0]), prompt_len=P, tag=len(out) + 1))
    return out


def dispatch_replay(samples, steps, gas, mb, stale, ngen):
    """Independent: micro-batch m (1-based) is filled at version max(0, (m-2)//gas) (the dispatcher prefetches one
    micro-batch; HF fetches gas per step, then the sync). A full group completes at its last row, a short one at
    the next group's first row."""
    kept, out = 0, []
    for s in samples:
        m = kept // mb + 1
        if m > steps * gas + 1:
            break
        drop = max(0, (m - 2) // gas) - s.model_version > stale
        out.append((s, drop, m))
        kept += not drop
    groups, order = {}, []
    for i, (s, d, m) in enumerate(out):
        g = groups.get(s.group_id)
        if g is None:
            if order and groups[order[-1]]["done_mb"] is None:
                groups[order[-1]]["done_mb"] = m  # short group closed by the next group's first row
            g = groups[s.group_id] = {"lens": [], "nd": 0, "done_mb": None}
            order.append(s.group_id)
        g["lens"].append(s.completion_mask.count(1))
        g["nd"] += d
        if len(g["lens"]) == ngen:
            g["done_mb"] = m
    return out, groups


def test_dispatcher(short_p, seed, steps=10):
    from rlforge.prefix_share import GroupRowBatcher
    ngen, nproc, gas, stale = 32, 4, 8, 3
    mb = ngen * nproc
    acc = real_dispatcher()
    samples = dispatch_stream(32 * steps * 2, seed, short_p, ngen)
    by_tag = {s.input_ids[0]: s for s in samples}
    orig_fcb = agt.FixedCountBatcher
    agt.FixedCountBatcher = GroupRowBatcher
    try:
        tmp = tempfile.mkdtemp()
        ta, _, ba, sa, _ = run_trainer_path(samples, ngen, gas, stale, False, os.path.join(tmp, "a"), acc, steps)
        tb, audit, bb, sb, per_step_b = run_trainer_path(samples, ngen, gas, stale, True, os.path.join(tmp, "b"),
                                                         acc, steps)
    finally:
        agt.FixedCountBatcher = orig_fcb
    assert sa == sb == steps and len(ba) == len(bb) == steps * gas
    compare_runs(ta, ba, tb, bb)
    rep = load_report()
    path = os.path.join(tmp, "b", "drop_audit.jsonl")
    recs = rep.load(path, None, None, False)
    final = rep.load(path, None, None, True)[-1]
    assert [r["step"] for r in recs] == list(range(1, steps + 1)) and final["final"]
    out, groups = dispatch_replay(samples, steps, gas, mb, stale, ngen)
    names = audit.names
    for r, bs in zip(recs, per_step_b):
        s = r["step"]
        lo, hi = (s - 1) * gas + 1, s * gas
        # what this step trained, read back from the batch tensors
        tags = []
        for b in bs:
            starts = (b["position_ids"] == 0) & (b["attention_mask"] == 1)
            tags += b["input_ids"][starts].tolist()
        assert len(tags) == len(set(tags)) == gas * mb
        trained = [by_tag[t] for t in tags]
        exp_lens = sorted(x.completion_mask.count(1) for x in trained)
        mine = [(x, d) for x, d, m in out if lo <= m <= hi]
        assert sorted(x.input_ids[0] for x, d in mine if not d) == sorted(tags), s  # replay agrees with TRL
        assert r["samples"] == {"arrived": len(mine), "kept": gas * mb, "dropped": sum(d for _, d in mine),
                                "trained": gas * mb, "drop_frac": sum(d for _, d in mine) / len(mine)}, (s, r["samples"])
        assert r["len"]["trained"]["n"] == gas * mb and r["len"]["trained"]["sum"] == sum(exp_lens)
        assert r["len"]["trained"]["p50"] == da.nearest_rank(exp_lens, 0.5)
        assert r["len"]["trained"]["p90"] == da.nearest_rank(exp_lens, 0.9) and r["len"]["trained"]["max"] == exp_lens[-1]
        gen = sorted(x.completion_mask.count(1) for x, _ in mine)
        assert r["len"]["generated"]["sum"] == sum(gen) and r["len"]["generated"]["n"] == len(gen)
        rs = sum(x.metrics["reward"] for x in trained)
        assert approx(r["reward"]["trained_samples_sum"], rs, 1e-6) and r["reward"]["trained_samples_n"] == gas * mb
        assert r["microbatches"] == gas and r["prefetched"] == 1, (r["microbatches"], r["prefetched"])
        assert r["cum"]["backlog"] == 0 and r["cum"]["trained"] == s * gas * mb, r["cum"]
        assert r["cum"]["collated"] == (s * gas + 1) * mb
        assert r["decision_versions"] == ([0, 0] if s == 1 else [s - 2, s - 1]), (s, r["decision_versions"])
        assert zero_anomalies(r), (s, r["anomalies"])
        exp = {n: Counter() for n in names}
        for g in groups.values():
            if g["done_mb"] is None or not lo <= g["done_mb"] <= hi:
                continue
            lens = sorted(g["lens"])
            e = exp[names[audit.bucket(lens[-1], lens[-1] >= 16384)]]
            e["groups"] += 1
            e["short"] += len(lens) < ngen
            e["samples"] += len(lens)
            e["samples_dropped"] += g["nd"]
            e["kept_full" if g["nd"] == 0 else "dropped_full" if g["nd"] == len(lens) else "partial"] += 1
        for n in names:
            got = {k: r["buckets"][n][k] for k in ("groups", "kept_full", "dropped_full", "partial", "short",
                                                    "samples", "samples_dropped")}
            want = {k: exp[n].get(k, 0) for k in got}
            assert got == want, (s, n, got, want)
    agg = rep.aggregate(recs)
    assert agg["samples"]["trained"] == steps * gas * mb  # exact: no prefetched micro-batch leaks in
    assert final["untrained_collated"] == mb and final["samples"]["trained"] == 0
    tot = agg["total"]
    assert tot["dropped_full"] > 0, tot
    if short_p == 0:
        assert tot["partial"] == 0 and tot["short"] == 0, tot  # version bumps fall between whole groups
    else:
        assert tot["short"] > 0, tot
    nt = agg["bucketings"]["buckets_ntrunc"]
    assert sum(nt[n]["groups"] for n in nt) == tot["groups"]
    print(f"[test_dispatcher:short_p={short_p}] OK steps={steps} trained={agg['samples']['trained']} "
          f"arrived={agg['samples']['arrived']} dropped={agg['samples']['dropped']} groups={tot['groups']} "
          f"dropped_full={tot['dropped_full']} partial={tot['partial']} short={tot['short']} "
          f"buckets(groups/dropped)={[(n, agg['buckets'][n]['groups'], agg['buckets'][n]['dropped_full']) for n in names]}")


# ---------------------------------------------------------------------------------------------------- test 6
def test_wait_compensation():
    """A slow hook (2 ms per pull) must not show up in TRL's queue-wait accumulator."""
    n = 60
    audit = da.DropAudit(num_generations=4, max_completion=16384, path=None)
    DS, _ = da.audited_classes(agt, audit)
    orig = audit._arrive

    def slow(sample):
        time.sleep(0.002)
        orig(sample)

    audit._arrive = slow
    q = queue.Queue()
    for i in range(n):
        q.put(mk(i // 4, 0, 100, 1.0))
    ds = DS(rollout_queue=q, model_version_fn=lambda: 0, check_health_fn=_raise_stop, stale_after_s=1e9,
            metrics=defaultdict(list), max_staleness=3, poll_interval_s=0.01)
    it = iter(ds)
    for _ in range(n):
        next(it)
    assert audit.enabled
    assert -1e-3 < ds.wait_s < 0.25 * n * 0.002, ds.wait_s  # uncompensated it would be >= 0.12 s
    print(f"[test_wait_compensation] OK wait_s={ds.wait_s * 1e3:.2f} ms with {n * 2} ms of hook time taken out")


# ---------------------------------------------------------------------------------------------------- test 7
def test_install_robustness():
    tmp = tempfile.mkdtemp()
    # short cap: edges at/above it are dropped (no ValueError on any rank)
    a = da.DropAudit(num_generations=4, max_completion=4096, path=None)
    assert a.edges == (2048,) and a.names == ["lt2k", "2k_4k", "trunc"], (a.edges, a.names)
    assert da.DropAudit(num_generations=4, max_completion=1024, path=None).names == ["lt1k", "trunc"]
    samples = make_stream(40, 8, seed=3)
    tr, audit, batches, steps, _ = run_trainer_path(samples, 8, 2, 3, True, os.path.join(tmp, "short"), cap=4096)
    assert audit.enabled and steps > 0 and audit.names == ["lt2k", "2k_4k", "trunc"]
    recs = load_report().load(os.path.join(tmp, "short", "drop_audit.jsonl"), None, None, False)
    assert len(recs) == steps and all(zero_anomalies(r) for r in recs)
    assert sum(r["len"]["trained"]["hist"]["trunc"] for r in recs) > 0
    # patch failure (e.g. a TRL rename): TRL's own dataloader is used, training goes on
    q = queue.Queue()
    for s in samples:
        q.put(s)
    tr = StandInTrainer(q, 8, 3, os.path.join(tmp, "patchfail"), 2)
    audit = da.install(tr)
    orig = da.audited_classes
    da.audited_classes = lambda *_a, **_k: (_ for _ in ()).throw(AttributeError("RolloutQueueDataset"))
    try:
        dl = tr.get_train_dataloader()
    finally:
        da.audited_classes = orig
    assert type(tr._rollout_dataset) is agt.RolloutQueueDataset and not audit.enabled
    next(iter(dl))
    # broken trainer: install returns None and leaves get_train_dataloader alone
    bad = SimpleNamespace(args=SimpleNamespace(output_dir=tmp))  # no num_generations, no add_callback
    bad.get_train_dataloader = lambda: "orig"
    assert da.install(bad) is None and bad.get_train_dataloader() == "orig"
    print("[test_install_robustness] OK")


# ---------------------------------------------------------------------------------------------------- test 8
def test_jsonl_robustness():
    tmp = tempfile.mkdtemp()
    blocker = os.path.join(tmp, "blocker")
    open(blocker, "w").close()
    path = os.path.join(blocker, "drop_audit.jsonl")  # parent is a file -> every write fails
    audit = da.DropAudit(num_generations=2, max_completion=16384, path=path)
    DS, COL = da.audited_classes(agt, audit)
    q = queue.Queue()
    rewards = [1.0, float("inf"), 0.0, 1.0, 0.5, 0.0, 1.0, 1.0]
    for i, r in enumerate(rewards):
        q.put(mk(i // 2, 0, 100 * (i + 1), r))
    ds = DS(rollout_queue=q, model_version_fn=lambda: 0, check_health_fn=_raise_stop, stale_after_s=1e9,
            metrics=defaultdict(list), max_staleness=3, poll_interval_s=0.01)
    col = COL(0, 1, metrics=defaultdict(list))
    it = iter(ds)
    sink = defaultdict(list)
    for step in range(1, 5):
        col.torch_call([[[next(it), next(it)]]])  # one micro-batch, one row of two samples
        if step == 3:  # the filesystem comes back: the next flush writes every kept line
            os.remove(blocker)
        audit.flush(step, step, sink, n_trained=1)
    assert audit.enabled and audit.jsonl_failures == 2
    assert sink["drop_audit/jsonl_fail_total"] == [1.0, 1.0, 0.0, 0.0]
    lines = [json.loads(x) for x in open(path) if x.strip()]
    assert [x["type"] for x in lines] == ["header", "step", "step", "step", "step"]
    assert [x["step"] for x in lines[1:]] == [1, 2, 3, 4]
    assert lines[1]["reward"]["trained_samples_sum"] is None  # inf -> null, line kept
    # relaunch into the same file (resume from step 2): last record per step wins
    a2 = da.DropAudit(num_generations=2, max_completion=16384, path=path)
    a2.enabled = True
    for step in (3, 4, 5):
        a2._fifo.append(da._Acc(a2._sizes))
        a2.flush(step, step, None, n_trained=1)
    with open(path, "a") as f:
        f.write('{"step": 9, "trunc')  # a torn line is skipped
    rep = load_report()
    info = {}
    recs = rep.load(path, None, None, False, info)
    assert [r["step"] for r in recs] == [1, 2, 3, 4, 5]
    assert [r["seg"] for r in recs] == [audit.seg, audit.seg, a2.seg, a2.seg, a2.seg]
    assert info["duplicates_replaced"] == 2 and info["bad_lines"] == 1 and len(info["headers"]) == 2
    assert info["headers"][1]["resume_step"] == 2
    assert rep.load(path, 1, 7, False, info) and info["missing_steps"] == [6, 7]
    print("[test_jsonl_robustness] OK (2 failed writes retried, inf -> null, 2 segments deduplicated)")


# ---------------------------------------------------------------------------------------------------- test 9
def bench(n=8192, reps=5):
    rng = random.Random(0)
    pool = []  # shared token lists keep memory flat; count(1) cost is per element, unchanged
    for _ in range(64):
        L = 16384 if rng.random() < 0.1 else int(min(16383, rng.lognormvariate(math.log(4000), 0.7)))
        P = 1500
        pool.append(([1] * (P + L), [0] * P + [1] * L, [0.0] * (P + L)))
    ngen = 32
    samples = []
    for i in range(n):
        ids, cm, lp = pool[(i // ngen) % len(pool)]
        ver = 0 if (i // ngen) % 5 else -9  # every 5th group stale -> dropped
        samples.append(RolloutSample(prompt=[], completion=[], input_ids=list(ids), completion_mask=cm,
                                     old_log_probs=lp, advantage=0.0, model_version=ver, group_id=i // ngen,
                                     metrics={"reward": 1.0}))
    n_kept = sum(s.model_version == 0 for s in samples)

    def run(audited):
        q = queue.Queue()
        for s in samples:
            q.put(s)
        audit = da.DropAudit(num_generations=ngen, max_completion=16384, path=None)
        cls = da.audited_classes(agt, audit)[0] if audited else agt.RolloutQueueDataset
        ds = cls(rollout_queue=q, model_version_fn=lambda: 0, check_health_fn=_raise_stop, stale_after_s=1e9,
                 metrics=defaultdict(list), max_staleness=3, poll_interval_s=0.001)
        it = iter(ds)
        t0 = time.perf_counter()
        items = [next(it) for _ in range(n_kept)]
        dt = time.perf_counter() - t0
        tc = tf = 0.0
        if audited:
            t1 = time.perf_counter()
            for i in range(0, len(items) - 127, 128):  # the collate hook, 128-sample micro-batches of 4 rows
                mb = items[i:i + 128]
                audit.on_collated([[mb[j * 32:(j + 1) * 32] for j in range(4)]])
            tc = time.perf_counter() - t1
            t2 = time.perf_counter()
            audit.flush(1, 1, defaultdict(list))
            tf = time.perf_counter() - t2
            assert audit.enabled and audit._cum["collated"] == len(items) // 128 * 128
        return dt, tc, tf

    def collate_ref():
        # Reference: TRL's own rank-0 collation of the same samples (128-sample micro-batches, 4 rows of 32).
        col = agt.DataCollatorForRollout(0, 4, metrics=defaultdict(list))
        kept = [s for s in samples if s.model_version == 0][:4096]
        items = [{"input_ids": s.input_ids, "completion_mask": s.completion_mask, "old_log_probs": s.old_log_probs,
                  "advantage": 0.0, "group_id": s.group_id, "metrics": s.metrics} for s in kept]
        t0 = time.perf_counter()
        for i in range(0, len(items) - 127, 128):
            mb = items[i:i + 128]
            col.torch_call([[mb[j * 32:(j + 1) * 32] for j in range(4)]])
        return (time.perf_counter() - t0) / len(items)

    base = [run(False)[0] for _ in range(reps)]
    wrapped = [run(True) for _ in range(reps)]
    tb, tw = min(base), min(w[0] for w in wrapped)
    tcol = statistics.median(w[1] for w in wrapped)
    tflush = statistics.median(w[2] for w in wrapped)
    per_sample_us = (tw - tb) / n * 1e6 + tcol / n_kept * 1e6
    per_step_ms = per_sample_us * 1024 / 1e3 + tflush * 1e3 * 1024 / n
    print(f"[bench] {n} pulls ({n - n_kept} stale-dropped), mean completion "
          f"{statistics.mean(s.completion_mask.count(1) for s in samples[::ngen]):.0f} tok, prompt 1500 tok")
    print(f"[bench] unwrapped {tb * 1e3:.1f} ms  wrapped {tw * 1e3:.1f} ms  collate hook {tcol * 1e3:.2f} ms "
          f"-> +{per_sample_us:.2f} us/sample; flush of a {n}-sample step {tflush * 1e3:.2f} ms")
    print(f"[bench] per 1024-sample step: ~{per_step_ms:.2f} ms of audit work "
          f"(= {per_step_ms / 1e3 / 60 * 100:.4f}% of a 60 s step; the hook time inside the queue get is taken "
          "back out of perf/rollout_wait_s)")
    col_us = collate_ref() * 1e6
    print(f"[bench] reference: TRL's DataCollatorForRollout on the same samples costs {col_us:.0f} us/sample "
          f"(~{col_us * 1024 / 1e3:.0f} ms per 1024-sample step) -> audit = {per_sample_us / col_us * 100:.1f}% of "
          "TRL's existing rank-0 per-sample work")
    assert per_step_ms < 200, per_step_ms


if __name__ == "__main__":
    test_hand_scenario()
    test_short_and_interleaved()
    test_fence()
    test_equivalence("FixedCountBatcher")
    test_equivalence("GroupRowBatcher")
    test_dispatcher(0.0, 1)
    test_dispatcher(0.05, 2)
    test_wait_compensation()
    test_install_robustness()
    test_jsonl_robustness()
    bench()
    print("ALL OK")
