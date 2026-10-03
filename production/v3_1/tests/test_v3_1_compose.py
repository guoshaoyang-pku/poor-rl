#!/usr/bin/env python3
"""CPU test for the rlforge_v3_1 merge seam: drop audit + non-blocking scorer on the same TRL queue dataset.

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=src python tests/test_v3_1_compose.py   (plain asserts, no pytest)

Both changes subclass ``trl...async_grpo_trainer.RolloutQueueDataset`` and wrap its ``queue``:
  * rlforge.score_loop.install() rebinds the module global to JudgedStalenessRolloutQueueDataset (at trainer
    start, before the trainer is built), whose queue proxy switches ``max_staleness`` per sample (judged
    compensation);
  * rlforge.drop_audit.install() subclasses whatever the module global is when get_train_dataloader runs and wraps
    the (already proxied) queue in _AuditedQueue.
So the composed dataset is Audited(JudgedStaleness(RolloutQueueDataset)). Checked here, through TRL's own
AsyncGRPOTrainer.get_train_dataloader at production geometry (NGEN=8 x 4 ranks, GroupRowBatcher, STALE=3,
auto cap 5):
  1. the MRO / queue wrapping order is as above and the module globals are restored by drop_audit;
  2. the audit does not change TRL's decisions under the scorer: identical collated batches and identical TRL
     metrics (incl. sample/judged_*) with and without the audit;
  3. TRL's drops == an independent replay of the judged rule (thr = 3 + min(judge_versions, 2) for judged rows);
  4. the audit's arrived/kept/dropped == TRL's, and its only anomaly is decision_mismatch, equal to the judged
     samples kept by the allowance (it predicts from the base max_staleness) -- the documented caveat;
  5. without the scorer (v3_1 defaults: SCORE_CONC=0) the dataset is Audited(RolloutQueueDataset) and
     decision_mismatch == 0.
"""

import importlib.util
import os
import random
import sys
import tempfile
from types import SimpleNamespace

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("tda", os.path.join(HERE, "test_drop_audit.py"))
tda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tda)  # helpers only (its tests run under __main__)

agt = tda.agt
da = tda.da
import rlforge.score_loop as sl  # noqa: E402
from rlforge.prefix_share import GroupRowBatcher  # noqa: E402

JK, JV = sl.JUDGED_KEY, sl.JUDGE_VERSIONS_KEY


def stream(n_groups, ngen, seed, judged_frac=0.25):
    rng = random.Random(seed)
    samples = tda.make_stream(n_groups, ngen, seed)
    judged = {}
    for s in samples:
        if s.group_id not in judged:
            judged[s.group_id] = rng.choice([0, 1, 2, 3, 4]) if rng.random() < judged_frac else None
        jv = judged[s.group_id]
        s.metrics[JK] = 0.0 if jv is None else 1.0
        if jv is not None:
            s.metrics[JV] = float(jv)
    return samples


def run_path(samples, ngen, gas, stale, audited, out):
    """tda.run_trainer_path without its 'module global is the stock class' assertion."""
    import queue
    q = queue.Queue()
    for s in samples:
        q.put(s)
    tr = tda.StandInTrainer(q, ngen, stale, out, gas)
    audit = da.install(tr, path=os.path.join(out, "drop_audit.jsonl")) if audited else None
    assert (audit is not None) == audited
    before = agt.RolloutQueueDataset
    dl = tr.get_train_dataloader()
    assert agt.RolloutQueueDataset is before  # drop_audit's patch is scoped to get_train_dataloader
    tr._rollout_dataset.poll_interval_s = 0.001
    batches, step = [], 0
    it = iter(dl)
    try:
        while True:
            bs = [next(it) for _ in range(gas)]
            batches.extend(bs)
            tr.current_gradient_accumulation_steps = len(bs)
            step += 1
            tr.model_version += 1
            for cb in tr.callbacks:
                cb.on_step_end(tr.args, SimpleNamespace(global_step=step), None)
    except tda.StopFeed:
        pass
    for cb in tr.callbacks:
        cb.on_train_end(tr.args, SimpleNamespace(global_step=step), None)
    return tr, audit, batches, step


def replay(samples, per_step_kept, base, cap):
    v, k, drops, kept_by_allow = 0, 0, 0, 0
    for s in samples:
        thr = base
        if s.metrics.get(JK, 0.0) > 0:
            thr = base + max(0, min(int(s.metrics.get(JV, cap - base)), cap - base))
        st = v - s.model_version
        if st > thr:
            drops += 1
            continue
        if st > base:
            kept_by_allow += 1
        k += 1
        if k == per_step_kept:
            v, k = v + 1, 0
    return drops, kept_by_allow


def anomalies(audit_path):
    rep = tda.load_report()
    recs = rep.load(audit_path, None, None, True)
    return rep.aggregate(recs)


def main():
    ngen, gas, stale, nproc = 8, 2, 3, 4
    per_step = ngen * nproc * gas
    samples = stream(160, ngen, seed=11)
    orig_fcb = agt.FixedCountBatcher
    agt.FixedCountBatcher = GroupRowBatcher
    tmp = tempfile.mkdtemp()
    try:
        # --- 5. v3_1 defaults: scorer not installed -----------------------------------------------------
        tr0, audit0, _, _ = run_path(samples, ngen, gas, stale, True, os.path.join(tmp, "noscore"))
        mro0 = [c.__name__ for c in type(tr0._rollout_dataset).__mro__[:2]]
        assert mro0 == ["AuditedRolloutQueueDataset", "RolloutQueueDataset"], mro0
        agg0 = anomalies(os.path.join(tmp, "noscore", "drop_audit.jsonl"))
        assert sum(agg0["anomalies"].values()) == 0, agg0["anomalies"]
        print(f"[defaults] scorer off: dataset {mro0}, dropped {agg0['samples']['dropped']}, anomalies 0  OK")

        # --- scorer installed exactly as the trainer does for SCORE_CONC=32, JUDGED_STALE empty ------------
        assert sl.install(score_concurrency=32, judged_max_staleness="auto", early_hooks=False,
                          max_staleness=stale, score_task_max_s=None)
        cap = sl.JudgedStalenessRolloutQueueDataset.judged_max_staleness
        assert cap == stale + 2 and agt.RolloutQueueDataset is sl.JudgedStalenessRolloutQueueDataset
        ta, _, ba, sa = run_path(samples, ngen, gas, stale, False, os.path.join(tmp, "a"))
        tb, audit, bb, sb = run_path(samples, ngen, gas, stale, True, os.path.join(tmp, "b"))

        # 1. composition order
        ds = tb._rollout_dataset
        mro = [c.__name__ for c in type(ds).__mro__[:3]]
        assert mro == ["AuditedRolloutQueueDataset", "JudgedStalenessRolloutQueueDataset", "RolloutQueueDataset"], mro
        assert isinstance(ds.queue, da._AuditedQueue) and isinstance(ds.queue._q, sl._JudgedStalenessQueue)
        assert type(ta._rollout_dataset) is sl.JudgedStalenessRolloutQueueDataset
        # 2. audit is observe-only under the scorer
        assert sa == sb > 0
        tda.compare_runs(ta, ba, tb, bb)
        m = ta._metrics["train"]
        n_drop = len(m["sample/dropped_stale_total"])
        n_allow = len(m["sample/judged_kept_by_allowance_total"])
        n_jdrop = len(m["sample/judged_dropped_stale_total"])
        # 3. TRL's decisions == the judged rule
        r_drop, r_allow = replay(samples, per_step, stale, cap)
        assert (n_drop, n_allow) == (r_drop, r_allow), ((n_drop, n_allow), (r_drop, r_allow))
        assert n_allow > 0 and n_jdrop > 0 and n_drop > n_jdrop, (n_allow, n_jdrop, n_drop)
        # 4. audit totals and its one expected anomaly
        agg = anomalies(os.path.join(tmp, "b", "drop_audit.jsonl"))
        assert agg["samples"]["arrived"] == len(samples)
        assert agg["samples"]["dropped"] == n_drop, (agg["samples"], n_drop)
        an = dict(agg["anomalies"])
        assert an.pop("decision_mismatch") == n_allow, (agg["anomalies"], n_allow)
        assert sum(an.values()) == 0, an
        print(f"[scorer on] dataset {mro}; queue _AuditedQueue(_JudgedStalenessQueue); {sb} steps, "
              f"{len(samples)} samples, TRL dropped {n_drop} (judged {n_jdrop}), judged kept by allowance {n_allow} "
              f"== replay; batches + TRL metrics identical with/without audit; audit decision_mismatch "
              f"{agg['anomalies']['decision_mismatch']} == kept-by-allowance, other anomalies 0  OK")
    finally:
        agt.FixedCountBatcher = orig_fcb
        sl.uninstall()
    assert agt.RolloutQueueDataset.__name__ == "RolloutQueueDataset"
    assert agt.AsyncRolloutWorker.__name__ == "AsyncRolloutWorker"
    print("ALL OK")


if __name__ == "__main__":
    main()
