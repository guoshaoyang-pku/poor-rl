"""Length-bucketed stale-drop audit for TRL's async GRPO trainer (rlforge_v3_next, 2026-10-03, schema 2).

Question: does the staleness filter remove long chains of thought preferentially? TRL stamps a group's
``model_version`` when its FIRST member is dispatched (``AsyncRolloutWorker._generate_loop``) and scores and
pushes the group only when its SLOWEST member finishes (``_score_loop``: all rows back to back). The trainer
drops a sample in ``RolloutQueueDataset.__iter__`` when ``model_version - sample.model_version > max_staleness``.
So long groups reach the trainer older and are more likely to be dropped whole.

The audit only OBSERVES. TRL makes every decision and the audit never sees it directly:
  * ``_AuditedQueue`` wraps the queue that ``RolloutQueueDataset`` reads. Each ``get`` first resolves the
    sample pulled before it as DROPPED (TRL pulled again without yielding it), then records the new sample
    as pending.
  * ``RolloutQueueDataset.__iter__`` is wrapped in a pass-through generator. It resolves the pending sample
    as KEPT when TRL yields it, checking identity: the yielded dict holds the sample's own ``input_ids``
    list. The audit predicts TRL's rule from the same version function and counts any disagreement as
    ``decision_mismatch`` (expected 0).
  * ``DataCollatorForRollout.torch_call`` closes one micro-batch: the rows it is given are exactly the samples
    that micro-batch trains (the collator drops nothing).
TRL still produces the yielded objects, their order, the batching and the drop decision. The audit reads
``completion_mask``, ``model_version``, ``group_id``, ``input_ids`` (identity only) and ``metrics["reward"]``.
The one thing it writes is the dataset's ``wait_s`` accumulator (a perf metric, never read by training): it
subtracts its own hook time, which runs inside TRL's queue-wait timer, so ``perf/rollout_wait_s`` stays what
it would be without the audit. All hooks run where the queue is consumed (rank 0, main thread,
``num_workers=0``). Every hook is fenced, so an audit bug turns the audit off and leaves training running;
an error while installing leaves TRL's own dataloader in place.

Per-step alignment (exact):
  * Every sample pulled from the queue is attributed to the micro-batch being built when it was pulled (kept
    and dropped alike), and that micro-batch is closed when the collator receives it. Closed micro-batches
    wait in a FIFO; at optimizer step N's end the audit pops the ``current_gradient_accumulation_steps``
    micro-batches step N trained (rank 0 collates in exactly the order HF consumes). Accelerate's dispatcher
    prefetches one micro-batch, so one closed micro-batch stays in the FIFO at every step end (``prefetched``
    = 1; it is step N+1's first micro-batch). The audit pops by count, so the prefetch never leaks into
    step N's numbers.
  * So step N's record describes exactly the samples step N trained ("trained", from the collated rows) and
    the samples TRL dropped while building them. generated = trained + dropped exactly in the production
    config (fixed-count batchers, ``token_budget == 0``), and summed over steps trained == steps x
    samples/step exactly. The micro-batch prefetched during step N-1 was filled at the version before the
    last weight sync, so a step's drop decisions use two versions (``decision_versions`` = [min, max]).
  * With ``token_budget > 0`` the planner can carry a pulled sample into the next micro-batch or drop an
    oversized one; "trained" stays exact (collated rows), kept/dropped attribution can shift by those samples.

Definitions (bucket edges in tokens, k = 1024):
  * sample length = completion tokens = ``completion_mask.count(1)`` (equal to ``len(completion_ids)`` for
    single-turn rollouts). A sample is truncated when its length is at least the cap (``--max-completion``),
    which is the reward functions' rule (``n_tokens >= cap`` -> -2). ``RolloutSample`` has no finish_reason.
  * buckets: lt2k [0, 2k), 2k_4k, 4k_8k, 8k_16k [8k, cap), trunc (>= cap). Edges at or above the cap are
    dropped (a short-cap smoke run gets fewer buckets instead of an error).
  * group bucketings, all from the same groups:
      - ``buckets``: by the group's LONGEST member, trunc if any member is truncated (the coordinator's spec).
        At G=32 this is dominated by the longest of 32 draws: lt2k is nearly empty and trunc is mostly groups
        of short completions plus one straggler at the cap. When such a group is dropped its short members go
        with it, so the dropped SAMPLE length histogram looks short-dominated even when long members cause the
        drop. Read it together with the next two.
      - ``buckets_median``: by the group's median member length (lower median).
      - ``buckets_ntrunc``: by the number of truncated members: t0 (none), t1 (one), t2p (two or more).
  * sample histograms use each sample's own length.
  * group outcome: kept_full (no member dropped), dropped_full (all members dropped), partial (some
    members dropped). A group is finalized once ``num_generations`` members have arrived, or when the next
    group starts (the scorer pushes groups contiguously); a group cut short this way is flagged ``short``.
    Group counts land in the step whose micro-batches were being built when the group's last member arrived.
    In production (micro-batch = whole groups, versions change only at step ends) ``partial`` is ~0 and the
    length bias shows up as ``dropped_full``; non-zero ``partial`` means groups with != 32 rows.
  * lag = staleness of a group's first member when it reached the trainer (version at pull minus
    ``sample.model_version``).

Anomalies (``drop_audit/anomalies_total`` must stay 0 in production): decision_mismatch, identity_mismatch,
kept_without_pending, late_samples (rows for an already-closed group), collate_unmatched (a collated row
the audit never saw kept), kept_meta_evicted, fifo_underrun / fifo_overrun (collated micro-batches vs what
the step trained + the one prefetched), and, for ``token_budget == 0`` only, kept_ne_trained (per step) and
backlog_nonzero (cumulative kept - collated != 0 at the step end).

Output per optimizer step: ``drop_audit/...`` entries in the trainer's metric sink (counts named ``*_total``,
fractions and means as ``(numerator, denominator)`` pairs, so TRL's ``_reduce_metric`` stays right for any
``logging_steps``), plus one JSON line in ``path`` (full detail and additive sums; see
scripts/drop_audit_report.py). The file starts each process's segment with a ``"type": "header"`` line
(pid, host, start time, resume step, config); a relaunch into the same file appends a new segment and the
report keeps the last record per step. A failed write is retried at the next flush (the unwritten lines are
kept) and counted in ``drop_audit/jsonl_fail_total``; non-finite floats are written as null.
"""

import bisect
import json
import math
import os
import socket
from collections import deque
from time import perf_counter, time

SCHEMA = 2
EDGES = (2048, 4096, 8192)
FINE_BIN = 512  # resolution of the additive length histogram in the jsonl (range percentiles offline)
NTRUNC_NAMES = ("t0", "t1", "t2p")
_DONE_MEMORY = 8192  # finalized group ids remembered for late-arrival detection
_META_MEMORY = 65536  # kept samples awaiting collation (production: at most one micro-batch)
_UNWRITTEN_MAX = 4096  # jsonl lines kept for retry after failed writes

_GROUP_KEYS = (
    "groups", "kept_full", "dropped_full", "partial", "short",
    "samples", "samples_dropped", "samples_dropped_partial",
    "r_kept_full_sum", "r_kept_full_n", "r_dropped_full_sum", "r_dropped_full_n",
    "r_partial_kept_sum", "r_partial_kept_n", "r_partial_dropped_sum", "r_partial_dropped_n",
    "lag_sum", "lag_n",
)
_SAMPLE_ANOMALIES = ("decision_mismatch", "identity_mismatch", "kept_without_pending", "late_samples",
                     "collate_unmatched", "kept_meta_evicted")


def bucket_names(edges=EDGES, cap=16384):
    names, lo = [], 0
    for e in edges:
        names.append(f"lt{e // 1024}k" if lo == 0 else f"{lo // 1024}k_{e // 1024}k")
        lo = e
    names.append(f"lt{cap // 1024}k" if lo == 0 else f"{lo // 1024}k_{cap // 1024}k")
    names.append("trunc")
    return names


def nearest_rank(sorted_vals, q):
    """Nearest-rank percentile: the smallest value with at least q of the data at or below it."""
    if not sorted_vals:
        return None
    k = max(1, math.ceil(q * len(sorted_vals)))
    return sorted_vals[min(k, len(sorted_vals)) - 1]


def _mean(s, n):
    return s / n if n else None


def _finite(x):
    """JSON-safe copy: non-finite floats become None."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


class _Group:
    __slots__ = ("lens", "n_drop", "ntrunc", "rk", "nk", "rd", "nd", "lag")

    def __init__(self, lag):
        self.lens = []
        self.n_drop = self.ntrunc = self.nk = self.nd = 0
        self.rk = self.rd = 0.0
        self.lag = lag


class _Acc:
    """Additive accounting of one micro-batch: the pulls made while building it and the rows it collated.
    A step record merges the micro-batches that step trained."""

    __slots__ = ("kept", "drop", "trained", "rk", "rd", "rt", "stale_k", "stale_d", "vmin", "vmax", "cnt", "g",
                 "n_mb", "hook_s")

    def __init__(self, sizes):
        self.kept, self.drop, self.trained = [], [], []  # sample lengths
        self.rk, self.rd, self.rt = [0.0, 0], [0.0, 0], [0.0, 0]  # reward sum / count (NaN rewards skipped)
        self.stale_k, self.stale_d = {}, {}
        self.vmin = self.vmax = None
        self.cnt = dict.fromkeys(_SAMPLE_ANOMALIES, 0)
        self.g = {kind: [dict.fromkeys(_GROUP_KEYS, 0) for _ in range(n)] for kind, n in sizes.items()}
        self.n_mb = 0
        self.hook_s = 0.0

    def merge(self, o):
        self.kept += o.kept
        self.drop += o.drop
        self.trained += o.trained
        for a, b in ((self.rk, o.rk), (self.rd, o.rd), (self.rt, o.rt)):
            a[0] += b[0]
            a[1] += b[1]
        for d, e in ((self.stale_k, o.stale_k), (self.stale_d, o.stale_d)):
            for k, v in e.items():
                d[k] = d.get(k, 0) + v
        if o.vmin is not None and (self.vmin is None or o.vmin < self.vmin):
            self.vmin = o.vmin
        if o.vmax is not None and (self.vmax is None or o.vmax > self.vmax):
            self.vmax = o.vmax
        for k, v in o.cnt.items():
            self.cnt[k] = self.cnt.get(k, 0) + v
        for kind, rows in o.g.items():
            for a, b in zip(self.g[kind], rows):
                for k in _GROUP_KEYS:
                    a[k] += b[k]
        self.n_mb += o.n_mb
        self.hook_s += o.hook_s


def _add_group(a, g, n):
    a["groups"] += 1
    a["samples"] += n
    a["samples_dropped"] += g.n_drop
    if g.n_drop == 0:
        a["kept_full"] += 1
        a["r_kept_full_sum"] += g.rk
        a["r_kept_full_n"] += g.nk
    elif g.n_drop == n:
        a["dropped_full"] += 1
        a["r_dropped_full_sum"] += g.rd
        a["r_dropped_full_n"] += g.nd
    else:
        a["partial"] += 1
        a["samples_dropped_partial"] += g.n_drop
        a["r_partial_kept_sum"] += g.rk
        a["r_partial_kept_n"] += g.nk
        a["r_partial_dropped_sum"] += g.rd
        a["r_partial_dropped_n"] += g.nd
    if g.lag is not None:
        a["lag_sum"] += g.lag
        a["lag_n"] += 1


class DropAudit:
    def __init__(self, *, num_generations, max_completion, path=None, edges=EDGES):
        self.ngen = int(num_generations)
        self.cap = int(max_completion)
        if self.cap <= 0:
            raise ValueError(f"max_completion must be positive, got {self.cap}")
        self.edges = tuple(sorted({int(e) for e in edges if 0 < int(e) < self.cap}))
        self.names = bucket_names(self.edges, self.cap)
        self.nb = len(self.names)
        self._sizes = {"max": self.nb, "median": self.nb, "ntrunc": len(NTRUNC_NAMES)}
        self.n_fine = -(-self.cap // FINE_BIN) + 1  # last fine bin = truncated
        self.path = path
        self.enabled = False  # set when bound to a queue-consuming dataset (rank 0 only)
        self.version_fn = None
        self.max_staleness = None
        self.token_budget = None
        self.exact_batcher = None  # token_budget == 0: each kept sample is collated in the micro-batch it completes
        self.gas = None
        self.config_extra = {}
        self._pending = None
        self._groups = {}
        self._last_gid = None
        self._done, self._done_q = set(), deque()
        self._meta = {}  # id(input_ids) -> (input_ids, length, reward) of kept samples not yet collated
        self._cur = _Acc(self._sizes)  # micro-batch being built
        self._fifo = deque()  # collated micro-batches not yet attributed to a step
        self._cum = {"arrived": 0, "kept": 0, "dropped": 0, "collated": 0, "trained": 0}
        self.seg = f"{os.getpid()}-{int(time() * 1000)}-{os.urandom(2).hex()}"
        self.t_start = time()
        self._t_last = self.t_start
        self._unwritten = deque()
        self._header_queued = False
        self._need_sep = False
        self.jsonl_failures = 0
        self.flushes = 0

    # ------------------------------------------------------------------ wiring
    def bind(self, version_fn, max_staleness):
        self.version_fn = version_fn
        self.max_staleness = None if max_staleness is None else int(max_staleness)
        self.enabled = True

    def configure(self, token_budget=None, gas=None):
        self.token_budget = token_budget
        self.exact_batcher = token_budget is not None and token_budget <= 0
        self.gas = gas

    def _fail(self, where, exc):
        self.enabled = False
        print(f"[rlforge][drop_audit] DISABLED after error in {where}: {type(exc).__name__}: {exc} "
              "(training unaffected)", flush=True)

    def bucket(self, length, trunc):
        return self.nb - 1 if trunc else bisect.bisect_right(self.edges, length)

    @staticmethod
    def compensate(owner, seconds):
        """Take the audit's own time back out of TRL's queue-wait accumulator (perf/rollout_wait_s)."""
        if owner is not None and seconds > 0:
            try:
                owner.wait_s -= seconds
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ hooks
    def before_pull(self):
        """The dataset is about to pull again: a still-pending sample was not yielded, so TRL dropped it."""
        if self._pending is None or not self.enabled:
            return
        t0 = perf_counter()
        try:
            self._resolve(dropped=True)
        except Exception as e:  # noqa: BLE001
            self._fail("before_pull", e)
        self._cur.hook_s += perf_counter() - t0

    def on_pull(self, sample):
        if not self.enabled:
            return
        t0 = perf_counter()
        try:
            self._arrive(sample)
        except Exception as e:  # noqa: BLE001
            self._fail("on_pull", e)
        self._cur.hook_s += perf_counter() - t0

    def on_yield(self, item):
        if not self.enabled:
            return
        t0 = perf_counter()
        try:
            p = self._pending
            if p is None:
                self._cur.cnt["kept_without_pending"] += 1
            else:
                if item.get("input_ids") is not p[0].input_ids:
                    self._cur.cnt["identity_mismatch"] += 1
                self._resolve(dropped=False)
        except Exception as e:  # noqa: BLE001
            self._fail("on_yield", e)
        self._cur.hook_s += perf_counter() - t0

    def on_collated(self, examples):
        """The collator received one micro-batch: its rows are what that micro-batch trains."""
        if not self.enabled:
            return
        t0 = perf_counter()
        try:
            (rows,) = examples
            acc = self._cur
            n = 0
            for row in rows:
                for ex in row:
                    meta = self._meta.pop(id(ex["input_ids"]), None)
                    if meta is None:
                        acc.cnt["collate_unmatched"] += 1
                        length = ex["completion_mask"].count(1)
                        r = (ex.get("metrics") or {}).get("reward")
                        r = float(r) if r is not None else math.nan
                    else:
                        _, length, r = meta
                    acc.trained.append(length)
                    if r == r:
                        acc.rt[0] += r
                        acc.rt[1] += 1
                    n += 1
            acc.n_mb += 1
            self._cum["collated"] += n
            acc.hook_s += perf_counter() - t0
            self._fifo.append(acc)
            self._cur = _Acc(self._sizes)
        except Exception as e:  # noqa: BLE001
            self._fail("on_collated", e)

    # ------------------------------------------------------------------ accounting
    def _arrive(self, sample):
        length = sample.completion_mask.count(1)
        trunc = length >= self.cap
        m = sample.metrics or {}
        r = m.get("reward")
        r = float(r) if r is not None else math.nan
        acc = self._cur
        stale = pred = None
        if self.version_fn is not None:
            v = int(self.version_fn())
            stale = v - int(sample.model_version)
            if acc.vmin is None or v < acc.vmin:
                acc.vmin = v
            if acc.vmax is None or v > acc.vmax:
                acc.vmax = v
            if self.max_staleness is not None:
                pred = stale > self.max_staleness
        gid = sample.group_id
        if gid != self._last_gid:
            # The scorer pushes a group's rows contiguously, so a new id closes the previous group (only
            # still open if it had fewer than num_generations rows).
            if self._last_gid in self._groups:
                self._finalize(self._last_gid)
            self._last_gid = gid
        g = None
        if gid in self._done:
            acc.cnt["late_samples"] += 1
        else:
            g = self._groups.get(gid)
            if g is None:
                g = self._groups[gid] = _Group(lag=stale)
            g.lens.append(length)
            if trunc:
                g.ntrunc += 1
        self._pending = (sample, length, r, stale, pred, gid, g)

    def _resolve(self, dropped):
        sample, length, r, stale, pred, gid, g = self._pending
        self._pending = None
        acc = self._cur
        ok = r == r  # not NaN
        self._cum["arrived"] += 1
        if dropped:
            self._cum["dropped"] += 1
            acc.drop.append(length)
            if ok:
                acc.rd[0] += r
                acc.rd[1] += 1
            hist = acc.stale_d
        else:
            self._cum["kept"] += 1
            acc.kept.append(length)
            if ok:
                acc.rk[0] += r
                acc.rk[1] += 1
            hist = acc.stale_k
            ids = sample.input_ids
            self._meta[id(ids)] = (ids, length, r)  # holding ids keeps its id() unique until collated
            if len(self._meta) > _META_MEMORY:
                self._meta.pop(next(iter(self._meta)))
                acc.cnt["kept_meta_evicted"] += 1
        if stale is not None:
            hist[stale] = hist.get(stale, 0) + 1
        if pred is not None and pred != dropped:
            acc.cnt["decision_mismatch"] += 1
        if g is not None:
            if dropped:
                g.n_drop += 1
                if ok:
                    g.rd += r
                    g.nd += 1
            elif ok:
                g.rk += r
                g.nk += 1
            if len(g.lens) >= self.ngen:
                self._finalize(gid)

    def _finalize(self, gid):
        g = self._groups.pop(gid)
        self._done.add(gid)
        self._done_q.append(gid)
        if len(self._done_q) > _DONE_MEMORY:
            self._done.discard(self._done_q.popleft())
        lens = sorted(g.lens)
        n = len(lens)
        mx = lens[-1] if lens else 0
        med = nearest_rank(lens, 0.5) or 0
        acc = self._cur
        for kind, idx in (("max", self.bucket(mx, g.ntrunc > 0)), ("median", self.bucket(med, med >= self.cap)),
                          ("ntrunc", min(g.ntrunc, 2))):
            a = acc.g[kind][idx]
            if n < self.ngen:
                a["short"] += 1
            _add_group(a, g, n)

    # ------------------------------------------------------------------ per-step output
    def _len_stats(self, lens):
        s = sorted(lens)
        hist = [0] * self.nb
        fine = [0] * self.n_fine
        for x in s:
            t = x >= self.cap
            hist[self.bucket(x, t)] += 1
            fine[self.n_fine - 1 if t else min(x // FINE_BIN, self.n_fine - 2)] += 1
        return {"n": len(s), "sum": sum(s), "mean": _mean(sum(s), len(s)),
                "p50": nearest_rank(s, 0.5), "p90": nearest_rank(s, 0.9), "max": s[-1] if s else None,
                "hist": dict(zip(self.names, hist)), "fine": fine}

    def record(self, step, model_version, final=False, n_trained=None):
        """Build step ``step``'s record from the ``n_trained`` oldest collated micro-batches (all, if None).
        ``final``: train end; everything left (prefetched and partly built micro-batches) goes into a record
        that is not trained."""
        anomalies = {}
        if final:
            for gid in list(self._groups):  # no more rows will arrive: close open groups (short)
                self._finalize(gid)
            take = list(self._fifo) + [self._cur]
            self._fifo.clear()
            self._cur = _Acc(self._sizes)
            anomalies["pending_at_end"] = int(self._pending is not None)
        else:
            n = len(self._fifo) if n_trained is None else int(n_trained)
            k = min(n, len(self._fifo))
            take = [self._fifo.popleft() for _ in range(k)]
            anomalies["fifo_underrun"] = n - k
            anomalies["fifo_overrun"] = max(0, len(self._fifo) - 1)  # beyond the dispatcher's 1 prefetch
        m = _Acc(self._sizes)
        for a in take:
            m.merge(a)
        untrained = []
        if final:
            untrained, m.trained = m.trained, []
        kept, drop, trained = len(m.kept), len(m.drop), len(m.trained)
        self._cum["trained"] += trained
        backlog = self._cum["kept"] - self._cum["collated"]
        if not final and self.exact_batcher:
            anomalies["kept_ne_trained"] = abs(kept - trained)
            anomalies["backlog_nonzero"] = int(backlog != 0)
        anomalies.update(m.cnt)
        tot = dict.fromkeys(_GROUP_KEYS, 0)
        for a in m.g["max"]:
            for k in _GROUP_KEYS:
                tot[k] += a[k]
        now = time()
        rec = {
            "type": "step", "schema": SCHEMA, "seg": self.seg,
            "step": int(step), "model_version": None if model_version is None else int(model_version),
            "final": bool(final), "time": round(now, 3), "step_s": round(now - self._t_last, 3),
            "microbatches": m.n_mb, "prefetched": len(self._fifo),
            "decision_versions": [m.vmin, m.vmax], "max_staleness": self.max_staleness,
            "num_generations": self.ngen, "cap": self.cap, "edges": list(self.edges), "fine_bin": FINE_BIN,
            "samples": {"arrived": kept + drop, "kept": kept, "dropped": drop, "trained": trained,
                        "drop_frac": _mean(drop, kept + drop)},
            "cum": dict(self._cum, backlog=backlog),
            "buckets": {name: self._bucket_view(a) for name, a in zip(self.names, m.g["max"])},
            "buckets_median": {name: self._bucket_view(a) for name, a in zip(self.names, m.g["median"])},
            "buckets_ntrunc": {name: self._bucket_view(a) for name, a in zip(NTRUNC_NAMES, m.g["ntrunc"])},
            "groups": self._bucket_view(tot),
            "len": {"generated": self._len_stats(m.kept + m.drop), "trained": self._len_stats(m.trained),
                    "dropped": self._len_stats(m.drop)},
            "reward": {
                "trained_samples": _mean(*m.rt), "kept_samples": _mean(*m.rk), "dropped_samples": _mean(*m.rd),
                "trained_samples_sum": m.rt[0], "trained_samples_n": m.rt[1],
                "kept_samples_sum": m.rk[0], "kept_samples_n": m.rk[1],
                "dropped_samples_sum": m.rd[0], "dropped_samples_n": m.rd[1],
                "trained_groups": _mean(tot["r_kept_full_sum"], tot["r_kept_full_n"]),
                "dropped_groups": _mean(tot["r_dropped_full_sum"], tot["r_dropped_full_n"]),
                "partial_groups": _mean(tot["r_partial_kept_sum"] + tot["r_partial_dropped_sum"],
                                        tot["r_partial_kept_n"] + tot["r_partial_dropped_n"]),
            },
            "staleness_at_arrival": {"kept": {str(k): v for k, v in sorted(m.stale_k.items())},
                                     "dropped": {str(k): v for k, v in sorted(m.stale_d.items())}},
            "open_groups": len(self._groups), "open_samples": sum(len(g.lens) for g in self._groups.values()),
            "open_mb_samples": len(self._cur.kept) + len(self._cur.drop),
            "anomalies": anomalies,
            "hook_ms": round(m.hook_s * 1e3, 3),
            "jsonl_failures": self.jsonl_failures,
        }
        if final:
            rec["untrained_collated"] = len(untrained)
        self._t_last = now
        return rec

    @staticmethod
    def _bucket_view(a):
        v = dict(a)
        v["drop_frac"] = _mean(a["samples_dropped"], a["samples"])
        v["group_drop_frac"] = _mean(a["dropped_full"], a["groups"])
        v["partial_frac"] = _mean(a["partial"], a["groups"])
        v["reward_trained"] = _mean(a["r_kept_full_sum"], a["r_kept_full_n"])
        v["reward_dropped"] = _mean(a["r_dropped_full_sum"], a["r_dropped_full_n"])
        v["reward_partial"] = _mean(a["r_partial_kept_sum"] + a["r_partial_dropped_sum"],
                                    a["r_partial_kept_n"] + a["r_partial_dropped_n"])
        v["lag_mean"] = _mean(a["lag_sum"], a["lag_n"])
        return v

    def emit_metrics(self, rec, sink, jsonl_failed=False):
        """Append one step's compact view to the trainer's metric sink (reduced by TRL's ``_reduce_metric``)."""
        P = "drop_audit/"

        def pair(key, num, den):
            if den:
                sink[key].append((float(num), float(den)))

        def val(key, x):
            sink[key].append(float(x))

        for name in self.names:
            b = rec["buckets"][name]
            p = f"{P}{name}/"
            val(p + "groups_total", b["groups"])
            val(p + "groups_dropped_total", b["dropped_full"])
            val(p + "groups_partial_total", b["partial"])
            val(p + "samples_total", b["samples"])
            val(p + "samples_dropped_total", b["samples_dropped"])
            pair(p + "drop_frac", b["samples_dropped"], b["samples"])
            pair(p + "group_drop_frac", b["dropped_full"], b["groups"])
            pair(p + "reward_trained", b["r_kept_full_sum"], b["r_kept_full_n"])
            pair(p + "reward_dropped", b["r_dropped_full_sum"], b["r_dropped_full_n"])
            pair(p + "lag_mean", b["lag_sum"], b["lag_n"])
        for fam, key, names in (("by_median", "buckets_median", self.names),
                                ("by_ntrunc", "buckets_ntrunc", NTRUNC_NAMES)):
            for name in names:
                b = rec[key][name]
                p = f"{P}{fam}/{name}/"
                val(p + "groups_total", b["groups"])
                val(p + "groups_dropped_total", b["dropped_full"])
                pair(p + "drop_frac", b["samples_dropped"], b["samples"])
        s, g = rec["samples"], rec["groups"]
        val(P + "samples_arrived_total", s["arrived"])
        val(P + "samples_kept_total", s["kept"])
        val(P + "samples_dropped_total", s["dropped"])
        val(P + "samples_trained_total", s["trained"])
        pair(P + "drop_frac", s["dropped"], s["arrived"])
        val(P + "groups_total", g["groups"])
        val(P + "groups_dropped_total", g["dropped_full"])
        val(P + "groups_partial_total", g["partial"])
        val(P + "groups_open", rec["open_groups"])
        for kind in ("generated", "trained"):
            ls = rec["len"][kind]
            pair(f"{P}len_{kind}_mean", ls["sum"], ls["n"])
            if ls["n"]:
                val(f"{P}len_{kind}_p50", ls["p50"])
                val(f"{P}len_{kind}_p90", ls["p90"])
                val(f"{P}len_{kind}_max", ls["max"])
            for name, c in ls["hist"].items():
                val(f"{P}len_hist/{name}/{kind}_total", c)
        rw = rec["reward"]
        pair(P + "reward_trained_samples", rw["trained_samples_sum"], rw["trained_samples_n"])
        pair(P + "reward_dropped_samples", rw["dropped_samples_sum"], rw["dropped_samples_n"])
        pair(P + "reward_trained_groups", g["r_kept_full_sum"], g["r_kept_full_n"])
        pair(P + "reward_dropped_groups", g["r_dropped_full_sum"], g["r_dropped_full_n"])
        pair(P + "reward_partial_groups", g["r_partial_kept_sum"] + g["r_partial_dropped_sum"],
             g["r_partial_kept_n"] + g["r_partial_dropped_n"])
        val(P + "anomalies_total", sum(rec["anomalies"].values()))
        val(P + "backlog", rec["cum"]["backlog"])
        val(P + "jsonl_fail_total", int(bool(jsonl_failed)))
        val(P + "hook_ms", rec["hook_ms"])

    # ------------------------------------------------------------------ jsonl
    @staticmethod
    def _dumps(rec):
        try:
            return json.dumps(rec, allow_nan=False)
        except ValueError:  # a non-finite float somewhere: write it as null rather than lose the line
            return json.dumps(_finite(rec), allow_nan=False)

    def _header(self, rec):
        return {
            "type": "header", "schema": SCHEMA, "seg": self.seg, "pid": os.getpid(), "host": socket.gethostname(),
            "start_time": round(self.t_start, 3), "first_step": None if rec["final"] else rec["step"],
            "resume_step": None if rec["final"] else rec["step"] - 1, "path": self.path,
            "config": dict({"cap": self.cap, "edges": list(self.edges), "names": self.names, "fine_bin": FINE_BIN,
                            "num_generations": self.ngen, "max_staleness": self.max_staleness,
                            "gradient_accumulation_steps": self.gas, "token_budget": self.token_budget,
                            "exact_batcher": self.exact_batcher,
                            "env": {k: os.environ.get(k) for k in ("RLFORGE_DROP_AUDIT", "RLFORGE_DROP_AUDIT_PATH",
                                                                   "RUN_NAME", "RLFORGE_V3")}},
                           **self.config_extra),
        }

    def _write(self, rec):
        """Append ``rec`` (and any lines a failed write left behind). Returns False if this write failed."""
        if not self.path:
            return True
        try:
            if not self._header_queued:
                self._unwritten.append(self._dumps(self._header(rec)))
                self._header_queued = True
            self._unwritten.append(self._dumps(rec))
            while len(self._unwritten) > _UNWRITTEN_MAX:
                self._unwritten.popleft()
            data = ("\n" if self._need_sep else "") + "\n".join(self._unwritten) + "\n"
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(data)
        except Exception as e:  # noqa: BLE001
            self.jsonl_failures += 1
            self._need_sep = True  # a partial line may be on disk: start the retry on a fresh line
            if self.jsonl_failures <= 3 or self.jsonl_failures % 100 == 0:
                print(f"[rlforge][drop_audit] jsonl write to {self.path} failed ({type(e).__name__}: {e}); "
                      f"{len(self._unwritten)} line(s) kept for retry, failures={self.jsonl_failures}", flush=True)
            return False
        self._unwritten.clear()
        self._need_sep = False
        return True

    def flush(self, step, model_version, sink=None, final=False, n_trained=None):
        """Close step ``step``: build its record, append the jsonl line, push metrics (unless final)."""
        if not self.enabled:
            return None
        try:
            rec = self.record(step, model_version, final=final, n_trained=n_trained)
        except Exception as e:  # noqa: BLE001
            self._fail("flush", e)
            return None
        ok = self._write(rec)
        if sink is not None and not final:
            try:
                self.emit_metrics(rec, sink, jsonl_failed=not ok)
            except Exception as e:  # noqa: BLE001
                self._fail("emit_metrics", e)
                return None
        self.flushes += 1
        return rec


class _AuditedQueue:
    """Pass-through view of the rollout queue: ``get``/``qsize``/anything else go to the real queue."""

    __slots__ = ("_q", "_audit", "_owner")

    def __init__(self, q, audit, owner=None):
        self._q = q
        self._audit = audit
        self._owner = owner

    def get(self, *args, **kwargs):
        audit, t0 = self._audit, time()
        audit.before_pull()
        hook = time() - t0
        try:
            sample = self._q.get(*args, **kwargs)  # queue.Empty propagates unchanged
            t1 = time()
            audit.on_pull(sample)
            hook += time() - t1
            return sample
        finally:
            audit.compensate(self._owner, hook)

    def qsize(self):
        return self._q.qsize()

    def __getattr__(self, name):
        if name in _AuditedQueue.__slots__:  # not yet set (e.g. half-built copy): no recursion
            raise AttributeError(name)
        return getattr(self._q, name)


def audited_classes(agt, audit):
    """Subclasses of TRL's queue dataset / collator that feed ``audit`` and otherwise defer to TRL."""
    base_ds, base_col = agt.RolloutQueueDataset, agt.DataCollatorForRollout

    class AuditedRolloutQueueDataset(base_ds):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            try:
                wrapped = _AuditedQueue(self.queue, audit, self)
                audit.bind(self.model_version_fn, self.max_staleness)
                self.queue = wrapped
            except Exception as e:  # noqa: BLE001
                audit._fail("dataset init", e)

        def __iter__(self):
            it = super().__iter__()
            try:
                for item in it:
                    audit.on_yield(item)
                    yield item
            finally:
                it.close()

    class AuditedDataCollatorForRollout(base_col):
        def torch_call(self, examples):
            audit.on_collated(examples)
            return super().torch_call(examples)

    return AuditedRolloutQueueDataset, AuditedDataCollatorForRollout


def _make_callback(trainer, audit):
    from transformers import TrainerCallback

    def n_trained():
        n = getattr(trainer, "current_gradient_accumulation_steps", None)  # HF: len(batch_samples) this step
        return int(n) if n else int(trainer.args.gradient_accumulation_steps)

    class DropAuditCallback(TrainerCallback):
        """Registered after TRL's callbacks, so it runs after this step's weight sync (version already bumped)
        and before ``_maybe_log_save_evaluate`` reads the metric sink."""

        def on_step_end(self, _args, state, _control, **_kwargs):
            if not audit.enabled:
                return
            try:
                n = n_trained()
            except Exception as e:  # noqa: BLE001
                audit._fail("on_step_end", e)
                return
            audit.flush(state.global_step, getattr(trainer, "model_version", None), trainer._metrics["train"],
                        n_trained=n)

        def on_train_end(self, _args, state, _control, **_kwargs):
            if audit.enabled:  # prefetched / partly built micro-batches after the last step; jsonl only
                audit.flush(state.global_step, getattr(trainer, "model_version", None), None, final=True)

    return DropAuditCallback()


def install(trainer, *, path=None, max_completion=None, num_generations=None, edges=EDGES):
    """Attach the audit to an (Async)GRPO trainer before ``trainer.train()``. Every process calls this; only the
    process that builds the rollout-queue dataset (rank 0) binds it, the others stay inert. Returns the audit,
    or None if it could not be installed (the trainer is then left exactly as it was)."""
    try:
        import trl.experimental.async_grpo.async_grpo_trainer as agt

        args = trainer.args
        if path is None:
            path = os.environ.get("RLFORGE_DROP_AUDIT_PATH") or os.path.join(args.output_dir, "drop_audit.jsonl")
        audit = DropAudit(num_generations=num_generations or args.num_generations,
                          max_completion=max_completion or args.max_completion_length, path=path, edges=edges)
        orig_get = trainer.get_train_dataloader
        trainer.add_callback(_make_callback(trainer, audit))
    except Exception as e:  # noqa: BLE001
        print(f"[rlforge][drop_audit] NOT installed: {type(e).__name__}: {e} (training unaffected)", flush=True)
        return None

    def get_train_dataloader(*a, **kw):
        if not trainer.accelerator.is_main_process:
            return orig_get(*a, **kw)
        try:
            classes = audited_classes(agt, audit)
            orig = agt.RolloutQueueDataset, agt.DataCollatorForRollout
        except Exception as e:  # noqa: BLE001
            audit._fail("patch", e)
            return orig_get(*a, **kw)
        agt.RolloutQueueDataset, agt.DataCollatorForRollout = classes
        try:
            dl = orig_get(*a, **kw)
        finally:
            agt.RolloutQueueDataset, agt.DataCollatorForRollout = orig
        try:  # TRL has resolved token_budget (None -> vLLM max_model_len) by now
            audit.configure(token_budget=getattr(trainer.args, "token_budget", None),
                            gas=getattr(trainer.args, "gradient_accumulation_steps", None))
        except Exception as e:  # noqa: BLE001
            audit._fail("configure", e)
        return dl

    trainer.get_train_dataloader = get_train_dataloader
    trainer._drop_audit = audit
    return audit
