"""Non-blocking (concurrent) group scoring for TRL's experimental async GRPO rollout worker.

Why: TRL 1.14 `_AsyncRolloutLoop._score_loop` awaits ONE group at a time. A group whose reward
calls an LLM judge (aiq_think_reward with AIQ_HALLUC=1: up to AIQ_HALLUC_TIMEOUT_S=90 s) stalls
the scoring of every group queued behind it; once `_groups_to_score` (maxsize 16) is full,
`_generate_loop` spins on `put_nowait` and stops harvesting completions and dispatching new
requests (metric `rollout/score_block_s`). Measured on 360-2 (2026-10-03): score_wait_s 35-61 s
mean, score_block_s 19-23 s mean, step_s 55-150 s mean with fwd_bwd only 7-10 s.

What this module does (all opt-in from rlforge.trainer, default = stock TRL behaviour):
  * ConcurrentScoreMixin (+ the concrete ConcurrentScoreLoop / ConcurrentScoreRoutedLoop):
    `_score_loop` scores up to `score_concurrency` groups at once (asyncio tasks + semaphore).
    Every metric the stock loop pushes is kept with the same meaning (score_s / score_wait_s per
    group, score_queue_size, inflight, windowed generated_tok_s, backpressure_s,
    `_total_groups_scored`, print samples). With score_concurrency=1 it is behaviour- and
    metric-identical to stock (test b). Sync reward funcs run on a dedicated pool of DAEMON
    threads (the loop's default executor), so an abandoned slow reward call can never hold up the
    child's exit. Shutdown cancels in-flight scoring tasks instead of waiting for them.
  * Composition with other loop classes: the mixin is combined with whatever loop class is
    installed on `AsyncRolloutWorker._loop_cls` when the worker is built/started (stock
    `_AsyncRolloutLoop`, or rlforge_v3's dp_route `GroupAffineRolloutLoop`), so it never shadows
    dp-route. Unknown loop classes are refused (RuntimeError), never silently replaced.
  * Liveness: each group's scoring is bounded by `score_task_max_s` (RLFORGE_SCORE_TASK_MAX_S,
    default max(600, 3 x AIQ_HALLUC_TIMEOUT_S)); a group that exceeds it fails the loop through
    TRL's failed_event / check_health path, like any reward exception. The heartbeat is only
    refreshed while no group has been scoring for longer than that bound, so even a scoring task
    that ignores cancellation lets the heartbeat go stale (stock: check_health raises after
    heartbeat_stale_after_s=300 s). A reward that raises CancelledError fails the loop (stock
    did too) instead of silently losing the group.
  * Judged tagging: when enabled, the loop hands the reward func a per-group `rlforge_info` dict
    (via the reward kwargs; only if the reward accepts **kwargs). The reward sets
    info["judged"] = True for groups it sent to the judge (decided before any verdict arrives)
    and every sample of that group carries metrics["rlforge/judged"] = 1.0 (0.0 otherwise) and
    metrics["rlforge/judge_versions"] = number of weight syncs that happened while the group was
    being scored (judged groups only).
  * Judged staleness compensation (trainer side): JudgedStalenessRolloutQueueDataset lets a
    judged sample be `min(judge_versions, judged_max_staleness - max_staleness)` versions staler
    than `max_staleness`, i.e. exactly the versions its own judge wait cost it, so judged groups
    are kept at the same rate as their unjudged siblings (no over-weighting, no starvation). The
    drop test itself is TRL's; only its threshold is switched per sample. Metrics:
    sample/judged_keep_frac vs sample/unjudged_keep_frac (should match), judged_staleness_* (kept
    samples only, like TRL's sample/staleness_*), judged_dropped_stale_total,
    judged_kept_by_allowance_total, judged_allowance_capped_total (cap too small).
  * Early reward hooks (EXPERIMENTAL, not recommended): if the reward function object (behind
    functools.partial) defines `rlforge_on_rollout(group_id=, prompt=, completion=,
    completion_ids=, **partial_kwargs)`, it is called on the event loop right after each single
    rollout finishes generating (must be non-blocking). The handles are passed back to the reward
    call as kwargs["rlforge_early"]. Under judge saturation this favours the rollouts that finish
    first (short CoTs) -- see NONBLOCKING_SCORING_2026-10-03.md, Review fixes.

Install: `rlforge.score_loop.install(score_concurrency=N, judged_max_staleness="auto"|S|None,
early_hooks=B, max_staleness=..., score_task_max_s=...)` BEFORE constructing the trainer. It
checks the installed TRL against the vetted 1.14.0 sources, then rebinds `AsyncRolloutWorker` /
`RolloutQueueDataset` in `trl.experimental.async_grpo.async_grpo_trainer` to the subclasses below
(site-packages is not edited). The rollout child is spawned: it unpickles the loop class by
qualified name, so this module must be importable in the child (it is: rlforge is on sys.path).
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import os
import queue
import sys
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

import trl
from trl.experimental.async_grpo import async_grpo_trainer as _agt
from trl.experimental.async_grpo import async_rollout_worker as _arw
from trl.trainer.utils import print_prompt_completions_sample

logger = _arw.logger

JUDGED_KEY = "rlforge/judged"
JUDGE_VERSIONS_KEY = "rlforge/judge_versions"

# --------------------------------------------------------------------------- TRL version guard
# The loop below copies TRL 1.14.0's `_score_loop` body and relies on private names of
# `_AsyncRolloutLoop` / `AsyncRolloutWorker` / `RolloutQueueDataset`. sha256[:16] of
# inspect.getsource() of every TRL member this module overrides, calls or depends on, as vetted
# on 360-1/360-2 (TRL 1.14.0, async_rollout_worker.py md5 a54c2e3d, async_grpo_trainer.py md5 513d5376).
VETTED_TRL_VERSION = "1.14.0"
VETTED_SOURCES = {
    "_AsyncRolloutLoop.__init__": "e426875a42d00877",
    "_AsyncRolloutLoop.run": "5010a1b9cf1080c7",
    "_AsyncRolloutLoop._generate_loop": "699c7c7bfbe2b7b0",
    "_AsyncRolloutLoop._score_loop": "b7581a38fb0e44b4",
    "_AsyncRolloutLoop._score_group": "808c50807e4c5276",
    "_AsyncRolloutLoop._generate_one": "26212cc454895e80",
    "_AsyncRolloutLoop._push_metrics": "0d51e6eb93c8adab",
    "AsyncRolloutWorker.__init__": "cb5869b9317921a1",
    "AsyncRolloutWorker.start": "32344ffaef4ab484",
    "RolloutQueueDataset.__iter__": "d92a084c61a3511c",
}


def _trl_members():
    return {
        "_AsyncRolloutLoop.__init__": _arw._AsyncRolloutLoop.__init__,
        "_AsyncRolloutLoop.run": _arw._AsyncRolloutLoop.run,
        "_AsyncRolloutLoop._generate_loop": _arw._AsyncRolloutLoop._generate_loop,
        "_AsyncRolloutLoop._score_loop": _arw._AsyncRolloutLoop._score_loop,
        "_AsyncRolloutLoop._score_group": _arw._AsyncRolloutLoop._score_group,
        "_AsyncRolloutLoop._generate_one": _arw._AsyncRolloutLoop._generate_one,
        "_AsyncRolloutLoop._push_metrics": _arw._AsyncRolloutLoop._push_metrics,
        "AsyncRolloutWorker.__init__": _arw.AsyncRolloutWorker.__init__,
        "AsyncRolloutWorker.start": _arw.AsyncRolloutWorker.start,
        "RolloutQueueDataset.__iter__": _ORIG["dataset"].__iter__,
    }


def trl_source_hashes() -> dict:
    out = {}
    for name, obj in _trl_members().items():
        try:
            out[name] = hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()[:16]
        except (OSError, TypeError):
            out[name] = None
    return out


def check_trl() -> list[str]:
    """-> list of mismatches against the vetted TRL (empty = OK). install() raises on a mismatch
    unless RLFORGE_SCORE_LOOP_UNVETTED=1 (then it only warns)."""
    bad = []
    if trl.__version__ != VETTED_TRL_VERSION:
        bad.append(f"trl.__version__={trl.__version__} (vetted {VETTED_TRL_VERSION})")
    for name, h in trl_source_hashes().items():
        if h != VETTED_SOURCES[name]:
            bad.append(f"{name}: source sha {h} != vetted {VETTED_SOURCES[name]}")
    return bad


# --------------------------------------------------------------------------- executor
class DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor whose workers are daemon threads and are NOT registered for the
    interpreter-exit join (`concurrent.futures.thread._python_exit`). asyncio requires the
    default executor to be a ThreadPoolExecutor instance, hence the subclass. Uses CPython
    3.10 private names; on any other layout it falls back to stock (non-daemon) threads."""

    _COMPATIBLE = sys.version_info[:2] in ((3, 10), (3, 11), (3, 12)) and hasattr(
        __import__("concurrent.futures.thread", fromlist=["_worker"]), "_worker")

    def _adjust_thread_count(self):
        if not self._COMPATIBLE:
            return super()._adjust_thread_count()
        from concurrent.futures import thread as _cft
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            t = threading.Thread(
                name=f"{self._thread_name_prefix or self}_{num_threads}",
                target=_cft._worker,
                args=(weakref.ref(self, weakref_cb), self._work_queue, self._initializer, self._initargs),
                daemon=True,
            )
            t.start()
            self._threads.add(t)  # deliberately not added to _cft._threads_queues


def _unwrap_partial(fn):
    """-> (innermost callable, merged partial keywords)."""
    kw = {}
    while isinstance(fn, functools.partial):
        kw = {**fn.keywords, **kw}
        fn = fn.func
    return fn, kw


def _accepts_kwargs(fn) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def default_score_task_max_s() -> float:
    env = os.environ.get("RLFORGE_SCORE_TASK_MAX_S", "").strip()
    if env:
        return float(env)
    try:
        judge_timeout = float(os.environ.get("AIQ_HALLUC_TIMEOUT_S", "90"))
    except ValueError:
        judge_timeout = 90.0
    return max(600.0, 3.0 * judge_timeout)


# --------------------------------------------------------------------------- rollout loop
class ConcurrentScoreMixin:
    """Concurrent score loop; combine with an `_AsyncRolloutLoop` subclass (see module docstring).
    Overrides __init__, _generate_one, _score_group, _score_loop and run only, so it composes with
    loops that override other methods (dp_route overrides _generate_one_turn)."""

    def __init__(self, *, score_concurrency: int = 1, tag_judged: bool = False,
                 early_hooks: bool = False, score_task_max_s: float | None = None,
                 saturation_warn_s: float = 30.0, **kwargs):
        super().__init__(**kwargs)
        self.score_concurrency = max(1, int(score_concurrency))
        self.score_task_max_s = float(score_task_max_s) if score_task_max_s else default_score_task_max_s()
        self.saturation_warn_s = float(saturation_warn_s)
        n_reward = max(1, len(self.reward_funcs))
        # Default executor of THIS loop = where asyncio.to_thread runs sync reward funcs. Stock
        # asyncio sizes it min(32, cpu+4) (= 32 on the 192-core 360 nodes); size it for N groups x
        # all reward funcs so concurrency is never silently capped below N.
        self._score_pool = DaemonThreadPoolExecutor(
            max_workers=max(32, self.score_concurrency * n_reward + 4), thread_name_prefix="rlforge-score")
        self._loop.set_default_executor(self._score_pool)

        inner = [_unwrap_partial(f) for f in self.reward_funcs]
        kw_ok = all(_accepts_kwargs(f) for f, _ in inner)
        self._pass_info = bool(tag_judged or early_hooks) and kw_ok
        if (tag_judged or early_hooks) and not kw_ok:
            logger.warning("[score] reward funcs do not accept **kwargs: judged tagging / early hooks disabled")
        self._tag_judged = bool(tag_judged) and self._pass_info
        self._early_hooks = []
        if early_hooks and self._pass_info:
            for f, kw in inner:
                hook = getattr(f, "rlforge_on_rollout", None)
                if callable(hook):
                    self._early_hooks.append((hook, kw))
            if not self._early_hooks:
                logger.warning("[score] --reward-early-hooks: reward defines no rlforge_on_rollout; ignored")
        # group_id -> list of (completion object, [handle per hook]); consumed in _score_group
        self._early: dict[int, list] = {}
        self._score_tasks: set[asyncio.Task] = set()
        self._score_started: dict[asyncio.Task, float] = {}  # task -> monotonic start of its scoring
        self._cancelling = False
        self._t_idle_start = time.monotonic()
        mro = " > ".join(c.__name__ for c in type(self).__mro__[:-1])
        # print: the child's stdout lands in the trainer log, so a smoke can grep this line
        print(f"[rlforge][score] rollout loop {type(self).__name__} (mro {mro}): "
              f"concurrency={self.score_concurrency} tag_judged={self._tag_judged} "
              f"early_hooks={len(self._early_hooks)} score_task_max_s={self.score_task_max_s:.0f}", flush=True)

    # ---- early per-rollout hook -------------------------------------------------------
    async def _generate_one(self, prompt, tool_dict, tools, group_id=0):
        result = await super()._generate_one(prompt, tool_dict, tools, group_id)
        if self._early_hooks:
            completion, completion_ids = result[0], result[1]
            handles = []
            for hook, kw in self._early_hooks:
                try:
                    handles.append(hook(group_id=group_id, prompt=prompt, completion=completion,
                                        completion_ids=completion_ids, **kw))
                except Exception as e:  # noqa: BLE001 - a hook must never break generation
                    logger.warning(f"[score] early hook failed: {type(e).__name__}: {e}")
                    handles.append(None)
            if any(h is not None for h in handles):
                self._early.setdefault(group_id, []).append((completion, handles))
        return result

    # ---- per-group scoring ------------------------------------------------------------
    async def _score_group(self, group):
        if not self._pass_info:
            return await super()._score_group(group)
        n = len(group.completions)
        info = {"group_id": int(group.group_id), "judged": False}
        kw = dict(group.reward_kwargs)
        kw["rlforge_info"] = [info] * n
        if self._early_hooks:
            entries = self._early.pop(group.group_id, [])
            aligned = []
            for c in group.completions:
                hit = next((h for (obj, h) in entries if obj is c), None)
                aligned.append(hit[0] if hit and len(hit) == 1 else hit)
            kw["rlforge_early"] = aligned
        group.reward_kwargs = kw
        v0 = self.model_version
        samples = await super()._score_group(group)
        if self._tag_judged:
            flag = 1.0 if info.get("judged") else 0.0
            versions = float(max(0, self.model_version - v0))
            for s in samples:
                s.metrics[JUDGED_KEY] = flag
                if flag:
                    s.metrics[JUDGE_VERSIONS_KEY] = versions
            if flag:
                self._counters["rollout/judged_groups_total"] += 1
        return samples

    # ---- score loop -------------------------------------------------------------------
    async def _score_loop(self, stop_event: asyncio.Event) -> None:
        sem = asyncio.Semaphore(self.score_concurrency)
        push_lock = asyncio.Lock()
        tasks = self._score_tasks
        started = self._score_started
        failure: list[BaseException] = []
        self._t_idle_start = time.monotonic()
        self._cancelling = False
        draining = False
        t_sat, sat_warned = None, False  # all N slots busy since t_sat (N > 1 only)
        t_prev = time.monotonic()

        def _done(t: asyncio.Task):
            tasks.discard(t)
            started.pop(t, None)
            sem.release()
            if failure:
                return
            if t.cancelled():
                # only this loop's own shutdown may cancel a scoring task; anything else (a reward
                # raising CancelledError) would otherwise lose the group silently
                if not self._cancelling and not stop_event.is_set():
                    failure.append(RuntimeError("score task cancelled unexpectedly (reward raised CancelledError?)"))
                return
            if t.exception() is not None:
                failure.append(t.exception())

        try:
            while not stop_event.is_set():
                now = time.monotonic()
                oldest = min(started.values(), default=None)
                if oldest is None or now - oldest <= self.score_task_max_s:
                    self._heartbeat_value.value = time.time()
                if failure:
                    raise failure[0]
                if self.score_concurrency > 1:
                    if sem.locked():
                        # seconds with every slot busy, summed per logging window (rides on the next push)
                        self._counters["rollout/score_saturated_s_total"] += now - t_prev
                        if t_sat is None:
                            t_sat = now
                        elif not sat_warned and now - t_sat > self.saturation_warn_s:
                            sat_warned = True
                            logger.warning(
                                f"[score] all {self.score_concurrency} scoring slots busy for {now - t_sat:.0f}s "
                                "(slow reward/judge or rollout-buffer backpressure): generation will block on the "
                                "score queue as in stock TRL; raise --score-concurrency if this persists")
                    else:
                        t_sat, sat_warned = None, False
                t_prev = now
                if draining:
                    if not tasks:
                        return
                    await asyncio.wait(set(tasks), timeout=0.5)
                    continue
                if sem.locked() and tasks:  # all N slots busy: wait for one to free (or for stop)
                    await asyncio.wait(set(tasks), timeout=0.5, return_when=asyncio.FIRST_COMPLETED)
                    continue
                try:
                    group = await asyncio.wait_for(self._groups_to_score.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if group is None:  # generator finished: drain what is in flight, then return
                    draining = True
                    continue
                await sem.acquire()  # never blocks: single consumer and not locked above
                if not tasks:
                    score_idle = time.monotonic() - self._t_idle_start
                    if score_idle > 0.5:
                        logger.info(f"[score] waited {score_idle:.1f}s for a group to score")
                wait_scoring = time.monotonic() - group.queued_at
                t = asyncio.create_task(self._score_and_push(group, wait_scoring, push_lock, stop_event))
                tasks.add(t)
                started[t] = time.monotonic()
                t.add_done_callback(_done)
        finally:
            self._cancelling = True
            for t in list(tasks):
                t.cancel()
            if tasks:
                await asyncio.gather(*list(tasks), return_exceptions=True)

    async def _score_and_push(self, group, wait_scoring: float, push_lock: asyncio.Lock,
                              stop_event: asyncio.Event) -> None:
        me = asyncio.current_task()
        t0 = time.monotonic()
        try:
            samples = await asyncio.wait_for(self._score_group(group), timeout=self.score_task_max_s)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"[score] group {group.group_id} not scored within {self.score_task_max_s:.0f}s "
                "(RLFORGE_SCORE_TASK_MAX_S / --score-task-max-s): reward or judge hung") from None
        except asyncio.CancelledError:
            if self._cancelling or stop_event.is_set():
                raise
            raise RuntimeError(f"[score] scoring of group {group.group_id} raised CancelledError "
                               "(not a shutdown): failing loudly instead of dropping the group") from None
        finally:
            self._score_started.pop(me, None)  # scoring done: push/backpressure time is not bounded
        scoring_time = time.monotonic() - t0
        async with push_lock:  # one group's metrics + samples go out together, in one block
            logger.info(
                f"[score] scored {len(samples)} samples in {scoring_time:.2f}s, "
                f"buffer_qsize={self.rollout_buffer.qsize()}"
            )
            now = time.monotonic()
            payload = {
                "rollout/score_s": scoring_time,
                "rollout/score_wait_s": wait_scoring,
                "rollout/score_queue_size": float(self._groups_to_score.qsize()),
                "rollout/inflight": float(self._inflight),
                "rollout/generated_tok_s": (
                    float(self._total_completion_tokens - self._pushed_completion_tokens),
                    now - self._pushed_at,
                ),
            }
            if self.score_concurrency > 1:
                # groups still being scored besides this one (0 = scorer had spare capacity)
                payload["rollout/score_inflight"] = float(max(0, len(self._score_tasks) - 1))
            self._push_metrics(payload)
            self._pushed_completion_tokens = self._total_completion_tokens
            self._pushed_at = now

            if self.log_completions and samples:
                print_prompt_completions_sample(
                    prompts=[s.prompt for s in samples],
                    completions=[s.completion for s in samples],
                    rewards={"reward": [s.metrics["reward"] for s in samples]},
                    advantages=[s.advantage for s in samples],
                    step=self._total_groups_scored,
                    num_samples=self.num_completions_to_print,
                )
            self._total_groups_scored += 1

            for sample in samples:
                t_blocked = None
                while True:
                    try:
                        sample.enqueued_at = time.time()
                        self.rollout_buffer.put_nowait(sample)
                        break
                    except queue.Full:
                        if stop_event.is_set():
                            return
                        if t_blocked is None:
                            t_blocked = time.monotonic()
                            logger.info(
                                f"[score] rollout buffer full (maxsize={self.queue_maxsize}), "
                                "waiting for trainer to consume..."
                            )
                        await asyncio.sleep(0.1)
                if t_blocked is not None:
                    self._push_metrics({"rollout/backpressure_s": time.monotonic() - t_blocked})
            self._t_idle_start = time.monotonic()

    def run(self) -> None:
        try:
            super().run()
        finally:
            self._score_pool.shutdown(wait=False, cancel_futures=True)


class ConcurrentScoreLoop(ConcurrentScoreMixin, _arw._AsyncRolloutLoop):
    """Concurrent score loop on top of stock TRL's `_AsyncRolloutLoop`."""


# base loop class (as installed on AsyncRolloutWorker._loop_cls) -> module-level combined class
# (module-level so the spawned child can unpickle it by qualified name)
_COMBINED: dict[type, type] = {_arw._AsyncRolloutLoop: ConcurrentScoreLoop}

try:  # rlforge_v3 only: group-affine routing for a DP vLLM server
    from rlforge import dp_route as _dp_route
except ImportError:  # older rlforge tree without dp_route
    _dp_route = None
else:
    class ConcurrentScoreRoutedLoop(ConcurrentScoreMixin, _dp_route.GroupAffineRolloutLoop):
        """Concurrent score loop + dp_route's group-affine request routing."""

    _COMBINED[_dp_route.GroupAffineRolloutLoop] = ConcurrentScoreRoutedLoop


def resolve_loop_cls(base: type) -> type:
    """Combine the concurrent score loop with the currently installed rollout loop class."""
    if isinstance(base, type) and issubclass(base, ConcurrentScoreMixin):
        return base
    combined = _COMBINED.get(base)
    if combined is None:
        raise RuntimeError(
            f"rlforge.score_loop: AsyncRolloutWorker._loop_cls is {base!r}, which has no concurrent-score "
            f"combination (known: {[c.__qualname__ for c in _COMBINED]}). Add a module-level "
            "class ConcurrentScoreX(ConcurrentScoreMixin, X) to rlforge/score_loop.py instead of dropping X.")
    return combined


class ConcurrentScoreWorker(_arw.AsyncRolloutWorker):
    """AsyncRolloutWorker that spawns the concurrent score loop, combined with whatever loop class
    is installed on the base AsyncRolloutWorker (e.g. by rlforge.dp_route.install(), in either
    install order), with `extra_loop_kwargs` added."""

    extra_loop_kwargs: dict = {}

    def __init__(self, **kwargs):
        super().__init__(**{**kwargs, **type(self).extra_loop_kwargs})
        self._loop_cls = resolve_loop_cls(_arw.AsyncRolloutWorker._loop_cls)

    def start(self) -> None:
        # re-resolve: a loop class installed after this worker was built still gets composed
        self._loop_cls = resolve_loop_cls(_arw.AsyncRolloutWorker._loop_cls)
        print(f"[rlforge][score] spawning rollout child with loop class {self._loop_cls.__qualname__} "
              f"(base installed: {_arw.AsyncRolloutWorker._loop_cls.__qualname__})", flush=True)
        return super().start()


# --------------------------------------------------------------------------- trainer side
class _JudgedStalenessQueue:
    """Proxy for the rollout mp.Queue: on every get() it switches the owning dataset's
    `max_staleness` to the threshold of the sample about to be checked. RolloutQueueDataset
    reads `self.max_staleness` right after get() in the same thread, and the trainer bumps
    the model version only between batches (same thread), so the switch is race-free.

    Judged samples get `base + min(judge_versions, cap - base)`: the weight syncs that happened
    while their own group was being scored (judge wait) are not counted against them, so they
    are kept at the rate of an unjudged sibling. A judged sample without judge_versions (older
    worker) gets the full cap."""

    def __init__(self, inner, dataset, base: int, cap: int):
        self._inner, self._ds, self._base, self._cap = inner, dataset, int(base), int(cap)

    def get(self, *args, **kwargs):
        sample = self._inner.get(*args, **kwargs)
        m = sample.metrics or {}
        is_judged = m.get(JUDGED_KEY, 0.0) > 0
        base, room = self._base, self._cap - self._base
        staleness = self._ds.model_version_fn() - sample.model_version
        mm = self._ds.metrics
        if is_judged:
            jv = m.get(JUDGE_VERSIONS_KEY)
            jv = room if jv is None else int(jv)
            extra = max(0, min(jv, room))
            if jv > room:
                mm["sample/judged_allowance_capped_total"].append(1.0)
            thr = base + extra
            kept = staleness <= thr
            mm["sample/judged_keep_frac"].append((float(kept), 1.0))
            mm["sample/judged_allowance_mean"].append(float(extra))
            if kept:
                mm["sample/judged_staleness_mean"].append(float(staleness))
                mm["sample/judged_staleness_max"].append(float(staleness))
                if staleness > base:
                    mm["sample/judged_kept_by_allowance_total"].append(1.0)
            else:
                mm["sample/judged_dropped_stale_total"].append(1.0)
        else:
            thr = base
            mm["sample/unjudged_keep_frac"].append((float(staleness <= thr), 1.0))
        self._ds.max_staleness = thr
        return sample

    def __getattr__(self, name):
        return getattr(self._inner, name)


class JudgedStalenessRolloutQueueDataset(_agt.RolloutQueueDataset):
    judged_max_staleness: int | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        js = type(self).judged_max_staleness
        if js is not None and int(js) > int(self.max_staleness):
            self.queue = _JudgedStalenessQueue(self.queue, self, self.max_staleness, js)


# --------------------------------------------------------------------------- install
_ORIG = {"worker": _agt.AsyncRolloutWorker, "dataset": _agt.RolloutQueueDataset}
JUDGED_AUTO_EXTRA = 2  # auto cap = max_staleness + 2 (covers a 90 s judge wait at >= 45 s/step)


def install(score_concurrency: int = 0, judged_max_staleness: int | str | None = "auto",
            early_hooks: bool = False, max_staleness: int | None = None,
            score_task_max_s: float | None = None) -> bool:
    """Patch TRL's async GRPO trainer module to use the classes above. Returns True if anything
    was installed. score_concurrency=0 and no other option = no-op (stock TRL).

    judged_max_staleness: "auto" (default) = max_staleness + 2 when score_concurrency > 1, else
    off; None = off; an int S = cap on the judged compensation, must be > max_staleness."""
    if judged_max_staleness == "auto":
        judged_max_staleness = (int(max_staleness) + JUDGED_AUTO_EXTRA
                                if score_concurrency > 1 and max_staleness is not None else None)
    if judged_max_staleness is not None:
        if max_staleness is None:
            raise ValueError("judged_max_staleness needs max_staleness")
        if int(judged_max_staleness) <= int(max_staleness):
            raise ValueError(f"judged_max_staleness={judged_max_staleness} must be > max_staleness={max_staleness} "
                             "(it is the cap on judged samples' extra staleness); use -1 to switch it off")
    tag = judged_max_staleness is not None
    if score_concurrency <= 0 and not tag and not early_hooks:
        return False
    bad = check_trl()
    if bad:
        msg = "rlforge.score_loop: installed TRL differs from the vetted sources: " + "; ".join(bad)
        if os.environ.get("RLFORGE_SCORE_LOOP_UNVETTED", "0") != "1":
            raise RuntimeError(msg + " (re-vet the copied _score_loop, or set RLFORGE_SCORE_LOOP_UNVETTED=1)")
        print(f"[rlforge] WARNING {msg} (RLFORGE_SCORE_LOOP_UNVETTED=1: continuing)", flush=True)
    if score_concurrency > 1 and not tag:
        print("[rlforge] WARNING --score-concurrency > 1 without the judged staleness compensation: judged groups "
              "reach the trainer later than their siblings and are dropped as stale more often, and no "
              "rlforge/judged metric is logged", flush=True)
    if early_hooks:
        print("[rlforge] WARNING --reward-early-hooks is experimental: under judge saturation it favours the "
              "rollouts that finish first (short CoTs); see NONBLOCKING_SCORING_2026-10-03.md", flush=True)
    ConcurrentScoreWorker.extra_loop_kwargs = {
        "score_concurrency": max(1, int(score_concurrency)),
        "tag_judged": tag,
        "early_hooks": bool(early_hooks),
        "score_task_max_s": float(score_task_max_s) if score_task_max_s else default_score_task_max_s(),
    }
    _agt.AsyncRolloutWorker = ConcurrentScoreWorker
    if tag:
        JudgedStalenessRolloutQueueDataset.judged_max_staleness = int(judged_max_staleness)
        _agt.RolloutQueueDataset = JudgedStalenessRolloutQueueDataset
    print(f"[rlforge] non-blocking scoring: {ConcurrentScoreWorker.extra_loop_kwargs} "
          f"max_staleness={max_staleness} judged_max_staleness={judged_max_staleness}", flush=True)
    return True


def uninstall() -> None:
    _agt.AsyncRolloutWorker = _ORIG["worker"]
    _agt.RolloutQueueDataset = _ORIG["dataset"]
    JudgedStalenessRolloutQueueDataset.judged_max_staleness = None
    ConcurrentScoreWorker.extra_loop_kwargs = {}
