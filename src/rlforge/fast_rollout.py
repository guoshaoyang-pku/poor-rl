"""CPU fixes for the single-threaded TRL rollout child (rlforge fastroll, 2026-10-17).

Why: on the H100 0.8B run (bsz 4096, max_inflight_tasks 8192) the rollout child's event loop ran
at ~97% of one core and both the vLLM node and the trainer sat idle >60% of every step. py-spy
over one 100 s step (/cephfs/guoshaoyang/tools/pyspy_rollout_worker_0712.txt):
  * 33%  TRL `_generate_one` re-tokenizes the whole ~3.6k-token prompt with apply_chat_template
         for every completion (32x per group, identical output);
  * ~30% asyncio timer churn from the ahead gate (every waiting task polls `sleep(0.01)`);
  * ~13% TRL `_generate_loop` calls `asyncio.wait(inflight_tasks, FIRST_COMPLETED)`, which adds
         and removes a done-callback on all ~8192 in-flight tasks on every call.
This module removes all three without changing what is generated or trained:
  * `cache_chat_template(tok)`: memoizes apply_chat_template for plain-text conversations without
    tools (same inputs -> the same token ids; a fresh list is returned on every call);
  * the ahead gate (score_loop.ConcurrentScoreMixin) admits groups from one pump coroutine
    instead of per-task polling (same admission order and limit);
  * `install_fast_wait(module)`: rebinds the `asyncio` global of TRL's rollout module to a proxy
    whose create_task registers a done-callback once and whose FIRST_COMPLETED wait is O(done).
Only the spawned rollout child calls these; site-packages is not edited.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict

_PRIMITIVES = (str, bool, int, float, type(None))


def _conversation_key(conversation, args, kwargs):
    if args or kwargs.get("tools") or kwargs.get("return_dict", True) is not False:
        return None
    if not isinstance(conversation, (list, tuple)):
        return None
    msgs = []
    for m in conversation:
        if not isinstance(m, dict) or not all(isinstance(v, str) for v in m.values()):
            return None
        msgs.append(tuple(sorted(m.items())))
    if not all(isinstance(v, _PRIMITIVES) for v in kwargs.values()):
        return None
    return tuple(msgs), tuple(sorted(kwargs.items()))


def cache_chat_template(tok, maxsize: int = 1024):
    """Memoize `tok.apply_chat_template` on this tokenizer instance (child process only)."""
    if getattr(tok, "_rlforge_chat_cache", None) is not None:
        return tok._rlforge_chat_cache
    orig = tok.apply_chat_template
    cache: OrderedDict = OrderedDict()
    stats = {"hits": 0, "misses": 0, "bypass": 0}

    def apply_chat_template(conversation, *args, **kwargs):
        key = _conversation_key(conversation, args, kwargs)
        if key is None:
            stats["bypass"] += 1
            return orig(conversation, *args, **kwargs)
        hit = cache.get(key)
        if hit is not None:
            cache.move_to_end(key)
            stats["hits"] += 1
            return list(hit)
        out = orig(conversation, *args, **kwargs)
        stats["misses"] += 1
        if isinstance(out, list) and all(type(x) is int for x in out):
            cache[key] = tuple(out)
            if len(cache) > maxsize:
                cache.popitem(last=False)
        return out

    tok.apply_chat_template = apply_chat_template
    tok._rlforge_chat_cache = stats
    return stats


class _FastWaitAsyncio:
    """Proxy for the `asyncio` module inside TRL's async_rollout_worker.

    create_task(): the real task plus one done-callback that records completion.
    wait(fs, return_when=FIRST_COMPLETED, timeout=...): returns (done tasks among fs, set()) from the
    recorded completions, awaiting a completion signal (or the timeout) when none is ready; on a
    timeout it also scans fs once, so a task not created here is still reported. Callers in TRL
    discard the pending set (`done, _ = await asyncio.wait(...)`). Every other attribute and every
    other wait mode is the real asyncio."""

    def __init__(self):
        self._finished: set = set()
        self._signal: asyncio.Event | None = None
        self.calls = 0

    def __getattr__(self, name):
        return getattr(asyncio, name)

    def _on_done(self, task):
        self._finished.add(task)
        if self._signal is not None:
            self._signal.set()

    def create_task(self, coro, **kwargs):
        task = asyncio.create_task(coro, **kwargs)
        task.add_done_callback(self._on_done)
        return task

    async def wait(self, fs, *, timeout=None, return_when=asyncio.ALL_COMPLETED):
        if return_when != asyncio.FIRST_COMPLETED or not isinstance(fs, dict) or not fs:
            return await asyncio.wait(fs, timeout=timeout, return_when=return_when)
        self.calls += 1
        if self._signal is None:
            self._signal = asyncio.Event()
        self._signal.clear()
        ready = {t for t in self._finished if t in fs}
        if not ready:
            try:
                await asyncio.wait_for(self._signal.wait(), timeout)
            except asyncio.TimeoutError:
                ready = {t for t in fs if t.done()}
            ready |= {t for t in self._finished if t in fs}
        self._finished -= ready
        # Drop records of tasks the caller no longer tracks (e.g. the loop tasks themselves).
        if len(self._finished) > 4 * len(fs) + 64:
            self._finished = {t for t in self._finished if t in fs}
        return ready, set()


def install_fast_wait(module) -> _FastWaitAsyncio:
    proxy = module.asyncio
    if not isinstance(proxy, _FastWaitAsyncio):
        proxy = _FastWaitAsyncio()
        module.asyncio = proxy
    return proxy
