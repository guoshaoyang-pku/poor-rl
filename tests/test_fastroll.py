"""Offline checks for rlforge fastroll (CPU only).

pytest runs T2 and T3; T1 also runs when FASTROLL_TOKENIZER (tokenizer dir) and FASTROLL_TRAIN (train jsonl) are set.
Script: python tests/test_fastroll.py <tokenizer dir> <train jsonl>
"""
import asyncio
import json
import os
import random
import sys
import time
from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer

import trl.experimental.async_grpo.async_rollout_worker as arw
from rlforge import fast_rollout as FR
from rlforge import score_loop as SL

# ---- T1: chat-template cache returns identical ids -----------------------------------------
def t1(tok_dir, data):
    tok = AutoTokenizer.from_pretrained(tok_dir)
    rows = [json.loads(line) for _, line in zip(range(300), open(data))]
    kw = dict(return_dict=False, add_generation_prompt=True, tools=None, chat_template=None, enable_thinking=True)
    t0 = time.perf_counter()
    ref = [tok.apply_chat_template(list(r["prompt"]), **kw) for r in rows]
    t_ref = (time.perf_counter() - t0) / len(rows)
    stats = FR.cache_chat_template(tok)
    t0 = time.perf_counter()
    for _ in range(3):
        for r, want in zip(rows, ref):
            got = tok.apply_chat_template(list(r["prompt"]), **kw)
            assert got == want and type(got) is list and got is not want
            got.append(-1)  # caller mutation must not leak into the cache
    t_cached = (time.perf_counter() - t0) / (3 * len(rows))
    assert stats["misses"] == len({json.dumps(r["prompt"]) for r in rows}) and stats["bypass"] == 0, stats
    # tools / return_dict variants bypass the cache
    tok.apply_chat_template(list(rows[0]["prompt"]), return_dict=True, add_generation_prompt=True)
    assert stats["bypass"] == 1
    print(f"T1 ok: {len(rows)} prompts, mean prompt {sum(map(len, ref)) / len(ref):.0f} tok, "
          f"uncached {t_ref * 1e3:.2f} ms/call, cached {t_cached * 1e3:.3f} ms/call, stats {stats}")


# ---- T2: O(done) FIRST_COMPLETED wait reports every task exactly once ----------------------
async def drive(aio, total, inflight_max):
    inflight, made, seen = {}, 0, []
    while made < total or inflight:
        while made < total and len(inflight) < inflight_max:
            inflight[aio.create_task(asyncio.sleep(random.random() * 0.05, result=made))] = made
            made += 1
        done, _ = await aio.wait(inflight, return_when=asyncio.FIRST_COMPLETED, timeout=0.1)
        for t in done:
            assert inflight.pop(t) == t.result()
            seen.append(t.result())
    return seen


def t2():
    out = {}
    for name, aio in (("stock", asyncio), ("fast", FR._FastWaitAsyncio())):
        random.seed(0)
        c0, w0 = time.process_time(), time.perf_counter()
        seen = asyncio.run(drive(aio, 40000, 8192))
        out[name] = (time.process_time() - c0, time.perf_counter() - w0)
        assert sorted(seen) == list(range(40000)), name
    print(f"T2 ok: 40000 tasks, inflight 8192; cpu/wall s stock {out['stock'][0]:.2f}/{out['stock'][1]:.2f}, "
          f"fast {out['fast'][0]:.2f}/{out['fast'][1]:.2f}")
    # module rebinding
    proxy = FR.install_fast_wait(arw)
    assert arw.asyncio is proxy and arw.asyncio.Queue is asyncio.Queue and FR.install_fast_wait(arw) is proxy
    arw.asyncio = asyncio


# ---- T3: ahead gate (new pump vs old polling) ----------------------------------------------
G, B, INFLIGHT, STEPS = 32, 4096, 8192, 5


class Base:
    async def _generate_one(self, prompt, tool_dict, tools, group_id=0):
        await asyncio.sleep(prompt)
        return ("c", [1], None, 0, 0, None)


class New(SL.ConcurrentScoreMixin, Base):
    pass


class Old(Base):
    async def _generate_one(self, prompt, tool_dict, tools, group_id=0):  # verbatim pre-fastroll gate
        if self._ahead_batch_samples:
            while group_id not in self._ahead_groups:
                if self._ahead_origin is None:
                    self._ahead_origin = self.model_version
                limit = ((self.model_version - self._ahead_origin + 2) * self._ahead_batch_samples
                         + self._ahead_dropped.value)
                if (group_id == self._ahead_next_group
                        and self._ahead_admitted + self.num_generations <= limit):
                    self._ahead_admitted += self.num_generations
                    self._ahead_next_group += 1
                    self._ahead_groups[group_id] = self.model_version
                    break
                await asyncio.sleep(0.01)
        return await super()._generate_one(prompt, tool_dict, tools, group_id)


def make(cls):
    o = cls.__new__(cls)
    o.__dict__.update(_ahead_batch_samples=B, _ahead_dropped=SimpleNamespace(value=0), _ahead_admitted=0,
                      _ahead_origin=None, _ahead_groups={}, _ahead_next_group=0, _ahead_waiters={},
                      _ahead_pump=None, num_generations=G, model_version=0, _early_hooks=[])
    return o


async def sim(o):
    o._stop_event = asyncio.Event()
    rng = random.Random(1)
    done_groups, counts, violations = [], {}, []
    inflight, gid, k = {}, 0, 0

    async def monitor():
        while True:
            if o._ahead_admitted > (o.model_version + 2) * B:
                violations.append((o._ahead_admitted, o.model_version))
            await asyncio.sleep(0.001)

    async def trainer():
        stale = []
        for step in range(STEPS):
            while len(done_groups) < (step + 1) * (B // G):
                await asyncio.sleep(0.002)
            batch = done_groups[step * (B // G):(step + 1) * (B // G)]
            stale += [o.model_version - v for _, v in batch]
            await asyncio.sleep(0.4)  # "fwd_bwd"
            o.model_version += 1
        return stale

    async def generator():
        nonlocal gid, k
        while True:
            while len(inflight) < INFLIGHT:
                t = asyncio.ensure_future(o._generate_one(rng.uniform(0.05, 0.5) if rng.random() > 0.02 else 1.5,
                                                          None, None, gid))
                inflight[t] = gid
                k += 1
                if k % G == 0:
                    gid += 1
            done, _ = await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED, timeout=0.1)
            for t in done:
                g = inflight.pop(t)
                t.result()
                counts[g] = counts.get(g, 0) + 1
                if counts[g] == G:
                    done_groups.append((g, o._ahead_groups.pop(g)))

    mon, gen = asyncio.ensure_future(monitor()), asyncio.ensure_future(generator())
    c0, w0 = time.process_time(), time.perf_counter()
    stale = await asyncio.wait_for(trainer(), 120)
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    o._stop_event.set()
    for t in [mon, gen, *inflight]:
        t.cancel()
    await asyncio.gather(mon, gen, *inflight, return_exceptions=True)
    order = [g for g, _ in done_groups]
    return dict(cpu=round(cpu, 2), wall=round(wall, 2), violations=len(violations),
                max_stale=max(stale), mean_stale=round(sum(stale) / len(stale), 3), groups=len(order))


def t3():
    res = {name: asyncio.run(sim(make(cls))) for name, cls in (("old", Old), ("new", New))}
    # out-of-order group completion (long tails) lets a group land one batch late in both gates;
    # the trainer's max_staleness (3) bounds what is trained
    for name, r in res.items():
        assert r["violations"] == 0 and r["max_stale"] <= 3, (name, r)
    assert res["new"]["max_stale"] <= res["old"]["max_stale"] + 1, res
    print(f"T3 ok: {STEPS} steps x {B} samples, inflight {INFLIGHT}: old {res['old']}, new {res['new']}")


_TOK, _TRAIN = os.environ.get("FASTROLL_TOKENIZER"), os.environ.get("FASTROLL_TRAIN")


@pytest.mark.skipif(not (_TOK and _TRAIN), reason="needs FASTROLL_TOKENIZER and FASTROLL_TRAIN")
def test_chat_template_cache():
    t1(_TOK, _TRAIN)


def test_fast_wait():
    t2()


def test_ahead_pump():
    t3()


if __name__ == "__main__":
    t1(sys.argv[1], sys.argv[2])
    t2()
    t3()
    print("ALL OK")
