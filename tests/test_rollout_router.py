"""CPU tests for the multi-backend rollout router + adapter-only LoRA sync (src/rlforge/rollout).

Everything runs in-process on 127.0.0.1: N mock vLLM servers (tests/rollout_mock_vllm.py), one
lora_agent per emulated remote host, the router, and TRL 1.14's real VLLMClient /
select_adapter_sync when TRL is importable. Run: pytest -q tests/test_rollout_router.py
"""
from __future__ import annotations

import asyncio
import collections
import json
import os
import socket
import sys
import tempfile

import aiohttp
import pytest
import torch
from aiohttp import web
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from rlforge.rollout.lora_agent import Agent  # noqa: E402
from rlforge.rollout.router import Affinity, Router, parse_backend, prompt_key  # noqa: E402
from rollout_mock_vllm import MockVLLM, adapter_identity  # noqa: E402


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


async def serve(app):
    runner = web.AppRunner(app)
    await runner.setup()
    port = free_port()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    return runner, f"http://127.0.0.1:{port}"


def write_adapter(d, seed, r=4, dtype=torch.float32, layers=2, zero_b=False):
    """PEFT-layout adapter (adapter_config.json + adapter_model.safetensors), TRL key naming."""
    os.makedirs(d, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for L in range(layers):
        for sub, t, fi, fo in [("self_attn", "q_proj", 64, 128), ("linear_attn", "in_proj_qkv", 64, 96),
                               ("linear_attn", "in_proj_a", 64, 4), ("mlp", "down_proj", 128, 64)]:
            k = f"base_model.model.model.layers.{L}.{sub}.{t}"
            sd[k + ".lora_A.weight"] = torch.randn(r, fi, generator=g).to(dtype)
            b = torch.zeros(fo, r) if zero_b else torch.randn(fo, r, generator=g) * 0.02
            sd[k + ".lora_B.weight"] = b.to(dtype)
    save_file(sd, os.path.join(d, "adapter_model.safetensors"))
    json.dump({"peft_type": "LORA", "r": r, "lora_alpha": 2 * r, "target_modules": ["q_proj"], "bias": "none"},
              open(os.path.join(d, "adapter_config.json"), "w"))
    open(os.path.join(d, "README.md"), "w").write("peft readme")
    return sd


class Cluster:
    """`spec`: list of host labels, one per backend; "local" shares the router FS, others get an agent."""

    def __init__(self, spec, weights=None, delay_s=0.0, fail=(), **router_kw):
        self.spec = spec
        self.weights = weights or [1.0] * len(spec)
        self.delay_s = delay_s
        self.fail = set(fail)
        self.router_kw = router_kw

    async def __aenter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.runners = []
        self.mocks, self.mock_urls = [], []
        for i, _ in enumerate(self.spec):
            m = MockVLLM(delay_s=self.delay_s, fail_load=i in self.fail)
            run, url = await serve(m.app())
            self.runners.append(run)
            self.mocks.append(m)
            self.mock_urls.append(url)
        self.agents, self.agent_urls = {}, {}
        for h in sorted(set(self.spec) - {"local"}):
            a = Agent(os.path.join(self.tmp.name, f"recv_{h}"))
            run, url = await serve(a.app())
            self.runners.append(run)
            self.agents[h], self.agent_urls[h] = a, url
        specs = []
        for i, h in enumerate(self.spec):
            s = f"{self.mock_urls[i]},weight={self.weights[i]}"
            if h != "local":
                s += f",agent={self.agent_urls[h]},host={h}"
            specs.append(s)
        self.backends = [parse_backend(s, i) for i, s in enumerate(specs)]
        kw = dict(health_s=0.2)
        kw.update(self.router_kw)
        self.router = Router(self.backends, os.path.join(self.tmp.name, "stage"), **kw)
        self.rrun, self.url = await serve(self.router.app())
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *a):
        await self.session.close()
        await self.rrun.cleanup()
        for r in self.runners:
            await r.cleanup()
        self.tmp.cleanup()

    async def complete(self, prompt, model="base", max_tokens=8):
        async with self.session.post(f"{self.url}/v1/completions", json={
                "model": model, "prompt": prompt, "max_tokens": max_tokens, "n": 1,
                "return_token_ids": True, "logprobs": 0}) as r:
            body = await r.json()
            return r.status, int(r.headers.get("X-Router-Backend", -1)), body


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------------------------ routing

def test_prompt_key_stable_and_distinct():
    assert prompt_key({"prompt": [1, 2, 3]}) == prompt_key({"prompt": [1, 2, 3], "model": "x"})
    assert prompt_key({"prompt": [1, 2, 3]}) != prompt_key({"prompt": [1, 2, 4]})
    assert prompt_key({"prompt": "ab"}) != prompt_key({"prompt": [97, 98]})
    assert prompt_key({"messages": [{"role": "user", "content": "hi"}]}) is not None
    assert prompt_key({}) is None


def test_group_affinity_and_balance():
    """64 groups x G=16 concurrent requests over 4 backends: each group on exactly one backend, balanced."""
    async def body():
        async with Cluster(["local"] * 4, delay_s=0.02) as c:
            groups = [[g * 1000 + i for i in range(50)] for g in range(64)]
            reqs = [c.complete(p) for p in groups for _ in range(16)]
            res = await asyncio.gather(*reqs)
            assert all(st == 200 for st, _, _ in res)
            per_group = collections.defaultdict(set)
            for k, (st, b, body) in enumerate(res):
                per_group[k // 16].add(b)
            assert all(len(v) == 1 for v in per_group.values()), per_group
            counts = collections.Counter(next(iter(v)) for v in per_group.values())
            assert set(counts) == {0, 1, 2, 3} and max(counts.values()) - min(counts.values()) <= 2, counts
            # outputs of the same group are identical (same prompt, deterministic mock), so routing is transparent
            st = c.router.aff
            assert st.new == 64 and st.hits == 64 * 15
            assert all(b.inflight == 0 for b in c.backends)
    run(body())


def test_weighted_balance():
    async def body():
        async with Cluster(["local"] * 3, weights=[1, 1, 2], delay_s=0.05) as c:
            res = await asyncio.gather(*[c.complete([g, 7, 7]) for g in range(400)])
            counts = collections.Counter(b for _, b, _ in res)
            assert counts[2] > 1.6 * counts[0] and counts[2] > 1.6 * counts[1], counts
    run(body())


def test_sequential_group_stays_pinned_via_lru():
    """Closed-loop clients (bench_sat) issue a group's requests one after another; the LRU keeps them together."""
    async def body():
        async with Cluster(["local"] * 3) as c:
            seen = collections.defaultdict(set)
            for g in range(10):
                for _ in range(8):
                    _, b, _ = await c.complete([g, 1, 2, 3])
                    seen[g].add(b)
            assert all(len(v) == 1 for v in seen.values())
            assert len({next(iter(v)) for v in seen.values()}) == 3  # still spread over backends
    run(body())


def test_lru_pin_moves_off_overloaded_backend():
    bs = [parse_backend(f"http://x{i}", i) for i in range(2)]
    for b in bs:
        b.healthy = True
    aff = Affinity(bs, slack_ratio=1.5, slack_abs=1.0)
    k = b"k" * 16
    i = aff.pick(k)
    aff.release(k, i)
    bs[i].inflight = 20       # someone else's load piles onto the pinned backend
    j = aff.pick(k)
    assert j != i and aff.moved == 1


def test_failover_and_readmit_with_adapters():
    async def body():
        async with Cluster(["local", "local"]) as c:
            d = os.path.join(c.tmp.name, "trainer", "trl-policy-v1")
            write_adapter(d, 1)
            async with c.session.post(f"{c.url}/v1/load_lora_adapter",
                                      json={"lora_name": "trl-policy-v1", "lora_path": d}) as r:
                assert r.status == 200, await r.text()
            # take backend 0 down
            await c.runners[0].cleanup()
            res = [await c.complete([g, 5], model="trl-policy-v1") for g in range(6)]
            assert all(st == 200 and b == 1 for st, b, _ in res), res
            await asyncio.sleep(0.6)
            assert not c.backends[0].healthy
            # bring a FRESH server (no adapters) back on the same port: router must re-load v1 before admitting it
            port = int(c.mock_urls[0].rsplit(":", 1)[1])
            m = MockVLLM()
            runner = web.AppRunner(m.app())
            await runner.setup()
            await web.TCPSite(runner, "127.0.0.1", port).start()
            c.runners[0] = runner
            for _ in range(30):
                await asyncio.sleep(0.2)
                if c.backends[0].healthy:
                    break
            assert c.backends[0].healthy and "trl-policy-v1" in m.loras
            res = await asyncio.gather(*[c.complete([g, 9], model="trl-policy-v1") for g in range(20)])
            assert all(st == 200 for st, _, _ in res) and {b for _, b, _ in res} == {0, 1}
    run(body())


# --------------------------------------------------------------------------------- fan-out

def test_pause_blocks_and_resume_releases_on_all_backends():
    async def body():
        async with Cluster(["local"] * 3) as c:
            async with c.session.post(f"{c.url}/pause", params={"mode": "keep"}) as r:
                assert r.status == 200
            assert all(("pause", "keep") in m.calls for m in c.mocks)
            pend = [asyncio.create_task(c.complete([g, 1])) for g in range(9)]
            await asyncio.sleep(0.3)
            assert not any(t.done() for t in pend)
            async with c.session.post(f"{c.url}/resume") as r:
                assert r.status == 200
            res = await asyncio.gather(*pend)
            assert all(st == 200 for st, _, _ in res)
    run(body())


def test_metrics_aggregate_parse_like_bench_sat():
    async def body():
        async with Cluster(["local"] * 2) as c:
            await asyncio.gather(*[c.complete([g, 1, 2], max_tokens=10) for g in range(10)])
            async with c.session.get(f"{c.url}/metrics") as r:
                text = await r.text()
            import re
            vals = [float(m.group(1)) for m in re.finditer(
                r"^vllm:generation_tokens_total(?:\{[^}]*\})? ([0-9.eE+-]+)$", text, re.M)]
            assert len(vals) == 2 and sum(vals) == 100
            assert 'backend="1"' in text and "router_affinity_new_total 10" in text
    run(body())


def test_merged_sync_endpoints_refused():
    async def body():
        async with Cluster(["local"]) as c:
            async with c.session.post(f"{c.url}/init_weight_transfer_engine", json={}) as r:
                assert r.status == 501
    run(body())


# ------------------------------------------------------------------------------- LoRA sync

def test_lora_sync_cross_host_bf16_cast_and_outputs():
    """2 local backends + 2 remote hosts (agent each, 1 and 2 backends): fp32 adapter -> bf16 shipped everywhere."""
    async def body():
        async with Cluster(["local", "local", "hostB", "hostC", "hostC"], push_streams=3) as c:
            d1 = os.path.join(c.tmp.name, "trainer", "trl-policy-v1")
            sd = write_adapter(d1, 1, layers=40)  # > 16 MiB so the multi-stream path runs
            async with c.session.post(f"{c.url}/v1/load_lora_adapter",
                                      json={"lora_name": "trl-policy-v1", "lora_path": d1}) as r:
                rec = await r.json()
                assert r.status == 200, rec
            assert rec["n_backends"] == 5 and len(rec["pushes"]) == 2
            assert rec["ship_bytes"] < 0.55 * rec["src_bytes"]
            paths = [m.loras["trl-policy-v1"]["path"] for m in c.mocks]
            assert paths[0] == paths[1] == os.path.join(c.router.stage_root, "trl-policy-v1")
            assert paths[2].startswith(c.agents["hostB"].root) and paths[3] == paths[4]
            assert paths[3].startswith(c.agents["hostC"].root)
            shipped = load_file(os.path.join(paths[2], "adapter_model.safetensors"))
            assert set(shipped) == set(sd)
            assert all(shipped[k].dtype == torch.bfloat16 and torch.equal(shipped[k], sd[k].to(torch.bfloat16))
                       for k in sd)
            assert not os.path.exists(os.path.join(paths[2], "README.md"))
            # identity as vLLM serves it (bf16) is the same for the fp32 source and every shipped copy
            ids = {m.loras["trl-policy-v1"]["id"] for m in c.mocks}
            assert ids == {adapter_identity(d1)}
            # same prompt + adapter -> same outputs on every backend; differs from the base model
            prompt = list(range(30))
            outs = set()
            for m, u in zip(c.mocks, c.mock_urls):
                async with c.session.post(f"{u}/v1/completions", json={"model": "trl-policy-v1", "prompt": prompt,
                                                                       "max_tokens": 6}) as r:
                    outs.add(tuple((await r.json())["choices"][0]["token_ids"]))
            assert len(outs) == 1
            _, _, base = await c.complete(prompt, model="base", max_tokens=6)
            assert tuple(base["choices"][0]["token_ids"]) not in outs
    run(body())


def test_lora_sync_all_or_nothing():
    async def body():
        async with Cluster(["local", "hostB", "hostB"], fail={2}) as c:
            d = os.path.join(c.tmp.name, "trainer", "trl-policy-v1")
            write_adapter(d, 1)
            async with c.session.post(f"{c.url}/v1/load_lora_adapter",
                                      json={"lora_name": "trl-policy-v1", "lora_path": d}) as r:
                rec = await r.json()
                assert r.status == 500 and "injected failure" in rec["error"]
            assert all("trl-policy-v1" not in m.loras for m in c.mocks)
            assert "trl-policy-v1" not in c.router.registry
            st, _, _ = await c.complete([1, 2], model="trl-policy-v1")
            assert st == 404
    run(body())


def test_unload_deletes_copies_one_sync_later():
    async def body():
        async with Cluster(["local", "hostB"]) as c:
            dirs = {}
            for v in (1, 2, 3):
                d = os.path.join(c.tmp.name, "trainer", f"trl-policy-v{v}")
                write_adapter(d, v)
                async with c.session.post(f"{c.url}/v1/load_lora_adapter",
                                          json={"lora_name": f"trl-policy-v{v}", "lora_path": d}) as r:
                    assert r.status == 200
                dirs[v] = (c.mocks[0].loras[f"trl-policy-v{v}"]["path"], c.mocks[1].loras[f"trl-policy-v{v}"]["path"])
            async with c.session.post(f"{c.url}/v1/unload_lora_adapter", json={"lora_name": "trl-policy-v1"}) as r:
                assert r.status == 200
            assert all(os.path.isdir(p) for p in dirs[1])          # still there: vLLM resolves paths lazily
            async with c.session.post(f"{c.url}/v1/unload_lora_adapter", json={"lora_name": "trl-policy-v2"}) as r:
                assert r.status == 200
            assert not any(os.path.exists(p) for p in dirs[1])
            assert all(os.path.isdir(p) for p in dirs[2] + dirs[3])
            assert all(set(m.loras) == {"trl-policy-v3"} for m in c.mocks)
            # unknown name: TRL ignores the status; router answers 200 (backends say 404)
            async with c.session.post(f"{c.url}/v1/unload_lora_adapter", json={"lora_name": "nope"}) as r:
                assert r.status == 200
    run(body())


def test_agent_rejects_corrupt_push():
    async def body():
        with tempfile.TemporaryDirectory() as t:
            a = Agent(os.path.join(t, "recv"))
            run_, url = await serve(a.app())
            async with aiohttp.ClientSession() as s:
                await s.post(f"{url}/adapters/x/begin")
                await s.put(f"{url}/adapters/x/f.bin", params={"offset": "0", "total": "4"}, data=b"abcd")
                async with s.post(f"{url}/adapters/x/commit",
                                  json={"files": [{"name": "f.bin", "size": 4, "sha256": "0" * 64}]}) as r:
                    assert r.status == 409
                async with s.put(f"{url}/adapters/..%2Fevil/f.bin", data=b"x") as r:
                    assert r.status in (400, 404)
            assert not os.path.exists(os.path.join(t, "recv", "x"))
            await run_.cleanup()
    run(body())


# ------------------------------------------------------------------- TRL 1.14 client contract

trl_client = pytest.importorskip("trl.experimental.async_grpo.vllm_client", reason="TRL not installed")


def test_trl_vllm_client_lifecycle_through_router():
    """TRL's own VLLMClient (sync, requests) drives the sequence _sync_weight_lora uses, via the router."""
    from trl.experimental.async_grpo.vllm_client import VLLMClient

    async def body():
        async with Cluster(["local", "hostB", "hostC"]) as c:
            cl = VLLMClient(c.url, server_timeout=10)
            await asyncio.to_thread(cl.wait_for_server_ready)
            assert await asyncio.to_thread(cl.get_max_model_len) == 16384
            info = await asyncio.to_thread(cl.get_server_info)
            assert info["parallel_config"]["data_parallel_size"] == 1 and info["lora_config"]["max_loras"] == 4
            assert await asyncio.to_thread(cl.get_dtype) == "torch.bfloat16"
            lora_dir = os.path.join(c.tmp.name, "out", ".vllm_lora")
            for v in range(1, 5):   # 4 syncs, max_staleness=1 -> unload v-2
                d = os.path.join(lora_dir, f"trl-policy-v{v}")
                write_adapter(d, v)
                await asyncio.to_thread(cl.pause)
                await asyncio.to_thread(cl.load_lora_adapter, f"trl-policy-v{v}", d)
                await asyncio.to_thread(cl.resume)
                if v - 2 > 0:
                    await asyncio.to_thread(cl.unload_lora_adapter, f"trl-policy-v{v - 2}")
                res = await asyncio.gather(*[c.complete([g, v], model=f"trl-policy-v{v}") for g in range(6)])
                assert all(st == 200 for st, _, _ in res)
            assert all(set(m.loras) == {"trl-policy-v3", "trl-policy-v4"} for m in c.mocks)
            assert len(c.router.sync_hist) == 4
    run(body())


def test_trl_select_adapter_sync_accepts_router_server_info():
    peft = pytest.importorskip("peft")
    from transformers import LlamaConfig, LlamaForCausalLM
    from trl.experimental.async_grpo.async_grpo_trainer import select_adapter_sync

    class Args:
        max_staleness = 2

    async def body():
        async with Cluster(["local", "hostB"]) as c:
            async with c.session.get(f"{c.url}/server_info", params={"config_format": "json"}) as r:
                return (await r.json())["vllm_config"]

    vc = run(body())
    cfg = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                      num_key_value_heads=1, vocab_size=128)
    model = peft.get_peft_model(LlamaForCausalLM(cfg), peft.LoraConfig(r=16, lora_alpha=32, target_modules="all-linear"))
    assert select_adapter_sync(vc, model, Args()) is True
