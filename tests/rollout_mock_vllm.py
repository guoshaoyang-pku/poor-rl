"""Minimal stand-in for `vllm serve` (dev mode + --enable-lora) for CPU tests of the router.

Emulates what the router and TRL touch: /health, /v1/models, /server_info, /get_world_size,
/v1/completions (token-id or text prompt, n=1, return_token_ids, logprobs), /pause, /resume
(requests issued while paused wait, like mode=keep), /v1/load_lora_adapter (reads the dir from
disk; the adapter's identity is the sha of its tensors cast to bf16, as vLLM serves them),
/v1/unload_lora_adapter, /reset_prefix_cache and /metrics with the counters bench_sat reads.
Outputs are a deterministic function of (prompt, adapter identity, seed); prefix-cache hits are
modelled per (adapter NAME, prompt), the key vLLM uses.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os

from aiohttp import web


def adapter_identity(path: str) -> str:
    import torch
    from safetensors.torch import load_file

    sd = load_file(os.path.join(path, "adapter_model.safetensors"))
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes())
    return h.hexdigest()[:16]


class MockVLLM:
    def __init__(self, name="base", max_loras=4, max_lora_rank=16, delay_s=0.0, fail_load=False,
                 max_model_len=16384):
        self.name = name
        self.max_loras = max_loras
        self.max_lora_rank = max_lora_rank
        self.delay_s = delay_s
        self.fail_load = fail_load
        self.max_model_len = max_model_len
        self.loras: dict[str, dict] = {}
        self.paused = asyncio.Event()
        self.paused.set()  # set == running
        self.calls: list[tuple] = []
        self.ctr = {"gen": 0, "prompt": 0, "pq": 0, "ph": 0, "req": 0}
        self.cache: set = set()
        self.inflight = 0
        self.max_inflight = 0

    async def health(self, r):
        return web.Response(text="")

    async def models(self, r):
        data = [{"id": self.name, "object": "model", "max_model_len": self.max_model_len}]
        data += [{"id": n, "object": "model", "parent": self.name, "max_model_len": self.max_model_len}
                 for n in self.loras]
        return web.json_response({"object": "list", "data": data})

    async def server_info(self, r):
        return web.json_response({"vllm_config": {
            "model_config": {"dtype": "torch.bfloat16", "max_model_len": self.max_model_len},
            "parallel_config": {"data_parallel_size": 1, "tensor_parallel_size": 1},
            "lora_config": {"max_loras": self.max_loras, "max_lora_rank": self.max_lora_rank,
                            "max_cpu_loras": self.max_loras}}})

    async def world(self, r):
        return web.json_response({"world_size": 1})

    async def completions(self, r):
        body = await r.json()
        self.calls.append(("gen", body.get("model")))
        await self.paused.wait()
        model = body.get("model")
        if model != self.name and model not in self.loras:
            return web.json_response({"error": {"message": f"The model `{model}` does not exist."}}, status=404)
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            p = body["prompt"]
            pkey = json.dumps(p)
            ident = self.loras[model]["id"] if model in self.loras else "base"
            n = int(body.get("max_tokens", 8))
            seed = hashlib.sha256(f"{pkey}|{ident}".encode()).digest()
            toks = [seed[i % 32] + 1000 * (i // 32) for i in range(n)]
            lps = [-(seed[(i + 7) % 32] / 255.0) for i in range(n)]
            plen = len(p) if isinstance(p, list) else len(p.split())
            self.ctr["req"] += 1
            self.ctr["gen"] += n
            self.ctr["prompt"] += plen
            self.ctr["pq"] += plen
            ck = (model, pkey)  # vLLM: block hash extra key = lora_name
            if ck in self.cache:
                self.ctr["ph"] += plen
            self.cache.add(ck)
            return web.json_response({"id": "cmpl-x", "object": "text_completion", "model": model, "choices": [
                {"index": 0, "text": " ".join(map(str, toks)), "token_ids": toks,
                 "logprobs": {"token_logprobs": lps, "tokens": [str(t) for t in toks]}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": plen, "completion_tokens": n}})
        finally:
            self.inflight -= 1

    async def pause(self, r):
        self.calls.append(("pause", r.query.get("mode")))
        self.paused.clear()
        return web.json_response({"ok": True})

    async def resume(self, r):
        self.calls.append(("resume",))
        self.paused.set()
        return web.json_response({"ok": True})

    async def reset_prefix_cache(self, r):
        self.calls.append(("reset_prefix_cache",))
        self.cache.clear()
        return web.json_response({"ok": True})

    async def load(self, r):
        body = await r.json()
        self.calls.append(("load", body["lora_name"], body["lora_path"]))
        if self.fail_load:
            return web.Response(status=500, text="injected failure")
        name, path = body["lora_name"], body["lora_path"]
        if name in self.loras and not body.get("load_inplace"):
            return web.Response(status=400, text=f"The lora adapter '{name}' has already been loaded.")
        if not os.path.isfile(os.path.join(path, "adapter_config.json")):
            return web.Response(status=404, text=f"no adapter at {path}")
        self.loras[name] = {"path": path, "id": adapter_identity(path)}
        return web.Response(text=f"Success: LoRA adapter '{name}' added successfully.")

    async def unload(self, r):
        body = await r.json()
        self.calls.append(("unload", body["lora_name"]))
        if self.loras.pop(body["lora_name"], None) is None:
            return web.Response(status=404, text="not found")
        return web.Response(text="Success")

    async def metrics(self, r):
        c = self.ctr
        lines = [
            "# HELP vllm:generation_tokens_total x", "# TYPE vllm:generation_tokens_total counter",
            f'vllm:generation_tokens_total{{engine="0",model_name="{self.name}"}} {float(c["gen"])}',
            f'vllm:prompt_tokens_total{{engine="0",model_name="{self.name}"}} {float(c["prompt"])}',
            f'vllm:prefix_cache_queries_total{{engine="0",model_name="{self.name}"}} {float(c["pq"])}',
            f'vllm:prefix_cache_hits_total{{engine="0",model_name="{self.name}"}} {float(c["ph"])}',
            f'vllm:num_requests_running{{engine="0",model_name="{self.name}"}} {float(self.inflight)}',
            f'vllm:kv_cache_usage_perc{{engine="0",model_name="{self.name}"}} 0.25',
            f'vllm:request_success_total{{engine="0",finished_reason="length",model_name="{self.name}"}} {float(c["req"])}',
        ]
        return web.Response(text="\n".join(lines) + "\n")

    def app(self):
        app = web.Application(client_max_size=1 << 30)
        app.router.add_get("/health", self.health)
        app.router.add_get("/v1/models", self.models)
        app.router.add_get("/server_info", self.server_info)
        app.router.add_get("/get_world_size", self.world)
        app.router.add_post("/v1/completions", self.completions)
        app.router.add_post("/pause", self.pause)
        app.router.add_post("/resume", self.resume)
        app.router.add_post("/reset_prefix_cache", self.reset_prefix_cache)
        app.router.add_post("/v1/load_lora_adapter", self.load)
        app.router.add_post("/v1/unload_lora_adapter", self.unload)
        app.router.add_get("/metrics", self.metrics)
        return app
