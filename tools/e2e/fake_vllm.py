#!/usr/bin/env python
"""CPU stand-in for one `vllm serve --enable-lora` backend, for the e2e CPU test.

Samples with the real (tiny) HF model in fp32 and returns vLLM-shaped /v1/completions responses (token_ids +
sampled-token logprobs). /v1/load_lora_adapter really applies the PEFT adapter: every key must map onto an
existing Linear under vLLM's `base_model.model.model.language_model.layers.*` naming, otherwise the load fails
(400), exactly the failure mode vLLM has silently (keys it does not recognise are ignored -> adapter == base).
So the e2e test's |log rho| ~ 0 at staleness 0 proves: adapter naming, the gather/save path, the router push and
the token/logprob alignment of the prefix-shared trainer.

Run: python fake_vllm.py --model TINY_DIR --port 8501 [--host 127.0.0.1]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import threading

for _m in ("fla", "causal_conv1d", "flash_attn"):
    sys.modules[_m] = None

import torch  # noqa: E402
from aiohttp import web  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

KEY_RE = re.compile(r"^base_model\.model\.model\.language_model\.(layers\.\d+\..+)\.lora_([AB])\.weight$")


class Fake:
    def __init__(self, model_dir, eos=None):
        torch.set_num_threads(int(os.environ.get("FAKE_THREADS", "4")))
        self.m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, device_map="cpu",
                                                      attn_implementation="sdpa").eval()
        self.base = {k: v.clone() for k, v in self.m.state_dict().items()}
        self.inner = self.m.model
        self.loras: dict[str, dict] = {}
        self.cur = None
        self.lock = threading.Lock()
        self.eos = eos
        self.stats = {"requests": 0, "by_model": {}, "loads": 0}

    # ---- adapters
    def build(self, path):
        cfg = json.load(open(os.path.join(path, "adapter_config.json")))
        sd = load_file(os.path.join(path, "adapter_model.safetensors"))
        scale = cfg["lora_alpha"] / cfg["r"]
        pairs = {}
        for k, v in sd.items():
            m = KEY_RE.match(k)
            if not m:
                raise ValueError(f"unrecognised adapter key {k}")
            pairs.setdefault(m.group(1), {})[m.group(2)] = v.float()
        deltas = {}
        mods = dict(self.inner.named_modules())
        for mod, ab in pairs.items():
            if mod not in mods or not isinstance(mods[mod], torch.nn.Linear):
                raise ValueError(f"adapter module {mod} not in model")
            deltas[f"model.{mod}.weight"] = scale * (ab["B"] @ ab["A"])
        return deltas

    def apply(self, name):
        if name == self.cur:
            return
        sd = {k: v.clone() for k, v in self.base.items()}
        if name != "base":
            for k, d in self.loras[name].items():
                sd[k] = sd[k] + d
        self.m.load_state_dict(sd)
        self.cur = name

    # ---- generation
    @torch.no_grad()
    def generate(self, name, ids, n, max_tokens, temperature, seed):
        with self.lock:
            self.apply(name)
            g = torch.Generator().manual_seed(int(seed or 0))
            out = []
            for i in range(n):
                seq = list(ids)
                toks, lps = [], []
                fin = "length"
                for _ in range(max_tokens):
                    lg = self.m(input_ids=torch.tensor([seq]), use_cache=False).logits[0, -1].float()
                    lp = torch.log_softmax(lg / temperature, -1)
                    t = int(torch.multinomial(lp.exp(), 1, generator=g))
                    toks.append(t)
                    lps.append(float(lp[t]))
                    seq.append(t)
                    if self.eos is not None and t == self.eos:
                        fin = "stop"
                        break
                txt = "<answer>A</answer>" if i % 2 == 0 else "no tag"
                out.append({"index": i, "text": txt, "token_ids": toks, "finish_reason": fin,
                            "logprobs": {"token_logprobs": lps, "tokens": [str(x) for x in toks]}})
            return out

    # ---- http
    async def completions(self, req):
        b = await req.json()
        name = b.get("model", "base")
        if name != "base" and name not in self.loras:
            return web.json_response({"error": f"model {name} not found"}, status=404)
        self.stats["requests"] += 1
        self.stats["by_model"][name] = self.stats["by_model"].get(name, 0) + 1
        ch = await asyncio.to_thread(self.generate, name, b["prompt"], int(b.get("n", 1)), int(b["max_tokens"]),
                                     float(b.get("temperature", 1.0)), b.get("seed"))
        return web.json_response({"id": "x", "object": "text_completion", "model": name, "choices": ch})

    async def load(self, req):
        b = await req.json()
        name, path = b["lora_name"], b["lora_path"]
        if name in self.loras and not b.get("load_inplace"):
            return web.json_response({"error": f"adapter {name} already loaded"}, status=400)
        try:
            d = await asyncio.to_thread(self.build, path)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=400)
        with self.lock:
            self.loras[name] = d
            if self.cur == name:
                self.cur = None
        self.stats["loads"] += 1
        return web.Response(text=f"Success: LoRA adapter '{name}' added successfully.")

    async def unload(self, req):
        b = await req.json()
        name = b["lora_name"]
        if name not in self.loras:
            return web.json_response({"error": "not found"}, status=404)
        with self.lock:
            self.loras.pop(name)
            if self.cur == name:
                self.cur = None
        return web.Response(text=f"Success: LoRA adapter '{name}' removed successfully.")

    async def health(self, req):
        return web.Response(text="")

    async def server_info(self, req):
        return web.json_response({"lora_config": {"max_loras": 4}, "parallel_config": {"data_parallel_size": 1},
                                  "fake": True, **self.stats})

    async def models(self, req):
        return web.json_response({"object": "list", "data": [{"id": "base"}] + [{"id": k} for k in self.loras]})

    async def metrics(self, req):
        return web.Response(text=f"fake_requests_total {self.stats['requests']}\n")

    async def ok(self, req):
        return web.json_response({"ok": True})

    def app(self):
        a = web.Application(client_max_size=1 << 30)
        a.router.add_post("/v1/completions", self.completions)
        a.router.add_post("/v1/load_lora_adapter", self.load)
        a.router.add_post("/v1/unload_lora_adapter", self.unload)
        a.router.add_get("/health", self.health)
        a.router.add_get("/server_info", self.server_info)
        a.router.add_get("/v1/models", self.models)
        a.router.add_get("/metrics", self.metrics)
        for pth in ("/pause", "/resume", "/reset_prefix_cache", "/sleep", "/wake_up"):
            a.router.add_post(pth, self.ok)
        return a


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--eos", type=int, default=None)
    a = ap.parse_args()
    web.run_app(Fake(a.model, a.eos).app(), host=a.host, port=a.port, access_log=None)
