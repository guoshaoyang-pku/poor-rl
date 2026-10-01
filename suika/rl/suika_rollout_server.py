"""OpenAI/vLLM-compatible rollout server for trl AsyncGRPO on backbones vLLM
cannot serve (Qwen3.5 hybrid; later the vision QwenViTQ).

Endpoint subset trl 1.14's async stack calls:
  GET  /health, /v1/models, /get_world_size
  GET  /server_info?config_format=json   (lora capability probe)
  POST /v1/completions                   (prompt = token-id list; returns
                                          token_ids + per-token logprobs)
  POST /pause?mode=keep  /resume         (weight-sync barrier)
  POST /v1/load_lora_adapter             {lora_name, lora_path}
  POST /v1/unload_lora_adapter           {lora_name}

LoRA versioning: trl syncs `trl-policy-v{N}` adapters while stale
(<= max_staleness) in-flight requests still hit older names, so the server
keeps NSLOTS independent base+adapter copies and routes by model name.
Per-slot micro-batching: requests queue for up to `batch_window` ms (or
`max_batch` requests) and run one padded batched generate.

Usage:
  CUDA_VISIBLE_DEVICES=0 python suika_rollout_server.py \
      --model /path/to/merged_bc_hf --port 8000 --slots 4
"""
import argparse
import asyncio
import time

import torch
import uvicorn
from fastapi import FastAPI, Request

_BATCH_WINDOW = 0.02
_MAX_BATCH = 32


class Slot:
    def __init__(self, model_path, dtype):
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype).cuda().eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.version = None
        self.queue = asyncio.Queue()
        self.lock = asyncio.Lock()

    def load_adapter(self, name, path):
        from peft import PeftModel
        m = self.model
        if isinstance(m, PeftModel):
            m = m.unload()
        self.model = PeftModel.from_pretrained(
            m, path, is_trainable=False).cuda().eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.version = name


class Server:
    def __init__(self, model_path, slots, dtype=torch.bfloat16):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.slots = [Slot(model_path, dtype) for _ in range(slots)]
        self.paused = asyncio.Event()
        self.app = self._build()

    def _slot(self, name):
        for s in self.slots:
            if s.version == name:
                return s
        raise KeyError(f"unknown model/adapter {name!r}; loaded: "
                       f"{[s.version for s in self.slots]}")

    async def _batcher(self, slot):
        tok = self.tok
        while True:
            first = await slot.queue.get()
            batch = [first]
            t0 = time.monotonic()
            while len(batch) < _MAX_BATCH:
                try:
                    item = await asyncio.wait_for(
                        slot.queue.get(),
                        timeout=max(0.0, _BATCH_WINDOW
                                    - (time.monotonic() - t0)))
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
            try:
                results = await asyncio.to_thread(
                    _generate_batch, slot.model, tok,
                    [b[1] for b in batch])
                for (fut, _), res in zip(batch, results):
                    if not fut.done():
                        fut.set_result(res)
            except Exception as e:                     # noqa: BLE001
                for fut, _ in batch:
                    if not fut.done():
                        fut.set_exception(e)

    def _build(self):
        app = FastAPI()
        srv = self

        @app.on_event("startup")
        async def startup():
            for s in srv.slots:
                asyncio.create_task(srv._batcher(s))

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.get("/v1/models")
        async def models():
            return {"object": "list", "data": [
                {"id": s.version, "object": "model"}
                for s in srv.slots if s.version]}

        @app.get("/server_info")
        async def server_info(config_format: str = "json"):
            # trl probes ["lora_config"] to decide adapter sync is possible
            return {"lora_config": {"max_lora_rank": 64},
                    "max_model_len": 24576}

        @app.get("/get_world_size")
        async def world_size():
            return {"world_size": 1}

        @app.post("/pause")
        async def pause(mode: str = "keep"):
            srv.paused.set()
            return {"status": "paused"}

        @app.post("/resume")
        async def resume():
            srv.paused.clear()
            return {"status": "resumed"}

        @app.post("/v1/load_lora_adapter")
        async def load_lora(req: Request):
            body = await req.json()
            name, path = body["lora_name"], body["lora_path"]
            victim = next((s for s in srv.slots if s.version == name), None)
            if victim is None:
                free = [s for s in srv.slots if s.version is None]
                victim = free[0] if free else min(
                    srv.slots, key=lambda s: s.version or "")
            async with victim.lock:
                await asyncio.to_thread(victim.load_adapter, name, path)
            return {"status": "loaded", "lora_name": name}

        @app.post("/v1/unload_lora_adapter")
        async def unload_lora(req: Request):
            body = await req.json()
            for s in srv.slots:
                if s.version == body["lora_name"]:
                    s.version = None       # keep weights; just unroute
            return {"status": "unloaded"}

        @app.post("/v1/completions")
        async def completions(req: Request):
            body = await req.json()
            while srv.paused.is_set():
                await asyncio.sleep(0.05)
            name = body["model"]
            ids = body["prompt"]
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = ids[0]
            payload = {
                "ids": ids,
                "max_tokens": int(body.get("max_tokens", 16)),
                "temperature": float(body.get("temperature", 1.0)),
                "top_p": float(body.get("top_p", 1.0)),
                "top_k": int(body.get("top_k") or 0),
            }
            slot = srv._slot(name)
            fut = asyncio.get_event_loop().create_future()
            await slot.queue.put((fut, payload))
            out_ids, logprobs = await fut
            return {
                "id": f"cmpl-{time.time_ns()}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": name,
                "choices": [{
                    "index": 0,
                    "text": srv.tok.decode(out_ids),
                    "token_ids": out_ids,
                    "logprobs": {"token_logprobs": logprobs,
                                 "tokens": [srv.tok.decode([t])
                                            for t in out_ids]},
                    "finish_reason": "stop" if len(out_ids)
                    < payload["max_tokens"] else "length",
                }],
                "usage": {"prompt_tokens": len(ids),
                          "completion_tokens": len(out_ids),
                          "total_tokens": len(ids) + len(out_ids)},
            }

        return app


def _generate_batch(model, tok, payloads):
    """Padded batched generate; returns [(token_ids, logprobs), ...] with
    per-sequence EOS trimming."""
    max_tokens = max(p["max_tokens"] for p in payloads)
    idlists = [p["ids"] for p in payloads]
    n = len(idlists)
    L = max(len(x) for x in idlists)
    pad = tok.pad_token_id
    x = torch.full((n, L), pad, dtype=torch.long)
    attn = torch.zeros((n, L), dtype=torch.long)
    for i, ids in enumerate(idlists):          # left pad: generate appends
        x[i, L - len(ids):] = torch.tensor(ids)
        attn[i, L - len(ids):] = 1
    x, attn = x.cuda(), attn.cuda()
    temps = {p["temperature"] for p in payloads}
    temp = max(temps) if temps else 1.0
    top_p = max(p["top_p"] for p in payloads)
    top_k = max(p["top_k"] for p in payloads)
    with torch.no_grad():
        out = model.generate(
            x, attention_mask=attn, max_new_tokens=max_tokens,
            do_sample=temp > 0, temperature=max(temp, 1e-5),
            top_p=top_p, top_k=top_k if top_k > 0 else None,
            return_dict_in_generate=True, output_scores=True,
            pad_token_id=pad)
    gen = out.sequences[:, L:]
    eos = tok.eos_token_id
    results = []
    for i in range(n):
        ids = gen[i].tolist()
        lps = []
        for t, score in enumerate(out.scores):
            lp = torch.log_softmax(score[i].float(), dim=-1)
            lps.append(float(lp[ids[t]]))
        if eos in ids:
            cut = ids.index(eos) + 1
            ids, lps = ids[:cut], lps[:cut]
        results.append((ids, lps))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--slots", type=int, default=4)
    args = ap.parse_args()
    srv = Server(args.model, args.slots)
    uvicorn.run(srv.app, host="127.0.0.1", port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()
