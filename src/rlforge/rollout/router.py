"""Multi-backend vLLM rollout router with group affinity and adapter-only LoRA sync.

One OpenAI-compatible endpoint in front of N independent `vllm serve` processes (any mix of
TP sizes, any hosts reachable over TCP). TRL 1.14's AsyncGRPOTrainer / AsyncRolloutWorker talk to
ONE `vllm_server_base_url`; pointing it at this router needs no TRL change:

* `/v1/completions`, `/v1/chat/completions`: group-affine dispatch. A prompt is hashed; requests
  with the same prompt go to the backend the prompt is pinned to (prefix-cache affinity, the
  same idea as production/v3_1/src/rlforge/dp_route.py). New prompts go to the backend with the
  lowest in-flight/weight. A pin is held while any request of the prompt is in flight and is
  then kept in an LRU, re-placed only if its backend became clearly busier than the least
  loaded one. Connection failures fail over to another backend.
* `/pause`, `/resume`, `/reset_prefix_cache`, `/sleep`, `/wake_up`: fanned out to all backends.
* `/v1/load_lora_adapter`: the adapter directory named by `lora_path` is read on the ROUTER host
  (where the trainer wrote it), optionally cast to bf16 (vLLM serves LoRA in the model dtype,
  so this is lossless for serving and halves the bytes), pushed over HTTP to a `lora_agent`
  on every remote host, and loaded into every backend under the same name. All-or-nothing:
  if any backend fails, the backends that succeeded unload it again and the call fails, so
  TRL never bumps `model_version` to a name some backend lacks.
* `/v1/unload_lora_adapter`: fanned out; staged copies are deleted one unload later (vLLM
  resolves `lora_path` lazily, a missing directory is fatal for it).
* `/server_info`: the first backend's, with `lora_config.max_loras` reduced to the minimum over
  backends, `parallel_config.data_parallel_size` == 1 (true per backend), so TRL's
  `select_adapter_sync` picks the adapter path. A `router` key lists the backends.
* `/metrics`: every backend's Prometheus text with a `backend="i"` label injected into each
  sample, plus `router_*` series. Counter sums over label sets (what bench_sat does) are
  therefore totals over backends; gauges such as kv_cache_usage_perc are summed too.
* merged-weight NCCL endpoints (`/init_weight_transfer_engine`, `/update_weights`, ...)
  return 501: a full-weight sync of 27B (54 GB) over Ethernet is what this router avoids.
* health loop: a backend that stops answering is taken out of rotation; when it is back, every
  adapter currently registered is re-loaded into it before it serves again.

Run: python -m rlforge.rollout.router --port 8300 \
        --backend http://127.0.0.1:8201 --backend http://172.16.0.15:8201,agent=http://172.16.0.15:8399
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import itertools
import json
import logging
import os
import re
import shutil
import time
from array import array
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from aiohttp import web

logger = logging.getLogger("rlforge.rollout.router")

FANOUT_POST = ("/pause", "/resume", "/reset_prefix_cache", "/sleep", "/wake_up",
               "/reset_mm_cache", "/collective_rpc_disabled")
MERGED_SYNC = ("/init_weight_transfer_engine", "/start_weight_update", "/update_weights",
               "/finish_weight_update", "/update_weights_from_disk")
HOP = {"host", "content-length", "transfer-encoding", "connection", "keep-alive",
       "content-encoding", "accept-encoding"}
NAME_RE = re.compile(r"^[A-Za-z0-9._@+-]{1,200}$")


# --------------------------------------------------------------------------------------------
# backends and affinity

@dataclass
class Backend:
    idx: int
    url: str
    weight: float = 1.0
    agent: str | None = None          # lora_agent URL on the backend's host; None = shares the router's FS
    host: str = ""                   # grouping key for adapter pushes (one push per host)
    healthy: bool = False
    inflight: int = 0
    requests: int = 0
    errors: int = 0
    failovers: int = 0
    last_ok: float = 0.0
    loras: set = field(default_factory=set)


def parse_backend(spec: str, idx: int) -> Backend:
    """`URL[,weight=W][,agent=URL][,host=NAME]`."""
    parts = spec.split(",")
    b = Backend(idx=idx, url=parts[0].rstrip("/"))
    for p in parts[1:]:
        k, _, v = p.partition("=")
        if k == "weight":
            b.weight = float(v)
        elif k == "agent":
            b.agent = v.rstrip("/")
        elif k == "host":
            b.host = v
        else:
            raise ValueError(f"unknown backend option {k!r} in {spec!r}")
    if not b.host:
        b.host = b.agent or "local"
    return b


def prompt_key(body: dict) -> bytes | None:
    """Affinity key of a completions / chat request: hash of the prompt (token ids or text)."""
    h = hashlib.blake2b(digest_size=16)
    p = body.get("prompt")
    if p is None:
        msgs = body.get("messages")
        if msgs is None:
            return None
        h.update(json.dumps(msgs, sort_keys=True, ensure_ascii=False).encode())
        return h.digest()
    if isinstance(p, list) and p and isinstance(p[0], list):
        p = p[0]  # batched token prompts: route the batch by its first prompt
    elif isinstance(p, list) and p and isinstance(p[0], str):
        p = p[0]
    if isinstance(p, str):
        h.update(b"s")
        h.update(p.encode())
    else:
        try:
            h.update(b"t")
            h.update(array("q", p).tobytes())
        except (TypeError, OverflowError):
            h.update(json.dumps(p).encode())
    return h.digest()


class Affinity:
    """Sticky prompt -> backend map.

    `pick` pins a new key to the healthy backend with the lowest (inflight + 1) / weight
    (rotating tie-break). A key with requests in flight always stays on its backend, so the G
    samples of one group are prefilled once. After its last request ends the pin stays in an
    LRU (default 65536 keys) and is re-used while the backend's normalized load is at most
    `slack_ratio` x the least-loaded backend's + `slack_abs`; otherwise the key is re-placed.
    """

    def __init__(self, backends: list[Backend], lru: int = 65536, slack_ratio: float = 1.5,
                 slack_abs: float = 4.0):
        self.b = backends
        self.lru = lru
        self.slack_ratio = slack_ratio
        self.slack_abs = slack_abs
        self.pins: collections.OrderedDict[bytes, list[int]] = collections.OrderedDict()
        self._rr = 0
        self.hits = 0      # request routed to an existing pin
        self.new = 0       # new pin
        self.moved = 0     # LRU pin re-placed because its backend was overloaded

    def _norm(self, i: int) -> float:
        return (self.b[i].inflight + 1) / self.b[i].weight

    def _least(self, exclude: set[int] = frozenset()) -> int:
        cand = [i for i, b in enumerate(self.b) if b.healthy and i not in exclude]
        if not cand:
            raise RuntimeError("no healthy backend")
        self._rr = (self._rr + 1) % len(self.b)
        return min(cand, key=lambda i: (self._norm(i), (i - self._rr) % len(self.b)))

    def pick(self, key: bytes | None, exclude: set[int] = frozenset()) -> int:
        if key is None:
            i = self._least(exclude)
            self.b[i].inflight += 1
            return i
        e = self.pins.get(key)
        if e is not None and self.b[e[0]].healthy and e[0] not in exclude:
            keep = e[1] > 0
            if not keep:
                least = self._least(exclude)
                keep = self._norm(e[0]) <= self.slack_ratio * self._norm(least) + self.slack_abs / self.b[e[0]].weight
                if not keep:
                    self.moved += 1
                    e[0] = least
            if keep:
                self.hits += 1
            self.pins.move_to_end(key)
        else:
            i = self._least(exclude)
            if e is None:
                e = [i, 0]
                self.pins[key] = e
                self.new += 1
                if len(self.pins) > self.lru:   # evict the oldest pins that have nothing in flight
                    for k0 in list(itertools.islice(self.pins.keys(), 0, 256)):
                        if self.pins[k0][1] == 0 and k0 != key:
                            del self.pins[k0]
                            if len(self.pins) <= self.lru:
                                break
            else:                       # pinned backend unhealthy / excluded: re-place
                self.moved += 1
                e[0] = i
        e[1] += 1
        self.b[e[0]].inflight += 1
        return e[0]

    def release(self, key: bytes | None, i: int) -> None:
        self.b[i].inflight -= 1
        if key is None:
            return
        e = self.pins.get(key)
        if e is not None and e[1] > 0:
            e[1] -= 1


# --------------------------------------------------------------------------------------------
# adapter staging (cast + push)

def stage_adapter(src: str, dst: str, ship_dtype: str) -> dict:
    """Copy (or cast to bf16) a PEFT adapter dir into `dst` (atomic rename). Returns file meta."""
    import torch
    from safetensors.torch import load_file, save_file

    t0 = time.time()
    tmp = os.path.join(os.path.dirname(dst), f".{os.path.basename(dst)}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    files = []
    src_bytes = 0
    n_cast = 0
    for fn in sorted(os.listdir(src)):
        sp = os.path.join(src, fn)
        if not os.path.isfile(sp):
            continue
        src_bytes += os.path.getsize(sp)
        dp = os.path.join(tmp, fn)
        if fn.endswith(".safetensors") and ship_dtype == "bf16":
            sd = load_file(sp)
            out = {}
            for k, v in sd.items():
                if v.is_floating_point() and v.dtype != torch.bfloat16:
                    v = v.to(torch.bfloat16)
                    n_cast += 1
                out[k] = v.contiguous()
            save_file(out, dp, metadata={"format": "pt"})
        elif fn.endswith((".json", ".safetensors", ".bin", ".model", ".txt")):
            shutil.copyfile(sp, dp)
        else:
            continue  # README.md etc.
        files.append(fn)
    t_write = time.time()
    meta = []
    for fn in files:
        p = os.path.join(tmp, fn)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                h.update(chunk)
        meta.append({"name": fn, "size": os.path.getsize(p), "sha256": h.hexdigest()})
    if os.path.exists(dst):
        old = dst + f".old{time.time_ns()}"
        os.rename(dst, old)
        shutil.rmtree(old, ignore_errors=True)
    os.rename(tmp, dst)
    return {"files": meta, "src_bytes": src_bytes, "ship_bytes": sum(m["size"] for m in meta),
            "n_cast": n_cast, "stage_s": round(t_write - t0, 4), "hash_s": round(time.time() - t_write, 4)}


async def _push_range(session, url, path, offset, length, total, token):
    def reader():
        with open(path, "rb") as f:
            f.seek(offset)
            left = length
            while left > 0:
                b = f.read(min(4 << 20, left))
                if not b:
                    break
                left -= len(b)
                yield b

    async def agen():
        for b in reader():
            yield b

    headers = {"X-Lora-Token": token} if token else {}
    async with session.put(url, params={"offset": str(offset), "total": str(total)}, data=agen(),
                           headers=headers) as r:
        if r.status != 200:
            raise RuntimeError(f"PUT {url} -> {r.status}: {(await r.text())[:300]}")
        return await r.json()


async def push_to_agent(session: aiohttp.ClientSession, agent: str, name: str, stage_dir: str,
                        meta: list[dict], streams: int, token: str | None) -> dict:
    """Push the staged adapter to one agent with `streams` parallel byte-range PUTs per big file."""
    t0 = time.time()
    headers = {"X-Lora-Token": token} if token else {}
    async with session.post(f"{agent}/adapters/{name}/begin", headers=headers) as r:
        if r.status != 200:
            raise RuntimeError(f"begin on {agent} failed: {r.status} {(await r.text())[:200]}")
    tasks = []
    for m in meta:
        p = os.path.join(stage_dir, m["name"])
        size = m["size"]
        n = streams if size >= (16 << 20) else 1
        step = -(-size // n) if size else 0
        url = f"{agent}/adapters/{name}/{m['name']}"
        if size == 0:
            tasks.append(_push_range(session, url, p, 0, 0, 0, token))
        for off in range(0, size, max(step, 1)):
            tasks.append(_push_range(session, url, p, off, min(step, size - off), size, token))
    await asyncio.gather(*tasks)
    t1 = time.time()
    async with session.post(f"{agent}/adapters/{name}/commit", json={"files": meta}, headers=headers) as r:
        body = await r.json()
        if r.status != 200:
            raise RuntimeError(f"commit on {agent} failed: {body}")
    return {"agent": agent, "path": body["path"], "push_s": round(t1 - t0, 4),
            "commit_s": round(time.time() - t1, 4), "verify_s": body.get("verify_s"),
            "bytes": sum(m["size"] for m in meta)}


# --------------------------------------------------------------------------------------------
# the router

class Router:
    def __init__(self, backends: list[Backend], stage_root: str, ship_dtype: str = "bf16",
                 push_streams: int = 4, agent_token: str | None = None, health_s: float = 5.0,
                 request_timeout: float = 3600.0, min_healthy: int | None = None,
                 log_path: str | None = None, affinity: Affinity | None = None):
        self.backends = backends
        self.aff = affinity or Affinity(backends)
        self.stage_root = os.path.abspath(stage_root)
        os.makedirs(self.stage_root, exist_ok=True)
        self.ship_dtype = ship_dtype
        self.push_streams = push_streams
        self.agent_token = agent_token
        self.health_s = health_s
        self.request_timeout = request_timeout
        self.min_healthy = len(backends) if min_healthy is None else min_healthy
        self.registry: dict[str, dict] = {}       # lora name -> {"stage": dir, "remote": {agent: path}}
        self._pending_delete: list[tuple[str, dict]] = []
        self.admin_lock = asyncio.Lock()
        self.session: aiohttp.ClientSession | None = None
        self.log_path = log_path
        self.sync_hist: list[dict] = []
        self.t_start = time.time()

    # ---- helpers
    def _log(self, rec: dict) -> None:
        rec = {"t": round(time.time(), 3), **rec}
        logger.info(json.dumps(rec))
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    async def _req(self, b: Backend, method: str, path: str, *, timeout: float = 1800, **kw):
        async with self.session.request(method, b.url + path,
                                        timeout=aiohttp.ClientTimeout(total=timeout), **kw) as r:
            return r.status, await r.read(), r.headers.get("Content-Type", "")

    async def _fanout(self, method: str, path: str, *, only_healthy: bool = True, timeout: float = 1800, **kw):
        targets = [b for b in self.backends if b.healthy or not only_healthy]

        async def one(b):
            t = time.time()
            try:
                st, body, _ = await self._req(b, method, path, timeout=timeout, **kw)
                return {"backend": b.idx, "status": st, "body": body[:400].decode(errors="replace"),
                        "s": round(time.time() - t, 4)}
            except Exception as e:  # noqa: BLE001
                return {"backend": b.idx, "status": 599, "body": f"{type(e).__name__}: {e}",
                        "s": round(time.time() - t, 4)}

        return await asyncio.gather(*(one(b) for b in targets))

    # ---- lifecycle
    async def start(self, app):
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0, ttl_dns_cache=None),
                                             auto_decompress=False)
        for b in self.backends:
            await self._check(b, initial=True)
        self._health_task = asyncio.create_task(self._health_loop())
        self._log({"event": "start", "backends": [(b.idx, b.url, b.weight, b.agent, b.healthy) for b in self.backends]})

    async def stop(self, app):
        self._health_task.cancel()
        await self.session.close()

    async def _check(self, b: Backend, initial: bool = False) -> None:
        try:
            st, _, _ = await self._req(b, "GET", "/health", timeout=10)
            ok = st == 200
        except Exception:  # noqa: BLE001
            ok = False
        if ok and not b.healthy:
            if not initial and self.registry:
                # a backend that came back lost its adapters: re-load all registered ones first
                try:
                    async with self.admin_lock:
                        for name, reg in self.registry.items():
                            await self._load_one(b, name, reg)
                except Exception as e:  # noqa: BLE001
                    self._log({"event": "readmit_failed", "backend": b.idx, "err": str(e)[:300]})
                    return
            b.healthy = True
            b.last_ok = time.time()
            self._log({"event": "backend_up", "backend": b.idx})
        elif not ok and b.healthy:
            b.healthy = False
            self._log({"event": "backend_down", "backend": b.idx})
        elif ok:
            b.last_ok = time.time()

    async def _health_loop(self):
        while True:
            await asyncio.sleep(self.health_s)
            await asyncio.gather(*(self._check(b) for b in self.backends), return_exceptions=True)

    # ---- generation proxy
    async def handle_generate(self, request: web.Request) -> web.StreamResponse:
        raw = await request.read()
        try:
            body = json.loads(raw)
        except ValueError:
            return web.json_response({"error": "invalid json"}, status=400)
        key = prompt_key(body)
        stream = bool(body.get("stream"))
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        tried: set[int] = set()
        last_err = None
        for _attempt in range(len(self.backends)):
            try:
                i = self.aff.pick(key, exclude=tried)
            except RuntimeError as e:
                last_err = str(e)
                break
            b = self.backends[i]
            b.requests += 1
            try:
                async with self.session.post(b.url + request.path_qs, data=raw, headers=headers,
                                             timeout=aiohttp.ClientTimeout(total=self.request_timeout)) as r:
                    if not stream:
                        data = await r.read()
                        resp = web.Response(body=data, status=r.status,
                                            content_type=(r.content_type or "application/json"))
                        resp.headers["X-Router-Backend"] = str(i)
                        return resp
                    resp = web.StreamResponse(status=r.status)
                    resp.content_type = r.content_type or "text/event-stream"
                    resp.headers["X-Router-Backend"] = str(i)
                    await resp.prepare(request)
                    async for chunk in r.content.iter_any():
                        await resp.write(chunk)
                    await resp.write_eof()
                    return resp
            except (aiohttp.ClientConnectorError, aiohttp.ServerDisconnectedError,
                    aiohttp.ClientOSError) as e:
                # connection-level failure before a response: fail over to another backend
                b.errors += 1
                b.failovers += 1
                tried.add(i)
                last_err = f"{type(e).__name__}: {e}"
                asyncio.create_task(self._check(b))
            finally:
                self.aff.release(key, i)
        return web.json_response({"error": f"all backends failed: {last_err}"}, status=503)

    # ---- fan-out admin endpoints
    async def handle_fanout(self, request: web.Request) -> web.Response:
        raw = await request.read()
        async with self.admin_lock:
            t = time.time()
            res = await self._fanout(request.method, request.path_qs, data=raw or None,
                                     headers={"Content-Type": request.headers.get("Content-Type", "application/json")})
            ok = all(r["status"] == 200 for r in res)
            self._log({"event": "fanout", "path": request.path_qs, "ok": ok, "s": round(time.time() - t, 4),
                       "per_backend": [(r["backend"], r["status"], r["s"]) for r in res]})
        return web.json_response({"ok": ok, "results": res}, status=200 if ok else 502)

    async def _load_one(self, b: Backend, name: str, reg: dict, inplace: bool = False) -> dict:
        path = reg["remote"][b.agent] if b.agent else reg["stage"]
        t = time.time()
        st, body, _ = await self._req(b, "POST", "/v1/load_lora_adapter",
                                      json={"lora_name": name, "lora_path": path, "load_inplace": inplace})
        dt = time.time() - t
        if st != 200:
            raise RuntimeError(f"backend {b.idx} load {name}: {st} {body[:300]!r}")
        b.loras.add(name)
        return {"backend": b.idx, "load_s": round(dt, 4)}

    async def handle_load_lora(self, request: web.Request) -> web.Response:
        req = await request.json()
        name, src = req.get("lora_name"), req.get("lora_path")
        inplace = bool(req.get("load_inplace", False))
        if not name or not NAME_RE.match(name) or not src:
            return web.json_response({"error": "lora_name/lora_path required"}, status=400)
        async with self.admin_lock:
            t0 = time.time()
            rec: dict[str, Any] = {"event": "load_lora", "name": name, "src": src, "inplace": inplace}
            try:
                if not os.path.isdir(src):
                    raise FileNotFoundError(f"lora_path {src} is not a directory on the router host")
                stage = os.path.join(self.stage_root, name)
                meta = await asyncio.to_thread(stage_adapter, src, stage, self.ship_dtype)
                rec.update({k: meta[k] for k in ("src_bytes", "ship_bytes", "n_cast", "stage_s", "hash_s")})
                t1 = time.time()
                agents = sorted({b.agent for b in self.backends if b.agent})
                pushes = await asyncio.gather(*(push_to_agent(self.session, a, name, stage, meta["files"],
                                                              self.push_streams, self.agent_token)
                                                for a in agents))
                t2 = time.time()
                reg = {"stage": stage, "remote": {p["agent"]: p["path"] for p in pushes}, "meta": meta["files"]}
                targets = [b for b in self.backends if b.healthy]
                loads = await asyncio.gather(*(self._load_one(b, name, reg, inplace) for b in targets),
                                             return_exceptions=True)
                t3 = time.time()
                errs = [str(x) for x in loads if isinstance(x, Exception)]
                if errs:
                    if not inplace:  # all-or-nothing: roll back the backends that took it
                        await asyncio.gather(*(self._unload_one(b, name) for b in targets if name in b.loras),
                                             return_exceptions=True)
                    raise RuntimeError("; ".join(errs))
                self.registry[name] = reg
                rec.update({"ok": True, "pushes": pushes, "loads": loads, "push_wall_s": round(t2 - t1, 4),
                            "load_wall_s": round(t3 - t2, 4), "total_s": round(t3 - t0, 4),
                            "n_backends": len(targets)})
                self.sync_hist.append(rec)
                self._log(rec)
                return web.json_response(rec)
            except Exception as e:  # noqa: BLE001
                rec.update({"ok": False, "error": f"{type(e).__name__}: {e}", "total_s": round(time.time() - t0, 4)})
                self._log(rec)
                return web.json_response(rec, status=500)

    async def _unload_one(self, b: Backend, name: str):
        st, body, _ = await self._req(b, "POST", "/v1/unload_lora_adapter", json={"lora_name": name})
        b.loras.discard(name)
        return st

    async def handle_unload_lora(self, request: web.Request) -> web.Response:
        req = await request.json()
        name = req.get("lora_name", "")
        async with self.admin_lock:
            t = time.time()
            res = await self._fanout("POST", "/v1/unload_lora_adapter", json={"lora_name": name})
            for b in self.backends:
                b.loras.discard(name)
            # delete the copies of the adapters unloaded at the PREVIOUS unload (lazy path resolution)
            for old_name, reg in self._pending_delete:
                shutil.rmtree(reg["stage"], ignore_errors=True)
                for agent in reg["remote"]:
                    try:
                        await self.session.delete(f"{agent}/adapters/{old_name}",
                                                  headers={"X-Lora-Token": self.agent_token} if self.agent_token else {})
                    except Exception:  # noqa: BLE001
                        pass
            reg = self.registry.pop(name, None)
            self._pending_delete = [(name, reg)] if reg else []
            # vLLM answers 404 for unknown names; TRL ignores the status. Report 200 unless a backend is unreachable.
            ok = all(r["status"] in (200, 404, 400) for r in res)
            self._log({"event": "unload_lora", "name": name, "ok": ok, "s": round(time.time() - t, 4),
                       "per_backend": [(r["backend"], r["status"]) for r in res]})
        return web.json_response({"ok": ok, "results": res}, status=200 if ok else 502)

    # ---- single-backend / aggregate GETs
    def _first(self) -> Backend:
        for b in self.backends:
            if b.healthy:
                return b
        raise web.HTTPServiceUnavailable(text="no healthy backend")

    async def handle_health(self, request):
        n = sum(b.healthy for b in self.backends)
        return web.Response(status=200 if n >= self.min_healthy else 503,
                            text=f"{n}/{len(self.backends)} backends healthy")

    async def handle_server_info(self, request):
        infos = await asyncio.gather(*(self._req(b, "GET", "/server_info" + ("?" + request.query_string if request.query_string else ""), timeout=60)
                                       for b in self.backends if b.healthy), return_exceptions=True)
        good = [json.loads(x[1]) for x in infos if not isinstance(x, Exception) and x[0] == 200]
        if not good:
            return web.json_response({"error": "no backend server_info"}, status=503)
        out = good[0]
        vc = out.get("vllm_config", {})
        lcs = [g.get("vllm_config", {}).get("lora_config") for g in good]
        if any(lc is None for lc in lcs):
            vc["lora_config"] = None  # a backend without --enable-lora: adapter sync impossible
        elif lcs:
            vc["lora_config"]["max_loras"] = min(lc["max_loras"] for lc in lcs)
            vc["lora_config"]["max_lora_rank"] = min(lc["max_lora_rank"] for lc in lcs)
        out["router"] = {"backends": [{"idx": b.idx, "url": b.url, "weight": b.weight, "agent": b.agent,
                                       "healthy": b.healthy} for b in self.backends]}
        return web.json_response(out)

    async def handle_metrics(self, request):
        res = await asyncio.gather(*(self._req(b, "GET", "/metrics", timeout=30) for b in self.backends if b.healthy),
                                   return_exceptions=True)
        out = []
        healthy = [b for b in self.backends if b.healthy]
        for b, r in zip(healthy, res):
            if isinstance(r, Exception) or r[0] != 200:
                continue
            lab = f'backend="{b.idx}"'
            for line in r[1].decode(errors="replace").splitlines():
                if not line or line.startswith("#"):
                    continue
                m = re.match(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?(\s.*)$", line)
                if not m:
                    continue
                labels = m.group(2)
                labels = "{" + lab + ("," + labels[1:-1] if labels and labels != "{}" else "") + "}"
                out.append(f"{m.group(1)}{labels}{m.group(3)}")
        for b in self.backends:
            lab = f'backend="{b.idx}"'
            out += [f"router_inflight{{{lab}}} {b.inflight}", f"router_requests_total{{{lab}}} {b.requests}",
                    f"router_errors_total{{{lab}}} {b.errors}", f"router_healthy{{{lab}}} {int(b.healthy)}"]
        out += [f"router_affinity_hits_total {self.aff.hits}", f"router_affinity_new_total {self.aff.new}",
                f"router_affinity_moved_total {self.aff.moved}", f"router_pins {len(self.aff.pins)}",
                f"router_lora_syncs_total {len(self.sync_hist)}"]
        return web.Response(text="\n".join(out) + "\n", content_type="text/plain")

    async def handle_router_state(self, request):
        return web.json_response({
            "backends": [{"idx": b.idx, "url": b.url, "weight": b.weight, "agent": b.agent, "healthy": b.healthy,
                          "inflight": b.inflight, "requests": b.requests, "errors": b.errors,
                          "failovers": b.failovers, "loras": sorted(b.loras)} for b in self.backends],
            "affinity": {"hits": self.aff.hits, "new": self.aff.new, "moved": self.aff.moved, "pins": len(self.aff.pins)},
            "registry": sorted(self.registry), "syncs": self.sync_hist[-20:]})

    async def handle_merged(self, request):
        return web.json_response({"error": "merged full-weight sync is not supported by the rollout router; "
                                           "start backends with --enable-lora for adapter-only sync"}, status=501)

    async def handle_passthrough(self, request: web.Request):
        b = self._first()
        raw = await request.read()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        st, body, ct = await self._req(b, request.method, request.path_qs, data=raw or None, headers=headers,
                                       timeout=self.request_timeout)
        return web.Response(body=body, status=st, content_type=(ct.split(";")[0] or None))

    def app(self) -> web.Application:
        app = web.Application(client_max_size=1 << 30)
        app.router.add_post("/v1/completions", self.handle_generate)
        app.router.add_post("/v1/chat/completions", self.handle_generate)
        for p in FANOUT_POST:
            app.router.add_post(p, self.handle_fanout)
        for p in MERGED_SYNC:
            app.router.add_post(p, self.handle_merged)
        app.router.add_post("/v1/load_lora_adapter", self.handle_load_lora)
        app.router.add_post("/v1/unload_lora_adapter", self.handle_unload_lora)
        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/server_info", self.handle_server_info)
        app.router.add_get("/metrics", self.handle_metrics)
        app.router.add_get("/router/state", self.handle_router_state)
        app.router.add_route("*", "/{tail:.*}", self.handle_passthrough)
        app.on_startup.append(self.start)
        app.on_cleanup.append(self.stop)
        return app


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("--backend", action="append", required=True,
                    help="URL[,weight=W][,agent=URL][,host=NAME]; repeat per vLLM server")
    ap.add_argument("--stage-root", required=True, help="router-host dir for staged (cast) adapters")
    ap.add_argument("--ship-dtype", choices=["bf16", "keep"], default="bf16")
    ap.add_argument("--push-streams", type=int, default=4)
    ap.add_argument("--agent-token", default=os.environ.get("RLFORGE_LORA_TOKEN"))
    ap.add_argument("--health-s", type=float, default=5.0)
    ap.add_argument("--request-timeout", type=float, default=3600.0)
    ap.add_argument("--min-healthy", type=int, default=None)
    ap.add_argument("--lru", type=int, default=65536)
    ap.add_argument("--slack-ratio", type=float, default=1.5)
    ap.add_argument("--slack-abs", type=float, default=4.0)
    ap.add_argument("--log", default=None, help="JSONL event log (syncs, fan-outs, health)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    backends = [parse_backend(s, i) for i, s in enumerate(a.backend)]
    r = Router(backends, a.stage_root, a.ship_dtype, a.push_streams, a.agent_token, a.health_s,
               a.request_timeout, a.min_healthy, a.log,
               Affinity(backends, a.lru, a.slack_ratio, a.slack_abs))
    web.run_app(r.app(), host=a.host, port=a.port, access_log=None, backlog=4096)


if __name__ == "__main__":
    main()
