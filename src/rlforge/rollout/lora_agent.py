"""Receiver for LoRA adapters pushed by the rollout router (one per rollout host).

The router pushes each adapter file with parallel byte-range PUTs, then commits; the agent
verifies size and sha256 of every file and publishes the directory with an atomic rename, so a
vLLM server on this host never sees a half-written adapter. The returned path is what the
router passes to that host's vLLM `/v1/load_lora_adapter`.

  POST   /adapters/{name}/begin                     wipe the staging dir of `name`
  PUT    /adapters/{name}/{file}?offset=O&total=T   body = bytes [O, O+len)
  POST   /adapters/{name}/commit                    {"files": [{"name","size","sha256"}]}
  DELETE /adapters/{name}
  GET    /health

Bind it to the host's private IP. Optional shared token (`X-Lora-Token`, env RLFORGE_LORA_TOKEN).

Run: python -m rlforge.rollout.lora_agent --host 172.16.0.15 --port 8399 --root /path/lora_recv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time

from aiohttp import web

NAME_RE = re.compile(r"^[A-Za-z0-9._@+-]{1,200}$")


class Agent:
    def __init__(self, root: str, token: str | None = None):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.token = token
        self.stats = {"bytes": 0, "puts": 0, "commits": 0, "deletes": 0}

    def _auth(self, request):
        if self.token and request.headers.get("X-Lora-Token") != self.token:
            raise web.HTTPForbidden(text="bad token")

    def _names(self, request, file=True):
        name = request.match_info["name"]
        if not NAME_RE.match(name) or name.startswith("."):
            raise web.HTTPBadRequest(text="bad adapter name")
        fn = request.match_info.get("file") if file else None
        if file and (not fn or not NAME_RE.match(fn) or fn.startswith(".")):
            raise web.HTTPBadRequest(text="bad file name")
        return name, fn

    def _tmp(self, name):
        return os.path.join(self.root, f".{name}.tmp")

    async def begin(self, request):
        self._auth(request)
        name, _ = self._names(request, file=False)
        shutil.rmtree(self._tmp(name), ignore_errors=True)
        os.makedirs(self._tmp(name))
        return web.json_response({"ok": True})

    async def put(self, request):
        self._auth(request)
        name, fn = self._names(request)
        off = int(request.query.get("offset", "0"))
        total = int(request.query.get("total", "0"))
        d = self._tmp(name)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, fn)
        t0 = time.time()
        fd = os.open(p, os.O_WRONLY | os.O_CREAT, 0o644)
        n = 0
        try:
            if total and os.fstat(fd).st_size < total:
                os.ftruncate(fd, total)
            pos = off
            async for chunk in request.content.iter_chunked(4 << 20):
                os.pwrite(fd, chunk, pos)
                pos += len(chunk)
                n += len(chunk)
        finally:
            os.close(fd)
        self.stats["bytes"] += n
        self.stats["puts"] += 1
        return web.json_response({"bytes": n, "s": round(time.time() - t0, 4)})

    async def commit(self, request):
        self._auth(request)
        name, _ = self._names(request, file=False)
        body = await request.json()
        d = self._tmp(name)
        t0 = time.time()
        for m in body["files"]:
            p = os.path.join(d, m["name"])
            if not os.path.isfile(p) or os.path.getsize(p) != m["size"]:
                return web.json_response({"error": f"{m['name']}: missing or size mismatch"}, status=409)
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(8 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != m["sha256"]:
                return web.json_response({"error": f"{m['name']}: sha256 mismatch"}, status=409)
        dst = os.path.join(self.root, name)
        if os.path.exists(dst):
            old = os.path.join(self.root, f".{name}.old{time.time_ns()}")
            os.rename(dst, old)
            shutil.rmtree(old, ignore_errors=True)
        os.rename(d, dst)
        self.stats["commits"] += 1
        return web.json_response({"path": dst, "verify_s": round(time.time() - t0, 4)})

    async def delete(self, request):
        self._auth(request)
        name, _ = self._names(request, file=False)
        shutil.rmtree(os.path.join(self.root, name), ignore_errors=True)
        shutil.rmtree(self._tmp(name), ignore_errors=True)
        self.stats["deletes"] += 1
        return web.json_response({"ok": True})

    async def health(self, request):
        return web.json_response({"ok": True, "root": self.root, **self.stats})

    def app(self):
        app = web.Application(client_max_size=1 << 34)
        app.router.add_post("/adapters/{name}/begin", self.begin)
        app.router.add_put("/adapters/{name}/{file}", self.put)
        app.router.add_post("/adapters/{name}/commit", self.commit)
        app.router.add_delete("/adapters/{name}", self.delete)
        app.router.add_get("/health", self.health)
        return app


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8399)
    ap.add_argument("--root", required=True)
    ap.add_argument("--token", default=os.environ.get("RLFORGE_LORA_TOKEN"))
    a = ap.parse_args(argv)
    web.run_app(Agent(a.root, a.token).app(), host=a.host, port=a.port, access_log=None)


if __name__ == "__main__":
    main()
