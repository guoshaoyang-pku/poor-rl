from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import samples_view


REPO_ROOT = Path(__file__).resolve().parents[1]
HOME_PAGE = REPO_ROOT / "dashboard_home.html"
STATUS_JSON = REPO_ROOT / "artifacts" / "gpu_status.json"
FAULTS_JSON = REPO_ROOT / "artifacts" / "gpu_faults.json"

CSP = ("default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
       "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_bytes(HOME_PAGE.read_bytes(), "text/html; charset=utf-8")
            return
        if self.path == "/api/gpu_status":
            if STATUS_JSON.is_file():
                self._send_bytes(STATUS_JSON.read_bytes(), "application/json; charset=utf-8")
            else:
                self._send_bytes(b'{"updated_at":null,"clusters":{},"faults":[],"faults_history":[]}',
                                 "application/json; charset=utf-8")
            return
        if self.path == "/api/samples_runs":
            root = REPO_ROOT / "artifacts" / "samples"
            runs = []
            if root.is_dir():
                for p in sorted(root.iterdir()):
                    if not p.is_dir():
                        continue
                    evals = sorted(f.stem[len("eval_"):] for f in p.glob("eval_*.jsonl"))
                    train = p / "train.jsonl"
                    n_train = 0
                    if train.is_file():
                        with open(train, "rb") as fh:
                            n_train = sum(chunk.count(b"\n") for chunk in
                                          iter(lambda: fh.read(1 << 20), b""))
                    runs.append({"id": p.name, "evals": evals, "train_rows": n_train})
            self._send_bytes(json.dumps(runs).encode(), "application/json; charset=utf-8")
            return
        if self.path.startswith("/samples/"):
            run_id = self.path[len("/samples/"):].strip("/")
            if run_id and "/" not in run_id and ".." not in run_id:
                self._send_bytes(samples_view.render_run(run_id), "text/html; charset=utf-8")
            else:
                self.send_error(404)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/api/gpu_faults/clear":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            key = str(payload.get("key", ""))
        except Exception:
            self.send_error(400, "invalid payload")
            return
        if not key or not FAULTS_JSON.is_file():
            self.send_error(404, "fault not found")
            return
        faults = json.loads(FAULTS_JSON.read_text())
        entry = faults.get(key)
        if entry is None or entry.get("cleared_at"):
            self.send_error(404, "fault not found")
            return
        ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        entry["cleared_at"] = ts
        entry["cleared_by"] = "manual"
        tmp = FAULTS_JSON.with_suffix(".tmp")
        tmp.write_text(json.dumps(faults, ensure_ascii=False, indent=1))
        tmp.replace(FAULTS_JSON)
        # reflect the clear in the served snapshot immediately
        if STATUS_JSON.is_file():
            try:
                snap = json.loads(STATUS_JSON.read_text())
                snap["faults"] = [f for f in snap.get("faults", []) if f.get("key") != key]
                history = snap.setdefault("faults_history", [])
                history.append({"key": key, **entry})
                tmp = STATUS_JSON.with_suffix(".tmp")
                tmp.write_text(json.dumps(snap, ensure_ascii=False, indent=1))
                tmp.replace(STATUS_JSON)
            except Exception:
                pass
        self._send_bytes(b'{"ok":true}', "application/json; charset=utf-8")

    def _send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=63400)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[rlforge home] http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
