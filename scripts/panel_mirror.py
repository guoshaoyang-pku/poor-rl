"""Mirror remote RL runs into the local SwanLab AIQ panel + hub samples viewer.

Each cycle:
  1. discover remote runs matching the opus_corr lineage (or use registry list)
  2. ssh (via jump) the remote helper -> metrics / run.json / eval summaries / liveness
  3. rewrite training steps to the global step axis (offset = max(0, K-200) for step{K} runs)
  4. re-import only changed runs into the AIQ panel (swanlab --replace)
  5. sync rollout samples + extract 10 eval display samples per new held-out eval

Usage:
  python3 scripts/panel_mirror.py --once          # single cycle
  python3 scripts/panel_mirror.py --loop 300      # daemon, 300s cadence
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MIRROR_DIR = REPO_ROOT / "artifacts" / "panel_mirror"
SAMPLES_ROOT = REPO_ROOT / "artifacts" / "samples"
IMPORTER = REPO_ROOT / "scripts" / "import_experiments_swanlab.py"
VENV_PY = os.environ.get("RLFORGE_VENV_PY", "/tmp/rlforge-swanlab-preview-venv/bin/python")

JUMP = os.environ.get("RLFORGE_SSH_JUMP", "360-1")
HOST = os.environ.get("RLFORGE_TRAIN_HOST", "10.234.161.3")
REMOTE_HELPER = "/tmp/panel_mirror_remote.py"
LOCAL_HELPER = Path(__file__).with_name("panel_mirror_remote.py")
RUN_GLOB = os.environ.get(
    "RLFORGE_RUN_GLOB",
    "ls /data/home/guoshaoyang/aiq_rl/runs/ | grep -E 'opus_corr.*step[0-9]+' || true")
REGISTRY = REPO_ROOT / "artifacts" / "panel_mirror.json"

# Runs that should never be mirrored into the panel (failed/duplicated attempts
# whose eval glob would collide with the canonical run of the same step tag).
DEFAULT_EXCLUDE = [
    "async_dp_opus_corr_e178_userformula_step10_20261001_failed_flashattn3_20261001",
    "async_dp_opus_corr_e178_userformula_step200_20261001",
]


def load_registry() -> dict:
    if REGISTRY.is_file():
        try:
            return json.loads(REGISTRY.read_text())
        except Exception:
            pass
    return {}


def excluded_runs() -> set[str]:
    reg = load_registry()
    return set(reg.get("exclude", DEFAULT_EXCLUDE))


def ssh_remote(script: str, timeout: int = 240) -> str:
    """Run a command on the training host via the jump host; return stdout."""
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", JUMP,
           f"ssh -o BatchMode=yes -o ConnectTimeout=15 {HOST} '{script}'"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"ssh failed: {r.stderr[-300:]}")
    return r.stdout


def step_offset(run_id: str) -> int:
    m = re.search(r"step(\d+)", run_id)
    return max(0, int(m.group(1)) - 200) if m else 0


def discover_runs() -> list[str]:
    out = ssh_remote(RUN_GLOB)
    excluded = excluded_runs()
    return sorted(line.strip() for line in out.splitlines()
                  if line.strip() and line.strip() not in excluded)


def push_helper() -> None:
    """Ship scripts/panel_mirror_remote.py to the training host (via jump)."""
    for target, dest in ((JUMP, "/tmp/"), (HOST, "/tmp/")):
        r = subprocess.run(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                            str(LOCAL_HELPER), f"{target}:{dest}"],
                           capture_output=True, text=True, timeout=90)
        if r.returncode != 0:
            raise RuntimeError(f"push helper to {target} failed: {r.stderr[-200:]}")


def fetch_bundle(run_id: str) -> dict:
    out = ssh_remote(f"python3 {REMOTE_HELPER} {run_id} {step_offset(run_id)}")
    return json.loads(out)


def apply_offset(bundle: dict) -> dict:
    """Return a cache record with global step axis."""
    off = step_offset(bundle["run_id"])
    metrics = bundle.get("metrics") or {}
    steps = [dict(r, step=int(r.get("step", 0)) + off) for r in metrics.get("steps", [])]
    run_meta = dict(bundle.get("run") or {})
    if bundle.get("running"):
        run_meta["stop_reason"] = "running"
    evals = [{"step": int(e["step_global"]), "overall": e.get("overall", {}),
              "by_task": e.get("by_task", {}), "path": e.get("path")}
             for e in bundle.get("evals", [])]
    return {"run_id": bundle["run_id"], "run": run_meta, "steps": steps,
            "task_bins": metrics.get("task_bins", []), "evals": evals,
            "running": bool(bundle.get("running")),
            "rollout_samples_path": bundle.get("rollout_samples_path"),
            "fetched_at": time.time()}


def cache_path(run_id: str) -> Path:
    return MIRROR_DIR / f"{run_id}.json"


def digest(rec: dict) -> str:
    payload = json.dumps({k: rec[k] for k in ("run", "steps", "task_bins", "evals", "running")},
                         sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


# ---------------- samples sync ----------------

def sync_samples(rec: dict) -> None:
    run_id = rec["run_id"]
    out_dir = SAMPLES_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_file = out_dir / "meta.json"
    if rec.get("run") and not meta_file.is_file():
        meta_file.write_text(json.dumps(rec["run"], ensure_ascii=False, indent=1))

    rsp = rec.get("rollout_samples_path")
    if rsp:
        try:
            data = ssh_remote(f"cat {rsp}")
            if data.strip():
                tmp = out_dir / "train.jsonl.tmp"
                tmp.write_text(data)
                tmp.replace(out_dir / "train.jsonl")
        except Exception as e:
            print(f"[mirror] rollout pull failed {run_id}: {e}", flush=True)

    for ev in rec.get("evals", []):
        sample_file = out_dir / f"eval_ckpt{ev['step']}.jsonl"
        if sample_file.is_file() or not ev.get("path"):
            continue
        try:
            full = json.loads(ssh_remote(f"cat {ev['path']}"))
            records = full.get("records") or []
            rng = random.Random(42 + ev["step"])
            idx = list(range(len(records)))
            rng.shuffle(idx)
            rows = []
            for i in idx[:10]:
                r = records[i]
                s = rng.choice(r.get("samples") or [{}])
                rows.append({
                    "kind": "eval", "question_id": r.get("question_id"),
                    "task": r.get("task"), "family": r.get("family"),
                    "gold": r.get("answer"), "pred": s.get("pred"),
                    "reward": s.get("reward"), "exact": s.get("exact", False),
                    "truncated": s.get("truncated", False),
                    "completion_tokens": s.get("completion_tokens"),
                    "completion": s.get("response_text", ""),
                    "thinking_chars": s.get("visible_thinking_chars"),
                    "strict_format": s.get("strict_answer_format"),
                })
            if rows:
                with open(sample_file, "w") as f:
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[mirror] eval samples {run_id} ckpt{ev['step']}: {len(rows)}", flush=True)
        except Exception as e:
            print(f"[mirror] eval sample pull failed {run_id} ckpt{ev['step']}: {e}", flush=True)


# ---------------- panel import ----------------

def import_runs(records: list[dict]) -> None:
    if not records:
        return
    history = {"aiq": {r["run_id"]: {"run": r["run"], "steps": r["steps"],
                                     "task_bins": r["task_bins"],
                                     "evals": [{k: e[k] for k in ("step", "overall", "by_task")}
                                               for e in r["evals"]]}
                       for r in records}}
    MIRROR_DIR.mkdir(parents=True, exist_ok=True)
    hist_file = MIRROR_DIR / "history.json"
    hist_file.write_text(json.dumps(history))
    r = subprocess.run([VENV_PY, str(IMPORTER), "--history-json", str(hist_file),
                        "--only", "aiq", "--replace"],
                       cwd=REPO_ROOT, env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
                       capture_output=True, text=True)
    names = ", ".join(r["run_id"].split("_userformula_")[-1] for r in records)
    if r.returncode != 0:
        print(f"[mirror] import FAILED ({names}): {r.stderr[-300:]}", flush=True)
    else:
        print(f"[mirror] imported: {names}", flush=True)


def cycle() -> None:
    MIRROR_DIR.mkdir(parents=True, exist_ok=True)
    try:
        run_ids = discover_runs()
    except Exception as e:
        print(f"[mirror] discover failed: {e}", flush=True)
        return
    changed: list[dict] = []
    for run_id in run_ids:
        try:
            rec = apply_offset(fetch_bundle(run_id))
        except Exception as e:
            print(f"[mirror] fetch failed {run_id}: {e}", flush=True)
            cp = cache_path(run_id)
            if cp.is_file():
                rec = json.loads(cp.read_text())
            else:
                continue
        cp = cache_path(run_id)
        old = json.loads(cp.read_text()) if cp.is_file() else None
        if old is None or digest(old) != digest(rec):
            cp.write_text(json.dumps(rec))
            changed.append(rec)
        sync_samples(rec)
    import_runs(changed)
    if not changed:
        print("[mirror] no changes", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", type=int, default=0, metavar="SECONDS")
    args = ap.parse_args()
    try:
        push_helper()
    except Exception as e:
        print(f"[mirror] helper push failed (using remote copy if present): {e}", flush=True)
    if args.loop:
        while True:
            t0 = time.time()
            cycle()
            time.sleep(max(30, args.loop - (time.time() - t0)))
    else:
        cycle()


if __name__ == "__main__":
    main()
