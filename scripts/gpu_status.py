"""Collect per-cluster GPU/memory status and maintain sticky fault flags.

Writes rlforge/artifacts/gpu_status.json (current snapshot + active faults +
fault history). Faults are sticky: they stay visible until the underlying
error clears on a later probe, or until manually cleared via the home API
(cleared faults re-raise on the next probe if the condition persists).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
SNAPSHOT = ARTIFACTS / "gpu_status.json"
FAULTS = ARTIFACTS / "gpu_faults.json"

CLUSTERS = {
    "360-1": {"expected_gpus": 8, "note": "GPU01"},
    "360-2": {"expected_gpus": 8, "note": "GPU02"},
    "ophis-gpu": {"expected_gpus": 8, "note": "Ophis H100"},
    "a100_perm": {"expected_gpus": 8, "note": "A100 perm（私有）", "private": True},
    "a100_t1": {"expected_gpus": 8, "note": "A100 t1（私有）", "private": True},
    "a100_t1_2": {"expected_gpus": 8, "note": "A100 t1-2（私有）", "private": True},
    "a100_t1_3": {"expected_gpus": 6, "note": "A100 t1-3（私有，6 卡）", "private": True},
}

SELF_USERS = {"guoshaoyang"}
OCCUPIED_MEM_MIB = 2048

PROBE = r'''
set -o pipefail
echo "---GPU---"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo "---APPS---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
echo "---PS---"
ps -eo pid=,user=,comm= 2>/dev/null
echo "---ECC---"
nvidia-smi --query-gpu=index,ecc.errors.uncorrected.volatile.total --format=csv,noheader 2>/dev/null || true
echo "---MEM---"
free -m | awk '/^Mem:/{print $3" "$2}'
'''


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def run_probe(host: str, timeout: int = 30, private: bool = False) -> dict:
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout - 10}",
         host, "bash", "-s"],
        input=PROBE, capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or "ssh failed").strip()[:200])
    sections: dict[str, list[str]] = {}
    current = None
    for line in proc.stdout.splitlines():
        if line.startswith("---") and line.endswith("---"):
            current = line.strip("-")
            sections[current] = []
        elif current is not None and line.strip():
            sections[current].append(line.strip())

    gpus: dict[int, dict] = {}
    uuid_to_index: dict[str, int] = {}
    for line in sections.get("GPU", []):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        idx = int(parts[0])
        uuid_to_index[parts[1]] = idx
        gpus[idx] = {
            "index": idx, "name": parts[2],
            "mem_used_mib": int(parts[3]), "mem_total_mib": int(parts[4]),
            "util_pct": int(parts[5]), "procs": [],
        }
    owners: dict[str, str] = {}
    for line in sections.get("PS", []):
        parts = line.split(None, 2)
        if len(parts) >= 2:
            owners[parts[0]] = parts[1]
    for line in sections.get("APPS", []):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx = uuid_to_index.get(parts[0])
        if idx is None or idx not in gpus:
            continue
        user = owners.get(parts[1], "?")
        foreign = False if private else user not in SELF_USERS
        gpus[idx]["procs"].append({
            "pid": parts[1], "name": parts[2],
            "mem_mib": int(parts[3].split()[0]) if parts[3].split() else 0,
            "user": user,
            # "?" = PID not visible in our container (host/其他容器进程)；私有集群一律视为自己
            "foreign": foreign,
            "owner_unknown": user == "?",
        })
    for line in sections.get("ECC", []):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            idx = int(parts[0])
            if idx in gpus:
                gpus[idx]["ecc_uncorrected"] = int(parts[1])
    ram = None
    mem_lines = sections.get("MEM", [])
    if mem_lines:
        used, total = mem_lines[0].split()[:2]
        ram = {"used_mb": int(used), "total_mb": int(total)}

    for gpu in gpus.values():
        gpu["foreign_busy"] = any(p["foreign"] for p in gpu["procs"])
        # a GPU hosting compute processes is working even when the instant
        # util sample is low: physics-bound pipelines idle the GPU between
        # batches and tiny MLP forwards barely register on util.
        gpu["busy"] = (bool(gpu["procs"])
                       or gpu["mem_used_mib"] >= OCCUPIED_MEM_MIB
                       or gpu["util_pct"] >= 10)
    return {"gpus": [gpus[i] for i in sorted(gpus)], "ram": ram}


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def save_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    tmp.replace(path)


def update_faults(faults: dict, key: str, detail: str, ts: str, active_now: bool) -> None:
    entry = faults.setdefault(key, {
        "first_seen": ts, "last_seen": ts, "detail": detail,
        "cleared_at": None, "cleared_by": None, "occurrences": 0,
    })
    if active_now:
        if entry.get("cleared_at"):
            entry["cleared_at"] = None
            entry["cleared_by"] = None
            entry["first_seen"] = ts
            entry["occurrences"] = entry.get("occurrences", 0) + 1
        entry["last_seen"] = ts
        entry["detail"] = detail
    else:
        if not entry.get("cleared_at"):
            entry["cleared_at"] = ts
            entry["cleared_by"] = "auto-recovered"


def collect() -> dict:
    ts = now_iso()
    faults = load_json(FAULTS, {})
    clusters = {}
    for host, meta in CLUSTERS.items():
        node_key = f"{host}:node"
        try:
            result = run_probe(host, private=bool(meta.get("private")))
            clusters[host] = {"ok": True, "error": None, "note": meta["note"], **result}
            update_faults(faults, node_key, "", ts, active_now=False)
            seen = {g["index"] for g in result["gpus"]}
            expected = meta["expected_gpus"]
            for idx in range(expected):
                card_key = f"{host}:gpu{idx}"
                if idx not in seen:
                    update_faults(faults, card_key, f"GPU{idx} 未在 nvidia-smi 中出现（掉卡）", ts, True)
                    continue
                gpu = result["gpus"][[g["index"] for g in result["gpus"]].index(idx)]
                ecc = gpu.get("ecc_uncorrected", 0)
                if ecc and ecc > 0:
                    update_faults(faults, card_key, f"GPU{idx} ECC 不可纠正错误累计 {ecc}", ts, True)
                else:
                    update_faults(faults, card_key, "", ts, active_now=False)
        except Exception as exc:  # noqa: BLE001 - any probe failure is a node fault
            clusters[host] = {"ok": False, "error": str(exc)[:300], "note": meta["note"],
                              "gpus": [], "ram": None}
            update_faults(faults, node_key, f"节点不可达或探测失败: {str(exc)[:200]}", ts, True)
    active = [{"key": k, **v} for k, v in sorted(faults.items()) if not v.get("cleared_at")]
    history = [{"key": k, **v} for k, v in sorted(faults.items()) if v.get("cleared_at")]
    snapshot = {"updated_at": ts, "clusters": clusters,
                "faults": active, "faults_history": history[-50:]}
    save_json(SNAPSHOT, snapshot)
    save_json(FAULTS, faults)
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", type=int, default=0, help="poll interval seconds")
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if args.once or not args.loop:
        snapshot = collect()
        n_active = len(snapshot["faults"])
        print(f"[gpu_status] {snapshot['updated_at']} clusters={len(snapshot['clusters'])} active_faults={n_active}")
        return
    while True:
        try:
            snapshot = collect()
            print(f"[gpu_status] {snapshot['updated_at']} active_faults={len(snapshot['faults'])}", flush=True)
        except Exception as exc:  # noqa: BLE001 - keep the daemon alive
            print(f"[gpu_status] collect failed: {exc}", flush=True)
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
