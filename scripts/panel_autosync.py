#!/usr/bin/env python3
"""Auto-sync AIQ RL trials from 360-2 into the local SwanLab panel.

Polls 360-2 every POLL_S seconds for runs matching WATCH_GLOB that have a
log_history.json. For any run whose step count changed since last import
(or that is new), pulls the panel export + eval JSONs, merges them into the
master history file, and re-imports with --replace.

State: /tmp/panel_import/autosync_state.json
Master history: /tmp/panel_import/history.json  (keyed by TECH run id;
user-facing merged/renamed entries live in history_merged.json and are NOT
touched here.)
"""
import json, os, subprocess, sys, time, glob, re

SSH = ["ssh", "-o", "ConnectTimeout=20", "-o", "BatchMode=yes", "360-2"]
RUNS_GLOB = "/data/home/guoshaoyang/aiq_rl/runs/async_dp_*"
ROOT = "/data/home/guoshaoyang/aiq_rl"
IMPORT_DIR = "/tmp/panel_import"
HISTORY = os.path.join(IMPORT_DIR, "history.json")
STATE = os.path.join(IMPORT_DIR, "autosync_state.json")
RLFORGE = "/Users/guoshaoyang/Desktop/workdir/rlforge"
VENV_PY = "/tmp/rlforge-swanlab-preview-venv/bin/python"
POLL_S = 300

def sh(args, timeout=120, **kw):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, **kw)

def remote(cmd, timeout=120):
    r = sh(SSH + [cmd], timeout=timeout)
    return r.stdout if r.returncode == 0 else None

def list_runs():
    out = remote(f"ls -d {RUNS_GLOB} 2>/dev/null")
    if not out:
        return []
    return [p.strip().split("/")[-1] for p in out.splitlines() if p.strip()]

def run_step_count(name):
    out = remote(
        "python3 -c \"import json;d=json.load(open('%s/runs/%s/log_history.json'));"
        "s=[x for x in d if x.get('step')];print(len(s),s[-1]['step'])\"" % (ROOT, name))
    if not out:
        return None
    try:
        n, last = out.split()
        return int(n), int(last)
    except ValueError:
        return None

def pull_export(name):
    remote(f"cd {ROOT} && ./venv/bin/python report_watch.py --run runs/{name} "
           f"--trainer-log logs/trainer_dp_{name[len('async_dp_'):]}.log "
           f"--out /tmp/panel_export/{name} --once >/dev/null 2>&1")
    dest = os.path.join(IMPORT_DIR, "exports", name)
    os.makedirs(dest, exist_ok=True)
    r = sh(["scp", "-q", f"360-2:/tmp/panel_export/{name}/metrics.json",
            f"360-2:{ROOT}/runs/{name}/run.json", dest], timeout=180)
    return (r.returncode == 0 and
            os.path.exists(os.path.join(dest, "metrics.json")) and
            os.path.exists(os.path.join(dest, "run.json")))

def pull_evals():
    """Pull any new eval JSONs under the shared eval dir."""
    os.makedirs("/tmp/ladder_eval", exist_ok=True)
    out = remote("ls /data/shared/guoshaoyang/aiq_rl_store/evals/*.json 2>/dev/null")
    if not out:
        return
    for p in out.splitlines():
        p = p.strip()
        fn = os.path.basename(p)
        local = os.path.join("/tmp/ladder_eval", fn)
        if not os.path.exists(local):
            sh(["scp", "-q", f"360-2:{p}", local], timeout=300)

def pull_eval_index():
    """Pull eval index + any new eval JSONs; returns {run_name: [ {step,file} ]}."""
    out = remote("cat /data/shared/guoshaoyang/aiq_rl_store/evals/index.jsonl 2>/dev/null")
    mapping = {}
    if not out:
        return mapping
    os.makedirs("/tmp/ladder_eval", exist_ok=True)
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        local = os.path.join("/tmp/ladder_eval", e["file"])
        if not os.path.exists(local):
            r = sh(["scp", "-q", f"360-2:/data/shared/guoshaoyang/aiq_rl_store/evals/{e['file']}",
                    local], timeout=300)
            if r.returncode != 0:
                continue
        mapping.setdefault(e["run"], []).append({"step": e["step"], "file": e["file"]})
    return mapping

def attach_evals(history, mapping):
    """Attach eval summaries from /tmp/ladder_eval to history records; True if changed."""
    changed = False
    for name, entries in mapping.items():
        rec = history["aiq"].get(name)
        if rec is None:
            continue
        have = {e["step"] for e in rec.get("evals", [])}
        for ent in entries:
            if ent["step"] in have:
                continue
            path = os.path.join("/tmp/ladder_eval", ent["file"])
            try:
                d = json.load(open(path))
            except Exception:
                continue
            rec.setdefault("evals", []).append(
                {"step": ent["step"], "overall": d["overall"], "by_task": d["by_task"]})
            rec["evals"].sort(key=lambda x: x["step"])
            changed = True
            print(f"[autosync] eval attached: {name} step {ent['step']}", flush=True)
    return changed
MERGE_LINES = [
    ("Opus e178 → RL（4+4 大batch，global 0-600 步）",
     [("async_dp_opus_corr_e178_userformula_step200_4plus4cps256_20261001", 0),
      ("async_dp_opus_corr_e178_userformula_step400_from_ckpt200_4plus4cps256_20261002", 200),
      ("async_dp_opus_corr_e178_userformula_step600_from_ckpt400_4plus4cps256_20261002", 400)],
     "技术 run id 三段续训合并；起点 Opus corrected SFT ckpt-178；GSPO ±0.1%，CPS=256，lr 2e-6，fp8 KV；峰顶 g400（79.8%）。"),
    ("多教师 e813 → RL（4+4 大batch，global 0-400 步）",
     [("async_dp_allteachers_e813_userformula_step200_4plus4cps256_20261002", 0),
      ("async_dp_allteachers_e813_userformula_step400_from_rl200_4plus4cps256_20261002", 200)],
     "技术 run id 两段续训合并；起点多教师 SFT e813；同 setting；峰顶 g350（81.0%/MCQ 71.5%，当前最优）。"),
]
SEGMENT_IDS = {sid for _, segs, _ in MERGE_LINES for sid, _ in segs}
SEGMENT_IDS.add("async_dp_opus_corr_e178_userformula_step10_20261001")

NAME_MAP = {
    "async_dp_opus_corr_e178_userformula_step10_20261001":
        ("Opus e178 → RL 10步小batch（已停损）",
         "7 卡 3+4 小 batch canary；ranking 漏字母恶化触发停损。"),
    "async_dp_allteachers_e813_rl200_eps2pct_step100_4plus4cps256_20261002":
        ("消融 · GSPO eps ±2%（at-e813 RL200 起点，100步）",
         "宽信任域消融：held-out −9.7pp vs ±0.1%，漂移/verbose 失败，否决。"),
    "async_dp_allteachers_e813_rl200_eps05pct_step100_4plus4cps256_20261002":
        ("消融 · GSPO eps ±0.5%（at-e813 RL200 起点，100步）",
         "中点消融：held-out −9.5pp vs ±0.1%，同样显著更差；结论：±0.1% 紧 clip 最优。"),
    "async_dp_ablate_lr4e6_eps01pct_step100_20261002":
        ("消融 · lr 4e-6（eps ±0.1%，at-e813 RL200 起点，100步）",
         "学习率加倍消融（eps 保持 ±0.1%）。"),
    "async_dp_ablate_cps512_eps01pct_step100_20261002":
        ("消融 · CPS 512 大batch（eps ±0.1%，at-e813 RL200 起点，100步）",
         "批量加倍消融（全局 2048 completions/step）。"),
    "async_dp_ablate_eps02pct_step100_20261003":
        ("消融 · GSPO eps ±0.2%（at-e813 RL200 起点，100步）",
         "剂量曲线补点：0.1%-0.5% 之间的中间档。"),
    "async_dp_ablate_eps005pct_step100_20261003":
        ("消融 · GSPO eps ±0.05%（at-e813 RL200 起点，100步）",
         "剂量曲线补点：比主线更紧的信任域。"),
    "async_dp_ablate_dynlo20_hi001_step100_20261003":
        ("消融 · 动态低侧clip（目标20%裁剪率 / high +0.1%，100步）",
         "低侧按批内 log-ratio 20% 分位数 EMA 自适应（用前值平均），高侧固定 +0.1%。"),
    "async_dp_v3full": ("早期 GRPO · v3 全池（历史，已 kill）", "早期配方验证，分数与现契约不可比。"),
    "async_dp_flip450": ("早期 GRPO · dataflip 450（历史，已 kill）", "早期配方验证。"),
    "async_dp_pool2134cap8k": ("早期 GRPO · pool2134 cap8k（历史，已停）", "早期配方验证。"),
}

def build_view(aiq_history):
    """tech-id history -> user-facing merged/renamed view."""
    view = {}
    for disp, segs, note in MERGE_LINES:
        steps, task_bins, evals, run_meta = [], [], [], None
        for sid, off in segs:
            rec = aiq_history.get(sid)
            if not rec:
                run_meta = None
                break
            run_meta = run_meta or dict(rec["run"])
            steps += [dict(s, step=s["step"] + off) for s in rec["steps"]]
            task_bins += rec.get("task_bins", [])
            evals += [dict(e) for e in rec.get("evals", [])]
        if run_meta is None:
            continue
        run_meta["stop_reason"] = "completed"
        run_meta["note"] = note
        steps.sort(key=lambda x: x["step"])
        evals.sort(key=lambda x: x["step"])
        view[disp] = {"run": run_meta, "steps": steps, "task_bins": task_bins, "evals": evals}
    for name, rec in aiq_history.items():
        if name in SEGMENT_IDS and name not in NAME_MAP:
            continue
        disp, note = NAME_MAP.get(name, (name, ""))
        meta = dict(rec["run"])
        if note:
            meta["note"] = note
        view[disp] = {"run": meta, "steps": rec["steps"],
                      "task_bins": rec.get("task_bins", []), "evals": rec.get("evals", [])}
    return view

def main():
    os.makedirs(os.path.join(IMPORT_DIR, "exports"), exist_ok=True)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    history = json.load(open(HISTORY)) if os.path.exists(HISTORY) else {"aiq": {}}
    changed = attach_evals(history, pull_eval_index())
    for name in list_runs():
        cnt = run_step_count(name)
        if not cnt:
            continue
        prev = state.get(name, {}).get("steps")
        if prev == cnt[0]:
            continue
        if not pull_export(name):
            print(f"[autosync] export failed for {name}", flush=True)
            continue
        dest = os.path.join(IMPORT_DIR, "exports", name)
        m = json.load(open(os.path.join(dest, "metrics.json")))
        r = json.load(open(os.path.join(dest, "run.json")))
        old = history["aiq"].get(name, {})
        history["aiq"][name] = {
            "run": r, "steps": m["steps"],
            "task_bins": m.get("task_bins", []),
            "evals": old.get("evals", []),
        }
        state[name] = {"steps": cnt[0], "last_step": cnt[1]}
        changed = True
        print(f"[autosync] refreshed {name} ({cnt[0]} steps)", flush=True)
    if changed:
        json.dump(history, open(HISTORY, "w"), ensure_ascii=False)
        json.dump(state, open(STATE, "w"))
        view = {"aiq": build_view(history["aiq"])}
        view_path = os.path.join(IMPORT_DIR, "history_view.json")
        json.dump(view, open(view_path, "w"), ensure_ascii=False)
        env = dict(os.environ, PYTHONPATH=f"{RLFORGE}/src:{RLFORGE}/scripts")
        r = sh([VENV_PY, "scripts/import_experiments_swanlab.py",
                "--history-json", view_path, "--only", "aiq", "--replace"],
               timeout=600, cwd=RLFORGE, env=env)
        tail = (r.stdout or r.stderr).strip().splitlines()[-3:]
        print("[autosync] import:", tail, flush=True)
        # remove raw segment entries so only the merged/renamed view remains
        cleanup = sh([VENV_PY, "-c", CLEANUP_SNIPPET], timeout=300, cwd=RLFORGE, env=env)
        print("[autosync] cleanup:", cleanup.stdout.strip().splitlines()[-1:]
              if cleanup.stdout.strip() else cleanup.returncode, flush=True)

CLEANUP_SNIPPET = (
    "import sys;sys.path.insert(0,'scripts');from pathlib import Path;"
    "from import_experiments_swanlab import delete_run;"
    "from panel_autosync import SEGMENT_IDS;"
    "ld=Path('artifacts/swanlab/AIQ/swanlog');"
    "[delete_run(ld,'AIQ',n) for n in sorted(SEGMENT_IDS)];"
    "print('segments cleaned')"
)

if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as e:
            print(f"[autosync] error: {e}", flush=True)
        time.sleep(POLL_S)
