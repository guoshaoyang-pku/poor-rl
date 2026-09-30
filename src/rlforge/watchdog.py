#!/usr/bin/env python3
"""Unattended-run watchdog for one async-GRPO/GSPO run on a single node.

Polls the trainer log + run directory every --poll seconds and:
  * runs a held-out eval on each new checkpoint-N that is a multiple of --eval-every
    (on a dedicated eval GPU, via eval_mcq.py, which loads the model itself);
  * copies the best-scoring checkpoint to <run>/keep_best/ so save_total_limit
    rotation can never delete the peak (the arm-A peak was lost exactly this way);
  * stops the run (kill launcher process group -> launcher trap writes stop_reason)
    when any stop rule fires, and records which rule fired in run.json.

Stop rules (all off by setting the threshold empty/0):
  --clip-frac 0.5 --clip-steps 15   clipped_ratio >= X for Y consecutive steps
  --reward-floor -1.0 --reward-win 20  mean reward over last Y steps <= floor
  --stall-min 30                    no new step for N minutes
  --disk-min-gb 50                  free space on the run's filesystem below N GB
  --wall-hours 10.5                 absolute cap

Usage:
  python run_watchdog.py --run runs/async_dpX --trainer-log logs/trainer_dpX.log \
      --launcher-pid 12345 --eval-gpu 7 --eval-data data/eval300.jsonl \
      --eval-every 100 --eval-tokens 16384 --n-samples 2
"""
import argparse
import collections
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

STEP_RE = re.compile(r"\{'loss':|\"loss\":")
METRIC_RE = re.compile(r"'(?P<k>[A-Za-z0-9_./]+)': (?P<v>-?[0-9.eE+]+)")


def read_steps(trainer_log: Path):
    """Parse per-step metric dicts from the trainer log (best effort)."""
    steps = []
    try:
        for line in open(trainer_log, errors="ignore"):
            line = line.strip()
            if not (line.startswith("{") and "'loss'" in line):
                continue
            d = {}
            for m in METRIC_RE.finditer(line):
                try:
                    d[m.group("k")] = float(m.group("v"))
                except ValueError:
                    pass
            if d:
                steps.append(d)
    except OSError:
        pass
    return steps


def checkpoints(run: Path):
    out = {}
    for p in run.glob("checkpoint-*"):
        if p.is_dir():
            try:
                out[int(p.name.split("-")[1])] = p
            except ValueError:
                pass
    return out


def ckpt_step(ckpt: Path):
    try:
        return json.load(open(ckpt / "trainer_state.json"))["global_step"]
    except Exception:
        return None


def stop(launcher_pid: int, run: Path, reason: str):
    print(f"[watchdog] STOP: {reason}", flush=True)
    try:
        m = json.load(open(run / "run.json"))
        m["watchdog_stop"] = reason
        m["watchdog_stop_t"] = time.strftime("%Y-%m-%d %H:%M:%S")
        json.dump(m, open(run / "run.json", "w"), indent=1)
    except Exception as e:
        print(f"[watchdog] could not annotate run.json: {e}", flush=True)
    try:
        os.killpg(os.getpgid(launcher_pid), signal.SIGTERM)
    except Exception as e:
        print(f"[watchdog] killpg failed: {e}; trying SIGKILL", flush=True)
        try:
            os.killpg(os.getpgid(launcher_pid), signal.SIGKILL)
        except Exception:
            pass
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--trainer-log", required=True, type=Path)
    ap.add_argument("--launcher-pid", type=int, required=True)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--eval-gpu", default="7")
    ap.add_argument("--eval-data", required=True,
                    help="held-out eval jsonl, relative to the project root "
                         "(the parent of the run dir's parent) or absolute")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-tokens", type=int, default=16384)
    ap.add_argument("--eval-temp", default="1.0")
    ap.add_argument("--n-samples", type=int, default=2)
    ap.add_argument("--clip-frac", type=float, default=0.5)
    ap.add_argument("--clip-steps", type=int, default=15)
    ap.add_argument("--reward-floor", type=float, default=-1.0)
    ap.add_argument("--reward-win", type=int, default=20)
    ap.add_argument("--stall-min", type=float, default=30)
    ap.add_argument("--disk-min-gb", type=float, default=50)
    ap.add_argument("--wall-hours", type=float, default=10.5)
    ap.add_argument("--eval-script", default=str(Path(__file__).parent / "eval_mcq.py"))
    ap.add_argument("--eval-mode", choices=["spawn", "server"], default="spawn",
                    help="spawn: boot a per-checkpoint vLLM on --eval-gpu (exact ckpt, "
                         "slow). server: probe the live rollout server via "
                         "--eval-server-url (fast, no extra GPU, but scores the live "
                         "policy which may be a few steps past the checkpoint).")
    ap.add_argument("--eval-server-url", default="http://localhost:8000")
    ap.add_argument("--eval-server-model", default=None,
                    help="served model name for eval-mode=server (default: $MODEL)")
    ap.add_argument("--eval-gpu-frac", type=float, default=0.45,
                    help="vLLM memory fraction for eval; the eval GPU also hosts a "
                         "trainer rank (~25 GB), so the eval cannot take 0.9")
    args = ap.parse_args()
    if args.eval_server_model is None:
        import os as _os
        args.eval_server_model = _os.environ.get("MODEL") or _os.environ.get("RLFORGE_BASE_MODEL", "")

    run = args.run
    root = run.parent.parent
    eval_dir = root / "evals" / run.name
    eval_dir.mkdir(parents=True, exist_ok=True)
    hist = eval_dir / "history.jsonl"

    t0 = time.time()
    last_n_steps = 0
    last_step_t = time.time()
    evaluated = set()
    best = {"acc": -1.0, "step": None}
    clip_streak = 0
    reward_window = collections.deque(maxlen=args.reward_win)

    print(f"[watchdog] watching {run} (pid {args.launcher_pid}), evals -> {eval_dir}", flush=True)

    while True:
        time.sleep(args.poll)
        now = time.time()

        # --- wall clock
        if args.wall_hours and now - t0 > args.wall_hours * 3600:
            stop(args.launcher_pid, run, f"wall clock {args.wall_hours} h reached")

        # --- disk
        free_gb = shutil.disk_usage(run).free / 1e9
        if args.disk_min_gb and free_gb < args.disk_min_gb:
            stop(args.launcher_pid, run, f"disk below {args.disk_min_gb} GB ({free_gb:.1f})")

        # --- trainer alive?
        try:
            os.kill(args.launcher_pid, 0)
        except OSError:
            print("[watchdog] launcher gone; exiting", flush=True)
            return

        # --- steps
        steps = read_steps(args.trainer_log)
        if len(steps) > last_n_steps:
            last_n_steps = len(steps)
            last_step_t = now
        elif args.stall_min and now - last_step_t > args.stall_min * 60 and last_n_steps > 0:
            stop(args.launcher_pid, run, f"no new step for {args.stall_min} min")

        for s in steps[len(steps) - 1:]:
            cr = s.get("completions/clipped_ratio")
            if cr is not None:
                clip_streak = clip_streak + 1 if cr >= args.clip_frac else 0
            if "reward" in s:
                reward_window.append(s["reward"])
        if args.clip_steps and clip_streak >= args.clip_steps:
            stop(args.launcher_pid, run,
                 f"clipped_ratio >= {args.clip_frac} for {clip_streak} consecutive steps")
        if (args.reward_win and len(reward_window) == args.reward_win
                and sum(reward_window) / args.reward_win <= args.reward_floor):
            stop(args.launcher_pid, run,
                 f"mean reward over {args.reward_win} steps <= {args.reward_floor}")

        # --- checkpoint evals
        for n, path in sorted(checkpoints(run).items()):
            if n in evaluated or n % args.eval_every != 0:
                continue
            evaluated.add(n)
            step = ckpt_step(path) or n
            out = eval_dir / f"{run.name}_step{step}_T{args.eval_temp}_eval300.json"
            cmd = [
                sys.executable, args.eval_script,
                "--model", str(path), "--data", str(root / args.eval_data),
                "--max-tokens", str(args.eval_tokens),
                "--temperature", args.eval_temp, "--top-p", "1.0",
                "--gpu-frac", str(args.eval_gpu_frac), "--n-samples", str(args.n_samples),
                "--out", str(out),
            ]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.eval_gpu)
            if args.eval_mode == "server":
                # Probe the LIVE policy on the rollout server: no second vLLM boot,
                # no trainer-GPU contention -- but scores the weights the server
                # holds at eval time, which may be a few steps past this checkpoint.
                cmd[cmd.index(str(path))] = args.eval_server_model
                cmd += ["--server-url", args.eval_server_url]
                env.pop("CUDA_VISIBLE_DEVICES")
            print(f"[watchdog] eval ckpt {n} (step {step}) ...", flush=True)
            t_eval = time.time()
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if proc.returncode != 0 or not out.exists():
                line = {"step": step, "error": proc.stderr[-800:]}
                print(f"[watchdog] eval ckpt {n} FAILED: {proc.stderr[-300:]}", flush=True)
            else:
                summary = json.load(open(out))["summary"]
                line = {"step": step, "ckpt": n,
                        "overall": summary["overall"], "by_task": summary["by_task"],
                        "constant_baseline": summary["constant_baseline"],
                        "eval_s": round(time.time() - t_eval)}
                acc = summary["overall"].get("accuracy", 0) or 0
                print(f"[watchdog] ckpt {n}: acc {acc:.3f} "
                      f"(mcq {summary['by_task'].get('mcq', {}).get('accuracy')}, "
                      f"rank {summary['by_task'].get('ranking', {}).get('accuracy')})", flush=True)
                if acc > best["acc"]:
                    best = {"acc": acc, "step": step}
                    keep = run / "keep_best"
                    tmp = run / "keep_best.tmp"
                    if tmp.exists():
                        shutil.rmtree(tmp)
                    shutil.copytree(path, tmp)
                    if keep.exists():
                        shutil.rmtree(keep)
                    tmp.rename(keep)
                    (run / "keep_best.json").write_text(json.dumps(best, indent=1))
                    print(f"[watchdog] new best {acc:.3f} @ step {step} -> keep_best/", flush=True)
            with open(hist, "a") as fh:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
