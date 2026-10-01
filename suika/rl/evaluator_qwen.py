"""Periodic fixed-seed greedy evaluator for the Qwen arm.

Fork of evaluator.py — builds the model ONCE (HF ckpt load takes ~1 min) and
only reloads the published state_dict each round. Eval is greedy argmax-Q
(nothink). Env / seeds / output format identical to evaluator.py.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from env import DQNEnv
from model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--obs-dim", type=int, required=True)
    ap.add_argument("--seeds", default="0:16")
    ap.add_argument("--interval-s", type=float, default=900.0)
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(args.config))
    lo, hi = args.seeds.split(":")
    seeds = list(range(int(lo), int(hi)))
    out_path = os.path.join(args.run_dir, "eval.jsonl")
    policy_path = os.path.join(args.run_dir, "policy.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[eval] building model (HF ckpt load, ~1 min)...", flush=True)
    model = build_model(cfg, args.obs_dim).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    K = int(cfg["K"])
    # pre-trigger JIT for eval-shaped batches (shares its GPU with the
    # inference server; a cold autotune stall would time actors out)
    from model_qwen import warmup
    warmup(model, batches=(16,), log=lambda s: print(s, flush=True))

    seen_grad = -1
    while True:
        try:
            mt = os.path.getmtime(policy_path)
        except OSError:
            time.sleep(10)
            continue
        payload = None
        try:
            payload = torch.load(policy_path, map_location="cpu",
                                 weights_only=False)
        except Exception:
            time.sleep(10)
            continue
        gs = int(payload.get("grad_steps", 0))
        if gs == seen_grad:
            time.sleep(args.interval_s / 3)
            continue
        seen_grad = gs
        model.load_state_dict(payload["state_dict"])

        # lockstep batched eval: one forward per move for ALL active seeds
        scores = np.full(len(seeds), np.nan)
        moves_l = np.zeros(len(seeds), dtype=np.int64)
        maxf_l = [0] * len(seeds)
        envs, obs = [], []
        t0 = time.time()
        for si, s in enumerate(seeds):
            e = DQNEnv(seed=None, K=K, max_fruits=int(cfg["max_fruits"]),
                       boundary=bool(cfg.get("boundary", True)),
                       tempo=bool(cfg.get("tempo", False)),
                       obs_format=cfg.get("obs_format", "tokens"))
            obs.append(e.reset(seed=int(s)))
            envs.append(e)
        active = list(range(len(seeds)))
        while active:
            x = torch.from_numpy(np.stack([obs[i] for i in active])).to(device)
            with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                if cfg.get("eval_decode", "qhead") == "pi":
                    q = model.pi_logits(x).float()
                else:
                    q = model.q_values(x).float()
            acts = q.argmax(dim=1).tolist()
            still = []
            for j, i in enumerate(active):
                o, r, done, info = envs[i].step(int(acts[j]))
                moves_l[i] += 1
                if done:
                    scores[i] = float(envs[i].score)
                    fruits = envs[i].env.get_state()["fruits"]
                    maxf_l[i] = max((f["type"] for f in fruits), default=-1)
                else:
                    obs[i] = o
                    still.append(i)
            active = still
        for i in np.where(np.isnan(scores))[0]:   # safety net
            scores[i] = float(envs[i].score)
        rec = {
            "n": len(seeds),
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
            "p25": float(np.percentile(scores, 25)),
            "min": float(scores.min()),
            "max": float(scores.max()),
            "p2000": float((scores >= 2000).mean()),
            "moves_mean": float(np.mean(moves_l)),
            "maxfruit_max": int(max(maxf_l)) if maxf_l else -1,
            "env_steps": int(payload.get("env_steps", 0)),
            "grad_steps": gs,
            "eval_s": round(time.time() - t0, 1),
            "time": time.time(),
        }
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[eval] gs={gs} mean={rec['mean']:.1f} max={rec['max']:.0f} "
              f"({rec['eval_s']}s)", flush=True)
        time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
