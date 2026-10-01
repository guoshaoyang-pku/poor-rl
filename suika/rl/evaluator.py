"""Periodic fixed-seed greedy evaluator for a run's published policy."""
import argparse
import json
import os
import time

import numpy as np
import torch

from env import DQNEnv
from model import build_model


def _play(model, env, seed, device, geom=None, max_moves=None):
    obs = env.reset(seed=int(seed), geom=geom)
    done = False
    moves = 0
    acts = bytearray()
    while not done and (max_moves is None or moves < max_moves):
        x = torch.from_numpy(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            a = int(model.q_values(x)[0].argmax().item())
        acts.append(a)
        obs, r, done, info = env.step(a)
        moves += 1
    fruits = env.env.get_state()["fruits"]
    return (float(env.score), moves,
            max((f["type"] for f in fruits), default=-1), not done,
            acts.hex())


def evaluate_grid(cfg, obs_dim, run_dir, seeds, grid, max_moves=None,
                  action_sink=None):
    """Greedy eval over a list of fixed (width, height) boards.

    Same seed list on every board (common random numbers), so boards and
    policies can be compared paired. `max_moves` bounds one episode so a
    near-endless policy cannot stall the evaluator; capped episodes are
    counted as `censored` and their score is a lower bound. If `action_sink`
    is a list, one replayable record (seed, board, actions) per episode is
    appended to it.
    """
    policy_path = os.path.join(run_dir, "policy.pt")
    payload = torch.load(policy_path, map_location="cpu", weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg, obs_dim).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    env = DQNEnv(seed=None, K=int(cfg["K"]), max_fruits=int(cfg["max_fruits"]),
                 boundary=bool(cfg.get("boundary", True)),
                 tempo=bool(cfg.get("tempo", False)),
                 obs_format=cfg.get("obs_format", "flat"),
                 geo_dim=int(cfg.get("geo_dim", 0)))
    pts, all_means, pooled = {}, [], []
    for (w, h) in grid:
        sc, mv, mf, cen = [], [], [], 0
        for s in seeds:
            a, b, c, d, hx = _play(model, env, s, device,
                                   geom=(int(w), int(h)), max_moves=max_moves)
            sc.append(a); mv.append(b); mf.append(c); cen += int(d)
            if action_sink is not None:
                action_sink.append({
                    "seed": int(s), "w": int(w), "h": int(h),
                    "score": a, "moves": b, "censored": bool(d),
                    "actions": hx})
        sc = np.array(sc)
        pooled.extend(sc.tolist())
        pts[f"{int(w)}x{int(h)}"] = {
            "mean": float(sc.mean()), "median": float(np.median(sc)),
            "max": float(sc.max()), "moves_mean": float(np.mean(mv)),
            "maxfruit_max": int(max(mf)), "censored": cen, "n": len(sc)}
        all_means.append(float(sc.mean()))
    return {
        "n": len(seeds) * len(grid),
        "mean": float(np.mean(all_means)),
        "median": float(np.median(pooled)),
        "p25": float(np.percentile(pooled, 25)),
        "p2000": float((np.array(pooled) >= 2000).mean()),
        "maxfruit_max": int(max(v["maxfruit_max"] for v in pts.values())),
        "min": float(np.min(all_means)),
        "max": float(np.max([v["max"] for v in pts.values()])),
        "moves_mean": float(np.mean([v["moves_mean"] for v in pts.values()])),
        "censored": int(sum(v["censored"] for v in pts.values())),
        "grid": pts,
        "env_steps": int(payload.get("env_steps", 0)),
        "grad_steps": int(payload.get("grad_steps", 0)),
    }


def evaluate_once(cfg, obs_dim, run_dir, seeds, eps=0.0, return_per_seed=False):
    policy_path = os.path.join(run_dir, "policy.pt")
    payload = torch.load(policy_path, map_location="cpu", weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg, obs_dim).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    K = int(cfg["K"])
    env = DQNEnv(seed=None, K=K, max_fruits=int(cfg["max_fruits"]),
                 boundary=bool(cfg.get("boundary", True)),
                 tempo=bool(cfg.get("tempo", False)),
                 obs_format=cfg.get("obs_format", "flat"),
                 geometry=cfg.get("geometry"),
                 geo_dim=int(cfg.get("geo_dim", 0)))
    scores, moves_l, maxf_l = [], [], []
    for s in seeds:
        obs = env.reset(seed=int(s))
        done = False
        moves = 0
        while not done:
            x = torch.from_numpy(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                a = int(model.q_values(x)[0].argmax().item())
            obs, r, done, info = env.step(a)
            moves += 1
        scores.append(float(env.score))
        moves_l.append(moves)
        fruits = env.env.get_state()["fruits"]
        maxf_l.append(max((f["type"] for f in fruits), default=-1))
    scores = np.array(scores)
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
        "grad_steps": int(payload.get("grad_steps", 0)),
    }
    if return_per_seed:
        # per-seed arrays let a caller pair runs that share a seed list
        # (common random numbers) instead of comparing unpaired means.
        rec["scores"] = [float(s) for s in scores]
        rec["moves"] = [int(m) for m in moves_l]
        rec["maxfruits"] = [int(m) for m in maxf_l]
        rec["seeds"] = [int(s) for s in seeds]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--obs-dim", type=int, required=True)
    ap.add_argument("--seeds", default="0:16")
    ap.add_argument("--interval-s", type=float, default=300.0)
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(args.config))
    lo, hi = args.seeds.split(":")
    seeds = list(range(int(lo), int(hi)))
    out_path = os.path.join(args.run_dir, "eval.jsonl")
    policy_path = os.path.join(args.run_dir, "policy.pt")
    best_path = os.path.join(args.run_dir, "policy_best.pt")
    best_meta = os.path.join(args.run_dir, "policy_best.json")
    best_mean = -1e18
    seen_grad = -1
    # D15b: best tracking must survive an evaluator restart (wave6 00:19
    # incident: restart reset best_mean and overwrote the peak snapshot with a
    # worse policy).
    if os.path.exists(best_meta):
        try:
            best_mean = float(json.load(open(best_meta))["mean"])
        except Exception:
            pass
    while True:
        try:
            mt = os.path.getmtime(policy_path)
        except OSError:
            time.sleep(5)
            continue
        payload = None
        try:
            payload = torch.load(policy_path, map_location="cpu",
                                 weights_only=False)
        except Exception:
            time.sleep(5)
            continue
        gs = int(payload.get("grad_steps", 0))
        if gs == seen_grad:
            time.sleep(args.interval_s / 3)
            continue
        seen_grad = gs
        if cfg.get("eval_grid"):
            gs = seeds[:int(cfg.get("eval_grid_seeds", 8))]
            sink = []
            rec = evaluate_grid(cfg, args.obs_dim, args.run_dir, gs,
                                cfg["eval_grid"],
                                max_moves=cfg.get("eval_max_moves"),
                                action_sink=sink)
            with open(os.path.join(args.run_dir, "eval_actions.jsonl"),
                      "a") as f:
                for e in sink:
                    e.update(grad_steps=rec["grad_steps"],
                             env_steps=rec["env_steps"])
                    f.write(json.dumps(e) + "\n")
        else:
            rec = evaluate_once(cfg, args.obs_dim, args.run_dir, seeds)
        rec["time"] = time.time()
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        # D15: never lose the peak policy to ckpt rotation again -- snapshot
        # the published policy whenever the eval mean improves.
        if rec["mean"] > best_mean:
            best_mean = rec["mean"]
            import shutil
            shutil.copyfile(policy_path, best_path + ".tmp")
            os.replace(best_path + ".tmp", best_path)
            with open(best_meta, "w") as f:
                json.dump({"mean": best_mean,
                           "grad_steps": rec["grad_steps"],
                           "env_steps": rec["env_steps"],
                           "time": rec["time"]}, f)
        print(f"[eval] gs={gs} mean={rec['mean']:.1f} max={rec['max']:.0f}"
              + (f" min_pt={rec['min']:.0f} censored={rec['censored']}"
                 if "grid" in rec else ""), flush=True)
        time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
