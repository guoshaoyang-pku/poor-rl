"""Controlled kill-line cross-eval: frozen weights x killy in {170, 200}.

Why this is a clean control
---------------------------
`config.pad.killy` (part2/config.py, overridable via SUIKA_KILLY) is read by
the termination predicate in `suika_env._check_game_over` (and by the C settle
scan's gate). The set-transformer token observation (`encode_tokens`) encodes
(type, x, y, vx, vy) normalized by the *play* box only -- it never touches
killy -- so changing killy changes only "when do I die", not "what do I see".
We verify that empirically by hashing a fixed-action probe rollout under both
kill lines (the digest must match).

The wave3c/step3801 checkpoint is bit-identical in both wave4 run dirs (md5
verified), so evaluating that single weight set under killy=170 and killy=200
measures the pure environment effect of moving the death line, with the policy
held fixed.

SUIKA_KILLY must be set *before* `part2/config.py` is imported (it is a module
singleton), which is why this script sets it from --killy at the top and then
imports the evaluator.

Usage (one job per (ckpt, killy) pair; run several with different --out):
  python xeval_killy.py --config configs/w3c_tf_xl.yaml \
      --ckpt runs/.../checkpoints/step3801_env482771285.pt --killy 200 \
      --seeds 0:32 --out /tmp/xeval_fork_k200.json
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True,
                    help="checkpoints/step*.pt (ema is served) or a "
                         "published policy.pt")
    ap.add_argument("--killy", type=int, required=True)
    ap.add_argument("--seeds", default="0:32")
    ap.add_argument("--tempo", action="store_true",
                    help="override cfg tempo (default: use the config)")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage-dir", default=None,
                    help="where to write the staged policy.pt "
                         "(default: a temp dir)")
    return ap.parse_args()


def probe_digest(env, seeds=(0, 1, 2, 3), steps=3, action=64):
    """Hash obs+reward over a fixed-action probe; identical across kill lines."""
    h = hashlib.sha256()
    for s in seeds:
        obs = env.reset(seed=int(s))
        h.update(np.asarray(obs, dtype=np.float32).tobytes())
        for _ in range(steps):
            obs, r, done, _ = env.step(action)
            h.update(np.asarray(obs, dtype=np.float32).tobytes())
            h.update(np.float32(r).tobytes())
            if done:
                h.update(b"DEAD")
                break
    return h.hexdigest()[:16]


def git_rev():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(here), "rev-parse", "--short", "HEAD"],
            text=True).strip()
    except Exception:
        import hashlib as _h
        d = _h.sha256()
        d.update(open(os.path.abspath(__file__), "rb").read())
        return f"sha256:{d.hexdigest()[:12]}"


def main():
    args = parse_args()
    # MUST precede the engine import: part2/config.py reads it at import time.
    os.environ["SUIKA_KILLY"] = str(int(args.killy))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import torch
    torch.set_num_threads(1)
    import yaml
    from evaluator import evaluate_once

    from config import config  # part2 singleton, killy frozen from SUIKA_KILLY
    from env import DQNEnv

    cfg = yaml.safe_load(open(args.config))
    obs_dim = int(cfg["obs_dim"])
    tempo = bool(args.tempo or cfg.get("tempo", False))
    lo, hi = args.seeds.split(":")
    seeds = list(range(int(lo), int(hi)))

    if int(config.pad.killy) != int(args.killy):
        raise SystemExit(f"[xeval] killy not applied: config={config.pad.killy} "
                         f"requested={args.killy}")

    src = os.path.abspath(args.ckpt)
    payload = torch.load(src, map_location="cpu", weights_only=False)
    weights_key = "ema" if "ema" in payload else "state_dict"
    stage = args.stage_dir or tempfile.mkdtemp(prefix="xeval_killy_")
    os.makedirs(stage, exist_ok=True)
    staged = os.path.join(stage, "policy.pt")
    torch.save({"state_dict": payload[weights_key],
                "grad_steps": int(payload.get("grad_steps", -1)),
                "env_steps": int(payload.get("env_steps", -1))}, staged)

    env = DQNEnv(seed=None, K=int(cfg["K"]), max_fruits=int(cfg["max_fruits"]),
                 boundary=bool(cfg.get("boundary", True)), tempo=tempo,
                 obs_format=cfg.get("obs_format", "flat"))
    digest = probe_digest(env)
    del env

    t0 = time.time()
    # per-seed arrays: every job here shares the same seed list, so the
    # (ckpt, killy) cells can be compared paired (common random numbers).
    rec = evaluate_once(cfg, obs_dim, stage, seeds, return_per_seed=True)
    rec.update({"label": args.label or os.path.basename(src),
                "killy": int(args.killy),
                "src_ckpt": src,
                "src_size": os.path.getsize(src),
                "weights": weights_key,
                "tempo": tempo,
                "obs_digest": digest,
                "git": git_rev(),
                "wall_s": round(time.time() - t0, 1)})
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[xeval] {rec['label']:22s} killy={rec['killy']} "
          f"src={weights_key} digest={digest} "
          f"mean={rec['mean']:7.1f} med={rec['median']:7.1f} "
          f"p25={rec['p25']:7.1f} max={rec['max']:6.0f} "
          f"p2000={rec['p2000']:.2f} maxfruit={rec['maxfruit_max']} "
          f"moves={rec['moves_mean']:.0f} n={rec['n']} "
          f"wall={rec['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
