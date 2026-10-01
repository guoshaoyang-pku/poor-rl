"""Capture a full game of a trained cluster DQN policy as an observatory trace.

Per-frame physics via ``viz.CaptureEnv`` (complete settle process); each
decision records the policy's full Q-values over the K drop columns so the
dashboard can draw the purple "Q bar" (where the policy wants to drop).

Usage:
  python capture_policy.py --ckpt <policy.pt> --config <run.yaml> --seed 0 \
      --name cluster_w3b_mlp_deep_seed0 --out-dir <serve/traces>
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

from paths import setup_engine_path
setup_engine_path()

from model import build_model
from encoding import col_to_x, encode_state, encode_tokens, input_dim, input_dim_tokens
from config import config as _cfg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz import capture as CAP  # noqa: E402
from viz.make_demo import _manifest_entry  # noqa: E402


def pack(f, radius=None):
    t = int(f["type"])
    return {"type": t, "name": _cfg.fruit_names[t],
            "radius": float(radius or f.get("radius") or _cfg[t, "radius"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--frame-stride", type=int, default=2)
    a = ap.parse_args()

    ycfg = yaml.safe_load(open(a.config)) or {}
    payload = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = dict(payload["cfg"])
    cfg.update(ycfg)
    K = int(cfg["K"]); MF = int(cfg["max_fruits"]); B = bool(cfg.get("boundary", True))
    tokens = str(cfg.get("obs_format", "flat")) == "tokens"
    obs_dim = input_dim_tokens() if tokens else input_dim(MF, boundary=B)
    net = build_model(cfg, obs_dim)
    net.load_state_dict(payload["state_dict"])
    net.eval()

    env = CAP.CaptureEnv(seed=a.seed, rule_mode="loop",
                         frame_stride=a.frame_stride, max_frames_per_step=300)
    st = env.reset(seed=a.seed)
    env._snapshot_frame()

    decisions = []
    done = False
    step = 0
    maxf = 0
    t0 = time.time()
    while not done and step < a.max_steps:
        obs = encode_tokens(st) if tokens else encode_state(st, K, MF, B)
        with torch.no_grad():
            q = net.q_values(torch.from_numpy(obs).unsqueeze(0))[0]
        col = int(q.argmax().item())
        x = float(col_to_x(col, K))
        cur, nxt = pack(st["current"]), pack(st["next"])
        f0 = len(env.frames)
        st, r, done, info = env.step(x)
        f1 = len(env.frames)
        fruits = st["fruits"]
        if fruits:
            maxf = max([maxf] + [f["type"] for f in fruits])
        top_y = min((f["y"] - f["radius"] for f in fruits), default=_cfg.pad.bot)
        decisions.append({
            "step": step + 1, "kind": str(cfg.get("name", "cluster-dqn")),
            "col": col, "x": round(x, 1), "current": cur, "next": nxt,
            "reward": float(r), "score": int(st["score"]),
            "fruit_count": len(fruits),
            "max_height": round(float(_cfg.pad.bot - top_y), 1),
            "game_over": bool(done), "frame_start": f0, "frame_end": f1,
            "internals": {"K": K, "col": col,
                          "q": [round(float(v), 2) for v in q.tolist()]},
        })
        step += 1

    meta = {
        "agent": str(cfg.get("name", "cluster-dqn")), "seed": a.seed,
        "rule_mode": "loop", "steps": step, "final_score": int(env.score),
        "max_fruit_type": int(maxf), "max_fruit_name": _cfg.fruit_names[int(maxf)],
        "game_over": bool(done), "num_frames": len(env.frames),
        "frame_stride": a.frame_stride, "fps": int(_cfg.screen.fps),
        "play_area": {"left": _cfg.pad.left, "right": _cfg.pad.right,
                      "top": _cfg.pad.top, "bot": _cfg.pad.bot,
                      "killy": _cfg.pad.killy},
        "world": CAP._world_box(_cfg.pad.left, _cfg.pad.right,
                                _cfg.pad.top, _cfg.pad.bot),
        "fruit_table": CAP._fruit_table(),
        "engine": {"id": "cluster-dqn",
                   "name": "cluster DQN (%s, %s)" % (
                       cfg.get("arch", "mlp"),
                       "tokens" if tokens else "flat"),
                   "physics": "pymunk 6.11.1 (settle)",
                   "screen": {"width": _cfg.screen.width,
                              "height": _cfg.screen.height}},
        "train_env_steps": int(payload.get("env_steps", 0)),
        "arch": str(cfg.get("arch", "mlp")),
    }
    trace = {"meta": meta, "frames": env.frames, "decisions": decisions}
    os.makedirs(a.out_dir, exist_ok=True)
    path = CAP.write_trace_js(trace, a.name, a.out_dir)
    entry = _manifest_entry(a.name, trace)
    entry["file"] = os.path.basename(path)
    entry["agent"] = meta["agent"]
    with open(os.path.join(a.out_dir, "_entry_%s.json" % a.name), "w") as f:
        json.dump(entry, f)
    print("[capture_policy] %s -> steps=%d score=%d maxfruit=%s frames=%d %.0fs"
          % (a.name, step, meta["final_score"], meta["max_fruit_name"],
             len(env.frames), time.time() - t0))


if __name__ == "__main__":
    main()
