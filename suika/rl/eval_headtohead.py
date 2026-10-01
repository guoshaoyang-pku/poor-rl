"""Same-environment head-to-head: cluster DQN vs community DQN vs AlphaZero vs heuristic.

All four agents play the IDENTICAL settle-dynamics engine (suika_dqn.env.DQNEnv,
K=128, new merge rule, max_fruits=80) with identical seeds. The community
(MattJacobs30) DQN is driven by synthesizing ITS flat 209-dim observation from
the shared state, so it can act in this env despite being trained elsewhere.

Usage:
  python eval_headtohead.py --ckpt <policy.pt> --seeds 0:5 \
      --community-zip <suika_dqn_mlp_final.zip> --az-ckpt <latest.pt>
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

from env import DQNEnv
from model import build_model
from encoding import col_to_x

from paths import setup_engine_path
setup_engine_path()
from config import config  # noqa: E402

W = float(config.screen.width)
H = float(config.screen.height)
LEFT, RIGHT = config.pad.left, config.pad.right
WIDTH = RIGHT - LEFT
MAX_TYPE, MAX_RADIUS = 11.0, 150.0


def synth_community_obs(state):
    """Reproduce MattJacobs30 SuikaEnv._get_obs() exactly from a shared state."""
    obs = np.zeros(9 + 50 * 4, dtype=np.float32)
    cur, nxt = state["current"], state["next"]
    cur_r = float(cur.get("radius", config[int(cur["type"]), "radius"]))
    obs[0] = int(cur["type"]) / MAX_TYPE
    obs[1] = cur_r / MAX_RADIUS
    obs[2] = int(nxt["type"]) / MAX_TYPE
    obs[3] = float(config[int(nxt["type"]), "radius"]) / MAX_RADIUS
    obs[4] = LEFT / W
    obs[5] = RIGHT / W
    obs[6] = config.pad.bot / H
    obs[7] = config.pad.killy / H
    fruits = state["fruits"]
    min_y = H
    for f in fruits:
        min_y = min(min_y, f["y"] - f["radius"])
    obs[8] = min_y / H
    fr = sorted(fruits, key=lambda f: (f["y"], f["x"]))[:50]
    i = 9
    for f in fr:
        obs[i] = int(f["type"]) / MAX_TYPE
        obs[i + 1] = f["x"] / W
        obs[i + 2] = f["y"] / H
        obs[i + 3] = f["radius"] / MAX_RADIUS
        i += 4
    return obs


def x_to_col128(x):
    return int(np.clip(int((float(x) - LEFT) / WIDTH * 128), 0, 127))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds", default="0:5")
    ap.add_argument("--community-zip", default=None)
    ap.add_argument("--az-ckpt", default=None)
    ap.add_argument("--az-visits", type=int, default=48)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--agents", default="cluster,community,alphazero,heuristic")
    ap.add_argument("--config", default=None,
                    help="run config yaml; merges over payload cfg "
                         "(arch/hidden/obs_format/tempo/...)")
    a = ap.parse_args()
    lo, hi = a.seeds.split(":")
    seeds = list(range(int(lo), int(hi)))
    wanted = [s for s in a.agents.split(",") if s.strip()]
    ycfg = {}
    if a.config:
        import yaml
        ycfg = yaml.safe_load(open(a.config)) or {}

    agents = {}
    if "cluster" in wanted:
        payload = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        cfg = dict(payload["cfg"])
        cfg.update(ycfg)
        probe = DQNEnv(seed=0, K=int(cfg["K"]),
                       max_fruits=int(cfg["max_fruits"]),
                       boundary=bool(cfg.get("boundary", True)),
                       tempo=bool(cfg.get("tempo", False)),
                       obs_format=str(cfg.get("obs_format", "flat")))
        obs_dim = probe.obs_dim
        net = build_model(cfg, obs_dim)
        net.load_state_dict(payload["state_dict"])
        net.eval()

        def cluster_act(st, obs):
            with torch.no_grad():
                return int(net.q_values(torch.from_numpy(obs).unsqueeze(0))[0]
                           .argmax().item()), None
        agents["cluster"] = cluster_act

    if "community" in wanted and a.community_zip:
        from stable_baselines3 import DQN
        cdqn = DQN.load(a.community_zip)

        def community_act(st, obs):
            o209 = synth_community_obs(st)
            bin_idx, _ = cdqn.predict(o209, deterministic=True)
            act = -1.0 + (int(bin_idx) / 127) * 2.0
            x = LEFT + (act + 1.0) * 0.5 * WIDTH
            return x_to_col128(x), None
        agents["community"] = community_act

    if "alphazero" in wanted and a.az_ckpt:
        sys.path.insert(0, os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))  # platform root -> import viz
        # rl/mcts.py does `from model import SuikaModel`, but sys.modules
        # ['model'] is THIS dir's DuelingQ module -> swap in the rl model.
        import importlib.util as _ilu
        _rl = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "suika", "rl")
        if _rl not in sys.path:
            sys.path.insert(0, _rl)
        _spec = _ilu.spec_from_file_location("suika_rl_model",
                                             os.path.join(_rl, "model.py"))
        _m = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_m)
        sys.modules["model"] = _m
        from viz import agents as VAG
        az = VAG.AlphaZeroAgent(ckpt_path=a.az_ckpt, visits=a.az_visits,
                                device="cpu", seed=0)

        def az_act(st, obs):
            d = az.decide(st)
            return x_to_col128(d["x"]), None
        agents["alphazero"] = az_act

    if "heuristic" in wanted:
        if "viz" not in sys.modules:
            sys.path.insert(0, os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
        from viz import agents as VAG
        he = VAG.HeuristicAgent(num_columns=14, seed=0)

        def he_act(st, obs):
            d = he.decide(st)
            return x_to_col128(d["x"]), None
        agents["heuristic"] = he_act

    print("%-11s %8s %8s %8s %6s %6s" %
          ("agent", "mean", "max", "min", "mfruit", "moves"))
    eK, eMF, eB = 128, 80, True
    eTempo, eFmt = False, "flat"
    if "cluster" in wanted and ycfg:
        eK, eMF = int(ycfg.get("K", eK)), int(ycfg.get("max_fruits", eMF))
        eB = bool(ycfg.get("boundary", eB))
        eTempo = bool(ycfg.get("tempo", eTempo))
        eFmt = str(ycfg.get("obs_format", eFmt))
    results = {}
    for name, act in agents.items():
        scores, moves_l, mf = [], [], []
        t0 = time.time()
        for s in seeds:
            env = DQNEnv(seed=s, K=eK, max_fruits=eMF, boundary=eB,
                         tempo=eTempo, obs_format=eFmt)
            obs = env.reset(seed=s)
            done, moves = False, 0
            while not done and moves < a.max_steps:
                st = env.env.get_state()
                col, _ = act(st, obs)
                obs, r, done, info = env.step(col)
                moves += 1
            scores.append(float(env.score))
            moves_l.append(moves)
            fr = env.env.get_state()["fruits"]
            mf.append(max((f["type"] for f in fr), default=-1))
        sc = np.array(scores)
        results[name] = sc
        print("%-11s %8.1f %8.0f %8.0f %6d %6.0f   (%.0fs)" %
              (name, sc.mean(), sc.max(), sc.min(), max(mf),
               np.mean(moves_l), time.time() - t0))
        sys.stdout.flush()
    print("per-seed scores:")
    for name, sc in results.items():
        print("  %-10s %s" % (name, np.array2string(sc, precision=0)))


if __name__ == "__main__":
    main()
