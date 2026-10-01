"""Cross-evaluation: {our DQN ckpt, mattjacobs30 SB3 weights} x {settle, tempo}.

Our physics + two-watermelon rule everywhere; only tempo/termination and the
policy's own obs/action adapters differ. Multiprocess over (policy, env, seed).
"""
import argparse
import io
import json
import os
import zipfile
from multiprocessing import Pool

import numpy as np

MJ_W, MJ_H, MJ_MAXR, MJ_FRUITS = 1280.0, 720.0, 150.0, 50
MJ_ZIP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "baselines", "SuikaReinforcement", "suika_dqn_mlp_final.zip")


def encode_mj_obs(state):
    """mattjacobs30's exact 209-dim observation from a get_state() dict."""
    obs = np.zeros(9 + MJ_FRUITS * 4, dtype=np.float32)
    cur, nxt = state["current"], state["next"]
    obs[0] = cur["type"] / 11.0
    obs[1] = cur.get("radius", 0.0) / MJ_MAXR
    obs[2] = nxt["type"] / 11.0
    # get_state() does not carry next.radius, but the original baseline
    # derives it from the fruit type when constructing its observation.
    from config import config
    obs[3] = float(config[int(nxt["type"]), "radius"]) / MJ_MAXR
    obs[4] = config.pad.left / MJ_W
    obs[5] = config.pad.right / MJ_W
    obs[6] = config.pad.bot / MJ_H
    obs[7] = config.pad.killy / MJ_H
    fruits = state["fruits"]
    min_y = MJ_H
    for f in fruits:
        min_y = min(min_y, f["y"] - f["radius"])
    obs[8] = min_y / MJ_H
    for i, f in enumerate(sorted(fruits, key=lambda f: (f["y"], f["x"]))[:MJ_FRUITS]):
        b = 9 + i * 4
        obs[b] = f["type"] / 11.0
        obs[b + 1] = f["x"] / MJ_W
        obs[b + 2] = f["y"] / MJ_H
        obs[b + 3] = f["radius"] / MJ_MAXR
    return obs


def mj_action_to_x(a):
    # Match mattjacobs30's discrete mapping: action 0..127 is first mapped
    # to [-1, 1], then to the configured pad interval.
    from config import config
    return config.pad.left + (float(a) / 127.0) * (config.pad.right - config.pad.left)


def load_their_policy():
    import torch
    import torch.nn as nn
    z = zipfile.ZipFile(MJ_ZIP)
    sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu",
                    weights_only=False)
    net = nn.Sequential(
        nn.Linear(209, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 128))
    net[0].weight.data = sd["q_net.q_net.0.weight"]
    net[0].bias.data = sd["q_net.q_net.0.bias"]
    net[2].weight.data = sd["q_net.q_net.2.weight"]
    net[2].bias.data = sd["q_net.q_net.2.bias"]
    net[4].weight.data = sd["q_net.q_net.4.weight"]
    net[4].bias.data = sd["q_net.q_net.4.bias"]
    net.eval()
    return net


def load_our_policy(ckpt_path):
    import torch
    from model import build_model
    cfg = {"K": 128, "max_fruits": 80, "boundary": True, "n_quant": 1,
           "hidden": [1024, 1024], "head_dim": 512}
    model = build_model(cfg, 333)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def make_env(mode, seed):
    from suika_env import SuikaEnv
    from tempo_env import TempoSuikaEnv
    env = TempoSuikaEnv(seed=seed) if mode == "tempo" else SuikaEnv(seed=seed)
    env.reset(seed=seed)
    return env


def play_one(job):
    policy_kind, ckpt_path, env_mode, seed, max_steps = job
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import torch
    torch.set_num_threads(1)
    from encoding import col_to_x, encode_state
    env = make_env(env_mode, seed)
    if policy_kind == "theirs":
        net = load_their_policy()
    else:
        net, _ = load_our_policy(ckpt_path)
    state = env.get_state()
    moves = 0
    done = False
    while not done and moves < max_steps:
        if policy_kind == "theirs":
            obs = encode_mj_obs(state)
            with torch.no_grad():
                a = int(net(torch.from_numpy(obs).unsqueeze(0))[0].argmax())
            x = mj_action_to_x(a)
        else:
            obs = encode_state(state, 128, 80, True)
            with torch.no_grad():
                q = net.q_values(torch.from_numpy(obs).unsqueeze(0))[0]
            x = col_to_x(int(q.argmax()), 128)
        state, _, done, _ = env.step(x)
        moves += 1
    fruits = state["fruits"]
    return {
        "policy": policy_kind, "env": env_mode, "seed": seed,
        "score": float(env.score), "moves": moves,
        "maxfruit": max((f["type"] for f in fruits), default=-1),
        "watermelons": sum(1 for f in fruits if f["type"] == 10),
    }


def summarize(rows, label):
    sc = np.array([r["score"] for r in rows])
    mv = np.array([r["moves"] for r in rows])
    wm = sum(r["watermelons"] for r in rows)
    f10 = sum(1 for r in rows if r["maxfruit"] >= 10)
    return (f"{label:38s} mean={sc.mean():7.1f} med={np.median(sc):7.1f} "
            f"p25={np.percentile(sc,25):7.1f} max={sc.max():6.0f} "
            f"p2000={np.mean(sc>=2000):.2f} fruit10={f10}/{len(rows)} "
            f"wm_total={wm} moves={mv.mean():.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--our-ckpts", nargs="+", required=True,
                    help="label=path pairs")
    ap.add_argument("--seeds", default="0:32")
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default="xeval_results.json")
    args = ap.parse_args()
    lo, hi = args.seeds.split(":")
    seeds = list(range(int(lo), int(hi)))

    jobs = []
    for spec in args.our_ckpts:
        label, path = spec.split("=", 1)
        path = os.path.abspath(path)
        for env_mode in ("settle", "tempo"):
            for s in seeds:
                jobs.append((f"ours:{label}", path, env_mode, s, args.max_steps))
    for env_mode in ("settle", "tempo"):
        for s in seeds:
            jobs.append(("theirs", None, env_mode, s, args.max_steps))

    with Pool(args.procs) as pool:
        rows = pool.map(play_one, jobs)

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)

    groups = {}
    for r in rows:
        groups.setdefault((r["policy"], r["env"]), []).append(r)
    for (pol, env_mode), rs in sorted(groups.items()):
        print(summarize(rs, f"{pol} x {env_mode}"))


if __name__ == "__main__":
    main()
