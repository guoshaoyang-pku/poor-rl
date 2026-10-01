"""Vision-arm BC data collector: teacher (w3b_mlp_deep MLP) rollouts with
synchronised board images.

Same n-step assembly and shard semantics as bc_collector.py (recipe
alignment), but each transition additionally stores a JPEG of the rendered
board at decision time (pre-action, overlays via vis_render). The token obs
is dropped; flat-333 is kept (f16) for teacher input provenance and hybrid
probes.

Shard fields: img (uint8 1-D concat of JPEG bytes) + img_off (int64 N+1)
              act rew done gam (identical semantics to actor.py)
              + qteach[128] f16 (teacher Q at obs) + argmax u8
              + flat[333] f16 + geom[2] u16 (width, killy)
NOTE: no nobs / next image - the v3.1 listwise BC recipe needs only s.
"""
import argparse
import collections
import json
import os
import time

import numpy as np
import torch

from paths import setup_engine_path
setup_engine_path()

from env import DQNEnv  # noqa: E402
from model import build_model  # noqa: E402
from vis_render import render_jpeg  # noqa: E402

TEACHER_CFG = {
    "arch": "mlp", "hidden": [2048, 2048, 2048], "head_dim": 1024,
    "K": 128, "double": True, "n_quant": 1,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--teacher", required=True)   # policy.pt of mlp_deep
    ap.add_argument("--actor", type=int, required=True)
    ap.add_argument("--actors", type=int, default=48)
    ap.add_argument("--max-transitions", type=int, default=600000)
    ap.add_argument("--eps-min", type=float, default=0.01)
    ap.add_argument("--eps-max", type=float, default=0.10)
    ap.add_argument("--n-step", type=int, default=3)
    ap.add_argument("--killy", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=770001)
    ap.add_argument("--img-w", type=int, default=288)
    ap.add_argument("--img-h", type=int, default=416)
    ap.add_argument("--jpeg-quality", type=int, default=85)
    args = ap.parse_args()

    torch.set_num_threads(1)
    K, n_step, gamma = 128, args.n_step, 1.0
    shard_size = 4096
    rng = np.random.default_rng(args.seed_base + args.actor)
    eps = (args.eps_max * (args.eps_min / args.eps_max)
           ** (args.actor / max(args.actors - 1, 1)))
    out_size = (args.img_w, args.img_h)

    teacher = build_model(TEACHER_CFG, 333)
    payload = torch.load(args.teacher, map_location="cpu", weights_only=False)
    teacher.load_state_dict(payload["state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    env = DQNEnv(seed=None, K=K, max_fruits=80, boundary=True,
                 reward_scale=1.0, tempo=False, obs_format="flat",
                 geometry={"killy": args.killy} if args.killy else None)
    inbox = os.path.join(args.run_dir, "inbox")
    os.makedirs(inbox, exist_ok=True)
    ep_log = open(os.path.join(args.run_dir,
                               f"episodes_a{args.actor}.jsonl"), "a")

    bufs = {k: [] for k in
            ("img", "act", "rew", "done", "gam", "qteach", "argmax",
             "flat", "geom")}

    def flush():
        if not bufs["act"]:
            return
        img_bytes = b"".join(bufs["img"])
        off = np.zeros(len(bufs["img"]) + 1, dtype=np.int64)
        np.cumsum([len(b) for b in bufs["img"]], out=off[1:])
        fn = f"a{args.actor}_{time.time_ns()}.npz"
        tmp = os.path.join(inbox, fn + ".tmp.npz")
        np.savez(tmp,
                 img=np.frombuffer(img_bytes, dtype=np.uint8),
                 img_off=off,
                 act=np.asarray(bufs["act"], dtype=np.int64),
                 rew=np.asarray(bufs["rew"], dtype=np.float32),
                 done=np.asarray(bufs["done"], dtype=np.float32),
                 gam=np.asarray(bufs["gam"], dtype=np.float32),
                 qteach=np.asarray(bufs["qteach"], dtype=np.float16),
                 argmax=np.asarray(bufs["argmax"], dtype=np.uint8),
                 flat=np.asarray(bufs["flat"], dtype=np.float16),
                 geom=np.asarray(bufs["geom"], dtype=np.uint16))
        os.replace(tmp, os.path.join(inbox, fn))
        for v in bufs.values():
            v.clear()

    def teacher_q(flat):
        with torch.no_grad():
            x = torch.from_numpy(flat).unsqueeze(0)
            q = teacher.q_values(x)[0]
        return q.numpy().astype(np.float16), int(q.argmax().item())

    # n-step assembly identical to bc_collector.py (fixed semantics)
    pend = collections.deque()
    episode, total, t0 = 0, 0, time.time()
    while total < args.max_transitions:
        ep_seed = args.seed_base * 1000003 + args.actor * 100000 + episode
        flat = env.reset(seed=ep_seed)
        pend.clear()
        ep_score, moves = 0.0, 0
        geom = env.cur_geometry  # (width, killy) pinned by geometry spec
        while True:
            q16, a_t = teacher_q(flat)
            if rng.random() < eps:
                a = int(rng.integers(K))
            else:
                a = a_t
            img = render_jpeg(env.env, out_size=out_size,
                              quality=args.jpeg_quality)
            nflat, r, done, info = env.step(a)
            ep_score += r
            moves += 1
            pend.append((img, a, r, done, q16, a == a_t, flat))
            if len(pend) >= n_step or done:
                Rn = 0.0
                for i, p in enumerate(pend):
                    Rn += (gamma ** i) * p[2]
                    if p[3]:
                        break
                im0, a0, _, dn, q0, am0, fl0 = pend[0]
                n_eff = len(pend)
                bufs["img"].append(im0); bufs["act"].append(a0)
                bufs["rew"].append(Rn)
                bufs["done"].append(float(dn)); bufs["gam"].append(gamma ** n_eff)
                bufs["qteach"].append(q0); bufs["argmax"].append(am0)
                bufs["flat"].append(fl0); bufs["geom"].append(geom)
                pend.popleft()
                total += 1
            if done:
                while pend:
                    Rn = 0.0
                    for i, p in enumerate(pend):
                        Rn += (gamma ** i) * p[2]
                        if p[3]:
                            break
                    im0, a0, _, dn, q0, am0, fl0 = pend[0]
                    bufs["img"].append(im0); bufs["act"].append(a0)
                    bufs["rew"].append(Rn)
                    bufs["done"].append(1.0)
                    bufs["gam"].append(gamma ** len(pend))
                    bufs["qteach"].append(q0); bufs["argmax"].append(am0)
                    bufs["flat"].append(fl0); bufs["geom"].append(geom)
                    pend.popleft()
                    total += 1
                ep_log.write(json.dumps({
                    "t": round(time.time() - t0, 1), "actor": args.actor,
                    "episode": episode, "score": ep_score, "moves": moves,
                    "eps": round(eps, 4), "seed": ep_seed,
                    "geom": list(geom)}) + "\n")
                ep_log.flush()
                break
            flat = nflat
            if len(bufs["act"]) >= shard_size:
                flush()
        episode += 1
    flush()
    ep_log.write(json.dumps({"t": round(time.time() - t0, 1),
                             "actor": args.actor, "done": True,
                             "total": total}) + "\n")
    ep_log.flush()


if __name__ == "__main__":
    main()
