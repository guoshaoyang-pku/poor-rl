"""BC data collector: teacher (w3b_mlp_deep MLP) rollouts in Qwen token format.

Per transition stores the SAME n-step assembly as actor.py (recipe alignment),
plus teacher's full Q vector at s (for value distillation) and an is_argmax
flag (CE loss only on expert actions; eps-mixed actions still usable for TD).

Shard fields: obs[T*5] act rew nobs done gam (identical semantics to actor.py)
              + qteach[128] f16 (teacher Q at obs) + argmax u8 (1 if act==argmax)
Teacher input = flat-333 encoding recomputed from the SAME raw state.
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

from encoding import col_to_x, encode_state, encode_tokens  # noqa: E402
from env import DQNEnv  # noqa: E402
from model import build_model  # noqa: E402


class DualEnv(DQNEnv):
    """DQNEnv that returns (tokens_obs, flat_obs) from the same raw state."""

    def reset_dual(self, seed=None):
        if seed is not None:
            self._seed = seed
        state = self.env.reset(seed=self._seed)
        return self._obs(state), encode_state(
            state, self.K, self.max_fruits, self.boundary)

    def step_dual(self, col):
        x = col_to_x(col, self.K)
        state, reward, done, info = self.env.step(x)
        return (self._obs(state),
                encode_state(state, self.K, self.max_fruits, self.boundary),
                float(reward) * self.reward_scale, bool(done), info)


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
    args = ap.parse_args()

    torch.set_num_threads(1)
    K, n_step, gamma = 128, args.n_step, 1.0
    shard_size = 4096
    rng = np.random.default_rng(990001 + args.actor)
    eps = (args.eps_max * (args.eps_min / args.eps_max)
           ** (args.actor / max(args.actors - 1, 1)))

    teacher = build_model(TEACHER_CFG, 333)
    payload = torch.load(args.teacher, map_location="cpu", weights_only=False)
    teacher.load_state_dict(payload["state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    env = DualEnv(seed=None, K=K, max_fruits=80, boundary=True,
                  reward_scale=1.0, tempo=False, obs_format="tokens")
    obs_dim = env.obs_dim
    inbox = os.path.join(args.run_dir, "inbox")
    os.makedirs(inbox, exist_ok=True)
    ep_log = open(os.path.join(args.run_dir,
                               f"episodes_a{args.actor}.jsonl"), "a")

    bufs = {k: [] for k in
            ("obs", "act", "rew", "nobs", "done", "gam", "qteach", "argmax")}

    def flush():
        if not bufs["obs"]:
            return
        fn = f"a{args.actor}_{time.time_ns()}.npz"
        tmp = os.path.join(inbox, fn + ".tmp.npz")
        np.savez(tmp,
                 obs=np.asarray(bufs["obs"], dtype=np.float32),
                 act=np.asarray(bufs["act"], dtype=np.int64),
                 rew=np.asarray(bufs["rew"], dtype=np.float32),
                 nobs=np.asarray(bufs["nobs"], dtype=np.float32),
                 done=np.asarray(bufs["done"], dtype=np.float32),
                 gam=np.asarray(bufs["gam"], dtype=np.float32),
                 qteach=np.asarray(bufs["qteach"], dtype=np.float16),
                 argmax=np.asarray(bufs["argmax"], dtype=np.uint8))
        os.replace(tmp, os.path.join(inbox, fn))
        for v in bufs.values():
            v.clear()

    def teacher_q(flat):
        with torch.no_grad():
            x = torch.from_numpy(flat).unsqueeze(0)
            q = teacher.q_values(x)[0]
        return q.numpy().astype(np.float16), int(q.argmax().item())

    # n-step assembly identical to actor.py (fixed semantics)
    pend = collections.deque()
    episode, total, t0 = 0, 0, time.time()
    while total < args.max_transitions:
        ep_seed = 990001 * 1000003 + args.actor * 100000 + episode
        obs, flat = env.reset_dual(seed=ep_seed)
        pend.clear()
        ep_score, moves = 0.0, 0
        while True:
            q16, a_t = teacher_q(flat)
            if rng.random() < eps:
                a = int(rng.integers(K))
            else:
                a = a_t
            nobs_t, nflat, r, done, info = env.step_dual(a)
            ep_score += r
            moves += 1
            pend.append((obs, a, r, nobs_t, done, q16, a == a_t))
            if len(pend) >= n_step or done:
                Rn = 0.0
                for i, p in enumerate(pend):
                    Rn += (gamma ** i) * p[2]
                    if p[4]:
                        break
                o0, a0, _, on, dn, q0, am0 = pend[0]
                n_eff = len(pend)
                bufs["obs"].append(o0); bufs["act"].append(a0)
                bufs["rew"].append(Rn); bufs["nobs"].append(on)
                bufs["done"].append(float(dn)); bufs["gam"].append(gamma ** n_eff)
                bufs["qteach"].append(q0); bufs["argmax"].append(am0)
                pend.popleft()
                total += 1
            if done:
                while pend:
                    Rn = 0.0
                    for i, p in enumerate(pend):
                        Rn += (gamma ** i) * p[2]
                        if p[4]:
                            break
                    o0, a0, _, on, dn, q0, am0 = pend[0]
                    bufs["obs"].append(o0); bufs["act"].append(a0)
                    bufs["rew"].append(Rn); bufs["nobs"].append(on)
                    bufs["done"].append(1.0)
                    bufs["gam"].append(gamma ** len(pend))
                    bufs["qteach"].append(q0); bufs["argmax"].append(am0)
                    pend.popleft()
                    total += 1
                ep_log.write(json.dumps({
                    "t": round(time.time() - t0, 1), "actor": args.actor,
                    "episode": episode, "score": ep_score, "moves": moves,
                    "eps": round(eps, 4), "seed": ep_seed}) + "\n")
                ep_log.flush()
                break
            obs, flat = nobs_t, nflat
            if len(bufs["obs"]) >= shard_size:
                flush()
        episode += 1
    flush()
    ep_log.write(json.dumps({"t": round(time.time() - t0, 1),
                             "actor": args.actor, "done": True,
                             "total": total}) + "\n")
    ep_log.flush()


if __name__ == "__main__":
    main()
