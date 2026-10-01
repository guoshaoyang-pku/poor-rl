"""Actor process: epsilon-greedy rollouts, n-step assembly, shard upload."""
import argparse
import collections
import json
import os
import time

import numpy as np
import torch

from env import DQNEnv
from model import build_model


class PolicyCache:
    def __init__(self, path, cfg, obs_dim):
        self.path = path
        self.model = build_model(cfg, obs_dim)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.mtime = 0.0
        self.env_steps = 0

    def maybe_reload(self):
        try:
            mt = os.path.getmtime(self.path)
        except OSError:
            return False
        if mt <= self.mtime:
            return False
        try:
            payload = torch.load(self.path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(payload["state_dict"])
            self.mtime = mt
            self.env_steps = int(payload.get("env_steps", 0))
            return True
        except Exception:
            return False

    @torch.no_grad()
    def act(self, obs, K, eps, rng):
        if rng.random() < eps:
            return int(rng.integers(K))
        x = torch.from_numpy(obs).unsqueeze(0)
        q = self.model.q_values(x)[0]
        return int(q.argmax().item())


class ServerPolicy:
    """Q-values from the arm's GPU inference server (zmq REQ per step).

    Policy reload happens server-side; actors stay thin CPU processes.
    """

    def __init__(self, addr):
        import zmq
        self._zmq = zmq
        self.addr = addr
        self.ctx = zmq.Context.instance()
        self.sock = None
        self._connect()
        self.env_steps = 0

    def _connect(self):
        if self.sock is not None:
            self.sock.close(0)
        self.sock = self.ctx.socket(self._zmq.REQ)
        self.sock.setsockopt(self._zmq.RCVTIMEO, 10000)
        self.sock.setsockopt(self._zmq.SNDTIMEO, 10000)
        self.sock.setsockopt(self._zmq.LINGER, 0)
        self.sock.connect(self.addr)

    def maybe_reload(self):
        return False

    def act(self, obs, K, eps, rng):
        if rng.random() < eps:
            return int(rng.integers(K))
        payload = np.ascontiguousarray(obs, dtype=np.float32).tobytes()
        for attempt in (0, 1):
            try:
                self.sock.send(payload)
                q = np.frombuffer(self.sock.recv(), dtype=np.float32)
                return int(q[:K].argmax())
            except self._zmq.Again:
                if attempt == 0:
                    self._connect()
        raise RuntimeError("inference server unreachable at " + self.addr)


def run_actor(cfg, actor_idx, n_actors, run_dir, obs_dim):
    torch.set_num_threads(1)
    K = int(cfg["K"])
    n_step = int(cfg.get("n_step", 3))
    gamma = float(cfg["gamma"])
    shard_size = int(cfg.get("actor_shard", 4096))
    seed0 = int(cfg.get("seed", 0))
    rng = np.random.default_rng(seed0 * 100003 + actor_idx)
    # Ape-X style per-actor epsilon, geometric spread.
    eps_min = float(cfg.get("eps_min", 0.02))
    eps_max = float(cfg.get("eps_max", 0.5))
    if n_actors > 1:
        eps = eps_max * (eps_min / eps_max) ** (actor_idx / (n_actors - 1))
    else:
        eps = eps_min

    env = DQNEnv(seed=None, K=K, max_fruits=int(cfg["max_fruits"]),
                 boundary=bool(cfg.get("boundary", True)),
                 reward_scale=float(cfg.get("reward_scale", 1.0)),
                 tempo=bool(cfg.get("tempo", False)),
                 obs_format=cfg.get("obs_format", "flat"),
                 geometry=cfg.get("geometry"),
                 geo_dim=int(cfg.get("geo_dim", 0)))
    if cfg.get("infer") == "server":
        policy = ServerPolicy(os.environ["SUIKA_INFER_ADDR"])
    else:
        policy = PolicyCache(os.path.join(run_dir, "policy.pt"), cfg, obs_dim)
    inbox = os.path.join(run_dir, "inbox")
    os.makedirs(inbox, exist_ok=True)
    ep_log = open(os.path.join(run_dir, f"episodes_a{actor_idx}.jsonl"), "a")
    # full action sequences go to their own per-actor file (episodes_a*.jsonl
    # stays slim and git-friendly). Replay = seed + board + actions; see
    # replay_actions.py.
    log_actions = bool(cfg.get("log_actions", True)) and K <= 256
    act_log = (open(os.path.join(run_dir, f"actions_a{actor_idx}.jsonl"), "a")
               if log_actions else None)

    buf_obs, buf_act, buf_rew, buf_nobs, buf_done, buf_gam = [], [], [], [], [], []
    pend = collections.deque()   # (obs, act, rew)
    episode = 0
    t0 = time.time()
    steps_done = 0

    def flush_shard():
        if not buf_obs:
            return
        fn = f"a{actor_idx}_{time.time_ns()}.npz"
        tmp = os.path.join(inbox, fn + ".tmp.npz")  # .npz suffix: no auto-rename
        np.savez(tmp, obs=np.asarray(buf_obs, dtype=np.float32),
                 act=np.asarray(buf_act, dtype=np.int64),
                 rew=np.asarray(buf_rew, dtype=np.float32),
                 nobs=np.asarray(buf_nobs, dtype=np.float32),
                 done=np.asarray(buf_done, dtype=np.float32),
                 gam=np.asarray(buf_gam, dtype=np.float32))
        os.replace(tmp, os.path.join(inbox, fn))
        buf_obs.clear(); buf_act.clear(); buf_rew.clear()
        buf_nobs.clear(); buf_done.clear(); buf_gam.clear()

    while True:
        policy.maybe_reload()
        ep_seed = seed0 * 1000003 + actor_idx * 100000 + episode
        obs = env.reset(seed=ep_seed)
        pend.clear()
        ep_score = 0.0
        moves = 0
        max_type = 0
        acts = bytearray()      # K<=256 column ids; replay = seed+geometry+acts
        while True:
            a = policy.act(obs, K, eps, rng)
            acts.append(a)
            nobs, r, done, info = env.step(a)
            ep_score += r / float(cfg.get("reward_scale", 1.0))
            moves += 1
            pend.append((obs, a, r, nobs, done))
            if len(pend) >= n_step or done:
                Rn = 0.0
                for i, (_, _, ri, _, di) in enumerate(pend):
                    Rn += (gamma ** i) * ri
                    if di:
                        break
                # The return is accumulated over the whole pending window, so
                # the bootstrap state must be the state after the last step
                # in that window (not pend[0]'s one-step successor).
                o0, a0, _, _, _ = pend[0]
                _, _, _, on, dn = pend[-1]
                n_eff = len(pend)
                buf_obs.append(o0); buf_act.append(a0); buf_rew.append(Rn)
                buf_nobs.append(on); buf_done.append(float(dn))
                buf_gam.append(gamma ** n_eff)
                pend.popleft()
                steps_done += 1
            if done:
                # drain remaining n-step tail
                while pend:
                    Rn = 0.0
                    for i, (_, _, ri, _, di) in enumerate(pend):
                        Rn += (gamma ** i) * ri
                        if di:
                            break
                    o0, a0, _, _, _ = pend[0]
                    _, _, _, on, dn = pend[-1]
                    buf_obs.append(o0); buf_act.append(a0); buf_rew.append(Rn)
                    buf_nobs.append(on); buf_done.append(1.0)
                    buf_gam.append(gamma ** len(pend))
                    pend.popleft()
                    steps_done += 1
                fruits = info.get("score", ep_score)
                ep_log.write(json.dumps({
                    "t": round(time.time() - t0, 1), "actor": actor_idx,
                    "episode": episode, "score": ep_score, "moves": moves,
                    "eps": round(eps, 4), "seed": ep_seed,
                    "w": env.cur_geometry[0],
                    "killy": env.cur_geometry[1],
                    "h": env.cur_geometry[2]}) + "\n")
                ep_log.flush()
                if act_log is not None:
                    act_log.write(json.dumps({
                        "actor": actor_idx, "episode": episode,
                        "seed": ep_seed, "w": env.cur_geometry[0],
                        "killy": env.cur_geometry[1],
                        "h": env.cur_geometry[2], "eps": round(eps, 4),
                        "score": ep_score, "moves": moves,
                        "actions": acts.hex()}) + "\n")
                    act_log.flush()
                break
            obs = nobs
            if len(buf_obs) >= shard_size:
                flush_shard()
        episode += 1
        if episode % 20 == 0:
            flush_shard()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--obs-dim", type=int, required=True)
    ap.add_argument("--actor-idx", type=int, required=True)
    ap.add_argument("--n-actors", type=int, required=True)
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(args.config))
    run_actor(cfg, args.actor_idx, args.n_actors, args.run_dir, args.obs_dim)


if __name__ == "__main__":
    main()
