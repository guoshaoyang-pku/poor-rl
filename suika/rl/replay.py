"""Proportional prioritized replay with n-step transitions (numpy)."""
import numpy as np


class PrioritizedReplay:
    def __init__(self, capacity, obs_dim, alpha=0.6):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros(self.capacity, dtype=np.int64)
        self.rew = np.zeros(self.capacity, dtype=np.float32)
        self.nobs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(self.capacity, dtype=np.float32)
        self.gam = np.zeros(self.capacity, dtype=np.float32)   # gamma^n
        self.prio = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.max_prio = 1.0
        self.inserts = 0

    def add_batch(self, obs, act, rew, nobs, done, gam):
        n = len(obs)
        idx = (self.pos + np.arange(n)) % self.capacity
        self.obs[idx] = obs
        self.act[idx] = act
        self.rew[idx] = rew
        self.nobs[idx] = nobs
        self.done[idx] = done
        self.gam[idx] = gam
        self.prio[idx] = self.max_prio
        self.pos = int((self.pos + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)
        self.inserts += n

    def sample(self, batch, beta, rng):
        p = self.prio[:self.size].astype(np.float64) ** self.alpha
        p /= p.sum()
        idx = rng.choice(self.size, size=batch, replace=True, p=p)
        w = (self.size * p[idx]) ** (-beta)
        w /= w.max()
        return (idx, self.obs[idx], self.act[idx], self.rew[idx],
                self.nobs[idx], self.done[idx], self.gam[idx],
                w.astype(np.float32))

    def update_priorities(self, idx, td_abs):
        pr = np.abs(td_abs) + 1e-3
        self.prio[idx] = pr
        self.max_prio = max(self.max_prio, float(pr.max()))
