"""Physics settle ablation for the headless Suika env.

Replays FIXED seeds + FIXED action scripts across engine variants and compares:
  - throughput (env-only drops/s, ms per settle frame)
  - score / moves / frames distributions
  - per-seed divergence vs baseline (score path L1, final board diff)
  - numerical stability (max overlap, tunneling escapes, NaN)

Variants (all keep the same game rules / reward):
  base        original SuikaEnv settle code path (verbatim copy + frame counter)
  fast        single-pass per-frame scan; physics op sequence unchanged
              -> must be bit-identical to base (sanity check)
  fast_it8/6/4  solver iterations 10 -> 8/6/4
  fast_dt45   dt=1/45, settle budget 180 frames (same 4 s wall-sim time)
  fast_dt30   dt=1/30, settle budget 120 frames
  fast_dt30it15  dt=1/30 with iterations=15 (compensate accuracy)
  fast_check3 scan game-over/settle every 3 frames instead of 1
  fast_check3_it6  both of the above pragmatic knobs combined

Usage: python phy_ablation.py [--seeds 30] [--variants ...] [--json OUT]
"""
import argparse
import json
import math
import os
import time

import numpy as np

from paths import setup_engine_path
setup_engine_path()

from suika_env import SuikaEnv          # noqa: E402
from particle import Particle           # noqa: E402
from config import config               # noqa: E402
from encoding import encode_state       # noqa: E402
from pymunk._chipmunk import lib as _cp  # noqa: E402  (direct C API reads)

K = 128
KILLY = config.pad.killy
BOT = config.pad.bot
LEFT = config.pad.left
RIGHT = config.pad.right
_HERE = os.path.dirname(os.path.abspath(__file__))  # setup_engine_path chdirs


class AblateEnv(SuikaEnv):
    """SuikaEnv with knobs; fast=False reproduces the original settle loop."""

    def __init__(self, seed=None, max_settle_steps=240, settle_velocity=2.0,
                 fps=None, iterations=None, slop=None, sleep=None,
                 fast=True, check_every=1):
        self._knob_iters = iterations
        self._knob_slop = slop
        self._knob_sleep = sleep
        self._fast = fast
        self._check_every = max(1, int(check_every))
        super().__init__(max_settle_steps=max_settle_steps,
                         settle_velocity=settle_velocity, seed=seed)
        if fps is not None:
            self.fps = fps
        self.last_settle_frames = 0

    def reset(self, seed=None):
        st = super().reset(seed=seed)
        if self._knob_iters is not None:
            self.space.iterations = self._knob_iters
        if self._knob_slop is not None:
            self.space.collision_slop = self._knob_slop
        if self._knob_sleep is not None:
            self.space.sleep_time_threshold = self._knob_sleep
        return st

    def _settle(self):
        if not self._fast:
            # verbatim copy of SuikaEnv._settle + frame counter
            n = 0
            for _ in range(self.max_settle_steps):
                self.space.step(1 / self.fps)
                n += 1
                if self._check_game_over():
                    self.game_over = True
                    break
                vmax = max(
                    (p.body.velocity.length for p in self._live_particles()),
                    default=0.0,
                )
                if vmax < self.settle_velocity:
                    break
            self.last_settle_frames = n
            return

        dt = 1.0 / self.fps
        sv2 = self.settle_velocity * self.settle_velocity
        every = self._check_every
        step = self.space.step
        space = self.space
        n = 0
        for i in range(self.max_settle_steps):
            step(dt)
            n += 1
            if every > 1 and (i % every) != every - 1 \
                    and i != self.max_settle_steps - 1:
                continue
            # single pass; identical break decisions to the original loop:
            #  - game over: any live particle with has_collided and y < killy
            #  - settled:   max velocity < sv  <=>  all sq(x^2+y^2) < sv^2
            over = False
            vmax_sq = 0.0
            for s in space.shapes:
                if isinstance(s, Particle) and s.alive:
                    b = s.body
                    if s.has_collided and _cp.cpBodyGetPosition(b._body).y < KILLY:
                        over = True
                        break
                    v = _cp.cpBodyGetVelocity(b._body)
                    sq = v.x * v.x + v.y * v.y
                    if sq > vmax_sq:
                        vmax_sq = sq
            if over:
                self.game_over = True
                break
            if vmax_sq < sv2:
                break
        self.last_settle_frames = n


# ------------------------------------------------------------------ #

VARIANTS = {
    "base":          dict(fast=False),
    "fast":          dict(fast=True),
    "fast_it8":      dict(fast=True, iterations=8),
    "fast_it6":      dict(fast=True, iterations=6),
    "fast_it4":      dict(fast=True, iterations=4),
    "fast_dt45":     dict(fast=True, fps=45, max_settle_steps=180),
    "fast_dt30":     dict(fast=True, fps=30, max_settle_steps=120),
    "fast_dt30it15": dict(fast=True, fps=30, max_settle_steps=120, iterations=15),
    "fast_check3":   dict(fast=True, check_every=3),
    "fast_check3_it6": dict(fast=True, check_every=3, iterations=6),
}
# NOTE: sleeping (space.sleep_time_threshold) was tried and HANGS inside
# cpSpaceStep: this game kills/spawns bodies inside the collision begin
# callback (collision.resolve_collision), which is illegal during a step
# and breaks Chipmunk's sleep bookkeeping. Sleeping would require
# restructuring merges into post-step callbacks first.


def action_script(seed, n=600):
    return np.random.default_rng(seed * 7919 + 13).integers(0, K, size=n)


def live_shapes(env):
    return [s for s in env.space.shapes
            if isinstance(s, Particle) and s.alive]


def board(env):
    """sorted list of (type, x, y) for divergence comparison."""
    out = []
    for s in live_shapes(env):
        p = s.body.position
        out.append((s.n, p.x, p.y))
    out.sort()
    return out


def stability(env):
    """(max_overlap, max_vel, nan, escape, n_fruits) at settle end."""
    shapes = live_shapes(env)
    maxpen = 0.0
    maxvel = 0.0
    nan = False
    escape = False
    n = len(shapes)
    for i in range(n):
        si = shapes[i]
        pi = si.body.position
        if not (math.isfinite(pi.x) and math.isfinite(pi.y)):
            nan = True
        if pi.y - si.radius > BOT + 1.0 or pi.x + si.radius < LEFT - 1.0 \
                or pi.x - si.radius > RIGHT + 1.0:
            escape = True
        v = si.body.velocity.length
        if v > maxvel:
            maxvel = v
        for j in range(i + 1, n):
            sj = shapes[j]
            pj = sj.body.position
            pen = si.radius + sj.radius - math.hypot(pi.x - pj.x, pi.y - pj.y)
            if pen > maxpen:
                maxpen = pen
    return maxpen, maxvel, nan, escape, n


def run_episode(env, seed):
    """Returns per-episode record incl. per-drop score path and boards."""
    env.reset(seed=seed)
    acts = action_script(seed)
    t0 = time.perf_counter()
    score_path = []
    boards = []
    frames = 0
    drops = 0
    maxpen = 0.0
    maxvel = 0.0
    nan = False
    escape = False
    done = False
    i = 0
    settle_t = 0.0
    while not done and i < len(acts):
        ts = time.perf_counter()
        _, r, done, info = env.step(int(acts[i]))
        settle_t += time.perf_counter() - ts
        i += 1
        drops += 1
        frames += getattr(env, "last_settle_frames", 0)
        score_path.append(env.score)
        boards.append(board(env))
        pen, vel, nn, esc, _ = stability(env)
        maxpen = max(maxpen, pen)
        maxvel = max(maxvel, vel)
        nan = nan or nn
        escape = escape or esc
    wall = time.perf_counter() - t0
    return dict(
        seed=seed, score=env.score, drops=drops, frames=frames,
        settle_s=settle_t, wall_s=wall,
        ms_per_frame=1000.0 * settle_t / max(1, frames),
        ms_per_drop=1000.0 * wall / max(1, drops),
        score_path=score_path, boards=boards,
        max_overlap=maxpen, max_vel=maxvel, nan=nan, escape=escape,
    )


def board_diff(a, b):
    """(type-multiset L1, mean matched distance) between two boards."""
    from collections import Counter
    ca = Counter(t for t, _, _ in a)
    cb = Counter(t for t, _, _ in b)
    l1 = sum(abs(ca[t] - cb[t]) for t in set(ca) | set(cb))
    dists = []
    for t in set(ca) & set(cb):
        pa = sorted((x, y) for tt, x, y in a if tt == t)
        pb = sorted((x, y) for tt, x, y in b if tt == t)
        for (x1, y1), (x2, y2) in zip(pa, pb):
            dists.append(math.hypot(x1 - x2, y1 - y2))
    return l1, (float(np.mean(dists)) if dists else 0.0)


def summarize(name, recs, base_recs):
    sc = np.array([r["score"] for r in recs], dtype=float)
    mv = np.array([r["drops"] for r in recs], dtype=float)
    fr = np.array([r["frames"] / max(1, r["drops"]) for r in recs])
    msf = np.array([r["ms_per_frame"] for r in recs])
    msd = np.array([r["ms_per_drop"] for r in recs])
    out = dict(
        name=name,
        score_mean=round(sc.mean(), 1), score_std=round(sc.std(), 1),
        moves_mean=round(mv.mean(), 1),
        frames_per_drop=round(fr.mean(), 1),
        ms_per_frame=round(msf.mean(), 3),
        ms_per_drop=round(msd.mean(), 2),
        drops_per_s_per_core=round(1000.0 / msd.mean(), 1),
        max_overlap_max=round(max(r["max_overlap"] for r in recs), 2),
        max_overlap_mean=round(float(np.mean([r["max_overlap"] for r in recs])), 2),
        max_vel_max=round(max(r["max_vel"] for r in recs), 1),
        nan=any(r["nan"] for r in recs),
        escapes=sum(1 for r in recs if r["escape"]),
        bit_exact_with_base=None,
    )
    if base_recs is not None:
        bm = {r["seed"]: r for r in base_recs}
        score_l1, tl1, mdist, exact = [], [], [], True
        for r in recs:
            b = bm[r["seed"]]
            m = min(len(r["score_path"]), len(b["score_path"]))
            score_l1.append(sum(abs(r["score_path"][k] - b["score_path"][k])
                                for k in range(m)))
            l1, dd = board_diff(r["boards"][-1], b["boards"][-1])
            tl1.append(l1)
            mdist.append(dd)
            exact = exact and (r["score"] == b["score"] and r["drops"] == b["drops"]
                               and r["score_path"] == b["score_path"]
                               and r["boards"] == b["boards"])
        out.update(
            score_path_l1_vs_base=round(float(np.mean(score_l1)), 1),
            final_type_l1_vs_base=round(float(np.mean(tl1)), 2),
            final_pos_meandist_vs_base=round(float(np.mean(mdist)), 2),
            bit_exact_with_base=bool(exact),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    names = args.variants or list(VARIANTS)
    seeds = [args.seed0 + i for i in range(args.seeds)]

    # sanity: copied base loop must match untouched SuikaEnv bit-for-bit
    plain = SuikaEnv(seed=seeds[0])
    ab = AblateEnv(seed=seeds[0], fast=False)
    for s in seeds[:3]:
        p_rec = run_episode(plain, s)
        a_rec = run_episode(ab, s)
        assert p_rec["score_path"] == a_rec["score_path"], "base copy diverged!"
    print("sanity: copied base settle loop == untouched SuikaEnv (3 seeds) OK")
    del plain, ab

    all_recs, results = {}, []
    for name in names:
        kw = VARIANTS[name]
        env = AblateEnv(seed=seeds[0], **kw)
        t0 = time.perf_counter()
        recs = [run_episode(env, s) for s in seeds]
        tot = time.perf_counter() - t0
        all_recs[name] = recs
        r = summarize(name, recs, all_recs.get("base"))
        r["wall_s_total"] = round(tot, 1)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False))
        print(f"  [{name}] {args.seeds} eps in {tot:.1f}s  "
              f"score {r['score_mean']}±{r['score_std']}  "
              f"moves {r['moves_mean']}  {r['ms_per_drop']} ms/drop  "
              f"{r['drops_per_s_per_core']} drops/s/core")

    if args.json:
        out = args.json if os.path.isabs(args.json) else os.path.join(_HERE, args.json)
        with open(out, "w") as f:
            json.dump(results, f, indent=1)
        print("wrote", out)


if __name__ == "__main__":
    main()
