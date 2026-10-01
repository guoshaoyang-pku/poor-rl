"""A "smarter than random" Suika agent (no training required).

Strategy: **heuristic + 1-step lookahead**.

The headless ``SuikaEnv`` cannot be ``copy.deepcopy``-ed nor ``space.copy()``-ed
because the custom ``Wall`` / ``Particle`` subclasses break pymunk's pickle
reconstruction (``Wall.__init__() takes 4 positional arguments but 5 were
given``). So instead of cloning the live env, we **rebuild an independent
pymunk space from the structured ``state`` snapshot** and simulate dropping the
current fruit there. This gives a *real physics* 1-step lookahead (merges,
settling, game-over) without ever touching the live env's global RNG. For
speed, the collision-free free-fall before first contact can be fast-forwarded;
all contact / merge dynamics still run through pymunk.

If the rebuild-based simulation fails for any reason, the agent gracefully
falls back to a pure state-based heuristic.
"""
import os
import sys
import math

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
import pymunk

from config import config, CollisionTypes
from particle import Particle
from wall import Wall
from collision import collide


BOTTOM_WALL_RADIUS = 2.0
FREEFALL_CLEARANCE = 1.0
FREEFALL_MIN_DISTANCE = 4.0


# --------------------------------------------------------------------------- #
# Lookahead simulation: rebuild a throwaway space from a state snapshot.
# --------------------------------------------------------------------------- #
def _build_space():
    space = pymunk.Space()
    space.gravity = (0, config.physics.gravity)
    space.damping = config.physics.damping
    space.collision_bias = config.physics.bias
    # keep refs to walls so they are not GC'd
    walls = [
        Wall(config.top_left, config.bot_left, space),
        Wall(config.bot_left, config.bot_right, space),
        Wall(config.bot_right, config.top_right, space),
    ]
    handler = space.add_collision_handler(
        CollisionTypes.PARTICLE, CollisionTypes.PARTICLE
    )
    handler.begin = collide
    handler.data["score"] = 0
    return space, handler, walls


def _live(space):
    return [s for s in space.shapes
            if isinstance(s, Particle) and getattr(s, "alive", False)]


def _check_game_over(space):
    for p in _live(space):
        if p.pos[1] < config.pad.killy and p.has_collided:
            return True
    return False


def _first_contact_y(fruits, x, radius):
    best_y = config.pad.bot - float(radius) - BOTTOM_WALL_RADIUS
    for f in fruits:
        reach = float(radius) + float(f["radius"])
        dx = float(f["x"]) - float(x)
        if abs(dx) >= reach:
            continue
        dy2 = reach * reach - dx * dx
        contact_y = float(f["y"]) - math.sqrt(max(0.0, dy2))
        if config.pad.top <= contact_y < best_y:
            best_y = contact_y
    return best_y


def _freefall_state_after(frames):
    frames = int(frames)
    if frames <= 0:
        return config.pad.top, 0.0
    dt = 1.0 / float(config.screen.fps)
    gravity = float(config.physics.gravity)
    d = float(config.physics.damping) ** dt
    a = gravity * dt
    if abs(1.0 - d) < 1e-12:
        velocity = a * frames
        distance = dt * a * frames * (frames + 1) * 0.5
    else:
        velocity = a * (1.0 - d ** frames) / (1.0 - d)
        distance = (
            dt * a * (frames - d * (1.0 - d ** frames) / (1.0 - d))
            / (1.0 - d)
        )
    return config.pad.top + distance, velocity


def _freefall_frame_count(distance, max_frames=None):
    distance = float(distance)
    if distance <= 0.0:
        return 0
    lo, hi = 0, 1
    while _freefall_state_after(hi)[0] - config.pad.top <= distance:
        hi *= 2
        if max_frames is not None and hi >= max_frames:
            hi = int(max_frames)
            break
    if max_frames is not None:
        hi = min(hi, int(max_frames))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _freefall_state_after(mid)[0] - config.pad.top <= distance:
            lo = mid
        else:
            hi = mid - 1
    return int(lo)


def _skip_freefall(state, particle, x, radius, max_skip_steps=None):
    """Move the dropped fruit to just before first contact, with impact speed."""
    contact_y = _first_contact_y(state.get("fruits") or [], x, radius)
    safe_distance = contact_y - FREEFALL_CLEARANCE - config.pad.top
    skipped = _freefall_frame_count(safe_distance, max_skip_steps)
    target_y, velocity = _freefall_state_after(skipped)
    if skipped <= 0 or target_y - config.pad.top < FREEFALL_MIN_DISTANCE:
        return 0
    particle.body.position = (float(x), float(target_y))
    particle.body.velocity = (0.0, velocity)
    return int(skipped)


def _settle(space, max_steps=160, settle_velocity=2.0, fps=60):
    over = False
    for _ in range(max_steps):
        space.step(1 / fps)
        if _check_game_over(space):
            over = True
            break
        vmax = max((p.body.velocity.length for p in _live(space)), default=0.0)
        if vmax < settle_velocity:
            break
    return over


def simulate_drop(state, drop_x, max_steps=160, skip_freefall=True):
    """Rebuild the board from ``state`` and drop the current fruit at ``drop_x``.

    Returns a dict with: ``score_gain``, ``game_over``, ``fruits`` (resulting
    live fruits as {x,y,radius,type}). Raises on failure so the caller can fall
    back to the pure heuristic.
    """
    space, handler, _walls = _build_space()
    cur_type = int(state["current"]["type"])

    # recreate existing fruits; treat them as already-collided / settled so the
    # game-over rule behaves like the real engine.
    for f in state["fruits"]:
        p = Particle((float(f["x"]), float(f["y"])), int(f["type"]), space)
        p.has_collided = True

    # drop the current fruit from the top, clamped into the legal x-range.
    radius = config[cur_type, "radius"]
    x = float(np.clip(drop_x,
                      config.pad.left + radius,
                      config.pad.right - radius))
    dropped = Particle((x, config.pad.top), cur_type, space)
    skipped_steps = 0
    if skip_freefall:
        skipped_steps = _skip_freefall(
            state, dropped, x, radius, max(0, max_steps - 1))

    over = _settle(space, max_steps=max(1, max_steps - skipped_steps))
    fruits = [{
        "x": float(p.pos[0]), "y": float(p.pos[1]),
        "radius": float(p.radius), "type": int(p.n),
    } for p in _live(space)]
    return {
        "score_gain": float(handler.data["score"]),
        "game_over": bool(over),
        "fruits": fruits,
    }


# --------------------------------------------------------------------------- #
# Board metrics used by the scoring function.
# --------------------------------------------------------------------------- #
def _top_y(fruits):
    """Smallest top-edge y over all fruits (small == stack reaches near ceiling
    == dangerous). Larger is safer."""
    if not fruits:
        return float(config.pad.bot)
    return min(f["y"] - f["radius"] for f in fruits)


def _merge_potential(fruits):
    """Reward setups where two same-type fruits sit close together (about to
    merge). Returns a bonus that grows with the points value of the fruit."""
    bonus = 0.0
    n = len(fruits)
    for i in range(n):
        fi = fruits[i]
        for j in range(i + 1, n):
            fj = fruits[j]
            if fi["type"] != fj["type"]:
                continue
            d = np.hypot(fi["x"] - fj["x"], fi["y"] - fj["y"])
            reach = fi["radius"] + fj["radius"]
            if d < reach * 1.6:
                closeness = max(0.0, 1.0 - (d - reach) / (reach * 0.6 + 1e-9))
                pts = config[fi["type"], "points"]
                bonus += closeness * pts
    return bonus


# --------------------------------------------------------------------------- #
# The agent.
# --------------------------------------------------------------------------- #
class HeuristicLookaheadAgent:
    """Heuristic + 1-step physics lookahead Suika agent.

    Scoring weights (per candidate column):

        value = W_MERGE      * immediate merge score gained this drop
              + W_SAFETY      * top_y (higher stack-top y == lower/safer board)
              + W_POTENTIAL   * same-type adjacency bonus (future merges)
              - W_GAMEOVER    * (huge) if the drop ends the game
              - W_FRUITS      * number of fruits left on the board
    """

    W_MERGE = 4.0
    W_SAFETY = 1.0
    W_POTENTIAL = 0.6
    W_GAMEOVER = 1.0e6
    W_FRUITS = 1.5

    def __init__(self, num_columns=14, lookahead_steps=160, seed=0,
                 skip_freefall=True):
        self.num_columns = num_columns
        self.lookahead_steps = lookahead_steps
        self.skip_freefall = bool(skip_freefall)
        self.rng = np.random.default_rng(seed)
        self.lookahead_failures = 0
        self.lookahead_used = 0

    # -- candidate drop x positions (column centers) ----------------------- #
    def _candidate_xs(self, radius):
        lo = config.pad.left + radius
        hi = config.pad.right - radius
        if hi <= lo:
            return [float((config.pad.left + config.pad.right) / 2)]
        xs = []
        for c in range(self.num_columns):
            xs.append(lo + (hi - lo) * (c + 0.5) / self.num_columns)
        return xs

    def _score_outcome(self, outcome):
        if outcome["game_over"]:
            return -self.W_GAMEOVER + self.W_MERGE * outcome["score_gain"]
        fruits = outcome["fruits"]
        value = 0.0
        value += self.W_MERGE * outcome["score_gain"]
        value += self.W_SAFETY * _top_y(fruits)
        value += self.W_POTENTIAL * _merge_potential(fruits)
        value -= self.W_FRUITS * len(fruits)
        return value

    # -- pure-heuristic fallback (no physics) ------------------------------ #
    def _heuristic_decide(self, state):
        cur_type = int(state["current"]["type"])
        radius = config[cur_type, "radius"]
        fruits = state["fruits"]
        xs = self._candidate_xs(radius)
        best_x, best_val = xs[0], -1e18
        for x in xs:
            val = 0.0
            # prefer landing next to a same-type fruit (potential merge)
            same = [f for f in fruits if f["type"] == cur_type]
            if same:
                nearest = min(abs(f["x"] - x) for f in same)
                val += 3.0 * config[cur_type, "points"] * max(
                    0.0, 1.0 - nearest / 80.0)
            # estimate local surface height at this x (min y of fruits whose
            # column overlaps x); prefer dropping into low / empty spots.
            overlap = [f for f in fruits
                       if abs(f["x"] - x) < f["radius"] + radius]
            if overlap:
                surface_y = min(f["y"] - f["radius"] for f in overlap)
            else:
                surface_y = config.pad.bot
            val += 1.0 * surface_y          # lower surface (large y) is better
            # avoid the very top danger zone
            if surface_y < config.pad.killy + 60:
                val -= 500.0
            if val > best_val:
                best_val, best_x = val, x
        return float(best_x)

    # -- main entry point -------------------------------------------------- #
    def decide(self, state):
        if state.get("game_over"):
            return float((config.pad.left + config.pad.right) / 2)
        cur_type = int(state["current"]["type"])
        radius = config[cur_type, "radius"]
        xs = self._candidate_xs(radius)

        scored = []
        try:
            for x in xs:
                outcome = simulate_drop(
                    state, x, max_steps=self.lookahead_steps,
                    skip_freefall=self.skip_freefall)
                scored.append((self._score_outcome(outcome), x))
            self.lookahead_used += 1
        except Exception:
            # physics rebuild failed -> degrade to pure heuristic
            self.lookahead_failures += 1
            return self._heuristic_decide(state)

        best_val = max(v for v, _ in scored)
        # tie-break: among near-best columns pick the one closest to center
        center = (config.pad.left + config.pad.right) / 2
        eps = 1e-6
        best = [x for v, x in scored if v >= best_val - eps]
        best.sort(key=lambda x: abs(x - center))
        return float(best[0])
