"""Headless / programmatic interface for the Suika game (part2).

This module lets an AI agent drive the Suika ("Watermelon") game purely in
code, without ever opening an OS window. The original human-playable entry
point (``part2/main.py``) is left completely untouched.

Quick start::

    from suika_env import SuikaEnv
    env = SuikaEnv(seed=0)
    state = env.reset(seed=0)
    state, reward, done, info = env.step(640)   # 640 = drop x in pixels

Action: the x pixel coordinate where the current fruit is dropped (it is
clamped to the valid play area). ``step_column`` is provided as a convenience
for discrete-column agents.
"""
import os

# Force a headless SDL video driver *before* pygame is imported so that no
# window is ever created. The caller may override SDL_VIDEODRIVER beforehand.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import sys

# config.py loads its YAML/PNG assets using paths relative to the project
# root, so make both the imports and the asset paths resolvable no matter
# what the current working directory is.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../suika/part2
_ROOT = os.path.dirname(_THIS_DIR)                       # .../suika
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
os.chdir(_ROOT)

import numpy as np
import pygame
import pymunk

try:  # fast direct C reads; probe once, fall back to pymunk properties
    from pymunk._chipmunk import lib as _cp
    _cp.cpBodyGetPosition(pymunk.Body(1, 1)._body)
except Exception:  # pragma: no cover  (private API changed / missing)
    _cp = None

pygame.init()

from config import config, CollisionTypes
from particle import Particle
from preparticle import PreParticle
from wall import Wall
from collision import collide
import preparticle as _preparticle_mod

try:
    import csettle_loader as _cs
    _CS_OK = _cs.ok
except Exception:
    _cs = None
    _CS_OK = False


class SuikaEnv:
    """A minimal, gym-like headless wrapper around the Suika physics game."""

    def __init__(self, max_settle_steps=240, settle_velocity=2.0, seed=None):
        self.max_settle_steps = max_settle_steps
        self.settle_velocity = settle_velocity
        self.fps = config.screen.fps
        self._surface = None
        self.reset(seed=seed)

    # ------------------------------------------------------------------ #
    # play-area helpers
    # ------------------------------------------------------------------ #
    @property
    def play_left(self):
        return config.pad.left

    @property
    def play_right(self):
        return config.pad.right

    @property
    def score(self):
        return self.handler.data["score"]

    # ------------------------------------------------------------------ #
    # core API
    # ------------------------------------------------------------------ #
    def reset(self, seed=None):
        """Start a fresh game and return the initial state dict."""
        if seed is not None:
            _preparticle_mod.rng = np.random.default_rng(seed)
        self.space = pymunk.Space()
        self.space.gravity = (0, config.physics.gravity)
        self.space.damping = config.physics.damping
        self.space.collision_bias = config.physics.bias
        self.walls = [
            Wall(config.top_left, config.bot_left, self.space),
            Wall(config.bot_left, config.bot_right, self.space),
            Wall(config.bot_right, config.top_right, self.space),
        ]
        self.handler = self.space.add_collision_handler(
            CollisionTypes.PARTICLE, CollisionTypes.PARTICLE
        )
        self.handler.begin = collide
        self.handler.data["score"] = 0
        self.curr = PreParticle()
        self.next = PreParticle()
        self.game_over = False
        self.steps_taken = 0
        return self.get_state()

    def step(self, action):
        """Drop the current fruit at x=``action`` (pixels), then let physics
        settle. Returns ``(state, reward, done, info)`` where reward is the
        score gained on this step."""
        if self.game_over:
            return self.get_state(), 0.0, True, {"score": self.score}
        prev_score = self.score
        self.curr.set_x(float(action))
        self.curr.release(self.space)
        # advance the "next fruit" preview, just like Cloud.step()
        self.curr = self.next
        self.next = PreParticle()
        self._settle()
        if not self.game_over:
            self.game_over = self._check_game_over()
        self.steps_taken += 1
        reward = float(self.score - prev_score)
        return self.get_state(), reward, self.game_over, {"score": self.score}

    def step_column(self, col, num_columns=12):
        """Convenience action: drop into one of ``num_columns`` evenly spaced
        columns (0 .. num_columns-1)."""
        lo, hi = self.play_left, self.play_right
        col = int(np.clip(col, 0, num_columns - 1))
        x = lo + (hi - lo) * (col + 0.5) / num_columns
        return self.step(x)

    def get_state(self):
        """Return a structured snapshot of the board."""
        fruits = []
        for p in self._live_particles():
            x, y = p.body.position
            vx, vy = p.body.velocity
            fruits.append({
                "x": float(x),
                "y": float(y),
                "vx": float(vx),
                "vy": float(vy),
                "radius": float(p.radius),
                "type": int(p.n),
                "name": config.fruit_names[p.n],
            })
        return {
            "fruits": fruits,
            "current": {
                "type": int(self.curr.n),
                "name": config.fruit_names[self.curr.n],
                "radius": float(self.curr.radius),
            },
            "next": {
                "type": int(self.next.n),
                "name": config.fruit_names[self.next.n],
            },
            "score": int(self.score),
            "game_over": bool(self.game_over),
            "play_area": {
                "left": config.pad.left,
                "right": config.pad.right,
                "top": config.pad.top,
                "bottom": config.pad.bot,
            },
        }

    def render(self, mode="rgb_array"):
        """``mode='none'`` -> None. ``mode='rgb_array'`` -> (H, W, 3) uint8
        numpy frame rendered to an off-screen surface. No OS window is opened
        (this env is headless); use ``part2/main.py`` for interactive play."""
        if mode == "none":
            return None
        if self._surface is None:
            self._surface = pygame.Surface(
                (config.screen.width, config.screen.height)
            )
        self._surface.blit(config.background_blit, (0, 0))
        for p in self._live_particles():
            p.draw(self._surface)
        from text import score as _draw_score
        _draw_score(self.score, self._surface)
        arr = pygame.surfarray.array3d(self._surface)   # (W, H, 3)
        return np.transpose(arr, (1, 0, 2))             # (H, W, 3)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _live_particles(self):
        return [
            s for s in self.space.shapes
            if isinstance(s, Particle) and getattr(s, "alive", False)
        ]

    def _check_game_over(self):
        for p in self._live_particles():
            if p.pos[1] < config.pad.killy and p.has_collided:
                return True
        return False

    def _settle(self):
        """Advance physics until settled / dead / budget used.

        Break decisions are bit-identical to the original two-scan loop
        (game-over check then max-velocity check), verified by
        suika_dqn/phy_ablation.py.  Preferred path: one stateless C call
        per frame (csettle.c) returning (vmax_sq, min_y); game over is only
        re-evaluated in Python when min_y < killy (a necessary condition),
        so the common frame costs zero Python work.  Falls back to a
        single-pass Python scan if the extension is not built.
        vmax < v  <=>  max(x^2+y^2) < v^2.
        """
        dt = 1 / self.fps
        sv2 = self.settle_velocity * self.settle_velocity
        killy = config.pad.killy
        step = self.space.step
        space = self.space
        if _CS_OK:
            scan = _cs.scan
            check = self._check_game_over
            for _ in range(self.max_settle_steps):
                step(dt)
                vmax_sq, min_y = scan(space)
                if min_y < killy and check():
                    self.game_over = True
                    break
                if vmax_sq < sv2:
                    break
            return
        for _ in range(self.max_settle_steps):
            step(dt)
            over = False
            vmax_sq = 0.0
            for s in space.shapes:
                if isinstance(s, Particle) and s.alive:
                    b = s.body
                    if s.has_collided:
                        if _cp is not None:
                            y = _cp.cpBodyGetPosition(b._body).y
                        else:
                            y = b.position.y
                        if y < killy:
                            over = True
                            break
                    if _cp is not None:
                        v = _cp.cpBodyGetVelocity(b._body)
                        sq = v.x * v.x + v.y * v.y
                    else:
                        v = b.velocity
                        sq = v.x * v.x + v.y * v.y
                    if sq > vmax_sq:
                        vmax_sq = sq
            if over:
                self.game_over = True
                break
            if vmax_sq < sv2:
                break
