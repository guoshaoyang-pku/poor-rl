"""Gym-like DQN wrapper around the headless SuikaEnv (new merge rule)."""
import numpy as np

from paths import setup_engine_path
setup_engine_path()

from suika_env import SuikaEnv  # noqa: E402
from config import config  # noqa: E402
from encoding import (col_to_x, encode_state, encode_tokens, input_dim,  # noqa: E402
                      input_dim_tokens)

# stock board: centre column of the container, used to keep it centred when
# the width is resampled.
_STOCK_CX = (config.pad.left + config.pad.right) / 2.0
_STOCK_KILLY = int(config.pad.killy)
_STOCK_W = int(config.pad.right - config.pad.left)
_STOCK_H = int(config.pad.bot - config.pad.killy)   # 505: killy -> floor


def set_geometry(width, height):
    """Rewrite the shared board config in place (takes effect at next reset).

    width: container width, kept symmetric about the stock centre line so the
    mirror augmentation stays valid. height: killy -> floor distance; the
    FLOOR moves (bot = killy + height) while top/killy stay pinned, fruit
    sizes and gravity untouched.
    """
    width = int(width) // 2 * 2
    config.pad.left = int(_STOCK_CX - width // 2)
    config.pad.right = int(_STOCK_CX + width // 2)
    config.pad.bot = int(config.pad.killy) + int(height)


def sample_geometry(spec, seed):
    """Deterministic (width, height) draw for an episode seed.

    spec: {"width": [lo, hi] | int | None, "height": [lo, hi] | int | None};
    an omitted axis keeps the stock value. The same seed always yields the
    same board, so evaluation seeds are common random numbers across policies.
    """
    rng = np.random.default_rng([int(seed) & 0x7FFFFFFF, 0x6E0])

    def draw(v, default):
        if v is None:
            return default
        if isinstance(v, (int, float)):
            return int(v)
        lo, hi = int(v[0]), int(v[1])
        return int(rng.integers(lo, hi + 1))

    return (draw(spec.get("width"), _STOCK_W),
            draw(spec.get("height"), _STOCK_H))


class DQNEnv:
    """One discrete drop per transition; reward = raw score delta."""

    def __init__(self, seed, K=128, max_fruits=80, boundary=True,
                 reward_scale=1.0, tempo=False, obs_format="flat",
                 geometry=None, geo_dim=0):
        self.K = int(K)
        self.max_fruits = int(max_fruits)
        self.boundary = bool(boundary)
        self.reward_scale = float(reward_scale)
        self.tokens = obs_format == "tokens"
        self.geometry = dict(geometry) if geometry else None
        self.geo_dim = int(geo_dim)
        self.cur_geometry = None
        if self.geometry and seed is not None:
            set_geometry(*sample_geometry(self.geometry, seed))
        if tempo:
            from tempo_env import TempoSuikaEnv
            self.env = TempoSuikaEnv(seed=seed)
        else:
            self.env = SuikaEnv(seed=seed)
        self._seed = seed
        self.obs_dim = (input_dim_tokens(geo_dim=self.geo_dim) if self.tokens
                        else input_dim(max_fruits, boundary=boundary))

    def _obs(self, state):
        if self.tokens:
            return encode_tokens(state, geo_dim=self.geo_dim)
        return encode_state(state, self.K, self.max_fruits, self.boundary)

    def reset(self, seed=None, geom=None):
        """geom=(width, height) pins the board (evaluation grids); otherwise
        it is drawn from the configured spec by episode seed."""
        if seed is not None:
            self._seed = seed
        if geom is not None:
            set_geometry(*geom)
        elif self.geometry:
            set_geometry(*sample_geometry(
                self.geometry,
                self._seed if self._seed is not None
                else int(np.random.SeedSequence().entropy) & 0x7FFFFFFF))
        self.cur_geometry = (int(config.pad.right - config.pad.left),
                             int(config.pad.killy),
                             int(config.pad.bot - config.pad.killy))
        state = self.env.reset(seed=self._seed)
        return self._obs(state)

    def step(self, col):
        x = col_to_x(col, self.K)
        state, reward, done, info = self.env.step(x)
        return self._obs(state), float(reward) * self.reward_scale, bool(done), info

    # convenience: raw game score of current episode
    @property
    def score(self):
        return self.env.score
