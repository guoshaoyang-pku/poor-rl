"""Object-centric state encoding (ported from rl_rft_cem/common.py).

Flat layout: globals[G] + max_fruits*4 tokens, fruits sorted by (y, x).
G = 9 base + 4 boundary (boundary_features=True) = 13.
"""
import numpy as np

from paths import setup_engine_path
setup_engine_path()

from config import config  # noqa: E402  (part2 config)

PLAY_LEFT = config.pad.left
PLAY_RIGHT = config.pad.right
PLAY_TOP = config.pad.top
PLAY_BOT = config.pad.bot
KILLY = config.pad.killy
PLAY_W = PLAY_RIGHT - PLAY_LEFT
PLAY_H = PLAY_BOT - PLAY_TOP
NUM_FRUIT_TYPES = 11
MAX_RADIUS = config[10, "radius"]
SCORE_NORM = 2000.0
FRUIT_FEATS = 4
GLOBAL_FEATS = 9
BOUNDARY_EXTRA_FEATS = 4


def geom():
    """Live board geometry (left, right, top, bot, killy).

    config.pad is a mutable singleton; a variable-geometry env rewrites
    left/right/killy at reset, so anything geometry-dependent must read it
    here rather than from the import-time PLAY_* constants above (which are
    kept only for the flat-vector encoder and legacy importers).
    """
    p = config.pad
    return (float(p.left), float(p.right), float(p.top), float(p.bot),
            float(p.killy))


def col_to_x(col, K):
    col = int(np.clip(col, 0, K - 1))
    left, right, _, _, _ = geom()
    return left + (right - left) * (col + 0.5) / K


def _radius_of(t):
    return float(config[int(t) % NUM_FRUIT_TYPES, "radius"])


def _top_y(fruits):
    if not fruits:
        return float(PLAY_BOT)
    return min(f["y"] - f["radius"] for f in fruits)


def global_features(state, max_fruits, boundary=True):
    fruits = state["fruits"]
    cur_t = float(state["current"]["type"])
    nxt_t = float(state["next"]["type"])
    cur_r = float(state["current"].get("radius", _radius_of(cur_t)))
    n = len(fruits)
    if fruits:
        ys = np.array([f["y"] for f in fruits], dtype=np.float32)
        rs = np.array([f["radius"] for f in fruits], dtype=np.float32)
        mean_y = float(ys.mean())
        max_r = float(rs.max())
        fill = float(np.sum(np.pi * rs ** 2) / (PLAY_W * PLAY_H))
    else:
        mean_y, max_r, fill = float(PLAY_BOT), 0.0, 0.0
    feats = [
        cur_t / 10.0,
        nxt_t / 10.0,
        cur_r / MAX_RADIUS,
        n / float(max_fruits),
        float(state["score"]) / SCORE_NORM,
        (_top_y(fruits) - PLAY_TOP) / PLAY_H,
        (mean_y - PLAY_TOP) / PLAY_H,
        max_r / MAX_RADIUS,
        min(fill, 2.0),
    ]
    if boundary:
        if fruits:
            left_clear = min(max(0.0, float(f["x"]) - float(f["radius"]) - PLAY_LEFT)
                             for f in fruits)
            right_clear = min(max(0.0, PLAY_RIGHT - (float(f["x"]) + float(f["radius"])))
                              for f in fruits)
            bottom_clear = min(max(0.0, PLAY_BOT - (float(f["y"]) + float(f["radius"])))
                               for f in fruits)
        else:
            left_clear = right_clear = PLAY_W
            bottom_clear = PLAY_H
        top_margin = max(0.0, _top_y(fruits) - KILLY)
        legal_span = max(0.0, (PLAY_W - 2.0 * cur_r) / PLAY_W)
        feats.extend([
            min(left_clear, PLAY_W) / PLAY_W,
            min(right_clear, PLAY_W) / PLAY_W,
            min(top_margin, PLAY_H) / PLAY_H,
            0.5 * min(bottom_clear, PLAY_H) / PLAY_H + 0.5 * legal_span,
        ])
    return np.array(feats, dtype=np.float32)


def encode_state(state, K, max_fruits, boundary=True):
    g = global_features(state, max_fruits, boundary=boundary)
    tokens = np.zeros((max_fruits, FRUIT_FEATS), dtype=np.float32)
    fr = sorted(state["fruits"], key=lambda f: (f["y"], f["x"]))[:max_fruits]
    for i, f in enumerate(fr):
        tokens[i, 0] = (f["x"] - PLAY_LEFT) / PLAY_W
        tokens[i, 1] = (f["y"] - PLAY_TOP) / PLAY_H
        tokens[i, 2] = f["radius"] / MAX_RADIUS
        tokens[i, 3] = f["type"] / 10.0
    return np.concatenate([g, tokens.reshape(-1)]).astype(np.float32)


def input_dim(max_fruits, boundary=True):
    g = GLOBAL_FEATS + (BOUNDARY_EXTRA_FEATS if boundary else 0)
    return g + max_fruits * FRUIT_FEATS


def mirror_vec(vec, max_fruits, boundary=True):
    """Mirror a flat encoded state horizontally (data augmentation)."""
    out = np.asarray(vec, dtype=np.float32).copy()
    gdim = GLOBAL_FEATS + (BOUNDARY_EXTRA_FEATS if boundary else 0)
    if boundary:
        out[GLOBAL_FEATS + 0], out[GLOBAL_FEATS + 1] = (
            out[GLOBAL_FEATS + 1], out[GLOBAL_FEATS + 0])
    tokens = out[gdim:].reshape(max_fruits, FRUIT_FEATS)
    valid = tokens[:, 2] > 0.0
    tokens[valid, 0] = 1.0 - tokens[valid, 0]
    tokens[~valid] = 0.0
    return out


def mirror_action(col, K):
    return K - 1 - int(col)


# ---- token set encoding (architecture v2 / set transformer) ----
# Rows: 0 = current fruit, 1 = next fruit, 2.. = board fruits sorted by (y,x).
# Features per row: (type, x_n, y_n, vx_n, vy_n); padded rows have type=-1.
TOK_FEATS = 5
TOK_T = 160
VEL_NORM = 240.0


# Board-geometry reference (stock layout) and the optional geo block appended
# after the token rows: [(W-448)/75, ((bot-killy)-505)/110, ((killy-top)-85)/85]
# = [width, floor distance, headroom above the death line]. killy/top are
# pinned in wave6 (headroom is a constant 0) but the model still receives the
# feature so the headroom axis stays first-class. All zeros at the stock
# geometry, so a geo-aware model sees exactly the stock observation plus a
# zero vector there.
GEO_REF_W, GEO_SCALE_W = 448.0, 75.0
GEO_REF_H, GEO_SCALE_H = 505.0, 110.0     # bot - killy (distance to floor)
GEO_REF_A, GEO_SCALE_A = 85.0, 85.0       # killy - top (headroom)
GEO_FEATS = 3


def geo_vector():
    left, right, top, bot, killy = geom()
    return np.array([((right - left) - GEO_REF_W) / GEO_SCALE_W,
                     ((bot - killy) - GEO_REF_H) / GEO_SCALE_H,
                     ((killy - top) - GEO_REF_A) / GEO_SCALE_A],
                    dtype=np.float32)


def encode_tokens(state, T=TOK_T, geo_dim=0):
    left, right, top, bot, _ = geom()
    play_w, play_h = right - left, bot - top
    out = np.zeros((T, TOK_FEATS), dtype=np.float32)
    out[:, 0] = -1.0
    out[0, 0] = float(state["current"]["type"])
    out[1, 0] = float(state["next"]["type"])
    fr = sorted(state["fruits"], key=lambda f: (f["y"], f["x"]))[: T - 2]
    for i, f in enumerate(fr):
        r = out[2 + i]
        r[0] = float(f["type"])
        r[1] = (f["x"] - left) / play_w
        r[2] = (f["y"] - top) / play_h
        r[3] = np.clip(f.get("vx", 0.0) / VEL_NORM, -4.0, 4.0)
        r[4] = np.clip(f.get("vy", 0.0) / VEL_NORM, -4.0, 4.0)
    flat = out.reshape(-1)
    if geo_dim:
        assert geo_dim == GEO_FEATS, geo_dim
        flat = np.concatenate([flat, geo_vector()])
    return flat


def input_dim_tokens(T=TOK_T, geo_dim=0):
    return T * TOK_FEATS + int(geo_dim)


def mirror_tokens(vec, T=TOK_T):
    """Mirror a flat token-set state horizontally (x -> 1-x, vx -> -vx).

    Rows 0/1 (current/next fruit) carry no position and are left untouched.
    Anything after the T*TOK_FEATS token block (the geo vector) is symmetric
    under a left/right flip and is passed through unchanged.
    """
    full = np.asarray(vec, dtype=np.float32).copy()
    out = full[:T * TOK_FEATS].reshape(T, TOK_FEATS)
    valid = out[:, 0] >= 0.0
    valid[:2] = False
    out[valid, 1] = 1.0 - out[valid, 1]
    out[valid, 3] = -out[valid, 3]
    return full
