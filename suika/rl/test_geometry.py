"""Geometry / warm-start checks. Run on a GPU/CPU box, never on the laptop.

  cd suika_dqn && PYTHONPATH=. python test_geometry.py

1. stock-geometry plumbing is a no-op (obs bit-equal with/without geometry spec)
2. geometry actually moves the walls / death line and is deterministic per seed
3. warm start: old (eager, no geo) model  ==  adapted (sdpa + geo) model, exactly
   at init, even with a non-trivial geo block
4. learner mirror keeps the geo block and mirrors only the token block
"""
import hashlib
import sys

import numpy as np
import torch

from paths import setup_engine_path
setup_engine_path()
from config import config  # noqa: E402
from env import DQNEnv, sample_geometry, set_geometry  # noqa: E402
from encoding import GEO_FEATS, TOK_T, geo_vector  # noqa: E402
from model import build_model  # noqa: E402
from model_v2 import adapt_state_dict  # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""),
          flush=True)


def rollout_hash(env, seed, n=25, geom=None):
    obs = env.reset(seed=seed, geom=geom)
    h = hashlib.sha256(np.ascontiguousarray(obs).tobytes())
    rng = np.random.default_rng(seed)
    done = False
    steps = 0
    while not done and steps < n:
        obs, r, done, _ = env.step(int(rng.integers(128)))
        h.update(np.ascontiguousarray(obs).tobytes())
        steps += 1
    return h.hexdigest()[:16], steps


# 1 ---------------------------------------------------------------------------
stock = DQNEnv(seed=None, obs_format="tokens")
a = [rollout_hash(stock, s) for s in (3, 4)]
geo_env = DQNEnv(seed=None, obs_format="tokens",
                 geometry={"width": 448, "height": 505})
b = [rollout_hash(geo_env, s) for s in (3, 4)]
check("stock geometry spec is bit-identical to no spec", a == b, f"{a} vs {b}")

# 2 ---------------------------------------------------------------------------
gv = DQNEnv(seed=None, obs_format="tokens", geo_dim=GEO_FEATS,
            geometry={"width": [400, 550], "height": [500, 720]})
seen = set()
for s in range(40):
    gv.reset(seed=s)
    seen.add(gv.cur_geometry)
    w = config.pad.right - config.pad.left
    assert (w == gv.cur_geometry[0] and config.pad.killy == gv.cur_geometry[1]
            and config.pad.bot - config.pad.killy == gv.cur_geometry[2])
check("width/height vary, killy pinned", len(seen) > 20
      and all(g[1] == 170 for g in seen), f"{len(seen)} distinct")
ws = [g[0] for g in seen]; hs = [g[2] for g in seen]
check("draws inside spec", min(ws) >= 400 and max(ws) <= 550
      and min(hs) >= 500 and max(hs) <= 720,
      f"w[{min(ws)},{max(ws)}] h[{min(hs)},{max(hs)}]")
check("geometry deterministic per seed",
      sample_geometry({"width": [400, 550], "height": [500, 720]}, 7)
      == sample_geometry({"width": [400, 550], "height": [500, 720]}, 7))
o = gv.reset(seed=1, geom=(400, 720))
gvec = o[TOK_T * 5:]
check("geo vector matches pinned board",
      np.allclose(gvec, [(400 - 448) / 75, (720 - 505) / 110, 0.0], atol=1e-6),
      str(gvec))
check("floor actually moved (bot = killy + h)", config.pad.bot == 170 + 720,
      f"bot={config.pad.bot}")
# fruits dropped on the tall board must be able to rest BELOW the stock floor
for _ in range(6):
    obs, _, done, _ = gv.step(64)
    if done:
        break
max_y = max(f["y"] for f in gv.env.get_state()["fruits"])
check("fruits rest below the stock floor on h=720", max_y > 700,
      f"max fruit y={max_y:.0f} (stock floor 675)")
# a fruit dropped at the leftmost column must land inside the narrower box
gv.reset(seed=2, geom=(400, 505))
for _ in range(8):
    obs, _, done, _ = gv.step(0)
    if done:
        break
tok = obs[:TOK_T * 5].reshape(TOK_T, 5)
xs = tok[2:][tok[2:, 0] >= 0][:, 1]
check("fruits stay inside narrow box (x in [0,1])",
      len(xs) > 0 and xs.min() >= -1e-3 and xs.max() <= 1 + 1e-3,
      f"x[{xs.min():.3f},{xs.max():.3f}] n={len(xs)}")
set_geometry(448, 505)

# 3 ---------------------------------------------------------------------------
base = dict(K=128, T=TOK_T, arch="settf", d_tok=128, d_lat=512, n_lat=64,
            stage1=2, stage3=16, tf_heads=8, head_dim=512)
torch.manual_seed(0)
old = build_model({**base, "attn_impl": "eager", "geo_dim": 0}, 800).eval()
new = build_model({**base, "attn_impl": "sdpa", "geo_dim": GEO_FEATS}, 803).eval()
new.load_state_dict(adapt_state_dict(old.state_dict(), new))
rng = np.random.default_rng(0)
x = torch.zeros(6, TOK_T, 5)
x[:, :, 0] = -1.0                       # padding rows
for i in range(6):
    n = int(rng.integers(3, 30))        # rows 0/1 = current/next, rest = board
    x[i, :n, 0] = torch.tensor(np.r_[rng.integers(0, 5, 2),
                                     rng.integers(0, 11, n - 2)],
                               dtype=torch.float32)
    x[i, 2:n, 1:3] = torch.tensor(rng.random((n - 2, 2)), dtype=torch.float32)
    x[i, 2:n, 3:5] = torch.tensor(rng.normal(0, .3, (n - 2, 2)),
                                  dtype=torch.float32)
x = x.reshape(6, -1)
xg = torch.cat([x, torch.tensor(rng.normal(0, 1, (6, GEO_FEATS)),
                                dtype=torch.float32)], dim=1)
with torch.no_grad():
    qo, qn = old.q_values(x), new.q_values(xg)
err = (qo - qn).abs().max().item()
check("warm start is function-preserving (eager/no-geo -> sdpa+geo)",
      err < 1e-4, f"max|dq|={err:.2e} (scale {qo.abs().max().item():.3f})")
# after training-like perturbation the geo path must actually matter
with torch.no_grad():
    new.geo_tok.weight.add_(0.05 * torch.randn_like(new.geo_tok.weight))
    new.in_proj.weight[:, -3:].add_(0.05 * torch.randn_like(new.in_proj.weight[:, -3:]))
    q2 = new.q_values(xg)
check("geo path is live once weights move", (q2 - qn).abs().max().item() > 1e-4)

# 4 ---------------------------------------------------------------------------
from learner import Learner  # noqa: E402
L = Learner.__new__(Learner)
L.tokens, L.T, L.K, L.boundary, L.max_fruits = True, TOK_T, 128, True, 80
xm = xg.clone()
om, am = L._mirror(xm.clone(), torch.tensor([0, 5, 127, 64, 1, 2]))
check("mirror keeps geo block", torch.equal(om[:, TOK_T * 5:], xg[:, TOK_T * 5:]))
check("mirror action flips", am.tolist() == [127, 122, 0, 63, 126, 125])
back, _ = L._mirror(om.clone(), am)
check("mirror is an involution", torch.allclose(back, xg, atol=1e-6))

# 5 ---------------------------------------------------------------------------
# action-sequence storage: seed + board + actions must reproduce the game
from replay_actions import replay_record  # noqa: E402
rng = np.random.default_rng(11)
spec = {"width": [400, 550], "height": [500, 720]}
renv = DQNEnv(seed=None, obs_format="tokens", geo_dim=GEO_FEATS, geometry=spec)
good = 0
for sd in (5, 6, 7):
    obs = renv.reset(seed=sd)
    acts, done = bytearray(), False
    while not done and len(acts) < 400:
        a = int(rng.integers(128))
        acts.append(a)
        obs, r, done, _ = renv.step(a)
    rec = {"seed": sd, "w": renv.cur_geometry[0], "h": renv.cur_geometry[2],
           "score": float(renv.score), "moves": len(acts), "actions": acts.hex()}
    good += int(replay_record(rec)["match"])
check("replay reproduces score for seed+board+actions", good == 3, f"{good}/3")
# wave5 records (w + killy, fixed 675 floor) must still replay
set_geometry(448, 505)
r5 = DQNEnv(seed=None, obs_format="tokens")
obs = r5.reset(seed=9)
acts, done = bytearray(), False
while not done and len(acts) < 300:
    a = int(rng.integers(128))
    acts.append(a)
    obs, r, done, _ = r5.step(a)
rec5 = {"seed": 9, "w": 448, "killy": 170, "score": float(r5.score),
        "moves": len(acts), "actions": acts.hex()}
check("wave5-format record still replays", replay_record(rec5)["match"])

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
