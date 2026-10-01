"""Verify the two-watermelon merge rule: both vanish and score += 66."""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "part2"))
os.chdir(_THIS_DIR)

import numpy as np
from suika_env import SuikaEnv
from particle import Particle
from config import config


def run_case(rule, expect_score, expect_watermelons_gone):
    config.rules.two_watermelon = rule
    env = SuikaEnv(seed=0)
    r = config[10, "radius"]
    # Two watermelons resting side by side on the floor, just touching.
    y = config.pad.bot - r - 1
    Particle(np.array([640.0 - r, y]), 10, env.space)
    Particle(np.array([640.0 + r, y]), 10, env.space)
    score_before = env.score
    for _ in range(120):
        env.space.step(1 / 60)
    gained = env.score - score_before
    n_watermelon = sum(1 for p in env._live_particles() if p.n == 10)
    print(f"rule={rule!r}: score gained={gained}, watermelons left={n_watermelon}")
    assert gained == expect_score, f"expected +{expect_score}, got +{gained}"
    if expect_watermelons_gone:
        assert n_watermelon == 0, "watermelons should have vanished"
    else:
        assert n_watermelon == 2, "watermelons should have survived"
    print("  OK")


# New official rule: both disappear, +66 points.
run_case("merge_disappear_score", expect_score=66, expect_watermelons_gone=True)
# Legacy rule: plain collision, no score, both stay.
run_case("no_merge_no_score_keep_both", expect_score=0, expect_watermelons_gone=False)

# Sanity: a normal merge (two cherries) still works under the new rule.
config.rules.two_watermelon = "merge_disappear_score"
env = SuikaEnv(seed=1)
r = config[0, "radius"]
y = config.pad.bot - r - 1
Particle(np.array([640.0 - 0.95 * r, y]), 0, env.space)
Particle(np.array([640.0 + 0.95 * r, y]), 0, env.space)
for _ in range(120):
    env.space.step(1 / 60)
strawberries = [p for p in env._live_particles() if p.n == 1]
assert env.score == 1 and len(strawberries) == 1, (
    f"normal merge broken: score={env.score}, strawberries={len(strawberries)}")
print("normal cherry->strawberry merge still works. ALL TESTS PASSED")
