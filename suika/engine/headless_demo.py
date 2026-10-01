"""Demo: drive the Suika game headlessly with a simple random policy.

Run from anywhere with the `suika` conda env:
    python headless_demo.py
This proves an AI model can take over the game without any window.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "part2"))

import numpy as np
from suika_env import SuikaEnv


def main(steps=80, seed=42):
    env = SuikaEnv(seed=seed)
    state = env.reset(seed=seed)
    policy_rng = np.random.default_rng(0)
    total_reward = 0.0

    print(f"play area x in [{env.play_left}, {env.play_right}]  "
          f"fruits on board start={len(state['fruits'])}")
    for i in range(steps):
        # naive policy: drop the current fruit at a random valid x
        x = policy_rng.uniform(env.play_left, env.play_right)
        state, reward, done, info = env.step(x)
        total_reward += reward
        print(f"step {i:3d}  drop_x={x:7.1f}  reward={reward:4.0f}  "
              f"score={info['score']:5d}  on_board={len(state['fruits']):2d}  "
              f"current={state['current']['name']:>10s}  "
              f"next={state['next']['name']:>10s}")
        if done:
            print(f">>> GAME OVER at step {i}")
            break

    # show that a full structured state is available to a model
    print("\nExample structured state (first 3 fruits):")
    for f in state["fruits"][:3]:
        print("  ", f)

    # render a frame to a numpy array (no window) to prove headless rendering
    frame = env.render(mode="rgb_array")
    print(f"\nrender(mode='rgb_array') -> numpy frame shape={frame.shape}, "
          f"dtype={frame.dtype}")
    print(f"\nFINISHED  final_score={env.score}  total_reward={total_reward:.0f}  "
          f"steps_played={env.steps_taken}")


if __name__ == "__main__":
    main()
