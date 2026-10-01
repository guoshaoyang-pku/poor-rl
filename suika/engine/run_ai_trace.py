"""Run the heuristic-lookahead Suika agent for one game, record a full trace,
and produce visualizations (frames, GIF, score curve, final-state image).

Usage:
    python3 run_ai_trace.py
"""
import csv
import json
import os
import sys

# Resolve paths BEFORE importing suika_env (its import does an os.chdir to the
# suika project root, which would otherwise break our relative paths).
_HERE = os.path.dirname(os.path.abspath(__file__))          # .../合成大西瓜/suika
_BASE = os.path.dirname(_HERE)                              # .../合成大西瓜
_TRACES = os.path.join(_BASE, "traces")
_FRAMES = os.path.join(_TRACES, "frames")
os.makedirs(_FRAMES, exist_ok=True)

sys.path.insert(0, os.path.join(_HERE, "part2"))

import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from suika_env import SuikaEnv          # triggers headless setup + chdir
from ai_agent import HeuristicLookaheadAgent
from config import config

SEED = 0
MAX_STEPS = 150
NUM_COLUMNS = 14
FRAME_EVERY = 1                          # save a frame every N steps


def stack_height(fruits):
    """Height of the tallest point of the pile, measured up from the floor.
    0 == empty board; larger == more dangerous (closer to the top line)."""
    if not fruits:
        return 0.0
    top_edge = min(f["y"] - f["radius"] for f in fruits)
    return float(config.pad.bot - top_edge)


def x_to_column(x, lo, hi, num):
    if hi <= lo:
        return 0
    col = int(np.clip(round((x - lo) / (hi - lo) * num - 0.5), 0, num - 1))
    return col


def main():
    env = SuikaEnv(seed=SEED)
    state = env.reset(seed=SEED)
    agent = HeuristicLookaheadAgent(num_columns=NUM_COLUMNS, seed=SEED)

    lo, hi = config.pad.left, config.pad.right
    trace = []
    frame_files = []
    max_fruit_type = max((f["type"] for f in state["fruits"]), default=0)

    # initial frame
    img0 = env.render(mode="rgb_array")
    f0 = os.path.join(_FRAMES, "step_000.png")
    imageio.imwrite(f0, img0)
    frame_files.append((0, f0))

    done = False
    step = 0
    while not done and step < MAX_STEPS:
        pre = env.get_state()
        cur = pre["current"]
        nxt = pre["next"]
        x = agent.decide(pre)
        col = x_to_column(x, lo, hi, NUM_COLUMNS)

        state, reward, done, info = env.step(x)
        step += 1

        fruits = state["fruits"]
        h = stack_height(fruits)
        max_fruit_type = max([max_fruit_type] + [f["type"] for f in fruits])

        rec = {
            "step": step,
            "current_type": int(cur["type"]),
            "current_name": cur["name"],
            "next_type": int(nxt["type"]),
            "next_name": nxt["name"],
            "drop_x": round(float(x), 2),
            "column": int(col),
            "reward": float(reward),
            "score": int(state["score"]),
            "num_fruits": len(fruits),
            "max_height": round(h, 1),
            "game_over": bool(done),
        }
        trace.append(rec)
        print(f"step {step:3d} cur={cur['name']:11s} next={nxt['name']:11s} "
              f"col={col:2d} x={x:6.1f} r={reward:5.0f} score={state['score']:6d} "
              f"fruits={len(fruits):2d} h={h:5.1f} done={done}")

        if step % FRAME_EVERY == 0 or done:
            img = env.render(mode="rgb_array")
            fp = os.path.join(_FRAMES, f"step_{step:03d}.png")
            imageio.imwrite(fp, img)
            frame_files.append((step, fp))

    final_score = env.score
    final_name = config.fruit_names[max_fruit_type]
    print(f"\nDONE: steps={step} final_score={final_score} "
          f"max_fruit={final_name} lookahead_used={agent.lookahead_used} "
          f"lookahead_failures={agent.lookahead_failures}")

    # ----- write trace.jsonl + trace.csv ------------------------------- #
    jsonl_path = os.path.join(_TRACES, "trace.jsonl")
    with open(jsonl_path, "w") as f:
        for rec in trace:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    csv_path = os.path.join(_TRACES, "trace.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
        w.writeheader()
        w.writerows(trace)

    # ----- GIF (downscaled for size) ----------------------------------- #
    gif_path = os.path.join(_TRACES, "playthrough.gif")
    gif_frames = []
    for _, fp in frame_files:
        im = imageio.imread(fp)
        gif_frames.append(im[::2, ::2])          # half resolution
    imageio.mimsave(gif_path, gif_frames, duration=0.35, loop=0)

    # ----- score / height curves --------------------------------------- #
    steps = [r["step"] for r in trace]
    scores = [r["score"] for r in trace]
    heights = [r["max_height"] for r in trace]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(steps, scores, color="tab:red", lw=2, label="score")
    ax1.set_xlabel("step")
    ax1.set_ylabel("cumulative score", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(steps, heights, color="tab:blue", lw=1.5, ls="--",
             label="max stack height")
    ax2.set_ylabel("max stack height (px)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    plt.title(f"Suika AI (heuristic + 1-step lookahead)  "
              f"final score={final_score}, max fruit={final_name}")
    fig.tight_layout()
    curve_path = os.path.join(_TRACES, "score_curve.png")
    fig.savefig(curve_path, dpi=120)
    plt.close(fig)

    # ----- final-state image ------------------------------------------- #
    final_img = env.render(mode="rgb_array")
    final_path = os.path.join(_TRACES, "final_state.png")
    imageio.imwrite(final_path, final_img)

    summary = {
        "seed": SEED,
        "steps": step,
        "final_score": int(final_score),
        "max_fruit": final_name,
        "max_fruit_type": int(max_fruit_type),
        "num_columns": NUM_COLUMNS,
        "lookahead_used": agent.lookahead_used,
        "lookahead_failures": agent.lookahead_failures,
        "num_frames": len(frame_files),
        "files": {
            "trace_jsonl": jsonl_path,
            "trace_csv": csv_path,
            "gif": gif_path,
            "score_curve": curve_path,
            "final_state": final_path,
            "frames_dir": _FRAMES,
        },
    }
    with open(os.path.join(_TRACES, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nSUMMARY:\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
