"""Wave6 status panel: eval curves + stability dashboard from pulled data.

Reads dqn_runs/node_c_{2,3}/wave6_20260930/{w6_var,w6_h720_fixed}/ (filled by
pull_wave6.sh) and writes dqn_runs/wave6_status.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "dqn_runs")
ARMS = [("w6_h720_fixed", "node_a", "#d62728", "448x720 fixed"),
        ("w6_var", "node_b", "#1f77b4", "variable grid"),
        ("w6_var_xl", "node_f", "#2ca02c", "var XL (stage3=24)"),
        ("w6_max_fixed", "node_b", "#ff7f0e", "550x720 fixed")]


def load(arm, node, name):
    p = os.path.join(ROOT, node, "wave6_20260930", arm, name)
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p)]


fig, axes = plt.subplots(1, 3, figsize=(21, 6.2))

ax = axes[0]
for arm, node, color, label in ARMS:
    ev = load(arm, node, "eval.jsonl")
    if not ev:
        continue
    ax.plot([r["grad_steps"] for r in ev], [r["mean"] for r in ev],
            "-o", color=color, ms=4, label=label)
    if arm in ("w6_var", "w6_var_xl"):
        cells = sorted(ev[-1].get("grid", {}).keys())
        for c in cells:
            ys = [r["grid"][c]["mean"] for r in ev if c in r.get("grid", {})]
            xs = [r["grad_steps"] for r in ev if c in r.get("grid", {})]
            hi = c.endswith("720")
            ax.plot(xs, ys, "-", color=color, lw=0.8,
                    alpha=0.75 if hi else 0.25)
        ax.plot([], [], "-", color=color, lw=0.8, alpha=0.75,
                label=f"{label} cells h=720")
        ax.plot([], [], "-", color=color, lw=0.8, alpha=0.25,
                label=f"{label} cells h=500/610")
ax.axhline(2400, color="gray", ls=":", lw=1)
ax.text(2, 2450, "stock-board (k170) all-time ceiling ~2400", fontsize=8,
        color="gray")
ax.set_xlabel("grad steps")
ax.set_ylabel("greedy eval mean score")
ax.set_title("wave6 eval vs training progress")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)

ax = axes[1]
var_ev = load("w6_var", "node_b", "eval.jsonl")
if var_ev and "grid" in var_ev[-1]:
    cells = sorted(var_ev[-1]["grid"].keys())
    for i, r in enumerate(var_ev):
        ys = [r["grid"][c]["mean"] for c in cells if c in r["grid"]]
        ax.plot([cells.index(c) for c in cells if c in r["grid"]], ys,
                "-o", ms=3, color="#1f77b4",
                alpha=0.25 + 0.75 * i / max(1, len(var_ev) - 1), lw=1)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(cells, rotation=45, ha="right", fontsize=8)
    ax.set_title("w6_var per-board mean (light -> latest)")
    ax.set_ylabel("greedy eval mean")
    ax.grid(alpha=0.3)

ax = axes[2]
for arm, node, color, label in ARMS:
    m = load(arm, node, "metrics.jsonl")
    if not m:
        continue
    ax.plot([r["grad_steps"] for r in m], [r["q_mean"] for r in m],
            "-", color=color, lw=0.8, label=f"{label} q_mean")
ax.axhline(300, color="gray", ls=":", lw=1)
ax.set_xlabel("grad steps")
ax.set_ylabel("q_mean")
ax.set_title("value stability (wave5 diverged to 2816 by step 2400)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

import time
fig.suptitle("wave6 status @ " + time.strftime("%m-%d %H:%M"), fontsize=12)
fig.tight_layout()
out = os.path.join(ROOT, "wave6_status.png")
fig.savefig(out, dpi=130)
print(out)
