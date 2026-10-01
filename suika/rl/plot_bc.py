"""Plot the w4 BC (Qwen) pretraining dashboard.

Usage: python plot_bc.py [--log PATH] [--metrics PATH] [--out PATH]
Dense per-50-step curves come from launch.log ([bc] step ...), the
teacher-agreement / distill-error gate curves come from metrics.jsonl
([bc-val] cadence = val_every grad steps).
"""
import argparse
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DEFAULT_RUN = os.path.join(PROJ, "dqn_runs", "t1_2", "wave4_20260929", "w4_bc_qwen")

STEP_RE = re.compile(
    r"\[bc\] step (\d+)/(\d+) loss=([\d.]+) pi=([\d.]+) qd=([\d.]+) "
    r"td=([\d.]+) ([\d.]+)s")
VAL_RE = re.compile(r"\[bc-val\] agree_pi=([\d.]+) agree_q=([\d.]+) val_qd=([\d.]+)")


def parse_log(path):
    rows = []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                m = STEP_RE.search(line)
                if m:
                    rows.append(dict(
                        step=int(m.group(1)), total=int(m.group(2)),
                        loss=float(m.group(3)), l_pi=float(m.group(4)),
                        l_qd=float(m.group(5)), l_td=float(m.group(6)),
                        step_s=float(m.group(7))))
                    continue
                m = VAL_RE.search(line)
                if m and rows:
                    rows[-1].update(agree_pi=float(m.group(1)),
                                    agree_q=float(m.group(2)),
                                    val_qd=float(m.group(3)))
    except OSError:
        pass
    return rows


def parse_metrics(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError:
        pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.join(DEFAULT_RUN, "launch.log"))
    ap.add_argument("--metrics", default=os.path.join(DEFAULT_RUN, "metrics.jsonl"))
    ap.add_argument("--out", default=os.path.join(PROJ, "dqn_runs", "wave4_bc_curves.png"))
    args = ap.parse_args()

    dense = parse_log(args.log)
    val = parse_metrics(args.metrics) or [
        r for r in dense if "agree_pi" in r]
    if not dense and not val:
        print(f"[plot_bc] no data yet: {args.log}")
        return 1

    fig, ax = plt.subplots(2, 2, figsize=(13, 7.5))
    st = [r["step"] for r in dense]

    if dense:
        ax[0, 0].plot(st, [r["l_pi"] for r in dense], color="#d62728", lw=1.2, label="L_pi (CE)")
        ax[0, 0].plot(st, [r["l_qd"] for r in dense], color="#1f77b4", lw=1.2, label="L_qd (distill)")
        ax[0, 0].plot(st, [r["l_td"] for r in dense], color="#2ca02c", lw=1.2, label="L_td")
        ax[0, 0].plot(st, [r["loss"] for r in dense], color="#555", lw=1.0, ls="--", label="total")
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_title("loss components")
    ax[0, 0].set_xlabel("grad step")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(alpha=.3)

    if val:
        vs = [r["grad_steps"] for r in val]
        ax[0, 1].plot(vs, [r["agree_pi"] * 100 for r in val], "o-", color="#d62728",
                      ms=3, lw=1.2, label="agree_pi (π head vs teacher)")
        ax[0, 1].plot(vs, [r["agree_q"] * 100 for r in val], "s-", color="#1f77b4",
                      ms=3, lw=1.2, label="agree_q (argmax Q vs teacher)")
        ax[0, 1].axhline(85, color="#888", ls=":", lw=1.2)
        ax[0, 1].text(vs[0], 86, "gate 85%", fontsize=8, color="#666")
        ax[0, 1].set_title("teacher agreement (%)")
        ax[0, 1].set_xlabel("grad step")
        ax[0, 1].legend(fontsize=8, loc="lower right")
        ax[0, 1].grid(alpha=.3)

        ax[1, 0].plot(vs, [r["val_qd"] for r in val], "o-", color="#ff7f0e", ms=3, lw=1.2)
        ax[1, 0].set_title("val_qd — smooth-L1 vs teacher Q128 (raw scale)")
        ax[1, 0].set_xlabel("grad step")
        ax[1, 0].grid(alpha=.3)

    if dense:
        s = np.array(st, dtype=float)
        t = np.array([r["step_s"] for r in dense], dtype=float)
        ax[1, 1].plot(s, t, color="#9467bd", lw=1.2, label="s / grad step")
        ax[1, 1].set_ylabel("s per step")
        ax[1, 1].set_xlabel("grad step")
        ax2 = ax[1, 1].twinx()
        w = t[s > 0]
        if len(w):
            ax2.plot(s, w.mean() and 1024.0 / np.clip(t, 1e-6, None), color="#2ca02c",
                     lw=1.0, alpha=.7, label="samples/s (batch 1024)")
            ax2.set_ylabel("samples/s")
            ax2.tick_params(axis="y", labelcolor="#2ca02c")
        ax[1, 1].set_title("throughput")
        h1, l1 = ax[1, 1].get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax[1, 1].legend(h1 + h2, l1 + l2, fontsize=8)
        ax[1, 1].grid(alpha=.3)

    fig.suptitle(f"w4 BC (Qwen3.5-0.8B + LoRA) pretrain — {len(dense)} logged steps, "
                 f"last={dense[-1]['step'] if dense else 0}", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"[plot_bc] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
