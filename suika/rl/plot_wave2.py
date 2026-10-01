"""Plot training-metric dashboard: greedy eval + train rollout + progress + throughput.

Usage: python plot_wave2.py [--out PATH]
Reads local mirrors under dqn_runs/<node>/<wave>/<arm>/{eval,episodes_a*}.jsonl
Segments of the same arm (wave3 -> wave3b -> wave3c) are merged into one curve
(env_steps counters continue across the resume).
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
LOCAL_RUNS = os.path.join(PROJ, "dqn_runs")

# label -> list of run segments (plotted as one continuous curve)
WAVE3 = {
    "mlp_deep": [
        "node_d/wave3_20260928/w3_mlp_deep",
        "node_d/wave3b_20260928/w3b_mlp_deep",
    ],
    "tf_deep SAB8": [
        "node_c/wave3_20260928/w3_tf_deep",
        "node_c/wave3b_20260928/w3b_tf_deep",
    ],
    "tf_xl SAB16 56M": [
        "node_c/wave3c_20260929/w3c_tf_xl",
    ],
    "qwen_text (VLM)": [
        "node_d/wave4_20260928/w4_qwen_text",
    ],
}
REF = {"wave2 warm 参考": ["node_d/wave2_20260928/w2_warm_g999"]}

COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]


def read_jsonl(path):
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


def split_segments(rows, min_drop=1.0):
    """Split one actor's append-only stream at process restarts.

    "t" is seconds since that actor process started, so a relaunch resets it
    (and the episode counter restarts from 0). A drop in t marks a segment.
    """
    segs, cur = [], []
    for r in rows:
        if cur and r.get("t", 0.0) < cur[-1].get("t", 0.0) - min_drop:
            segs.append(cur)
            cur = []
        cur.append(r)
    if cur:
        segs.append(cur)
    return segs


def load_segments(keys):
    evals, episodes = [], []
    t_off = 0.0
    for key in keys:
        d = os.path.join(LOCAL_RUNS, key)
        ev = read_jsonl(os.path.join(d, "eval.jsonl"))
        streams = [split_segments(read_jsonl(ef))
                   for ef in sorted(glob.glob(os.path.join(d, "episodes_a*.jsonl")))]
        streams = [s for s in streams if s]
        nseg = max((len(s) for s in streams), default=0)
        # Segments are aligned from the end: an actor that crashed extra times
        # keeps its extra early segments as extra, and every actor's final
        # segment is the arm's final segment. Segment k is offset by the summed
        # duration of all earlier segments, so the concatenated t axis is
        # monotone and comparable across actors.
        dur = [0.0] * nseg
        for s in streams:
            shift = nseg - len(s)
            for j, seg in enumerate(s):
                dur[j + shift] = max(dur[j + shift],
                                     max(r.get("t", 0.0) for r in seg))
        offsets = [t_off + float(sum(dur[:k])) for k in range(nseg)]
        for s in streams:
            shift = nseg - len(s)
            for j, seg in enumerate(s):
                off = offsets[j + shift]
                seen = set()
                for r in seg:
                    rc = (r.get("actor"), r.get("episode"))
                    if rc in seen:
                        continue
                    seen.add(rc)
                    r = dict(r)
                    r["t"] = r.get("t", 0.0) + off
                    episodes.append(r)
        t_off += float(sum(dur))
        evals.extend(ev)
    return dict(evals=evals, episodes=episodes)


def throughput(ev):
    """M env steps/h between consecutive evals; skips restart gaps (>1h)."""
    ev = sorted(ev, key=lambda e: e["env_steps"])
    xs, ys = [], []
    for a, b in zip(ev, ev[1:]):
        dt = b.get("time", 0) - a.get("time", 0)
        ds = b["env_steps"] - a["env_steps"]
        if 180 < dt < 3600 and ds > 0:
            xs.append(b["env_steps"] / 1e6)
            ys.append(ds / dt * 3600 / 1e6)
    return xs, ys


def rolling(x, w):
    k = np.convolve(x, np.ones(w) / w, mode="valid")
    return k


def main():
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(LOCAL_RUNS, "wave3_curves.png")
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10))

    groups = [(WAVE3, COLORS, 1.7), (REF, ["#888888"], 1.1)]
    for group, colors, lw in groups:
        for (label, keys), color in zip(group.items(), colors):
            d = load_segments(keys)
            if d["evals"]:
                ev = sorted(d["evals"], key=lambda e: e["env_steps"])
                xs = np.array([e["env_steps"] for e in ev], dtype=float) / 1e6
                mean = np.array([e["mean"] for e in ev], dtype=float)
                m = xs > 0
                axes[0][0].plot(xs[m], mean[m], color=color, lw=lw, marker=".", ms=3, label=label)
                if group is WAVE3:
                    p25 = np.array([e.get("p25", np.nan) for e in ev], dtype=float)
                    axes[0][0].plot(xs[m], p25[m], color=color, lw=0.8, ls="--", alpha=0.45)
                    mf = np.array([e.get("maxfruit_max", np.nan) for e in ev], dtype=float)
                    ok = m & np.isfinite(mf)
                    if ok.any():
                        axes[1][0].plot(xs[ok], mf[ok], color=color, lw=lw, marker=".", ms=3, label=label)
                    tx, ty = throughput(ev)
                    if tx:
                        axes[1][1].plot(tx, ty, color=color, lw=lw, marker=".", ms=3, label=label)
            if d["episodes"]:
                ss = sorted(d["episodes"], key=lambda e: e["t"])
                t = np.array([e["t"] for e in ss], dtype=float) / 3600.0
                sc = np.array([e["score"] for e in ss], dtype=float)
                w = max(200, len(sc) // 200)
                axes[0][1].plot(t[w - 1:], rolling(sc, w), color=color, lw=lw, label=label)

    axes[0][0].axhline(1844, color="k", ls=":", lw=1)
    axes[0][0].text(2, 1860, "历史 beam 教师 1844", fontsize=8, color="k")

    axes[0][0].set_title("greedy eval 分数 vs 训练量（实线=mean，虚线=p25）")
    axes[0][0].set_ylabel("score")
    axes[0][1].set_title("训练 rollout 单局得分（滚动均值，含 ε 噪声）")
    axes[0][1].set_ylabel("score")
    axes[1][0].set_title("greedy eval 最大果型（10=西瓜）")
    axes[1][0].set_ylabel("max fruit index")
    axes[1][1].set_title("实测吞吐（由 eval 间隔推算）")
    axes[1][1].set_ylabel("M env steps / h")

    for ax in (axes[0][0], axes[1][0], axes[1][1]):
        ax.set_xscale("log")
        ax.set_xlabel("env steps (M, log)")
    axes[0][1].set_xlabel("wall time (h, 段间墙钟已拼接)")
    for row in axes:
        for ax in row:
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("saved ->", out)


if __name__ == "__main__":
    main()
