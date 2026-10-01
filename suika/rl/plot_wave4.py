"""Kill-line ablation dashboard (wave4): killy 170 vs 200 on the tf_xl recipe.

The two k200 arms fork off the k170 baseline at the same checkpoint
(w4_k200_cont is seeded with w3c_tf_xl step3801, env_steps 482.77M), so on a
shared env-steps axis the continue arm traces the same trajectory as the
baseline up to the fork and then diverges. That fork is the experiment.

Row 2 adds the from-scratch-vs-baseline matched-progress view and the critic
diagnostics (q_mean / loss), which is what answers "did the critic break?"

Usage: python plot_wave4.py [--out PATH]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB",
                                          "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

from plot_wave2 import LOCAL_RUNS, load_segments, read_jsonl, rolling

# label -> local mirror segments, drawn as one continuous curve
ARMS = {
    "tf_xl k170 基线": (["node_c/wave3c_20260929/w3c_tf_xl"], "#d62728", 1.7),
    "tf_xl k200 续训": (["node_c/wave4_20260929/w4_k200_cont"], "#1f77b4", 1.9),
    "tf_xl k200 从头": (["node_a/wave4_20260929/w4_k200_scratch"], "#2ca02c", 1.9),
}
FORK_ENV_STEPS = 482_771_285      # w3c step3801, the k170->k200 switch point
FORK_GRAD = 3801


def load_metrics(keys):
    out = []
    for key in keys:
        out.extend(read_jsonl(os.path.join(LOCAL_RUNS, key, "metrics.jsonl")))
    return sorted(out, key=lambda r: r.get("grad_steps", 0))


def ema(x, alpha):
    """Exponential moving average; jumps (actor/learner restarts) still show."""
    out = np.empty_like(x, dtype=float)
    acc = x[0]
    for i, v in enumerate(x):
        acc = alpha * v + (1.0 - alpha) * acc
        out[i] = acc
    return out


def main():
    out = (sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv
           else os.path.join(LOCAL_RUNS, "wave4_killy.png"))
    fig, axes = plt.subplots(3, 3, figsize=(21, 14))
    (ax_eval, ax_zoom, ax_match,
     ax_roll, ax_fruit, ax_q,
     ax_loss, ax_gn, ax_fork) = axes.ravel()
    fork = FORK_ENV_STEPS / 1e6

    for label, (keys, color, lw) in ARMS.items():
        d = load_segments(keys)
        if d["evals"]:
            ev = sorted(d["evals"], key=lambda e: e["env_steps"])
            xs = np.array([e["env_steps"] for e in ev], float) / 1e6
            mean = np.array([e["mean"] for e in ev], float)
            m = xs > 0
            for ax in (ax_eval, ax_zoom, ax_match):
                ax.plot(xs[m], mean[m], color=color, lw=lw, marker=".", ms=3,
                        label=label)
            p25 = np.array([e.get("p25", np.nan) for e in ev], float)
            ax_eval.plot(xs[m], p25[m], color=color, lw=0.8, ls="--",
                         alpha=0.45)
            mf = np.array([e.get("maxfruit_max", np.nan) for e in ev], float)
            ok = m & np.isfinite(mf)
            if ok.any():
                ax_fruit.plot(xs[ok], mf[ok], color=color, lw=lw,
                              marker=".", ms=3, label=label)
            mv = np.array([e.get("moves_mean", np.nan) for e in ev], float)
            ok = m & np.isfinite(mv)
            if ok.any():
                ax_fork.plot(xs[ok], mv[ok], color=color, lw=lw,
                             marker=".", ms=3, label=label)
                ax_fork.set_ylim(bottom=0)
            if xs[m].size:
                print(f"{label:22s} evals={int(m.sum()):4d} "
                      f"steps={xs[m].min():.0f}-{xs[m].max():.0f}M "
                      f"last_mean={mean[m][-1]:.0f}")
        if d["episodes"]:
            ss = sorted(d["episodes"], key=lambda e: e["t"])
            t = np.array([e["t"] for e in ss], float) / 3600.0
            sc = np.array([e["score"] for e in ss], float)
            w = max(50, len(sc) // 200)
            ax_roll.plot(t[w - 1:], rolling(sc, w), color=color, lw=lw,
                         label=label)

        mt = load_metrics(keys)
        if mt:
            g = np.array([r["grad_steps"] for r in mt], float)
            q = np.array([r["q_mean"] for r in mt], float)
            ls = ema(np.array([r["loss"] for r in mt], float), 0.05)
            gm = ema(np.array([r["grad_norm"] for r in mt], float), 0.05)
            for ax, y, ttl in ((ax_q, q, "q_mean"), (ax_loss, ls, "loss"),
                               (ax_gn, gm, "grad_norm")):
                ax.plot(g, y, color=color, lw=lw, label=label)
                ax.set_yscale("log")
            print(f"{label:22s} metrics={len(mt):5d} last q_mean={q[-1]:8.2f} "
                  f"loss_ema={ls[-1]:6.3f} gn_ema={gm[-1]:5.2f} "
                  f"| loss@1k/2k/3k/end="
                  f"{ls[999] if len(ls) > 999 else float('nan'):.2f}/"
                  f"{ls[1999] if len(ls) > 1999 else float('nan'):.2f}/"
                  f"{ls[2999] if len(ls) > 2999 else float('nan'):.2f}")

    # panel 1: full greedy-eval history
    ax_eval.axvline(fork, color="k", ls=":", lw=1.2)
    ax_eval.text(fork + 4, ax_eval.get_ylim()[0] + 60,
                 f"分叉点 {fork:.0f}M\n(killy 170→200)", fontsize=8)
    ax_eval.axhline(1844, color="gray", ls=":", lw=1)
    ax_eval.text(6, 1860, "历史 beam 教师 1844", fontsize=8, color="gray")
    ax_eval.set_title("greedy eval 分数 vs 训练量（实线=mean，细虚线=p25）")
    ax_eval.set_xlabel("env steps (M)")
    ax_eval.set_ylabel("score")

    # panel 2: the fork, zoomed
    ax_zoom.axvline(fork, color="k", ls=":", lw=1.2)
    ax_zoom.set_title("分叉区放大（k200 续训 vs k170 基线，同起点）")
    ax_zoom.set_xlabel("env steps (M)")
    ax_zoom.set_ylabel("score")
    ax_zoom.set_xlim(fork - 60, fork + 220)

    # panel 3: from-scratch arm vs the baseline at matched training volume
    ax_match.axvline(fork, color="k", ls=":", lw=0.8, alpha=0.4)
    ax_match.set_title("从头臂 vs 基线：对齐训练量（0–45M）")
    ax_match.set_xlabel("env steps (M)")
    ax_match.set_ylabel("score")
    ax_match.set_xlim(0, 45)

    # panel 4: training rollouts (wall time, each arm from its own start)
    ax_roll.set_title("训练 rollout 单局得分（滚动均值，含 ε 噪声）")
    ax_roll.set_xlabel("wall time (h, 各臂自启动起)")
    ax_roll.set_ylabel("score")

    # panel 5: max merged fruit
    ax_fruit.set_title("greedy eval 最大果型（10=西瓜）")
    ax_fruit.set_xlabel("env steps (M)")
    ax_fruit.set_ylabel("max fruit index")

    # panels 6-8: critic diagnostics -- does the critic break?
    for ax, ttl in ((ax_q, "critic q_mean（EMA 无，原始值）"),
                    (ax_loss, "TD loss（EMA 0.05）"),
                    (ax_gn, "grad norm（EMA 0.05）")):
        ax.axvline(FORK_GRAD, color="k", ls=":", lw=1.2)
        ax.set_title(ttl)
        ax.set_xlabel("grad steps")
        ax.set_ylabel("value (log)")

    # panel 9: episode length -- the mechanism behind a lower score
    ax_fork.axvline(fork, color="k", ls=":", lw=1.2)
    ax_fork.set_title("greedy eval 平均步数（每步收益 × 步数 = 分数）")
    ax_fork.set_xlabel("env steps (M)")
    ax_fork.set_ylabel("moves")

    for ax in (ax_eval, ax_zoom, ax_match, ax_roll, ax_fruit, ax_q, ax_loss,
               ax_gn):
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("saved ->", out)


if __name__ == "__main__":
    main()
