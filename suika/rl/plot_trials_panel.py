"""One panel holding every trial's core info (wave1 -> wave4).

Nine curves on top answer "does capacity/sample efficiency buy anything?":
  row 1  greedy eval vs env steps / vs grad steps / vs wall-clock hours
  row 2  actor-side env throughput, learner-side sample throughput,
         and best score vs cumulative training samples consumed
  row 3  watermelon progress (max fruit type), episode length, P(score>=2000)

The full-width bottom table is the per-trial ledger (arch, params, batch,
steps, throughput, score, host) built from dqn_runs/trials.json.

Arms trained at the shorter death line (killy=200) are drawn dashed + hollow:
their scores are NOT comparable to the killy=170 arms and the xeuqal sweep
showed the same weights lose ~620 points when moved to killy=200.

Usage: python plot_trials_panel.py [--out PATH]
"""
import json
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

# label -> (segments, colour, killy, arch tag)
ARMS = {
    "MLP 13.4M deep": (["node_d/wave3_20260928/w3_mlp_deep",
                        "node_d/wave3b_20260928/w3b_mlp_deep"],
                       "#1f77b4", 170),
    "tf_deep 8.2M (SAB8)": (["node_c/wave3_20260928/w3_tf_deep",
                             "node_c/wave3b_20260928/w3b_tf_deep"],
                            "#ff7f0e", 170),
    "tf_xl 55.9M (SAB16)": (["node_c/wave3c_20260929/w3c_tf_xl"],
                            "#d62728", 170),
    "tf_base 5.1M": (["node_d/wave3_20260928/w3_tf_base"],
                     "#9467bd", 170),
    "w2 warm MLP 2.5M ($\\gamma$=0.999)":
        (["node_d/wave2_20260928/w2_warm_g999"], "#7f7f7f", 170),
    "tf_xl k200 续训": (["node_c/wave4_20260929/w4_k200_cont"],
                        "#17becf", 200),
    "tf_xl k200 从头": (["node_a/wave4_20260929/w4_k200_scratch"],
                        "#2ca02c", 200),
}
DASHED = {"tf_xl k200 续训", "tf_xl k200 从头"}

DQN_BASELINE = 1604      # mattjacobs30-dqn, their dynamics, 160 steps
WATERMELON = 3000        # community-elite line: reached a watermelon


def load(keys):
    """Merged eval stream with a continuous hour axis across resume segments."""
    ev, hours = [], 0.0
    for key in keys:
        rows = sorted(read_jsonl(os.path.join(LOCAL_RUNS, key, "eval.jsonl")),
                      key=lambda r: r.get("time", 0))
        if not rows:
            continue
        t0 = rows[0].get("time", 0)
        span = (rows[-1].get("time", t0) - t0) / 3600.0
        for r in rows:
            r = dict(r)
            r["h"] = hours + (r.get("time", t0) - t0) / 3600.0
            ev.append(r)
        hours += span
    ev.sort(key=lambda r: (r.get("grad_steps", 0), r.get("h", 0)))
    return ev


def ema(x, a=0.35):
    out, acc = np.empty(len(x)), x[0]
    for i, v in enumerate(x):
        acc = a * v + (1 - a) * acc
        out[i] = acc
    return out


def rpad(x, w):
    """Centred moving average that keeps the original length."""
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    y = rolling(x, w)
    return np.concatenate([np.full(w // 2, y[0]), y,
                           np.full(len(x) - len(y) - w // 2, y[-1])])


def curve(ax, xs, ys, label, color, dashed=False, lw=2.0):
    (ls, mfc, mew, ms) = ("--", "none", 1.4, 5) if dashed else ("-", color, 0, 0)
    ax.plot(xs, ys, ls, color=color, lw=lw, label=label,
            marker="o", ms=ms, mfc=mfc, mew=mew, markevery=max(1, len(xs) // 14))


def main():
    out = (sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv
           else os.path.join(LOCAL_RUNS, "trials_panel.png"))
    trials = json.load(open(os.path.join(LOCAL_RUNS, "trials.json")))
    by_name = {t["trial"]: t for t in trials}

    loaded = {lab: load(seg) for lab, (seg, _, _) in ARMS.items()}

    fig = plt.figure(figsize=(24, 27))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.35],
                          hspace=0.30, wspace=0.20)
    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(3)]
    (ax_env, ax_gs, ax_h,
     ax_tp, ax_samp, ax_compute,
     ax_fruit, ax_moves, ax_p2k) = axes

    def draw(ax, xkey):
        for lab, (_, color, _killy) in ARMS.items():
            ev = loaded[lab]
            if not ev:
                continue
            xs = np.asarray([r.get(xkey, 0) for r in ev], dtype=float)
            ys = np.asarray([r.get("mean", 0) for r in ev], dtype=float)
            if len(ys) > 3:
                ys = ema(ys)
            curve(ax, xs, ys, lab, color, dashed=(lab in DASHED))

    # --- row 1: score vs compute axis -----------------------------------
    draw(ax_env, "env_steps")
    ax_env.set_xscale("log")
    ax_env.set_xlabel("环境步数 env_steps (log)")
    ax_env.set_ylabel("greedy eval 平均分 (16 seeds)")
    ax_env.set_title("A. 分数 vs 环境步数（数据量）", fontsize=13, fontweight="bold")
    ax_env.axhline(DQN_BASELINE, color="k", ls=":", lw=1.4)
    ax_env.text(2e5, DQN_BASELINE + 55, f"DQN 基线 {DQN_BASELINE}",
                fontsize=9, color="k")
    ax_env.axhline(WATERMELON, color="green", ls=":", lw=1.4)
    ax_env.text(2e5, WATERMELON + 55, "社区精英线 ~3000（出西瓜）",
                fontsize=9, color="green")
    ax_env.legend(fontsize=8.5, loc="upper left")

    draw(ax_gs, "grad_steps")
    ax_gs.set_xscale("log")
    ax_gs.set_xlabel("梯度步数 grad_steps (log)")
    ax_gs.set_ylabel("greedy eval 平均分")
    ax_gs.set_title("B. 分数 vs 梯度步数（优化器样本效率）", fontsize=13,
                    fontweight="bold")
    ax_gs.legend(fontsize=8.5, loc="upper left")

    draw(ax_h, "h")
    ax_h.set_xlabel("墙钟小时（该 trial 本地 eval 时间跨度）")
    ax_h.set_ylabel("greedy eval 平均分")
    ax_h.set_title("C. 分数 vs 墙钟时间（效率）", fontsize=13, fontweight="bold")
    ax_h.legend(fontsize=8.5, loc="upper left")

    # --- row 2: throughput ---------------------------------------------
    names = [t["trial"] for t in trials]
    colors = {"mlp": "#1f77b4", "settf": "#d62728", "qwen": "#8c564b"}
    order = sorted(trials, key=lambda t: -(t.get("env_sps") or 0))
    x = np.arange(len(order))
    ax_tp.bar(x, [t.get("env_sps") or 0 for t in order],
              color=[colors.get(t["arch"], "#999") for t in order])
    ax_tp.set_yscale("log")
    ax_tp.set_xticks(x)
    ax_tp.set_xticklabels([t["trial"] for t in order], rotation=90, fontsize=6)
    ax_tp.set_ylabel("env steps / s (learner 侧稳态)")
    ax_tp.set_title("D. 采样吞吐：actor 送入 learner 的步/秒", fontsize=13,
                    fontweight="bold")
    ax_tp.set_ylim(50, 1.1e5)
    for xi, t in zip(x, order):
        v = t.get("env_sps") or 0
        ax_tp.text(xi, v * 1.25, f"{v/1000:.0f}k", ha="center", fontsize=6,
                   rotation=90)

    order2 = sorted(trials, key=lambda t: -((t.get("grad_sps") or 0)
                                            * (t.get("batch") or 1)))
    x2 = np.arange(len(order2))
    samp = [((t.get("grad_sps") or 0) * (t.get("batch") or 1)) for t in order2]
    ax_samp.bar(x2, samp, color=[colors.get(t["arch"], "#999") for t in order2])
    ax_samp.set_yscale("log")
    ax_samp.set_xticks(x2)
    ax_samp.set_xticklabels([t["trial"] for t in order2], rotation=90, fontsize=6)
    ax_samp.set_ylabel("训练样本 / s = grad/s x batch")
    ax_samp.set_title("E. 优化吞吐：每秒钟过多少条训练样本", fontsize=13,
                      fontweight="bold")
    for xi, t, v in zip(x2, order2, samp):
        ax_samp.set_ylim(0.5, 3e6)
        ax_samp.text(xi, v * 1.25, f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}",
                     ha="center", fontsize=6, rotation=90)

    # best score vs total training samples consumed (grad_steps x batch)
    for lab, (_seg, color, _killy) in ARMS.items():
        # arms whose segment list spans two trials (wave3 -> wave3b): sum both
        segs = ARMS[lab][0]
        tot = 0
        for s in segs:
            r = by_name.get(s.split("/")[-1])
            if r and r.get("grad_steps") and r.get("batch"):
                tot += r["grad_steps"] * r["batch"]
        if tot:
            best = max((r.get("mean", 0) for r in loaded[lab]), default=0)
            ax_compute.scatter(tot, best, s=110, color=color,
                               marker="o" if lab not in DASHED else "^",
                               edgecolor="k", zorder=3)
            ax_compute.annotate(lab, (tot, best), fontsize=8,
                                xytext=(4, 5), textcoords="offset points")
    ax_compute.set_xscale("log")
    ax_compute.set_xlabel("累计训练样本 = grad_steps x batch (log)")
    ax_compute.set_ylabel("该臂最高 eval 分")
    ax_compute.set_title("F. 算力投入 vs 分数上限", fontsize=13, fontweight="bold")
    ax_compute.margins(x=0.30, y=0.14)
    ax_compute.grid(alpha=0.3)

    # --- row 3: behaviour ----------------------------------------------
    for lab, (_, color, _) in ARMS.items():
        ev = loaded[lab]
        if not ev:
            continue
        xs = [e.get("env_steps", 0) for e in ev]
        curve(ax_fruit, xs, [e.get("maxfruit_max", 0) for e in ev], lab, color,
              dashed=(lab in DASHED))
        curve(ax_moves, xs, rpad([e.get("moves_mean", 0) for e in ev], 3), lab,
              color, dashed=(lab in DASHED))
        curve(ax_p2k, xs, rpad([e.get("p2000", 0) for e in ev], 3), lab, color,
              dashed=(lab in DASHED))
    for ax, title, ylab in [
            (ax_fruit, "G. 最大果型（10 = 西瓜）", "max fruit type"),
            (ax_moves, "H. 单局平均存活步数", "moves_mean"),
            (ax_p2k, "I. P(score>=2000)  # 接近西瓜的比例", "p2000")]:
        ax.set_xscale("log")
        ax.set_xlabel("env_steps (log)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=8.5, loc="upper left")

    # --- bottom: the per-trial ledger -----------------------------------
    ax_t = fig.add_subplot(gs[3, :])
    ax_t.axis("off")
    hdr = ["trial", "arch", "params", "batch", "grad_steps", "env_steps",
           "env/s", "样本/s", "best", "last", "maxfr", "存活步", "θ", "host"]
    order3 = sorted(trials, key=lambda t: (t["wave"], -(t["eval_last"] or 0)))
    body = []
    for t in order3:
        p = t.get("params")
        s = (t.get("grad_sps") or 0) * (t.get("batch") or 1)
        k = "200" if t["trial"].startswith("w4_k200") else ("170" if t["arch"] in
                                                            ("mlp", "settf") else "-")
        body.append([
            t["trial"], t["arch"] or "-",
            f"{p/1e6:.2f}M" if isinstance(p, int) else "-",
            str(t.get("batch") or "-"),
            f"{t.get('grad_steps') or 0:,}",
            f"{(t.get('env_steps') or 0)/1e6:,.0f}M",
            f"{t.get('env_sps') or 0:,.0f}",
            f"{s/1000:,.1f}k" if s else "-",
            f"{t.get('eval_best') or 0:,.0f}",
            f"{t.get('eval_last') or 0:,.0f}",
            str(t.get("maxfruit_max") if t.get("maxfruit_max") is not None else "-"),
            f"{t.get('moves_last') or 0:,.0f}",
            k, t["host"],
        ])
    tb = ax_t.table(cellText=body, colLabels=hdr, loc="center", cellLoc="center",
                    bbox=[0.0, 0.0, 1.0, 0.90])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.0)
    tb.scale(1, 1.30)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#2b2b2b")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            row = order3[r - 1]
            if row["trial"].startswith("w4_k200"):
                cell.set_facecolor("#fff3e0")
            elif row["arch"] == "mlp":
                cell.set_facecolor("#eaf2fb")
            elif row["arch"] == "settf":
                cell.set_facecolor("#fdeaea")
    ax_t.set_title(
        "J. 全部 %d 个 trial 台账（θ = 死亡线 killy；橙底 = killy 200，"
        "分数与 killy 170 不可比；蓝底 = MLP，红底 = set-transformer）" % len(order3),
        fontsize=13, fontweight="bold", y=0.96)

    fig.suptitle("合成大西瓜 RL —— 全 trial 总览 panel（架构 / 吞吐 / 上限 / 台账）",
                 fontsize=18, fontweight="bold", y=0.995)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
