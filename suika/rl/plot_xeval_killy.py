"""Kill-line cross-eval dashboard: frozen weights x killy in {170, 200}.

Reads the per-seed JSONs written by suika_dqn/xeval_killy.py (pulled from the
node that holds the run dirs) and separates the two effects that the training
curves conflate:

  * environment effect  -- same weights, death line moved 170 -> 200
  * policy effect       -- k170-trained vs k200-trained weights, same line

Panel A: grouped bars with bootstrap 95% CI (64 paired seeds per cell).
Panel B: per-seed paired differences under the same kill line (common random
numbers), which is the test the aggregate curves cannot do.

Usage: python plot_xeval_killy.py [--dir dqn_runs/xeval_killy/batch2]
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB",
                                          "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DEFAULT_DIR = os.path.join(PROJ, "dqn_runs", "xeval_killy", "batch2")
OUT = os.path.join(PROJ, "dqn_runs", "xeval_killy.png")

# label -> (short name, colour)
FAMILY = {
    "fork": ("分叉点权重 step3801", "#7f7f7f"),
    "w3c_k170": ("k170 续训后 (w3c)", "#d62728"),
    "w4cont_k200": ("k200 续训后 (w4cont)", "#1f77b4"),
}
ORDER = ["fork", "w3c_k170", "w4cont_k200"]


def load(cdir):
    cells = {}
    for d in cdir.split(","):
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            if os.path.basename(p).startswith("smoke"):
                continue
            r = json.load(open(p))
            fam = r["label"].split("@")[0]
            cells[(fam, int(r["killy"]))] = r
    return cells


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if x.size == 0:
        return (np.nan, np.nan)
    m = rng.choice(x, size=(n, x.size), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(a, b, n=10000, seed=1):
    """Bootstrap CI + sign test for mean(a) - mean(b) over paired seeds."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, d.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    p_sign = float(np.mean(d <= 0))
    return float(d.mean()), float(lo), float(hi), p_sign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seeds", default="all",
                    help="'all' or LO:HI to restrict the seed subset")
    args = ap.parse_args()

    cells = load(args.dir)
    if not cells:
        sys.exit(f"no JSON in {args.dir}")
    per_seed = all("scores" in c for c in cells.values())
    lo, hi = (0, 10 ** 9)
    if args.seeds != "all":
        a, b = args.seeds.split(":")
        lo, hi = int(a), int(b)

    def series(cell):
        if per_seed:
            s = np.array(cell["scores"], float)
            sd = np.array(cell["seeds"], int)
            return s[(sd >= lo) & (sd < hi)]
        return np.full(1, cell["mean"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    width = 0.26
    for i, fam in enumerate(ORDER):
        for j, killy in enumerate((170, 200)):
            cell = cells.get((fam, killy))
            if cell is None:
                continue
            x = j + (i - 1) * width
            s = series(cell)
            m = float(s.mean())
            err = None
            if per_seed and s.size > 1:
                l, h = boot_ci(s)
                err = [[m - l], [h - m]]
            names = FAMILY[fam][0]
            ax.bar(x, m, width * 0.92, color=FAMILY[fam][1],
                   yerr=err, capsize=3, alpha=0.9,
                   label=names if j == 0 else None)
            ax.text(x, m + 60, f"{m:.0f}", ha="center", fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["killy = 170（旧死亡线）", "killy = 200（新死亡线）"])
    ax.set_ylabel("greedy eval 平均分（同种子集）")
    ax.set_title("同一权重、只移动死亡线：环境效应\n"
                 "vs 同一死亡线、不同训练权重：策略效应")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9, loc="upper right")

    ax = axes[1]
    lines = []
    labels = []
    if per_seed:
        f_k170, f_k200 = cells.get(("fork", 170)), cells.get(("fork", 200))
        c_k200, w_k200 = cells.get(("w4cont_k200", 200)), cells.get(("w3c_k170", 200))
        c_k170, w_k170 = cells.get(("w4cont_k200", 170)), cells.get(("w3c_k170", 170))
        panels = []
        if f_k170 and f_k200:
            panels.append(("环境效应\n同权重 k170→k200",
                           series(f_k170), series(f_k200), "#7f7f7f"))
        if c_k200 and w_k200:
            panels.append(("策略效应 @killy=200\nk200续训 vs k170续训",
                           series(c_k200), series(w_k200), "#1f77b4"))
        if c_k170 and w_k170:
            panels.append(("换回旧死亡线 @killy=170\nk200续训 vs k170续训",
                           series(c_k170), series(w_k170), "#2ca02c"))
        for i, (name, a, b, colour) in enumerate(panels):
            d = a - b
            ax.bar(i, d.mean(), 0.5, color=colour, alpha=0.9)
            l, h = boot_ci(d)
            ax.errorbar(i, d.mean(), yerr=[[d.mean() - l], [h - d.mean()]],
                        fmt="none", ecolor="k", capsize=4)
            ax.text(i, d.mean() + (30 if d.mean() >= 0 else -70),
                    f"{d.mean():+.0f}\n95%CI [{l:+.0f},{h:+.0f}]\n"
                    f"赢 {int((d > 0).sum())}/{d.size} 局",
                    ha="center", fontsize=9)
            labels.append(name)
        ax.set_xticks(range(len(panels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("配对分差（逐种子，共同随机数）")
        ax.set_title("配对检验：逐种子对照（64 个种子）")
        ax.grid(axis="y", alpha=0.25)

    for fam in ORDER:
        for killy in (170, 200):
            c = cells.get((fam, killy))
            if c is None:
                continue
            s = series(c)
            extra = ""
            if per_seed:
                l, h = boot_ci(s)
                extra = f" CI[{l:.0f},{h:.0f}]"
            print(f"{fam:14s} killy={killy}  mean={s.mean():7.1f}{extra} "
                  f"med={np.median(s):7.1f} n={s.size} "
                  f"p2000={c['p2000']:.2f} moves={c['moves_mean']:.0f} "
                  f"digest={c['obs_digest']}")
    if per_seed:
        contrasts = [
            ("env  same weights, k170 -> k200",
             series(cells[("fork", 170)]), series(cells[("fork", 200)])),
            ("policy @k200: w4cont - w3c",
             series(cells[("w4cont_k200", 200)]), series(cells[("w3c_k170", 200)])),
            ("policy @k170: w4cont - w3c",
             series(cells[("w4cont_k200", 170)]), series(cells[("w3c_k170", 170)])),
        ]
        for name, a, b in contrasts:
            dm, l, h, p = paired(a, b)
            wins = int((a - b > 0).sum())
            print(f"PAIRED {name:34s} {dm:+8.1f} "
                  f"CI[{l:+7.1f},{h:+7.1f}] wins={wins}/{a.size} "
                  f"P(d<=0)={p:.3f}")

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")

    killies = sorted({k for _, k in cells})
    if len(killies) >= 3:
        plot_sweep(cells, series, per_seed, killies, args)


def plot_sweep(cells, series, per_seed, killies, args):
    """Score vs death line: how much room does the line cost, and where?"""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    for fam in ORDER:
        ks = [k for k in killies if (fam, k) in cells]
        if len(ks) < 2:
            continue
        ms, los, his, mk = [], [], [], []
        for k in ks:
            s = series(cells[(fam, k)])
            ms.append(s.mean())
            mk.append(cells[(fam, k)]["moves_mean"])
            l, h = boot_ci(s) if per_seed and s.size > 1 else (s.mean(), s.mean())
            los.append(s.mean() - l)
            his.append(h - s.mean())
        ax.errorbar(ks, ms, yerr=[los, his], marker="o", capsize=3,
                    color=FAMILY.get(fam, (None, "#333"))[1],
                    label=FAMILY.get(fam, (fam, "#333"))[0])
        ax2.plot(ks, mk, marker="s", color=FAMILY.get(fam, (None, "#333"))[1],
                 label=FAMILY.get(fam, (fam, "#333"))[0])
    ax.axvline(170, color="k", ls=":", lw=1)
    ax.text(171, ax.get_ylim()[1] * 0.97, "旧死亡线 170", fontsize=8, va="top")
    ax.set_xlabel("killy（死亡线 y，越大 = 越靠下 = 可用高度越小）")
    ax.set_ylabel("greedy eval 平均分")
    ax.set_title("死亡线响应曲线：同权重横扫 killy（36 种子）")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    ax2.set_xlabel("killy")
    ax2.set_ylabel("平均存活步数")
    ax2.set_title("存活步数：局有多早被截断")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.splitext(args.out)[0] + "_sweep.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    print("SWEEP table:")
    for fam in ORDER:
        for k in sorted(k for f, k in cells if f == fam):
            s = series(cells[(fam, k)])
            print(f"  {fam:14s} killy={k:3d} mean={s.mean():7.1f} "
                  f"moves={cells[(fam, k)]['moves_mean']:5.0f} "
                  f"p2000={cells[(fam, k)]['p2000']:.2f} "
                  f"fruit_max={cells[(fam, k)]['maxfruit_max']}")


if __name__ == "__main__":
    main()
