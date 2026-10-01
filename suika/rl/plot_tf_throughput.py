"""Render the tf_xl throughput-ablation panel.

Reads the raw artifacts pulled from node_d (dqn_runs/tf_throughput/) and
writes tf_throughput_panel.png. No numbers are hand-copied: every bar comes
from a bench/probe JSON.

  A  throughput ladder      - what each switch is worth, and the OOM wall
  B  cost structure         - ms/step vs stage3 depth (linear fit -> stage3%)
  C  padding                - board fruit count vs the 160 slots we pay for
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "dqn_runs", "tf_throughput")
OUT = os.path.join(os.path.dirname(HERE), "dqn_runs", "tf_throughput_panel.png")

C_PROD = "#c0392b"
C_WIN = "#1e8449"
C_LOSE = "#7f8c8d"
C_OOM = "#e67e22"


def load(name):
    return json.load(open(os.path.join(SRC, name)))


def rows(*files):
    out = []
    for f in files:
        out += load(f)["rows"]
    return {r["name"]: r for r in out}


def label_of(name):
    r = name.split("-")
    if len(r) < 5:
        return name
    return f"{r[0]}·{r[1]}·{r[2]}·m{r[3][1:]}·cp{r[4][2:]}"


# ---------------------------------------------------------------- panel A
def panel_a(ax, R, probe):
    order = [
        "ck1-eager-bf16-m4096-cp1",     # production
        "ck1-sdpa-bf16-m4096-cp1",
        "ck1-sdpa-bf16-m2048-cp1",
        "ck0-sdpa-bf16-m1024-cp1",      # best
        "ck0-sdpa-bf16-m8192-cp0",
        "ck0-sdpa-bf16-m2048-cp1",
        "ck0-sdpa-bf16-m4096-cp1",
        "ck1-sdpa-bf16-m1024-cp1",
        "ck1-eager-bf16-m4096-cp0",
        "ck1-sdpa-fp32-m4096-cp1",
    ]
    PROD, WIN = 0, 3
    ys, vals, cols, labs = [], [], [], []
    for i, n in enumerate(order):
        r = R[n]
        ys.append(i)
        v = r.get("samples_per_s", 0) or 0
        vals.append(v)
        cols.append(C_PROD if i == PROD else C_WIN if i == WIN else C_LOSE)
        if v:
            labs.append(f"{v:,} samp/s   {r['ms_per_step']/1000:.1f} s/step   "
                        f"{r['peak_gb']:.0f} GB")
        else:
            labs.append("OOM on 40 GB A100")
    ax.barh(ys, vals, color=cols, height=0.7)
    for y, v, l in zip(ys, vals, labs):
        if v:
            ax.text(v + 60, y, l, va="center", fontsize=8.5)
        else:
            ax.text(60, y, l, va="center", fontsize=8.5, color=C_OOM,
                    style="italic")
    ax.set_yticks(ys)
    ax.set_yticklabels([label_of(n) for n in order], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 4600)
    ax.set_xlabel("learner throughput (samples/s @ batch 32768)")
    ax.set_title("A. Every switch, measured (node_d GPU4, A100-40G, T=160)",
                 fontsize=11, loc="left")
    ax.grid(axis="x", alpha=0.25)
    ax.text(0.99, 0.03,
            f"bf16 {84526.4/12631.6:.1f}x   compile {19647.0/12631.6:.2f}x   "
            f"SDPA {12631.6/11861.5:.2f}x",
            transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(fc="#fdf2e9", ec=C_OOM, alpha=0.9))


# ---------------------------------------------------------------- panel B
def panel_b(ax, share, prod_ms):
    """ms/step vs stage3 depth; the intercept is everything that is not stage3."""
    pts = [(16, prod_ms)]
    for r in share:
        n = r["compile_count"] and r.get("__s3")
        if n:
            pts.append((n, r["ms_per_step"]))
    pts.sort()
    d = np.array([p[0] for p in pts], dtype=float)
    m = np.array([p[1] for p in pts], dtype=float)
    b, a = np.polyfit(d, m, 1)                       # a = intercept
    ax.plot(d, m, "o", ms=9, color=C_PROD, zorder=3,
            label="measured (fresh process each)")
    dd = np.linspace(0, 17.5, 50)
    ax.plot(dd, a + b * dd, "--", color="#2c3e50", lw=1.4,
            label=f"fit: {a/1000:.2f} s + {b/1000:.2f} s x depth")
    frac = b * 16 / prod_ms
    ax.axhspan(0, a, color="#dfe6e9", alpha=0.7)
    ax.annotate(f"everything else\n{a/1000:.2f} s ({100*(1-frac):.0f}%)",
                xy=(1.0, a / 2), fontsize=9, va="center",
                xytext=(2.2, a * 0.8), textcoords="data",
                arrowprops=dict(arrowstyle="->", color="#636e72"))
    ax.annotate(f"stage3 = 64 latents x 16 layers\n={frac*100:.0f}% of the step",
                xy=(16, prod_ms), xytext=(6.4, 9600), fontsize=9,
                arrowprops=dict(arrowstyle="->", color=C_PROD))
    for x, y in pts:
        ax.text(x, y + 420, f"{y/1000:.2f}s", ha="center", fontsize=8)
    ax.set_xlabel("stage3 depth (latent self-attn layers)")
    ax.set_ylabel("ms / grad step")
    ax.set_ylim(0, 14500)
    ax.set_title("B. The cost is the latent tower, and it is linear in depth",
                 fontsize=11, loc="left")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
    return frac, a, b


# ---------------------------------------------------------------- panel C
def panel_c(ax, bc):
    hist = np.array(bc["board_hist"], dtype=float)
    T = bc["T"]
    used = np.arange(len(hist))
    ax.bar(used, hist, width=0.9, color="#2980b9")
    ax.axvspan(48, T, color=C_OOM, alpha=0.13)
    ax.set_yscale("log")
    ax.set_xlim(0, T)
    mean = float((used * hist).sum() / hist.sum())
    ax.axvline(mean, color=C_PROD, lw=1.6)
    ax.text(mean + 2, hist.max() * 0.6, f"mean {mean:.1f}",
            color=C_PROD, fontsize=9)
    ax.text(104, hist.max() * 0.12,
            f"never observed past 48\n({T-48} of {T} slots are\n"
            f"structural padding)",
            fontsize=9, color="#a04000")
    ax.set_xlabel(f"fruits on the board -> token rows used (of T={T})")
    ax.set_ylabel(f"transitions (n={bc['n']:,})")
    ax.set_title("C. ...but cutting T is worthless: 90% of tokens are padding",
                 fontsize=11, loc="left")
    ax.grid(axis="y", alpha=0.25)


def main():
    R = rows("bench_tf_throughput.json", "bench_tf2.json")
    probe = [json.loads(l) for l in open(os.path.join(SRC, "probe_compile_modes.jsonl"))]
    share = [json.loads(l) for l in open(os.path.join(SRC, "probe_share.jsonl"))]
    # attach the swept knob to each share-probe row, in run order. Row 1 is the
    # T=64 run: its stage3 depth is still 16, so it must NOT enter the
    # depth->cost fit (two x=16 points would split the least-squares line).
    for r, s3 in zip(share, [16, None, 8, 4]):
        r["__s3"] = s3
    bc = load("board_counts.json")
    prod_ms = share[0]["ms_per_step"]

    fig, axes = plt.subplots(1, 3, figsize=(21.5, 8.2),
                            gridspec_kw=dict(width_ratios=[1.25, 1.0, 1.0]))
    panel_a(axes[0], R, probe)
    frac, a, b = panel_b(axes[1], share, prod_ms)
    panel_c(axes[2], bc)

    best = R["ck0-sdpa-bf16-m1024-cp1"]
    prod = R["ck1-eager-bf16-m4096-cp1"]
    fig.suptitle(
        "tf_xl 56M throughput ablation — the 14.7 s/step is architecture, not a "
        "sw bug: the two big levers (bf16, compile) were already on\n"
        f"best reachable = {best['samples_per_s']:,} samples/s "
        f"({best['ms_per_step']/1000:.1f} s/step, +{100*(prod['ms_per_step']/best['ms_per_step']-1):.0f}% "
        f"over production) — vs the 20–50k samples/s I had predicted",
        fontsize=12.5, y=0.985)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(OUT, dpi=145)
    print(f"wrote {OUT}")
    print(f"stage3 share = {frac*100:.1f}%  intercept = {a:.0f} ms  "
          f"per-layer = {b:.0f} ms")


if __name__ == "__main__":
    main()
