#!/usr/bin/env python3
"""AIQ RL live reporter -- local wandb substitute for offline nodes.

Parses the trainer log / task_split.jsonl / eval history of one run and writes
a self-contained report directory:

  <out>/metrics.json     flat time series (per step + per eval)
  <out>/fig{1..4}_*.png  figures
  <out>/report.html      self-contained HTML (base64 images, glossary, tables)

Usage:
  python report_watch.py --run runs/async_dp_v3full --trainer-log logs/trainer_dp_v3full.log \
      --eval-history evals/async_dp_v3full/history.jsonl --base-eval evals/base_eval300_v3.json \
      --pool-manifest data/pool_manifest_v3.json --out runs/async_dp_v3full/report [--once] [--interval 600]
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NUM = r"(-?[0-9.e+\-]+)"


def parse_trainer_log(path: Path) -> list[dict]:
    txt = path.read_text(errors="replace")
    rows = []
    for m in re.finditer(r"'reward': '" + NUM + "'", txt):
        seg = txt[m.start(): m.start() + 4500]

        def g(key, default=None):
            mm = re.search(r"'" + re.escape(key) + r"': '" + NUM + "'", seg)
            return float(mm.group(1)) if mm else default

        rows.append({
            "reward": float(m.group(1)),
            "loss": g("loss"),
            "kl": g("kl"),
            "entropy": g("entropy"),
            "meanlen": g("completions/mean_length"),
            "trunc": g("completions/clipped_ratio"),
            "seq_clip_low": g("gspo/seq_clip_low_frac"),
            "lr": g("learning_rate"),
            "step_s": g("perf/step_s"),
        })
    for i, r in enumerate(rows, 1):
        r["step"] = i
    return rows


UNPARSED_REWARD = -0.5
TRUNCATED_REWARD = -2.0
# MCQ wrong answer scores 0.0 (see async_grpo_train.mcq_reward); ranking uses concordance.


def parse_task_split(path: Path, n_bins: int = 40) -> list[dict]:
    """Per-bin per-task reward / accuracy / truncation, reconstructed exactly from the
    reward counters: mcq_reward = (correct - 0.5*unparsed - 2*trunc)/n,
    rank_reward = (score_sum - 2*trunc)/n. Accuracy excludes the truncation penalty."""
    calls = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not calls:
        return []
    t0, t1 = calls[0]["t"], calls[-1]["t"]
    span = max(t1 - t0, 1)
    bins = [dict(mc=0, mu=0, mt=0, nm=0, re_=0, ru=0, rt=0, rs=0.0, nr=0,
                 src={}) for _ in range(n_bins)]
    for c in calls:
        b = min(int((c["t"] - t0) / span * n_bins), n_bins - 1)
        d = bins[b]
        d["nm"] += c["n_mcq"]; d["mc"] += c["mcq_correct"]; d["mu"] += c["mcq_unparsed"]; d["mt"] += c["mcq_trunc"]
        d["nr"] += c["n_rank"]; d["re_"] += c["rank_exact"]; d["ru"] += c["rank_unparsed"]
        d["rt"] += c["rank_trunc"]; d["rs"] += c["rank_score_sum"]
        for src, (n, rsum, cor, tr) in (c.get("by_source") or {}).items():
            s = d["src"].setdefault(src, [0, 0.0, 0, 0])
            s[0] += n; s[1] += rsum; s[2] += cor; s[3] += tr
    out = []
    for i, d in enumerate(bins):
        row = {
            "hour": round((i + 0.5) * span / n_bins / 3600, 2),
            "mcq_reward": (d["mc"] + UNPARSED_REWARD * d["mu"] + TRUNCATED_REWARD * d["mt"]) / d["nm"] if d["nm"] else None,
            "mcq_correct": d["mc"] / d["nm"] if d["nm"] else None,
            "mcq_trunc": d["mt"] / d["nm"] if d["nm"] else None,
            "rank_reward": (d["rs"] + TRUNCATED_REWARD * d["rt"]) / d["nr"] if d["nr"] else None,
            "rank_exact": d["re_"] / d["nr"] if d["nr"] else None,
            "rank_score": d["rs"] / d["nr"] if d["nr"] else None,
            "rank_trunc": d["rt"] / d["nr"] if d["nr"] else None,
            "by_source": {s: {"reward": v[1] / v[0], "correct": v[2] / v[0], "trunc": v[3] / v[0], "n": v[0]}
                          for s, v in d["src"].items() if v[0]},
        }
        out.append(row)
    return out


def parse_eval_history(path: Path) -> list[dict]:
    out = []
    for l in path.read_text().splitlines():
        if not l.strip():
            continue
        d = json.loads(l)
        if "overall" not in d:
            continue
        out.append({
            "step": d["step"],
            "overall": d["overall"]["accuracy"],
            "mcq": d["by_task"].get("mcq", {}).get("accuracy"),
            "ranking": d["by_task"].get("ranking", {}).get("accuracy"),
            "parse": d["overall"].get("parse_rate"),
            "trunc": d["overall"].get("truncation_rate"),
            "mean_tokens": d["overall"].get("mean_completion_tokens"),
            "const_mcq": d.get("constant_baseline", {}).get("mcq", {}).get("accuracy"),
            "const_rank": d.get("constant_baseline", {}).get("ranking", {}).get("accuracy"),
        })
    return sorted(out, key=lambda r: r["step"])


def rolling(xs, w=20):
    out, s = [], 0.0
    q = []
    for x in xs:
        q.append(x); s += x
        if len(q) > w:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def figs(steps, tasks, evals, base, out: Path):
    plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

    # fig1: held-out ladder
    if evals:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        xs = [e["step"] for e in evals]
        ax.plot(xs, [e["overall"] for e in evals], "o-", label="overall", color="black")
        ax.plot(xs, [e["mcq"] for e in evals], "s-", label="MCQ (200q)", color="tab:blue")
        ax.plot(xs, [e["ranking"] for e in evals], "^-", label="ranking (100q)", color="tab:green")
        if base is not None:
            ax.scatter([0], [base["overall"]["accuracy"]], marker="D", color="tab:red",
                       label="base (step 0)", zorder=5)
        cm = next((e["const_mcq"] for e in evals if e["const_mcq"]), None)
        cr = next((e["const_rank"] for e in evals if e["const_rank"]), None)
        if cm:
            ax.axhline(cm, ls="--", color="tab:blue", alpha=0.5, label=f"const-A MCQ {cm:.2f}")
        if cr:
            ax.axhline(cr, ls="--", color="tab:green", alpha=0.5, label=f"const ranking {cr:.2f}")
        ax.set_xlabel("checkpoint step"); ax.set_ylabel("held-out accuracy (eval300, T=1, n=2)")
        ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "fig1_heldout.png"); plt.close(fig)

    # fig2: train reward + truncation
    if steps:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        xs = [r["step"] for r in steps]
        rw = [r["reward"] for r in steps]
        ax.plot(xs, rw, color="tab:blue", alpha=0.15)
        ax.plot(xs, rolling(rw), color="tab:blue", label="reward (20-step mean)")
        ax.set_xlabel("step"); ax.set_ylabel("mean reward per completion")
        ax.set_ylim(-2.1, 1.1)
        ax2 = ax.twinx()
        ax2.plot(xs, [r["trunc"] for r in steps], color="tab:red", alpha=0.6, label="truncation frac")
        ax2.set_ylabel("truncation frac", color="tab:red"); ax2.set_ylim(0, 1)
        ax2.grid(False)
        fig.tight_layout(); fig.savefig(out / "fig2_train_reward.png"); plt.close(fig)

    # fig3: per-task curves -- reward / accuracy / truncation
    if tasks:
        fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
        xs = [t["hour"] for t in tasks]

        def y(key, task=None, metric=None):
            if task is None:
                return [t[key] for t in tasks]
            return [(t["by_source"].get(task) or {}).get(metric) for t in tasks]

        axes[0].plot(xs, y("mcq_reward"), color="tab:blue", label="MCQ")
        axes[0].plot(xs, y("rank_reward"), color="tab:green", label="ranking")
        axes[0].set_ylabel("train reward mean")
        axes[0].set_title("reward (includes -2 truncation penalty, -0.5 unparseable)")
        axes[0].legend(fontsize=8)
        axes[1].plot(xs, y("mcq_correct"), color="tab:blue", label="MCQ correct")
        axes[1].plot(xs, y("rank_exact"), color="tab:green", label="ranking exact")
        axes[1].plot(xs, y("rank_score"), "--", color="tab:green", alpha=0.5,
                     label="ranking partial score")
        axes[1].set_ylabel("train accuracy")
        axes[1].set_title("accuracy (correctness only, no truncation penalty)")
        axes[1].set_ylim(0, 1.02); axes[1].legend(fontsize=8)
        axes[2].plot(xs, y("mcq_trunc"), color="tab:blue", label="MCQ")
        axes[2].plot(xs, y("rank_trunc"), color="tab:green", label="ranking")
        axes[2].set_ylabel("truncation frac"); axes[2].set_xlabel("hours into run")
        axes[2].set_ylim(0, 1)
        fig.tight_layout(); fig.savefig(out / "fig3_task_curves.png"); plt.close(fig)

    # fig5 (optional): per-source curves, only when by_source logging exists (post-edit runs)
    srcs = sorted({s for t in tasks for s in t.get("by_source", {})}) if tasks else []
    if srcs:
        fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
        xs = [t["hour"] for t in tasks]
        for s in srcs:
            rw = [(t["by_source"].get(s) or {}).get("reward") for t in tasks]
            cr = [(t["by_source"].get(s) or {}).get("correct") for t in tasks]
            tr = [(t["by_source"].get(s) or {}).get("trunc") for t in tasks]
            axes[0].plot(xs, rw, label=s)
            axes[1].plot(xs, cr, label=s)
            axes[2].plot(xs, tr, label=s)
        axes[0].set_ylabel("reward"); axes[0].legend(fontsize=7)
        axes[1].set_ylabel("correct rate"); axes[1].set_ylim(0, 1.02)
        axes[2].set_ylabel("truncation"); axes[2].set_xlabel("hours into run"); axes[2].set_ylim(0, 1)
        fig.suptitle("per-source curves")
        fig.tight_layout(); fig.savefig(out / "fig5_per_source.png"); plt.close(fig)

    # fig4: length / entropy / gspo health
    if steps:
        fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), sharex=True)
        xs = [r["step"] for r in steps]
        axes[0, 0].plot(xs, [r["meanlen"] for r in steps], color="tab:purple")
        axes[0, 0].set_yscale("log"); axes[0, 0].set_ylabel("mean completion tokens")
        axes[0, 1].plot(xs, [r["entropy"] for r in steps], color="tab:orange")
        axes[0, 1].set_ylabel("entropy")
        axes[1, 0].plot(xs, [r["seq_clip_low"] for r in steps], color="tab:brown")
        axes[1, 0].set_ylabel("GSPO seq_clip_low frac"); axes[1, 0].set_xlabel("step")
        axes[1, 0].axhline(0.5, ls="--", color="red", alpha=0.5)
        axes[1, 1].plot(xs, [r["kl"] for r in steps], color="tab:cyan")
        axes[1, 1].set_ylabel("KL"); axes[1, 1].set_xlabel("step")
        fig.tight_layout(); fig.savefig(out / "fig4_length_health.png"); plt.close(fig)


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


GLOSSARY = """
<h3>Metric glossary</h3>
<ul>
<li><b>reward (train, per step / per task bin)</b>: mean scalar reward over completions.
MCQ: +1 correct / 0 wrong / -0.5 unparseable / -2 truncated at the 16k cap.
Ranking (gold <code>A&lt;B&lt;C&lt;D&lt;E</code>, lower held-out loss first): concordance mapped to
[-1,+1] — exact order +1, full reversal -1, coin flip 0, unparseable -0.5, truncated -2.
Per-task curves are reconstructed exactly from the reward-call counters in task_split.jsonl;
per-step values in the trainer log mix tasks by batch (TRL clusters 16 completions per
prompt), so read smoothed curves, not single steps.</li>
<li><b>accuracy (train)</b>: correctness only — MCQ correct rate / ranking exact rate —
deliberately excluding the -2 truncation and -0.5 parse penalties, which is why it diverges
from reward early in the run.</li>
<li><b>truncation frac</b>: share of completions hitting the 16384-token cap.</li>
<li><b>held-out accuracy (eval300)</b>: fraction of exact-correct samples on the 300-question
held-out set (200 MCQ + 100 ranking), T=1.0, n=2 samples per question. MCQ = right letter;
ranking = full order exact. <i>mean_reward</i> in the eval json is the same samples averaged
under the training reward mapping.</li>
<li><b>constant baseline</b>: always answering the majority gold (MCQ: "A"; ranking: the most
common order). The number a data-blind policy gets.</li>
<li><b>mean completion tokens</b>: generated length. This run converged to direct answers
(~16 tokens) from verbose CoT (~10k).</li>
<li><b>truncation frac</b>: share of completions hitting the 16384-token cap (reward -2).</li>
<li><b>GSPO seq_clip_low frac</b>: share of sequences whose sequence-level importance ratio
rho hit the lower clip epsilon (0.007). Sustained &ge; 0.5 = the arm-C collapse signature;
this run stays &lt; 0.1.</li>
</ul>
"""


def write_html(out: Path, steps, tasks, evals, base, manifest, run_meta):
    imgs = "".join(
        f'<h3>{p.stem}</h3><img style="max-width:100%" src="data:image/png;base64,{b64(p)}"/>'
        for p in sorted(out.glob("fig*.png"))
    )
    eval_rows = "".join(
        f"<tr><td>{e['step']}</td><td>{e['overall']:.3f}</td><td>{e['mcq']:.3f}</td>"
        f"<td>{e['ranking']:.3f}</td><td>{e['parse']:.3f}</td><td>{e['trunc']:.3f}</td>"
        f"<td>{e['mean_tokens']:.0f}</td></tr>"
        for e in evals
    )
    base_row = ""
    if base:
        o = base["overall"]
        base_row = (f"<tr><td>0 (base)</td><td>{o['accuracy']:.3f}</td><td>-</td><td>-</td>"
                    f"<td>{o['parse_rate']:.3f}</td><td>{o['truncation_rate']:.3f}</td>"
                    f"<td>{o['mean_completion_tokens']:.0f}</td></tr>")
    data_sec = ""
    if manifest:
        tb, sb = manifest["task_balance"], manifest["source_balance"]
        data_sec = f"""
<h3>Data</h3>
<p><b>Train</b> {manifest['splits']['train']} questions
(MCQ {tb['train']['mcq']}: {', '.join(f'{k} {v}' for k, v in sb['train'].items() if k != 'loss_ranked')};
ranking {tb['train']['ranking']}: loss_ranked).<br/>
<b>Held-out eval</b> {manifest['splits']['eval']} questions
(MCQ {tb['eval']['mcq']}: {', '.join(f'{k} {v}' for k, v in sb['eval'].items() if k != 'loss_ranked')};
ranking {tb['eval']['ranking']}: loss_ranked holdout, incl. the 50 Opus-5 pre-graded).
Three-way disjointness audited: question id / orig-cf pair / cross-source dataset instance.</p>"""
    hp = ""
    if run_meta and "hyperparams" in run_meta:
        hp = f"<h3>Run</h3><pre>{json.dumps(run_meta['hyperparams'], indent=1)}</pre>"
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>AIQ RL report</title>
<style>body{{font-family:-apple-system,sans-serif;max-width:900px;margin:2em auto;padding:0 1em}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:3px 8px;font-size:13px}}</style>
</head><body>
<h2>AIQ RL live report — {out.parent.name}</h2>
<p>generated {time.strftime('%Y-%m-%d %H:%M:%S')} · {len(steps)} steps logged · {len(evals)} held-out evals</p>
<h3>Held-out eval300</h3>
<table><tr><th>step</th><th>overall</th><th>MCQ</th><th>ranking</th><th>parse</th><th>trunc</th><th>mean tok</th></tr>
{base_row}{eval_rows}</table>
{data_sec}{hp}{GLOSSARY}{imgs}
</body></html>"""
    (out / "report.html").write_text(html)


def build(args):
    run = Path(args.run)
    steps = parse_trainer_log(Path(args.trainer_log))
    tasks = parse_task_split(run / "task_split.jsonl") if (run / "task_split.jsonl").exists() else []
    evals = parse_eval_history(Path(args.eval_history)) if args.eval_history and Path(args.eval_history).exists() else []
    base = json.load(open(args.base_eval)) if args.base_eval and Path(args.base_eval).exists() else None
    manifest = json.load(open(args.pool_manifest)) if args.pool_manifest and Path(args.pool_manifest).exists() else None
    run_meta = json.load(open(run / "run.json")) if (run / "run.json").exists() else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("fig*.png"):
        stale.unlink()
    figs(steps, tasks, evals, base, out)
    json.dump({"steps": steps, "task_bins": tasks, "evals": evals},
              open(out / "metrics.json", "w"))
    write_html(out, steps, tasks, evals, base, manifest, run_meta)
    last = evals[-1] if evals else None
    print(f"[report] {len(steps)} steps, {len(evals)} evals"
          + (f", latest held-out @{last['step']}: {last['overall']:.3f}" if last else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--trainer-log", required=True)
    ap.add_argument("--eval-history", default=None)
    ap.add_argument("--base-eval", default=None)
    ap.add_argument("--pool-manifest", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    if args.once:
        build(args)
        return
    while True:
        try:
            build(args)
        except Exception as e:  # keep the reporter alive across partial writes
            print(f"[report] error: {e}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
