"""Pull run metrics from both A100 nodes and build a comparison report.

Usage:
    python aggregate.py pull           # rsync metrics from nodes
    python aggregate.py report         # print table + write curves.png/csv
"""
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
LOCAL_RUNS = os.path.join(PROJ, "dqn_runs")
NODES = {
    "node_d": "/data/user/suika-dqn/runs/",
    "node_c": "/path/to/suika-dqn/runs/",
}


def pull():
    os.makedirs(LOCAL_RUNS, exist_ok=True)
    for node, remote in NODES.items():
        dst = os.path.join(LOCAL_RUNS, node) + "/"
        os.makedirs(dst, exist_ok=True)
        if node == "node_c":
            # node lacks rsync: use tar over ssh
            cmd = (f"ssh {node} 'cd {remote} && tar czf - "
                   f"--exclude=inbox --exclude=checkpoints --exclude=policy.pt "
                   f"--exclude=*.npz .' | tar xzf - -C {dst}")
            subprocess.run(cmd, shell=True, check=False)
        else:
            subprocess.run([
                "rsync", "-az",
                "--exclude", "inbox/", "--exclude", "checkpoints/",
                "--exclude", "policy.pt", "--exclude", "*.npz",
                f"{node}:{remote}", dst], check=False)
    print("pulled ->", LOCAL_RUNS)


def _read_jsonl(path):
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


def collect():
    """arm -> dict(metrics=..., evals=..., episodes=...); key = node/wave/arm."""
    arms = {}
    for node in NODES:
        node_root = os.path.join(LOCAL_RUNS, node)
        for rundir in sorted(glob.glob(os.path.join(node_root, "*", "*"))):
            if not os.path.isdir(rundir):
                continue
            key = os.path.relpath(rundir, LOCAL_RUNS)
            eps = []
            for ef in glob.glob(os.path.join(rundir, "episodes_a*.jsonl")):
                eps.extend(_read_jsonl(ef))
            arms[key] = dict(
                metrics=_read_jsonl(os.path.join(rundir, "metrics.jsonl")),
                evals=_read_jsonl(os.path.join(rundir, "eval.jsonl")),
                episodes=eps,
            )
    return arms


def report():
    arms = collect()
    rows = []
    for key, d in arms.items():
        ev = d["evals"][-1] if d["evals"] else {}
        mt = d["metrics"][-1] if d["metrics"] else {}
        scores = [e["score"] for e in d["episodes"][-2000:]]
        rows.append(dict(
            arm=key,
            eval_mean=ev.get("mean", float("nan")),
            eval_p25=ev.get("p25", float("nan")),
            eval_max=ev.get("max", float("nan")),
            p2000=ev.get("p2000", float("nan")),
            maxfruit=ev.get("maxfruit_max", -1),
            env_steps=mt.get("env_steps", ev.get("env_steps", 0)),
            grad_steps=mt.get("grad_steps", ev.get("grad_steps", 0)),
            train_score=(sum(scores) / len(scores)) if scores else float("nan"),
            n_evals=len(d["evals"]),
        ))
    rows.sort(key=lambda r: -(r["eval_mean"] if r["eval_mean"] == r["eval_mean"] else -1))
    hdr = ("arm", "eval_mean", "p25", "max", "p2000", "fruit",
           "env_M", "grad_k", "train_score", "#ev")
    print(("%-24s %9s %7s %6s %6s %5s %7s %7s %11s %4s" % hdr))
    for r in rows:
        print("%-24s %9.1f %7.1f %6.0f %6.3f %5d %7.1f %7.1f %11.1f %4d" % (
            r["arm"], r["eval_mean"], r["eval_p25"], r["eval_max"],
            r["p2000"], r["maxfruit"], r["env_steps"] / 1e6,
            r["grad_steps"] / 1e3, r["train_score"], r["n_evals"]))
    # persist csv
    import csv
    os.makedirs(LOCAL_RUNS, exist_ok=True)
    with open(os.path.join(LOCAL_RUNS, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)
    # curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for key, d in arms.items():
            if d["evals"]:
                xs = [e["env_steps"] / 1e6 for e in d["evals"]]
                ys = [e["mean"] for e in d["evals"]]
                axes[0].plot(xs, ys, marker=".", ms=3, label=key.split("/")[-1])
            if d["episodes"]:
                import numpy as np
                ss = sorted(d["episodes"], key=lambda e: e["t"])
                sc = np.array([e["score"] for e in ss])
                if len(sc) > 50:
                    k = np.convolve(sc, np.ones(50) / 50, mode="valid")
                    axes[1].plot(k, label=key.split("/")[-1], lw=0.8)
        axes[0].set_title("greedy eval mean vs env steps (M)")
        axes[1].set_title("train episode score (ema50)")
        for ax in axes:
            ax.legend(fontsize=6)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(LOCAL_RUNS, "curves.png"), dpi=120)
        print("curves ->", os.path.join(LOCAL_RUNS, "curves.png"))
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pull":
        pull()
    report()
