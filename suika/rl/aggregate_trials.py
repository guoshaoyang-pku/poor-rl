"""Aggregate every local DQN trial into one JSON+CSV table.

Reads dqn_runs/<host>/<wave>/<trial>/{metrics,eval}.jsonl, joins the matching
config (configs/<trial>.yaml) for arch hyper-params, and instantiates the model
on CPU purely to count parameters.  No training, no GPU.

Usage:  python aggregate_trials.py [--out dqn_runs/trials.json]
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def read_jsonl(p):
    out = []
    if not p.exists():
        return out
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_cfg(name):
    p = HERE / "configs" / f"{name}.yaml"
    if not (p.exists() and yaml):
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def param_count(cfg):
    """Instantiate on CPU (meta-free, tiny) just to count parameters."""
    if not cfg:
        return None
    try:
        sys.path.insert(0, str(HERE))
        from model import build_model
        arch = cfg.get("arch", "mlp")
        obs_dim = int(cfg.get("obs_dim", 333))
        if arch == "qwen":
            return None  # needs the HF checkpoint; skip
        m = build_model(cfg, obs_dim)
        return int(sum(p.numel() for p in m.parameters()))
    except Exception as e:  # noqa: BLE001
        return f"err:{type(e).__name__}"


def throughput(metrics):
    """Steady-state (env_steps/s, grad_steps/s) from consecutive log deltas.

    NOTE: learner.py logs env_sps/grad_sps as cumulative averages since *this
    process* started (`steps / (now - t0)`), but a resumed run seeds env_steps
    from the parent checkpoint.  The logged value is therefore inflated by
    (resumed_env_steps / elapsed) and only slowly converges.  We recompute from
    deltas between consecutive log lines instead, which is resume-invariant.
    """
    pairs = [(m.get("t"), m.get("env_steps"), m.get("grad_steps")) for m in metrics]
    pairs = [p for p in pairs if p[0] is not None]
    e_rates, g_rates = [], []
    for (t1, e1, g1), (t2, e2, g2) in zip(pairs, pairs[1:]):
        dt = t2 - t1
        if dt <= 0:
            continue
        if e1 is not None and e2 is not None and e2 >= e1:
            e_rates.append((e2 - e1) / dt)
        if g1 is not None and g2 is not None and g2 >= g1:
            g_rates.append((g2 - g1) / dt)

    def med(v):
        if not v:
            return None
        v = sorted(v)
        return v[len(v) // 2]

    return med(e_rates), med(g_rates)


def summarize(trial_dir):
    metrics = read_jsonl(trial_dir / "metrics.jsonl")
    evals = read_jsonl(trial_dir / "eval.jsonl")
    name = trial_dir.name
    cfg = load_cfg(name)
    env_sps, grad_sps = throughput(metrics)
    last = metrics[-1] if metrics else {}
    rec = {
        "trial": name,
        "host": trial_dir.parent.parent.name,
        "wave": trial_dir.parent.name,
        "arch": cfg.get("arch", "mlp" if cfg else None),
        "params": param_count(cfg),
        "killy": cfg.get("killy"),
        "gamma": cfg.get("gamma"),
        "batch": cfg.get("batch"),
        "lr": cfg.get("lr"),
        "K": cfg.get("K"),
        "stage1": cfg.get("stage1"),
        "stage3": cfg.get("stage3"),
        "n_lat": cfg.get("n_lat"),
        "d_lat": cfg.get("d_lat"),
        "d_tok": cfg.get("d_tok"),
        "hidden": cfg.get("hidden"),
        "n_metrics": len(metrics),
        "n_evals": len(evals),
        "env_steps": last.get("env_steps"),
        "grad_steps": last.get("grad_steps"),
        "env_sps": env_sps,
        "grad_sps": grad_sps,
        "q_mean": last.get("q_mean"),
        "loss": last.get("loss"),
        "grad_norm": last.get("grad_norm"),
        "eval_first": evals[0].get("mean") if evals else None,
        "eval_last": evals[-1].get("mean") if evals else None,
        "eval_best": max((e.get("mean", -1e9) for e in evals), default=None),
        "eval_last_env_steps": evals[-1].get("env_steps") if evals else None,
        "moves_last": evals[-1].get("moves_mean") if evals else None,
        "maxfruit_max": max((e.get("maxfruit_max", -1) for e in evals), default=None),
    }
    if evals:
        rec["eval_best_env_steps"] = evals[
            max(range(len(evals)), key=lambda i: evals[i].get("mean", -1e9))
        ].get("env_steps")
        # wall clock spanned by the eval records (eval.jsonl stamps epoch `time`)
        t0, t1 = evals[0].get("time"), evals[-1].get("time")
        rec["wall_h"] = round((t1 - t0) / 3600.0, 2) if (t0 and t1) else None
    else:
        rec["wall_h"] = None
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dqn_runs" / "trials.json"))
    args = ap.parse_args()

    rows = []
    for metrics in sorted((ROOT / "dqn_runs").glob("*/wave*/**/metrics.jsonl")):
        rows.append(summarize(metrics.parent))
    rows.sort(key=lambda r: (r["wave"], -(r["eval_last"] or 0)))

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    csv_path = Path(args.out).with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    hdr = f"{'trial':22s} {'arch':7s} {'params':>10s} {'eval_last':>9s} {'best':>7s} {'env_steps':>12s} {'env/s':>8s} {'grad/s':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        p = r["params"]
        p = f"{p/1e6:.2f}M" if isinstance(p, int) else str(p)
        print(f"{r['trial']:22s} {str(r['arch']):7s} {p:>10s} "
              f"{(r['eval_last'] or 0):9.0f} {(r['eval_best'] or 0):7.0f} "
              f"{(r['env_steps'] or 0):12.0f} {(r['env_sps'] or 0):8.0f} "
              f"{(r['grad_sps'] or 0):8.1f}")
    print(f"\n{len(rows)} trials -> {args.out}")


if __name__ == "__main__":
    main()
