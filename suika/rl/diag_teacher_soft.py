"""Temperature sweep for soft-target pi distillation from teacher Q.

Hard-argmax CE is unattainable for this teacher (top1-top2 gap p50 = 1.0 on
Q ~ 1096 while val_qd ~ 100), so the pi head should be trained on soft targets
softmax((q_T - mean_a q_T) / tau). This script sweeps tau and reports the mean
target entropy (nats) and top-5 mass, to pick a temperature that is sharp
enough to be a real policy signal but achievable by a regressor.
"""
import argparse
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--shards", type=int, default=200)
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))[:args.shards]
    qs = []
    for f in files:
        d = np.load(f)
        q = d["qteach"].astype(np.float32)
        qs.append(q)
    q = np.concatenate(qs)
    c = q - q.mean(1, keepdims=True)             # centered per state
    print(f"shards={len(files)} states={q.shape[0]:,} K={q.shape[1]}")
    for tau in args.taus:
        p = np.exp((c / tau) - (c / tau).max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        ent = -(p * np.log(np.maximum(p, 1e-12))).sum(1)
        top5 = np.sort(p, 1)[:, -5:].sum(1)
        print(f"  tau={tau:5.1f}  mean H={ent.mean():.3f} nats  "
              f"med H={np.median(ent):.3f}  top5 mass={top5.mean():.3f}  "
              f"max-prob={p.max(1).mean():.3f}")
    # regret yardsticks on the same states
    srt = np.sort(q, 1)
    best_marg = int(np.bincount(np.argmax(q, 1), minlength=q.shape[1]).argmax())
    reg_rand = float((srt[:, -1] - q.mean(1)).mean())
    reg_marg = float((srt[:, -1] - q[:, best_marg]).mean())
    print(f"  top1-top2 gap: med={np.median(srt[:, -1]-srt[:, -2]):.2f} "
          f"mean={(srt[:, -1]-srt[:, -2]).mean():.2f}")
    print(f"  regret(random action)={reg_rand:.1f} "
          f"regret(always-marginal-best)={reg_marg:.1f}")


if __name__ == "__main__":
    main()
