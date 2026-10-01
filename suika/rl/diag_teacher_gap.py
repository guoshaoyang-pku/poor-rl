"""Teacher Q gap statistics: what pi-CE is reachable at the current Q fidelity?

For each expert (argmax) state in a bc_collector shard, compute the gap between
the teacher's top-1 and top-2 Q values. The pi head can only resolve the argmax
when its Q error is smaller than that gap -- compare the gap distribution with
val_qd (current mean |dQ| ~ 97) to tell whether l_pi ~ 3.79 (== teacher action
marginal entropy 3.787) is an optimization artifact or the intrinsic ceiling.
"""
import argparse
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--shards", type=int, default=200)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))[:args.shards]
    gaps, qstd, qall, n = [], [], [], 0
    for f in files:
        d = np.load(f)
        q = d["qteach"].astype(np.float32)[d["argmax"] > 0]
        if q.size == 0:
            continue
        part = np.partition(q, -2, axis=1)
        gaps.append(part[:, -1] - part[:, -2])
        qstd.append(q.std(axis=1))
        qall.append(q.ravel())
        n += q.shape[0]
    g = np.concatenate(gaps)
    s = np.concatenate(qstd)
    qa = np.concatenate(qall)
    print(f"shards={len(files)} expert_states={n:,}")
    print("  top1-top2 gap pcts: "
          + " ".join(f"p{p}={np.percentile(g, p):.1f}"
                     for p in (1, 5, 10, 25, 50, 75, 90)))
    print("  frac gap< " + " ".join(
        f"{x}:{float((g < x).mean()):.3f}"
        for x in (1, 2, 5, 10, 20, 50, 100, 200)))
    print(f"  per-state Q std: p25={np.percentile(s, 25):.1f} "
          f"med={np.median(s):.1f} p75={np.percentile(s, 75):.1f}")
    print(f"  Q values: p1={np.percentile(qa, 1):.1f} p25={np.percentile(qa, 25):.1f} "
          f"med={np.median(qa):.1f} p75={np.percentile(qa, 75):.1f} "
          f"p99={np.percentile(qa, 99):.1f} max={qa.max():.1f}")


if __name__ == "__main__":
    main()
