"""Teacher action-marginal diagnostic: floor for the BC pi-head CE loss.

Reads N shards from a bc_collector inbox and reports the marginal action
distribution over expert (argmax) samples. The entropy (in nats) is the CE
a state-independent pi head would achieve -- i.e. the floor that l_pi ~ 3.7
must be compared against; the best-constant accuracy is the matching floor
for agree_pi in bc_learner_qwen validate().
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
    hist = np.zeros(128, dtype=np.int64)
    n = n_exp = 0
    for f in files:
        d = np.load(f)
        a, am = d["act"], d["argmax"]
        hist += np.bincount(a[am > 0], minlength=128)
        n_exp += int((am > 0).sum())
        n += len(a)
    p = hist / max(n_exp, 1)
    nz = p[p > 0]
    H = float(-(nz * np.log(nz)).sum())
    order = np.argsort(-p)[:5]
    print(f"shards={len(files)} transitions={n:,} "
          f"expert_frac={n_exp / max(n, 1):.3f}")
    print(f"marginal CE floor = {H:.3f} nats (uniform = 4.852)")
    print(f"best constant-action acc = {p.max():.4f} (col {int(order[0])})")
    print(f"top5 cols {order.tolist()} probs {np.round(p[order], 4).tolist()}")
    print(f"effective classes exp(H) = {np.exp(H):.1f}; "
          f"col coverage {(p > 0).sum()}/128")


if __name__ == "__main__":
    main()
