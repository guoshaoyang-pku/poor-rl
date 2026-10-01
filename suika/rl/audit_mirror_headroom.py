"""Mirror-augmentation audit + ranking headroom probe (CPU only, no GPU).

Two questions, both answered with closed-form readouts on the BC shards:

1. Does the learner's mirror augmentation transfer?  `_mirror` flips x -> 1-x
   and the action space (a -> K-1-a, qteach -> flip(1)) but does NOT re-sort
   the board rows (encoding sorts by (y, x), so a true mirrored encoding would
   re-sort).  Fit a ridge on mirrored data and evaluate on ORIGINAL val: if the
   convention is self-consistent it transfers (~same regret as orig->orig); if
   it is broken the regret collapses to the constant-policy level (~31).

2. What regret is reachable from this input at all?  Linear ridge is the floor;
   a random-feature (nonlinear) ridge shows headroom.  Regret yardsticks:
   constant policy 31, current model 29, gate < 10.
"""
import argparse
import glob
import os

import numpy as np


def mirror_np(obs, T, resort):
    """Mirror [N, T*5] token obs exactly like bc_learner_qwen._mirror.

    resort=True additionally re-sorts board rows by (y, x) after mirroring,
    i.e. produces the encoding of the mirrored state.
    """
    B = obs.shape[0]
    tok = obs.reshape(B, T, 5).copy()
    valid = tok[:, :, 0] >= 0
    valid[:, :2] = False
    x = tok[:, :, 1].copy()
    vx = tok[:, :, 3].copy()
    tok[:, :, 1] = np.where(valid, 1.0 - x, x)
    tok[:, :, 3] = np.where(valid, -vx, vx)
    if resort:
        for b in range(B):
            board = tok[b, 2:]
            keep = board[board[:, 0] >= 0]
            order = np.lexsort((keep[:, 1], keep[:, 2]))   # by y then x
            keep = keep[order]
            out = np.full_like(board, -1.0)
            out[:len(keep)] = keep
            tok[b, 2:] = out
    return tok.reshape(B, -1)


def smooth_l1_mean(a, b):
    d = np.abs(a - b)
    return float(np.mean(np.where(d < 1.0, 0.5 * d * d, d - 0.5)))


def ridge(F, Y, Fv, alpha, n_feat=None):
    Fb = np.concatenate([F, np.ones((len(F), 1), np.float64)], 1)
    G = Fb.T @ Fb + alpha * np.eye(Fb.shape[1])
    W = np.linalg.solve(G, Fb.T @ Y.astype(np.float64))
    Fvb = np.concatenate([Fv, np.ones((len(Fv), 1), np.float64)], 1)
    return (Fvb @ W).astype(np.float32)


def report(name, Qh, Yva, Ava):
    za = Qh.argmax(1)
    reg = float((Yva.max(1) - Yva[np.arange(len(Yva)), za]).mean())
    print(f"  {name:34s} q_regret={reg:7.2f} "
          f"agree={float((za == Ava).mean()):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-shards", type=int, default=8)
    ap.add_argument("--val-shards", type=int, default=2)
    ap.add_argument("--T", type=int, default=160)
    ap.add_argument("--max-n", type=int, default=40000)
    ap.add_argument("--rf-dim", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="npz cache of the splits")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    tf = files[:args.train_shards]
    vf = files[args.train_shards:args.train_shards + args.val_shards]

    def load(flist, cap):
        X, Y, A = [], [], []
        n = 0
        for f in flist:
            d = np.load(f)
            obs, q, act = d["obs"].astype(np.float32), \
                d["qteach"].astype(np.float32), d["act"]
            k = min(len(obs), max(0, cap - n))
            X.append(obs[:k]); Y.append(q[:k]); A.append(act[:k])
            n += k
            if n >= cap:
                break
        return np.concatenate(X), np.concatenate(Y), np.concatenate(A)

    if args.out and os.path.exists(args.out):
        z = np.load(args.out)
        Xtr, Ytr, Atr = z["Xtr"], z["Ytr"], z["Atr"]
        Xva, Yva, Ava = z["Xva"], z["Yva"], z["Ava"]
    else:
        Xtr, Ytr, Atr = load(tf, args.max_n)
        Xva, Yva, Ava = load(vf, args.max_n // 2)
        if args.out:
            np.savez_compressed(args.out, Xtr=Xtr, Ytr=Ytr, Atr=Atr,
                                Xva=Xva, Yva=Yva, Ava=Ava)
    K = Ytr.shape[1]
    print(f"[audit] train={len(Xtr):,} val={len(Xva):,} K={K} T={args.T}",
          flush=True)

    # per-state contrast / gap stats (what precision a student would need)
    srt = np.sort(Yva, 1)
    std = Yva.std(1)
    print(f"  contrast: per-state std p25/med/p75 = "
          f"{np.percentile(std, 25):.1f}/{np.median(std):.1f}/"
          f"{np.percentile(std, 75):.1f}; top1-top2 gap med="
          f"{np.median(srt[:, -1] - srt[:, -2]):.2f}; "
          f"level p25/med/p75 = {np.percentile(Yva.mean(1), 25):.0f}/"
          f"{np.median(Yva.mean(1)):.0f}/{np.percentile(Yva.mean(1), 75):.0f}",
          flush=True)

    def mirror(X, Y, A, resort):
        return (mirror_np(X, args.T, resort), Y[:, ::-1].copy(),
                K - 1 - A)

    Xm, Ym, Am = mirror(Xtr, Ytr, Atr, resort=False)
    Xr, Yr, Ar = mirror(Xtr, Ytr, Atr, resort=True)

    print("[audit] mirror transfer (train -> original val):", flush=True)
    report("orig -> orig (reference)",
           ridge(Xtr, Ytr, Xva, 1.0), Yva, Ava)
    report("mirror-unsorted -> orig (as trained)",
           ridge(Xm, Ym, Xva, 1.0), Yva, Ava)
    report("mirror-resorted -> orig",
           ridge(Xr, Yr, Xva, 1.0), Yva, Ava)
    report("orig+mirror-unsorted -> orig",
           ridge(np.concatenate([Xtr, Xm]), np.concatenate([Ytr, Ym]),
                 Xva, 1.0), Yva, Ava)
    report("orig+mirror-resorted -> orig",
           ridge(np.concatenate([Xtr, Xr]), np.concatenate([Ytr, Yr]),
                 Xva, 1.0), Yva, Ava)

    # self-consistency: does each augmentation fit its own mirrored val set?
    Xvm, Yvm, Avm = mirror(Xva, Yva, Ava, resort=False)
    print("[audit] same-distribution fits (sanity: should all be high):",
          flush=True)
    report("mirror-unsorted -> mirror val",
           ridge(Xm, Ym, Xvm, 1.0), Yvm, Avm)
    report("mirror-resorted -> mirror val",
           ridge(Xr, Yr, Xvm, 1.0), Yvm, Avm)

    # centred targets: pure ranking, level removed from supervision
    print("[audit] centred (ranking-only) targets, orig -> orig:", flush=True)
    report("orig centred -> orig",
           ridge(Xtr, Ytr - Ytr.mean(1, keepdims=True), Xva, 1.0), Yva, Ava)
    report("orig+mirror-resorted centred -> orig",
           ridge(np.concatenate([Xtr, Xr]),
                 np.concatenate([Ytr, Yr]) -
                 np.concatenate([Ytr, Yr]).mean(1, keepdims=True),
                 Xva, 1.0), Yva, Ava)

    # nonlinear headroom: random ReLU features + ridge
    rng = np.random.default_rng(args.seed)
    W1 = (rng.standard_normal((Xtr.shape[1], args.rf_dim)) /
          np.sqrt(Xtr.shape[1])).astype(np.float32)
    b1 = rng.standard_normal(args.rf_dim).astype(np.float32)
    print(f"[audit] random-feature (ReLU, {args.rf_dim}d) headroom:",
          flush=True)

    def rff(X):
        return np.maximum(X @ W1 + b1, 0.0)

    Ftr, Fva = rff(Xtr), rff(Xva)
    report("RF orig -> orig",
           ridge(Ftr, Ytr, Fva, 1e-1), Yva, Ava)
    report("RF orig+mirror-resorted -> orig",
           ridge(np.concatenate([Ftr, rff(Xr)]),
                 np.concatenate([Ytr, Yr]), Fva, 1e-1), Yva, Ava)
    report("RF centred (orig) -> orig",
           ridge(Ftr, Ytr - Ytr.mean(1, keepdims=True), Fva, 1e-1), Yva, Ava)
    report("RF centred (orig+mir) -> orig",
           ridge(np.concatenate([Ftr, rff(Xr)]),
                 np.concatenate([Ytr, Yr]) -
                 np.concatenate([Ytr, Yr]).mean(1, keepdims=True),
                 Fva, 1e-1), Yva, Ava)


if __name__ == "__main__":
    main()
