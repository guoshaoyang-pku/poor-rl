"""Is the BC failure a feature problem or a head/optimization problem?

Take the trunk hidden features that feed the v/a/pi heads from the current BC
policy, then fit closed-form ridge readouts (hidden -> teacher Q) and compare
against the model's own heads on held-out shards. Regret is the yardstick:
constant-policy baseline 30.4, untrained net 29.96, gate < 10.

  model heads        regret ~= 30  -> heads are not extracting ranking
  ridge full         if << 30      -> features DO contain the ranking
  ridge centred      ranking-only linear readout (level removed)
"""
import argparse
import glob
import os

import numpy as np
import torch
import yaml

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_model            # noqa: E402
from model_qwen import warmup            # noqa: E402


def smooth_l1_mean(a, b):
    d = np.abs(a - b)
    return float(np.mean(np.where(d < 1.0, 0.5 * d * d, d - 0.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-shards", type=int, default=8)
    ap.add_argument("--val-shards", type=int, default=2)
    ap.add_argument("--max-n", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda"
    model = build_model(cfg, int(cfg["T"]) * 5).to(device)
    ck = torch.load(args.policy, map_location="cpu", weights_only=False)
    missing = model.load_state_dict(ck["state_dict"], strict=False)
    print(f"[probe] loaded {args.policy}: missing={len(missing.missing_keys)} "
          f"unexpected={len(missing.unexpected_keys)}", flush=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    warmup(model, batches=(args.batch,), log=lambda s: print(s, flush=True))

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    tf = files[:args.train_shards]
    vf = files[args.train_shards:args.train_shards + args.val_shards]

    def load(flist, cap):
        X, Y, A = [], [], []
        n = 0
        for f in flist:
            d = np.load(f)
            obs, q, act = d["obs"], d["qteach"].astype(np.float32), d["act"]
            k = min(len(obs), max(0, cap - n))
            X.append(obs[:k]); Y.append(q[:k]); A.append(act[:k])
            n += k
            if n >= cap:
                break
        return (np.concatenate(X), np.concatenate(Y), np.concatenate(A))

    Xtr, Ytr, _ = load(tf, args.max_n)
    Xva, Yva, Ava = load(vf, args.max_n // 2)
    print(f"[probe] train={len(Xtr):,} val={len(Xva):,} "
          f"obs_dim={Xtr.shape[1]}", flush=True)

    def run(batches_fn, X, cap=None):
        out, n = [], 0
        with torch.no_grad():
            for s in range(0, len(X), args.batch):
                xb = torch.from_numpy(X[s:s + args.batch]).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = batches_fn(xb)
                o = o.float().cpu().numpy()
                out.append(o)
                n += len(o)
                if cap and n >= cap:
                    break
        return np.concatenate(out)

    Ftr = run(lambda x: model._hidden(x), Xtr)
    Fva = run(lambda x: model._hidden(x), Xva)
    Qva = run(lambda x: model.q_values(x).squeeze(-1), Xva)
    Pva = run(lambda x: model.pi_head(model._hidden(x)), Xva)

    def report(name, Qh):
        za = Qh.argmax(1)
        reg = float((Yva.max(1) - Yva[np.arange(len(Yva)), za]).mean())
        print(f"  {name:24s} val_qd={smooth_l1_mean(Qh, Yva):8.2f} "
              f"q_regret={reg:7.2f} agree={float((za == Ava).mean()):.3f}",
              flush=True)

    print("[probe] held-out readouts (regret: constant 30.4, gate <10):",
          flush=True)
    report("model Q head", Qva)
    report("model pi head", Pva)

    # probe on the raw obs (what the trunk got as input, linear baseline)
    Ftr_o, Fva_o = Xtr.astype(np.float32), Xva.astype(np.float32)

    def ridge(F, Y, Fv, alpha):
        Fb = np.concatenate([F, np.ones((len(F), 1), np.float32)], 1).astype(
            np.float64)
        G = Fb.T @ Fb + alpha * np.eye(Fb.shape[1], dtype=np.float64)
        W = np.linalg.solve(G, Fb.T @ Y.astype(np.float64))
        Fvb = np.concatenate([Fv, np.ones((len(Fv), 1), np.float32)], 1)
        return (Fvb.astype(np.float64) @ W).astype(np.float32)

    for alpha in (1e-2, 1.0, 100.0):
        report(f"ridge hidden a={alpha:g}", ridge(Ftr, Ytr, Fva, alpha))
    report("ridge hidden centred",
           ridge(Ftr, Ytr - Ytr.mean(1, keepdims=True), Fva, 1.0))
    report("ridge raw-obs a=1", ridge(Ftr_o, Ytr, Fva_o, 1.0))
    report("ridge raw-obs centred",
           ridge(Ftr_o, Ytr - Ytr.mean(1, keepdims=True), Fva_o, 1.0))

    # best-constant reference: always pick the globally most-picked column
    top_col = int(np.bincount(Ava, minlength=Yva.shape[1]).argmax())
    reg_c = float((Yva.max(1) - Yva[:, top_col]).mean())
    print(f"  reference: best-constant({top_col}) q_regret={reg_c:.2f}  "
          f"teacher-top1 share={float((Yva.argmax(1) == Ava).mean()):.3f}")


if __name__ == "__main__":
    main()
