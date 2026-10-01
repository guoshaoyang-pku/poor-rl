"""BC ceiling probe: what regret can a plain MLP student reach on the same data?

Runs the exact v3 objective (configs/w4_bc_qwen_v3.yaml) on the raw 800d token
obs with a 3-layer MLP head (no LM, no text interface, local MPS/CPU). This is
the "data/input ceiling" for the BC target: if a plain MLP gets regret well
below the constant-policy baseline (~31) then the ranking IS in the data and
the LM's job is capacity/optimization; if even the MLP stalls near ~22 (the
random-feature readout), the target itself is near-tie-noise dominated.

Usage:
  python3 probe_mlp_ceiling.py --split /tmp/bc_audit_split.npz
"""
import argparse

import numpy as np
import torch
import torch.nn as nn

DEV = ("mps" if torch.backends.mps.is_available()
       else ("cuda" if torch.cuda.is_available() else "cpu"))


def smooth_l1(a, b):
    d = (a - b).abs()
    return torch.where(d < 1.0, 0.5 * d * d, d - 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam-adv", type=float, default=1.0)
    ap.add_argument("--lam-v", type=float, default=1.0)
    ap.add_argument("--adv-ref", type=float, default=12.0)
    ap.add_argument("--tau", type=float, default=1.0, help="pi KD temperature")
    ap.add_argument("--lam-pi", type=float, default=1.0)
    ap.add_argument("--lam-qrank", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    z = np.load(args.split)
    Xtr, Ytr, Atr = z["Xtr"], z["Ytr"], z["Atr"]
    Xva, Yva, Ava = z["Xva"], z["Yva"], z["Ava"]
    # test shards come after the val shards in the audit split
    if "Xte" in z:
        Xte, Yte, Ate = z["Xte"], z["Yte"], z["Ate"]
    else:
        Xte = Yte = Ate = None
    K = Ytr.shape[1]
    print(f"[mlp] dev={DEV} train={Xtr.shape} val={Xva.shape} K={K}", flush=True)

    dev = torch.device(DEV)
    Xtr_t = torch.from_numpy(Xtr).to(dev)
    Ytr_t = torch.from_numpy(Ytr).to(dev)
    Xva_t = torch.from_numpy(Xva).to(dev)

    net = nn.Sequential(
        nn.Linear(Xtr.shape[1], args.hidden), nn.SiLU(),
        nn.Linear(args.hidden, args.hidden // 2), nn.SiLU(),
        nn.Linear(args.hidden // 2, K)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)

    n = len(Xtr_t)
    steps_per_epoch = max(1, n // args.batch)
    total = args.epochs * steps_per_epoch
    g = torch.Generator(device="cpu").manual_seed(args.seed)

    @torch.no_grad()
    def evaluate(X, Y, A, name):
        net.eval()
        zs = []
        for s in range(0, len(X), 1024):
            zs.append(net(X[s:s + 1024]))
        Z = torch.cat(zs)
        za = Z.argmax(1)
        Yt = torch.from_numpy(Y).to(dev)
        At = torch.from_numpy(A).to(dev)
        reg = float((Yt.max(1).values - Yt.gather(1, za[:, None]).squeeze(1)
                     ).mean())
        agree = float((za == At).float().mean())
        c_t = Yt - Yt.mean(1, keepdim=True)
        c_p = Z - Z.mean(1, keepdim=True)
        rmse = float(((c_p - c_t) ** 2).mean().sqrt())
        print(f"  [{name:5s}] q_regret={reg:7.2f} agree={agree:.3f} "
              f"contrast_rmse={rmse:6.2f}", flush=True)
        net.train()
        return reg

    print("[mlp] reference: constant action 31.1, RF readout 22.3, "
          "model step1200 27.8, gate < 10", flush=True)
    step = 0
    for ep in range(args.epochs):
        perm = torch.randperm(n, generator=g)
        acc = [0.0, 0.0, 0.0]
        for s in range(0, n - args.batch + 1, args.batch):
            idx = perm[s:s + args.batch]
            xb = Xtr_t[idx]
            yb = Ytr_t[idx]
            ab = torch.from_numpy(Atr[s:s + args.batch]).to(dev)
            Z = net(xb)
            lvl_t = yb.mean(1)
            c_t = yb - lvl_t[:, None]
            lvl_p = Z.mean(1)
            l_adv = smooth_l1((Z - lvl_p[:, None]) / args.adv_ref,
                              c_t / args.adv_ref).mean()
            l_v = smooth_l1(lvl_p / 1000.0, lvl_t / 1000.0).mean()
            p_t = torch.softmax(c_t / args.tau, dim=1)
            l_pi = -(p_t * torch.log_softmax(Z / args.tau, dim=1)).sum(
                1).mean()
            l_qr = -(p_t * torch.log_softmax(Z, dim=1)).sum(1).mean()
            loss = (args.lam_adv * l_adv + args.lam_v * l_v
                    + args.lam_pi * l_pi + args.lam_qrank * l_qr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            acc[0] += float(loss)
            acc[1] += float(l_adv)
            acc[2] += float(l_pi)
            step += 1
        print(f"[mlp] epoch {ep+1}/{args.epochs} loss={acc[0]/steps_per_epoch:.3f} "
              f"adv={acc[1]/steps_per_epoch:.3f} pi={acc[2]/steps_per_epoch:.3f}",
              flush=True)
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            evaluate(Xva_t, Yva, Ava, "val")
            if Xte is not None:
                evaluate(torch.from_numpy(Xte).to(dev), Yte, Ate, "test")


if __name__ == "__main__":
    main()
