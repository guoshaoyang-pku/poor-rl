"""EI soft dump: replay top episodes, score obs with the CURRENT policy on GPU,
and write BC shards whose qteach = policy's own centred Q tilted toward the
taken action:

    qc_policy = q_policy - mean_a(q_policy)          (contrast only)
    target_logits = qc_policy + beta * (onehot(act) - 1/K)
    qteach = target_logits        (stored f16; learner recenters + softmaxes)

Why not pure one-hot (ei_dump c1 lesson): sharp labels (p_act=0.76) made the
dueling A head inflate margins via generalization interference; 400 steps
wrecked Q-argmax (eval 1724 -> 921) while the pi head learned fine. Keeping
the policy's own dark knowledge + a bounded tilt (beta~2) nudges ranking
without blowing up margins.

Runs on t1: CPU replay workers + one GPU for policy scoring.
Usage:
  python ei_dump_soft.py --actions-dir <dir with actions_a*.jsonl> \
      --ckpt <policy.pt> --config <yaml> --out-dir <dir> \
      --top-frac 0.10 --max-episodes 4000 --beta 2.0 --gpu 0
"""
import argparse
import glob
import json
import os
import time

import numpy as np


def _replay_obs(rec):
    """CPU worker: replay one episode, return (obses [n,obs_dim], acts [n])
    or None on drift. Env created lazily per process."""
    if not hasattr(_replay_obs, "env"):
        from paths import setup_engine_path
        setup_engine_path()
        from env import DQNEnv
        _replay_obs.env = DQNEnv(seed=None, K=128, max_fruits=80,
                                 boundary=True, obs_format="tokens")
    env = _replay_obs.env
    obs = env.reset(seed=int(rec["seed"]),
                    geom=(int(rec["w"]), int(rec["h"])))
    O, A = [], []
    for a in bytes.fromhex(rec["actions"]):
        a = int(a)
        nobs, _, done, _ = env.step(a)
        O.append(obs); A.append(a)
        obs = nobs
        if done:
            break
    if abs(float(env.score) - float(rec["score"])) > 1e-6:
        return None
    return (np.asarray(O, dtype=np.float32), np.asarray(A, dtype=np.int64),
            float(rec["score"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-frac", type=float, default=0.10)
    ap.add_argument("--max-episodes", type=int, default=4000)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--micro", type=int, default=48)
    ap.add_argument("--shard-size", type=int, default=8192)
    ap.add_argument("--K", type=int, default=128)
    args = ap.parse_args()

    t0 = time.time()
    recs = []
    for fn in sorted(glob.glob(os.path.join(args.actions_dir,
                                            "actions_a*.jsonl"))):
        with open(fn) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("actions"):
                    recs.append(r)
    scores = np.array([r["score"] for r in recs])
    cutoff = float(np.quantile(scores, 1.0 - args.top_frac))
    sel = sorted((r for r in recs if r["score"] >= cutoff),
                 key=lambda r: -r["score"])[: args.max_episodes]
    print(f"[ei-soft] {len(recs)} episodes, cutoff {cutoff:.0f}, "
          f"selected {len(sel)} ({sel[-1]['score']:.0f}.."
          f"{sel[0]['score']:.0f})", flush=True)

    # ---- phase 1: parallel CPU replay ----
    import multiprocessing as mp
    eps, drift = [], 0
    with mp.Pool(args.workers) as pool:
        for i, out in enumerate(pool.imap_unordered(_replay_obs, sel,
                                                    chunksize=4)):
            if out is None:
                drift += 1
                continue
            eps.append(out)
            if (i + 1) % 500 == 0:
                print(f"[ei-soft] replay {i+1}/{len(sel)} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    ntr = sum(len(e[1]) for e in eps)
    print(f"[ei-soft] replayed {len(eps)} episodes -> {ntr:,} transitions, "
          f"drift {drift}, {time.time()-t0:.0f}s", flush=True)

    # ---- phase 2: GPU scoring with the current policy ----
    import torch
    import yaml
    from model import build_model
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(f"cuda:{args.gpu}")
    model = build_model(cfg, 800).to(device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ck.get("state_dict", ck),
                                                strict=False)
    assert not unexpected, unexpected[:5]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[ei-soft] policy loaded ({len(missing)} frozen skipped), "
          f"scoring on {device}...", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    nshard, nwritten = 0, 0
    bufs = {k: [] for k in ("obs", "act", "rew", "nobs", "done", "gam",
                            "qteach", "argmax")}
    nbuf = 0

    def flush():
        nonlocal nshard, nbuf, nwritten
        if nbuf == 0:
            return
        out = {k: np.concatenate(v) for k, v in bufs.items()}
        np.savez(os.path.join(args.out_dir, f"eis_{nshard:05d}.npz"), **out)
        nshard += 1
        nwritten += nbuf
        nbuf = 0
        for v in bufs.values():
            v.clear()

    K = args.K
    with torch.no_grad():
        for O, A, ep_score in eps:
            q_all = []
            for s in range(0, len(O), args.micro):
                x = torch.from_numpy(O[s:s + args.micro]).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    q = model.q_values(x)
                q_all.append(q.float().cpu().numpy())
            q = np.concatenate(q_all)                     # [n,K] f32
            qc = q - q.mean(1, keepdims=True)
            tgt = qc + args.beta * (
                np.eye(K, dtype=np.float32)[A] - 1.0 / K)
            n = len(A)
            bufs["obs"].append(O)
            bufs["act"].append(A)
            # rew/nobs/done/gam unused by the EI loss config; store cheap
            # stand-ins with correct shapes (nobs needed by the loader only)
            bufs["rew"].append(np.full(n, ep_score, dtype=np.float32))
            bufs["nobs"].append(np.zeros_like(O))
            bufs["done"].append(np.zeros(n, dtype=np.float32))
            bufs["gam"].append(np.ones(n, dtype=np.float32))
            bufs["qteach"].append(tgt.astype(np.float16))
            bufs["argmax"].append(np.ones(n, dtype=bool))
            nbuf += n
            if nbuf >= args.shard_size:
                flush()
    flush()
    print(f"[ei-soft] DONE: {nwritten:,} transitions in {nshard} shards -> "
          f"{args.out_dir} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
