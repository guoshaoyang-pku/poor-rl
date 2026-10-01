"""Expert-iteration dump: replay top-scoring episodes from actions_a*.jsonl
into BC-format shards (hard-label soft targets for the taken action).

Each record in actions_a*.jsonl is (seed, w, h, actions, score, moves) --
a complete game (physics is deterministic; verified bit-exact on-node with
replay_actions.py). We replay the top episodes and write the same 8-field
shard layout bc_learner_qwen expects:
  obs [N,T*5] f32, act [N] i64, rew/nobs/done/gam (real, from replay),
  qteach [N,K] f16 = one_hot(act) * boost   (centered softmax target
       p_act = sigmoid-ish ~0.76 at boost=6, tau=1),
  argmax [N] bool = True                    (act IS the label here)

Usage:
  python ei_dump.py --run-dir runs/x/actions_dir --out-dir out/ \
      --top-frac 0.10 --max-episodes 4000 --workers 48
"""
import argparse
import glob
import json
import os
import time

import numpy as np

_ENV = None


def _env():
    global _ENV
    if _ENV is None:
        from paths import setup_engine_path
        setup_engine_path()
        from env import DQNEnv
        _ENV = DQNEnv(seed=None, K=128, max_fruits=80, boundary=True,
                      obs_format="tokens")
    return _ENV


def _replay(rec, K):
    """Replay one episode; return dict of per-move arrays or None on drift."""
    env = _env()
    obs = env.reset(seed=int(rec["seed"]),
                    geom=(int(rec["w"]), int(rec["h"])))
    acts = bytes.fromhex(rec["actions"])
    O, A, R, NO, D = [], [], [], [], []
    done = False
    for a in acts:
        a = int(a)
        nobs, r, done, _ = env.step(a)
        O.append(obs); A.append(a); R.append(r); NO.append(nobs)
        D.append(float(done))
        obs = nobs
        if done:
            break
    if abs(float(env.score) - float(rec["score"])) > 1e-6:
        return None     # replay drifted -- skip this episode
    return {
        "obs": np.asarray(O, dtype=np.float32),
        "act": np.asarray(A, dtype=np.int64),
        "rew": np.asarray(R, dtype=np.float32),
        "nobs": np.asarray(NO, dtype=np.float32),
        "done": np.asarray(D, dtype=np.float32),
        "gam": np.ones(len(A), dtype=np.float32),
    }


def _worker(args):
    rec, K, boost = args
    out = _replay(rec, K)
    if out is None:
        return None
    n = len(out["act"])
    qt = np.zeros((n, K), dtype=np.float16)
    qt[np.arange(n), out["act"]] = np.float16(boost)
    out["qteach"] = qt
    out["argmax"] = np.ones(n, dtype=bool)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="dir containing actions_a*.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-frac", type=float, default=0.10)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--max-episodes", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--boost", type=float, default=6.0,
                    help="one-hot magnitude; centered softmax p_act~0.76 @6")
    ap.add_argument("--shard-size", type=int, default=8192)
    ap.add_argument("--K", type=int, default=128)
    args = ap.parse_args()

    t0 = time.time()
    recs = []
    for fn in sorted(glob.glob(os.path.join(args.run_dir,
                                            "actions_a*.jsonl"))):
        with open(fn) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue          # partial tail of a live file
                if r.get("actions"):
                    recs.append(r)
    scores = np.array([r["score"] for r in recs])
    cutoff = max(float(np.quantile(scores, 1.0 - args.top_frac)),
                 args.min_score)
    sel = [r for r in recs if r["score"] >= cutoff]
    sel.sort(key=lambda r: -r["score"])
    sel = sel[: args.max_episodes]
    print(f"[ei-dump] {len(recs)} episodes, cutoff {cutoff:.0f} "
          f"(top {args.top_frac:.0%}), selected {len(sel)} "
          f"(score {sel[-1]['score']:.0f}..{sel[0]['score']:.0f})",
          flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    drift = 0
    bufs = {k: [] for k in ("obs", "act", "rew", "nobs", "done", "gam",
                            "qteach", "argmax")}
    nbuf, nshard, ntrans = 0, 0, 0

    def flush():
        nonlocal nbuf, nshard, ntrans
        if nbuf == 0:
            return
        out = {k: np.concatenate(v) for k, v in bufs.items()}
        fn = os.path.join(args.out_dir, f"ei_{nshard:05d}.npz")
        np.savez(fn, **out)
        nshard += 1
        ntrans += nbuf
        nbuf = 0
        for v in bufs.values():
            v.clear()

    import multiprocessing as mp
    with mp.Pool(args.workers) as pool:
        for i, out in enumerate(pool.imap_unordered(
                _worker, [(r, args.K, args.boost) for r in sel],
                chunksize=4)):
            if out is None:
                drift += 1
                continue
            for k, v in bufs.items():
                v.append(out[k])
            nbuf += len(out["act"])
            if nbuf >= args.shard_size:
                flush()
            if (i + 1) % 250 == 0:
                print(f"[ei-dump] {i+1}/{len(sel)} replayed, "
                      f"drift {drift}, {time.time()-t0:.0f}s", flush=True)
    flush()
    print(f"[ei-dump] DONE: {ntrans:,} transitions in {nshard} shards "
          f"-> {args.out_dir}  (drift-skipped {drift}, "
          f"{time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
