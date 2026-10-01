"""Remote-side digest of one run dir (stdlib only; piped over ssh, never run locally).

usage: python3 - RUN_DIR [TOPK] < digest_actions.py

Writes two small files into RUN_DIR so the laptop never has to pull the raw
per-actor logs (actions_a*.jsonl grows ~GB/day):
  digest_top_actions.jsonl  top-K training episodes by score, with full action bytes
                            (replayable via replay_actions.py: seed + w + killy + actions)
  digest_episode_bins.jsonl 10-min bins of training episodes (n, mean/p90/max score,
                            mean moves) overall and per killy bucket
"""
import glob
import heapq
import json
import os
import sys

run = sys.argv[1]
topk = int(sys.argv[2]) if len(sys.argv) > 2 else 100
BIN_S = 600


def _lines(path):
    with open(path) as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except ValueError:
                continue  # partially written tail line


heap, n_seen = [], 0
for f in glob.glob(os.path.join(run, "actions_a*.jsonl")):
    for r in _lines(f):
        n_seen += 1
        key = (float(r["score"]), n_seen)
        if len(heap) < topk:
            heapq.heappush(heap, (key, r))
        elif key > heap[0][0]:
            heapq.heapreplace(heap, (key, r))
top = [r for _, r in sorted(heap, key=lambda x: -x[0][0])]
tmp = os.path.join(run, "digest_top_actions.jsonl.tmp")
with open(tmp, "w") as fh:
    for r in top:
        fh.write(json.dumps(r) + "\n")
os.replace(tmp, os.path.join(run, "digest_top_actions.jsonl"))


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


bins = {}
for f in glob.glob(os.path.join(run, "episodes_a*.jsonl")):
    for r in _lines(f):
        b = int(r["t"] // BIN_S)
        kb = None if "h" not in r else int(r["h"] // 50 * 50)
        bins.setdefault(b, {}).setdefault(kb, []).append((float(r["score"]), float(r["moves"])))


def _stat(v):
    sc = [s for s, _ in v]
    return {"n": len(v), "mean": sum(sc) / len(sc), "p90": _pct(sc, 0.9), "max": max(sc),
            "moves": sum(m for _, m in v) / len(v)}


tmp = os.path.join(run, "digest_episode_bins.jsonl.tmp")
with open(tmp, "w") as fh:
    for b in sorted(bins):
        allv = [x for v in bins[b].values() for x in v]
        row = {"t_min": b * BIN_S / 60.0, **_stat(allv)}
        if len(bins[b]) > 1 or None not in bins[b]:
            row["by_h"] = {str(k): _stat(v) for k, v in sorted(bins[b].items()) if k is not None}
        fh.write(json.dumps(row) + "\n")
os.replace(tmp, os.path.join(run, "digest_episode_bins.jsonl"))
print("digest %s: actions_seen=%d top=%d (best %.0f) bins=%d"
      % (run, n_seen, len(top), top[0]["score"] if top else 0, len(bins)))
