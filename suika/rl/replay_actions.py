"""Replay stored action sequences and verify they reproduce the recorded game.

A record is {seed, w, h, actions(hex), score, moves} (wave6+; wave5 records
carry killy instead of h, with the floor fixed at 675). The triple
(seed, board, actions) is the complete state of a game, because the fruit
queue is a function of `seed`, the walls/floor of (w, h), and physics
is deterministic. Sources: runs/*/actions_a*.jsonl (every training episode) and
runs/*/eval_actions.jsonl (every greedy eval episode).

  python replay_actions.py runs/x/eval_actions.jsonl --top 3      # verify
  python replay_actions.py runs/x/actions_a0.jsonl --top 2 --dump out.json
      # --dump writes per-move states (fruits, score) for the visualiser
"""
import argparse
import json
import sys

from paths import setup_engine_path
setup_engine_path()
from env import DQNEnv  # noqa: E402


def record_geom(rec):
    """(width, height) for env.reset; wave5 records stored (w, killy) with a
    fixed 675 floor, which maps exactly to height = 675 - killy."""
    if "h" in rec:
        return int(rec["w"]), int(rec["h"])
    return int(rec["w"]), 675 - int(rec["killy"])


def replay_record(rec, dump=False):
    env = DQNEnv(seed=None, obs_format="tokens")
    env.reset(seed=int(rec["seed"]), geom=record_geom(rec))
    acts = bytes.fromhex(rec["actions"])
    frames = []
    done = False
    n = 0
    for a in acts:
        _, _, done, _ = env.step(int(a))
        n += 1
        if dump:
            st = env.env.get_state()
            frames.append({"score": float(env.score), "fruits": [
                [f["type"], round(f["x"], 1), round(f["y"], 1)]
                for f in st["fruits"]]})
        if done:
            break
    out = {"score": float(env.score), "moves": n, "done": bool(done),
           "match": abs(float(env.score) - float(rec["score"])) < 1e-6
           and n == int(rec["moves"])}
    if dump:
        out["frames"] = frames
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=3, help="verify N best-scoring")
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(args.path) if l.strip()]
    recs = [r for r in recs if r.get("actions")]
    recs.sort(key=lambda r: -r["score"])
    bad, dumped = 0, []
    for r in recs[:args.top]:
        res = replay_record(r, dump=bool(args.dump))
        bad += int(not res["match"])
        w, h = record_geom(r)
        print(f"seed={r['seed']} {w}x{h} recorded="
              f"{r['score']:.0f}/{r['moves']} replayed={res['score']:.0f}/"
              f"{res['moves']} match={res['match']}")
        if args.dump:
            dumped.append({**{k: r[k] for k in ("seed", "score", "moves")},
                           "w": w, "h": h, "frames": res["frames"]})
    if args.dump:
        json.dump(dumped, open(args.dump, "w"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
