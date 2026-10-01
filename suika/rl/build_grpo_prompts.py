"""Build the suika nothink GRPO prompt dataset from BC text shards.

Each row: {"prompt": [system/user messages], "answer": "<teacher argmax col>",
           "obs": "<b64 npz {obs: [T*5] f16}>", "source": "w4bc_teacher"}

- states are sampled uniformly across BC shards (teacher eps-mixed visits at
  killy=200), so the distribution matches the BC pretrain;
- the obs column lets the reward reconstruct the exact critic input
  (model_qwen.QwenQ consumes compact obs directly);
- teacher argmax goes in `answer` for logging (teacher_hit in the reward).

Usage:
  python build_grpo_prompts.py --shards <bc_data/inbox> --out data/suika_grpo.jsonl \
      --n 120000 [--n-text 96]
"""
import argparse
import base64
import io
import json
import os

import numpy as np

from paths import setup_engine_path
setup_engine_path()
import qwen_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=120000)
    ap.add_argument("--n-text", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.shards) if f.endswith(".npz"))
    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    written = 0
    with open(args.out, "w") as fh:
        while written < args.n:
            f = files[int(rng.integers(len(files)))]
            d = np.load(os.path.join(args.shards, f))
            m = len(d["act"])
            take = min(64, m, args.n - written)
            idx = rng.choice(m, size=take, replace=False)
            for i in idx:
                obs = d["obs"][i].astype(np.float32)
                buf = io.BytesIO()
                np.savez(buf, obs=obs.astype(np.float16))
                row = {
                    "prompt": qwen_text.build_messages(obs, args.n_text),
                    "answer": str(int(d["qteach"][i].argmax())),
                    "obs": base64.b64encode(buf.getvalue()).decode("ascii"),
                    "source": "w4bc_teacher",
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
    print(f"wrote {written} rows -> {args.out}")


if __name__ == "__main__":
    main()
