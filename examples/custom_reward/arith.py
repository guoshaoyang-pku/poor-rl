"""Minimal custom reward for rlforge: arithmetic with a strict answer tag.

Dataset rows: {"prompt": "Compute 17*23. Finish with <answer>391</answer>.",
               "answer": "391", "source": "arith"}

Run:
    python -m rlforge.trainer --model <model> --train train.jsonl \
        --reward examples.custom_reward.arith:arith_reward ...

This file documents the contract a reward function must satisfy:
  * signature (completions, prompts, completion_ids, answer, cap, **dataset_columns)
  * return one float per completion
  * use len(ids) >= cap to detect truncation (the trainer's max_completion_length)
  * any extra dataset column arrives as a keyword argument (here: source)
  * optionally write per-call stats to $RLFORGE_TASK_LOG for the auto-report
"""
import json
import os
import re
import time

TAG = re.compile(r"<answer>\s*([-0-9.]+)\s*</answer>")
TRUNCATED_REWARD = -2.0


def arith_reward(completions, prompts, completion_ids, answer, cap=16384, **kwargs):
    rewards, stats = [], {"t": round(time.time(), 1), "n": 0, "correct": 0,
                          "unparsed": 0, "trunc": 0}
    for comp, ids, gold in zip(completions, completion_ids, answer):
        stats["n"] += 1
        if len(ids) >= cap:
            rewards.append(TRUNCATED_REWARD)
            stats["trunc"] += 1
            continue
        text = comp if isinstance(comp, str) else "".join(
            m.get("content", "") for m in comp)
        m = TAG.findall(text)
        if not m:
            rewards.append(-0.5)
            stats["unparsed"] += 1
        elif abs(float(m[-1]) - float(gold)) < 1e-9:
            rewards.append(1.0)
            stats["correct"] += 1
        else:
            rewards.append(0.0)
    log = os.environ.get("RLFORGE_TASK_LOG", "")
    if log:
        try:
            with open(log, "a") as fh:
                fh.write(json.dumps(stats) + "\n")
        except OSError:
            pass
    return rewards
