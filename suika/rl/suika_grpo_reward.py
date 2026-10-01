"""Suika nothink GRPO reward (rlforge protocol): critic-scored decimal drops.

Contract (rlforge.rewards): fn(completions, prompts, completion_ids, answer,
cap, **dataset_columns) -> list[float].

Dataset rows (build_grpo_prompts.py):
  {"prompt": [system/user chat messages], "answer": "<teacher argmax col>",
   "obs": "<b64 npz of compact obs [T*5] f16>", "source": "w4bc_teacher"}

Reward v1 (text arm, nothink):
  parse completion as decimal -> col (decimal_to_col);
  score = Q_critic(state, col) from the FROZEN BC critic (model_qwen.QwenQ,
  policy.pt full weights) — group-relative GRPO advantage over G candidates
  of the same state then equals a listwise Q-ranking objective with
  on-policy candidate sampling. Also directly attacks the measured decode
  fork (Q head 1540 vs token path 500): the generator is paid to agree with
  the Q head.
  penalties (rlforge contract): len(ids) >= cap -> -2; unparseable -> -0.5.

The critic is loaded lazily once per trainer process (CRITIC env:
path to policy.pt; CRITIC_DEVICE: cuda:N). Q values enter GRPO only through
per-group z-scores, so raw scale does not matter.
"""
import base64
import io
import json
import os
import re
import time

import numpy as np

NUM = re.compile(r"-?\d+(?:\.\d+)?")
_critic = None


def _load_critic():
    global _critic
    if _critic is not None:
        return _critic
    import torch
    import model_qwen
    path = os.environ["CRITIC"]
    dev = os.environ.get("CRITIC_DEVICE", "cuda:0")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {}) if isinstance(ck, dict) else {}
    K = int(cfg.get("K", 128))
    model_path = os.environ["CRITIC_MODEL_PATH"]
    m = model_qwen.QwenQ(K, model_path=model_path,
                         lora_r=int(cfg.get("lora_r", 64)),
                         lora_alpha=int(cfg.get("lora_alpha", 128)))
    sd = ck["state_dict"] if "state_dict" in ck else ck
    m.load_state_dict(sd, strict=False)
    m.eval().to(dev)
    for p in m.parameters():
        p.requires_grad_(False)
    _critic = (m, torch, dev)
    return _critic


def _decode_obs(obs_b64):
    raw = base64.b64decode(obs_b64)
    return np.load(io.BytesIO(raw))["obs"].astype(np.float32)


def suika_reward(completions, prompts, completion_ids, answer, cap=8,
                 obs=None, **kwargs):
    stats = {"t": round(time.time(), 1), "n": 0, "parsed": 0,
             "unparsed": 0, "trunc": 0, "teacher_hit": 0}
    rewards = [0.0] * len(completions)
    # group completions by prompt index: one critic forward per unique state
    uniq = {}
    for i, ob in enumerate(obs):
        uniq.setdefault(ob, []).append(i)          # b64 string is the key
    m, torch, dev = _load_critic()
    q_cache = {}
    with torch.no_grad():
        for key, idxs in uniq.items():
            x = torch.from_numpy(_decode_obs(obs[idxs[0]])).unsqueeze(0).to(dev)
            q_cache[key] = m.q_values(x)[0].float().cpu().numpy()

    for i, (comp, ids, gold, ob) in enumerate(
            zip(completions, completion_ids, answer, obs)):
        stats["n"] += 1
        if len(ids) >= cap:
            rewards[i] = -2.0
            stats["trunc"] += 1
            continue
        text = comp if isinstance(comp, str) else "".join(
            c.get("content", "") for c in comp)
        hit = NUM.search(text)
        if not hit:
            rewards[i] = -0.5
            stats["unparsed"] += 1
            continue
        x = float(hit.group(0))
        if not (0.0 <= x <= 1.0):
            rewards[i] = -0.5
            stats["unparsed"] += 1
            continue
        col = int(min(127, max(0, np.floor(x * 128))))
        q = q_cache[ob]
        rewards[i] = float(q[col])
        stats["parsed"] += 1
        if col == int(gold):
            stats["teacher_hit"] += 1
    log = os.environ.get("RLFORGE_TASK_LOG", "")
    if log:
        try:
            with open(log, "a") as fh:
                fh.write(json.dumps(stats) + "\n")
        except OSError:
            pass
    return rewards
