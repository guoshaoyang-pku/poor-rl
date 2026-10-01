"""Profile the Qwen arm forward path: tokenize vs pure fwd vs full q_values."""
import sys
import time

import numpy as np
import torch
import yaml

cfg = yaml.safe_load(open("configs/w4_qwen_text.yaml"))
from model import build_model
from env import DQNEnv
from qwen_text import build_user_text, encode_prompt_batch

m = build_model(cfg, 800).to("cuda").eval()

# realistic obs: play 60 random moves
e = DQNEnv(seed=7, K=128, max_fruits=80, boundary=True, obs_format="tokens")
obs = e.reset(seed=7)
rng = np.random.default_rng(0)
moves = 0
while moves < 60:
    o, r, d, info = e.step(int(rng.integers(128)))
    moves += 1
    if d:
        obs = e.reset(seed=int(rng.integers(10 ** 6)))
    else:
        obs = o
txt = build_user_text(obs)
print(f"[text] {len(txt)} chars, board rows={txt.count('|') + 1}", flush=True)

enc = encode_prompt_batch(np.stack([obs] * 64), m.tokenizer)
print(f"[tokens] padded shape={tuple(enc['input_ids'].shape)}", flush=True)

t0 = time.time()
for _ in range(3):
    encode_prompt_batch(np.stack([obs] * 64), m.tokenizer)
print(f"[tokenize] 64 states: {(time.time() - t0) / 3 * 1000:.0f} ms", flush=True)

ids = enc["input_ids"].to("cuda")
mask = enc["attention_mask"].to("cuda")
with torch.no_grad():
    for _ in range(2):
        m.trunk(input_ids=ids, attention_mask=mask, use_cache=False)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        m.trunk(input_ids=ids, attention_mask=mask, use_cache=False)
    torch.cuda.synchronize()
    print(f"[fwd-pure] batch64: {(time.time() - t0) / 5 * 1000:.0f} ms", flush=True)

x = torch.from_numpy(np.stack([obs] * 64)).to("cuda")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for _ in range(2):
        m.q_values(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        m.q_values(x)
    torch.cuda.synchronize()
    print(f"[fwd-full] batch64 incl tokenize+H2D: {(time.time() - t0) / 5 * 1000:.0f} ms",
          flush=True)
print("[maxmem]", round(torch.cuda.max_memory_allocated() / 2 ** 30, 1), "GiB")
