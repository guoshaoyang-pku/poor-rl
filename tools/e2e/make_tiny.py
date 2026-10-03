#!/usr/bin/env python
"""Tiny random Qwen3.8 (qwen3_5 text) checkpoint for CPU tests: same layer pattern, real vocab, tiny widths.

python make_tiny.py --src /home/tione/guoshaoyang/models/Qwen3.8-27B --out DIR [--layers 4] [--hidden 128]
"""
import argparse
import sys

for _m in ("fla", "causal_conv1d", "flash_attn"):
    sys.modules[_m] = None

import torch  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

cfg = AutoConfig.from_pretrained(a.src)
t = cfg.text_config if hasattr(cfg, "text_config") else cfg
t.hidden_size = a.hidden
t.intermediate_size = 2 * a.hidden
t.num_hidden_layers = a.layers
# keep the real 3:1 linear/full pattern (last layer full attention)
t.layer_types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(a.layers)]
t.num_attention_heads = 2          # head_dim (256) and the partial/mrope rope layout stay as in the real model
t.num_key_value_heads = 1
for k, v in (("linear_num_key_heads", 1), ("linear_num_value_heads", 2)):
    if hasattr(t, k):
        setattr(t, k, v)
if hasattr(t, "full_attention_interval"):
    t.full_attention_interval = 4
torch.manual_seed(a.seed)
m = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
with torch.no_grad():  # larger-than-default init so a few LoRA steps visibly move the logprobs
    for n, p in m.named_parameters():
        if p.ndim == 2 and "embed" not in n:
            p.normal_(0, 0.05)
m.save_pretrained(a.out)
print("saved", a.out, type(m).__name__, sum(p.numel() for p in m.parameters()) / 1e6, "M params",
      "layer_types", t.layer_types)
