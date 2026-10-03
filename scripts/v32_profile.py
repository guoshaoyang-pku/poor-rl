"""Kernel-level profile of one micro-batch row (v32_gate data/model helpers). usage: v32_profile.py KIND SB:CKPT"""
import sys, torch
sys.path.insert(0, __import__("os").path.dirname(__file__))
import v32_gate as G
from transformers import AutoTokenizer
kind, cfg = sys.argv[1], sys.argv[2]
sbt, ck = int(cfg.split(":")[0]), cfg.split(":")[1]
tok = AutoTokenizer.from_pretrained(G.CK)
rows = G.build_rows(tok, [kind])
model, ref = G.load_models(with_ref=True)
inputs = G.make_inputs(rows[kind], seed=1)
for _ in range(2):
    G.run_path(model, ref, inputs, sbt, ck)
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    r = G.run_path(model, ref, inputs, sbt, ck)
print("wall", r[4])
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=45, max_name_column_width=70))
