"""Reproduce learner_qwen.train_step memory blowup single-process (gpu7).

Prints memory_allocated after every phase/chunk to find where the ~37GB
comes from and whether it accumulates across micro chunks.
"""
import yaml
import numpy as np
import torch

cfg = yaml.safe_load(open("configs/w4_qwen_text.yaml"))
from model import build_model, param_count


def mb(x):
    return f"{x / 2 ** 20:8.0f} MiB"


m = build_model(cfg, 800)
print(f"[build] total={param_count(m) / 1e6:.1f}M "
      f"trainable={m.trainable_param_count() / 1e6:.2f}M", flush=True)
gc_on = [type(mod).__name__ for mod in m.trunk.modules()
         if getattr(mod, "gradient_checkpointing", False)]
print(f"[ckpt] modules with gradient_checkpointing flag: "
      f"{len(gc_on)} {gc_on[:3]}", flush=True)

dev = "cuda"
m = m.to(dev)
target = build_model(cfg, 800).to(dev)
target.load_state_dict(m.state_dict())
for p in target.parameters():
    p.requires_grad_(False)
head_p, lora_p = [], []
for n, p in m.named_parameters():
    if p.requires_grad:
        (lora_p if "lora_" in n else head_p).append(p)
opt = torch.optim.AdamW([{"params": head_p}, {"params": lora_p}], lr=1e-4)
print(f"[mem] after build online+target+opt: {mb(torch.cuda.memory_allocated())}",
      flush=True)

B, T5, micro, K = 1024, 800, 32, 128
rng = np.random.default_rng(0)
# synthetic but realistic: ~55 board fruits + cur/next rows, type in [0,10)
obs = -np.ones((B, T5), dtype=np.float32)
for i in range(B):
    n = int(rng.integers(40, 70))
    obs[i, :5] = [rng.integers(0, 5), 0.5, 0.5, 0, 0]
    obs[i, 5:10] = [rng.integers(0, 5), 0.5, 0.5, 0, 0]
    for j in range(n):
        obs[i, 10 + j * 5:15 + j * 5] = [rng.integers(0, 11), rng.random(),
                                         rng.random(), 0, 0]
o = torch.from_numpy(obs).to(dev)
no = torch.from_numpy(obs).to(dev)
act = torch.from_numpy(rng.integers(0, K, B)).to(dev)
rew = torch.rand(B).to(dev)
done = torch.zeros(B).to(dev)
gam = torch.ones(B).to(dev)
w = torch.ones(B).to(dev)
print(f"[mem] after batch tensors: {mb(torch.cuda.memory_allocated())}", flush=True)

print("[phase] target pass (no_grad)", flush=True)
a_star = torch.empty(B, dtype=torch.long, device=dev)
nxt = torch.empty(B, 1, device=dev)
with torch.no_grad():
    for s in range(0, B, micro):
        e = min(s + micro, B)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a_star[s:e] = m(no[s:e]).mean(-1).argmax(1)
            zt = target(no[s:e])
        nxt[s:e] = zt.gather(1, a_star[s:e].view(-1, 1, 1).expand(-1, 1, 1)).squeeze(1)
        if s < 3 * micro or e == B:
            print(f"  [tgt] chunk {s}-{e}: alloc={mb(torch.cuda.memory_allocated())} "
                  f"peak={mb(torch.cuda.max_memory_allocated())}", flush=True)
tgt = rew.unsqueeze(1) + (gam * (1 - done)).unsqueeze(1) * nxt
print(f"[mem] after target pass: alloc={mb(torch.cuda.memory_allocated())} "
      f"peak={mb(torch.cuda.max_memory_allocated())}", flush=True)

print("[phase] online pass (grad accum)", flush=True)
opt.zero_grad(set_to_none=True)
preds = torch.empty(B, 1, device=dev)
for s in range(0, B, micro):
    e = min(s + micro, B)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = m(o[s:e])
        pred = z.gather(1, act[s:e].view(-1, 1, 1).expand(-1, 1, 1)).squeeze(1)
    loss = (pred * w[s:e]).mean() * (e - s) / B
    loss.backward()
    preds[s:e] = pred.detach()
    print(f"  [fwd] chunk {s}-{e}: alloc={mb(torch.cuda.memory_allocated())} "
          f"peak={mb(torch.cuda.max_memory_allocated())}", flush=True)
opt.step()
print(f"[mem] after step: alloc={mb(torch.cuda.memory_allocated())} "
      f"peak={mb(torch.cuda.max_memory_allocated())}", flush=True)
print("[ok] single train step completed without OOM", flush=True)
