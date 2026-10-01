"""Is torch.compile silently falling back to eager in the tf_xl learner?

Observed in profile_tf_stage.log:

  torch._dynamo hit config.recompile_limit (8)
  last reason: 0/7: GLOBAL_STATE changed: grad_mode autocast(cuda)_enabled

The learner calls the compiled online net under two global states per step
(grad+autocast for obs, no_grad+autocast for the double-DQN argmax) and the
compiled target net under a third. Dynamo keys its cache on that state, so
each new (grad_mode, autocast) combination is a fresh compile; once the limit
is hit the wrapper permanently runs the eager fallback while still looking
compiled. If true, production tf_xl never actually ran compiled.

This measures, in one fresh process per configuration:
  eager          no compile at all
  compile_both   compile online + target (production)
  compile_lim64  production + torch._dynamo.config.recompile_limit = 64
  compile_online compile the online net only, target left eager

and prints the dynamo counters so the graph count is visible, not inferred.

  CUDA_VISIBLE_DEVICES=4 python probe_compile_modes.py --config configs/w3c_tf_xl.yaml
"""
import argparse
import json
import os
import time

import torch

_LAUNCH_CWD = os.getcwd()

import yaml  # noqa: E402

from model import build_model  # noqa: E402
from bench_tf_throughput import make_batch, make_targets, timed_steps, _abs  # noqa: E402


def dynamo_graphs():
    try:
        from torch._dynamo.utils import counters
        st = counters.get("stats", {})
        return {k: int(v) for k, v in st.items() if v}
    except Exception as e:      # counters API moves between versions
        return {"err": str(e)[:60]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/w3c_tf_xl.yaml")
    ap.add_argument("--mode", required=True,
                    choices=["eager", "compile_both", "compile_lim64",
                             "compile_online"])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--T", type=int, default=0, help="override token slots")
    ap.add_argument("--stage3", type=int, default=0, help="override depth")
    ap.add_argument("--out", default="probe_compile_modes.jsonl")
    args = ap.parse_args()
    if args.mode == "compile_lim64":
        torch._dynamo.config.recompile_limit = 64

    dev = torch.device("cuda")
    cfg = yaml.safe_load(open(_abs(args.config)))
    cfg["grad_ckpt"] = True
    if args.T:
        cfg["T"] = args.T
        cfg["obs_dim"] = args.T * 5
    if args.stage3:
        cfg["stage3"] = args.stage3
    torch.manual_seed(0)
    model = build_model(cfg, int(cfg["obs_dim"])).to(dev)
    target = build_model(cfg, int(cfg["obs_dim"])).to(dev)
    target.load_state_dict(model.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
    mc, tc = model, target
    if args.mode in ("compile_both", "compile_lim64"):
        mc = torch.compile(model, dynamic=False)
        tc = torch.compile(target, dynamic=False)
    elif args.mode == "compile_online":
        mc = torch.compile(model, dynamic=False)

    B = int(cfg["batch"])
    T = int(cfg["T"])
    micro = B // 8
    obs = make_batch(B, T, dev, seed=0)
    nobs = make_batch(B, T, dev, seed=7)
    tgt = make_targets(B, int(cfg["K"]), dev)

    fn = lambda: timed_steps(mc, tc, opt, obs, nobs, tgt, micro, 8, "bf16", 0)
    t0 = time.time()
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    warm_s = time.time() - t0
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(args.steps):
        fn()
    e1.record()
    torch.cuda.synchronize()
    ms = e0.elapsed_time(e1) / args.steps
    gf = dynamo_graphs()
    rec = {"mode": args.mode, "ms_per_step": round(ms, 1),
           "samples_per_s": int(B * 1000 / ms),
           "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
           "warmup_s": round(warm_s, 1), "dynamo": gf,
           "compile_count": int(gf.get("unique_graphs", 0)
                                or gf.get("frames_total", 0))}
    print(f"[probe] mode={args.mode:14s} {rec['ms_per_step']:8.1f} ms/step "
          f"{rec['samples_per_s']:6d} samples/s  warmup={rec['warmup_s']}s  "
          f"dynamo={gf}", flush=True)
    with open(_abs(args.out), "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
