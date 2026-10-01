"""Empirical cost split of the tf_xl learner step on one A100.

The throughput sweep only says which switches help. This says *where the
12.6 s/step actually goes*, by leave-one-out timing on the production graph
(ck1 + eager + bf16 + torch.compile + micro 4096):

  fwd_only      one no_grad forward (the actor's inference shape)
  prod16        full learner step, stage3 depth 16 (production)
  s3_8 / s3_4   stage3 truncated -> isolates latent self-attn cost / layer
  T64           T=160 -> 64 token slots (3.3x less padding; see probe)
  no_ckpt       recompute off (only fits on a >=80 GB card)

  python profile_tf_stage.py --config configs/w3c_tf_xl.yaml
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

VARIANTS = [
    dict(name="prod16", s3=None, T=None, ckpt=True, compile=True),
    dict(name="s3_8", s3=8, T=None, ckpt=True, compile=True),
    dict(name="s3_4", s3=4, T=None, ckpt=True, compile=True),
    dict(name="T64", s3=None, T=64, ckpt=True, compile=True),
    dict(name="no_compile", s3=None, T=None, ckpt=True, compile=False),
    dict(name="no_ckpt", s3=None, T=None, ckpt=False, compile=True),
]


def build(cfg0, v, B, dev):
    cfg = dict(cfg0)
    if v["s3"] is not None:
        cfg["stage3"] = v["s3"]
    if v["T"] is not None:
        cfg["T"] = v["T"]
        cfg["obs_dim"] = v["T"] * 5
    cfg["grad_ckpt"] = v["ckpt"]
    torch.manual_seed(0)
    m = build_model(cfg, int(cfg["obs_dim"])).to(dev)
    t = build_model(cfg, int(cfg["obs_dim"])).to(dev)
    t.load_state_dict(m.state_dict())
    for p in t.parameters():
        p.requires_grad_(False)
    mc, tc = m, t
    if v["compile"]:
        mc = torch.compile(m, dynamic=False)
        tc = torch.compile(t, dynamic=False)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, betas=(0.9, 0.95))
    obs = make_batch(B, int(cfg["T"]), dev, seed=0)
    nobs = make_batch(B, int(cfg["T"]), dev, seed=7)
    tgt = make_targets(B, int(cfg["K"]), dev)
    return mc, tc, opt, obs, nobs, tgt, cfg


def timeit(fn, warmup, steps):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(steps):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/w3c_tf_xl.yaml")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="profile_tf_stage.json")
    args = ap.parse_args()
    dev = torch.device("cuda")
    cfg0 = yaml.safe_load(open(_abs(args.config)))
    B = int(cfg0["batch"])
    print(f"[prof] gpu={torch.cuda.get_device_name(0)} batch={B} "
          f"T={cfg0['T']} stage3={cfg0['stage3']}", flush=True)

    rows = []
    for v in VARIANTS:
        try:
            mc, tc, opt, obs, nobs, tgt, cfg = build(cfg0, v, B, dev)
            micro = int(cfg0["batch"]) // 8
            if micro > obs.shape[0]:
                micro = obs.shape[0]
            fn = lambda: timed_steps(mc, tc, opt, obs, nobs, tgt, micro, 8,
                                     "bf16", 0)
            ms = timeit(fn, args.warmup, args.steps)
            # pure inference rate: what one actor forward costs (no_grad)
            with torch.no_grad():
                f_ms = timeit(lambda: mc(obs[:4096]), args.warmup, args.steps)
            row = dict(name=v["name"], stage3=cfg["stage3"], T=cfg["T"],
                       ckpt=v["ckpt"], compile=v["compile"],
                       ms_per_step=round(ms, 1),
                       samples_per_s=int(B * 1000 / ms),
                       fwd_ms_4096=round(f_ms, 2),
                       fwd_samples_per_s=int(4096 * 1000 / f_ms))
            print(f"[prof] {row['name']:11s} s3={row['stage3']:2d} T={row['T']:3d}"
                  f" ck={int(row['ckpt'])} cp={int(row['compile'])}"
                  f"  {row['ms_per_step']:8.1f} ms/step "
                  f"{row['samples_per_s']:6d} samples/s   "
                  f"fwd4096 {row['fwd_ms_4096']:7.2f} ms "
                  f"({row['fwd_samples_per_s']} samp/s)", flush=True)
        except RuntimeError as e:
            msg = str(e).splitlines()[0]
            row = dict(name=v["name"], error=msg[:160])
            print(f"[prof] {v['name']:11s} FAILED: {msg[:120]}", flush=True)
        rows.append(row)
        del v
        torch.cuda.empty_cache()

    base = next((r for r in rows if r["name"] == "prod16"
                 and "ms_per_step" in r), None)
    if base:
        print(f"\n[prof] production = {base['ms_per_step']} ms/step "
              f"({base['samples_per_s']} samples/s)", flush=True)
        for r in rows:
            if r.get("ms_per_step") and r["name"] != "prod16":
                print(f"[prof]   {r['name']}: "
                      f"{base['ms_per_step'] / r['ms_per_step']:.2f}x vs prod",
                      flush=True)
    with open(_abs(args.out), "w") as f:
        json.dump({"config": args.config, "batch": B, "rows": rows}, f, indent=1)
    print(f"[prof] wrote {_abs(args.out)}", flush=True)


if __name__ == "__main__":
    main()
