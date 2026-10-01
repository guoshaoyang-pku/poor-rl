"""tf_xl throughput ablation: where do the 14.7 s/grad-step go?

Reproduces the learner's exact step path (bf16 autocast, micro-batch
accumulation, double-DQN target forwards, quantile-huber loss) and sweeps
four switches that are the suspected causes of the ~5.6 TFLOP/s plateau:

  grad_ckpt   full activation recompute of every attention block + cross KV
  attn_impl   eager  = nn.MultiheadAttention (materialises [B,H,T,T] scores)
              sdpa   = F.scaled_dot_product_attention (flash-mem-efficient)
  amp         bf16 autocast vs fp32
  micro       micro-batch per forward (batch is fixed at 32768)

One GPU, one process. Reports ms/step, grad steps/s, learner samples/s and
peak allocated memory per combo.

  CUDA_VISIBLE_DEVICES=4 python bench_tf_throughput.py \
      --config configs/w3c_tf_xl.yaml [--only sdpa] [--steps 3]
"""
import argparse
import json
import os
import time

import torch

# `model` -> paths.setup_engine_path() chdir()s into the engine root, so any
# relative path the caller passed must be resolved against the launch cwd.
_LAUNCH_CWD = os.getcwd()

import yaml  # noqa: E402

from model import build_model, param_count  # noqa: E402
from learner import quantile_huber_loss  # noqa: E402
from model_v2 import SetTransformerQ, to_sdpa_state_dict  # noqa: E402


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(_LAUNCH_CWD, p)


def make_batch(B, T, device, seed=0):
    """Synthetic token obs with the real encoding's invariants: rows 0/1 are
    current/next fruit (always valid), the rest are board fruits or type=-1."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(B, T, 5, generator=g) * 0.3
    tid = torch.randint(-1, 11, (B, T), generator=g)
    tid[:, 0] = torch.randint(0, 11, (B,), generator=g)
    tid[:, 1] = torch.randint(0, 11, (B,), generator=g)
    x[..., 0] = tid
    x[..., 2] = torch.rand(B, T, generator=g)          # y
    x[..., 3:5] *= 0.5                                 # velocities
    return x.reshape(B, -1).to(device)


def make_targets(B, K, device, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    act = torch.randint(0, K, (B,), generator=g)
    rew = torch.rand(B, generator=g) * 10
    gam = torch.ones(B) * 0.99
    done = (torch.rand(B, generator=g) < 0.02).float()
    return [t.to(device) for t in (act, rew, gam, done)]


def timed_steps(model, target, opt, obs, nobs, tgt, micro, accum, amp, n_sub):
    """Run one optimiser step; `tgt` = (act, rew, gam, done)."""
    dev = obs.device
    act, rew, gam, done = tgt
    n_quant = 1
    use_amp = amp == "bf16" and dev.type == "cuda"
    opt.zero_grad(set_to_none=True)
    for i in range(0, obs.shape[0], micro):
        sl = slice(i, i + micro)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            z = model(obs[sl])
            pred = z.gather(1, act[sl].view(-1, 1, 1)
                            .expand(-1, 1, n_quant)).squeeze(1)
            with torch.no_grad():
                a_star = model(nobs[sl]).mean(-1).argmax(1)      # double DQN
                nxt = target(nobs[sl]).gather(
                    1, a_star.view(-1, 1, 1).expand(-1, 1, n_quant)).squeeze(1)
                t = rew[sl].unsqueeze(1) + (gam[sl] * (1 - done[sl])
                                            ).unsqueeze(1) * nxt
            w = torch.ones(pred.shape[0], device=dev)
            loss, _ = quantile_huber_loss(pred, t, w)
        (loss / accum).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    opt.step()


def run_combo(combo, base_cfg, args, dev):
    cfg = dict(base_cfg)
    cfg["grad_ckpt"] = combo["ckpt"]
    cfg["attn_impl"] = combo["attn"]
    torch.manual_seed(0)
    model = build_model(cfg, int(cfg["obs_dim"])).to(dev)
    target = build_model(cfg, int(cfg["obs_dim"])).to(dev)
    target.load_state_dict(model.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
    mc, tc = model, target
    if combo["compile"]:
        mc = torch.compile(model, dynamic=False)
        tc = torch.compile(target, dynamic=False)

    B = int(cfg["batch"])
    micro = combo["micro"]
    accum = max(1, B // micro)
    obs = make_batch(B, int(cfg["T"]), dev, seed=0)
    nobs = make_batch(B, int(cfg["T"]), dev, seed=7)
    tgt = make_targets(B, int(cfg["K"]), dev)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.warmup):
        timed_steps(mc, tc, opt, obs, nobs, tgt, micro, accum, combo["amp"], 0)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(args.steps):
        timed_steps(mc, tc, opt, obs, nobs, tgt, micro, accum, combo["amp"], 0)
    ev1.record()
    torch.cuda.synchronize()
    ms = ev0.elapsed_time(ev1) / args.steps
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    return {
        "name": combo["name"],
        "ckpt": combo["ckpt"], "attn": combo["attn"], "amp": combo["amp"],
        "micro": micro, "accum": accum, "compile": combo["compile"],
        "params_m": round(param_count(model) / 1e6, 2),
        "ms_per_step": round(ms, 1),
        "grad_per_s": round(1000.0 / ms, 4),
        "samples_per_s": int(B * 1000.0 / ms),
        "peak_gb": round(peak, 1),
    }


COMBOS = [
    dict(name="ck1-eager-bf16-m4096-cp1", ckpt=True, attn="eager",
         amp="bf16", micro=4096, compile=True),      # current production
    dict(name="ck1-sdpa-bf16-m4096-cp1", ckpt=True, attn="sdpa",
         amp="bf16", micro=4096, compile=True),
    dict(name="ck0-sdpa-bf16-m4096-cp1", ckpt=False, attn="sdpa",
         amp="bf16", micro=4096, compile=True),
    dict(name="ck0-eager-bf16-m4096-cp1", ckpt=False, attn="eager",
         amp="bf16", micro=4096, compile=True),
    dict(name="ck1-eager-bf16-m4096-cp0", ckpt=True, attn="eager",
         amp="bf16", micro=4096, compile=False),
    dict(name="ck1-sdpa-bf16-m8192-cp1", ckpt=True, attn="sdpa",
         amp="bf16", micro=8192, compile=True),
    dict(name="ck0-sdpa-bf16-m8192-cp1", ckpt=False, attn="sdpa",
         amp="bf16", micro=8192, compile=True),
    dict(name="ck1-sdpa-fp32-m4096-cp1", ckpt=True, attn="sdpa",
         amp="fp32", micro=4096, compile=True),
    dict(name="ck0-sdpa-bf16-m8192-cp0", ckpt=False, attn="sdpa",
         amp="bf16", micro=8192, compile=False),
    # micro 4096 has no room for the un-recomputed graph on a 40 GB A100
    # (peak would be >39.5 GB), so pay for it with a smaller micro-batch:
    # the optimiser batch stays 32768, only the accumulation boundary moves.
    dict(name="ck0-sdpa-bf16-m2048-cp1", ckpt=False, attn="sdpa",
         amp="bf16", micro=2048, compile=True),
    dict(name="ck0-sdpa-bf16-m1024-cp1", ckpt=False, attn="sdpa",
         amp="bf16", micro=1024, compile=True),
    dict(name="ck0-eager-bf16-m2048-cp1", ckpt=False, attn="eager",
         amp="bf16", micro=2048, compile=True),
    dict(name="ck1-sdpa-bf16-m2048-cp1", ckpt=True, attn="sdpa",
         amp="bf16", micro=2048, compile=True),
    dict(name="ck1-sdpa-bf16-m1024-cp1", ckpt=True, attn="sdpa",
         amp="bf16", micro=1024, compile=True),
    dict(name="ck0-sdpa-bf16-m2048-cp0", ckpt=False, attn="sdpa",
         amp="bf16", micro=2048, compile=False),
]


def check_equivalence(base_cfg, dev):
    """Eager vs SDPA must agree on the same weights, else the ablation is
    measuring two different models."""
    cfg = dict(base_cfg)
    cfg["grad_ckpt"] = False
    cfg["attn_impl"] = "eager"
    a = build_model(cfg, int(cfg["obs_dim"])).to(dev).float()
    cfg["attn_impl"] = "sdpa"
    b = SetTransformerQ(
        n_actions=int(cfg["K"]), T=int(cfg["T"]), d_tok=int(cfg["d_tok"]),
        d_lat=int(cfg["d_lat"]), n_lat=int(cfg["n_lat"]),
        stage1=int(cfg["stage1"]), stage3=int(cfg["stage3"]),
        heads=int(cfg["tf_heads"]), head_dim=int(cfg["head_dim"]),
        grad_ckpt=False, attn_impl="sdpa").to(dev).float()
    b.load_state_dict(to_sdpa_state_dict(a.state_dict()))
    x = make_batch(8, int(cfg["T"]), dev)
    with torch.no_grad():
        qa, qb = a.q_values(x), b.q_values(x)
    d = (qa - qb).abs().max().item()
    rel = d / max(1e-6, qa.abs().max().item())
    print(f"[check] eager vs sdpa: max|dq|={d:.3e} (rel {rel:.2e}) "
          f"q_scale={qa.abs().max().item():.3f}", flush=True)
    assert rel < 1e-3, "SDPA block does not match eager block"
    # padding must be ignored the same way
    x2 = make_batch(8, int(cfg["T"]), dev)
    x2.view(8, -1, 5)[:, 40:, 0] = -1
    with torch.no_grad():
        qa, qb = a.q_values(x2), b.q_values(x2)
    rel2 = ((qa - qb).abs().max() / max(1e-6, qa.abs().max().item())).item()
    print(f"[check] padded rows: rel={rel2:.2e}", flush=True)
    assert rel2 < 1e-3, "pad_mask semantics differ"
    print("[check] OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/w3c_tf_xl.yaml")
    ap.add_argument("--only", default="", help="substring filter on combo name")
    ap.add_argument("--combos", default="",
                    help="comma-separated exact combo names to run")
    ap.add_argument("--batch", type=int, default=0, help="0 = cfg batch")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="bench_tf_throughput.json")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_cfg = yaml.safe_load(open(_abs(args.config)))
    if args.batch:
        base_cfg["batch"] = args.batch
    if args.check:
        check_equivalence(base_cfg, dev)   # runs on CPU too (tiny batch)
        return
    assert torch.cuda.is_available(), "throughput bench needs a GPU"

    print(f"[bench] gpu={torch.cuda.get_device_name(0)} "
          f"torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    print(f"[bench] cfg={base_cfg['name']} batch={base_cfg['batch']} "
          f"T={base_cfg['T']} K={base_cfg['K']}", flush=True)

    rows, t0 = [], time.time()
    for combo in COMBOS:
        if args.only and args.only not in combo["name"]:
            continue
        if args.combos and combo["name"] not in args.combos.split(","):
            continue
        print(f"[bench] {combo['name']} ...", flush=True)
        try:
            row = run_combo(combo, base_cfg, args, dev)
            print(f"[bench]   {row['ms_per_step']} ms/step  "
                  f"{row['grad_per_s']} grad/s  "
                  f"{row['samples_per_s']} samples/s  "
                  f"{row['peak_gb']} GB peak", flush=True)
        except RuntimeError as e:
            msg = str(e).splitlines()[0]
            oom = "out of memory" in msg.lower()
            print(f"[bench]   FAILED{' (OOM)' if oom else ''}: {msg[:160]}",
                  flush=True)
            row = {"name": combo["name"], "ckpt": combo["ckpt"],
                   "attn": combo["attn"], "amp": combo["amp"],
                   "micro": combo["micro"], "compile": combo["compile"],
                   "error": "oom" if oom else msg[:160]}
        rows.append(row)
        torch.cuda.empty_cache()

    out = {"config": args.config, "batch": base_cfg["batch"],
           "steps": args.steps, "warmup": args.warmup,
           "gpu": torch.cuda.get_device_name(0), "rows": rows}
    with open(_abs(args.out), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[bench] wrote {_abs(args.out)} "
          f"({time.time() - t0:.0f}s total)", flush=True)

    ok = [r for r in rows if "ms_per_step" in r]
    if ok:
        best = max(ok, key=lambda r: r["samples_per_s"])
        print(f"[bench] best: {best['name']} = {best['samples_per_s']} samples/s "
              f"({best['ms_per_step']} ms/step, {best['peak_gb']} GB)",
              flush=True)


if __name__ == "__main__":
    main()
