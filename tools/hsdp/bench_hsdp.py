#!/usr/bin/env python
"""Multi-host HSDP (FSDP2 2D mesh) LoRA trainer benchmark for Qwen3.8-27B on A100-40GB over TCP.

Mesh: (replicate = #hosts, shard = ranks per host). The frozen bf16 base is sharded within a host only; across hosts
only the LoRA gradients travel (FSDP2 HSDP: reduce-scatter in the shard dim, all-reduce of the grad shards in the
replicate dim). Gradient accumulation: set_requires_gradient_sync(False) on non-final micro-steps, so exactly one
reduce (intra-host RS + cross-host AR) happens per optimizer step.

Per (comp_len) it times every micro-step (fwd, bwd), the optimizer step, and then standalone collectives on the real
process groups:
  ar_hsdp_{fp32,bf16} : every rank all-reduces its 1/S slice of the LoRA grads over the replicate group (the exact
                        cross-host volume of one HSDP optimizer step), all local ranks concurrently
  ar_single_{fp32,bf16}: one 467 MB (fp32) / 233 MB (bf16) flat all-reduce between local-rank-0s only
Run with --device cpu --tiny for gloo CPU correctness/launch tests (no GPU).
Writes one JSON line per comp_len on global rank 0.
"""
import argparse, json, os, sys, time, math, traceback, gc, contextlib
from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.checkpoint

p = argparse.ArgumentParser()
p.add_argument("--model", default="/home/tione/guoshaoyang/models/Qwen3.8-27B")
p.add_argument("--data", default="/home/tione/guoshaoyang/a100_rl/data/prompts_27b_tok.jsonl")
p.add_argument("--r", type=int, default=16)
p.add_argument("--alpha", type=int, default=32)
p.add_argument("--targets", default="all")
p.add_argument("--prompt-len", type=int, default=6144)
p.add_argument("--comp-lens", default="2048,8192")
p.add_argument("--micro", type=int, default=2, help="micro-steps (sequences per rank) per optimizer step")
p.add_argument("--steps", type=int, default=2, help="measured optimizer steps")
p.add_argument("--warmup", type=int, default=1)
p.add_argument("--replicate", type=int, default=0, help="replicate dim; 0 = #hosts (WORLD/LOCAL_WORLD)")
p.add_argument("--sync", default="native", choices=["native", "every"],
               help="native: sync only on final micro-step; every: sync every micro-step (baseline)")
p.add_argument("--lora-mp", default="bf16", help="param_dtype for LoRA compute: bf16 | none (fp32)")
p.add_argument("--reduce-dtype", default="fp32", help="FSDP2 reduce_dtype: fp32 | bf16")
p.add_argument("--attn", default="sdpa")
p.add_argument("--ckpt", type=int, default=1)
p.add_argument("--logits-chunk", type=int, default=1024)
p.add_argument("--layers", type=int, default=0, help="truncate to N layers (debug)")
p.add_argument("--ar-bench", type=int, default=1)
p.add_argument("--ar-iters", type=int, default=5)
p.add_argument("--ar-numel", type=int, default=0, help="standalone AR size (default: this model's LoRA numel; pass "
               "116727808 = full 27B r16 when running a --layers-truncated model)")
p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
p.add_argument("--tiny", type=int, default=0, help="random tiny Qwen3.5 text model (CPU tests); value = hidden size")
p.add_argument("--check-grads", default="", help="CPU test: dump full LoRA grads (after reduce) to this .pt path")
p.add_argument("--fixed-seq", default="", help="CPU test: comma list of seq lens; rank-independent synthetic tokens")
p.add_argument("--ref-world", type=int, default=0, help="CPU test, world==1: emulate W data-parallel ranks sequentially (reference grads)")
p.add_argument("--out", required=True)
p.add_argument("--tag", default="")
p.add_argument("--host-tag", default=os.environ.get("HOST_TAG", ""))
args = p.parse_args()

ALL_T = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
         "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
targets = ALL_T if args.targets == "all" else args.targets.split(",")

CPU = args.device == "cpu"
dist.init_process_group("gloo" if CPU else "nccl", timeout=timedelta(minutes=15))
rank, world = dist.get_rank(), dist.get_world_size()
local = int(os.environ.get("LOCAL_RANK", 0))
local_world = int(os.environ.get("LOCAL_WORLD_SIZE", world))
if not CPU:
    torch.cuda.set_device(local)
dev = torch.device("cpu") if CPU else torch.device("cuda", local)
R = args.replicate or max(1, world // local_world)
S = world // R
assert R * S == world, (R, S, world)


def sync():
    if not CPU:
        torch.cuda.synchronize()


def log(*a):
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def gather_obj(x):
    out = [None] * world
    dist.all_gather_object(out, x)
    return out


T_START = time.time()
log("args", vars(args))
log("torch", torch.__version__, "world", world, "mesh (replicate, shard) =", (R, S),
    "NCCL env", {k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "GLOO_"))})
hosts = gather_obj(os.uname().nodename)
log("hosts per rank", hosts)
# [review fix] the frozen base may only be all-gathered intra-host: every shard group (row r of the (R,S) mesh,
# global ranks r*S .. r*S+S-1) must live on ONE host, else base params cross TCP every layer.
for r_ in range(R):
    hs = set(hosts[r_ * S:(r_ + 1) * S])
    assert len(hs) == 1, f"shard group {r_} spans hosts {hs}: --replicate {R} gives S={S} != ranks per host"

# ---------------- model ----------------
if CPU:
    # [review fix] transformers binds causal_conv1d / fla kernels at import time when the packages are installed (they
    # are, in a100_rl/venv) and they assert x.is_cuda -> the CPU test mode crashed in the first GDN layer. Block them
    # so the torch reference paths are used on CPU.
    for _m in ("causal_conv1d", "fla"):
        sys.modules[_m] = None
from transformers import AutoModelForCausalLM, AutoConfig
if args.tiny:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    H = args.tiny
    tcfg = Qwen3_5TextConfig(vocab_size=512, hidden_size=H, intermediate_size=2 * H, num_hidden_layers=args.layers or 4,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=H // 4,
                             linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=H // 4,
                             linear_value_head_dim=H // 4, full_attention_interval=2, tie_word_embeddings=False,
                             max_position_embeddings=4096)
    tcfg.layer_types = ["linear_attention" if (i + 1) % 2 else "full_attention" for i in range(tcfg.num_hidden_layers)]
    tcfg._attn_implementation = "sdpa" if args.attn == "sdpa" else args.attn
    torch.manual_seed(1234)  # identical base on every rank
    model = AutoModelForCausalLM.from_config(tcfg, dtype=torch.float32 if CPU else torch.bfloat16)
    cfg = tcfg
else:
    cfg = AutoConfig.from_pretrained(args.model)
    tcfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
    if args.layers:
        tcfg.num_hidden_layers = args.layers
        tcfg.layer_types = tcfg.layer_types[: args.layers]
t0 = time.time()
if not args.tiny:
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation=args.attn,
                                                 config=cfg, device_map="cpu")
t_load = time.time() - t0
log(f"loaded {type(model).__name__} in {t_load:.1f}s")

names = set(n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, nn.Linear))
from peft import LoraConfig, get_peft_model
torch.manual_seed(0)
lcfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.0, bias="none",
                  target_modules=[t for t in targets if t in names])
if args.ckpt:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
pm = get_peft_model(model, lcfg)
if args.tiny:
    # peft inits lora_B = 0 -> zero grads for lora_A; give B a deterministic nonzero init so the test checks all grads
    g = torch.Generator().manual_seed(7)
    for n, p_ in pm.named_parameters():
        if "lora_B" in n:
            with torch.no_grad():
                p_.copy_(torch.randn(p_.shape, generator=g) * 0.02)
lora_named = [(n, p_) for n, p_ in pm.named_parameters() if p_.requires_grad]
n_lora = sum(p_.numel() for _, p_ in lora_named)
log(f"LoRA r={args.r} alpha={args.alpha} params={n_lora/1e6:.2f}M dtypes={sorted(set(str(p_.dtype) for _, p_ in lora_named))}")
base = pm.base_model.model


class Policy(nn.Module):
    def __init__(self, pm):
        super().__init__()
        self.pm = pm

    def forward(self, input_ids, comp_start, labels, chunk):
        b = self.pm.base_model.model
        out = b.model(input_ids=input_ids, use_cache=False)
        h = out.last_hidden_state[0, comp_start - 1: -1]
        head = b.lm_head
        res = []
        for i in range(0, h.shape[0], chunk):
            hc, lc = h[i: i + chunk], labels[i: i + chunk]

            def f(hc, lc):
                lg = head(hc).float()
                return lg.gather(1, lc[:, None]).squeeze(1) - torch.logsumexp(lg, -1)
            res.append(torch.utils.checkpoint.checkpoint(f, hc, lc, use_reentrant=False)
                       if torch.is_grad_enabled() else f(hc, lc))
        return torch.cat(res)


pol = Policy(pm)
pol.train()

from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
t1 = time.time()
mesh = init_device_mesh("cpu" if CPU else "cuda", (R, S), mesh_dim_names=("replicate", "shard"))
rd = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.reduce_dtype]
pd = torch.bfloat16 if (args.lora_mp == "bf16" and not CPU) else None
mp = MixedPrecisionPolicy(param_dtype=pd, reduce_dtype=rd)
fsdp_mesh = mesh if R > 1 else mesh["shard"]  # R == 1 -> plain FSDP2 over the host
for layer in base.model.layers:
    fully_shard(layer, mesh=fsdp_mesh, mp_policy=mp)
fully_shard(base.model.embed_tokens, mesh=fsdp_mesh, mp_policy=mp)
fully_shard(base.lm_head, mesh=fsdp_mesh, mp_policy=mp, reshard_after_forward=False)
fully_shard(pol, mesh=fsdp_mesh, mp_policy=mp)
sync()
t_shard = time.time() - t1
mem_static = 0.0 if CPU else torch.cuda.memory_allocated(dev) / 2**30
log(f"shard {t_shard:.1f}s mem_alloc={mem_static:.2f}GiB fsdp_mesh={fsdp_mesh}")

lora_trainable = [p_ for p_ in pol.parameters() if p_.requires_grad]
# [review fix] lr=0 under --check-grads: the timed steps must not move the weights, otherwise the HSDP run and the
# world=1 reference (whose timed steps only see rank-0 data) dump grads at different LoRA weights.
opt = torch.optim.AdamW(lora_trainable, lr=0.0 if args.check_grads else 1e-6, betas=(0.9, 0.99), eps=1e-8,
                        weight_decay=0.0, foreach=not CPU)

# [review fix] count the collectives FSDP2 actually issues during training (it calls dist.reduce_scatter_single and
# dist.all_reduce through the module attribute, so wrapping them here intercepts its calls). Evidence for
# "exactly one intra-host RS + one cross-host AR per FSDP unit per optimizer step".
REP_NAME = mesh.get_group("replicate").group_name if R > 1 else None
COLL = {"on": False, "rs": 0, "rs_bytes": 0, "ar_rep": 0, "ar_rep_bytes": 0, "ar_other": 0}
_orig_ar, _orig_rs = dist.all_reduce, getattr(dist, "reduce_scatter_single", None)


def _ar_wrap(tensor, *a, **k):
    if COLL["on"]:
        g = k.get("group", a[1] if len(a) > 1 else None)
        if g is not None and R > 1 and getattr(g, "group_name", None) == REP_NAME:
            COLL["ar_rep"] += 1; COLL["ar_rep_bytes"] += tensor.numel() * tensor.element_size()
        else:
            COLL["ar_other"] += 1
    return _orig_ar(tensor, *a, **k)


def _rs_wrap(*a, **k):
    if COLL["on"]:
        t = k.get("input", a[1] if len(a) > 1 else None)
        COLL["rs"] += 1; COLL["rs_bytes"] += 0 if t is None else t.numel() * t.element_size()
    return _orig_rs(*a, **k)


dist.all_reduce = _ar_wrap
if _orig_rs is not None:
    dist.reduce_scatter_single = _rs_wrap


def coll_snapshot():
    return {k: v for k, v in COLL.items() if k != "on"}


def coll_reset():
    for k in COLL:
        if k != "on":
            COLL[k] = 0

# ---------------- data ----------------
rows = []
if not args.fixed_seq:
    with open(args.data) as f:
        for line in f:
            r_ = json.loads(line)
            if r_["n_tok"] >= args.prompt_len:
                rows.append(r_)
            if len(rows) >= 256:
                break
    filler = []
    with open(args.data) as f:
        for line in f:
            filler.extend(json.loads(line)["input_ids"])
            if len(filler) > 200000:
                break


def make_seq(micro_idx, C, rank=rank):
    """Real prompt (distinct per global rank and micro-step) + real-text completion of length C."""
    if args.fixed_seq:
        L = [int(x) for x in args.fixed_seq.split(",")]
        n = L[(rank * args.micro + micro_idx) % len(L)]
        g = torch.Generator().manual_seed(1000 + rank * args.micro + micro_idx)
        ids = torch.randint(0, tcfg.vocab_size, (n,), generator=g).tolist()
        P = n // 2
        return ids[:P], ids[P:]
    pr = rows[(rank * args.micro + micro_idx) % len(rows)]["input_ids"]
    prompt = pr[: args.prompt_len - 8] + pr[-8:]
    st = ((rank * args.micro + micro_idx) * 9973) % max(1, len(filler) - C)
    return prompt, filler[st: st + C]


def micro_step(prompt, comp, scale):
    ids = torch.tensor([prompt + comp], device=dev)
    labels = torch.tensor(comp, device=dev)
    sync(); ta = time.time()
    logp = pol(ids, len(prompt), labels, args.logits_chunk)
    old = logp.detach()
    ratio = torch.exp((logp - old).mean())
    loss = -torch.minimum(ratio, ratio.clamp(1 - 3e-4, 1 + 4e-4)) * scale
    if args.check_grads:  # deterministic, ratio-independent loss for exact grad comparison
        loss = -(logp.mean()) * scale
    sync(); tb = time.time()
    loss.backward()
    sync(); tc = time.time()
    return tb - ta, tc - tb, float(loss)


def opt_step():
    M = args.micro
    fw, bw, colls = [], [], []
    sync(); t0_ = time.time()
    for i in range(M):
        final = i == M - 1
        pol.set_requires_gradient_sync(final or args.sync == "every")
        pr, cp = SEQS[i]
        coll_reset(); COLL["on"] = True
        a, b, _ = micro_step(pr, cp, 1.0 / M)  # FSDP2 averages over all world ranks (shard x replicate)
        COLL["on"] = False; colls.append(coll_snapshot())
        fw.append(a); bw.append(b)
    sync(); t_bw_end = time.time()
    gn = torch.nn.utils.clip_grad_norm_(lora_trainable, 1e9)
    opt.step()
    opt.zero_grad(set_to_none=True)  # [review fix] was missing: FSDP2 then ADDS the next step's reduced grads
    sync(); t_opt = time.time() - t_bw_end
    gnv = float(gn.full_tensor() if hasattr(gn, "full_tensor") else gn)
    return dict(fwd_s=fw, bwd_s=bw, opt_s=t_opt, step_s=time.time() - t0_, grad_norm=gnv, colls=colls)


# ---------------- standalone collectives on the real groups ----------------
rep_group = mesh.get_group("replicate") if R > 1 else None


def time_coll(fn, iters):
    fn(); sync(); dist.barrier()
    ts = []
    for _ in range(iters):
        sync(); dist.barrier(); t = time.time()
        fn(); sync()
        ts.append(time.time() - t)
    return sorted(ts)[len(ts) // 2]


def ar_bench():
    if rep_group is None:
        return {}
    out = {}
    N = args.ar_numel or n_lora
    for dt_name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        # (a) HSDP pattern: every rank its 1/S slice, all concurrently
        x = torch.ones(math.ceil(N / S), dtype=dt, device=dev)
        t = time_coll(lambda: dist.all_reduce(x, group=rep_group), args.ar_iters)
        t = max(gather_obj(t))  # [review fix] slowest rank, not rank 0's view of S concurrent all-reduces
        tot = N * x.element_size()
        out[f"ar_hsdp_{dt_name}"] = {"s": t, "bytes_per_rank": x.numel() * x.element_size(), "bytes_total_per_host": tot,
                                     "host_algbw_GBps": tot / t / 1e9}
        del x
        # (b) one full flat LoRA-grad all-reduce between local-rank-0s only
        if local == 0:
            y = torch.ones(N, dtype=dt, device=dev)
            t = time_coll_sub(lambda: dist.all_reduce(y, group=rep_group), args.ar_iters)
            out[f"ar_single_{dt_name}"] = {"s": t, "bytes": tot, "algbw_GBps": tot / t / 1e9,
                                           "busbw_GBps": 2 * (R - 1) / R * tot / t / 1e9}
            del y
        dist.barrier()
    return out


def time_coll_sub(fn, iters):
    fn(); sync()
    ts = []
    for _ in range(iters):
        sync(); dist.barrier(group=rep_group); t = time.time()
        fn(); sync()
        ts.append(time.time() - t)
    return sorted(ts)[len(ts) // 2]


results = []
comp_lens = [int(x) for x in args.comp_lens.split(",")] if not args.fixed_seq else [0]
for C in comp_lens:
    SEQS = [make_seq(i, C) for i in range(args.micro)]
    if args.ref_world and world == 1:  # sequential emulation of W ranks: all (rank, micro) seqs, scaled 1/(M*W)
        REF = [make_seq(i, C, rank=rr) for rr in range(args.ref_world) for i in range(args.micro)]
    seq_tok = [len(a) + len(b) for a, b in SEQS]
    rec = dict(arm="wave2/hsdp/trainer", tag=args.tag, world=world, replicate=R, shard=S, hosts=sorted(set(hosts)),
               micro=args.micro, sync=args.sync, r=args.r, alpha=args.alpha, n_lora_params=n_lora,
               lora_mp=args.lora_mp, reduce_dtype=args.reduce_dtype, attn=args.attn, ckpt=args.ckpt,
               prompt_len=args.prompt_len, comp_len=C, seq_tokens_rank0=seq_tok, layers=tcfg.num_hidden_layers,
               load_s=t_load, shard_s=t_shard, mem_static_gib=mem_static,
               nccl_env={k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "GLOO_"))},
               time=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    try:
        steps = []
        for it in range(args.warmup + args.steps):
            if not CPU:
                torch.cuda.reset_peak_memory_stats(dev)
            m = opt_step()
            m["peak_alloc_gib"] = 0.0 if CPU else torch.cuda.max_memory_allocated(dev) / 2**30
            m["it"] = it
            steps.append(m)
            log(f"C={C} it={it} step={m['step_s']:.2f}s fwd={[round(x,2) for x in m['fwd_s']]} "
                f"bwd={[round(x,2) for x in m['bwd_s']]} opt={m['opt_s']:.3f} gn={m['grad_norm']:.4g} "
                f"peak={m['peak_alloc_gib']:.1f}GiB")
        meas = steps[args.warmup:] or steps
        st_all = gather_obj(sum(s["step_s"] for s in meas) / len(meas))
        rec["steps"] = steps
        rec["step_s"] = max(st_all)
        rec["step_s_per_rank"] = [round(x, 3) for x in st_all]
        # exposed sync cost: final micro-step bwd minus mean non-final bwd (same-length sequences)
        if args.micro > 1 and args.sync == "native":
            ex = [s["bwd_s"][-1] - sum(s["bwd_s"][:-1]) / (args.micro - 1) for s in meas]
            rec["exposed_sync_s_rank0"] = sum(ex) / len(ex)
        # collectives FSDP2 issued, per micro-step (rank 0); expected native: zeros except the final micro-step
        rec["fsdp_colls_rank0"] = [s["colls"] for s in meas]
        rec["peak_alloc_gib_per_rank"] = [round(x, 2) for x in gather_obj(max(s["peak_alloc_gib"] for s in steps))]
        tok_rank = gather_obj(sum(seq_tok))
        rec["tokens_per_step_total"] = sum(tok_rank)
        rec["train_tok_s_total"] = sum(tok_rank) / rec["step_s"]
        rec["train_tok_s_per_gpu"] = rec["train_tok_s_total"] / world
        if args.ar_bench:
            rec["collectives"] = ar_bench()
            log("collectives", json.dumps(rec["collectives"]))
        rec["status"] = "ok"
    except Exception as e:
        # [review fix] an OOM/error on one rank leaves the others blocked in NCCL collectives (15 min timeout) and the
        # old per-rank "recover and continue" could not work; record per rank and abort the whole job instead.
        rec["status"] = "OOM" if isinstance(e, torch.OutOfMemoryError) else "ERROR"
        rec["error"] = traceback.format_exc()[-2500:]
        rec["failed_rank"] = rank
        print(f"[rank {rank}] C={C} {rec['status']}: {rec['error'][-600:]}", flush=True)
        with open(args.out if rank == 0 else f"{args.out}.rank{rank}.err", "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(3 if rec["status"] == "OOM" else 4)
    if args.check_grads:
        # full (unsharded) reduced grads of the LAST optimizer step are in .grad before zero_grad; recompute here:
        pass
    results.append(rec)
    if rank == 0:
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    gc.collect()
    if not CPU:
        torch.cuda.empty_cache()

if args.check_grads:
    # One more accumulation step without optimizer, then dump full grads (DTensor.full_tensor) for comparison.
    opt.zero_grad(set_to_none=True)
    M = args.micro
    if args.ref_world and world == 1:
        for pr, cp in REF:
            micro_step(pr, cp, 1.0 / (M * args.ref_world))
    else:
        for i in range(M):
            pol.set_requires_gradient_sync(i == M - 1 or args.sync == "every")
            pr, cp = SEQS[i]
            micro_step(pr, cp, 1.0 / M)
    full = {}
    for n, p_ in pol.named_parameters():
        if p_.requires_grad and p_.grad is not None:
            g_ = p_.grad
            full[n] = (g_.full_tensor() if hasattr(g_, "full_tensor") else g_).detach().float().cpu()
    if rank == 0:
        torch.save(full, args.check_grads)
        log("dumped", len(full), "grads to", args.check_grads)

log("DONE total_s", round(time.time() - T_START, 1))
dist.barrier()
dist.destroy_process_group()
