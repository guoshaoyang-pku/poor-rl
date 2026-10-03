#!/usr/bin/env python
"""Wave-2 end-to-end async GSPO smoke for Qwen3.8-27B LoRA on A100-40GB (one launcher, three tracks).

  rollout : N vLLM backends (any TP / precision mix) behind rlforge.rollout.router (group affinity,
            adapter-only LoRA sync pushed to a lora_agent per host).        [rollout track]
  trainer : FSDP2 (1-D) or HSDP (--replicate R: shard inside a host, replicate across hosts, only LoRA
            grads cross hosts) over the frozen bf16 base, LoRA fp32 master.   [hsdp track]
  forward : chunked prefix sharing (prompt once per group, G branches in chunks of --chunk rows, one
            prompt backward), rlforge.prefix_share_sm80.                       [prefix track]
  sync    : after every optimizer step the LoRA tensors are gathered (DTensor.full_tensor), written as a
            PEFT adapter with vLLM's language_model naming and loaded through the router as policy-v{k}.

Rank 0 runs the rollout producer in a thread (HTTP only, no CUDA): batch b is submitted once the trainer has
finished step b-1-max_stale, with the newest loaded adapter, so batch b+1 is generated while step b trains.
Step 1 runs with lr 0 (weights == rollout weights), so its log-ratio vs the vLLM sampling logprobs is the pure
train/inference mismatch per rollout precision.

Reward: placeholder. MCQ letter (+1 correct / 0 wrong / -0.5 unparsed) + 0.05 * (1 - len/max_tokens), so groups
are never all-equal and the update is non-zero. Reward quality is not the point of this smoke.

Run (one host, 4 trainer GPUs):
  torchrun --nproc_per_node 4 tools/e2e/e2e_gspo.py --router http://127.0.0.1:8410 --out RUN_DIR \
      --backend-precision 0=bf16_tp2,1=w4a16_tp1
CPU test: --device cpu with a tiny model and tools/e2e/fake_vllm.py backends (tools/e2e/cpu_e2e_test.sh).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import json
import math
import os
import queue
import random
import re
import sys
import threading
import time
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

ALL_T = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
         "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]

p = argparse.ArgumentParser()
p.add_argument("--model", default="/home/tione/guoshaoyang/models/Qwen3.8-27B")
p.add_argument("--prompts", default="/home/tione/guoshaoyang/a100_rl/data/prompts_27b_tok.jsonl")
p.add_argument("--answers", default="/home/tione/guoshaoyang/a100_rl/data/src/train_think.jsonl")
p.add_argument("--router", required=True, help="router base URL (rank 0 only talks to it)")
p.add_argument("--out", required=True)
p.add_argument("--steps", type=int, default=5)
p.add_argument("--groups", type=int, default=4, help="groups per step (multiple of the DP world)")
p.add_argument("--G", type=int, default=32)
p.add_argument("--max-tokens", type=int, default=2048)
p.add_argument("--max-prompt", type=int, default=7000)
p.add_argument("--max-stale", type=int, default=1)
p.add_argument("--chunk", type=int, default=4, help="branches per prefix-share forward/backward chunk")
p.add_argument("--lm-chunk", type=int, default=1024)
p.add_argument("--offload", type=int, default=0, help="offload checkpointed layer inputs to pinned CPU")
p.add_argument("--r", type=int, default=16)
p.add_argument("--alpha", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--lr0-steps", type=int, default=1, help="first N steps run with lr 0 (numerics probe)")
p.add_argument("--eps-lo", type=float, default=3e-4)
p.add_argument("--eps-hi", type=float, default=4e-4)
p.add_argument("--clip-grad", type=float, default=1.0)
p.add_argument("--temperature", type=float, default=1.0)
p.add_argument("--replicate", type=int, default=1, help="HSDP replicate degree (hosts); 1 = plain FSDP2")
p.add_argument("--attn", default="sdpa")
p.add_argument("--layers", type=int, default=0, help="debug: truncate the model")
p.add_argument("--device", default="cuda")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--backend-precision", default="", help="router backend idx -> label, e.g. 0=bf16_tp2,1=w4a16_tp1")
p.add_argument("--rollout-workers", type=int, default=64, help="concurrent group requests from rank 0")
p.add_argument("--request-timeout", type=float, default=3600)
p.add_argument("--verify-prefix", type=int, default=2, help="step 1: per-seq logp of N branches/rank vs prefix")
p.add_argument("--keep-adapters", type=int, default=0)
p.add_argument("--adapter-name", default="policy")
p.add_argument("--max-step-s", type=float, default=3600, help="abort if a step exceeds this (hang guard)")
args = p.parse_args()

CPU = args.device == "cpu"
if CPU:  # transformers would pick fla / causal_conv1d kernels that need CUDA tensors
    for _m in ("fla", "causal_conv1d", "flash_attn"):
        sys.modules[_m] = None

dist.init_process_group("gloo" if CPU else "nccl", timeout=timedelta(minutes=20))
rank, world = dist.get_rank(), dist.get_world_size()
local = int(os.environ.get("LOCAL_RANK", 0))
if CPU:
    dev = torch.device("cpu")
else:
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
os.makedirs(args.out, exist_ok=True)
T0 = time.time()


def log(*a, all_ranks=False):
    if rank == 0 or all_ranks:
        print(f"[{time.strftime('%H:%M:%S')} r{rank} +{time.time() - T0:.0f}s]", *a, flush=True)


def sync():
    if not CPU:
        torch.cuda.synchronize()


def write_jsonl(name, rec):
    if rank == 0:
        with open(os.path.join(args.out, name), "a") as f:
            f.write(json.dumps(rec) + "\n")


def q(xs, f):
    if not xs:
        return None
    xs = sorted(xs)
    return float(xs[min(len(xs) - 1, int(f * len(xs)))])


assert args.groups % world == 0, "groups per step must be a multiple of the DP world"
assert world % args.replicate == 0

# ------------------------------------------------------------------------------------------------ model
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from torch.distributed.device_mesh import init_device_mesh  # noqa: E402
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard  # noqa: E402

import rlforge.prefix_share_sm80 as ps  # noqa: E402

cfg = AutoConfig.from_pretrained(args.model)
tcfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
if args.layers:
    tcfg.num_hidden_layers = args.layers
    tcfg.layer_types = tcfg.layer_types[: args.layers]
t = time.time()
model = AutoModelForCausalLM.from_pretrained(args.model, config=cfg, dtype=torch.float32 if CPU else torch.bfloat16,
                                             attn_implementation=args.attn, device_map="cpu")
t_load = time.time() - t
names = {n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, nn.Linear)}
targets = [x for x in ALL_T if x in names]
torch.manual_seed(args.seed)
pm = get_peft_model(model, LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.0, bias="none",
                                      target_modules=targets))
lm = ps.install(pm)                     # Qwen3_5ForCausalLM (inside peft), forward dispatch patched
lm._ps_force_ckpt = True                # prefix_share checkpoints every decoder layer itself
lm.train()
bb = ps._backbone(lm)
n_lora = sum(x.numel() for x in lm.parameters() if x.requires_grad)
log(f"loaded {type(model).__name__} {t_load:.0f}s layers={tcfg.num_hidden_layers} lora r{args.r} "
    f"params={n_lora / 1e6:.2f}M targets={targets}")

if args.replicate > 1:
    mesh = init_device_mesh(dev.type, (args.replicate, world // args.replicate), mesh_dim_names=("replicate", "shard"))
else:
    mesh = init_device_mesh(dev.type, (world,))
mp = None if CPU else MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
fs_kw = {"mesh": mesh} | ({"mp_policy": mp} if mp else {})
for layer in bb.layers:
    fully_shard(layer, **fs_kw)
# embed + lm_head replicated (frozen, no grads): the vocab-chunked logprob path calls lm_head a data-dependent
# number of times per rank, which must not issue FSDP collectives.
head_params = set(bb.embed_tokens.parameters()) | set(lm.lm_head.parameters())
bb.embed_tokens.to(dev)
lm.lm_head.to(dev)
fully_shard(lm, ignored_params=head_params, **fs_kw)
for name_, buf in lm.named_buffers():  # rotary inv_freq etc. that FSDP did not move
    if buf.device != dev:
        buf.data = buf.data.to(dev)
sync()
mem_static = torch.cuda.memory_allocated(dev) / 2**30 if not CPU else 0.0
trainable = [(n, x) for n, x in lm.named_parameters() if x.requires_grad]
opt = torch.optim.AdamW([x for _, x in trainable], lr=args.lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0)
log(f"sharded: mesh={tuple(mesh.shape)} static_mem={mem_static:.2f}GiB trainable_tensors={len(trainable)}")


def adapter_key(n):
    # lm param "model.layers.3.linear_attn.in_proj_qkv.lora_A.default.weight"
    #  -> vLLM/PEFT "base_model.model.model.language_model.layers.3.linear_attn.in_proj_qkv.lora_A.weight"
    n = n.replace(".default", "")
    if n.startswith("model.language_model.layers."):
        return "base_model.model." + n
    assert n.startswith("model.layers."), n
    return "base_model.model.model.language_model." + n[len("model."):]


def save_adapter(version):
    """Collective: every rank gathers each LoRA tensor; rank 0 writes the PEFT dir. Returns (dir, stats)."""
    from safetensors.torch import save_file
    t = time.time()
    sd = {}
    for n, x in trainable:
        full = x.full_tensor() if hasattr(x, "full_tensor") else x.data
        if rank == 0:
            sd[adapter_key(n)] = full.detach().float().cpu().contiguous()
        del full
    t_gather = time.time() - t
    d = os.path.join(args.out, "adapters", f"{args.adapter_name}-v{version}")
    if rank == 0:
        tmp = d + ".tmp"
        os.makedirs(tmp, exist_ok=True)
        save_file(sd, os.path.join(tmp, "adapter_model.safetensors"), metadata={"format": "pt"})
        json.dump({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": args.r, "lora_alpha": args.alpha,
                   "lora_dropout": 0.0, "bias": "none", "target_modules": targets, "fan_in_fan_out": False,
                   "inference_mode": True, "modules_to_save": None, "use_rslora": False, "use_dora": False,
                   "base_model_name_or_path": args.model}, open(os.path.join(tmp, "adapter_config.json"), "w"))
        if os.path.exists(d):
            import shutil
            shutil.rmtree(d)
        os.rename(tmp, d)
    dist.barrier()
    return d, {"gather_s": round(t_gather, 3), "save_s": round(time.time() - t, 3), "n_tensors": len(sd) or None}


# ------------------------------------------------------------------------------------------- rollout (rank 0)
ANSWER_RE = re.compile(r"<answer>\s*([^<]*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def reward_of(text, answer, n_tok, finish):
    tags = ANSWER_RE.findall(text or "")
    m = re.match(r"^[^A-Za-z]*([A-E])\b", tags[-1]) if tags else None
    if m is None:
        base = -0.5
    else:
        base = 1.0 if (answer and m.group(1).upper() == str(answer).strip().upper()[:1]) else 0.0
    return base + 0.05 * (1.0 - n_tok / args.max_tokens), base


class Rollout:
    def __init__(self):
        import requests
        self.s = requests.Session()
        self.s.mount("http://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=args.rollout_workers + 8))
        ans = {}
        if args.answers and os.path.exists(args.answers):
            for line in open(args.answers):
                r = json.loads(line)
                if r.get("answer") is not None:
                    ans[r["question_id"]] = r["answer"]
        rows = []
        for line in open(args.prompts):
            r = json.loads(line)
            ids = r["input_ids"]
            # > 2*ALIGN tokens so every group shares a prefix (P > 0): the per-seq fallback issues a different
            # number of FSDP collectives than the prefix path and would hang the other ranks
            if 2 * ps.ALIGN < len(ids) <= args.max_prompt and (not ans or r["question_id"] in ans):
                rows.append((r["question_id"], ids, ans.get(r["question_id"])))
        random.Random(args.seed).shuffle(rows)
        self.rows = rows
        self.version = 0                     # newest adapter loaded into every backend
        self.done_step = 0                   # trainer steps finished
        self.cv = threading.Condition()
        self.q: queue.Queue = queue.Queue()
        self.stop = False
        self.err = None
        self.prec = dict(kv.split("=", 1) for kv in args.backend_precision.split(",") if kv)
        self.th = threading.Thread(target=self.run, daemon=True)
        log(f"rollout: {len(rows)} prompts (<= {args.max_prompt} tok, with answer: {bool(ans)})")

    def one_group(self, gid, qid, ids, answer, version):
        name = f"{args.adapter_name}-v{version}"
        body = {"model": name, "prompt": ids, "n": args.G, "max_tokens": args.max_tokens,
                "temperature": args.temperature, "top_p": 1.0, "logprobs": 1,
                "return_token_ids": True, "skip_special_tokens": False, "seed": args.seed * 100003 + gid}
        t = time.time()
        r = self.s.post(args.router + "/v1/completions", json=body, timeout=args.request_timeout)
        if r.status_code != 200:
            raise RuntimeError(f"group {gid} on {name}: HTTP {r.status_code} {r.text[:300]}")
        be = r.headers.get("X-Router-Backend", "?")
        out = r.json()
        comps = []
        for ch in sorted(out["choices"], key=lambda c: c.get("index", 0)):
            toks = ch.get("token_ids")
            lps = ch["logprobs"]["token_logprobs"]
            assert toks is not None and len(toks) == len(lps) and len(toks) >= 1, "need token_ids + logprobs"
            rw, base = reward_of(ch.get("text", ""), answer, len(toks), ch.get("finish_reason"))
            comps.append({"ids": toks, "lp": [float(x) if x is not None else 0.0 for x in lps], "reward": rw,
                          "base": base, "finish": ch.get("finish_reason")})
        return {"gid": gid, "qid": qid, "prompt": ids, "comps": comps, "backend": be,
                "precision": self.prec.get(be, f"backend{be}"), "version": version,
                "gen_s": round(time.time() - t, 3), "t_done": time.time()}

    def run(self):
        try:
            gid = 0
            pool = cf.ThreadPoolExecutor(args.rollout_workers)
            for b in range(1, args.steps + 1):
                with self.cv:
                    while self.done_step < b - 1 - args.max_stale and not self.stop:
                        self.cv.wait(1.0)
                    if self.stop:
                        return
                    v = self.version
                t = time.time()
                futs = []
                for _ in range(args.groups):
                    qid, ids, a = self.rows[gid % len(self.rows)]
                    futs.append(pool.submit(self.one_group, gid, qid, ids, a, v))
                    gid += 1
                groups = [f.result() for f in futs]
                ntok = sum(len(c["ids"]) for g in groups for c in g["comps"])
                rec = {"batch": b, "version": v, "groups": groups, "gen_wall_s": round(time.time() - t, 3),
                       "gen_tokens": ntok, "t_submit": t, "t_ready": time.time()}
                log(f"rollout batch {b}: v{v} {len(groups)} groups {ntok} tok in {rec['gen_wall_s']:.1f}s "
                    f"({ntok / max(rec['gen_wall_s'], 1e-9):.0f} tok/s) backends="
                    f"{sorted({g['backend'] for g in groups})}")
                self.q.put(rec)
        except Exception as e:  # noqa: BLE001
            self.err = traceback.format_exc()
            log("ROLLOUT ERROR", self.err)
            self.q.put(None)

    def router(self, method, path, **kw):
        r = self.s.request(method, args.router + path, timeout=args.request_timeout, **kw)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {"text": r.text[:500]}


ro = None
if rank == 0:
    ro = Rollout()

# version 0 = zero-B adapter (identical to the base); every rollout goes through the LoRA path from step 1
d0, st0 = save_adapter(0)
if rank == 0:
    code, res = ro.router("POST", "/v1/load_lora_adapter", json={"lora_name": f"{args.adapter_name}-v0", "lora_path": d0})
    write_jsonl("sync.jsonl", {"version": 0, "http": code, **st0, "router": res})
    if code != 200:
        log("FATAL: router did not load v0", code, json.dumps(res)[:800])
ok = torch.tensor([1 if (rank != 0 or code == 200) else 0], device=dev)
dist.all_reduce(ok, op=dist.ReduceOp.MIN)
if int(ok.item()) != 1:
    dist.destroy_process_group()
    sys.exit(2)
if rank == 0:
    ro.th.start()


# ---------------------------------------------------------------------------------------------- training
def step_lr(step):
    return 0.0 if step <= args.lr0_steps else args.lr


def bcast(obj):
    lst = [obj]
    dist.broadcast_object_list(lst, src=0)
    return lst[0]


summary = []
peak_reset = (lambda: torch.cuda.reset_peak_memory_stats(dev)) if not CPU else (lambda: None)
peak = (lambda: torch.cuda.max_memory_allocated(dev) / 2**30) if not CPU else (lambda: 0.0)
status = "ok"
try:
    for step in range(1, args.steps + 1):
        t_step = time.time()
        # ---- wait for the batch (rank 0), ship it to all ranks
        if rank == 0:
            batch = ro.q.get(timeout=args.max_step_s)
            if batch is None:
                raise RuntimeError("rollout failed: " + str(ro.err)[-500:])
            compact = {"batch": batch["batch"], "version": batch["version"],
                       "groups": [{k: g[k] for k in ("gid", "prompt", "comps", "precision", "backend", "version")}
                                  for g in batch["groups"]]}
        else:
            compact = None
        t_wait = time.time() - t_step
        compact = bcast(compact)
        t_recv = time.time()
        groups = compact["groups"]
        mine = groups[rank::world]          # FSDP/HSDP ranks are all data-parallel
        n_seq_global = sum(len(g["comps"]) for g in groups)
        for gp in opt.param_groups:
            gp["lr"] = step_lr(step)
        peak_reset()
        # ---- forward/backward (prefix sharing, chunked branches)
        sync()
        t_fb = time.time()
        rho_tok, rho_seq = {}, {}
        verify = {}
        fwd_tokens = unshared = 0
        state_gn = []
        for gi, g in enumerate(mine):
            rs = torch.tensor([c["reward"] for c in g["comps"]], dtype=torch.float64)
            adv = ((rs - rs.mean()) / (rs.std(unbiased=False) + 1e-4)).tolist()
            prompt = torch.tensor(g["prompt"], device=dev)
            comps = [torch.tensor(c["ids"], device=dev) for c in g["comps"]]
            olds = [torch.tensor(c["lp"], device=dev, dtype=torch.float32) for c in g["comps"]]
            prec = g["precision"]

            def loss_fn(i, lp, olds=olds, adv=adv, prec=prec):
                d = lp.float() - olds[i]
                with torch.no_grad():
                    rho_tok.setdefault(prec, []).extend(d.abs().tolist())
                    rho_seq.setdefault(prec, []).append(float(d.mean()))
                s = torch.exp(d.mean())
                a = adv[i]
                surr = torch.minimum(s * a, s.clamp(1 - args.eps_lo, 1 + args.eps_hi) * a)
                return -surr / n_seq_global * world  # FSDP averages grads over ranks

            chunks_max = math.ceil(max(len(gg["comps"]) for gg in groups) / args.chunk)
            out = ps.prefix_group_backward(lm, prompt, comps, loss_fn, chunk=args.chunk,
                                           temperature=args.temperature, offload=bool(args.offload),
                                           lm_chunk=args.lm_chunk, fsdp_modules=[lm],
                                           n_chunks_sync=chunks_max, final_sync=(gi == len(mine) - 1))
            fwd_tokens += out["stats"]["forward_tokens"]
            unshared += out["stats"]["unshared_tokens"] if "unshared_tokens" in out["stats"] else 0
            if "state_grad_norm" in out["stats"]:
                state_gn.append(out["stats"]["state_grad_norm"])
            if step == 1 and gi == 0 and args.verify_prefix:
                k = min(args.verify_prefix, len(comps))
                ref = ps.perseq_group_logprobs(lm, prompt, comps[:k], args.temperature, args.lm_chunk)
                verify = {"n": k, "max_abs_logp_diff": max(float((a.float() - b.float()).abs().max())
                                                           for a, b in zip(ref, out["logps"][:k])),
                          "mean_abs_logp_diff": sum(float((a.float() - b.float()).abs().mean())
                                                    for a, b in zip(ref, out["logps"][:k])) / k}
        sync()
        t_fb = time.time() - t_fb
        # ---- optimizer
        t_opt = time.time()
        gn = torch.nn.utils.clip_grad_norm_([x for _, x in trainable], args.clip_grad)
        gn = float(gn.full_tensor() if hasattr(gn, "full_tensor") else gn)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync()
        t_opt = time.time() - t_opt
        # ---- adapter-only sync: gather + write + router push/load
        t_sync = time.time()
        dk, st = save_adapter(step)
        router_rec = {}
        if rank == 0:
            code, res = ro.router("POST", "/v1/load_lora_adapter",
                                  json={"lora_name": f"{args.adapter_name}-v{step}", "lora_path": dk})
            if code != 200:
                raise RuntimeError(f"router load v{step}: {code} {json.dumps(res)[:500]}")
            with ro.cv:
                ro.version = step
                ro.done_step = step
                ro.cv.notify_all()
            router_rec = {k: res.get(k) for k in ("ship_bytes", "stage_s", "hash_s", "push_wall_s", "load_wall_s",
                                                  "total_s", "n_backends", "loads", "pushes")}
            old = step - 2
            if old >= 0 and not args.keep_adapters:
                ro.router("POST", "/v1/unload_lora_adapter", json={"lora_name": f"{args.adapter_name}-v{old}"})
        t_sync = time.time() - t_sync
        # ---- metrics (gather per-rank lists to rank 0)
        loc = {"rho_tok": rho_tok, "rho_seq": rho_seq, "fb_s": t_fb, "peak": peak(), "fwd_tokens": fwd_tokens,
               "unshared": unshared, "verify": verify, "state_gn": state_gn}
        allv = [None] * world
        dist.all_gather_object(allv, loc)
        step_s = time.time() - t_step
        if rank == 0:
            tok, seqs = {}, {}
            for v in allv:
                for k_, xs in v["rho_tok"].items():
                    tok.setdefault(k_, []).extend(xs)
                for k_, xs in v["rho_seq"].items():
                    seqs.setdefault(k_, []).extend(xs)
            rho = {k_: {"n_tok": len(tok[k_]), "tok_abs_p50": q(tok[k_], .5), "tok_abs_p90": q(tok[k_], .9),
                        "tok_abs_p99": q(tok[k_], .99), "tok_abs_mean": sum(tok[k_]) / len(tok[k_]),
                        "n_seq": len(seqs[k_]), "seq_abs_p50": q([abs(x) for x in seqs[k_]], .5),
                        "seq_abs_p90": q([abs(x) for x in seqs[k_]], .9),
                        "seq_signed_mean": sum(seqs[k_]) / len(seqs[k_])} for k_ in tok}
            gen_tok = sum(len(c["ids"]) for g in groups for c in g["comps"])
            rews = [c["reward"] for g in groups for c in g["comps"]]
            base = [c["base"] for g in groups for c in g["comps"]]
            trunc = sum(1 for g in groups for c in g["comps"] if c["finish"] == "length")
            rec = {"step": step, "lr": step_lr(step), "batch_version": compact["version"],
                   "staleness": (step - 1) - compact["version"], "samples": n_seq_global, "groups": len(groups),
                   "gen_tokens": gen_tok, "prompt_tokens": sum(len(g["prompt"]) for g in groups),
                   "rollout_wait_s": round(t_wait, 3), "bcast_s": round(t_recv - t_step - t_wait, 3),
                   "fwdbwd_s_max": round(max(v["fb_s"] for v in allv), 3),
                   "fwdbwd_s_min": round(min(v["fb_s"] for v in allv), 3),
                   "opt_s": round(t_opt, 3), "sync_s": round(t_sync, 3), "adapter_save": st, "router": router_rec,
                   "step_s": round(step_s, 3), "grad_norm": gn,
                   "train_fwd_tokens": sum(v["fwd_tokens"] for v in allv),
                   "train_unshared_tokens": sum(v["unshared"] for v in allv),
                   "peak_alloc_gib_max": round(max(v["peak"] for v in allv), 2),
                   "reward_mean": sum(rews) / len(rews), "acc_letter": sum(1 for x in base if x == 1.0) / len(base),
                   "unparsed_frac": sum(1 for x in base if x == -0.5) / len(base), "trunc_frac": trunc / len(base),
                   "abs_log_rho": rho, "verify_prefix": [v["verify"] for v in allv if v["verify"]],
                   "backends": sorted({g["backend"] for g in groups}),
                   "precisions": sorted({g["precision"] for g in groups}),
                   "state_grad_norm": [x for v in allv for x in v["state_gn"]][:8],
                   "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            write_jsonl("steps.jsonl", rec)
            summary.append(rec)
            log(f"STEP {step} v{compact['version']} stale={rec['staleness']} samples={n_seq_global} "
                f"wait={t_wait:.1f}s fb={rec['fwdbwd_s_max']:.1f}s opt={t_opt:.2f}s sync={t_sync:.1f}s "
                f"step={step_s:.1f}s gn={gn:.3e} rew={rec['reward_mean']:.3f} peak={rec['peak_alloc_gib_max']}GiB "
                f"rho={json.dumps({k_: (round(v_['tok_abs_p50'], 5), round(v_['tok_abs_p90'], 5), round(v_['seq_abs_p50'], 5), round(v_['seq_abs_p90'], 5)) for k_, v_ in rho.items()})}")
except Exception:  # noqa: BLE001
    status = "error"
    err = traceback.format_exc()
    log("TRAINER ERROR", err, all_ranks=True)
    with open(os.path.join(args.out, f"error.rank{rank}.txt"), "w") as f:
        f.write(err)
    if ro is not None:
        ro.stop = True
    # a rank that failed cannot finish the collectives the others are blocked in: exit hard, the launcher's
    # teardown kills the process group
    os._exit(3)

# ---------------------------------------------------------------------------------------------- teardown
if rank == 0:
    ro.stop = True
    served = {}
    code, st = ro.router("GET", "/router/state")
    if code == 200:
        served = st
    meas = summary[1:] or summary
    tot = {"status": status, "steps": len(summary), "world": world, "replicate": args.replicate,
           "groups": args.groups, "G": args.G, "max_tokens": args.max_tokens, "chunk": args.chunk,
           "samples_per_step": summary[0]["samples"] if summary else None,
           "mean_step_s_excl_first": sum(r["step_s"] for r in meas) / len(meas) if meas else None,
           "samples_per_s": (sum(r["samples"] for r in meas) / sum(r["step_s"] for r in meas)) if meas else None,
           "gen_tok_per_s_e2e": (sum(r["gen_tokens"] for r in meas) / sum(r["step_s"] for r in meas)) if meas else None,
           "adapter_versions_loaded": [r["step"] for r in summary], "router_state": served,
           "total_s": round(time.time() - T0, 1), "load_s": t_load, "static_mem_gib": mem_static}
    json.dump(tot, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    log("SUMMARY", json.dumps({k: v for k, v in tot.items() if k != "router_state"}))
dist.barrier()
dist.destroy_process_group()
log("DONE")
