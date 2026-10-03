#!/usr/bin/env python3
"""v3_2 gate: sub-batched prefix-share loss (rlforge_v3_2) vs the v3_1 whole-row path, real ckpt weights.

Both paths run the REAL trainer code: GSPOAsyncGRPOTrainer.compute_loss (v3_1 branch, _sb_tokens=0) and
GSPOAsyncGRPOTrainer._compute_loss_subbatched (v3_2), bound to a minimal stand-in for the trainer object
(accelerator = identity reduce/gather, 1 process). Model construction = TRL AsyncGRPOTrainer
(create_model_from_path + FA3 + fp32 master + text-only requires_grad + patch_chunked_lm_head + non-reentrant grad
ckpt), policy under bf16 autocast (accelerate mixed_precision=bf16), KL ref = bf16 eval copy (RLFORGE_REF_MODEL or the
policy checkpoint), called without autocast under no_grad -- as in production.

Modes
  numerics  per group: per-seq log-ratio new-vs-old (|mean|, p99, max), loss, KL, grad cosine and norm ratio,
            for each sub-batch config in --configs.
  speed     fwd+bwd wall time / peak memory per config on the same rows (AdamW state allocated so the
            peak includes optimizer memory like a real rank).
  memcal    per-token activation cost (KB/token/layer) with ckpt all vs none -> RLFORGE_SB_MEM_* constants.
  ddp       (accelerate launch --num_processes 2 --mixed_precision bf16) real DDP: gas=2 with no_sync, ranks with
            different sub-batch counts; final grads vs manual all-reduce of per-rank v3_1-path grads; no hang.

Groups: real prompts (v3.1e task_split qids -> train pool), completions = token windows of real rollout text
(rollout_samples.jsonl of v3.1b/v3.1e) with lengths from a profile:
  longtail  32 x ~16k (24 truncated at 16384 + 8 in 8k-15k): the v3_1 no-ckpt OOM case (~0.5M tokens)
  typical   lognormal around the v3.1e median group (mean ~2k, one 16k truncation)
  short     mean ~700
  mixed2    row = typical group + 16 completions of a second (short) group  (multi-group / partial-group row)
"""
import argparse
import ast
import gzip
import json
import math
import os
import random
import sys
import time
import types
from collections import defaultdict

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch  # noqa: E402

CK = os.environ.get("GATE_CKPT", "/data/shared/guoshaoyang/aiq_rl_store/models/v3_1b_ckpt50_20261003")
REF = os.environ.get("GATE_REF", "")
BUNDLE = os.environ.get("GATE_BUNDLE", "bundle")
EPS, BETA, CAP = 4e-3, 0.05, 16384


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------------------------------- data
def build_rows(tok, kinds, seed=0):
    rng = random.Random(seed)
    groups = json.load(gzip.open(os.path.join(BUNDLE, "groups_v3_1e.json.gz")))
    texts = [json.loads(l)["completion"] for l in gzip.open(os.path.join(BUNDLE, "rollout_texts.jsonl.gz"), "rt")]
    rng.shuffle(texts)
    stream = []
    for t in texts:
        stream += tok(t, add_special_tokens=False)["input_ids"]
        if len(stream) > 3_000_000:
            break
    eos = tok.convert_tokens_to_ids("<|im_end|>")

    def comp(n, truncated):
        s = rng.randrange(0, len(stream) - n - 1)
        c = stream[s: s + n]
        if not truncated:
            c = c[:-1] + [eos]
        return c

    def prompt(i):
        msgs = ast.literal_eval(groups[i]["prompt"]) if isinstance(groups[i]["prompt"], str) else groups[i]["prompt"]
        return list(tok.apply_chat_template(msgs, tokenize=True, return_dict=True, add_generation_prompt=True,
                                            enable_thinking=True)["input_ids"])

    def lengths(kind):
        if kind == "longtail":
            return [CAP] * 24 + [rng.randint(8000, 15000) for _ in range(8)]
        if kind == "typical":
            ls = [min(CAP - 1, max(64, int(rng.lognormvariate(math.log(1700), 0.55)))) for _ in range(31)]
            return ls + [CAP]
        if kind == "short":
            return [max(32, int(rng.lognormvariate(math.log(650), 0.4))) for _ in range(32)]
        raise ValueError(kind)

    rows = {}
    for gi, kind in enumerate(kinds):
        if kind == "mixed2":
            pa, pb = prompt(100 + gi), prompt(200 + gi)
            seqs = [(pa, comp(n, n >= CAP)) for n in lengths("typical")]
            seqs += [(pb, comp(n, False)) for n in lengths("short")[:16]]
        else:
            p = prompt(10 * gi + 3)
            seqs = [(p, comp(n, n >= CAP)) for n in lengths(kind)]
        rows[kind] = seqs
        log(f"[data] {kind}: n={len(seqs)} prompt={len(seqs[0][0])} completion tokens={sum(len(c) for _, c in seqs)} "
            f"max={max(len(c) for _, c in seqs)} unshared={sum(len(p) + len(c) for p, c in seqs)}")
    return rows


def pack(seqs, dev="cuda"):
    ids, pos, cm = [], [], []
    for p, c in seqs:
        ids += p + c
        pos += list(range(len(p) + len(c)))
        cm += [0] * len(p) + [1] * len(c)
    t = lambda x, dt=torch.long: torch.tensor([x], device=dev, dtype=dt)  # noqa: E731
    return t(ids), t(pos), t(cm)


def make_inputs(seqs, base_lp=None, seed=0, n_mb=None, n_ranks=1):
    """compute_loss inputs for one row (batch of 1 row, no inter-rank padding). old_log_probs = base_lp (v3_1-path
    policy lp, detached) + per-seq offset N(0, 3e-3) + per-token N(0, 0.05) -> realistic |log rho| ~ eps with some
    sequence clipping; advantages per seq N(0,1)."""
    g = torch.Generator().manual_seed(seed)
    ids, pos, cm = pack(seqs)
    T = ids.shape[1]
    adv = torch.zeros(1, T)
    old = torch.zeros(1, T)
    a = 0
    for k, (p, c) in enumerate(seqs):
        L = len(p) + len(c)
        adv[0, a: a + L] = torch.randn(1, generator=g).item()
        a += L
    if base_lp is not None:
        offs = torch.zeros(T)
        a = 0
        for k, (p, c) in enumerate(seqs):
            L = len(p) + len(c)
            offs[a: a + L] = torch.randn(1, generator=g).item() * 3e-3
            a += L
        noise = torch.randn(T, generator=g) * 0.05
        # old_log_probs[t] is the lp of token t; base_lp[t-1] scores token t
        old[0, 1:] = (base_lp.cpu()[0] + offs[1:] + noise[1:]) * cm[0, 1:].cpu()
    n = len(seqs) if n_mb is None else n_mb
    ntok = float(cm.sum())
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "completion_mask": cm,
            "old_log_probs": old.cuda(), "position_ids": pos, "advantages": adv.cuda(),
            "global_n_tokens": torch.tensor([ntok]), "global_n_forward_tokens": torch.tensor([float(T * n / len(seqs))]),
            "mean_seq_len": torch.tensor([float(T) / len(seqs)])}


# ---------------------------------------------------------------------------------------------------- models
def load_models(with_ref=True):
    if os.environ.get("RLFORGE_FUSED_OPS"):
        from rlforge.fused_ops import install as _fi
        _fi()
    from transformers import PreTrainedModel
    from trl.trainer.utils import create_model_from_path, patch_chunked_lm_head
    import rlforge.prefix_share as ps
    m = create_model_from_path(CK, device_map=None, attn_implementation="kernels-community/flash-attn3",
                               dtype=torch.float32).cuda()
    tc = m.config.get_text_config()
    if tc is not m.config:
        text = next(mod for mod in m.modules()
                    if isinstance(mod, PreTrainedModel) and mod is not m and mod.config is tc)
        m.requires_grad_(False)
        text.requires_grad_(True)
        m.get_output_embeddings().requires_grad_(True)
    patch_chunked_lm_head(m, chunk_size=8192, temperature=1.0)
    ref = None
    if with_ref:
        if REF:
            ref = type(m).from_pretrained(REF, torch_dtype=torch.bfloat16,
                                          attn_implementation=m.config._attn_implementation).cuda().eval()
            patch_chunked_lm_head(ref, chunk_size=8192, temperature=1.0)
        else:
            import copy
            ref = copy.deepcopy(m).to(torch.bfloat16).eval()
            noise = float(os.environ.get("GATE_REF_NOISE", "0"))
            if noise > 0:  # a reference that differs from the policy (KL ~ production's 1e-2) when ckpt175 is absent
                g = torch.Generator(device="cuda").manual_seed(1234)
                with torch.no_grad():
                    for p in ref.parameters():
                        if p.dim() >= 2:
                            p.add_(torch.randn(p.shape, generator=g, device="cuda", dtype=torch.float32).to(p.dtype)
                                   * (p.float().std() * noise).to(p.dtype))
        ref.requires_grad_(False)
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.train()
    # accelerate mixed_precision=bf16: autocast wrapper on the module forward (as accelerator.prepare_model does)
    from accelerate.utils import convert_outputs_to_fp32
    ps.install(m, temperature=1.0)
    fwd = m.forward.__func__
    m.forward = types.MethodType(convert_outputs_to_fp32(torch.autocast("cuda", dtype=torch.bfloat16)(fwd)), m)
    if ref is not None:
        ps.install(ref, temperature=1.0)
    log("attn_impl", m.config._attn_implementation, "| trainable", sum(p.numel() for p in m.parameters() if p.requires_grad))
    return m, ref


class _Acc:
    num_processes = 1
    process_index = 0
    is_main_process = True

    def __init__(self, n=1, idx=0, real=None):
        self.num_processes, self.process_index, self.real = n, idx, real

    def reduce(self, t, reduction="sum"):
        return self.real.reduce(t, reduction=reduction) if self.real else t

    def gather(self, t):
        return self.real.gather(t) if self.real else t

    def backward(self, loss):
        if self.real:
            self.real.backward(loss)
        else:
            loss.backward()

    def unwrap_model(self, m):
        from rlforge.prefix_share import _unwrap
        return _unwrap(m)


def fake_trainer(model, ref, sb_tokens=0, sb_ckpt="all", gas=1, acc=None):
    from rlforge.trainer import GSPOAsyncGRPOTrainer as G
    f = types.SimpleNamespace()
    for name in ("compute_loss", "_compute_loss_subbatched", "_loss_metrics", "_sb_ckpt_layers", "_write_poslog"):
        setattr(f, name, types.MethodType(getattr(G, name), f))
    f.accelerator = acc or _Acc()
    f._prefix_share, f._per_seq_forward, f.aux_loss_enabled = True, True, False
    f.epsilon_low = f.epsilon_high = EPS
    f._dyn_low_frac, f._dyn_low_ema, f._gspo_norm = 0.0, None, "seq_mean"
    f._ref_model, f._kl_beta = ref, (BETA if ref is not None else 0.0)
    f.current_gradient_accumulation_steps = gas
    f._metrics = {"train": defaultdict(list)}
    f._audit_local_seqs, f._poslog_dir, f._poslog_steps = 0, None, 0
    f.state = types.SimpleNamespace(global_step=1)
    f._step_forward_tokens = f._step_trained_tokens = f._step_seq_len_weighted = 0.0
    f._step_samples = f._step_forward_s = 0.0
    f._sb_tokens, f._sb_ckpt = sb_tokens, sb_ckpt
    f._sb_n_layers = model.config.get_text_config().num_hidden_layers if hasattr(model, "config") else 24
    f._sb_act_bytes = float(os.environ.get("RLFORGE_SB_ACT_GB", "90")) * 2**30
    f._sb_kb_full = float(os.environ.get("RLFORGE_SB_MEM_FULL_KB", "100"))
    f._sb_kb_ckpt = float(os.environ.get("RLFORGE_SB_MEM_CKPT_KB", "8"))
    f._sb_kb_base = float(os.environ.get("RLFORGE_SB_MEM_BASE_KB", "64"))
    f._sb_calls = 0
    return f


def grads(model):
    return torch.cat([p.grad.detach().float().flatten() if p.grad is not None else torch.zeros(p.numel(), device="cuda")
                      for p in model.parameters() if p.requires_grad])


def zero(model):
    for p in model.parameters():
        p.grad = None


def run_path(model, ref, inputs, sb_tokens, sb_ckpt, want_grad=True):
    """One micro-batch through the real code. Returns (loss value, per-token lp (T-1) detached, grads|None, metrics,
    seconds, peak GiB)."""
    zero(model)
    f = fake_trainer(model, ref, sb_tokens, sb_ckpt)
    lp_holder = {}
    _set_variant(new=bool(sb_tokens) or BASE_IS_NEW)
    import rlforge.prefix_share as ps
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    if sb_tokens:
        loss = f.compute_loss(model, inputs)
    else:
        # v3_1 path: capture the policy log_probs the loss used (prefix_shared_logprobs output)
        orig = ps.prefix_shared_logprobs

        def cap(*a, **k):
            r = orig(*a, **k)
            if torch.is_grad_enabled():
                lp_holder["lp"] = r[0].detach()
            return r
        ps.prefix_shared_logprobs = cap
        try:
            loss = f.compute_loss(model, inputs)
        finally:
            ps.prefix_shared_logprobs = orig
    if want_grad:
        f.accelerator.backward(loss)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    g = grads(model) if want_grad else None
    return float(loss.detach()), lp_holder.get("lp"), g, {k: v for k, v in f._metrics["train"].items()}, dt, peak, f


BASE_IS_NEW = os.environ.get("GATE_BASE_NEW", "0") == "1"   # 1: v3_1 path also gets fast_logprob/fused ops
_FAST_ENV = os.environ.get("RLFORGE_FAST_LOGPROB", "0")


def _set_variant(new):
    """v3_1 reference = TRL log-prob + stock transformers ops; v3_2 = env settings (RLFORGE_FAST_LOGPROB,
    RLFORGE_FUSED_OPS)."""
    os.environ["RLFORGE_FAST_LOGPROB"] = _FAST_ENV if new else "0"
    if os.environ.get("RLFORGE_FUSED_OPS"):
        import rlforge.fused_ops as fo
        fo.set_enabled(new)


def subbatch_lp(model, inputs, sb_tokens):
    """policy per-token lp of the v3_2 path (no grad, same kernels) in the (1, T-1) layout"""
    import rlforge.prefix_share as ps
    mb = inputs["attention_mask"].bool()
    ids = inputs["input_ids"][mb][None]
    cm = inputs["completion_mask"][mb][None]
    pos = inputs["position_ids"][mb][None]
    out = torch.zeros(1, ids.shape[1] - 1, device="cuda")
    _set_variant(True)
    with torch.no_grad():
        for sb in ps.plan_subbatches(ids, pos, cm, sb_tokens):
            o = model(prefix_share=dict(input_ids=ids, completion_mask=cm, subbatch=sb, ckpt_layers=set()))
            out[0, o["slots"]] = o["log_probs"].float()
    return out


def seq_diff(lp_a, lp_b, inputs):
    pos = inputs["position_ids"][0]
    cm = inputs["completion_mask"][0, 1:].bool()
    seq = (pos == 0).cumsum(0)[1:] - 1
    n = int((pos == 0).sum())
    d = (lp_a[0] - lp_b[0]).float() * cm
    s = torch.zeros(n, device=d.device).index_add(0, seq, d)
    c = torch.zeros(n, device=d.device).index_add(0, seq, cm.float()).clamp(min=1)
    m = s / c
    a = m.abs()
    tok = d[cm].abs()
    return {"seq_mean": float(m.mean()), "seq_absmean": float(a.mean()), "seq_p99": float(torch.quantile(a, 0.99)),
            "seq_max": float(a.max()), "tok_absmean": float(tok.mean()), "tok_max": float(tok.max())}


# ---------------------------------------------------------------------------------------------------- modes
def parse_configs(s):
    out = []
    for c in s.split(","):
        tok, ck = c.split(":")
        out.append((int(tok), ck))
    return out


def mode_numerics(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CK)
    rows = build_rows(tok, args.kinds.split(","))
    model, ref = load_models(with_ref=not args.no_ref)
    res = {}
    for kind, seqs in rows.items():
        # base lp for old_log_probs: v3_1 path forward, no grad
        ids, pos, cm = pack(seqs)
        import rlforge.prefix_share as ps
        with torch.no_grad():
            base = model(prefix_share=dict(input_ids=ids, position_ids=pos, completion_mask=cm))["log_probs"].float()
        inputs = make_inputs(seqs, base_lp=base, seed=1)
        L0, lp0, g0, m0, t0, pk0, _ = run_path(model, ref, inputs, 0, "all")
        log(f"[{kind}] v3_1 loss={L0:.6e} kl_ref={m0.get('kl_ref')} seq_clip_low={m0['gspo/seq_clip_low_frac']} "
            f"t={t0:.2f}s peak={pk0:.1f}GiB |g|={g0.norm():.4e}")
        res[kind] = {"v3_1": {"loss": L0, "t": t0, "peak": pk0, "gnorm": float(g0.norm()),
                              "kl_ref": m0.get("kl_ref"), "abs_log_rho_p90": m0["gspo/abs_log_rho_p90"]}}
        for sbt, ck in parse_configs(args.configs):
            L1, _, g1, m1, t1, pk1, f1 = run_path(model, ref, inputs, sbt, ck)
            lp1 = subbatch_lp(model, inputs, sbt)
            sd = seq_diff(lp1, lp0, inputs)
            cos = float(torch.nn.functional.cosine_similarity(g0, g1, dim=0))
            nr = float(g1.norm() / g0.norm())
            r = {"loss": L1, "loss_rel": (L1 - L0) / max(abs(L0), 1e-12), "grad_cos": cos, "grad_norm_ratio": nr,
                 "t": t1, "peak": pk1, "subbatches": m1["v3_2/subbatches_per_row"], "pad_frac": m1["prefix_share/pad_frac"],
                 "kl_ref": m1.get("kl_ref"), "abs_log_rho_p90": m1["gspo/abs_log_rho_p90"], **sd,
                 "pass": abs(sd["seq_mean"]) < 2e-4 and sd["seq_p99"] < 1e-3 and cos > 0.99 and abs(nr - 1) < 0.02}
            res[kind][f"{sbt}:{ck}"] = r
            log(f"[{kind}] sb={sbt}:{ck} loss={L1:.6e} (rel {r['loss_rel']:+.2e}) cos={cos:.6f} norm_ratio={nr:.5f} "
                f"seq|mean|={sd['seq_mean']:+.2e} seq_p99={sd['seq_p99']:.2e} seq_max={sd['seq_max']:.2e} "
                f"tok_max={sd['tok_max']:.2e} kl_ref={r['kl_ref']} sub={r['subbatches']} pad={r['pad_frac']} "
                f"t={t1:.2f}s peak={pk1:.1f}GiB PASS={r['pass']}")
        del g0
        torch.cuda.empty_cache()
    json.dump(res, open(args.out, "w"), indent=1)
    log("wrote", args.out)


def mode_speed(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CK)
    rows = build_rows(tok, args.kinds.split(","))
    model, ref = load_models(with_ref=not args.no_ref)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0)
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.zeros_like(p)
    opt.step()  # allocate exp_avg / exp_avg_sq like a real rank
    res = {}
    cfgs = [(0, "all")] + parse_configs(args.configs)
    for kind, seqs in rows.items():
        inputs = make_inputs(seqs, base_lp=None, seed=1)
        res[kind] = {}
        for sbt, ck in cfgs:
            ts, pk = [], 0
            try:
                for rep in range(args.reps + 1):
                    L, _, _, m, t, p, f = run_path(model, ref, inputs, sbt, ck, want_grad=True)
                    if rep:
                        ts.append(t)
                    pk = max(pk, p)
                tmed = sorted(ts)[len(ts) // 2]
                ntok = inputs["input_ids"].shape[1]
                r = {"t": tmed, "peak": pk, "unshared_tok_s": ntok / tmed,
                     "subbatches": (m.get("v3_2/subbatches_per_row") or [1])[0],
                     "pad_frac": m["prefix_share/pad_frac"][0], "fwd_s": f._last_forward_time_s}
            except torch.OutOfMemoryError as e:  # noqa: F841
                r = {"oom": True}
                zero(model)
                torch.cuda.empty_cache()
            res[kind][f"{sbt}:{ck}"] = r
            log(f"[speed {kind}] {sbt}:{ck} {r}")
    json.dump(res, open(args.out, "w"), indent=1)
    log("wrote", args.out)


def mode_memcal(args):
    """KB per padded token: base (embeddings + logprob + prompt bookkeeping), per layer with/without ckpt."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CK)
    kind = args.kinds.split(",")[0]
    rows = build_rows(tok, [kind])
    model, _ = load_models(with_ref=False)
    seqs = rows[kind]
    inputs = make_inputs(seqs, seed=1)
    import rlforge.prefix_share as ps
    mb = inputs["attention_mask"].bool()
    plan = ps.plan_subbatches(inputs["input_ids"], inputs["position_ids"], inputs["completion_mask"], 1 << 30)
    sb = plan[0]
    T = sb["tokens"]
    nL = model.config.get_text_config().num_hidden_layers
    out = {}
    for name, layers in [("all", set(range(nL))), ("none", set()), ("half", set(range(nL // 2)))]:
        zero(model)
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        o = model(prefix_share=dict(input_ids=inputs["input_ids"], completion_mask=inputs["completion_mask"],
                                    subbatch=sb, ckpt_layers=layers))
        after_fwd = torch.cuda.memory_allocated()
        o["log_probs"].sum().backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        out[name] = {"held_after_fwd_kb_per_tok": (after_fwd - base) / T / 1024, "peak_kb_per_tok": (peak - base) / T / 1024}
        del o
        log(f"[memcal] {name}: T={T} {out[name]}")
    full = (out["none"]["held_after_fwd_kb_per_tok"] - out["all"]["held_after_fwd_kb_per_tok"]) / nL
    log(f"[memcal] per layer per token: no-ckpt minus ckpt = {full:.1f} KB; ckpt-all held {out['all']['held_after_fwd_kb_per_tok']:.1f} KB/token total; peak/held(none) {out['none']['peak_kb_per_tok']/max(out['none']['held_after_fwd_kb_per_tok'],1e-9):.2f}")
    json.dump(out, open(args.out, "w"), indent=1)


def mode_ddp(args):
    """Real torch DDP, 2 ranks (torchrun; NCCL if each rank has its own GPU, else gloo with both ranks on one GPU),
    gas=2 with no_sync on the first micro-batch, ranks with DIFFERENT sub-batch counts per micro-batch and a
    balanced-batcher row layout (unequal samples per rank). Checks: no hang, equal DDP forward count, and the final
    DDP-averaged grads == manual all-reduce of per-rank v3_1-path grads (cos / norm ratio)."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoTokenizer, PreTrainedModel
    from trl.trainer.utils import create_model_from_path, patch_chunked_lm_head
    from accelerate.utils import convert_outputs_to_fp32
    import rlforge.prefix_share as ps
    rank, R = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    ngpu = torch.cuda.device_count()
    dev = rank if ngpu >= R else 0
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl" if ngpu >= R else "gloo")
    tok = AutoTokenizer.from_pretrained(CK)
    mbs = build_step_samples(tok, 2, start=args.start, R=R)
    bal = ps.BalancedGroupRowBatcher([], R, 32 * R)
    plan = [bal._partition(mb)[rank] for mb in mbs]  # this rank's row of each micro-batch
    nmb = [len(mb) for mb in mbs]
    tmb = [sum(len(x["input_ids"]) for x in mb) for mb in mbs]

    m = create_model_from_path(CK, device_map=None, attn_implementation="kernels-community/flash-attn3",
                               dtype=torch.float32).cuda()
    tc = m.config.get_text_config()
    if tc is not m.config:
        text = next(mod for mod in m.modules() if isinstance(mod, PreTrainedModel) and mod is not m and mod.config is tc)
        m.requires_grad_(False)
        text.requires_grad_(True)
        m.get_output_embeddings().requires_grad_(True)
    patch_chunked_lm_head(m, chunk_size=8192, temperature=1.0)
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.train()
    ps.install(m, temperature=1.0)
    fwd = m.forward.__func__
    m.forward = types.MethodType(convert_outputs_to_fp32(torch.autocast("cuda", dtype=torch.bfloat16)(fwd)), m)
    ddp = DDP(m, device_ids=[dev])

    class A(_Acc):
        def reduce(self, t, reduction="sum"):
            t = t.clone()
            dist.all_reduce(t)
            return t / R if reduction == "mean" else t

        def gather(self, t):  # accelerate.gather semantics: 0-dim -> 1-dim, concat on dim 0
            t = t.reshape(-1) if t.dim() == 0 else t
            out = [torch.zeros_like(t) for _ in range(R)]
            dist.all_gather(out, t.contiguous())
            return torch.cat(out)

        def backward(self, loss):
            loss.backward()

    acc = A(R, rank)
    f = fake_trainer(ddp, None, sb_tokens=args.sb, sb_ckpt="auto", gas=2, acc=acc)
    f._sb_n_layers = tc.num_hidden_layers
    nsb, t0 = [], time.perf_counter()
    fwd_count = {"n": 0}
    h = ddp.register_forward_pre_hook(lambda *a, **k: fwd_count.__setitem__("n", fwd_count["n"] + 1))
    _set_variant(True)
    for j, row in enumerate(plan):
        inputs = row_inputs(row, nmb[j], tmb[j])
        ctx = ddp.no_sync() if j == 0 else torch.enable_grad()
        with ctx:
            loss = f.compute_loss(ddp, inputs)
            loss.backward()
        nsb.append(f._metrics["train"]["v3_2/subbatches_per_row"][-1])
    torch.cuda.synchronize()
    h.remove()
    log(f"[ddp] rank {rank}: rows n={[len(r) for r in plan]} sub-batches/mb={nsb} ddp_forwards={fwd_count['n']} "
        f"t={time.perf_counter() - t0:.1f}s (no hang)")
    g_ddp = torch.cat([p.grad.detach().float().flatten() for p in m.parameters() if p.requires_grad])
    for p in m.parameters():
        p.grad = None
    # reference: v3_1 math on the same rows without DDP, per-seq weight 1/(samples in micro-batch) x R
    # (= what DDP's 1/R averaging turns into 1/n_mb), accumulated over the 2 micro-batches, then all-reduced / R
    _set_variant(False)
    f2 = fake_trainer(m, None, sb_tokens=0, gas=2)
    for j, row in enumerate(plan):
        inputs = row_inputs(row, nmb[j], tmb[j])
        loss = f2.compute_loss(m, inputs)  # = mean over the row's seqs / gas
        (loss * len(row) * R / nmb[j]).backward()
    g_ref = torch.cat([p.grad.detach().float().flatten() for p in m.parameters() if p.requires_grad])
    dist.all_reduce(g_ref)
    g_ref /= R
    cos = float(torch.nn.functional.cosine_similarity(g_ddp, g_ref, dim=0))
    nr = float(g_ddp.norm() / g_ref.norm())
    log(f"[ddp] rank {rank} grad cos vs manual all-reduce of v3_1-path grads = {cos:.6f} norm ratio {nr:.5f}")
    if rank == 0:
        json.dump({"rows": [len(r) for r in plan], "subbatches_rank0": nsb, "ddp_forwards_rank0": fwd_count["n"],
                   "cos": cos, "norm_ratio": nr, "backend": dist.get_backend()}, open(args.out, "w"))
    dist.destroy_process_group()

def build_step_samples(tok, n_mb, start=0, cv=0.5, seed=0, ngen=32, R=4):
    """Micro-batches of the v3.1e group sequence (task_split order): real prompts, per-sample completion lengths
    lognormal around the group's logged mean_tokens (CV ``cv``) with the group's logged truncations at 16384,
    content = real rollout text windows. Returns [[sample dict, ...] x R*ngen] per micro-batch."""
    rng = random.Random(seed)
    groups = json.load(gzip.open(os.path.join(BUNDLE, "groups_v3_1e.json.gz")))
    texts = [json.loads(l)["completion"] for l in gzip.open(os.path.join(BUNDLE, "rollout_texts.jsonl.gz"), "rt")]
    rng.shuffle(texts)
    stream = []
    for t in texts:
        stream += tok(t, add_special_tokens=False)["input_ids"]
        if len(stream) > 3_000_000:
            break
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    s_ = math.sqrt(math.log(1 + cv * cv))
    mbs = []
    gi = start
    for _ in range(n_mb):
        batch = []
        for _r in range(R):
            gr = groups[gi]
            msgs = ast.literal_eval(gr["prompt"]) if isinstance(gr["prompt"], str) else gr["prompt"]
            p = list(tok.apply_chat_template(msgs, tokenize=True, return_dict=True, add_generation_prompt=True,
                                             enable_thinking=True)["input_ids"])
            nt = int(gr["trunc"])
            rest = ngen - nt
            target = (gr["mean_tokens"] * ngen - CAP * nt) / max(rest, 1)
            lens = [min(CAP - 1, max(16, int(rng.lognormvariate(math.log(max(target, 32)) - s_ * s_ / 2, s_))))
                    for _ in range(rest)] + [CAP] * nt
            for n in lens:
                o = rng.randrange(0, len(stream) - n - 1)
                c = stream[o: o + n]
                if n < CAP:
                    c = c[:-1] + [eos]
                batch.append({"group_id": gi, "input_ids": p + c, "completion_mask": [0] * len(p) + [1] * len(c),
                              "advantage": rng.gauss(0, 1)})
            gi += 1
        mbs.append(batch)
    return mbs


def row_inputs(row, n_mb_samples, mb_tokens):
    ids = [t for s in row for t in s["input_ids"]]
    cm = [m for s in row for m in s["completion_mask"]]
    pos = [i for s in row for i in range(len(s["input_ids"]))]
    adv = [s["advantage"] for s in row for _ in s["input_ids"]]
    t = lambda x, dt=torch.long: torch.tensor([x], device="cuda", dtype=dt)  # noqa: E731
    return {"input_ids": t(ids), "attention_mask": torch.ones(1, len(ids), device="cuda", dtype=torch.long),
            "completion_mask": t(cm), "old_log_probs": torch.zeros(1, len(ids), device="cuda"),
            "position_ids": t(pos), "advantages": t(adv, torch.float32),
            "global_n_tokens": torch.tensor([float(sum(cm))]), "global_n_forward_tokens": torch.tensor([float(mb_tokens)]),
            "mean_seq_len": torch.tensor([float(mb_tokens) / n_mb_samples])}


def mode_step(args):
    """Per-row fwd+bwd time of real-profile micro-batches, v3_1 layout (GroupRowBatcher, v3_1 path) vs the v3_2
    layout (BalancedGroupRowBatcher if --balance, v3_2 path with --configs[0]). Micro-batch time = max over the R
    rows (ranks wait for each other at the per-micro-batch collectives). Single GPU, rows run one after another."""
    from transformers import AutoTokenizer
    import rlforge.prefix_share as ps
    tok = AutoTokenizer.from_pretrained(CK)
    mbs = build_step_samples(tok, args.n_mb, start=args.start)
    model, ref = load_models(with_ref=not args.no_ref)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0)
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.zeros_like(p)
    opt.step()
    R = 4
    layouts = {"v3_1": (ps.GroupRowBatcher([], R, 128), 0, "all")}
    for c in parse_configs(args.configs):
        bat = ps.BalancedGroupRowBatcher([], R, 128) if args.balance else ps.GroupRowBatcher([], R, 128)
        layouts[f"v3_2:{c[0]}:{c[1]}" + (":bal" if args.balance else "")] = (bat, c[0], c[1])
    if args.only_v32:
        layouts.pop("v3_1")
    res = {k: {"mb_max_s": [], "mb_mean_s": [], "rows": [], "peak": 0.0} for k in layouts}
    warm = row_inputs(mbs[0][:32], 128, sum(len(s["input_ids"]) for s in mbs[0]))
    for name, (bat, sbt, ck) in layouts.items():
        run_path(model, ref, warm, sbt, ck)  # warm-up / compile
        for j, mb in enumerate(mbs):
            rows = bat._partition(mb)
            mbt = sum(len(s["input_ids"]) for s in mb)
            ts = []
            for r in rows:
                inp = row_inputs(r, len(mb), mbt)
                try:
                    _, _, _, m, t, pk, _ = run_path(model, ref, inp, sbt, ck)
                except torch.OutOfMemoryError:
                    t, pk = float("nan"), float("inf")
                    zero(model)
                    torch.cuda.empty_cache()
                ts.append(t)
                res[name]["peak"] = max(res[name]["peak"], pk)
                res[name]["rows"].append({"mb": j, "n": len(r), "tokens": inp["input_ids"].shape[1], "t": t, "peak": pk})
            res[name]["mb_max_s"].append(max(ts))
            res[name]["mb_mean_s"].append(sum(ts) / len(ts))
            log(f"[step {name}] mb {j}: rows n={[len(r) for r in rows]} t={[round(x, 2) for x in ts]} max={max(ts):.2f}")
        tot = sum(res[name]["mb_max_s"])
        res[name]["step_fwd_bwd_s_est"] = tot * 8 / len(mbs)
        res[name]["balance_bound_s"] = sum(res[name]["mb_mean_s"]) * 8 / len(mbs)
        log(f"[step {name}] est fwd_bwd/step (8 mb) = {res[name]['step_fwd_bwd_s_est']:.1f}s "
            f"(perfect-balance bound {res[name]['balance_bound_s']:.1f}s) peak {res[name]['peak']:.1f} GiB")
    json.dump(res, open(args.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["numerics", "speed", "memcal", "ddp", "step"])
    ap.add_argument("--n-mb", type=int, default=8)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--only-v32", action="store_true")
    ap.add_argument("--kinds", default="longtail,typical,short,mixed2")
    ap.add_argument("--configs", default="65536:none,131072:auto,196608:all")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--sb", type=int, default=32768)
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--out", default="v32_gate.json")
    args = ap.parse_args()
    torch.manual_seed(0)
    {"numerics": mode_numerics, "speed": mode_speed, "memcal": mode_memcal, "ddp": mode_ddp,
     "step": mode_step}[args.mode](args)


if __name__ == "__main__":
    main()
