"""Exactness + speed check: shared-prefix forward vs the production per-sequence forward.

  python check_prefix_share.py exact --G 6 [--ckpt]
  python check_prefix_share.py bench --G 32 --prompt-len 5500 --comp-len 2048 [--ckpt]

The model is loaded the way TRL's async trainer loads it (create_model_from_path + chunked lm head),
fp32 master weights under bf16 autocast (the production `--dtype none` recipe).
"""
import argparse, json, time

import torch

MODEL = "/data/home/guoshaoyang/models/Qwen3.5-0.8B-ms"
PROMPTS = "/data/home/guoshaoyang/aiq_rl/tools/bench/prompts_5k.jsonl"


def load(ckpt):
    from trl.trainer.utils import create_model_from_path, patch_chunked_lm_head
    m = create_model_from_path(MODEL, device_map=None, attn_implementation="sdpa", dtype=torch.float32).cuda()
    tc = m.config.get_text_config()
    text = next(mod for mod in m.modules() if mod is not m and getattr(mod, "config", None) is tc
                and hasattr(mod, "layers"))
    m.requires_grad_(False)
    text.requires_grad_(True)
    m.get_output_embeddings().requires_grad_(True)
    patch_chunked_lm_head(m, chunk_size=8192, temperature=1.0)
    if ckpt:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.train()
    from rlforge.prefix_share import install
    install(m, temperature=1.0)
    return m


def build_row(tok, G, prompt_len=None, comp_lens=None, seed=0):
    rows = [json.loads(l)["prompt"] for l in open(PROMPTS)]
    p = tok(tok.apply_chat_template(rows[0], tokenize=False, add_generation_prompt=True),
            add_special_tokens=False)["input_ids"]
    if prompt_len:
        while len(p) < prompt_len:
            p = p + p
        p = p[-prompt_len:]
    pool = tok("\n".join(r[-1]["content"] for r in rows[1:40]), add_special_tokens=False)["input_ids"]
    g = torch.Generator().manual_seed(seed)
    ids, pos, cm = [], [], []
    for k in range(G):
        L = comp_lens[k % len(comp_lens)]
        s = int(torch.randint(0, len(pool) - L, (1,), generator=g))
        c = pool[s : s + L]
        seq = p + c
        ids += seq
        pos += list(range(len(seq)))
        cm += [0] * len(p) + [1] * len(c)
    t = lambda x: torch.tensor([x], device="cuda")
    return t(ids), t(pos), t(cm), len(p)


def per_seq_single(model, input_ids, position_ids, completion_mask):
    """Noise-floor reference: every sequence alone at batch 1 (no padding at all)."""
    T = input_ids.shape[1]
    starts = (position_ids[0] == 0).nonzero().flatten().tolist()
    lp = torch.zeros((1, T - 1), device="cuda")
    en = torch.zeros((1, T - 1), device="cuda")
    for a, e in zip(starts, starts[1:] + [T]):
        out = model(input_ids=input_ids[:, a:e], position_ids=position_ids[:, a:e], labels=input_ids[:, a:e],
                    completion_mask=completion_mask[:, a:e], use_cache=False)
        lp[0, a : e - 1] = out["log_probs"][0]
        en[0, a : e - 1] = out["entropy"][0]
    return lp, en


AUTOCAST = True
REF = None  # frozen bf16 reference (KL term), forwarded under no_grad like the trainer does


def run(model, how, ids, pos, cm, w):
    from rlforge.trainer import per_seq_logprobs
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AUTOCAST):
        if how == "per_seq":
            lp, ent, _ = per_seq_logprobs(model, ids, pos, cm)
        elif how == "single":
            lp, ent = per_seq_single(model, ids, pos, cm)
        else:
            out = model(prefix_share=dict(input_ids=ids, position_ids=pos, completion_mask=cm))
            lp, ent = out["log_probs"], out["entropy"]
    valid = cm[:, 1:].float()
    n = valid.sum()
    loss = -(lp * w * valid).sum() / n - 0.01 * (ent * valid).sum() / n
    if REF is not None:
        with torch.no_grad():
            if how == "per_seq":
                from rlforge.trainer import per_seq_logprobs
                ref_lp, _, _ = per_seq_logprobs(REF, ids, pos, cm)
            else:
                ref_lp = REF(prefix_share=dict(input_ids=ids, position_ids=pos, completion_mask=cm))["log_probs"]
        d = (ref_lp - lp)
        loss = loss + 0.05 * ((torch.exp(d) - d - 1) * valid).sum() / n
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    grads = {k: p.grad.detach().float().clone() for k, p in model.named_parameters() if p.grad is not None}
    return dict(lp=lp.detach(), ent=ent.detach(), grads=grads, fwd_s=t1 - t0, bwd_s=t2 - t1,
                peak_gb=torch.cuda.max_memory_allocated() / 2**30)


def compare(a, b, cm):
    v = cm[0, 1:].bool()
    d = (a["lp"][0, v] - b["lp"][0, v]).abs()
    de = (a["ent"][0, v] - b["ent"][0, v]).abs()
    num = sum(((a["grads"][k] - b["grads"][k]) ** 2).sum() for k in a["grads"])
    den = sum((b["grads"][k] ** 2).sum() for k in a["grads"])
    per = sorted(((((a["grads"][k] - b["grads"][k]).norm() / b["grads"][k].norm().clamp_min(1e-30)).item(), k)
                  for k in a["grads"]), reverse=True)
    missing = sorted(set(a["grads"]) ^ set(b["grads"]))
    return {"lp_max_abs": d.max().item(), "lp_mean_abs": d.mean().item(),
            "ent_max_abs": de.max().item(), "grad_rel_l2": (num / den).sqrt().item(),
            "worst_param_rel": per[:3], "grad_keys_mismatch": missing[:5], "n_tokens": int(v.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["exact", "bench", "accel"])
    ap.add_argument("--G", type=int, default=6)
    ap.add_argument("--prompt-len", type=int, default=0)
    ap.add_argument("--comp-len", type=int, default=2048)
    ap.add_argument("--ckpt", action="store_true")
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="no autocast: isolates algorithmic error from bf16 noise")
    ap.add_argument("--detach", action="store_true", help="ablation: cut prompt<-branch grads (must FAIL)")
    ap.add_argument("--budget", type=int, default=24576, help="production token_budget (vLLM max_model_len)")
    ap.add_argument("--with-ref", action="store_true", help="add the KL reference forward (bf16, no_grad)")
    ap.add_argument("--memcap-gb", type=float, default=0, help="cap this process's GPU memory (co-tenancy)")
    ap.add_argument("--arms", default="per_seq,shared_budget,shared_whole")
    a = ap.parse_args()
    global AUTOCAST
    AUTOCAST = not a.fp32
    if a.detach:
        import rlforge.prefix_share as ps
        ps._DETACH = True
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if a.memcap_gb:
        tot = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.memcap_gb / tot))
    model = load(a.ckpt)
    if a.with_ref:
        import copy
        global REF
        REF = copy.deepcopy(model).to(torch.bfloat16).eval()
        REF.requires_grad_(False)
    if a.mode == "accel":
        # Wiring check: accelerate's prepare() (bf16 autocast + fp32 output conversion) wraps the
        # bound forward installed by prefix_share, kwargs pass through, and a deepcopy (KL reference)
        # stays bound to the copy rather than the policy.
        import copy
        from accelerate import Accelerator
        acc = Accelerator(mixed_precision="bf16")
        ref = copy.deepcopy(model).to(torch.bfloat16).eval()
        pm = acc.prepare(model)
        ids, pos, cm, P = build_row(tok, 4, 2000, [300, 500])
        out = pm(prefix_share=dict(input_ids=ids, position_ids=pos, completion_mask=cm))
        out["log_probs"].sum().backward()
        with torch.no_grad():
            r = ref(prefix_share=dict(input_ids=ids, position_ids=pos, completion_mask=cm))
            with torch.no_grad():
                for prm in model.parameters():
                    prm.add_(torch.randn_like(prm) * 1e-2) if prm.requires_grad else None
            r2 = ref(prefix_share=dict(input_ids=ids, position_ids=pos, completion_mask=cm))
        print("ACCEL ok: dtype", out["log_probs"].dtype, "stats", out["prefix_share_stats"],
              "ref_bound_to_copy", ref.forward.__self__ is ref,
              "ref_unchanged_after_policy_perturb", torch.equal(r["log_probs"], r2["log_probs"]))
        return
    if a.mode == "exact":
        ids, pos, cm, P = build_row(tok, a.G, a.prompt_len or None, [1500, 2048, 700, 1900, 1024, 333])
    else:
        ids, pos, cm, P = build_row(tok, a.G, a.prompt_len, [a.comp_len])
    torch.manual_seed(0)
    w = torch.randn(cm[:, 1:].shape, device="cuda")
    print(json.dumps({"G": a.G, "prompt_tokens": P, "row_tokens": ids.shape[1], "ckpt": a.ckpt, "autocast_bf16": AUTOCAST}), flush=True)
    if a.mode == "exact":
        base = run(model, "per_seq", ids, pos, cm, w)
        single = run(model, "single", ids, pos, cm, w)
        shared = run(model, "shared", ids, pos, cm, w)
        print("NOISE_FLOOR per_seq(padded batch) vs single(batch 1):", json.dumps(compare(single, base, cm)))
        print("SHARED vs per_seq:", json.dumps(compare(shared, base, cm)))
        print("SHARED vs single:", json.dumps(compare(shared, single, cm)))
    else:
        # One group = G x (P + C). Production planner (TokenBudgetBatcher, budget = max_model_len)
        # packs floor(budget / (P + C)) whole sequences per row -> per_seq forward per row.
        # Shared: (a) the group-aware budget planner -> rows of P + k*C <= budget; (b) whole group in one row.
        P_, C = P, a.comp_len
        k_un = max(1, a.budget // (P_ + C))
        k_sh = max(1, (a.budget - P_) // C)
        plans = {"per_seq": ("per_seq", k_un), "shared_budget": ("shared", k_sh), "shared_whole": ("shared", a.G)}
        plans = {k: v for k, v in plans.items() if k in a.arms.split(",")}
        res = {}
        for name, (how, k) in plans.items():
            def one_group():
                tot, peak, n_rows = 0.0, 0.0, 0
                for s0 in range(0, a.G, k):
                    n = min(k, a.G - s0)
                    ids, pos, cm, _ = build_row(tok, n, a.prompt_len, [C], seed=s0)
                    ww = torch.randn(cm[:, 1:].shape, device="cuda")
                    r = run(model, how, ids, pos, cm, ww)
                    tot += r["fwd_s"] + r["bwd_s"]
                    peak = max(peak, r["peak_gb"])
                    n_rows += 1
                return tot, peak, n_rows
            try:
                one_group()  # warmup
                t, pk, nr = min((one_group() for _ in range(2)), key=lambda x: x[0])
            except torch.OutOfMemoryError as e:
                print(name, "OOM", str(e)[:120], flush=True)
                torch.cuda.empty_cache()
                continue
            fwd_tok = nr * P_ + a.G * C if how == "shared" else a.G * (P_ + C)
            res[name] = {"seqs_per_row": k, "rows_per_group": nr, "group_fwd_bwd_s": round(t, 3),
                         "samples_per_s_per_gpu": round(a.G / t, 2), "forwarded_tokens_per_group": fwd_tok,
                         "peak_gb": round(pk, 1), "trained_tok_per_s": round(a.G * C / t),
                         "unshared_equiv_tok_per_s": round(a.G * (P_ + C) / t)}
            print(name, json.dumps(res[name]), flush=True)
            torch.cuda.empty_cache()
        if "per_seq" in res:
            for n in res:
                if n != "per_seq":
                    print("SPEEDUP", n, round(res["per_seq"]["group_fwd_bwd_s"] / res[n]["group_fwd_bwd_s"], 2))

if __name__ == "__main__":
    main()
