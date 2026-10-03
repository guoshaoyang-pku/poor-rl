#!/usr/bin/env python3
"""v3 GATE for rlforge.prefix_share: shared-prompt path (torch SDPA + FLA) vs the PRODUCTION FA3
per-sequence forward, ckpt57 weights, real ckpt57 probe completions, whole group per row.

Modes
  numerics  criteria (1) seq log-ratio, (2) per-token |d|, (5) per-position bias, (6) leakage,
            (3) grad check (G=8 subset + full G=32 row), (4) detach ablation
  speed     one G=32 group, ~5.5k prompt / ~2.2k mean completion: fwd+bwd(+KL ref fwd) wall time,
            peak memory, forwarded tokens; shared (whole group per row) vs production per-seq rows
            packed to token_budget 24576 (v2 TokenBudgetBatcher)
  ddp       (accelerate launch --num_processes 2 --mixed_precision bf16) shared forward under real
            DDP: unequal rows per rank, grad accumulation (no_sync), equal micro-step count, no hang
  batcher   CPU simulation of GroupRowBatcher / GroupTokenBudgetBatcher rank balance

Model construction = TRL AsyncGRPOTrainer: create_model_from_path(attn kernels-community/flash-attn3,
fp32 master) + VLM text-only requires_grad + patch_chunked_lm_head; policy forward under bf16 autocast
(accelerate mixed_precision=bf16), grad-ckpt non-reentrant; KL ref = deepcopy -> bf16, eval, called
WITHOUT autocast under no_grad (exactly as GSPOAsyncGRPOTrainer.compute_loss does).
The clean reference is the batch-1 FA3 forward of each sequence (no padding, no packing).
"""
import argparse
import copy
import glob
import json
import math
import os
import random
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch  # noqa: E402

CK = os.environ.get("GATE_CKPT", "/data/shared/guoshaoyang/aiq_rl_store/models/sft_rc_ckpt57_20261003")  # reference-deployment default
BASE = os.environ.get("GATE_BASE_MODEL", "/data/home/guoshaoyang/models/Qwen3.5-0.8B-ms")  # reference-deployment default
W = os.environ.get("GATE_WORKDIR", "/data/home/guoshaoyang/aiq_rl/work_g32think_20261003")  # reference-deployment default
QFILE = W + "/data/probe160.jsonl"
TRAIN = os.environ.get("GATE_TRAIN_JSONL", "/data/shared/guoshaoyang/aiq_rl_store/data/rl_pool_b_v0_think/train_think.jsonl")  # reference-deployment default
GROUPS = ["q_3d708d", "q_a10140", "q_88802d", "q_8aba00", "q_17f213", "q_e3db23", "q_fd2da1", "q_995163"]
GRAD_FULL, GRAD_SUB = "q_3d708d", "q_8aba00"
BUDGET = 24576
EPS, BETA = 3e-3, 0.05


def log(*a):
    print(*a, flush=True)


def tokenizer():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(CK)
    except Exception:
        return AutoTokenizer.from_pretrained(BASE)


def load_groups(tok, qids):
    Q = {}
    for line in open(QFILE):
        r = json.loads(line)
        Q[r["question_id"]] = r
    raw = {}
    for f in sorted(glob.glob(W + "/probe57/raw_shard*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r["question_id"] in qids:
                raw[r["question_id"]] = r
    out = {}
    for q in qids:
        pi = tok.apply_chat_template(Q[q]["prompt"], tokenize=True, return_dict=True,
                                     add_generation_prompt=True, enable_thinking=True)["input_ids"]
        comps = [tok(s["text"], add_special_tokens=False)["input_ids"] for s in raw[q]["samples"]]
        comps = [c for c in comps if len(c) >= 2]
        out[q] = (list(pi), comps)
    return out


def load_models(with_ref=True):
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
    if with_ref:  # GSPOAsyncGRPOTrainer.__init__: deepcopy(unwrapped policy).to(bf16).eval() BEFORE prepare/ckpt
        ref = copy.deepcopy(m).to(torch.bfloat16).eval()
        ref.requires_grad_(False)
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.train()
    ps.install(m, temperature=1.0)
    if ref is not None:
        ps.install(ref, temperature=1.0)
    log("attn_impl", m.config._attn_implementation, "| params trainable",
        sum(p.numel() for p in m.parameters() if p.requires_grad))
    return m, ref


def pack(prompt, comps):
    ids, pos, cm, bounds = [], [], [], []
    for c in comps:
        a = len(ids)
        seq = prompt + c
        ids += seq
        pos += list(range(len(seq)))
        cm += [0] * len(prompt) + [1] * len(c)
        bounds.append((a, len(ids)))
    t = lambda x: torch.tensor([x], device="cuda")  # noqa: E731
    return t(ids), t(pos), t(cm), bounds


def ac(on):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=on)


def fwd_clean(model, prompt, c, policy):
    """Production numerics, batch 1, no padding: model(...) on one sequence. Returns completion lp [len(c)]."""
    ids, pos, cm, _ = pack(prompt, [c])
    with ac(policy):
        out = model(input_ids=ids, position_ids=pos, labels=ids, completion_mask=cm, use_cache=False)
    P = len(prompt)
    return out["log_probs"][0, P - 1:].float()


def rows_by_budget(lens, budget=BUDGET):
    """Greedy in-order packing of whole sequences into rows of <= budget tokens (v2 TokenBudgetBatcher row)."""
    rows, cur, n = [], [], 0
    for k, L in enumerate(lens):
        if cur and n + L > budget:
            rows.append(cur)
            cur, n = [], 0
        cur.append(k)
        n += L
    if cur:
        rows.append(cur)
    return rows


def fwd_prod_rows(model, prompt, comps, policy, rows):
    """Production padded per_seq_logprobs over v2-style rows. Returns {k: completion lp}."""
    from rlforge.trainer import per_seq_logprobs
    out = {}
    P = len(prompt)
    for r in rows:
        ids, pos, cm, bounds = pack(prompt, [comps[k] for k in r])
        with ac(policy):
            lp, _, _ = per_seq_logprobs(model, ids, pos, cm)
        for k, (a, e) in zip(r, bounds):
            out[k] = lp[0, a + P - 1: e - 1].float()
    return out


def fwd_shared(model, prompt, comps, policy):
    ids, pos, cm, bounds = pack(prompt, comps)
    with ac(policy):
        o = model(prefix_share=dict(input_ids=ids, position_ids=pos, completion_mask=cm))
    P = len(prompt)
    return [o["log_probs"][0, a + P - 1: e - 1].float() for a, e in bounds], o["prefix_share_stats"]


def proc_order(comps):
    """Order in which the shared path lays the branches out: buckets of _buckets(lens), rows in order."""
    import rlforge.prefix_share as ps
    lens = [len(c) + 1 for c in comps]
    order = [k for b in ps._buckets(lens) for k in b]
    pos = [0] * len(comps)
    for i, k in enumerate(order):
        pos[k] = i + 1
    return pos, order


def seqstats(a_list, b_list):
    s, tok = [], []
    for a, b in zip(a_list, b_list):
        d = a - b
        s.append(d.mean().item())
        tok.append(d.abs())
    t = torch.cat(tok)
    sa = torch.tensor(s).abs()
    return {"n_seq": len(s), "seq_lr_mean": float(torch.tensor(s).mean()), "seq_lr_absmean": float(sa.mean()),
            "seq_lr_p99": float(torch.quantile(sa, 0.99)), "seq_lr_max": float(sa.max()),
            "tok_absmean": float(t.mean()), "tok_p99": float(torch.quantile(t[:2_000_000].float(), 0.99)),
            "tok_max": float(t.max()), "n_tok": int(t.numel())}, s


def spearman(x, y):
    import numpy as np
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def slope(x, y):
    import numpy as np
    return float(np.polyfit(np.array(x, float), np.array(y, float), 1)[0])


def position_table(per_group_s, per_group_pos, G=32):
    buckets = {i: [] for i in range(1, G + 1)}
    for s, pos in zip(per_group_s, per_group_pos):
        for v, p in zip(s, pos):
            if p <= G:
                buckets[p].append(v)
    xs, ms = [], []
    for p in range(1, G + 1):
        if buckets[p]:
            xs.append(p)
            ms.append(sum(buckets[p]) / len(buckets[p]))
    return {"n_per_bucket": min(len(v) for v in buckets.values() if v), "max_abs_bucket_mean": max(abs(m) for m in ms),
            "spearman": spearman(xs, ms), "slope_per_pos": slope(xs, ms), "drift_1_to_G": slope(xs, ms) * (len(xs) - 1),
            "bucket_means": [round(m, 6) for m in ms]}


# ------------------------------------------------------------------ loss / grads
def seq_terms(lp, ref_lp, old_lp, A, N):
    """GSPOAsyncGRPOTrainer seq_mean loss term of ONE sequence (completion tokens only), / N sequences."""
    s = (lp - old_lp).mean()
    rho = torch.exp(s)
    rho_c = torch.clamp(rho, 1 - EPS, 1 + EPS)
    pg = -torch.min(rho * A, rho_c * A)
    d = ref_lp.detach() - lp
    kl = (torch.exp(d) - d - 1.0).mean()
    return (pg + BETA * kl) / N


def flat_grad(model):
    gs = []
    for _, p in model.named_parameters():
        if p.requires_grad:
            gs.append(torch.zeros(p.numel(), device="cuda") if p.grad is None else p.grad.detach().float().flatten())
    return torch.cat(gs)


def missing_grads(model):
    return [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]


def gcmp(a, b):
    return {"cos": float(torch.nn.functional.cosine_similarity(a, b, dim=0)), "norm_ratio": float(a.norm() / b.norm()),
            "rel_l2": float((a - b).norm() / b.norm())}


def grad_arms(pol, ref, prompt, comps, adv, arms):
    import rlforge.prefix_share as ps
    N = len(comps)
    res, lps = {}, {}
    # clean: batch-1 FA3, per-sequence backward (loss is a sum of per-sequence terms)
    pol.zero_grad(set_to_none=True)
    old, ref_clean = [], []
    t0 = time.perf_counter()
    for k, c in enumerate(comps):
        with torch.no_grad():
            rl = fwd_clean(ref, prompt, c, policy=False)
        lp = fwd_clean(pol, prompt, c, policy=True)
        old.append(lp.detach())
        ref_clean.append(rl)
        seq_terms(lp, rl, lp.detach(), adv[k], N).backward()
    torch.cuda.synchronize()
    res["clean"] = flat_grad(pol)
    log(f"  clean done {time.perf_counter() - t0:.1f}s missing={missing_grads(pol)[:3]}")
    if "prod" in arms:
        pol.zero_grad(set_to_none=True)
        rows = rows_by_budget([len(prompt) + len(c) for c in comps])
        for r in rows:
            sub = [comps[k] for k in r]
            with torch.no_grad():
                rl = fwd_prod_rows(ref, prompt, sub, False, [list(range(len(r)))])
            lp = fwd_prod_rows(pol, prompt, sub, True, [list(range(len(r)))])
            loss = sum(seq_terms(lp[i], rl[i], old[k], adv[k], N) for i, k in enumerate(r))
            loss.backward()
        res["prod"] = flat_grad(pol)
    for name, det in (("shared", False), ("shared_detach", True)):
        if name not in arms:
            continue
        ps._DETACH = det
        pol.zero_grad(set_to_none=True)
        with torch.no_grad():
            rl, _ = fwd_shared(ref, prompt, comps, policy=False)
        lp, _ = fwd_shared(pol, prompt, comps, policy=True)
        loss = sum(seq_terms(lp[k], rl[k], old[k], adv[k], N) for k in range(N))
        loss.backward()  # exactly ONE backward for the whole row
        res[name] = flat_grad(pol)
        log(f"  {name} done missing={missing_grads(pol)[:3]}")
        lps[name] = lp
    ps._DETACH = False
    pol.zero_grad(set_to_none=True)
    out = {k: gcmp(v, res["clean"]) for k, v in res.items() if k != "clean"}
    out["clean_norm"] = float(res["clean"].norm())
    if "prod" in res and "shared" in res:
        out["shared_vs_prod"] = gcmp(res["shared"], res["prod"])
    del res
    torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------ modes
def mode_numerics(a):
    tok = tokenizer()
    groups = load_groups(tok, GROUPS)
    pol, ref = load_models()
    R = {"code_md5": a.md5}
    allS = {"policy": [[], [], []], "ref": [[], [], []]}  # shared vs clean / prod vs clean / shared vs prod
    pos_shared, pos_row, s_pol_groups, s_ref_groups, s_floor_groups = [], [], [], [], []
    grp_info = {}
    for q in GROUPS:
        prompt, comps = groups[q]
        lens = [len(prompt) + len(c) for c in comps]
        rows = rows_by_budget(lens)
        t0 = time.perf_counter()
        with torch.no_grad():
            per = {}
            for tag, model, policy in (("policy", pol, True), ("ref", ref, False)):
                clean = [fwd_clean(model, prompt, c, policy) for c in comps]
                prod = fwd_prod_rows(model, prompt, comps, policy, rows)
                prod = [prod[k] for k in range(len(comps))]
                shared, st = fwd_shared(model, prompt, comps, policy)
                per[tag] = (clean, prod, shared, st)
        torch.cuda.synchronize()
        g = {"P": len(prompt), "G": len(comps), "comp_mean": round(sum(map(len, comps)) / len(comps)),
             "comp_max": max(map(len, comps)), "comp_min": min(map(len, comps)),
             "fwd_tok_frac": per["policy"][3]["forward_tokens"] / per["policy"][3]["unshared_tokens"],
             "secs": round(time.perf_counter() - t0, 1)}
        for tag in ("policy", "ref"):
            clean, prod, shared, _ = per[tag]
            st_sc, s_sc = seqstats(shared, clean)
            st_pc, s_pc = seqstats(prod, clean)
            st_sp, _ = seqstats(shared, prod)
            g[tag] = {"shared_vs_clean": st_sc, "prod_vs_clean(floor)": st_pc, "shared_vs_prod": st_sp}
            for i, (x, y) in enumerate(((shared, clean), (prod, clean), (shared, prod))):
                allS[tag][i].append((x, y))
            if tag == "policy":
                s_pol_groups.append(s_sc)
                s_floor_groups.append(s_pc)
            else:
                s_ref_groups.append(s_sc)
        pos_shared.append(proc_order(comps)[0])
        pos_row.append(list(range(1, len(comps) + 1)))
        grp_info[q] = g
        log(q, json.dumps({k: g[k] for k in ("P", "G", "comp_mean", "comp_max", "fwd_tok_frac", "secs")}),
            "| pol shared-clean", json.dumps(g["policy"]["shared_vs_clean"]),
            "| floor", json.dumps(g["policy"]["prod_vs_clean(floor)"]))
    R["groups"] = grp_info
    agg = {}
    for tag in ("policy", "ref"):
        agg[tag] = {}
        for i, name in enumerate(("shared_vs_clean", "prod_vs_clean(floor)", "shared_vs_prod")):
            xs = [x for pair in allS[tag][i] for x in pair[0]]
            ys = [y for pair in allS[tag][i] for y in pair[1]]
            agg[tag][name] = seqstats(xs, ys)[0]
    R["aggregate"] = agg
    log("AGG", json.dumps(agg))
    # (5) per-position bias
    R["position"] = {
        "policy_shared_order": position_table(s_pol_groups, pos_shared),
        "policy_row_order": position_table(s_pol_groups, pos_row),
        "ref_shared_order": position_table(s_ref_groups, pos_shared),
        "ref_row_order": position_table(s_ref_groups, pos_row),
        "floor_row_order(prod vs clean)": position_table(s_floor_groups, pos_row),
    }
    log("POS", json.dumps(R["position"]))
    # (6) leakage: pick k, scramble the completion processed just before it, re-run shared
    rng = random.Random(0)
    leak = []
    for q in (GRAD_SUB, GRAD_FULL, "q_e3db23"):
        prompt, comps = groups[q]
        _, order = proc_order(comps)
        for trial in range(3):
            i = rng.randrange(1, len(order))
            k, j = order[i], order[i - 1]
            for variant in ("same_len_random", "half_len_random"):
                c2 = [list(c) for c in comps]
                L = len(comps[j]) if variant == "same_len_random" else max(2, len(comps[j]) // 2)
                c2[j] = [rng.randrange(0, 150000) for _ in range(L)]
                with torch.no_grad():
                    base, _ = fwd_shared(pol, prompt, comps, True)
                    pert, _ = fwd_shared(pol, prompt, c2, True)
                dk = (pert[k] - base[k]).abs()
                others = [(pert[m] - base[m]).abs().max().item() for m in range(len(comps)) if m != j]
                leak.append({"q": q, "k": k, "j": j, "variant": variant, "k_len": len(comps[k]), "j_len": len(comps[j]),
                             "k_tok_absmax": dk.max().item(), "k_seq_lr_shift": (pert[k] - base[k]).mean().item(),
                             "k_bitexact": bool(torch.equal(pert[k], base[k])), "others_absmax": max(others)})
                log("LEAK", json.dumps(leak[-1]))
    R["leakage"] = leak
    # (3)/(4) gradient checks
    g = torch.Generator().manual_seed(1)
    for name, q, sel in (("G8_subset", GRAD_SUB, 8), ("G32_full_row", GRAD_FULL, 32)):
        prompt, comps = groups[q]
        if sel < len(comps):
            idx = sorted(range(len(comps)), key=lambda k: len(comps[k]))
            pick = [idx[round(t * (len(idx) - 1) / (sel - 1))] for t in range(sel)]
            comps = [comps[k] for k in pick]
        r = torch.randn(len(comps), generator=g)
        adv = ((r - r.mean()) / r.std()).tolist()
        log(f"GRAD {name}: G={len(comps)} P={len(prompt)} lens={sorted(map(len, comps))}")
        R["grad_" + name] = grad_arms(pol, ref, prompt, comps, adv, ("prod", "shared", "shared_detach"))
        log("GRAD", name, json.dumps(R["grad_" + name]))
    R["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
    json.dump(R, open(a.out, "w"), indent=1)
    log("WROTE", a.out)


def mode_floor(a):
    """Kernel floor: the SAME unshared batch-1 forward with attention switched FA3 -> torch SDPA.
    Separates 'any kernel change' noise from anything the sharing itself adds."""
    tok = tokenizer()
    groups = load_groups(tok, GROUPS)
    pol, ref = load_models()
    R = {"code_md5": a.md5}
    acc = {t: {"sdpa_vs_fa3": [], "shared_vs_sdpa": [], "shared_vs_fa3": []} for t in ("policy", "ref")}
    s_floor, s_sh, pos_row = [], [], []
    for q in GROUPS:
        prompt, comps = groups[q]
        res = {}
        with torch.no_grad():
            for tag, model, policy in (("policy", pol, True), ("ref", ref, False)):
                model.set_attn_implementation("kernels-community/flash-attn3")
                fa3 = [fwd_clean(model, prompt, c, policy) for c in comps]
                shared, _ = fwd_shared(model, prompt, comps, policy)
                model.set_attn_implementation("sdpa")
                sd = [fwd_clean(model, prompt, c, policy) for c in comps]
                model.set_attn_implementation("kernels-community/flash-attn3")
                acc[tag]["sdpa_vs_fa3"].append((sd, fa3))
                acc[tag]["shared_vs_sdpa"].append((shared, sd))
                acc[tag]["shared_vs_fa3"].append((shared, fa3))
                res[tag] = {"sdpa_vs_fa3": seqstats(sd, fa3)[0], "shared_vs_sdpa": seqstats(shared, sd)[0]}
                if tag == "policy":
                    s_floor.append(seqstats(sd, fa3)[1])
                    s_sh.append(seqstats(shared, sd)[1])
        pos_row.append(list(range(1, len(comps) + 1)))
        log(q, json.dumps(res))
    agg = {}
    for tag in acc:
        agg[tag] = {}
        for name, pairs in acc[tag].items():
            agg[tag][name] = seqstats([x for p in pairs for x in p[0]], [y for p in pairs for y in p[1]])[0]
    R["aggregate"] = agg
    R["position_floor_sdpa_vs_fa3_row_order"] = position_table(s_floor, pos_row)
    R["position_shared_vs_sdpa_row_order"] = position_table(s_sh, pos_row)
    log("FLOOR_AGG", json.dumps(agg))
    log("FLOOR_POS", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "bucket_means"}
                                 for k, v in R.items() if k.startswith("position")}))
    # gradient floor: clean FA3 vs clean SDPA (same loss/adv/old as numerics mode)
    g = torch.Generator().manual_seed(1)
    for name, q, sel in (("G8_subset", GRAD_SUB, 8), ("G32_full_row", GRAD_FULL, 32)):
        prompt, comps = groups[q]
        if sel < len(comps):
            idx = sorted(range(len(comps)), key=lambda k: len(comps[k]))
            pick = [idx[round(t * (len(idx) - 1) / (sel - 1))] for t in range(sel)]
            comps = [comps[k] for k in pick]
        r = torch.randn(len(comps), generator=g)
        adv = ((r - r.mean()) / r.std()).tolist()
        N = len(comps)
        grads, old = {}, None
        for impl in ("kernels-community/flash-attn3", "sdpa"):
            pol.set_attn_implementation(impl)
            ref.set_attn_implementation(impl)
            pol.zero_grad(set_to_none=True)
            lps = []
            for k, c in enumerate(comps):
                with torch.no_grad():
                    rl = fwd_clean(ref, prompt, c, policy=False)
                lp = fwd_clean(pol, prompt, c, policy=True)
                lps.append(lp.detach())
                o = old[k] if old is not None else lp.detach()
                seq_terms(lp, rl, o, adv[k], N).backward()
            if old is None:
                old = lps
            grads[impl] = flat_grad(pol)
        pol.set_attn_implementation("kernels-community/flash-attn3")
        ref.set_attn_implementation("kernels-community/flash-attn3")
        R["grad_floor_" + name] = gcmp(grads["sdpa"], grads["kernels-community/flash-attn3"])
        log("GRAD_FLOOR", name, json.dumps(R["grad_floor_" + name]))
        del grads
        pol.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    json.dump(R, open(a.out, "w"), indent=1)
    log("WROTE", a.out)


def mode_speed(a):
    tok = tokenizer()
    # ~5.5k-token real train prompt + 32 real ckpt57 completions (mean ~2.2k) of a probe question
    best = None
    for i, line in enumerate(open(TRAIN)):
        if i > 600:
            break
        r = json.loads(line)
        n = len(tok.apply_chat_template(r["prompt"], tokenize=True, return_dict=True, add_generation_prompt=True,
                                        enable_thinking=True)["input_ids"])
        if best is None or abs(n - 5500) < abs(best[0] - 5500):
            best = (n, r["prompt"])
    prompt = tok.apply_chat_template(best[1], tokenize=True, return_dict=True, add_generation_prompt=True,
                                     enable_thinking=True)["input_ids"]
    _, comps = load_groups(tok, ["q_865c10"])["q_865c10"]
    comps = comps[:32]
    pol, ref = load_models()
    P, C = len(prompt), sum(map(len, comps)) / len(comps)
    log(f"SPEED group: P={P} G={len(comps)} comp_mean={C:.0f} max={max(map(len, comps))}")
    adv = torch.randn(len(comps)).tolist()
    N = len(comps)

    def prod_group():
        rows = rows_by_budget([P + len(c) for c in comps])
        for r in rows:
            sub = [comps[k] for k in r]
            with torch.no_grad():
                rl = fwd_prod_rows(ref, prompt, sub, False, [list(range(len(r)))])
            lp = fwd_prod_rows(pol, prompt, sub, True, [list(range(len(r)))])
            sum(seq_terms(lp[i], rl[i], lp[i].detach(), adv[k], N) for i, k in enumerate(r)).backward()
        return len(rows), sum(P + len(c) for c in comps)

    def shared_group():
        with torch.no_grad():
            rl, _ = fwd_shared(ref, prompt, comps, False)
        lp, st = fwd_shared(pol, prompt, comps, True)
        sum(seq_terms(lp[k], rl[k], lp[k].detach(), adv[k], N) for k in range(N)).backward()
        return 1, st["forward_tokens"]

    res = {}
    for name, fn in (("prod_perseq_rows24576", prod_group), ("shared_whole_group", shared_group)):
        times = []
        for it in range(3):
            pol.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            nrows, fwd_tok = fn()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        res[name] = {"rows": nrows, "s_per_group_best": round(min(times[1:]), 3), "s_all": [round(t, 2) for t in times],
                     "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1), "forwarded_tokens": fwd_tok}
        log("SPEED", name, json.dumps(res[name]))
    res["speedup"] = res["prod_perseq_rows24576"]["s_per_group_best"] / res["shared_whole_group"]["s_per_group_best"]
    res["forwarded_token_ratio"] = res["shared_whole_group"]["forwarded_tokens"] / res["prod_perseq_rows24576"]["forwarded_tokens"]
    res["P"], res["comp_mean"] = P, C
    log("SPEED_SUMMARY", json.dumps(res))
    json.dump(res, open(a.out, "w"), indent=1)


def mode_ddp(a):
    """Real DDP (accelerate, bf16 autocast) with the shared forward: 2 ranks, unequal rows, gas=2."""
    from accelerate import Accelerator
    acc = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=2)
    tok = tokenizer()
    groups = load_groups(tok, ["q_8aba00", "q_6d1022", "q_865c10"])
    pol, ref = load_models()
    opt = torch.optim.SGD([p for p in pol.parameters() if p.requires_grad], lr=0.0)
    pm, opt = acc.prepare(pol, opt)
    rk = acc.process_index
    # rank 0: one whole group of 32 (short completions trimmed for speed); rank 1: two partial groups 17 + 9
    def trim(cs, n, L=600):
        return [c[:L] for c in cs[:n]]
    if rk == 0:
        rows = [[("q_8aba00", trim(groups["q_8aba00"][1], 32))], [("q_6d1022", trim(groups["q_6d1022"][1], 20))]]
    else:
        rows = [[("q_6d1022", trim(groups["q_6d1022"][1], 17)), ("q_865c10", trim(groups["q_865c10"][1], 9))],
                [("q_865c10", trim(groups["q_865c10"][1], 31, 900))]]
    n_micro = 0
    for step in range(2):
        for row in rows:
            ids, pos, cm = [], [], []
            for q, cs in row:
                pr = groups[q][0]
                for c in cs:
                    seq = pr + c
                    ids += seq
                    pos += list(range(len(seq)))
                    cm += [0] * len(pr) + [1] * len(c)
            t = lambda x: torch.tensor([x], device=acc.device)  # noqa: E731
            ids, pos, cm = t(ids), t(pos), t(cm)
            with acc.accumulate(pm):
                out = pm(prefix_share=dict(input_ids=ids, position_ids=pos, completion_mask=cm))
                with torch.no_grad():
                    rl = ref(prefix_share=dict(input_ids=ids, position_ids=pos, completion_mask=cm))["log_probs"]
                v = cm[:, 1:].float()
                lp = out["log_probs"]
                d = rl - lp
                loss = (-(lp * v).sum() / v.sum()) + 0.05 * ((torch.exp(d) - d - 1) * v).sum() / v.sum()
                acc.backward(loss)
                n_micro += 1
                if acc.sync_gradients:
                    g = torch.cat([p.grad.detach().float().flatten() for p in pol.parameters() if p.requires_grad])
                    chk = torch.stack([g.norm(), g.sum(), g[:: 9973].sum()])
                    allc = acc.gather(chk[None])
                    if acc.is_main_process:
                        log(f"DDP step {step}: grad checks per rank {allc.tolist()} "
                            f"equal={bool(torch.allclose(allc[0], allc[1], rtol=0, atol=0))}")
                    opt.step()
                    opt.zero_grad()
            log(f"DDP rank {rk} step {step} micro {n_micro} out_dtype {lp.dtype} stats {out['prefix_share_stats']}")
    cnt = acc.gather(torch.tensor([n_micro], device=acc.device))
    if acc.is_main_process:
        log(f"DDP_OK micro-steps per rank {cnt.tolist()} (no hang, no unused-param error)")


def mode_batcher(a):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")
    from collections import defaultdict
    import rlforge.prefix_share as ps
    rng = random.Random(0)

    def stream(n_groups, G=32, drop=0.05, long_frac=0.1, P=(3500, 6000)):
        gid = 0
        while gid < n_groups:
            p = rng.randrange(*P)
            for _ in range(G):
                if rng.random() < drop:
                    continue
                c = 16384 if rng.random() < long_frac else rng.randrange(800, 6000)
                yield {"input_ids": [0] * (p + c), "completion_mask": [0] * p + [1] * c, "group_id": gid}
            gid += 1

    out = {}
    for R in (4, 5):
        for long_frac in (0.1, 0.5, 1.0):
            b = ps.GroupRowBatcher(list(stream(400, long_frac=long_frac)), R, 32 * R)
            n_mb, empty, sizes, maxcost, multi = 0, 0, set(), 0, 0
            for rows in b:
                n_mb += 1
                empty += sum(1 for r in rows if not r)
                sizes.add(sum(map(len, rows)))
                for r in rows:
                    gids = [s["group_id"] for s in r]
                    multi += len(set(gids)) > 1
                    cost = 0
                    for g_ in set(gids):
                        ss = [s for s in r if s["group_id"] == g_]
                        p = ps._prompt_len(ss[0])
                        cost += p + sum(len(s["input_ids"]) - p for s in ss)
                    maxcost = max(maxcost, cost)
            out[f"GroupRowBatcher_R{R}_long{long_frac}"] = {
                "microbatches": n_mb, "empty_rows": empty, "samples_per_mb": sorted(sizes),
                "max_row_shared_tokens": maxcost, "rows_with_2+_groups": multi}
        m = defaultdict(list)
        b = ps.GroupTokenBudgetBatcher(list(stream(200)), R, 24576, m)
        n_mb, empty, sizes = 0, 0, []
        for rows in b:
            n_mb += 1
            empty += sum(1 for r in rows if not r)
            sizes.append(sum(map(len, rows)))
        out[f"GroupTokenBudgetBatcher_R{R}"] = {"microbatches": n_mb, "empty_rows": empty,
                                                 "samples_per_mb_min_max": [min(sizes), max(sizes)]}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["numerics", "floor", "speed", "ddp", "batcher"])
    ap.add_argument("--out", default="/tmp/gate_out.json")
    ap.add_argument("--md5", default="")
    a = ap.parse_args()
    {"numerics": mode_numerics, "floor": mode_floor, "speed": mode_speed, "ddp": mode_ddp, "batcher": mode_batcher}[a.mode](a)
