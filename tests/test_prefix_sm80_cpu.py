"""CPU gradient-identity test for rlforge.prefix_share_sm80 on a tiny random Qwen3.8 (qwen3_5 text), fp32.

perseq (every sequence alone, standard HF forward) vs shared prefix unchunked vs chunked branches (k=2), with the
per-layer checkpoint on and off. Compares per-branch logp and every LoRA gradient.
Run: TINY=<dir from tools/e2e/make_tiny.py> python tests/test_prefix_sm80_cpu.py
"""
import json
import os
import sys

for _m in ("fla", "causal_conv1d", "flash_attn"):
    sys.modules[_m] = None
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

import rlforge.prefix_share_sm80 as ps  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
DT = getattr(torch, os.environ.get("DTYPE", "float32"))
TOL = 1e-4 if DT == torch.float32 else 1e-9
m = AutoModelForCausalLM.from_pretrained(os.environ["TINY"], dtype=DT, attn_implementation="sdpa")
T = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z",
     "in_proj_a", "in_proj_b", "out_proj"]
pm = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, target_modules=T, lora_dropout=0.0))
with torch.no_grad():  # non-zero B so every LoRA tensor gets a gradient and the adapter changes the forward
    for n, p in pm.named_parameters():
        if "lora_B" in n:
            p.normal_(0, 0.05)
lm = ps.install(pm)
lm.train()
params = [(n, p) for n, p in lm.named_parameters() if p.requires_grad]

g = torch.Generator().manual_seed(1)
V = lm.config.vocab_size
p_len = int(os.environ.get("PLEN", "203"))
prompt = torch.randint(0, V, (p_len,), generator=g)
lens = [17, 3, 25, 1, 9]
comps = [torch.randint(0, V, (n,), generator=g) for n in lens]
w = [0.7, -1.3, 0.4, 2.0, -0.5]


def loss_fn(i, lp):
    return (lp.float().mean() * w[i]) if os.environ.get("MEANLOSS") else (lp.float().sum() * w[i] / 10)


def run(kind, ckpt):
    lm._ps_force_ckpt = ckpt
    for _, p in params:
        p.grad = None
    if kind == "perseq":
        out = ps.perseq_group_backward(lm, prompt, comps, loss_fn)
    else:
        out = ps.prefix_group_backward(lm, prompt, comps, loss_fn, chunk=(None if kind == "shared" else 2))
    grads = {n: p.grad.detach().clone() for n, p in params if p.grad is not None}
    return out, grads


ref, gref = run("perseq", False)
res = {"P": ps.split_point(p_len), "p": p_len, "lens": lens, "n_lora": len(params), "cases": {}}
ok = True
for kind in ("shared", "chunked"):
    for ckpt in (False, True):
        out, gr = run(kind, ckpt)
        lp_diff = max(float((a - b).abs().max()) for a, b in zip(out["logps"], ref["logps"]))
        missing = [n for n in gref if n not in gr]
        rels = sorted(((float((gr[n] - gref[n]).norm() / (gref[n].norm() + 1e-30)), float(gref[n].norm()), n)
                       for n in gref if n in gr), reverse=True)
        rel = rels[0][0]
        # global relative error over all LoRA grads (robust to tensors whose grad is ~0)
        num = sum(float((gr[n] - gref[n]).double().norm() ** 2) for n in gref if n in gr) ** 0.5
        den = sum(float(gref[n].double().norm() ** 2) for n in gref) ** 0.5
        absd = max(float((gr[n] - gref[n]).abs().max()) for n in gref if n in gr)
        c = {"logp_max_abs": lp_diff, "grad_max_rel_l2": rel, "grad_global_rel_l2": num / den,
             "worst3": [(round(a, 8), round(b, 8), n.replace("base_model.model.", "")) for a, b, n in rels[:10]],
             "grad_max_abs": absd, "missing": len(missing),
             "n_grads": len(gr), "stats": {k: v for k, v in out.get("stats", {}).items() if k != "state_grad_norm"}}
        c["ok"] = lp_diff < TOL and num / den < TOL and not missing
        ok &= c["ok"]
        res["cases"][f"{kind}_ckpt{int(ckpt)}"] = c
res["n_grads_ref"] = len(gref)
res["ok"] = ok
print(json.dumps(res, indent=1))
sys.exit(0 if ok else 1)
