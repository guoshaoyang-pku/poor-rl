"""Which v3_2 component moves per-seq log-probs? no-grad forwards on one real-profile group."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("RLFORGE_FUSED_OPS", "all")
import v32_gate as G
import rlforge.prefix_share as ps
import rlforge.fused_ops as fo
from transformers import AutoTokenizer
kind = sys.argv[1] if len(sys.argv) > 1 else "typical"
tok = AutoTokenizer.from_pretrained(G.CK)
rows = G.build_rows(tok, [kind])
model, _ = G.load_models(with_ref=False)
seqs = rows[kind]
inp = G.make_inputs(seqs, seed=1)
ids, pos, cm = inp["input_ids"], inp["position_ids"], inp["completion_mask"]
lens = [len(c) for _, c in seqs]

def v31(fast, conv, norm):
    os.environ["RLFORGE_FAST_LOGPROB"] = "1" if fast else "0"
    fo.set_enabled(False)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    if conv: mq.causal_conv1d_fn = fo._PATCHED["conv"]
    if norm:
        mq.Qwen3_5RMSNorm.forward = fo._PATCHED["rms"]; mq.Qwen3_5RMSNormGated.forward = fo._PATCHED["rmsg"]
    with torch.no_grad():
        return model(prefix_share=dict(input_ids=ids, position_ids=pos, completion_mask=cm))["log_probs"].float()

def sb(budget, fast=False):
    os.environ["RLFORGE_FAST_LOGPROB"] = "1" if fast else "0"
    fo.set_enabled(False)
    return G.subbatch_lp.__wrapped__(model, inp, budget) if hasattr(G.subbatch_lp, "__wrapped__") else _sb(budget)

def _sb(budget):
    out = torch.zeros(1, ids.shape[1] - 1, device="cuda")
    with torch.no_grad():
        for s in ps.plan_subbatches(ids, pos, cm, budget):
            o = model(prefix_share=dict(input_ids=ids, completion_mask=cm, subbatch=s, ckpt_layers=set()))
            out[0, o["slots"]] = o["log_probs"].float()
    return out

base = v31(False, False, False)
base2 = v31(False, False, False)
def rep(name, lp):
    d = G.seq_diff(lp, base, inp)
    pos0 = inp["position_ids"][0]; c = inp["completion_mask"][0, 1:].bool()
    seq = (pos0 == 0).cumsum(0)[1:] - 1
    dd = (lp[0] - base[0]) * c
    s = torch.zeros(len(seqs), device="cuda").index_add(0, seq, dd) / torch.zeros(len(seqs), device="cuda").index_add(0, seq, c.float()).clamp(min=1)
    k = int(s.abs().argmax())
    print(f"{name:28s} seq|mean| {d['seq_mean']:+.2e} p99 {d['seq_p99']:.2e} max {d['seq_max']:.2e} (seq {k}, len {lens[k]}) tok_absmean {d['tok_absmean']:.2e} tok_max {d['tok_max']:.2e}", flush=True)
rep("rerun (determinism)", base2)
rep("fast_logprob", v31(True, False, False))
rep("fused conv", v31(False, True, False))
rep("fused norm", v31(False, False, True))
rep("fast+conv+norm", v31(True, True, True))
os.environ["RLFORGE_FAST_LOGPROB"] = "0"; fo.set_enabled(False)
rep("subbatch 1<<30 (1 sb, DP buckets)", _sb(1 << 30))
rep("subbatch 65536", _sb(65536))
rep("subbatch 32768", _sb(32768))
