"""Trainer-side check: TRL's NCCL weight sync into a (possibly cross-node) DP vLLM server.

Uses TRL's own pieces (create_model_from_path, _iter_vllm_named_params, VLLMClient,
WeightTransferClient), so the handshake, world size and wire format are exactly the
trainer's. Steps:
  1. greedy probe on every DP replica (X-data-parallel-rank header) -> baseline
  2. N timed syncs of the unchanged weights (pause / send / resume), as the trainer does
  3. sync a perturbed copy (noise on every weight) -> every replica's output must change
  4. sync the original back -> every replica must return to the baseline tokens exactly
"""
import argparse, json, time

import requests
import torch
from accelerate import PartialState
from trl.experimental.async_grpo.async_grpo_trainer import _iter_vllm_named_params
from trl.experimental.async_grpo.vllm_client import VLLMClient
from trl.experimental.async_grpo.weight_transfer import WeightTransferClient
from trl.trainer.utils import create_model_from_path

PROBE = [{"role": "user", "content": "List the first ten prime numbers and explain why 1 is not prime."}]


def probe(url, dp, tok):
    ids = tok.apply_chat_template(PROBE, tokenize=True, add_generation_prompt=True)
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    out = []
    for r in range(dp):
        resp = requests.post(f"{url}/v1/completions", headers={"X-data-parallel-rank": str(r)},
                             json={"model": "t", "prompt": list(ids), "max_tokens": 48, "temperature": 0.0,
                                   "return_token_ids": True, "logprobs": 0}, timeout=120)
        resp.raise_for_status()
        c = resp.json()["choices"][0]
        out.append((c["token_ids"], c["logprobs"]["token_logprobs"]))
    return out


def lp_gap(outs):
    """Max |logprob| gap between replica 0 and each other replica over the common greedy prefix."""
    ref_t, ref_l = outs[0]
    gaps = []
    for t, l in outs[1:]:
        n = next((i for i, (x, y) in enumerate(zip(ref_t, t)) if x != y), min(len(ref_t), len(t)))
        gaps.append({"common_prefix": n, "first_tok_gap": abs(ref_l[0] - l[0]),
                     "max_gap": max((abs(x - y) for x, y in zip(ref_l[:n], l[:n])), default=0.0)})
    return gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--noise", type=float, default=0.3, help="perturbation, in units of each tensor's std")
    a = ap.parse_args()
    PartialState()  # TRL's logger refuses to log before accelerate state exists
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)

    ws_dp = requests.get(f"{a.url}/get_world_size", params={"include_dp": "true"}).json()["world_size"]
    ws = requests.get(f"{a.url}/get_world_size", params={"include_dp": "false"}).json()["world_size"]
    dp = ws_dp // ws
    print("WORLD", json.dumps({"world_size_across_dp": ws_dp, "per_replica": ws, "dp": dp}), flush=True)

    model = create_model_from_path(a.model, device_map=None, dtype=torch.bfloat16,
                                   attn_implementation="sdpa").cuda()
    params = list(_iter_vllm_named_params(model))
    nbytes = sum(p.numel() * p.element_size() for _, p in params)
    info = {"names": [n for n, _ in params],
            "dtype_names": [str(p.dtype).split(".")[-1] for _, p in params],
            "shapes": [list(p.shape) for _, p in params]}
    print("PARAMS", json.dumps({"n": len(params), "GB": round(nbytes / 1e9, 3)}), flush=True)

    client = VLLMClient(a.url, 240)
    wt = WeightTransferClient(client, info, weight_sync_timeout=600)
    t0 = time.time()
    wt.init_weight_transfer()
    print("INIT", json.dumps({"init_s": round(time.time() - t0, 2)}), flush=True)

    base = probe(a.url, dp, tok)
    print("BASELINE replicas_agree", all(b[0] == base[0][0] for b in base), json.dumps(lp_gap(base)), flush=True)

    def sync(tensors):
        t = time.time()
        wt.pause()
        tp = time.time()
        wt.send_weights(iter(tensors))
        ts = time.time()
        wt.resume()
        return {"pause_s": round(tp - t, 3), "send_s": round(ts - tp, 3), "total_s": round(time.time() - t, 3)}

    orig = [(n, p.detach()) for n, p in params]
    times = [sync(orig) for _ in range(a.reps)]
    for t in times:
        print("SYNC", json.dumps(t), flush=True)
    send = sorted(t["send_s"] for t in times)
    print("SYNC_MEDIAN", json.dumps({"send_s": send[len(send) // 2],
                                     "GB_per_s": round(nbytes / 1e9 / send[len(send) // 2], 2)}), flush=True)

    torch.manual_seed(0)
    noisy = [(n, (p.float() + a.noise * p.float().std().nan_to_num(0) * torch.randn_like(p.float())).to(p.dtype))
             for n, p in orig]
    sync(noisy)
    pert = probe(a.url, dp, tok)
    sync(orig)
    back = probe(a.url, dp, tok)
    print("RESULT", json.dumps({
        "dp": dp,
        "perturbed_changed_per_replica": [p[0] != b[0] for p, b in zip(pert, base)],
        "perturbed_replicas_agree": all(p[0] == pert[0][0] for p in pert),
        "perturbed_lp_gap_vs_replica0": lp_gap(pert),
        "restored_exact_per_replica": [r == b for r, b in zip(back, base)],
    }), flush=True)
    wt.destroy()


main()
