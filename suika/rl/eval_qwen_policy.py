"""Generative (token-decoding) policy evaluator for the Qwen arm.

Fork of evaluator_qwen.py. Same env interaction / seed parsing / warmup /
jsonl append / CUDA selection conventions, but the action is chosen by
DECODING the canonical answer string instead of argmax over the Q head.

Canonical answer (format spec v1): f"{(col+0.5)/128:.3f}" -> "0.004".."0.996".
Every one of the 128 strings tokenizes to exactly 5 single tokens:
['0'=15, '.'=13, d1, d2, d3] with digit ids 15..24 for '0'..'9'.
The chat template with add_generation_prompt=True ends in
"<think>\n\n</think>\n\n", so the answer follows immediately.

Decoding (--decode tokens):
  1) model._serialize(x) -> prompt ids/mask (same serialization path as
     learner / inference server / evaluator);
  2) bf16 autocast forward of the trunk, take the logits at the LAST prompt
     position (mask.sum(1)-1) and argmax (or sample) over token ids 15..24
     -> d1;
  3) append d1's token id and forward again -> d2; again -> d3 (3 forwards,
     slow but correct);
  4) "0.{d1}{d2}{d3}" -> col = min(127, max(0, int(float(s)*128))).

--decode qhead reproduces the old Q-head argmax behaviour; --decode both
computes both in the same forward pass and keeps two separate statistics.

NOTE: model_qwen.py is read-only here (edited in parallel elsewhere); the
trunk is called directly through a small local helper.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from env import DQNEnv
from model import build_model

# digit '0'..'9' are single tokens with consecutive ids 15..24
DIGIT0 = 15
DIGIT_IDS = list(range(DIGIT0, DIGIT0 + 10))
DOT_ID = 13


def _trunk_logits(model, ids, mask):
    """Forward the trunk and return last_hidden_state (bf16 autocast)."""
    out = model.trunk(input_ids=ids, attention_mask=mask, use_cache=False)
    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return h


def _lm_head(model):
    """Return the (tied) output embedding matrix [vocab, hid]."""
    trunk = model.trunk
    for obj in (trunk, getattr(trunk, "base_model", None),
                getattr(getattr(trunk, "base_model", None), "model", None)):
        if obj is None:
            continue
        emb = obj.get_output_embeddings() if hasattr(
            obj, "get_output_embeddings") else None
        if emb is not None:
            return emb.weight
        if hasattr(obj, "lm_head") and obj.lm_head is not None:
            return obj.lm_head.weight
    # tied embeddings: the decoder's input embedding is the output projection
    for obj in (trunk, getattr(trunk, "base_model", None),
                getattr(getattr(trunk, "base_model", None), "model", None)):
        if obj is None:
            continue
        if hasattr(obj, "embed_tokens"):
            return obj.embed_tokens.weight
    raise RuntimeError("could not locate lm_head / embed_tokens on trunk")


def _digit_logits(h, pos, head_w):
    """h: [B, L, hid], pos: [B] -> logits over the 10 digit tokens [B, 10]."""
    b = torch.arange(h.shape[0], device=h.device)
    z = h[b, pos]                                   # [B, hid]
    w = head_w[torch.tensor(DIGIT_IDS, device=head_w.device)]   # [10, hid]
    return z.float() @ w.float().t()                # [B, 10]


def _pick(logits, temperature, generator):
    """Greedy (T<=0) or softmax sampling over the 10 digit logits."""
    if temperature and temperature > 0:
        probs = torch.softmax(logits / float(temperature), dim=-1)
        return torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    return logits.argmax(dim=-1)


def decode_tokens(model, x, temperature=0.0, generator=None):
    """x: [B, T*5] -> (cols [B] int64, digits [B,3] int64, prompts [B] str)."""
    ids, mask = model._serialize(x)
    head_w = _lm_head(model)
    b = ids.shape[0]
    last = (mask.sum(dim=1) - 1).clamp(min=0)
    digits = []
    cur_ids, cur_mask = ids, mask
    pos = last
    for step in range(3):
        h = _trunk_logits(model, cur_ids, cur_mask)
        lg = _digit_logits(h, pos, head_w)
        d = _pick(lg, temperature, generator)
        digits.append(d)
        # append the chosen token; next prediction reads its own position
        cur_ids = torch.cat([cur_ids, d.unsqueeze(1)], dim=1)
        cur_mask = torch.cat(
            [cur_mask, torch.ones((b, 1), dtype=cur_mask.dtype,
                                  device=cur_mask.device)], dim=1)
        pos = torch.full((b,), cur_ids.shape[1] - 1, dtype=torch.long,
                         device=cur_ids.device)
    digits = torch.stack(digits, dim=1)             # [B, 3] index into 15..24
    # argmax index i over DIGIT_IDS corresponds to digit value i (0..9)
    dvals = digits.float()
    dec = dvals[:, 0] * 0.1 + dvals[:, 1] * 0.01 + dvals[:, 2] * 0.001
    cols = torch.clamp((dec * 128.0).floor().long(), 0, 127)
    prompts = []
    for i in range(b):
        n = int(mask[i].sum().item())
        prompts.append(model.tokenizer.decode(ids[i, max(0, n - 60):n]))
    return cols, digits, prompts


def _stats(scores, moves_l, maxf_l, payload, gs, t0):
    scores = np.asarray(scores, dtype=np.float64)
    return {
        "n": len(scores),
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "p25": float(np.percentile(scores, 25)),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "p2000": float((scores >= 2000).mean()),
        "moves_mean": float(np.mean(moves_l)),
        "maxfruit_max": int(max(maxf_l)) if maxf_l else -1,
        "env_steps": int(payload.get("env_steps", 0)),
        "grad_steps": gs,
        "eval_s": round(time.time() - t0, 1),
        "time": time.time(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--obs-dim", type=int, required=True)
    ap.add_argument("--seeds", default="0:16")
    ap.add_argument("--interval-s", type=float, default=900.0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--decode", choices=["tokens", "qhead", "pi", "both"],
                    default="tokens")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--samples", type=int, default=0)
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(args.config))
    lo, hi = args.seeds.split(":")
    seeds = list(range(int(lo), int(hi)))
    os.makedirs(args.run_dir, exist_ok=True)
    out_path = os.path.join(args.run_dir, "eval.jsonl")
    policy_path = args.ckpt or os.path.join(args.run_dir, "policy.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[eval] building model (HF ckpt load, ~1 min)...", flush=True)
    model = build_model(cfg, args.obs_dim).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    K = int(cfg["K"])
    from model_qwen import warmup
    warmup(model, batches=(16,), log=lambda s: print(s, flush=True))

    gen = None
    if args.temperature and args.temperature > 0:
        gen = torch.Generator(device=device)
        gen.manual_seed(0)

    seen_grad = -1
    while True:
        try:
            os.path.getmtime(policy_path)
        except OSError:
            time.sleep(10)
            continue
        payload = None
        try:
            payload = torch.load(policy_path, map_location="cpu",
                                 weights_only=False)
        except Exception:
            time.sleep(10)
            continue
        gs = int(payload.get("grad_steps", 0))
        if gs == seen_grad:
            time.sleep(args.interval_s / 3)
            continue
        seen_grad = gs
        model.load_state_dict(payload["state_dict"])

        # lockstep batched eval: one forward per move for ALL active seeds
        scores = np.full(len(seeds), np.nan)
        moves_l = np.zeros(len(seeds), dtype=np.int64)
        maxf_l = [0] * len(seeds)
        q_scores = np.full(len(seeds), np.nan) if args.decode == "both" else None
        q_moves = np.zeros(len(seeds), dtype=np.int64)
        q_maxf = [0] * len(seeds)
        envs, obs = [], []
        t0 = time.time()
        for si, s in enumerate(seeds):
            e = DQNEnv(seed=None, K=K, max_fruits=int(cfg["max_fruits"]),
                       boundary=bool(cfg.get("boundary", True)),
                       tempo=bool(cfg.get("tempo", False)),
                       obs_format=cfg.get("obs_format", "tokens"))
            obs.append(e.reset(seed=int(s)))
            envs.append(e)
        active = list(range(len(seeds)))
        n_decisions = 0
        while active:
            x = torch.from_numpy(np.stack([obs[i] for i in active])).to(device)
            with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                if args.decode == "qhead":
                    acts = model.q_values(x).float().argmax(dim=1).tolist()
                    q_acts = acts
                elif args.decode == "pi":
                    acts = model.pi_logits(x).float().argmax(dim=1).tolist()
                    q_acts = acts
                elif args.decode == "tokens":
                    cols, digits, prompts = decode_tokens(
                        model, x, args.temperature, gen)
                    acts = cols.tolist()
                    q_acts = None
                else:  # both
                    q_acts = model.q_values(x).float().argmax(dim=1).tolist()
                    cols, digits, prompts = decode_tokens(
                        model, x, args.temperature, gen)
                    acts = cols.tolist()
            if args.samples > 0 and n_decisions < args.samples:
                for j in range(len(active)):
                    if n_decisions >= args.samples:
                        break
                    d = digits[j].tolist()
                    ds = "".join(str(int(v)) for v in d)
                    dec = f"0.{ds}"
                    qcol = q_acts[j] if q_acts is not None else -1
                    qdec = f"{(qcol + 0.5) / 128:.3f}" if qcol >= 0 else "n/a"
                    print(f"[sample {n_decisions}] prompt_tail="
                          f"{prompts[j][-60:]!r} digits={ds} dec={dec} "
                          f"col={acts[j]} | qhead_col={qcol} qhead_dec={qdec}",
                          flush=True)
                    n_decisions += 1
            still = []
            for j, i in enumerate(active):
                o, r, done, info = envs[i].step(int(acts[j]))
                moves_l[i] += 1
                if done:
                    scores[i] = float(envs[i].score)
                    fruits = envs[i].env.get_state()["fruits"]
                    maxf_l[i] = max((f["type"] for f in fruits), default=-1)
                else:
                    obs[i] = o
                    still.append(i)
            active = still
        for i in np.where(np.isnan(scores))[0]:   # safety net
            scores[i] = float(envs[i].score)

        rec = _stats(scores, moves_l, maxf_l, payload, gs, t0)
        rec["decode"] = args.decode
        rec["temperature"] = args.temperature
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[eval] decode={args.decode} gs={gs} mean={rec['mean']:.1f} "
              f"max={rec['max']:.0f} ({rec['eval_s']}s)", flush=True)
        time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
