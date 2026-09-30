#!/usr/bin/env python3
"""Zero-shot / checkpoint eval of a model on an MCQ/ranking jsonl via vLLM.

Usage: python -m rlforge.eval_mcq --model <path> --data eval.jsonl [--out results.json] \
           [--max-tokens 16384] [--temperature 1.0] [--top-p 1.0] [--n-samples 2]

Scoring is the SAME code path as the training reward (imported from
rlforge.rewards.mcq, with a fallback copy for standalone use), so an eval number
and a training reward mean the same thing. Reports per-task breakdown (mcq /
ranking), per-sample records, and a constant-letter baseline computed on the
same rows.

Checkpoints saved by a trainer sometimes lack tokenizer/processor files; those
are borrowed from a base model (RLFORGE_BASE_MODEL env, or the evaluated path
itself) into a shim directory -- never weights.
"""

import argparse
import collections
import hashlib
import json
import os
import re
import sys
from pathlib import Path

BASE_MODEL = os.environ.get("RLFORGE_BASE_MODEL") or os.environ.get("AIQ_BASE_MODEL", "")

# --- shared reward semantics (single source of truth with training) -------------
try:
    from rlforge.rewards.mcq import (  # noqa: E402
        UNPARSED_REWARD,
        TRUNCATED_REWARD,
        is_ranking_gold,
        parse_answer,
        parse_order,
        ranking_score,
    )
except ImportError:  # standalone copy (kept in lock-step; eval-only fallback)
    UNPARSED_REWARD = -0.5
    TRUNCATED_REWARD = -2.0
    ANSWER_RE = re.compile(r"<answer>\s*([^<]*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
    ORDER_TAG_RE = re.compile(
        r"<answer>\s*([A-Ea-e](?:\s*[<>]\s*[A-Ea-e])+)\s*</answer>",
        re.DOTALL | re.IGNORECASE,
    )
    LETTER_RE = re.compile(r"^[^A-Za-z]*([A-E])\b")
    SEPARATOR_RE = re.compile(r"\s*[<>]\s*")

    def _as_letters(value):
        if value is None:
            return None
        if isinstance(value, str):
            parts = [p.upper() for p in SEPARATOR_RE.split(value.strip()) if p]
        else:
            parts = [str(p).strip().upper() for p in value]
        return tuple(parts) or None

    def parse_answer(text):
        tags = ANSWER_RE.findall(text)
        if not tags:
            return None
        m = LETTER_RE.match(tags[-1])
        return m.group(1).upper() if m else None

    def parse_order(text):
        tags = ORDER_TAG_RE.findall(text)
        if not tags:
            return None
        letters = _as_letters(tags[-1])
        return letters if letters and len(letters) == 5 and set(letters) == set("ABCDE") else None

    def is_ranking_gold(gold):
        letters = _as_letters(gold)
        return bool(letters) and len(letters) > 1

    def ranking_score(prediction, gold):
        expected = _as_letters(gold)
        if expected is None or len(expected) != 5 or set(expected) != set("ABCDE"):
            raise ValueError(f"Invalid gold ranking: {gold!r}")
        predicted = _as_letters(prediction)
        if predicted is None or len(predicted) != 5 or set(predicted) != set("ABCDE"):
            return UNPARSED_REWARD
        if predicted == expected:
            return 1.0
        positions = {letter: index for index, letter in enumerate(expected)}
        concordant = sum(
            positions[predicted[l]] < positions[predicted[r]]
            for l in range(5)
            for r in range(l + 1, 5)
        )
        return concordant / 5.0 - 1.0


def build_shim(model_path: str) -> str:
    """TRL checkpoints lack preprocessor_config.json etc.; vLLM (VL-wrapped
    Qwen3.5) refuses to load them. Build a shim dir: base-model files with the
    checkpoint's weight/config files symlinked over them."""
    p = Path(model_path).resolve()
    if (p / "preprocessor_config.json").exists():
        return str(p)
    base = Path(BASE_MODEL).resolve() if BASE_MODEL else p
    if not (base / "preprocessor_config.json").exists():
        raise SystemExit(
            f"{p} has no preprocessor/tokenizer files and no base model to borrow them "
            "from; set RLFORGE_BASE_MODEL to the base model directory."
        )
    tag = f"{p.parent.name}_{p.name}_" + hashlib.md5(str(p).encode()).hexdigest()[:8]
    shim = p.parent / "_shims" / tag
    if (shim / "preprocessor_config.json").exists():
        return str(shim)
    shim.mkdir(parents=True, exist_ok=True)
    # base contributes ONLY processor/tokenizer/template files — never weights
    # (base's sharded weights + index would override the checkpoint's single
    # model.safetensors and silently evaluate the base model instead).
    weight_re = re.compile(
        r"^model\..*\.(safetensors|bin)$|^model\.safetensors\.index\.json$|^pytorch_model"
    )
    for src in base.iterdir():
        if src.is_file() and not weight_re.search(src.name):
            dst = shim / src.name
            if not dst.is_symlink() and not dst.exists():
                os.symlink(src, dst)
    for src in p.iterdir():
        if src.is_file():
            dst = shim / src.name
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            os.symlink(src, dst)
    return str(shim)


def constant_baseline(rows):
    """Accuracy of always answering the majority gold, per task, on these rows."""
    out = {}
    for task in ("mcq", "ranking"):
        golds = [r["answer"] for r in rows if is_ranking_gold(r["answer"]) == (task == "ranking")]
        if not golds:
            continue
        if task == "mcq":
            top = collections.Counter(golds).most_common(1)[0]
            out[task] = {"const_answer": top[0], "accuracy": top[1] / len(golds), "n": len(golds)}
        else:
            # ranking: exact match of the most common gold order (a data-blind constant
            # can essentially never hit a full order; reported for completeness)
            top = collections.Counter(golds).most_common(1)[0]
            out[task] = {"const_answer": top[0], "accuracy": top[1] / len(golds), "n": len(golds)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--gpu-frac", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=1,
                    help="samples per question (paired stats need >= 2)")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.data)]
    if args.limit:
        rows = rows[: args.limit]
    from vllm import LLM, SamplingParams

    load_path = build_shim(args.model)
    llm = LLM(
        model=load_path,
        gpu_memory_utilization=args.gpu_frac,
        max_model_len=24576,
        enable_prefix_caching=True,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=0,
        n=args.n_samples,
    )
    outs = llm.chat(
        [r["prompt"] for r in rows], sp, chat_template_kwargs={"enable_thinking": True}
    )

    recs = []
    for r, o in zip(rows, outs):
        gold = r["answer"].strip()
        ranking = is_ranking_gold(gold)
        samples = []
        for cand in o.outputs:
            text = cand.text
            n_tok = len(cand.token_ids)
            truncated = n_tok >= args.max_tokens
            if ranking:
                pred_letters = parse_order(text)
                pred = "<".join(pred_letters) if pred_letters else None
                reward = TRUNCATED_REWARD if truncated else ranking_score(pred_letters, gold)
                exact = float(reward == 1.0)
            else:
                pred = parse_answer(text)
                if truncated:
                    reward = TRUNCATED_REWARD
                elif pred is None:
                    reward = UNPARSED_REWARD
                else:
                    reward = 1.0 if pred == gold.upper() else 0.0
                exact = float(pred == gold.upper()) if pred is not None else 0.0
            samples.append({
                "pred": pred,
                "reward": reward,
                "exact": exact,
                "parsed": pred is not None,
                "truncated": truncated,
                "completion_tokens": n_tok,
            })
        recs.append({
            "question_id": r["question_id"],
            "family": r.get("family", "?"),
            "task": "ranking" if ranking else "mcq",
            "answer": gold,
            "samples": samples,
        })

    def agg(rs):
        n_q = len(rs)
        n_s = sum(len(x["samples"]) for x in rs)
        if not n_s:
            return {"n_questions": n_q, "n_samples": 0}
        rewards = [s["reward"] for x in rs for s in x["samples"]]
        return {
            "n_questions": n_q,
            "n_samples": n_s,
            "accuracy": sum(s["exact"] for x in rs for s in x["samples"]) / n_s,
            "mean_reward": sum(rewards) / n_s,
            "parse_rate": sum(s["parsed"] for x in rs for s in x["samples"]) / n_s,
            "truncation_rate": sum(s["truncated"] for x in rs for s in x["samples"]) / n_s,
            "mean_completion_tokens": sum(
                s["completion_tokens"] for x in rs for s in x["samples"]
            ) / n_s,
        }

    by_task = {}
    for task in ("mcq", "ranking"):
        sub = [x for x in recs if x["task"] == task]
        if sub:
            by_task[task] = agg(sub)
    by_family = {}
    for fam in sorted({x["family"] for x in recs}):
        by_family[fam] = agg([x for x in recs if x["family"] == fam])
    letter_dist = collections.Counter(
        s["pred"] or "?" for x in recs for s in x["samples"] if x["task"] == "mcq"
    )

    res = {
        "model": args.model,
        "data": args.data,
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "n_samples": args.n_samples,
            "thinking": True,
        },
        "overall": agg(recs),
        "by_task": by_task,
        "by_family": {k: {"accuracy": v.get("accuracy"), "n_questions": v["n_questions"]}
                      for k, v in by_family.items()},
        "letter_dist_mcq": dict(letter_dist),
        "constant_baseline": constant_baseline(rows),
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"summary": res, "records": recs}, ensure_ascii=False, indent=1)
        )


if __name__ == "__main__":
    main()
