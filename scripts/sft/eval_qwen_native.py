from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
MCQ_RE = re.compile(r"[A-E]", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


class PresencePenalty(LogitsProcessor):
    def __init__(self, prompt_tokens: int, penalty: float):
        self.prompt_tokens = prompt_tokens
        self.penalty = penalty

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        generated = input_ids[:, self.prompt_tokens :]
        if generated.shape[-1] == 0 or self.penalty == 0:
            return scores
        seen = torch.zeros_like(scores, dtype=torch.bool)
        seen.scatter_(1, generated, True)
        return scores - seen.to(scores.dtype) * self.penalty


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_response(
    text: str, gold: str, truncated: bool, completion_tokens: int
) -> dict[str, Any]:
    answer_tags = ANSWER_RE.findall(text)
    body = answer_tags[-1].strip() if answer_tags else None
    match = MCQ_RE.match(body or "")
    prediction = match.group(0).upper() if match else None
    parsed = prediction is not None
    exact = bool(parsed and prediction == gold.strip().upper())
    reward = -2.0 if truncated else (-0.5 if not parsed else (1.0 if exact else 0.0))
    last_close = list(THINK_CLOSE_RE.finditer(text))
    thought = text[: last_close[-1].start()].strip() if last_close else ""
    after_think = text[last_close[-1].end() :].strip() if last_close else ""
    answer_after_think = bool(
        last_close
        and re.fullmatch(
            r"<answer>.*?</answer>", after_think, re.IGNORECASE | re.DOTALL
        )
    )
    strict = bool(
        len(answer_tags) == 1 and body and re.fullmatch(r"[A-E]", body.upper())
    )
    return {
        "pred": prediction,
        "parsed": parsed,
        "exact": exact,
        "reward": reward,
        "answer_tag_count": len(answer_tags),
        "strict_answer_format": strict,
        "thinking_boundary": bool(last_close),
        "visible_thinking_text": bool(thought),
        "visible_thinking_tokens": len(re.findall(r"\S+", thought)),
        "answer_only_after_think": answer_after_think,
        "truncated": truncated,
        "completion_tokens": completion_tokens,
        "response_text": text,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [sample for row in rows for sample in row["samples"]]
    if not samples:
        return {"n_questions": len(rows), "n_samples": 0}
    count = len(samples)
    return {
        "n_questions": len(rows),
        "n_samples": count,
        "accuracy": sum(sample["exact"] for sample in samples) / count,
        "mean_reward": sum(sample["reward"] for sample in samples) / count,
        "parse_rate": sum(sample["parsed"] for sample in samples) / count,
        "strict_answer_format_rate": sum(
            sample["strict_answer_format"] for sample in samples
        )
        / count,
        "thinking_boundary_rate": sum(sample["thinking_boundary"] for sample in samples)
        / count,
        "visible_thinking_rate": sum(
            sample["visible_thinking_text"] for sample in samples
        )
        / count,
        "answer_after_thinking_rate": sum(
            sample["answer_only_after_think"] for sample in samples
        )
        / count,
        "truncation_rate": sum(sample["truncated"] for sample in samples) / count,
        "mean_completion_tokens": sum(sample["completion_tokens"] for sample in samples)
        / count,
        "mean_visible_thinking_tokens": sum(
            sample["visible_thinking_tokens"] for sample in samples
        )
        / count,
        "answer_distribution": dict(
            collections.Counter(sample["pred"] or "?" for sample in samples)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-samples", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len({row["question_id"] for row in rows}) != len(rows):
        raise ValueError("Evaluation question_id values are not unique")
    if any(row.get("task") != "mcq" for row in rows):
        raise ValueError("This pilot evaluator expects v1.5 multiple-choice questions")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        .cuda()
        .eval()
    )
    model.config.use_cache = True

    records = []
    for index, row in enumerate(rows, 1):
        prompt = tokenizer.apply_chat_template(
            row["prompt"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            model.device
        )
        prompt_tokens = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else 1.0,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_tokens,
                num_return_sequences=args.n_samples,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                logits_processor=LogitsProcessorList(
                    [PresencePenalty(prompt_tokens, args.presence_penalty)]
                ),
            )
        samples = []
        for output in generated:
            completion_ids = output[prompt_tokens:]
            text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            samples.append(
                score_response(
                    text,
                    row["answer"],
                    len(completion_ids) >= args.max_tokens,
                    len(completion_ids),
                )
            )
        records.append(
            {
                "question_id": row["question_id"],
                "family": row["family"],
                "question_type": row["question_type"],
                "task": row["task"],
                "answer": row["answer"],
                "samples": samples,
            }
        )
        if index % 10 == 0 or index == len(rows):
            print(
                json.dumps(
                    {"completed": index, "total": len(rows)}, ensure_ascii=False
                ),
                flush=True,
            )

    evaluation_by_type = {
        question_type: aggregate(
            [row for row in records if row["question_type"] == question_type]
        )
        for question_type in sorted({row["question_type"] for row in records})
    }
    result = {
        "model": args.model,
        "data": str(args.data.resolve()),
        "data_sha256": file_sha256(args.data),
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": 0.0,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": 1.0,
            "max_tokens": args.max_tokens,
            "n_samples": args.n_samples,
            "enable_thinking": True,
            "backend": "transformers_generate_sdpa_bf16",
            "note": "Qwen3.5-0.8B has no native low/medium/high reasoning_effort control; concise target and output cap are budget proxies only",
        },
        "overall": aggregate(records),
        "by_question_type": evaluation_by_type,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("data_sha256", "decoding", "overall", "by_question_type")
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
