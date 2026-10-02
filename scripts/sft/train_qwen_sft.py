from __future__ import annotations

import argparse
import atexit
import fcntl
import hashlib
import json
import math
import os
import platform
import random
import signal
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import transformers
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


class TokenizedDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int):
        self.samples = []
        prompt_token_counts = []
        total_prompt_tokens = 0
        total_target_tokens = 0
        for row in rows:
            messages = row["messages"]
            if [message["role"] for message in messages] != [
                "system",
                "user",
                "assistant",
            ]:
                raise ValueError(
                    f"Unexpected message roles for {row.get('question_id')}"
                )
            completion = messages[-1]["content"]
            if (
                "</think>" not in completion
                or "<answer>" not in completion
                or "<explanation>" in completion
            ):
                raise ValueError(
                    f"Expected private thinking and final answer targets for {row.get('question_id')}"
                )
            prompt = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                return_dict=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )["input_ids"]
            full = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=False,
                enable_thinking=True,
            )["input_ids"]
            shared = 0
            for prompt_token, full_token in zip(prompt, full):
                if prompt_token != full_token:
                    break
                shared += 1
            if shared < len(prompt) - 4:
                raise ValueError(
                    f"Chat-template boundary mismatch for {row.get('question_id')}: {shared}/{len(prompt)}"
                )
            if len(full) > max_length:
                raise ValueError(
                    f"Sequence too long for {row.get('question_id')}: {len(full)} > {max_length}"
                )
            labels = list(full)
            labels[:shared] = [-100] * shared
            target_count = sum(token != -100 for token in labels)
            if target_count < 8:
                raise ValueError(
                    f"Too few supervised target tokens for {row.get('question_id')}"
                )
            self.samples.append({"input_ids": list(full), "labels": labels})
            prompt_token_counts.append(len(prompt))
            total_prompt_tokens += shared
            total_target_tokens += target_count
        self.stats = {
            "examples": len(self.samples),
            "max_sequence_tokens": max(
                len(sample["input_ids"]) for sample in self.samples
            ),
            "mean_sequence_tokens": sum(
                len(sample["input_ids"]) for sample in self.samples
            )
            / len(self.samples),
            "mean_masked_prompt_tokens": total_prompt_tokens / len(self.samples),
            "mean_supervised_tokens": total_target_tokens / len(self.samples),
            "min_prompt_prefix_match_ratio": min(
                sample_prefix / prompt_length
                for sample_prefix, prompt_length in zip(
                    [
                        len(sample["input_ids"])
                        - sum(token != -100 for token in sample["labels"])
                        for sample in self.samples
                    ],
                    prompt_token_counts,
                )
            ),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class CompletionOnlyCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        max_length = (max_length + 7) // 8 * 8
        input_ids = torch.full(
            (len(features), max_length), self.pad_token_id, dtype=torch.long
        )
        labels = torch.full((len(features), max_length), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(features), max_length), dtype=torch.long)
        for index, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[index, :length] = torch.tensor(
                feature["input_ids"], dtype=torch.long
            )
            labels[index, :length] = torch.tensor(feature["labels"], dtype=torch.long)
            attention_mask[index, :length] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class RunStateCallback(TrainerCallback):
    def __init__(self, output: Path, manifest: dict[str, Any], tokenizer):
        self.output = output
        self.manifest = manifest
        self.tokenizer = tokenizer
        self.manifest_path = output / "run_manifest.json"
        self.events_path = output / "run_events.jsonl"

        if int(os.environ.get("RANK", "0")) == 0 and self.events_path.exists():
            archive = self.events_path.with_name(
                f"run_events_resume_{int(time.time())}.jsonl"
            )
            os.replace(self.events_path, archive)

    def write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)

    def append_event(self, payload: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def on_step_end(self, args, state, control, **kwargs):
        if STOP_REQUESTED or state.global_step >= state.max_steps:
            control.should_save = True
            control.should_training_stop = (
                control.should_training_stop
                or STOP_REQUESTED
                or state.global_step >= state.max_steps
            )
            if STOP_REQUESTED and state.is_world_process_zero:
                self.manifest["stop_reason"] = (
                    f"signal_{signal.Signals(STOP_SIGNAL).name if STOP_SIGNAL else 'UNKNOWN'}"
                )
                self.write_json_atomic(self.manifest_path, self.manifest)
        return control

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        checkpoint = self.output / f"checkpoint-{state.global_step}"
        if not checkpoint_is_complete(
            checkpoint, expected_step=state.global_step, require_marker=False
        ):
            raise RuntimeError(f"Refusing to mark incomplete checkpoint: {checkpoint}")
        marker = checkpoint / ".checkpoint_complete.json"
        temporary_marker = marker.with_suffix(".json.tmp")
        self.tokenizer.save_pretrained(checkpoint)
        temporary_marker.write_text(
            json.dumps({"step": state.global_step, "timestamp": time.time()}) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_marker, marker)
        self.append_event(
            {
                "event": "checkpoint_saved",
                "step": state.global_step,
                "path": str(checkpoint),
                "timestamp": time.time(),
            }
        )
        self.manifest["global_step"] = state.global_step
        self.manifest["last_saved_step"] = state.global_step
        self.manifest["last_checkpoint"] = str(checkpoint.resolve())
        self.manifest["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.write_json_atomic(self.manifest_path, self.manifest)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return control
        self.append_event(
            {
                "event": "train_log",
                "step": state.global_step,
                "logs": logs,
                "timestamp": time.time(),
            }
        )
        return control


STOP_REQUESTED = False
STOP_SIGNAL: int | None = None


def request_stop(signum, frame):
    global STOP_REQUESTED, STOP_SIGNAL
    STOP_REQUESTED = True
    STOP_SIGNAL = signum


def checkpoint_is_complete(
    path: Path, expected_step: int | None = None, require_marker: bool = True
) -> bool:
    if not path.is_dir():
        return False
    required = ["optimizer.pt", "scheduler.pt", "trainer_state.json"]
    if require_marker:
        required.append("tokenizer_config.json")
    if not all((path / name).is_file() for name in required):
        return False
    has_weights = any(path.glob("*.safetensors")) or any(
        path.glob("pytorch_model*.bin")
    )
    if not has_weights:
        return False
    try:
        trainer_state = json.loads(
            (path / "trainer_state.json").read_text(encoding="utf-8")
        )
        step = int(trainer_state["global_step"])
        if expected_step is not None and step != expected_step:
            return False
        marker_path = path / ".checkpoint_complete.json"
        if require_marker:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if int(marker["step"]) != step:
                return False
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return True


def last_checkpoint(output: Path) -> Path | None:
    checkpoints = [
        path
        for path in output.glob("checkpoint-*")
        if path.name.rsplit("-", 1)[-1].isdigit()
        and checkpoint_is_complete(path, expected_step=int(path.name.rsplit("-", 1)[1]))
    ]
    return (
        max(checkpoints, key=lambda path: int(path.name.rsplit("-", 1)[-1]))
        if checkpoints
        else None
    )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    if path.is_file():
        return file_sha256(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Model directory contains no files: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(item)))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument(
        "--save-steps",
        type=int,
        default=20,
        help="0 selects one fixed save interval per epoch",
    )
    parser.add_argument("--save-total-limit", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume", choices=("none", "auto"), default="none")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    if args.resume == "auto" and args.resume_from_checkpoint:
        parser.error(
            "--resume auto and --resume-from-checkpoint are mutually exclusive"
        )
    if args.resume_from_checkpoint and args.resume_from_checkpoint.is_dir() is False:
        parser.error(f"Checkpoint does not exist: {args.resume_from_checkpoint}")
    if args.save_steps < 0 or args.save_total_limit < 1:
        parser.error(
            "save-steps must be non-negative and save-total-limit must be positive"
        )

    rows = read_jsonl(args.train)
    random.Random(args.seed).shuffle(rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = TokenizedDataset(rows, tokenizer, args.max_length)
    print(
        json.dumps(
            {"train_rows": len(rows), "tokenization": dataset.stats}, ensure_ascii=False
        ),
        flush=True,
    )
    if args.preflight_only:
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    steps_per_epoch = math.ceil(
        len(dataset) / (args.batch_size * args.gradient_accumulation_steps * world_size)
    )
    save_steps = args.save_steps or max(1, steps_per_epoch)
    if args.max_steps > 0:
        save_steps = min(save_steps, args.max_steps)
    if args.gradient_checkpointing:
        print(
            "gradient checkpointing is enabled; expect lower throughput in exchange for lower activation memory",
            flush=True,
        )
    if world_size > 1 and os.environ.get("LOCAL_RANK") is None:
        raise ValueError("WORLD_SIZE > 1 requires torchrun/accelerate DDP environment")

    args.output.mkdir(parents=True, exist_ok=True)
    lock_path = args.output / ".run.lock"
    lock_stream = lock_path.open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(
            f"Another training process holds the run lock: {lock_path}"
        ) from exc
    existing_items = [path for path in args.output.iterdir() if path != lock_path]
    if existing_items and args.resume == "none" and args.resume_from_checkpoint is None:
        raise FileExistsError(
            f"Refusing to train into a non-empty run directory without resume: {args.output}"
        )
    existing_manifest_path = args.output / "run_manifest.json"
    existing_manifest = (
        json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if existing_manifest_path.exists()
        else None
    )
    if existing_manifest and existing_manifest.get("status") == "completed":
        raise ValueError(
            "Refusing to resume a completed run; start a new output directory"
        )
    if existing_manifest and existing_manifest.get("effective_config") is None:
        raise ValueError(
            "Existing manifest lacks resume-verification config; use a new output directory"
        )
    if args.resume == "auto" and existing_manifest is None:
        raise ValueError(
            "--resume auto requires an existing run_manifest.json to verify inputs"
        )
    if (
        args.resume == "auto" or args.resume_from_checkpoint
    ) and existing_manifest is None:
        raise ValueError(
            "Resume requires an existing run_manifest.json to verify inputs"
        )
    current_train_sha256 = file_sha256(args.train)
    current_model_sha256 = tree_sha256(Path(args.model).resolve())
    effective_config = {
        "base_model": str(Path(args.model).resolve()),
        "base_model_sha256": current_model_sha256,
        "train_sha256": current_train_sha256,
        "script_sha256": file_sha256(Path(__file__)),
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "warmup_steps": args.warmup_steps,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "world_size": world_size,
        "seed": args.seed,
        "bf16": True,
        "base_weights_dtype": "float32",
        "compute_dtype": "bf16",
        "gradient_checkpointing": args.gradient_checkpointing,
        "optimizer": "adamw_torch_fused",
        "scheduler": "cosine",
        "weight_decay": 0.01,
        "save_steps": save_steps,
        "save_total_limit": args.save_total_limit,
        "save_only_model": False,
    }
    if existing_manifest:
        if existing_manifest.get("effective_config") != effective_config:
            raise ValueError(
                "Cannot resume: model, data, script, or training configuration differs from run manifest"
            )
    if args.resume == "auto":
        resume_checkpoint = last_checkpoint(args.output)
        if resume_checkpoint is None:
            raise FileNotFoundError(
                "--resume auto requested but no checkpoint-* folder exists"
            )
    elif args.resume_from_checkpoint:
        resume_checkpoint = args.resume_from_checkpoint.resolve()
        if not checkpoint_is_complete(resume_checkpoint, require_marker=True):
            raise ValueError(
                f"Checkpoint is incomplete or missing a completion marker: {resume_checkpoint}"
            )
    else:
        resume_checkpoint = None
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    run_manifest = {
        **(existing_manifest or {}),
        "status": "running",
        "effective_config": effective_config,
        "host": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_count": torch.cuda.device_count(),
        "started": (existing_manifest or {}).get("started", started),
        "last_resumed": started
        if resume_checkpoint
        else (existing_manifest or {}).get("last_resumed"),
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "base_model": str(Path(args.model).resolve()),
        "chat_template": "Qwen tokenizer.apply_chat_template with enable_thinking=True",
        "student_native_thinking": True,
        "student_native_reasoning_effort_levels": False,
        "train_file": str(args.train.resolve()),
        "train_sha256": current_train_sha256,
        "train_rows": len(rows),
        "script_sha256": file_sha256(Path(__file__)),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "warmup_steps": args.warmup_steps,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "bf16": True,
        "base_weights_dtype": "float32",
        "compute_dtype": "bf16",
        "seed": args.seed,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "save_total_limit": args.save_total_limit,
        "save_only_model": False,
        "stop_reason": None,
        "last_saved_step": (existing_manifest or {}).get("last_saved_step"),
        "tokenization": dataset.stats,
    }
    manifest_path = args.output / "run_manifest.json"
    atomic_write_json(manifest_path, run_manifest)
    trainer = None

    def handle_signal(signum, frame):
        request_stop(signum, frame)

    def handle_graceful_exit():
        if (
            int(os.environ.get("RANK", "0")) == 0
            and run_manifest.get("status") == "running"
        ):
            run_manifest.update(
                {
                    "status": "interrupted" if STOP_REQUESTED else "failed",
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "stop_reason": f"signal_{signal.Signals(STOP_SIGNAL).name if STOP_SIGNAL else 'UNKNOWN'}"
                    if STOP_REQUESTED
                    else "incomplete_training_exit",
                    "last_checkpoint": str(last_checkpoint(args.output))
                    if last_checkpoint(args.output)
                    else None,
                }
            )
            atomic_write_json(manifest_path, run_manifest)

    atexit.register(handle_graceful_exit)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGUSR1, handle_signal)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        optim="adamw_torch_fused",
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=False,
        logging_strategy="steps",
        logging_steps=5,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        ddp_timeout=7200,
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CompletionOnlyCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=[RunStateCallback(args.output, run_manifest, tokenizer)],
    )
    resume_from_checkpoint = str(resume_checkpoint) if resume_checkpoint else None
    try:
        result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except BaseException as exc:
        if trainer.is_world_process_zero():
            run_manifest.update(
                {
                    "status": "interrupted"
                    if isinstance(exc, KeyboardInterrupt) or STOP_REQUESTED
                    else "failed",
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "stop_reason": f"signal_{STOP_SIGNAL}"
                    if STOP_REQUESTED
                    else f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-12000:],
                    "last_checkpoint": str(last_checkpoint(args.output))
                    if last_checkpoint(args.output)
                    else None,
                }
            )
            atomic_write_json(manifest_path, run_manifest)
        raise
    trainer.save_state()
    if trainer.is_world_process_zero():
        checkpoints = sorted(
            (
                path
                for path in args.output.glob("checkpoint-*")
                if path.is_dir()
                and path.name.rsplit("-", 1)[-1].isdigit()
                and checkpoint_is_complete(
                    path, expected_step=int(path.name.rsplit("-", 1)[1])
                )
            ),
            key=lambda path: int(path.name.rsplit("-", 1)[1]),
        )
        if not checkpoints:
            raise RuntimeError(
                "No complete checkpoint with optimizer state and tokenizer was created"
            )
        final_checkpoint = checkpoints[-1]
        interrupted = (
            STOP_REQUESTED
            or trainer.state.global_step < training_args.max_steps
            and not args.epochs
        )
        run_manifest.update(
            {
                "status": "interrupted" if interrupted else "completed",
                "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                "stop_reason": run_manifest.get("stop_reason")
                or ("signal" if STOP_REQUESTED else "completed"),
                "tokenization": dataset.stats,
                "train_metrics": result.metrics,
                "global_step": trainer.state.global_step,
                "epoch_checkpoints": [path.name for path in checkpoints],
                "final_checkpoint": final_checkpoint.name,
                "final_checkpoint_path": str(final_checkpoint.resolve()),
                "last_checkpoint": str(final_checkpoint.resolve()),
                "traceback": None,
            }
        )
        atomic_write_json(manifest_path, run_manifest)
        tokenizer.save_pretrained(final_checkpoint)
        print(
            json.dumps(
                {
                    "run_manifest": str(manifest_path),
                    "checkpoints": run_manifest["epoch_checkpoints"],
                    "final": str(final_checkpoint),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
