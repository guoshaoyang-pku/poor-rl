from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def checkpoint_is_complete(path: Path) -> bool:
    if not path.is_dir() or not path.name.startswith("checkpoint-"):
        return False
    suffix = path.name.removeprefix("checkpoint-")
    if not suffix.isdigit():
        return False
    required = [
        path / ".checkpoint_complete.json",
        path / "optimizer.pt",
        path / "scheduler.pt",
        path / "trainer_state.json",
        path / "tokenizer_config.json",
    ]
    if not all(item.is_file() for item in required):
        return False
    if not any(path.glob("*.safetensors")) and not any(path.glob("pytorch_model*.bin")):
        return False
    try:
        marker = json.loads(required[0].read_text(encoding="utf-8"))
        trainer_state = json.loads(required[3].read_text(encoding="utf-8"))
        return int(marker["step"]) == int(suffix) == int(trainer_state["global_step"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def run_process_group(command: list[str], log_path: Path, timeout: int) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise TimeoutError(f"Evaluator exceeded timeout of {timeout}s") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--eval-data", required=True, type=Path)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--checkpoints", default="all", help="all or comma-separated global steps"
    )
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu-frac", type=float, default=0.8)
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--gdn-prefill-backend", default="triton")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    eval_data = args.eval_data.resolve()
    evaluator = args.evaluator.resolve()
    run_manifest_path = run_dir / "run_manifest.json"
    if (
        not run_manifest_path.is_file()
        or not eval_data.is_file()
        or not evaluator.is_file()
    ):
        raise FileNotFoundError(
            "Expected run_manifest.json, held-out data, and evaluator to exist"
        )
    json.loads(run_manifest_path.read_text(encoding="utf-8"))
    checkpoints = sorted(
        (path for path in run_dir.glob("checkpoint-*") if checkpoint_is_complete(path)),
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
    )
    if args.checkpoints != "all":
        wanted = {
            int(value.strip()) for value in args.checkpoints.split(",") if value.strip()
        }
        checkpoints = [
            path for path in checkpoints if int(path.name.rsplit("-", 1)[1]) in wanted
        ]
        missing = wanted - {int(path.name.rsplit("-", 1)[1]) for path in checkpoints}
        if missing:
            raise FileNotFoundError(f"Missing requested checkpoints: {sorted(missing)}")
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")

    output_dir = (args.output_dir or run_dir / "checkpoint_evals").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "ladder_manifest.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {
            "status": "pending",
            "run_dir": str(run_dir),
            "run_manifest_sha256": sha256(run_manifest_path),
            "eval_data": str(eval_data),
            "eval_data_sha256": sha256(eval_data),
            "evaluator": str(evaluator),
            "evaluator_sha256": sha256(evaluator),
            "decoding": {
                "n_samples": args.n_samples,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "presence_penalty": args.presence_penalty,
                "repetition_penalty": args.repetition_penalty,
                "max_tokens": args.max_tokens,
                "batch_size": args.batch_size,
                "gpu_frac": args.gpu_frac,
                "kv_cache_dtype": args.kv_cache_dtype,
                "gdn_prefill_backend": args.gdn_prefill_backend,
            },
            "checkpoints": {},
        }
    )
    invariant_fields = {
        "run_dir": str(run_dir),
        "eval_data_sha256": sha256(eval_data),
        "evaluator_sha256": sha256(evaluator),
        "decoding": {
            "n_samples": args.n_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
            "max_tokens": args.max_tokens,
            "batch_size": args.batch_size,
            "gpu_frac": args.gpu_frac,
            "kv_cache_dtype": args.kv_cache_dtype,
            "gdn_prefill_backend": args.gdn_prefill_backend,
        },
    }
    for key, value in invariant_fields.items():
        if state.get(key) != value:
            raise ValueError(f"Cannot resume checkpoint ladder: {key} changed")
    state["status"] = "running"
    atomic_write(state_path, state)

    for checkpoint in checkpoints:
        step = int(checkpoint.name.rsplit("-", 1)[1])
        result_path = output_dir / f"checkpoint-{step}.json"
        record = state["checkpoints"].get(str(step), {})
        if (
            record.get("status") == "completed"
            and result_path.is_file()
            and record.get("result_sha256") == sha256(result_path)
        ):
            continue
        record.update(
            {
                "status": "running",
                "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "checkpoint": str(checkpoint.resolve()),
            }
        )
        state["checkpoints"][str(step)] = record
        atomic_write(state_path, state)
        command = [
            args.python,
            str(evaluator),
            "--model",
            str(checkpoint.resolve()),
            "--data",
            str(eval_data),
            "--out",
            str(result_path),
            "--n-samples",
            str(args.n_samples),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--top-k",
            str(args.top_k),
            "--presence-penalty",
            str(args.presence_penalty),
            "--repetition-penalty",
            str(args.repetition_penalty),
            "--max-tokens",
            str(args.max_tokens),
            "--batch-size",
            str(args.batch_size),
            "--gpu-frac",
            str(args.gpu_frac),
            "--kv-cache-dtype",
            args.kv_cache_dtype,
            "--gdn-prefill-backend",
            args.gdn_prefill_backend,
        ]
        log_path = output_dir / f"checkpoint-{step}.log"
        try:
            completed = run_process_group(command, log_path, args.timeout_seconds)
            if completed != 0 or not result_path.is_file():
                raise RuntimeError(f"Evaluator exited {completed}; see {log_path}")
            record.update(
                {
                    "status": "completed",
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "result": str(result_path),
                    "result_sha256": sha256(result_path),
                    "log": str(log_path),
                    "exit_code": completed,
                }
            )
            print(
                json.dumps(
                    {"step": step, "status": "completed", "result": str(result_path)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except BaseException as exc:
            record.update(
                {
                    "status": "failed",
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "log": str(log_path),
                }
            )
            state["status"] = "failed"
            atomic_write(state_path, state)
            if args.stop_on_error:
                raise
        atomic_write(state_path, state)

    failed = [
        step
        for step, record in state["checkpoints"].items()
        if record.get("status") != "completed"
    ]
    state["status"] = "failed" if failed else "completed"
    state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["failed_steps"] = failed
    atomic_write(state_path, state)
    print(
        json.dumps(
            {
                "status": state["status"],
                "manifest": str(state_path),
                "failed_steps": failed,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
