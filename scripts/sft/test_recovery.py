from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def stop_process_group(process: subprocess.Popen, timeout: int = 15) -> int | None:
    if process.poll() is not None:
        return process.returncode
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait()


def wait_for_checkpoint(path: Path, process: subprocess.Popen, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and (path / ".checkpoint_complete.json").is_file():
            return
        if process.poll() is not None:
            raise RuntimeError(
                f"Training exited before checkpoint appeared with code {process.returncode}"
            )
        time.sleep(1)
    stop_process_group(process)
    raise TimeoutError(f"Checkpoint not completed within {timeout}s: {path}")


def wait_for_resume(
    output: Path, process: subprocess.Popen, timeout: int, expected_steps: int
) -> int:
    manifest_path = output / "run_manifest.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )
        global_step = manifest.get("global_step")
        if isinstance(global_step, int) and global_step > expected_steps:
            stop_process_group(process)
            raise RuntimeError(
                f"Resumed training exceeded max_steps={expected_steps}: global_step={global_step}"
            )
        if return_code is not None:
            return return_code
        time.sleep(0.5)
    stop_process_group(process)
    raise TimeoutError(f"Resumed training did not finish within {timeout}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    command = [
        sys.executable,
        str(args.trainer),
        "--model",
        args.model,
        "--train",
        str(args.train),
        "--output",
        str(args.output),
        "--epochs",
        "1",
        "--max-steps",
        str(args.max_steps),
        "--save-steps",
        "1",
        "--learning-rate",
        "5e-6",
        "--max-length",
        "4096",
        "--batch-size",
        "1",
        "--gradient-accumulation-steps",
        "1",
        "--warmup-steps",
        "1",
        "--seed",
        "20261002",
        "--save-total-limit",
        "4",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    args.output.mkdir(parents=True, exist_ok=True)
    log_dir = args.output.parent / f"{args.output.name}_recovery_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    first_log = (log_dir / "first_pass.log").open("w", encoding="utf-8")
    first = subprocess.Popen(
        command,
        env=environment,
        stdout=first_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        first_checkpoint = args.output / "checkpoint-1"
        wait_for_checkpoint(first_checkpoint, first, args.timeout)
        first.send_signal(signal.SIGTERM)
        try:
            first_code = first.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            stop_process_group(first)
            raise TimeoutError(
                f"Interrupted training did not stop within {args.timeout}s"
            )
    finally:
        first_log.close()

    manifest_path = args.output / "run_manifest.json"
    interrupted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if interrupted_manifest.get("status") != "interrupted":
        raise RuntimeError(
            f"Expected interrupted status after SIGTERM, got {interrupted_manifest.get('status')}"
        )
    if interrupted_manifest.get("last_checkpoint") is None:
        raise RuntimeError(
            "Interrupted run manifest did not record its last complete checkpoint"
        )

    resume_command = command + ["--resume", "auto"]
    resume_log = (log_dir / "resume.log").open("w", encoding="utf-8")
    resume = subprocess.Popen(
        resume_command,
        env=environment,
        stdout=resume_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        resume_code = wait_for_resume(args.output, resume, args.timeout, args.max_steps)
    finally:
        resume_log.close()
    if resume_code:
        raise RuntimeError(
            f"Resume exited with code {resume_code}; inspect {log_dir / 'resume.log'}"
        )

    completed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_steps = args.max_steps
    if (
        completed_manifest.get("status") != "completed"
        or completed_manifest.get("global_step") != expected_steps
    ):
        raise RuntimeError(
            f"Unexpected resumed completion state: {completed_manifest.get('status')} step={completed_manifest.get('global_step')}"
        )
    checkpoint = args.output / f"checkpoint-{expected_steps}"
    if not (checkpoint / ".checkpoint_complete.json").is_file():
        raise RuntimeError(
            f"Resumed run did not produce a marked complete checkpoint-{expected_steps}"
        )
    checkpoint_state = json.loads(
        (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
    )
    if checkpoint_state.get("global_step") != expected_steps:
        raise RuntimeError(
            f"Final checkpoint has wrong trainer step: {checkpoint_state.get('global_step')}"
        )
    if completed_manifest.get("resume_checkpoint") != interrupted_manifest.get(
        "last_checkpoint"
    ):
        raise RuntimeError(
            "Resume did not select the last complete checkpoint from the interrupted run"
        )
    result = {
        "status": "passed",
        "first_process_exit_code": first_code,
        "interrupted_status": interrupted_manifest["status"],
        "interrupted_last_checkpoint": interrupted_manifest["last_checkpoint"],
        "resume_checkpoint": completed_manifest.get("resume_checkpoint"),
        "final_status": completed_manifest["status"],
        "final_global_step": completed_manifest["global_step"],
        "complete_checkpoints": completed_manifest["epoch_checkpoints"],
        "output": str(args.output),
    }
    (args.output / "recovery_test.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
