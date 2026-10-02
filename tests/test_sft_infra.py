import ast
import importlib.util
import json
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SFT_DIR = ROOT / "scripts" / "sft"


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SFT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_max_steps_callback_saves_and_stops():
    source = (SFT_DIR / "train_qwen_sft.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    trainer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RunStateCallback"
    )
    method = next(
        node
        for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "on_step_end"
    )
    callback_class = ast.ClassDef(
        name="Callback", bases=[], keywords=[], body=[method], decorator_list=[]
    )
    namespace = {"STOP_REQUESTED": False, "signal": signal, "STOP_SIGNAL": None}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[callback_class], type_ignores=[])
            ),
            "train_qwen_sft.py",
            "exec",
        ),
        namespace,
    )

    class State:
        global_step = 6
        max_steps = 6
        is_world_process_zero = False

    class Control:
        should_save = False
        should_training_stop = False

    result = namespace["Callback"]().on_step_end(None, State(), Control())
    assert result.should_save is True
    assert result.should_training_stop is True


def test_checkpoint_gate_requires_matching_marker_and_trainer_state(tmp_path):
    ladder = load_script("sft_ladder", "eval_checkpoint_ladder.py")
    checkpoint = tmp_path / "checkpoint-6"
    checkpoint.mkdir()
    for name in (
        "optimizer.pt",
        "scheduler.pt",
        "tokenizer_config.json",
        "model.safetensors",
    ):
        (checkpoint / name).touch()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 6}), encoding="utf-8"
    )
    marker = checkpoint / ".checkpoint_complete.json"
    marker.write_text(json.dumps({"step": 6}), encoding="utf-8")
    assert ladder.checkpoint_is_complete(checkpoint)
    marker.write_text(json.dumps({"step": 5}), encoding="utf-8")
    assert not ladder.checkpoint_is_complete(checkpoint)


def test_recovery_watchdog_stops_overrun_process_group(tmp_path):
    recovery = load_script("sft_recovery", "test_recovery.py")
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(json.dumps({"global_step": 7}), encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        recovery.wait_for_resume(tmp_path, process, timeout=3, expected_steps=6)
    except RuntimeError as error:
        assert "exceeded max_steps=6" in str(error)
        assert process.poll() is not None
    else:
        raise AssertionError("watchdog did not reject resumed step overflow")


def test_evaluator_timeout_kills_child_process_group(tmp_path):
    ladder = load_script("sft_timeout", "eval_checkpoint_ladder.py")
    try:
        ladder.run_process_group(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            tmp_path / "evaluation.log",
            timeout=1,
        )
    except TimeoutError:
        pass
    else:
        raise AssertionError("evaluator timeout was not enforced")
