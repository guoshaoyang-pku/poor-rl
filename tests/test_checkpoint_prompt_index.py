#!/usr/bin/env python3
"""The rollout_state prompt_index must survive dropped groups (RLFORGE_DROP_ZERO_VAR=1).

TRL writes prompt_index = dataset_start_index + first group id not in _trained_groups. A group the
rollout child drops (zero advantage variance) is never trained, so that id caps the recorded index for
the rest of the run and a resume replays prompts it already trained. rlforge rewrites the file with the
highest group the trainer saw: groups are admitted in order, so that is where the stream stands.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlforge.trainer import _fix_checkpoint_prompt_index  # noqa: E402


def make(tmp, trained, start, step=10):
    ck = tmp / f"checkpoint-{step}"
    ck.mkdir(parents=True)
    # what TRL's own _save_checkpoint writes
    first_untrained = next(g for g in range(10 ** 6) if g not in trained)
    (ck / "rollout_state.json").write_text(json.dumps(
        {"prompt_index": start + first_untrained, "model_version": step + 1}))
    return SimpleNamespace(
        _save_checkpoint=lambda model, trial: None,
        _get_output_dir=lambda trial: str(tmp),
        _trained_groups=trained,
        accelerator=SimpleNamespace(is_main_process=True),
        state=SimpleNamespace(global_step=step),
        rollout_worker=SimpleNamespace(_loop_kwargs={"dataset_start_index": start}),
    )


def test_dropped_groups_do_not_rewind_the_stream(tmp_path):
    trained = set(range(2560)) - {14, 97, 2559}  # three groups dropped before the trainer
    trainer = make(tmp_path, trained, start=6382)
    before = json.loads((tmp_path / "checkpoint-10" / "rollout_state.json").read_text())
    assert before["prompt_index"] == 6382 + 14  # the bug this guards against
    _fix_checkpoint_prompt_index(trainer)
    trainer._save_checkpoint(None, None)
    after = json.loads((tmp_path / "checkpoint-10" / "rollout_state.json").read_text())
    assert after["prompt_index"] == 6382 + 2558 + 1  # highest trained group, then the next prompt
    assert after["groups_trained"] == len(trained) and after["model_version"] == before["model_version"]


def test_no_drops_is_unchanged(tmp_path):
    trainer = make(tmp_path, set(range(640)), start=0)
    _fix_checkpoint_prompt_index(trainer)
    trainer._save_checkpoint(None, None)
    state = json.loads((tmp_path / "checkpoint-10" / "rollout_state.json").read_text())
    assert state["prompt_index"] == 640


def test_no_trained_groups_leaves_the_file_alone(tmp_path):
    trainer = make(tmp_path, set(), start=5)
    _fix_checkpoint_prompt_index(trainer)
    trainer._save_checkpoint(None, None)
    assert json.loads((tmp_path / "checkpoint-10" / "rollout_state.json").read_text())["prompt_index"] == 5
