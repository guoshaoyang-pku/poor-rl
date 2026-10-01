from types import SimpleNamespace

from rlforge.dashboard import RunSpec, State


def test_completed_run_uses_training_progress_not_checkpoint_shard_progress(tmp_path):
    trainer_log = tmp_path / "trainer.log"
    trainer_log.write_text(
        "100%|██████████| 500/500 [3:14:34<00:00, 14.27s/it]\n"
        "Writing model shards: 100%|██████████| 1/1 [00:03<00:00, 3.04s/it]\n"
    )
    state = State(SimpleNamespace(root=None, registry=None, min_steps=50, run=None,
                                  trainer_log=None, eval_history=None, base_eval=None))
    spec = RunSpec("flip450", tmp_path, trainer_log=trainer_log)

    assert state._progress(spec) == (500, 500)


def test_in_progress_run_ignores_interleaved_checkpoint_progress(tmp_path):
    trainer_log = tmp_path / "trainer.log"
    trainer_log.write_text(
        " 75%|███████▌  | 375/500 [2:40<00:53, 14.27s/it]\n"
        "Writing model shards: 100%|██████████| 1/1 [00:03<00:00, 3.04s/it]\n"
        " 76%|███████▌  | 380/500 [2:42<00:51, 14.27s/it]\n"
    )
    state = State(SimpleNamespace(root=None, registry=None, min_steps=50, run=None,
                                  trainer_log=None, eval_history=None, base_eval=None))
    spec = RunSpec("flip450", tmp_path, trainer_log=trainer_log)

    assert state._progress(spec) == (380, 500)
