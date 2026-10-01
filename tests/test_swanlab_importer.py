import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


sys.modules.setdefault("swanlab", SimpleNamespace(define_metric=lambda *args, **kwargs: None))
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "import_experiments_swanlab.py"
SPEC = importlib.util.spec_from_file_location("import_experiments_swanlab", SCRIPT)
IMPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMPORTER)


def test_aiq_rows_expose_reward_length_step_time_and_both_gspo_clip_sides():
    rows = IMPORTER.aiq_rows({
        "steps": [{
            "step": 3,
            "reward": 0.42,
            "completions/mean_length": 128,
            "perf/step_s": 11.5,
            "gspo/seq_clip_low_frac": 0.31,
            "gspo/seq_clip_high_frac": 0.12,
        }],
        "evals": [{
            "step": 50,
            "overall": {"accuracy": 0.71, "mean_reward": 0.66},
            "by_task": {"mcq": {"accuracy": 0.62}, "ranking": {"accuracy": 0.89}},
        }],
        "task_bins": [{"hour": 0.5, "mcq_correct": 0.6, "mcq_trunc": 0.1,
                       "rank_exact": 0.8, "rank_trunc": 0.05}],
    })

    first, detail, validation = rows
    assert first[0] == 3
    assert first[1]["train/mean_reward_per_completion"] == 0.42
    assert first[1]["train/mean_completion_tokens"] == 128
    assert first[1]["train/step_wallclock_seconds"] == 11.5
    assert first[1]["gspo/sequence_clip_low_percent"] == 31
    assert first[1]["gspo/sequence_clip_high_percent"] == 12
    assert validation[1]["validation/overall_accuracy_percent"] == 71
    assert validation[1]["validation/mcq_accuracy_percent"] == 62
    assert validation[1]["validation/ranking_accuracy_percent"] == 89
    assert detail[1]["detail/train_bin_mcq_accuracy_percent"] == 60
    assert detail[1]["detail/train_bin_mcq_truncation_percent"] == 10
    assert detail[1]["detail/train_bin_ranking_exact_accuracy_percent"] == 80
    assert detail[1]["detail/train_bin_ranking_truncation_percent"] == 5
    assert detail[1]["detail/train_bin_mean_completion_tokens"] == 128


def test_finite_metrics_omits_nan_infinity_and_booleans():
    assert IMPORTER.finite_metrics({"finite": 0.25, "nan": float("nan"),
                                   "infinity": float("inf"), "flag": True}) == {"finite": 0.25}


def test_configure_sections_prioritizes_reward_and_keeps_clip_sides_in_focus():
    definitions = []

    class Run:
        def define_metric(self, name, **kwargs):
            definitions.append((name, kwargs))

    IMPORTER.configure_sections(Run())
    sections = {name: options.get("section_name") for name, options in definitions}
    assert sections["train/mean_reward_per_completion"] == IMPORTER.BASIC
    assert sections["train/mean_completion_tokens"] == IMPORTER.BASIC
    assert sections["train/step_wallclock_seconds"] == IMPORTER.BASIC
    assert sections["validation/*"] == IMPORTER.FOCUS
    assert sections["gspo/sequence_clip_low_percent"] == IMPORTER.FOCUS
    assert sections["gspo/sequence_clip_high_percent"] == IMPORTER.FOCUS
    assert sections["detail/*"] == IMPORTER.DETAIL


def test_run_exists_is_scoped_to_project(tmp_path):
    import sqlite3

    database = tmp_path / "runs.swanlab"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE project (id INTEGER PRIMARY KEY, name TEXT);"
            "CREATE TABLE experiment (id INTEGER PRIMARY KEY, project_id INTEGER, name TEXT);"
            "INSERT INTO project VALUES (1, 'Suika');"
            "INSERT INTO project VALUES (2, 'AIQ');"
            "INSERT INTO experiment VALUES (1, 1, 'w1_trial');"
            "INSERT INTO experiment VALUES (2, 2, 'w1_trial');"
        )

    assert IMPORTER.run_exists(tmp_path, "Suika", "w1_trial")
    assert not IMPORTER.run_exists(tmp_path, "SFT", "w1_trial")


def test_short_suika_runs_are_hidden_without_deleting_records(tmp_path):
    import sqlite3

    database = tmp_path / "runs.swanlab"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE project (id INTEGER PRIMARY KEY, name TEXT);"
            "CREATE TABLE experiment (id INTEGER PRIMARY KEY, project_id INTEGER, name TEXT, show INTEGER);"
            "INSERT INTO project VALUES (1, 'Suika');"
            "INSERT INTO experiment VALUES (1, 1, 'short_run', 1);"
        )

    assert IMPORTER.hide_existing_runs(tmp_path, "Suika", "short_run") == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT name, show FROM experiment").fetchall() == [("short_run", 0)]


def test_sft_rows_pair_loss_and_validation_by_checkpoint():
    rows = IMPORTER.sft_rows({
        "trainer_state": {"log_history": [{"step": 10, "loss": 1.2, "learning_rate": 0.0001}]},
        "evals": [{"step": 10, "overall": {"accuracy": 0.58, "mean_reward": 0.51},
                   "by_task": {"mcq": {"accuracy": 0.55}, "ranking": {"accuracy": 0.64}}}],
    })

    assert len(rows) == 1
    assert rows[0][0] == 10
    assert rows[0][1]["train/loss"] == 1.2
    assert rows[0][1]["validation/overall_accuracy_percent"] == 58
    assert rows[0][1]["validation/reward_per_completion"] == 0.51
