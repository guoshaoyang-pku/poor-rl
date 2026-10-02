from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

from rlforge.rl_trials import DQNTrialAdapter

BASIC = "01 基础指标"
FOCUS = "02 指定指标"
DETAIL = "03 细节指标"
PROJECTS = {"aiq": "AIQ", "suika": "Suika", "sft": "SFT"}

BASIC_ORDER = {
    "train/mean_reward_per_completion": 0,
    "online/mean_score": 0,
    "train/mean_completion_tokens": 1,
    "train/elapsed_seconds": 2,
    "train/elapsed_hours": 2,
    "train/step_wallclock_seconds": 3,
    "train/forward_backward_seconds": 4,
    "train/weight_sync_seconds": 5,
    "train/rollout_wait_seconds": 6,
    "train/loss": 1,
}
FOCUS_ORDER = {
    "validation/overall_accuracy_percent": 0,
    "validation/mcq_accuracy_percent": 1,
    "validation/ranking_accuracy_percent": 2,
    "gspo/sequence_clip_low_percent": 3,
    "gspo/sequence_clip_high_percent": 4,
    "evaluation/mean_score": 5,
}


def fraction_percent(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return round(float(value) * 100, 6)
    return None


def finite_metrics(values: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in values.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value)
    }


def configure_sections(run: Any) -> None:
    run.define_metric("detail/train_bin_*", x_axis="detail/train_elapsed_hours", section_name=DETAIL)
    run.define_metric("evaluation/*", x_axis="evaluation/elapsed_hours", section_name=FOCUS)
    run.define_metric("online/*", x_axis="online/elapsed_hours", section_name=BASIC)
    definitions = {
        "train/mean_reward_per_completion": BASIC,
        "train/mean_completion_tokens": BASIC,
        "train/step_wallclock_seconds": BASIC,
        "train/forward_backward_seconds": BASIC,
        "train/weight_sync_seconds": BASIC,
        "train/rollout_wait_seconds": BASIC,
        "train/elapsed_hours": BASIC,
        "train/loss": BASIC,
        "validation/*": FOCUS,
        "gspo/sequence_clip_low_percent": FOCUS,
        "gspo/sequence_clip_high_percent": FOCUS,
        "detail/*": DETAIL,
    }
    for key, section in definitions.items():
        run.define_metric(key, section_name=section)


def run_exists(log_dir: Path, project: str, *names: str) -> bool:
    database = log_dir / "runs.swanlab"
    if not database.exists() or not names:
        return False
    placeholders = ", ".join("?" for _ in names)
    connection = sqlite3.connect(database)
    try:
        row = connection.execute(
            "SELECT experiment.id FROM experiment JOIN project ON project.id = experiment.project_id "
            f"WHERE project.name = ? AND experiment.name IN ({placeholders}) LIMIT 1",
            (project, *names),
        ).fetchone()
        return row is not None
    finally:
        connection.close()


def hide_existing_runs(log_dir: Path, project: str, *names: str) -> int:
    database = log_dir / "runs.swanlab"
    if not database.exists() or not names:
        return 0
    placeholders = ", ".join("?" for _ in names)
    connection = sqlite3.connect(database)
    try:
        cursor = connection.execute(
            "UPDATE experiment SET show = 0 WHERE project_id = "
            "(SELECT id FROM project WHERE name = ?) "
            f"AND name IN ({placeholders}) AND show != 0",
            (project, *names),
        )
        connection.commit()
        return cursor.rowcount
    finally:
        connection.close()


def delete_run(log_dir: Path, project: str, name: str) -> bool:
    """Remove an experiment and every row that references it, plus its run directory."""
    database = log_dir / "runs.swanlab"
    if not database.exists():
        return False
    connection = sqlite3.connect(database)
    try:
        row = connection.execute(
            "SELECT experiment.id, experiment.run_id FROM experiment "
            "JOIN project ON project.id = experiment.project_id "
            "WHERE project.name = ? AND experiment.name = ?",
            (project, name),
        ).fetchone()
        if row is None:
            return False
        experiment_id, run_dir_name = row
        chart_ids = [
            r[0]
            for r in connection.execute(
                "SELECT id FROM chart WHERE experiment_id = ?", (experiment_id,)
            )
        ]
        if chart_ids:
            placeholders = ", ".join("?" for _ in chart_ids)
            connection.execute(f"DELETE FROM source WHERE chart_id IN ({placeholders})", chart_ids)
            connection.execute(f"DELETE FROM display WHERE chart_id IN ({placeholders})", chart_ids)
        connection.execute("DELETE FROM chart WHERE experiment_id = ?", (experiment_id,))
        connection.execute("DELETE FROM tag WHERE experiment_id = ?", (experiment_id,))
        connection.execute("DELETE FROM namespace WHERE experiment_id = ?", (experiment_id,))
        connection.execute("DELETE FROM experiment WHERE id = ?", (experiment_id,))
        connection.commit()
    finally:
        connection.close()
    if run_dir_name:
        run_dir = log_dir / run_dir_name
        if run_dir.is_dir():
            shutil.rmtree(run_dir, ignore_errors=True)
    return True


def emit_run(*, log_dir: Path, project: str, name: str, group: str,
             description: str, config: dict[str, Any], tags: list[str],
             metric_rows: list[tuple[int, dict[str, Any]]], replace: bool = False,
             live: bool = False) -> bool:
    if run_exists(log_dir, project, name):
        if not replace:
            return False
        delete_run(log_dir, project, name)
    import swanlab

    run = swanlab.init(
        mode="local", log_dir=str(log_dir), project=project, name=name,
        group=group, description=description, config=config, tags=tags,
        reinit=True,
    )
    configure_sections(run)
    by_step: dict[int, dict[str, Any]] = {}
    for step, values in metric_rows:
        by_step.setdefault(step, {}).update(values)
    for step, values in sorted(by_step.items()):
        metrics = finite_metrics(values)
        if metrics:
            swanlab.log(metrics, step=step)
    if not live:
        run.finish()
    return True


def aiq_rows(record: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    rows = []
    elapsed_seconds = 0.0
    step_lengths = []
    for row in record.get("steps", []):
        step = int(row.get("step", len(rows) + 1))
        step_seconds = row.get("perf/step_s", row.get("step_s"))
        if isinstance(step_seconds, (int, float)) and math.isfinite(step_seconds):
            elapsed_seconds += float(step_seconds)
        completion_tokens = row.get("completions/mean_length", row.get("meanlen"))
        values = {
            "train/mean_reward_per_completion": row.get("reward"),
            "train/mean_completion_tokens": completion_tokens,
            "train/step_wallclock_seconds": row.get("perf/step_s", row.get("step_s")),
            "train/forward_backward_seconds": row.get("perf/fwd_bwd_s"),
            "train/weight_sync_seconds": row.get("perf/weight_sync_s"),
            "train/rollout_wait_seconds": row.get("perf/rollout_wait_s"),
            "train/elapsed_hours": elapsed_seconds / 3600,
            "train/learning_rate": row.get("learning_rate", row.get("lr")),
            "train/policy_loss": row.get("loss"),
            "train/entropy_nats": row.get("entropy"),
            "train/approx_kl": row.get("kl"),
            "train/truncation_percent": fraction_percent(row.get("completions/clipped_ratio", row.get("trunc"))),
            "gspo/sequence_clip_low_percent": fraction_percent(row.get("gspo/seq_clip_low_frac", row.get("seq_clip_low"))),
            "gspo/sequence_clip_high_percent": fraction_percent(row.get("gspo/seq_clip_high_frac", row.get("seq_clip_high"))),
            "gspo/sequence_ratio_mean": row.get("gspo/rho_mean"),
            "detail/min_completion_tokens": row.get("completions/min_length"),
            "detail/max_completion_tokens": row.get("completions/max_length"),
        }
        rows.append((step, values))
        if isinstance(completion_tokens, (int, float)) and math.isfinite(completion_tokens):
            step_lengths.append((elapsed_seconds / 3600, float(completion_tokens)))

    previous_hour = 0.0
    for index, time_bin in enumerate(record.get("task_bins", []), 1):
        bin_metrics = {
            "mcq_reward": time_bin.get("mcq_reward"),
            "mcq_accuracy_percent": fraction_percent(time_bin.get("mcq_correct")),
            "mcq_truncation_percent": fraction_percent(time_bin.get("mcq_trunc")),
            "ranking_reward": time_bin.get("rank_reward"),
            "ranking_exact_accuracy_percent": fraction_percent(time_bin.get("rank_exact")),
            "ranking_partial_score": time_bin.get("rank_score"),
            "ranking_truncation_percent": fraction_percent(time_bin.get("rank_trunc")),
        }
        hour = float(time_bin.get("hour", 0.0) or 0.0)
        interval_lengths = [
            length for elapsed, length in step_lengths
            if (elapsed <= hour if previous_hour == 0 else previous_hour < elapsed <= hour)
        ]
        values = {
            "detail/train_elapsed_hours": hour,
            "detail/train_bin_mean_completion_tokens": (
                sum(interval_lengths) / len(interval_lengths) if interval_lengths else None
            ),
        }
        previous_hour = hour
        values.update({f"detail/train_bin_{key}": value for key, value in bin_metrics.items()})
        rows.append((index, values))

    for index, evaluation in enumerate(record.get("evals", []), 1):
        if evaluation.get("error"):
            continue
        step = int(evaluation.get("step", evaluation.get("ckpt", index)))
        overall = evaluation.get("overall", {})
        summary = evaluation.get("summary", {})
        if not overall:
            overall = summary.get("overall", summary)
        if "accuracy" not in overall and "accuracy_overall" in overall:
            overall = {**overall, "accuracy": overall.get("accuracy_overall")}
        by_task = evaluation.get("by_task", {})
        if not by_task:
            by_task = summary.get("by_task", {})
        values = {
            "validation/overall_accuracy_percent": fraction_percent(overall.get("accuracy")),
            "validation/reward_per_completion": overall.get("mean_reward"),
            "validation/truncation_percent": fraction_percent(overall.get("truncation_rate")),
            "validation/mean_completion_tokens": overall.get("mean_completion_tokens"),
            "validation/parse_rate_percent": fraction_percent(overall.get("parse_rate")),
            "validation/mcq_accuracy_percent": fraction_percent(by_task.get("mcq", {}).get("accuracy")),
            "validation/ranking_accuracy_percent": fraction_percent(by_task.get("ranking", {}).get("accuracy")),
            "detail/mcq_mean_reward": by_task.get("mcq", {}).get("mean_reward"),
            "detail/ranking_mean_reward": by_task.get("ranking", {}).get("mean_reward"),
        }
        rows.append((step, values))
    return rows


def import_aiq(data: dict[str, Any], log_dir: Path, replace: bool = False) -> int:
    count = 0
    for run_name, record in data.get("aiq", {}).items():
        metadata = record.get("run", {})
        hyperparams = metadata.get("hyperparams", {})
        dataset = metadata.get("data", {})
        status = metadata.get("stop_reason", "historical")
        description = (
            f"Historical AIQ {metadata.get('mode', 'run')}; "
            f"steps={len(record.get('steps', []))}; data={dataset.get('rows', 'unknown')} rows; "
            f"stop={status}. Reward is a dimensionless task score, not accuracy."
        )
        if metadata.get("note"):
            description += " " + metadata["note"]
        config = {
            "run_id": metadata.get("run_id", run_name),
            "status": status,
            "node": metadata.get("node"),
            "start_time": metadata.get("started"),
            "end_time": metadata.get("ended"),
            "training_rows": dataset.get("rows"),
            "training_data_md5": dataset.get("md5"),
            "training_tasks": dataset.get("tasks"),
            "model": metadata.get("model", {}).get("path"),
            "algorithm": "GSPO" if hyperparams.get("gspo") else "GRPO",
            "hyperparameters": hyperparams,
            "metric_definitions": {
                "train/mean_reward_per_completion": "Mean task reward per generated completion; dimensionless score on this run's reward scale, not accuracy.",
                "train/mean_completion_tokens": "Mean output length per completion, in generated tokens.",
                "train/step_wallclock_seconds": "Wall-clock seconds for one optimizer step.",
                "train/elapsed_hours": "Elapsed wall-clock hours since the beginning of this training run.",
                "gspo/sequence_clip_low_percent": "Percent of completion sequences below the GSPO lower sequence importance-ratio boundary; sequence-level, not token-level.",
                "gspo/sequence_clip_high_percent": "Percent of completion sequences above the GSPO upper sequence importance-ratio boundary; sequence-level, not token-level.",
                "validation/overall_accuracy_percent": "Exact held-out answer accuracy in percent; does not include truncation reward penalties.",
                "detail/train_bin_mcq_accuracy_percent": "MCQ exact accuracy in percent during this elapsed-time interval.",
                "detail/train_bin_ranking_exact_accuracy_percent": "Ranking exact accuracy in percent during this elapsed-time interval.",
                "detail/train_bin_mean_completion_tokens": "Mean completion length in generated tokens during this elapsed-time interval.",
            },
        }
        groups = {
            "async_dp_v3full": "Full pool · 3,830 train / 300 held-out",
            "async_dp_flip450": "Data-flip · 450 train / 50 paired held-out",
            "async_dp_pool2134cap8k": "Historical pool · 8k prompt-cap run",
            "async_dp_opus_corr_e178_userformula_step10_20261001": "Opus e178 · user-formula pool",
            "async_dp_opus_corr_e178_userformula_step200_4plus4cps256_20261001": "Opus e178 · user-formula pool",
            "async_dp_opus_corr_e178_userformula_step400_from_ckpt200_4plus4cps256_20261002": "Opus e178 · user-formula pool",
        }
        emitted = emit_run(
            log_dir=log_dir, project="AIQ", name=run_name,
            group=groups.get(run_name, "Historical AIQ"),
            description=description, config=config,
            tags=["aiq", "historical", str(config["algorithm"]).lower(), str(status)],
            metric_rows=aiq_rows(record), replace=replace,
        )
        count += int(emitted)
    return count


def import_suika(root: Path, log_dir: Path, minimum_wall_hours: float,
                 max_points: int) -> tuple[int, list[str], list[str], list[str]]:
    adapter = DQNTrialAdapter(root)
    imported, excluded, preserved, hidden = 0, [], [], []
    for trial in adapter.list_trials():
        detail = adapter.get_trial(trial.id) or {}
        metrics = detail.get("training", {})
        evals = detail.get("evaluation", {})
        online = detail.get("online", {})
        duration = trial.summary.get("wall_h")
        if duration is None:
            elapsed = [value for value in metrics.get("t", []) if isinstance(value, (int, float))]
            duration = max(elapsed, default=0) / 3600
        if duration < minimum_wall_hours:
            excluded.append(f"{trial.name} ({duration:.2f}h)")
            legacy_name = (
                f"{trial.config.get('host', 'unknown')}__"
                f"{trial.experiment or 'historical'}__{trial.name}"
            )
            if run_exists(log_dir, "Suika", legacy_name):
                if hide_existing_runs(log_dir, "Suika", legacy_name):
                    hidden.append(trial.name)
            continue

        legacy_name = (
            f"{trial.config.get('host', 'unknown')}__"
            f"{trial.experiment or 'historical'}__{trial.name}"
        )
        if run_exists(log_dir, "Suika", trial.name, legacy_name):
            preserved.append(trial.name)
            continue

        total_points = max(
            (len(values) for series in (metrics, evals, online) for values in series.values()),
            default=0,
        )
        stride = max(1, math.ceil(total_points / max_points))
        rows_by_step: dict[int, dict[str, Any]] = {}
        for key, values in metrics.items():
            for index, value in enumerate(values):
                if value is None or (index % stride != 0 and index != len(values) - 1):
                    continue
                if key == "t":
                    rows_by_step.setdefault(index, {})["train/elapsed_hours"] = value / 3600
                elif key == "loss":
                    rows_by_step.setdefault(index, {})["detail/loss"] = value
                elif key == "lr":
                    rows_by_step.setdefault(index, {})["detail/learning_rate"] = value
                elif key == "grad_norm":
                    rows_by_step.setdefault(index, {})["detail/grad_norm"] = value
                elif key == "q_mean":
                    rows_by_step.setdefault(index, {})["detail/q_mean"] = value
                elif key == "env_sps":
                    rows_by_step.setdefault(index, {})["detail/environment_steps_per_second"] = value
                elif key == "grad_sps":
                    rows_by_step.setdefault(index, {})["detail/gradient_steps_per_second"] = value
                else:
                    rows_by_step.setdefault(index, {})[f"detail/{key}"] = value
        for key, values in evals.items():
            if key == "time":
                for index, value in enumerate(values):
                    if value is not None and (index % stride == 0 or index == len(values) - 1):
                        rows_by_step.setdefault(index, {})["evaluation/elapsed_hours"] = max(0.0, (value - values[0]) / 3600)
                continue
            for index, value in enumerate(values):
                if value is not None and (index % stride == 0 or index == len(values) - 1):
                    metric_name = "mean_score" if key == "mean" else f"{key}_score" if key in {"p25", "median", "max", "p2000", "p75", "p90"} else key
                    rows_by_step.setdefault(index, {})[f"evaluation/{metric_name}"] = value
        for key, values in online.items():
            if key == "time_hours":
                for index, value in enumerate(values):
                    if value is not None and (index % stride == 0 or index == len(values) - 1):
                        rows_by_step.setdefault(index, {})["online/elapsed_hours"] = value
                continue
            for index, value in enumerate(values):
                if value is not None and (index % stride == 0 or index == len(values) - 1):
                    metric_name = "mean_score" if key == "mean" else f"{key}_score" if key in {"p25", "median", "max", "p75", "p90"} else key
                    rows_by_step.setdefault(index, {})[f"online/{metric_name}"] = value

        source_id = trial.id.split("::", 1)[-1]
        config = {
            **trial.config,
            "source_trial_id": trial.id,
            "host": trial.config.get("host"),
            "wave": trial.experiment,
            "wall_time_hours": duration,
            "final_grad_steps": trial.summary.get("grad_steps"),
            "best_eval_score": trial.summary.get("eval_best"),
            "last_eval_score": trial.summary.get("eval_last"),
            "last_online_score": trial.summary.get("online_last"),
            "metric_definitions": {
                "online/mean_score": "Mean game score from online episodes; game-score units, higher is better.",
                "evaluation/mean_score": "Mean game score from periodic held-out evaluation; game-score units, higher is better.",
                "train/elapsed_hours": "Elapsed wall-clock hours since the beginning of this training run.",
                "detail/loss": "Training objective loss; dimensionless optimization scalar, not game score.",
                "detail/environment_steps_per_second": "Environment transitions processed per second.",
                "detail/gradient_steps_per_second": "Optimizer gradient steps per second.",
            },
        }
        rows = [(step, values) for step, values in sorted(rows_by_step.items())]
        emitted = emit_run(
            log_dir=log_dir, project="Suika", name=trial.name,
            group=trial.experiment or "Historical Suika", description=f"{trial.algorithm} · {source_id} · {duration:.2f} h",
            config=config, tags=["suika", "historical", str(trial.algorithm).lower(), str(trial.config.get("wave", ""))],
            metric_rows=rows, live=trial.status == "live",
        )
        imported += int(emitted)
    return imported, excluded, preserved, hidden


def sft_rows(record: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    steps: dict[int, dict[str, Any]] = {}
    for row in record.get("trainer_state", {}).get("log_history", []):
        step = row.get("step")
        if not isinstance(step, int):
            continue
        values = {
            "train/loss": row.get("loss"),
            "train/learning_rate": row.get("learning_rate"),
            "detail/grad_norm": row.get("grad_norm"),
        }
        steps.setdefault(step, {}).update(values)
    for evaluation in record.get("evals", []):
        step = int(evaluation.get("step", 0))
        overall = evaluation.get("overall", {})
        by_task = evaluation.get("by_task", {})
        values = {
            "validation/overall_accuracy_percent": fraction_percent(overall.get("accuracy")),
            "validation/reward_per_completion": overall.get("mean_reward"),
            "validation/parse_rate_percent": fraction_percent(overall.get("parse_rate")),
            "validation/truncation_percent": fraction_percent(overall.get("truncation_rate")),
            "validation/mean_completion_tokens": overall.get("mean_completion_tokens"),
            "validation/mcq_accuracy_percent": fraction_percent(by_task.get("mcq", {}).get("accuracy")),
            "validation/ranking_accuracy_percent": fraction_percent(by_task.get("ranking", {}).get("accuracy")),
        }
        steps.setdefault(step, {}).update(values)
    return sorted(steps.items())


def import_sft(data: dict[str, Any], log_dir: Path) -> int:
    count = 0
    for name, record in data.get("sft", {}).items():
        manifest = record.get("manifest", {})
        train = manifest.get("train_metrics", {})
        status = manifest.get("status", manifest.get("stop_reason", "unknown"))
        config = {
            "run_id": name,
            "status": status,
            "base_model": manifest.get("base_model"),
            "training_rows": manifest.get("train_rows"),
            "epochs": manifest.get("epochs"),
            "learning_rate": manifest.get("learning_rate"),
            "bf16": manifest.get("bf16"),
            "max_sequence_tokens": manifest.get("max_length"),
            "train_loss": train.get("train_loss"),
            "train_runtime_seconds": train.get("train_runtime"),
            "final_checkpoint": manifest.get("final_checkpoint"),
            "metric_definitions": {
                "train/loss": "Supervised fine-tuning cross-entropy loss (dimensionless).",
                "validation/overall_accuracy_percent": "Exact correctness as percent (0–100) on held-out eval, not reward.",
                "validation/reward_per_completion": "Mean task reward per generated completion; reward score, not accuracy percentage.",
                "validation/mean_completion_tokens": "Mean generated completion length, in tokens.",
            },
        }
        emitted = emit_run(
            log_dir=log_dir, project="SFT", name=name, group="Private-thinking SFT",
            description=f"SFT {manifest.get('train_rows')} examples × {manifest.get('epochs')} epochs; status={status}; eval checkpoints 89/178/267.",
            config=config, tags=["sft", "historical", str(status)], metric_rows=sft_rows(record),
        )
        count += int(emitted)
    return count


def relabel_legacy_suika_metrics(connection: sqlite3.Connection, project_id: int) -> None:
    metric_names = {
        "train/t": "train/elapsed_seconds",
        "train/loss": "detail/loss",
        "train/lr": "detail/learning_rate",
        "train/q_mean": "detail/mean_q_value",
        "train/env_sps": "detail/environment_steps_per_second",
        "train/grad_sps": "detail/gradient_steps_per_second",
        "eval/mean": "evaluation/mean_score",
        "eval/p25": "evaluation/p25_score",
        "eval/median": "evaluation/median_score",
        "eval/max": "evaluation/max_score",
        "eval/p2000": "evaluation/p2000_score",
        "eval/moves_mean": "evaluation/mean_moves_per_episode",
        "eval/maxfruit_max": "evaluation/max_fruit_count",
        "eval/env_steps": "evaluation/environment_steps",
        "eval/grad_steps": "evaluation/gradient_steps",
        "eval/time": "evaluation/unix_timestamp_seconds",
        "online/mean": "online/mean_score",
        "online/p25": "online/p25_score",
        "online/p75": "online/p75_score",
        "online/p90": "online/p90_score",
        "online/max": "online/max_score",
        "online/median": "online/median_score",
        "online/moves_mean": "online/mean_moves_per_episode",
        "online/eps_mean": "online/mean_epsilon",
        "online/time_hours": "online/elapsed_hours",
        "online/actors": "online/actor_count",
        "online/episodes": "online/episode_count",
        "online/env_steps": "online/environment_steps",
    }
    for old_name, new_name in metric_names.items():
        connection.execute(
            "UPDATE chart SET name = ? WHERE name = ? AND "
            "(project_id = ? OR experiment_id IN (SELECT id FROM experiment WHERE project_id = ?))",
            (new_name, old_name, project_id, project_id),
        )


def configure_local_sections(log_dir: Path) -> None:
    database = log_dir / "runs.swanlab"
    if not database.exists():
        return
    connection = sqlite3.connect(database)
    try:
        suika_projects = connection.execute("SELECT id FROM project WHERE name = 'Suika'").fetchall()
        for (project_id,) in suika_projects:
            relabel_legacy_suika_metrics(connection, project_id)
        scopes = connection.execute(
            "SELECT id, experiment_id, project_id FROM namespace"
        ).fetchall()
        scope_keys = {(experiment_id, project_id) for _, experiment_id, project_id in scopes}
        for experiment_id, project_id in scope_keys:
            namespaces = connection.execute(
                "SELECT id, name FROM namespace WHERE experiment_id IS ? AND project_id IS ? ORDER BY sort, id",
                (experiment_id, project_id),
            ).fetchall()
            section_ids = {}
            for index, section in enumerate((BASIC, FOCUS, DETAIL)):
                opened = 0 if index == 2 else 1
                description = {
                    BASIC: "Reward is the primary scalar task score (not accuracy); token lengths and timing use explicit units.",
                    FOCUS: "Held-out accuracy is in percent; GSPO low/high are sequence-level clipped fractions in percent.",
                    DETAIL: "Lower-priority per-bin task metrics and detailed diagnostics.",
                }[section]
                if index < len(namespaces):
                    namespace_id = namespaces[index][0]
                    connection.execute(
                        "UPDATE namespace SET name = ?, description = ?, sort = ?, opened = ? WHERE id = ?",
                        (section, description, index, opened, namespace_id),
                    )
                else:
                    timestamp = datetime.now(timezone.utc).isoformat()
                    cursor = connection.execute(
                        "INSERT INTO namespace (experiment_id, project_id, name, description, sort, opened, create_time, update_time) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (experiment_id, project_id, section, description, index, opened, timestamp, timestamp),
                    )
                    namespace_id = cursor.lastrowid
                section_ids[section] = namespace_id

            chart_scope = "experiment_id IS ? AND project_id IS ?"
            charts = connection.execute(
                f"SELECT id, name FROM chart WHERE {chart_scope}", (experiment_id, project_id)
            ).fetchall()
            for chart_id, name in charts:
                if name in BASIC_ORDER:
                    section, display_sort = BASIC, BASIC_ORDER[name]
                elif name in FOCUS_ORDER:
                    section, display_sort = FOCUS, FOCUS_ORDER[name]
                elif name.startswith(("validation/", "evaluation/", "gspo/")):
                    section, display_sort = FOCUS, 100
                else:
                    section, display_sort = DETAIL, 100
                display = connection.execute(
                    "SELECT id FROM display WHERE chart_id = ? ORDER BY id LIMIT 1", (chart_id,)
                ).fetchone()
                if display:
                    connection.execute(
                        "UPDATE display SET namespace_id = ?, sort = ? WHERE id = ?",
                        (section_ids[section], display_sort, display[0]),
                    )
                else:
                    timestamp = datetime.now(timezone.utc).isoformat()
                    connection.execute(
                        "INSERT INTO display (chart_id, namespace_id, sort, create_time, update_time) VALUES (?, ?, ?, ?, ?)",
                        (chart_id, section_ids[section], display_sort, timestamp, timestamp),
                    )
        connection.commit()
    finally:
        connection.close()


def mark_live_runs(suika_root: Path, log_dir: Path) -> int:
    """Post-process hook: swanlab's atexit finishes every open run when the
    import process exits, so live trials must be flipped back to RUNNING in a
    separate process after the import completes."""
    import sqlite3

    db = log_dir / "runs.swanlab"
    if not db.is_file():
        return 0
    adapter = DQNTrialAdapter(suika_root)
    live_names = [t.name for t in adapter.list_trials() if t.status == "live"]
    if not live_names:
        return 0
    conn = sqlite3.connect(db)
    try:
        placeholders = ", ".join("?" for _ in live_names)
        conn.execute(f"UPDATE experiment SET status = 0 WHERE name IN ({placeholders})", live_names)
        conn.commit()
    finally:
        conn.close()
    return len(live_names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-json", type=Path, required=True)
    parser.add_argument("--suika-root", type=Path, default=Path("examples/suika_trials"))
    parser.add_argument("--log-dir", type=Path, default=Path("artifacts/swanlab"))
    parser.add_argument("--minimum-suika-hours", type=float, default=4.0)
    parser.add_argument("--max-suika-points", type=int, default=800)
    parser.add_argument("--only", choices=("all", "aiq", "suika", "sft"), default="all")
    parser.add_argument("--replace", action="store_true",
                        help="re-import existing runs (AIQ only): delete and rebuild instead of skipping")
    parser.add_argument("--mark-live", action="store_true",
                        help="suika only: flip trials with fresh metrics back to RUNNING in the swanlab db "
                             "(run in a separate process after the import, swanlab atexit finishes open runs)")
    args = parser.parse_args()
    if args.minimum_suika_hours < 0 or args.max_suika_points < 1:
        parser.error("--minimum-suika-hours must be non-negative and --max-suika-points must be positive")

    if args.only == "all":
        results = {}
        for category in ("aiq", "suika", "sft"):
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--history-json", str(args.history_json),
                "--suika-root", str(args.suika_root),
                "--log-dir", str(args.log_dir),
                "--minimum-suika-hours", str(args.minimum_suika_hours),
                "--max-suika-points", str(args.max_suika_points),
                "--only", category,
            ]
            if args.replace:
                command.append("--replace")
            subprocess.run(command, check=True, env={
                **os.environ,
                "SWANLAB_MODE": "local",
                "SWANLAB_LOGDIR": str(args.log_dir / PROJECTS[category] / "swanlog"),
                "SWANLAB_PROJ_NAME": PROJECTS[category],
            })
            results[PROJECTS[category]] = str(args.log_dir / PROJECTS[category] / "swanlog")
        print(json.dumps({"projects": results}, ensure_ascii=False, indent=2))
        return

    data = json.loads(args.history_json.read_text())
    log_dir = args.log_dir / PROJECTS[args.only] / "swanlog"
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SWANLAB_MODE"] = "local"
    os.environ["SWANLAB_LOGDIR"] = str(log_dir)
    os.environ["SWANLAB_PROJ_NAME"] = PROJECTS[args.only]
    if args.only == "aiq":
        count = import_aiq(data, log_dir, replace=args.replace)
        summary = {"project": "AIQ", "runs": count, "log_dir": str(log_dir)}
    elif args.only == "suika":
        if args.mark_live:
            mark_live_runs(args.suika_root, log_dir)
            print(json.dumps({"project": "Suika", "mark_live": True, "log_dir": str(log_dir)}, ensure_ascii=False))
            return
        count, excluded, preserved, hidden = import_suika(args.suika_root, log_dir, args.minimum_suika_hours, args.max_suika_points)
        summary = {
            "project": "Suika", "imported_runs": count, "preserved_existing_runs": len(preserved),
            "hidden_short_existing_runs": hidden, "excluded_short_runs": excluded,
            "log_dir": str(log_dir),
        }
    else:
        count = import_sft(data, log_dir)
        summary = {"project": "SFT", "runs": count, "log_dir": str(log_dir)}
    configure_local_sections(log_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
