from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        pass
    return rows


def _mtime(path: Path | None) -> int:
    try:
        return path.stat().st_mtime_ns if path else -1
    except OSError:
        return -1


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


@dataclass(frozen=True)
class TrialRecord:
    id: str
    adapter: str
    algorithm: str
    name: str
    experiment: str | None
    status: str
    config: dict[str, Any]
    summary: dict[str, Any]


class TrialAdapter(Protocol):
    name: str

    def list_trials(self) -> list[TrialRecord]: ...

    def list_overview(self) -> list[dict[str, Any]]: ...

    def get_trial(self, trial_id: str) -> dict[str, Any] | None: ...


class TrialRegistry:
    def __init__(self):
        self._adapters: dict[str, TrialAdapter] = {}

    def register(self, adapter: TrialAdapter) -> None:
        if adapter.name in self._adapters:
            raise ValueError(f"trial adapter {adapter.name!r} is already registered")
        self._adapters[adapter.name] = adapter

    def list_trials(self) -> list[TrialRecord]:
        records = [record for adapter in self._adapters.values()
                   for record in adapter.list_trials()]
        records.sort(key=lambda record: (record.status == "live", record.name), reverse=True)
        return records

    def list_overview(self) -> list[dict[str, Any]]:
        overview = []
        for adapter in self._adapters.values():
            for record in adapter.list_trials():
                detail = adapter.get_trial(record.id) or {}
                overview.append({
                    "trial": record.__dict__,
                    "training": detail.get("training", {}),
                    "evaluation": detail.get("evaluation", {}),
                    "online": detail.get("online", {}),
                })
        overview.sort(key=lambda row: (
            row["trial"]["status"] == "live",
            row["trial"]["name"],
        ), reverse=True)
        return overview

    def get_trial(self, trial_id: str) -> dict[str, Any] | None:
        adapter_name, separator, _ = trial_id.partition("::")
        if not separator:
            return None
        adapter = self._adapters.get(adapter_name)
        return adapter.get_trial(trial_id) if adapter else None


class DQNTrialAdapter:
    name = "suika-dqn"
    _METRIC_FIELDS = ("t", "loss", "q_mean", "grad_norm", "lr", "env_sps", "grad_sps")
    _EVAL_FIELDS = ("mean", "p25", "median", "max", "p2000", "moves_mean",
                    "maxfruit_max", "grad_steps", "time")
    _ONLINE_FIELDS = ("time_hours", "env_steps", "mean", "p25", "median", "p75", "p90",
                      "max", "eps_mean", "moves_mean", "episodes", "actors")

    def __init__(self, root: str | Path, ledger: str | Path | None = None):
        self.root = Path(root).expanduser().resolve()
        self.ledger = Path(ledger).expanduser().resolve() if ledger else self.root / "trials.json"
        self._cache_key: tuple[Any, ...] | None = None
        self._records: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _key(host: str, wave: str, name: str) -> str:
        return f"{host}/{wave}/{name}"

    def _load(self) -> None:
        ledger_stamp = _mtime(self.ledger)
        candidates = sorted(self.root.glob("*/*/*/metrics.jsonl"))
        file_stamps = tuple((str(path), _mtime(path), _mtime(path.with_name("eval.jsonl")),
                             _mtime(path.with_name("online_score.jsonl")))
                            for path in candidates)
        cache_key = (ledger_stamp, _mtime(self.root / "configs.json"), file_stamps)
        if cache_key == self._cache_key:
            return

        ledger = _read_json(self.ledger)
        rows = ledger if isinstance(ledger, list) else []
        cfg_data = _read_json(self.root / "configs.json")
        configs = cfg_data if isinstance(cfg_data, dict) else {}
        merged: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict) or not row.get("trial"):
                continue
            host = str(row.get("host") or "unknown")
            wave = str(row.get("wave") or "historical")
            config = dict(configs.get(str(row["trial"]), {}))
            config.update({key: value for key, value in row.items()
                           if key not in {"n_metrics", "n_evals", "env_steps", "grad_steps",
                                          "env_sps", "grad_sps", "q_mean", "loss", "grad_norm",
                                          "eval_first", "eval_last", "eval_best",
                                          "eval_last_env_steps", "moves_last", "maxfruit_max",
                                          "eval_best_env_steps", "wall_h"} and value is not None})
            merged[self._key(host, wave, str(row["trial"]))] = {
                "host": host, "wave": wave, "name": str(row["trial"]),
                "config": config,
                "legacy_summary": row,
                "metrics": [], "evals": [], "online": [], "metrics_mtime": -1,
                "eval_mtime": -1, "online_mtime": -1,
            }

        for metrics_path in candidates:
            trial_dir = metrics_path.parent
            host, wave, name = trial_dir.parent.parent.name, trial_dir.parent.name, trial_dir.name
            key = self._key(host, wave, name)
            row = merged.setdefault(key, {
                "host": host, "wave": wave, "name": name,
                "config": dict(configs.get(name, {})),
                "legacy_summary": {}, "metrics": [], "evals": [], "online": [],
                "metrics_mtime": -1, "eval_mtime": -1, "online_mtime": -1,
            })
            eval_path = trial_dir / "eval.jsonl"
            online_path = trial_dir / "online_score.jsonl"
            row["metrics"] = _read_jsonl(metrics_path)
            row["evals"] = _read_jsonl(eval_path)
            row["online"] = _read_jsonl(online_path)
            row["metrics_mtime"] = _mtime(metrics_path)
            row["eval_mtime"] = _mtime(eval_path)
            row["online_mtime"] = _mtime(online_path)

        records = {}
        for key, row in merged.items():
            metrics, evals, online, summary = (row["metrics"], row["evals"],
                                                 row["online"], row["legacy_summary"])
            last_metric = metrics[-1] if metrics else {}
            last_eval = evals[-1] if evals else {}
            last_online = online[-1] if online else {}
            best_eval = max(evals, key=lambda item: item.get("mean", float("-inf")), default={})
            metrics_path = self.root / row["host"] / row["wave"] / row["name"] / "metrics.jsonl"
            status = "finished"
            try:
                import time
                if time.time() - metrics_path.stat().st_mtime <= 2700:
                    status = "live"
            except OSError:
                pass
            run_id = f"{self.name}::{key}"
            config = {field: row["config"].get(field) for field in
                      ("arch", "params", "gamma", "batch", "lr", "K", "stage1", "stage3",
                       "n_lat", "d_lat", "d_tok", "hidden", "host", "wave", "killy",
                       "geometry", "eval_grid", "n_step", "mirror_aug", "n_quant", "geo_dim",
                       "grad_accum", "replay_capacity", "max_reuse")
                      if row["config"].get(field) is not None}
            config["host"] = row["host"]
            config["wave"] = row["wave"]
            fields = {"env_steps", "grad_steps", "env_sps", "grad_sps", "eval_best",
                      "eval_best_env_steps", "wall_h", "maxfruit_max", "moves_last"}
            compact_summary = {field: summary[field] for field in fields if summary.get(field) is not None}
            compact_summary.update({
                "env_steps": last_metric.get("env_steps", compact_summary.get("env_steps")),
                "grad_steps": last_metric.get("grad_steps", compact_summary.get("grad_steps")),
                "env_sps": summary.get("env_sps") or self._median_rate(metrics, "env_steps"),
                "grad_sps": summary.get("grad_sps") or self._median_rate(metrics, "grad_steps"),
                "eval_last": last_eval.get("mean", summary.get("eval_last")),
                "eval_best": max((item.get("mean", float("-inf")) for item in evals),
                                 default=summary.get("eval_best")),
                "online_last": last_online.get("mean"),
                "online_last_p25": last_online.get("p25"),
                "online_last_p75": last_online.get("p75"),
                "online_last_p90": last_online.get("p90"),
                "online_episodes": sum(item.get("episodes", 0) for item in online),
                "online_epsilon": last_online.get("eps_mean"),
                "eval_best_env_steps": best_eval.get("env_steps", summary.get("eval_best_env_steps")),
                "maxfruit_max": max((item.get("maxfruit_max", 0) for item in evals),
                                    default=summary.get("maxfruit_max")),
                "updated": max(row["metrics_mtime"], row["eval_mtime"], row["online_mtime"]),
            })
            algorithm = "Qwen RL" if row["config"].get("arch") == "qwen" else "DQN"
            record = TrialRecord(run_id, self.name, algorithm, row["name"], row["wave"], status,
                                 _finite(config), _finite(compact_summary))
            records[run_id] = {
                "record": record,
                "metrics": metrics,
                "evals": evals,
                "online": online,
                "config": record.config,
                "summary": record.summary,
                "host": row["host"],
            }
        self._records = records
        self._cache_key = cache_key

    @staticmethod
    def _median_rate(metrics: list[dict[str, Any]], field: str) -> float | None:
        rates = []
        for previous, current in zip(metrics, metrics[1:]):
            elapsed = current.get("t", 0) - previous.get("t", 0)
            delta = current.get(field, 0) - previous.get(field, 0)
            if elapsed > 0 and delta >= 0:
                rates.append(delta / elapsed)
        if not rates:
            return None
        rates.sort()
        return rates[len(rates) // 2]

    def list_trials(self) -> list[TrialRecord]:
        self._load()
        return [item["record"] for item in self._records.values()]

    def get_trial(self, trial_id: str) -> dict[str, Any] | None:
        self._load()
        row = self._records.get(trial_id)
        if row is None:
            return None
        metrics, evals = row["metrics"], row["evals"]
        train_series = {"env_steps": [item.get("env_steps") for item in metrics]}
        for field in self._METRIC_FIELDS:
            train_series[field] = [item.get(field) for item in metrics]
        eval_series = {"env_steps": [item.get("env_steps") for item in evals]}
        for field in self._EVAL_FIELDS:
            eval_series[field] = [item.get(field) for item in evals]
        online = row["online"]
        online_series = {field: [item.get(field) for item in online]
                         for field in self._ONLINE_FIELDS}
        record = row["record"]
        return _finite({
            "trial": record.__dict__,
            "host": row["host"],
            "training": train_series,
            "evaluation": eval_series,
            "online": online_series,
        })
