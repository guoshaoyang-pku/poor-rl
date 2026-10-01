from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path):
    try:
        with path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row
    except OSError:
        return


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def normalize_elapsed(rows: list[dict[str, Any]]) -> list[tuple[float, dict[str, Any]]]:
    output = []
    offset = 0.0
    previous = None
    for row in rows:
        try:
            raw_time = float(row["t"])
        except (KeyError, TypeError, ValueError):
            continue
        elapsed = raw_time + offset
        if previous is not None and elapsed <= previous:
            offset += previous - elapsed + 0.01
            elapsed = raw_time + offset
        previous = elapsed
        output.append((elapsed, row))
    return output


def metric_value_at(metrics: list[dict[str, Any]], elapsed: float, field: str) -> float | None:
    points = []
    time_offset = value_offset = 0.0
    previous_time = previous_value = None
    for row in metrics:
        current_time, current_value = row.get("t"), row.get(field)
        if not isinstance(current_time, (int, float)) or not isinstance(current_value, (int, float)):
            continue
        if not math.isfinite(current_time) or not math.isfinite(current_value):
            continue
        current_time, current_value = float(current_time) + time_offset, float(current_value) + value_offset
        if previous_time is not None and current_time <= previous_time:
            time_offset += previous_time - current_time + 0.01
            current_time = float(row["t"]) + time_offset
        if previous_value is not None and current_value < previous_value:
            value_offset += previous_value - current_value
            current_value = float(row[field]) + value_offset
        points.append((current_time, current_value))
        previous_time, previous_value = current_time, current_value
    if not points:
        return None
    if elapsed <= points[0][0]:
        first_time, first_value = points[0]
        return first_value * max(0.0, elapsed) / first_time if first_time > 0 else first_value
    for (left_time, left_value), (right_time, right_value) in zip(points, points[1:]):
        if left_time <= elapsed <= right_time:
            fraction = (elapsed - left_time) / max(right_time - left_time, 1e-9)
            return left_value + fraction * (right_value - left_value)
    if len(points) >= 2:
        (left_time, left_value), (right_time, right_value) = points[-2:]
        rate = (right_value - left_value) / max(right_time - left_time, 1e-9)
        return max(right_value, right_value + max(0.0, elapsed - right_time) * rate)
    return points[0][1]


def aggregate_digest_trial(source: Path, destination: Path,
                           metrics: list[dict[str, Any]]) -> int:
    digest_rows = list(read_jsonl(source / "digest_episode_bins.jsonl"))
    if not digest_rows:
        return 0

    output = []
    episode_count = 0
    for row in digest_rows:
        try:
            elapsed = float(row["t_min"]) * 60 + 300
            count = int(row["n"])
            mean = float(row["mean"])
            p90 = float(row["p90"])
            maximum = float(row["max"])
            moves = float(row["moves"])
        except (KeyError, TypeError, ValueError):
            continue
        if count <= 0 or not all(math.isfinite(value) for value in
                                 (elapsed, mean, p90, maximum, moves)):
            continue
        env_steps = metric_value_at(metrics, elapsed, "env_steps") if metrics else None
        grad_steps = metric_value_at(metrics, elapsed, "grad_steps") if metrics else None
        episode_count += count
        output.append({
            "time_hours": round(elapsed / 3600, 5),
            "env_steps": round(env_steps) if env_steps is not None else None,
            "grad_steps": round(grad_steps) if grad_steps is not None else None,
            "mean": mean,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": p90,
            "max": maximum,
            "eps_mean": None,
            "moves_mean": moves,
            "episodes": count,
            "actors": None,
        })
    if not output:
        return 0

    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "online_score.jsonl").open("w") as stream:
        for row in output:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return episode_count


def aggregate_trial(source: Path, destination: Path, bucket_seconds: int,
                    digest_only: bool = False) -> int:
    metric_path = source / "metrics.jsonl"
    if not metric_path.exists():
        return 0
    digest_path = source / "digest_episode_bins.jsonl"
    if digest_only and not digest_path.exists():
        return 0
    metrics = list(read_jsonl(metric_path))
    if digest_only:
        return aggregate_digest_trial(source, destination, metrics)
    episode_paths = list(source.glob("episodes_a*.jsonl"))
    if not episode_paths:
        return aggregate_digest_trial(source, destination, metrics)
    buckets: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"scores": [], "eps": [], "moves": [], "actors": set(), "t_sum": 0.0})
    episode_count = 0
    for path in source.glob("episodes_a*.jsonl"):
        for elapsed, row in normalize_elapsed(list(read_jsonl(path))):
            try:
                score = float(row["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(elapsed) or not math.isfinite(score):
                continue
            bucket_index = max(0, int(elapsed // bucket_seconds))
            bucket = buckets[bucket_index]
            bucket["scores"].append(score)
            bucket["t_sum"] += elapsed
            bucket["actors"].add(row.get("actor"))
            for key, target in (("eps", "eps"), ("moves", "moves")):
                try:
                    value = float(row[key])
                    if math.isfinite(value):
                        bucket[target].append(value)
                except (KeyError, TypeError, ValueError):
                    pass
            episode_count += 1
    if not episode_count:
        return 0

    output = []
    for index, bucket in sorted(buckets.items()):
        scores = bucket["scores"]
        elapsed = bucket["t_sum"] / len(scores)
        env_steps = metric_value_at(metrics, elapsed, "env_steps") if metrics else None
        grad_steps = metric_value_at(metrics, elapsed, "grad_steps") if metrics else None
        output.append({
            "time_hours": round(elapsed / 3600, 5),
            "env_steps": round(env_steps) if env_steps is not None else None,
            "grad_steps": round(grad_steps) if grad_steps is not None else None,
            "mean": sum(scores) / len(scores),
            "p25": percentile(scores, 0.25),
            "median": percentile(scores, 0.5),
            "p75": percentile(scores, 0.75),
            "p90": percentile(scores, 0.9),
            "max": max(scores),
            "eps_mean": sum(bucket["eps"]) / len(bucket["eps"]) if bucket["eps"] else None,
            "moves_mean": sum(bucket["moves"]) / len(bucket["moves"]) if bucket["moves"] else None,
            "episodes": len(scores),
            "actors": len(bucket["actors"]),
        })
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "online_score.jsonl"
    with output_path.open("w") as stream:
        for row in output:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return episode_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--bucket-seconds", type=int, default=300)
    parser.add_argument("--digest-only", action="store_true",
                        help="prefer compact digest bins where available; do not read raw episode logs")
    args = parser.parse_args()
    if args.bucket_seconds < 1:
        parser.error("--bucket-seconds must be positive")
    source_root, destination_root = args.source.resolve(), args.destination.resolve()
    trials = episodes = 0
    for metrics_path in sorted(source_root.glob("*/*/*/metrics.jsonl")):
        source = metrics_path.parent
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        count = aggregate_trial(source, destination, args.bucket_seconds,
                                digest_only=args.digest_only)
        if count:
            trials += 1
            episodes += count
            print(f"{relative}: {count:,} episodes -> {destination / 'online_score.jsonl'}")
    print(f"aggregated {episodes:,} episodes across {trials} trials")


if __name__ == "__main__":
    main()
