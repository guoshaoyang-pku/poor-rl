import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from import_suika_online_scores import aggregate_trial


def test_online_score_importer_buckets_episodes_and_aligns_env_steps(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "metrics.jsonl").write_text(
        '{"t":0,"env_steps":0}\n{"t":600,"env_steps":60000}\n'
    )
    (source / "episodes_a0.jsonl").write_text(
        '{"t":20,"score":10,"moves":20,"eps":0.5,"actor":0}\n'
        '{"t":80,"score":30,"moves":40,"eps":0.3,"actor":0}\n'
        '{"t":340,"score":50,"moves":60,"eps":0.1,"actor":1}\n'
    )

    count = aggregate_trial(source, tmp_path / "output", bucket_seconds=300)
    rows = [json.loads(line) for line in (tmp_path / "output" / "online_score.jsonl").read_text().splitlines()]

    assert count == 3
    assert len(rows) == 2
    assert rows[0]["mean"] == 20
    assert rows[0]["p25"] == 15
    assert rows[0]["p75"] == 25
    assert rows[0]["episodes"] == 2
    assert rows[1]["env_steps"] == 34000
    assert rows[1]["mean"] == 50


def test_importer_ignores_missing_and_malformed_episode_lines(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "metrics.jsonl").write_text('{"t":10,"env_steps":100}\n')
    (source / "episodes_a0.jsonl").write_text(
        'not-json\n{"t":2,"score":4,"actor":0}\n{"score":8}\n'
    )

    count = aggregate_trial(source, tmp_path / "output", bucket_seconds=300)

    assert count == 1
    row = json.loads((tmp_path / "output" / "online_score.jsonl").read_text())
    assert row["mean"] == 4
    assert row["env_steps"] == 20
    assert row["grad_steps"] is None


def test_importer_preserves_elapsed_time_across_metric_restarts(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    metrics = [
        {"t": 100.0, "env_steps": 1000},
        {"t": 200.0, "env_steps": 2000},
        {"t": 10.0, "env_steps": 10},
        {"t": 20.0, "env_steps": 110},
    ]
    (source / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metrics))
    (source / "episodes_a0.jsonl").write_text(
        '{"t":150,"score":10,"actor":0}\n'
        '{"t":25,"score":20,"actor":0}\n'
    )

    aggregate_trial(source, tmp_path / "output", bucket_seconds=300)
    rows = [json.loads(line) for line in (tmp_path / "output" / "online_score.jsonl").read_text().splitlines()]

    assert len(rows) == 1
    assert rows[0]["mean"] == 15
    assert rows[0]["env_steps"] > 1000


def test_importer_uses_compact_digest_when_raw_episodes_are_absent(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "metrics.jsonl").write_text(
        '{"t":0,"env_steps":0}\n{"t":600,"env_steps":60000}\n'
    )
    (source / "digest_episode_bins.jsonl").write_text(
        '{"t_min":0,"n":200,"mean":1200,"p90":1800,"max":2400,"moves":140}\n'
        '{"t_min":10,"n":300,"mean":1500,"p90":2200,"max":2800,"moves":160}\n'
    )

    count = aggregate_trial(source, tmp_path / "output", bucket_seconds=300)
    rows = [json.loads(line) for line in
            (tmp_path / "output" / "online_score.jsonl").read_text().splitlines()]

    assert count == 500
    assert len(rows) == 2
    assert rows[1]["env_steps"] == 90000
    assert rows[1]["grad_steps"] is None
    assert rows[1]["mean"] == 1500
    assert rows[1]["p90"] == 2200
    assert rows[1]["episodes"] == 300
