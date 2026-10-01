import json
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib.request import urlopen

from rlforge.dashboard import State, make_handler
from rlforge.rl_trials import DQNTrialAdapter, TrialRecord, TrialRegistry


def test_dqn_adapter_joins_history_curves_and_returns_summary(tmp_path):
    ledger = [{
        "trial": "old-run", "host": "node-a", "wave": "wave1", "arch": "mlp",
        "eval_last": 10, "eval_best": 12, "env_steps": 100, "grad_steps": 20,
    }]
    (tmp_path / "trials.json").write_text(json.dumps(ledger))
    (tmp_path / "configs.json").write_text(json.dumps({"old-run": {
        "geometry": {"width": [400, 550], "height": [500, 720]},
        "params": 1000,
    }}))
    run_dir = tmp_path / "node-a" / "wave1" / "old-run"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text(
        '{"t":1,"env_steps":100,"grad_steps":20,"q_mean":8,"loss":2}\n'
        '{"t":2,"env_steps":200,"grad_steps":30,"q_mean":9,"loss":1}\n'
    )
    (run_dir / "eval.jsonl").write_text(
        '{"env_steps":100,"mean":10,"p25":4}\n'
        '{"env_steps":200,"mean":14,"p25":8}\n'
    )
    (run_dir / "online_score.jsonl").write_text(
        '{"time_hours":0.1,"env_steps":90,"mean":3,"p25":1,"p75":5,"p90":7,"episodes":100,"eps_mean":0.2}\n'
        '{"time_hours":0.2,"env_steps":190,"mean":6,"p25":2,"p75":9,"p90":12,"episodes":100,"eps_mean":0.1}\n'
    )

    adapter = DQNTrialAdapter(tmp_path)
    trials = adapter.list_trials()
    trial_id = "suika-dqn::node-a/wave1/old-run"
    trial = adapter.get_trial(trial_id)

    assert len(trials) == 1
    assert trials[0].id == trial_id
    assert trials[0].algorithm == "DQN"
    assert trials[0].summary["eval_last"] == 14
    assert trials[0].summary["eval_best"] == 14
    assert trials[0].config["geometry"]["width"] == [400, 550]
    assert trials[0].config["params"] == 1000
    assert trials[0].summary["online_last"] == 6
    assert trial["training"]["env_steps"] == [100, 200]
    assert trial["evaluation"]["p25"] == [4, 8]
    assert trial["online"]["mean"] == [3, 6]
    assert trial["online"]["p90"] == [7, 12]
    assert trial["online"]["episodes"] == [100, 100]


def test_ledger_only_trial_remains_visible_without_metric_files(tmp_path):
    ledger = [{"trial": "summary-only", "host": "node-b", "wave": "wave0",
               "arch": "mlp", "eval_last": 7.5, "env_steps": 42}]
    (tmp_path / "trials.json").write_text(json.dumps(ledger))

    trial = DQNTrialAdapter(tmp_path).list_trials()[0]

    assert trial.name == "summary-only"
    assert trial.status == "finished"
    assert trial.summary["eval_last"] == 7.5


def test_registry_routes_trials_by_adapter_prefix():
    class Adapter:
        name = "sample"

        def list_trials(self):
            return [TrialRecord("sample::x", "sample", "DQN", "x", None,
                                "finished", {}, {})]

        def get_trial(self, trial_id):
            return {"id": trial_id}

    registry = TrialRegistry()
    registry.register(Adapter())

    assert registry.list_trials()[0].id == "sample::x"
    assert registry.get_trial("sample::x") == {"id": "sample::x"}
    assert registry.get_trial("missing::x") is None


def test_dashboard_routes_rl_trials_and_serves_rl_panel(tmp_path):
    run_dir = tmp_path / "node-c" / "wave2" / "trial-c"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text('{"env_steps":1,"grad_steps":1}\n')
    (run_dir / "eval.jsonl").write_text('{"env_steps":1,"mean":3}\n')
    (run_dir / "online_score.jsonl").write_text('{"env_steps":1,"mean":2,"episodes":5}\n')
    args = SimpleNamespace(root=None, registry=None, min_steps=50, run=None,
                           trainer_log=None, eval_history=None, base_eval=None,
                           rl_data_root=tmp_path, rl_ledger=None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(State(args)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(base_url) as response:
            page = response.read().decode()
        with urlopen(base_url + "/api/rl/overview") as response:
            overview = json.loads(response.read())
        with urlopen(base_url + "/api/rl/trial?id=suika-dqn%3A%3Anode-c%2Fwave2%2Ftrial-c") as response:
            detail = json.loads(response.read())
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()

    assert "全 trial 总览 panel" in page
    assert overview[0]["trial"]["name"] == "trial-c"
    assert overview[0]["evaluation"]["mean"] == [3]
    assert overview[0]["online"]["mean"] == [2]
    assert detail["evaluation"]["mean"] == [3]
