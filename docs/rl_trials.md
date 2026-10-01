# Algorithm-neutral RL trials

The local HTML RL panel is served by the existing dashboard process and is enabled with `--rl-data-root`. Trial records share a small stable envelope (`id`, `adapter`, `algorithm`, `name`, `experiment`, `status`, `config`, `summary`); each adapter owns its native time-series schema. The panel uses the common fields for trial discovery and metrics, and displays algorithm metadata without assuming that every method is DQN.

## View the bundled Suika history

From the repository root:

```bash
PYTHONPATH=src python -m rlforge.dashboard \
  --rl-data-root examples/suika_trials \
  --port 8872
```

Open `http://localhost:8872`. This leaves the existing dashboard on port 8871 untouched. To point at an updated export, pass its directory; `trials.json` is discovered by default, while per-trial `metrics.jsonl` and `eval.jsonl` files are loaded from `<root>/<host>/<wave>/<trial>/`.

The bundled history includes the historical summary ledger, learner metrics, greedy evaluation curves, and compact online-score curves. Raw actor logs from 29 historical trials contribute 33.6 million five-minute episode aggregates; eight recent trials publish ten-minute digests rather than raw logs and contribute another 19.8 million episode outcomes. The panel therefore exposes online score on all 37 trials with episode data while keeping the compact exports below 1 MB. Raw episode/action/checkpoint/replay artifacts are not copied.

Chart A overlays the actor-side online episode return/score (solid) with fixed-seed greedy evaluation (same-color dashed); gray dashed lines show P25/P75 when raw episodes are available, or P90 when only compact digests exist. The actor's episode score is the accumulated game score and closely matches return under the current pure score-delta reward (γ=1, no shaping). Online score and greedy evaluation are different policy distributions: a rising online curve with flat greedy eval indicates a train/eval gap that may include ε exploration and evaluation-distribution differences, not necessarily pure generalization failure; both flat suggests little measured progress. Online episode timestamps are aligned to learner environment steps by interpolation, so that x-coordinate is approximate. Recent runs publish ten-minute digest bins rather than raw actor episode logs; these retain mean/P90/max and episode counts. Trials without either actor logs or digest bins have no online curve.

Refresh the compact online-score export from the Suika checkout with:

```bash
python scripts/import_suika_online_scores.py \\
  --source /path/to/suika/dqn_runs \\
  --destination examples/suika_trials \\
  --bucket-seconds 300
```

## Adding an algorithm

Implement `TrialAdapter` from `rlforge.rl_trials` and register it in `TrialRegistry`. An adapter returns `TrialRecord` values and resolves details by namespaced IDs (`adapter-name::native-id`). Keep the canonical record fields stable; put algorithm-specific hyperparameters in `config`, aggregate results in `summary`, and return its native curves from `get_trial`. The RL panel can then list runs from multiple registered adapters without modifying the shared trial model.

The initial `suika-dqn` adapter merges historical `trials.json` rows with per-trial JSONL data. New curve data overrides matching ledger summaries while ledger-only trials remain visible. Malformed partial JSONL lines are skipped so live writes do not break panel refreshes.
