# Training panels and experiment tracking

rlforge uses SwanLab as the default experiment tracker, in local mode. It gives a
single run-comparison surface for verl/LLM RL, Transformers SFT, and custom
PyTorch or Stable-Baselines3 experiments. The rlforge HTML report remains the
source for task-aware RL interpretation; verl users can optionally add RL-Insight
for distributed rollout and system observability.

## Fixed panel entry

Bookmark this guide: [github.com/guoshaoyang-pku/poor-rl/blob/main/docs/PANEL.md](https://github.com/guoshaoyang-pku/poor-rl/blob/main/docs/PANEL.md#fixed-panel-entry).
The dashboard runs on each viewer's machine; these localhost links are not a
publicly hosted copy of the author's experiments.

| Panel | Fixed local URL | Data directory |
|---|---|---|
| Home | [127.0.0.1:63400](http://127.0.0.1:63400/) | `dashboard_home.html`, `artifacts/samples/` |
| AIQ / LLM RL | [127.0.0.1:63401](http://127.0.0.1:63401/) | `artifacts/swanlab/AIQ/swanlog/` |
| Suika | [127.0.0.1:63402](http://127.0.0.1:63402/) | `artifacts/swanlab/Suika/swanlog/` |
| SFT | [127.0.0.1:63403](http://127.0.0.1:63403/) | `artifacts/swanlab/SFT/swanlog/` |

From the repository root, create a persistent dashboard environment:

```bash
python3 -m venv .venv-panel
.venv-panel/bin/python -m pip install -e ".[panel]"
```

Start Home and AIQ in separate terminals, both from the repository root:

```bash
# Terminal 1
.venv-panel/bin/python scripts/home.py --host 127.0.0.1 --port 63400

# Terminal 2
ROOT="$PWD" VENV="$PWD/.venv-panel" PROJECT=AIQ PORT=63401 bash scripts/panel.sh
```

For Suika or SFT, run the second command with `PROJECT=Suika PORT=63402` or
`PROJECT=SFT PORT=63403`. Existing offline data is displayed immediately. New
training runs must log to the matching data directory, using `SWANLAB_MODE=local`,
`SWANLAB_LOGDIR` and `SWANLAB_PROJ_NAME` as described below. To import and refresh
remote experiments and rollout samples, follow [the mirroring guide](PANEL_LINK.md).
Environment and data stay in the checkout, rather than a temporary directory.

## Unified local Home

Start the local launcher to reach all three isolated SwanLab panels from one
bookmark:

```bash
python scripts/home.py --host 127.0.0.1 --port 63400
```

Open `http://127.0.0.1:63400`. The Home opens each panel in a new browser tab;
the AIQ, Suika, and SFT SwanLab panels remain on ports `63401`, `63402`, and
`63403`, respectively. Start those panel processes separately with
`scripts/panel.sh` and the matching `PROJECT` value. The Home server only serves
its landing page and does not expose the repository directory. The current
long-running local instance uses tmux session `rlforge-panels` with one window
per service (`home`, `aiq`, `suika`, `sft`). Reattach with
`tmux attach -t rlforge-panels`, list windows with
`tmux list-windows -t rlforge-panels`, and stop all four with
`tmux kill-session -t rlforge-panels`.

## SwanLab: general experiment tracking

Install the local dashboard extra once:

```bash
pip install -e ".[panel]"
```

The launcher defaults to `REPORT_TO=swanlab`, `SWANLAB_MODE=local`, a shared
`$ROOT/swanlog` directory, and the `AIQ` project. The local store can hold
separate `AIQ`, `Suika`, and `SFT` projects; select the project in SwanLab rather
than mixing unlike workloads into one run list. No cloud account or GPU is
needed to view the local dashboard:

```bash
ROOT=/path/to/rlforge PROJECT=AIQ VENV=/path/to/venv bash scripts/panel.sh
```

Choose `PROJECT=Suika` or `PROJECT=SFT` to open those separate experiment groups
on another panel port if you want to compare them side by side. Alternatively,
set `SWANLAB_LOGDIR` to a shared root and use SwanLab's project selector.

Open `http://127.0.0.1:5092`. For a remote node, keep the panel bound to
localhost and tunnel it with `ssh -L 5092:localhost:5092 <host>`.

To opt in to cloud sync, set `SWANLAB_MODE=cloud` and configure SwanLab
credentials. Other Hugging Face integrations remain selectable with
`REPORT_TO=tensorboard`, `wandb`, `mlflow`, or `none`.

SwanLab can also track SFT through the Transformers/TRL callback and classic RL
through Stable-Baselines3 or `swanlab.log()` in a custom PyTorch loop. For those
launchers, use the same shared experiment store and project explicitly:

```bash
export SWANLAB_MODE=local
export SWANLAB_LOGDIR=/path/to/project/swanlog
export SWANLAB_PROJ_NAME=AIQ
```

Use `SWANLAB_PROJ_NAME=Suika` or `SFT` for those workloads. The importer keeps
separate local SwanLab stores for the three projects, preventing name collisions
and accidental cross-grouping. `scripts/panel.sh` opens one selected project at a
time; use another port to keep a second project visible alongside it. The
importer's three collapsible sections are ordered as follows: primary reward,
length, and time metrics; held-out accuracy and both GSPO sequence-clip sides;
then diagnostics, including per-interval accuracy, truncation, and completion
length. Metric keys state their units (`_percent`, `_tokens`, `_seconds`,
`_hours`); reward is a dimensionless task score, not accuracy, and its scale
depends on the task's grader.

Keep metric names and run metadata consistent to compare workloads. The dashboard
extra is not required on remote training workers if another machine hosts
`swanlab watch`.

## rlforge HTML report: RL-specific metrics

```bash
python -m rlforge.report --run <run> --trainer-log <trainer log> \
    --eval-history <eval history.jsonl> --base-eval <base eval.json> \
    --out <run>/report --interval 600
```

Refreshes every 10 minutes while the run is live:

- held-out eval ladder per checkpoint (with base model + constant baselines),
- per-task reward / accuracy / truncation curves reconstructed from reward-call
  counters; accuracy excludes the `-2` truncation penalty,
- per-source curves when the dataset carries a `source` column,
- completion length, entropy, and GSPO clip-fraction health metrics,
- self-contained `report.html` and `metrics.json`.

## Optional: RL-Insight for verl

RL-Insight adds trainer, rollout-engine, queue, trace, and hardware views for
verl workloads. It complements rather than replaces SwanLab; install and enable
it only when those distributed runtime diagnostics are needed:
[verl integration guide](https://verl.readthedocs.io/en/latest/advance/rl_insight.html).
