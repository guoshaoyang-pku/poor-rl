#!/bin/bash
# RL panel (wandb-style, fully offline): TensorBoard over every run under ROOT/runs.
#
# The trainer writes TB events to <run>/tb when REPORT_TO=tensorboard (see
# scripts/run_async_dp.sh). The RL-specific curves that a generic tracker cannot
# compute (per-task reward vs accuracy vs truncation, held-out ladder) come from
# rlforge.report's HTML -- run both, they complement each other.
#
# Usage: ROOT=/path/to/project [PORT=6006] bash scripts/panel.sh
set -euo pipefail
ROOT="${ROOT:?set ROOT}"
PORT="${PORT:-6006}"
VENV="${VENV:-}"
PY="${VENV:+$VENV/bin/}python"
exec "$PY" -m tensorboard.main --logdir "$ROOT/runs" --port "$PORT" --bind_all
