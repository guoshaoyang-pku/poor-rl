#!/bin/bash
# Usage: launch_wave6.sh ARM OBS_DIM [ACTORS] [LEARNER_GPU] [EVAL_GPU]
# Run from anywhere; uses the project venv. One arm = learner + infer servers
# (cfg.infer_gpus) + ACTORS physics actors + periodic evaluator.
set -eu
ARM=$1; OBS_DIM=$2
ACTORS=${3:-200}; LGPU=${4:-0}; EGPU=${5:-1}; EVAL_S=${6:-600}
CODE=$(cd "$(dirname "$0")" && pwd)
PROJ=$(dirname "$CODE")
RUN=$PROJ/runs/wave6_20260930/$ARM
mkdir -p "$RUN"
cd "$PROJ"
nohup "$PROJ/suika-venv/bin/python" "$CODE/run_arm.py" \
  --config "$CODE/configs/$ARM.yaml" --run-dir "$RUN" \
  --actors "$ACTORS" --gpu "$LGPU" --eval-gpu "$EGPU" --obs-dim "$OBS_DIM" \
  --eval-interval-s "$EVAL_S" \
  > "$RUN/orchestrator.log" 2>&1 &
echo "launched $ARM pid $! -> $RUN"
