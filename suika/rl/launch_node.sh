#!/bin/bash
# Usage: launch_node.sh RUN_ROOT arm1 arm2 ...  (arm i -> gpu i)
set -u
ROOT=$1; shift
CODE=$(cd "$(dirname "$0")" && pwd)
OBS_DIM=${OBS_DIM:-333}
ACTORS=${ACTORS:-20}
i=${GPU_OFFSET:-0}
for arm in "$@"; do
  rundir="$ROOT/$arm"
  mkdir -p "$rundir"
  nohup python3 "$CODE/run_arm.py" \
    --config "$CODE/configs/$arm.yaml" \
    --run-dir "$rundir" \
    --actors "$ACTORS" --gpu "$i" --obs-dim "$OBS_DIM" \
    > "$rundir/orchestrator.log" 2>&1 &
  echo "launched $arm on gpu $i pid $!"
  i=$((i+1))
done
