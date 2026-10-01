#!/bin/bash
# w6 vision BC data gen (node_d, CPU actors): teacher mlp_deep rollouts at
# killy=200 with synchronised board renders (288x416 JPEG). Runs alongside the
# GPU BC training (CPU-only workload). 64 actors x 400k = 25.6M transitions.
# Usage: nohup setsid bash run_bc_vis_gen.sh > runs/bcvis_data_20260930/gen.log 2>&1 &
set -e
cd /data/user/suika-dqn
RUN=/data/user/suika-dqn/runs/bcvis_data_20260930
TEACHER=/data/user/suika-dqn/runs/wave3b_20260928/w3b_mlp_deep/policy.pt
ACTORS=${ACTORS:-64}
MAXT=${MAXT:-400000}
mkdir -p "$RUN"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=suika_dqn
for a in $(seq 0 $((ACTORS-1))); do
  nohup ./suika-venv/bin/python suika_dqn/bc_collector_vis.py \
    --run-dir "$RUN" --teacher "$TEACHER" \
    --actor "$a" --actors "$ACTORS" --max-transitions "$MAXT" \
    --killy 200 > "$RUN/actor$a.log" 2>&1 &
  sleep 0.2
done
echo "launched $ACTORS actors -> $RUN"
wait
