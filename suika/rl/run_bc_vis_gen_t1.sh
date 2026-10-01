#!/bin/bash
# w6 vision BC data gen (node_c, CPU actors -> /dev/shm). Teacher mlp_deep
# rollouts at killy=200, 288x416 JPEG renders. Seed base 880001 (independent
# of the perm copy at 770001). 64 actors x 400k = 25.6M transitions max;
# tmpfs is 1.8T so the full set fits in RAM.
# Usage: nohup setsid bash run_bc_vis_gen_t1.sh > /dev/shm/bcvis_data/gen.log 2>&1 &
set -e
cd /path/to/suika-dqn
RUN=/dev/shm/bcvis_data
TEACHER=/path/to/suika-dqn/teacher_w3b.pt
ACTORS=${ACTORS:-64}
MAXT=${MAXT:-400000}
mkdir -p "$RUN"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=suika_dqn
for a in $(seq 0 $((ACTORS-1))); do
  nohup ./suika-venv/bin/python suika_dqn/bc_collector_vis.py \
    --run-dir "$RUN" --teacher "$TEACHER" \
    --actor "$a" --actors "$ACTORS" --max-transitions "$MAXT" \
    --killy 200 --seed-base 880001 > "$RUN/actor$a.log" 2>&1 &
  sleep 0.2
done
echo "launched $ACTORS actors -> $RUN"
wait
