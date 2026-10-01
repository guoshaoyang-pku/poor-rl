#!/bin/bash
# w6 vision BC (node_c, 8xA100): image-input Qwen3.5 VL BC pretraining,
# v3.1 listwise recipe, data from /dev/shm/bcvis_data (bc_collector_vis).
# Usage: nohup setsid bash run_bc_vit_t1.sh > runs/w6bc_vit_20260930/bc.log 2>&1 &
set -e
cd /path/to/suika-dqn
RUN=/path/to/suika-dqn/runs/w6bc_vit_20260930
DATA=/dev/shm/bcvis_data/inbox
CFG=suika_dqn/configs/w6_bc_qwen_vit_t1.yaml
NRANKS=${NRANKS:-8}
mkdir -p "$RUN"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
exec ./suika-venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=$NRANKS \
  suika_dqn/bc_learner_qwen_vit.py \
  --config "$CFG" --run-dir "$RUN" --data-dir "$DATA" --val-every 200 \
  ${RESUME:+--resume-from "$RESUME"}
