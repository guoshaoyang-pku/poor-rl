#!/bin/bash
# w4 BC v3.1 (Qwen3.5-0.8B + LoRA) — node_d, 8 ranks, resumed at 5200/12000.
# Prepared because node_b was cleared at 22:17 for the SFT deployment
# (which targets t1_2); perm has the full 48-actor BC data (7056 shards,
# 28.9M transitions) and a venv that runs the same code (2-rank smoke passed).
# Usage:  nohup setsid bash run_bc_v3_perm.sh > runs/w4bc_bc/v3_perm/bc.log 2>&1 &
set -e
cd /data/user/suika-dqn
RUN=runs/w4bc_bc/v3_perm
RESUME=${RESUME:-/data/user/suika-dqn/runs/w4bc_bc/step5200.pt}
mkdir -p "$RUN/checkpoints"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NCCL_DEBUG=WARN
exec ./suika-venv/bin/python -m torch.distributed.run \
  --nproc_per_node 8 --master_port 29597 \
  suika_dqn/bc_learner_qwen.py \
  --config suika_dqn/configs/w4_bc_qwen_v3_perm.yaml \
  --run-dir "$RUN" \
  --data-dir runs/bc_data_20260929/inbox \
  --max-transitions 12000000 --val-every 200 \
  --resume-from "$RESUME"
