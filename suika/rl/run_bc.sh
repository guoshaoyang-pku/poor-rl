#!/bin/bash
# w4 BC (Qwen3.5-0.8B + LoRA) pretraining launcher — t1_2, 8 ranks.
# Teacher = w3b_mlp_deep policy.pt @2033; data = 24-actor bc_collector subset
# (a0..a23, 14.45M transitions, 96GB) in bc_data/inbox.
set -e
cd /path/to/suika-dqn
RUN=runs/wave4_20260929/w4_bc_qwen
mkdir -p "$RUN"
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
exec ./suika-venv/bin/python -m torch.distributed.run \
  --nproc_per_node 8 --master_port 29541 \
  suika_dqn/bc_learner_qwen.py \
  --config suika_dqn/configs/w4_bc_qwen.yaml \
  --run-dir "$RUN" \
  --data-dir bc_data/inbox \
  --max-transitions 12000000 --val-every 200
