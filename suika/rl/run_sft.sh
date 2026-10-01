#!/bin/bash
# SFT stage 1 (token policy = normalized teacher Q, shared-backbone value head)
# — node_b, 8 ranks. Budget: ~1 h (user spec 2026-09-29).
# Companion evaluation (separate, light): eval_qwen_policy.py --decode tokens
# and --decode qhead, both watching <RUN>/policy.pt.
#   RESUME=<ckpt>   warm start (e.g. the v3.1 BC ckpt: same trunk/heads keys)
#   BUDGET=<n>      override grad_budget
#   MAXT=<n>        override --max-transitions (smoke tests)
set -e
# node-specific project root (t1_2 vs perm)
cd "${ROOT:-/path/to/suika-dqn}"
RUN=${RUN:-runs/w4sft_20260929/sft_qwen}
DATA=${DATA:-bc_data/inbox}
MAXT=${MAXT:-12000000}
CONFIG=${CONFIG:-suika_dqn/configs/sft_qwen.yaml}
mkdir -p "$RUN/checkpoints"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NCCL_DEBUG=WARN
EXTRA=""
if [ -n "$RESUME" ]; then EXTRA="$EXTRA --resume-from $RESUME"; fi
if [ -n "$BUDGET" ]; then EXTRA="$EXTRA --grad-budget $BUDGET"; fi
echo "[run_sft] run=$RUN data=$DATA config=$CONFIG resume=${RESUME:-none} budget=${BUDGET:-cfg}"
exec ./suika-venv/bin/python -m torch.distributed.run \
  --nproc_per_node "${NPROC:-8}" --master_port 29553 \
  suika_dqn/sft_qwen.py \
  --config "$CONFIG" \
  --run-dir "$RUN" \
  --data-dir "$DATA" \
  --max-transitions 12000000 --val-every "${VAL_EVERY:-100}" $EXTRA
