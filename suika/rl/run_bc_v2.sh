#!/bin/bash
# w4 BC v2 (Qwen3.5-0.8B + LoRA) — node_b, 8 ranks, warm-resumed.
# v2.0 (steps 400-600): soft pi KD (bc_pi_tau=1.0) instead of hard argmax CE.
# v2.1 (steps 600+): + soft RANKING distill on the Q vector
#   (bc_qrank_tau=1.0): at step 600 q_regret was 28.8 == the constant-policy
#   baseline 30.4 and eval 523.9 == random play, while val_qd had halved
#   (97->53) — i.e. the raw-Q smooth_l1 was learning the state level, not the
#   action ranking that actually decides argmax.
set -e
cd /path/to/suika-dqn
RUN=runs/w4bc_20260929/bc_qwen_v2
RESUME=${RESUME:-$RUN/checkpoints/step600.pt}
mkdir -p "$RUN/checkpoints"
if [ ! -f "$RUN/checkpoints/step400.pt" ]; then
  cp runs/w4bc_20260929/bc_qwen/checkpoints/step400.pt "$RUN/checkpoints/"
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NCCL_DEBUG=WARN
exec ./suika-venv/bin/python -m torch.distributed.run \
  --nproc_per_node 8 --master_port 29553 \
  suika_dqn/bc_learner_qwen.py \
  --config suika_dqn/configs/w4_bc_qwen_v2.yaml \
  --run-dir "$RUN" \
  --data-dir bc_data/inbox \
  --max-transitions 12000000 --val-every 200 \
  --resume-from "$RESUME"
