#!/bin/bash
# w4 BC v3 (Qwen3.5-0.8B + LoRA) — node_b, 8 ranks.
# v3 = normalized dueling split: L_v on level/1000, L_adv on contrast/12 (all
# the ranking), soft pi KD (tau=1), listwise qrank (tau=1); raw-scale qd and TD
# off. Rationale + probe evidence: configs/w4_bc_qwen_v3.yaml header.
# RESUME=<ckpt> overrides; default = latest v2 checkpoint (trunk warm start).
set -e
cd /path/to/suika-dqn
RUN=runs/w4bc_20260929/bc_qwen_v3
mkdir -p "$RUN/checkpoints"
SRC=${SRC:-/path/to/suika-dqn/runs/w4bc_20260929/bc_qwen_v2/checkpoints}
if [ -z "$RESUME" ]; then
  # prefer this run's own newest checkpoint, else fall back to the v2 chain
  RESUME=$(ls -t "$RUN"/checkpoints/step*.pt 2>/dev/null | head -1)
  [ -z "$RESUME" ] && RESUME=$(ls -t "$SRC"/step*.pt 2>/dev/null | head -1)
fi
if [ -n "$RESUME" ] && [ ! -f "$RUN/checkpoints/$(basename "$RESUME")" ]; then
  cp "$RESUME" "$RUN/checkpoints/"
  RESUME="$RUN/checkpoints/$(basename "$RESUME")"
fi
echo "[run_bc_v3] resume=$RESUME"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NCCL_DEBUG=WARN
exec ./suika-venv/bin/python -m torch.distributed.run \
  --nproc_per_node 8 --master_port 29553 \
  suika_dqn/bc_learner_qwen.py \
  --config suika_dqn/configs/w4_bc_qwen_v3.yaml \
  --run-dir "$RUN" \
  --data-dir bc_data/inbox \
  --max-transitions 12000000 --val-every 200 \
  --resume-from "$RESUME"
