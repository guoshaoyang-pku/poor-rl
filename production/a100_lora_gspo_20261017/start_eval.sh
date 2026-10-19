#!/bin/bash
# Start the co-located live eval (eval_live.py) detached on the head. Usage: start_eval.sh RUN_DIR [EVERY] [N]
set -eu
source /home/tione/guoshaoyang/a100_rl/env.sh
export CUDA_VISIBLE_DEVICES= PYTHONPATH=/home/cfs/guoshaoyang/aiq_rl/q35_4b_gspo_20261017/code/src
setsid nohup python /home/cfs/guoshaoyang/aiq_rl/q35_4b_gspo_20261017/code/eval_live.py "$1" "${2:-20}" "${3:-2}" \
  >> "$1/eval_live.log" 2>&1 < /dev/null &
echo "eval pid $!"
