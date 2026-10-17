#!/bin/bash
# Start the trainer-head supervisor (router + DDP trainer) detached. Usage: start_head.sh PLAN_JSON
set -eu
source /home/tione/guoshaoyang/a100_rl/env.sh
RUN=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['run_dir'])" "$1")
mkdir -p "$RUN"; cp "$1" "$RUN/plan.json"
setsid nohup python /home/cfs/guoshaoyang/aiq_rl/q35_4b_gspo_20261017/code/head.py "$RUN/plan.json" \
  > "$RUN/head.log" 2>&1 < /dev/null &
echo "head pid $! run $RUN"
