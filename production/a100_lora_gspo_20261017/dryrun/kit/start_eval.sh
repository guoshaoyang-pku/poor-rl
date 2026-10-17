#!/bin/bash
# Dry-drill stand-in for start_eval.sh: a fake live eval that exits on RUN/FAILED.json or RUN/DONE.json.
# Usage: start_eval.sh RUN_DIR [EVERY] [N]
setsid nohup python "$(dirname "$(readlink -f "$0")")/../fake_trainer.py" eval "$1" >> "$1/eval_live.log" 2>&1 < /dev/null &
echo "eval pid $!"
