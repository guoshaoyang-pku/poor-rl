#!/bin/bash
# Start the auto-resume watcher detached: `head` on a100_t1 (next to head.py), `node` on each worker node (a100_t1_2).
# Usage: start_watch.sh head|node RUN_DIR [resume_watch.py options]
# State, ledger, events and logs: /home/cfs/guoshaoyang/aiq_rl/q35_4b_gspo_20261017/resume_watch/
set -eu
source /home/tione/guoshaoyang/a100_rl/env.sh
NFS=/home/cfs/guoshaoyang/aiq_rl/q35_4b_gspo_20261017
ROLE=$1; RUN=$2; shift 2
mkdir -p "$NFS/resume_watch"
LOG=$NFS/resume_watch/${ROLE}_$(hostname -I | awk '{print $1}').log
setsid nohup python "$NFS/code/resume_watch.py" "$ROLE" "$RUN" "$@" >> "$LOG" 2>&1 < /dev/null &
echo "$ROLE watcher pid $! log $LOG"
