#!/bin/bash
# Usage: kill_node.sh RUN_ROOT  — terminate all procs of this run root.
set -u
ROOT=$1
pkill -TERM -f "$ROOT" || true
sleep 4
pkill -KILL -f "$ROOT" || true
echo "killed all procs matching $ROOT"
