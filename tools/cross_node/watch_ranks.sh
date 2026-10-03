#!/usr/bin/env bash
# Tear the run down if any headless rollout rank (on another host) dies.
#
# Why: a dead headless DP rank is invisible to the head. /health stays 200, requests the LB
# routes to that rank hang until the client timeout (TRL request_timeout, 3600-5400 s), and the
# next weight sync blocks (init: 300 s TCPStore timeout then HTTP 500; pause/update are
# broadcast to every engine). Without this watchdog the trainer silently stalls.
#
# Usage (on the head host, after the headless ranks are up):
#   REMOTE=10.234.161.2 REMOTE_PIDFILE=/data/home/guoshaoyang/crossnode_20261003/dp_headless.pid \
#   KILL_PIDS="<launcher pid> <head vllm pid>" bash watch_ranks.sh
# KILL_PIDS get SIGTERM, so run_async_dp.sh's TERM trap records stop_reason and stops vLLM.
set -u
: "${REMOTE:?}" "${REMOTE_PIDFILE:?}" "${KILL_PIDS:?}"
INTERVAL=${INTERVAL:-15}
MISSES_ALLOWED=${MISSES_ALLOWED:-2}   # tolerate brief ssh hiccups
misses=0
while true; do
  for p in $KILL_PIDS; do kill -0 "$p" 2>/dev/null || { echo "[watch] $(date +%T) local pid $p gone; exiting"; exit 0; }; done
  if ssh -o BatchMode=yes -o ConnectTimeout=5 "$REMOTE" "kill -0 \$(cat $REMOTE_PIDFILE)" 2>/dev/null; then
    misses=0
  else
    misses=$((misses + 1))
    echo "[watch] $(date +%T) remote rank check failed ($misses/$MISSES_ALLOWED)"
    if [ "$misses" -gt "$MISSES_ALLOWED" ]; then
      echo "[watch] $(date +%T) headless rank on $REMOTE is dead: SIGTERM $KILL_PIDS"
      kill -TERM $KILL_PIDS 2>/dev/null
      exit 1
    fi
  fi
  sleep "$INTERVAL"
done
