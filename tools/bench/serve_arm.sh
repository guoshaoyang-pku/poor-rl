#!/usr/bin/env bash
# Usage: serve_arm.sh GPU PORT NAME [extra vllm args...]
# Starts a 0.8B rollout server in the background, waits for /health, logs to $LOGDIR/NAME.log.
set -u
GPU=$1; PORT=$2; NAME=$3; shift 3
V=${VENV:-/data/home/guoshaoyang/aiq_rl/venv}/bin
MODEL=${MODEL:-/data/home/guoshaoyang/models/Qwen3.5-0.8B-ms}
LOGDIR=${LOGDIR:-/data/home/guoshaoyang/aiq_rl/runs/bench_sat_20261003}
mkdir -p "$LOGDIR"
export VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=$GPU
nohup "$V/vllm" serve "$MODEL" --port "$PORT" --served-model-name t \
  --max-model-len "${MAX_LEN:-18432}" --gpu-memory-utilization "${UTIL:-0.90}" \
  --language-model-only "$@" > "$LOGDIR/$NAME.log" 2>&1 &
echo $! > "$LOGDIR/$NAME.pid"
for i in $(seq 1 180); do
  c=$(curl -s -o /dev/null -w '%{http_code}' "localhost:$PORT/health")
  [ "$c" = "200" ] && { echo "UP $NAME after $((i*5))s"; exit 0; }
  kill -0 "$(cat "$LOGDIR/$NAME.pid")" 2>/dev/null || { echo "DIED $NAME"; grep -E "Error|error" "$LOGDIR/$NAME.log" | tail -5; exit 1; }
  sleep 5
done
echo "TIMEOUT $NAME"; exit 1
