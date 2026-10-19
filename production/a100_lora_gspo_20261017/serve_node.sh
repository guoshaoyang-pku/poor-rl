#!/bin/bash
# Rollout host for the Qwen3.5-4B LoRA GSPO run (2026-10-17).
# One TP1 `vllm serve` per GPU (ports BASE+i) behind rlforge.rollout.router on the trainer head,
# plus one rlforge.rollout.lora_agent (port 8399) that receives each pushed policy adapter.
# Each GPU slot is kept by a bounded restart loop (MAX_RESTARTS); the router drops a dead
# backend and re-loads every live adapter into it when it returns.
# Usage: serve_node.sh RUN_DIR GPU_LIST [BASE_PORT]      e.g. serve_node.sh $RUN 0,1,2,3,4,5,6,7 8200
set -u
RUN=$1; GPUS=$2; BASE=${3:-8200}
KIT=/home/cfs/guoshaoyang/aiq_rl/q35_4b_gspo_20261017/code
MODEL=${MODEL:-/home/tione/guoshaoyang/aiq_rl/models/q35_4b_sft_ep4_merged}
LORA_RANK=${LORA_RANK:-32}; MAX_LORAS=${MAX_LORAS:-5}; MAX_RESTARTS=${MAX_RESTARTS:-4}
GPU_UTIL=${GPU_UTIL:-0.88}; MAX_SEQS=${MAX_SEQS:-256}
# The bundled CUDA forward-compat libs (575) only help on the 535 kernel driver (t1*); perm runs 580.
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "$drv" -ge 575 ] && export A100RL_NO_COMPAT=1
source /home/tione/guoshaoyang/a100_rl/env.sh
export PYTHONPATH=$KIT/src VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
IP=$(hostname -I | awk '{print $1}')
D=$RUN/nodes/$IP
# perm (user ubuntu) cannot write the root-owned NFS run dir: keep its node logs on local disk.
mkdir -p "$D" 2>/dev/null && [ -w "$D" ] || D=/home/tione/guoshaoyang/aiq_rl/q35_nodes/$(basename "$RUN")/$IP
mkdir -p "$D"
RECV=/home/tione/guoshaoyang/aiq_rl/q35_lora_recv/$(basename "$RUN"); mkdir -p "$RECV"
ev() { echo "{\"utc\": \"$(date -u +%FT%TZ)\", \"host\": \"$IP\", $1}" >> "$D/events.jsonl"; }

setsid nohup python -m rlforge.rollout.lora_agent --host "$IP" --port 8399 --root "$RECV" \
  > "$D/lora_agent.log" 2>&1 < /dev/null &
echo $! > "$D/lora_agent.pid"; ev "\"event\": \"lora_agent_start\", \"pid\": $!"

i=0
for g in ${GPUS//,/ }; do
  port=$((BASE + i))
  setsid nohup bash -c '
    g=$1; port=$2; i=$3; D=$4; n=0
    while [ $n -le '"$MAX_RESTARTS"' ]; do
      echo "{\"utc\": \"$(date -u +%FT%TZ)\", \"event\": \"vllm_start\", \"gpu\": $g, \"port\": $port, \"attempt\": $n}" >> $D/events.jsonl
      CUDA_VISIBLE_DEVICES=$g VLLM_HOST_IP='"$IP"' vllm serve '"$MODEL"' --served-model-name '"$MODEL"' \
        --host 0.0.0.0 --port $port --dtype bfloat16 --max-model-len 24576 \
        --gpu-memory-utilization '"$GPU_UTIL"' --max-num-seqs '"$MAX_SEQS"' --enable-prefix-caching \
        --limit-mm-per-prompt "{\"image\": 0, \"video\": 0}" --seed $((20261017 + 1000 * i + n)) \
        --enable-lora --max-lora-rank '"$LORA_RANK"' --max-loras '"$MAX_LORAS"' >> $D/vllm_gpu$g.log 2>&1
      rc=$?
      echo "{\"utc\": \"$(date -u +%FT%TZ)\", \"event\": \"vllm_exit\", \"gpu\": $g, \"port\": $port, \"rc\": $rc, \"attempt\": $n}" >> $D/events.jsonl
      [ -e $D/STOP ] && exit 0
      n=$((n + 1)); sleep 30
    done
    echo "{\"utc\": \"$(date -u +%FT%TZ)\", \"event\": \"vllm_gave_up\", \"gpu\": $g, \"port\": $port}" >> $D/events.jsonl
  ' keeper "$g" "$port" "$i" "$D" > /dev/null 2>&1 < /dev/null &
  echo "$g $port $!" >> "$D/keepers.txt"
  i=$((i + 1))
done
ev "\"event\": \"node_started\", \"gpus\": \"$GPUS\", \"base_port\": $BASE, \"model\": \"$MODEL\""
echo "node $IP started gpus=$GPUS base=$BASE"
