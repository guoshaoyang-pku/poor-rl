#!/bin/bash
# Stop this host's rollout processes started by serve_node.sh for RUN_DIR (scoped: only its own pids).
RUN=$1
IP=$(hostname -I | awk '{print $1}'); D=$RUN/nodes/$IP
# perm (user ubuntu) cannot write the root-owned NFS run dir: keep its node logs on local disk.
mkdir -p "$D" 2>/dev/null && [ -w "$D" ] || D=/home/tione/guoshaoyang/aiq_rl/q35_nodes/$(basename "$RUN")/$IP
mkdir -p "$D"
touch "$D/STOP"
while read -r g port pid; do kill -TERM -- "-$pid" 2>/dev/null; done < "$D/keepers.txt"
[ -f "$D/lora_agent.pid" ] && kill -TERM -- "-$(cat "$D/lora_agent.pid")" 2>/dev/null
sleep 15
while read -r g port pid; do kill -KILL -- "-$pid" 2>/dev/null; done < "$D/keepers.txt"
echo "{\"utc\": \"$(date -u +%FT%TZ)\", \"host\": \"$IP\", \"event\": \"node_stopped\"}" >> "$D/events.jsonl"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
