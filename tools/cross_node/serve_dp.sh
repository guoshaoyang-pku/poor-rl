#!/usr/bin/env bash
# One vLLM data-parallel rollout server whose replicas may span hosts.
#
#   head (owns the HTTP API, LB, coordinator; runs DP_LOCAL replicas, ranks 0..DP_LOCAL-1):
#     ROLE=head     GPUS=2     DP_SIZE=2 DP_LOCAL=1 HEAD_IP=10.234.161.3 PORT=8201 bash serve_dp.sh
#   remote (headless; runs DP_LOCAL replicas starting at START_RANK, no HTTP):
#     ROLE=headless GPUS=2     DP_SIZE=2 DP_LOCAL=1 START_RANK=1 HEAD_IP=10.234.161.3 bash serve_dp.sh
#
# TRL keeps talking to one URL (the head). /get_world_size reports TP*DP, so TRL's NCCL
# weight-transfer group already includes every replica on every host, and pause / resume /
# update_weights are broadcast to all engines by the DP client (call_utility_async).
# Dense models: each DP rank is an independent engine (no cross-rank collectives during
# generation); only the weight-sync broadcast crosses the wire.
set -u
: "${ROLE:?head|headless}" "${GPUS:?}" "${DP_SIZE:?}" "${DP_LOCAL:?}" "${HEAD_IP:?}"
V=${VENV:-/data/home/guoshaoyang/aiq_rl/venv}
MODEL=${MODEL:-/data/home/guoshaoyang/models/Qwen3.5-0.8B-ms}
LOGDIR=${LOGDIR:-/data/home/guoshaoyang/crossnode_20261003}
RPC_PORT=${RPC_PORT:-13345}
NAME=${NAME:-dp_${ROLE}}
mkdir -p "$LOGDIR" "$LOGDIR/tmp"

# Pin every control/bootstrap socket to the bond4 network both hosts share (docker0 also
# exists and is unroutable across hosts). NCCL data still goes over IB (mlx5_*); the bonded
# RoCE device is Ethernet on both hosts and is excluded so the two sides agree on IB.
MYIP=$(ip -4 -o addr show bond4 | awk '{print $4}' | cut -d/ -f1)
export VLLM_HOST_IP=$MYIP NCCL_SOCKET_IFNAME=bond4 GLOO_SOCKET_IFNAME=bond4 \
       NCCL_IB_HCA=${NCCL_IB_HCA:-^mlx5_bond_0} \
       VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0 VLLM_SERVER_DEV_MODE=1 \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       TMPDIR=$LOGDIR/tmp VLLM_CACHE_DIR=${VLLM_CACHE_DIR:-/data/home/guoshaoyang/aiq_rl/tmp/vllm} \
       CUDA_VISIBLE_DEVICES=$GPUS

COMMON=(--served-model-name "${SERVED_NAME:-t}" --dtype bfloat16 --max-model-len "${MAX_LEN:-18432}"
        --gpu-memory-utilization "${UTIL:-0.85}"
        --weight-transfer-config '{"backend":"nccl"}'
        --data-parallel-size "$DP_SIZE" --data-parallel-size-local "$DP_LOCAL"
        --data-parallel-address "$HEAD_IP" --data-parallel-rpc-port "$RPC_PORT"
        ${VLLM_EXTRA:-})

if [ "$ROLE" = head ]; then
  nohup "$V/bin/vllm" serve "$MODEL" --host 0.0.0.0 --port "${PORT:-8201}" \
    --api-server-count "${API_SERVERS:-1}" "${COMMON[@]}" > "$LOGDIR/$NAME.log" 2>&1 &
else
  nohup "$V/bin/vllm" serve "$MODEL" --headless \
    --data-parallel-start-rank "${START_RANK:?}" "${COMMON[@]}" > "$LOGDIR/$NAME.log" 2>&1 &
fi
echo $! > "$LOGDIR/$NAME.pid"
echo "[serve_dp] $ROLE pid $(cat "$LOGDIR/$NAME.pid") on $(hostname -s) GPUs=$GPUS ip=$MYIP log=$LOGDIR/$NAME.log"
