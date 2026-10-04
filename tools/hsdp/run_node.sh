#!/usr/bin/env bash
# Launch ONE node of a multi-host HSDP bench (static torchrun rendezvous). Run it on every host (from the Mac via ssh),
# node 0 first. NOT RUN YET by the reviewer (GPU use was not authorized in the review session).
# usage: run_node.sh <tag> <node_rank> <nnodes> <gpus csv> <master_ip> <master_port> -- <bench_hsdp.py args...>
# example (2 hosts x 4 GPUs, R=2 S=4):
#   ssh a100_t1   'bash .../run_node.sh hsdp_2x4 0 2 1,2,3,4 172.16.0.116 29761 -- --comp-lens 2048,8192 --micro 4'
#   ssh a100_t1_2 'bash .../run_node.sh hsdp_2x4 1 2 0,1,2,3 172.16.0.116 29761 -- --comp-lens 2048,8192 --micro 4'
# Same-code single-host baseline (needed to attribute any slowdown to the cross-host AR):
#   ssh a100_t1   'bash .../run_node.sh fsdp_1x4 0 1 1,2,3,4 127.0.0.1 29763 -- --replicate 1 --comp-lens 2048,8192 --micro 4'
set -u
TAG=$1; NR=$2; NN=$3; GPUS=$4; MADDR=$5; PORT=$6; shift 6; [ "${1:-}" = "--" ] && shift
source /home/tione/guoshaoyang/a100_rl/env.sh
D=${HSDP_DIR:-/home/tione/guoshaoyang/a100_rl/wave2/hsdp}
RD=$D/runs/$TAG; mkdir -p $RD
# GPU RULE: every requested GPU < 100 MiB and 0 % util on two checks 60 s apart, else refuse.
busy() { nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
         awk -F', ' -v g=",$GPUS," 'index(g, ","$1",") && ($2 >= 100 || $3 != 0) {print $1}'; }
B1=$(busy); sleep 60; B2=$(busy)
if [ -n "$B1$B2" ]; then echo "REFUSE: GPUs busy: $B1 $B2" | tee $RD/refused.node$NR.txt; exit 2; fi
NP=$(echo $GPUS | tr ',' '\n' | wc -l)
export CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# no /dev/infiniband in these containers (mlx5_bond_* only visible in sysfs) -> NET/Socket over eth0
export NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1 NCCL_DEBUG=${NCCL_DEBUG:-INFO} NCCL_DEBUG_SUBSYS=INIT,NET
export HOST_TAG=$(hostname)
CMD=(torchrun --nnodes $NN --node_rank $NR --nproc_per_node $NP --master_addr $MADDR --master_port $PORT
     $D/code/bench_hsdp.py --out $RD/result.jsonl --tag $TAG "$@")
{ echo "# $(date -Is) host=$(hostname) node_rank=$NR"; env | grep -E '^(CUDA_|NCCL_|GLOO_|PYTORCH_|OMP_)' | sort
  printf '%q ' "${CMD[@]}"; echo; } > $RD/cmd.node$NR.txt
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $RD/nvsmi_before.node$NR.csv
setsid "${CMD[@]}" > $RD/train.node$NR.log 2>&1 < /dev/null &
PID=$!; echo $PID > $RD/run.node$NR.pid
echo "started tag=$TAG node=$NR pid=$PID pgid=$(ps -o pgid= -p $PID | tr -d ' ') log=$RD/train.node$NR.log"
# stop only this job:  kill -TERM -- -$(cat $RD/run.node$NR.pid)
