#!/usr/bin/env bash
# Usage: serve_backend.sh NAME GPUS PORT MODEL TP MAXSEQS [extra vllm args...]
# Starts one LoRA-enabled vLLM backend under setsid; writes RUN/<NAME>.pid ("pid pgid") and cmd.
# Refuses unless every requested GPU shows <100 MiB and 0% util right now (caller did the 60 s double check).
set -u
NAME=$1; GPUS=$2; PORT=$3; MODEL=$4; TP=$5; MAXSEQS=$6; shift 6
source /home/tione/guoshaoyang/a100_rl/env.sh
E2E=${E2E:-/home/tione/guoshaoyang/a100_rl/wave2/e2e}
RUN=${RUN:-$E2E/runs/servers}
mkdir -p $RUN
for g in ${GPUS//,/ }; do
  read -r used util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i $g | tr -d ',')
  if [ "$used" -ge 100 ] || [ "$util" -gt 0 ]; then echo "REFUSE: gpu$g busy (used=${used}MiB util=$util%)"; exit 3; fi
done
CMD=(vllm serve $MODEL --port $PORT --host 0.0.0.0 --served-model-name base --language-model-only
     --tensor-parallel-size $TP --max-model-len ${MAXLEN:-10240} --gpu-memory-utilization ${UTIL:-0.90}
     --enable-lora --max-lora-rank 16 --max-loras ${MAXLORAS:-2} --max-cpu-loras ${MAXCPULORAS:-4}
     --max-num-seqs $MAXSEQS --max-cudagraph-capture-size ${MAXCG:-64} --seed 0 "$@")
echo "CUDA_VISIBLE_DEVICES=$GPUS ${CMD[*]}" > $RUN/$NAME.cmd
CUDA_VISIBLE_DEVICES=$GPUS setsid nohup "${CMD[@]}" > $RUN/$NAME.log 2>&1 < /dev/null &
PID=$!
sleep 0.5
echo "$PID $(ps -o pgid= -p $PID | tr -d ' ')" > $RUN/$NAME.pid
echo "started $NAME pid/pgid $(cat $RUN/$NAME.pid) gpus=$GPUS port=$PORT"
