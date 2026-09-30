#!/bin/bash
# rlforge async-GRPO/GSPO launcher for a single shared GPU node, data-parallel trainer.
#
# Layout: vLLM rollout server on $SERVER_GPUS (TP=$TP) + trainer on $TRAINER_GPUS
# with $NUM_TRAINER DP ranks (accelerate). Example for an 8-GPU node:
#   rollout 0,2,3,4 (TP=4) + trainer 1,5,6,7 (DP=4)  -- set explicitly per node.
#
# Required env: ROOT (project dir with data/ logs/ runs/ evals/),
#               VENV (python env with rlforge installed), MODEL (local model path).
#
# Usage: [env overrides] bash run_async_dp.sh smoke|full <suffix>
# Off-policy depth: STALE (default 3). Batch: CPS samples per rank per step
# (gradient accumulation = CPS/NGEN); global completions/step = CPS x NUM_TRAINER.
set -euo pipefail

MODE="${1:-full}"
SUFFIX="${2:-}"
ROOT="${ROOT:?set ROOT}"
VENV="${VENV:?set VENV}"
MODEL="${MODEL:?set MODEL (also export RLFORGE_BASE_MODEL for checkpoint evals)}"
PORT="${PORT:-8000}"

source "$VENV/bin/activate"
cd "$ROOT"
mkdir -p logs runs evals

export TMPDIR="${TMPDIR:-$ROOT/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/tmp/triton}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-$ROOT/tmp/vllm}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_ALLREDUCE_USE_FLASHINFER="${VLLM_ALLREDUCE_USE_FLASHINFER:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TRL_EXPERIMENTAL_SILENCE="${TRL_EXPERIMENTAL_SILENCE:-1}"
mkdir -p "$TMPDIR"

DTYPE="${DTYPE:-none}"
GSPO="${GSPO:-1}"
GSPO_NORM="${GSPO_NORM:-seq_mean}"
GSPO_EPS_LOW="${GSPO_EPS_LOW:-3e-4}"
GSPO_EPS_HIGH="${GSPO_EPS_HIGH:-4e-4}"
CPS="${CPS:-256}"
NGEN="${NGEN:-16}"
STALE="${STALE:-3}"
INFLIGHT="${INFLIGHT:-512}"
EPOCHS="${EPOCHS:-40}"
SERVER_GPUS="${SERVER_GPUS:-0,2,3,4}"
TRAINER_GPUS="${TRAINER_GPUS:-1,5,6,7}"
NUM_TRAINER="${NUM_TRAINER:-4}"
TP="${TP:-4}"
LR="${LR:-2e-6}"
MAX_COMPLETION="${MAX_COMPLETION:-16384}"
# Step ceiling. Left unset, the trainer derives its own "safety ceiling" from EPOCHS and the
# pool size; with the constant LR schedule either value only sets how far the run can go, so
# pass MAX_STEPS explicitly to state the horizon instead of inferring it.
MAX_STEPS="${MAX_STEPS:-0}"

if [ "$MODE" = "smoke" ]; then
  DATA="${DATA:-data/smoke.jsonl}"; OUT=runs/dpsmoke${SUFFIX}; EPOCHS=1; CPS=32; NGEN=8; STALE=3; INFLIGHT=96; SAVE="${SAVE:-100}"
else
  DATA="${DATA:-data/train.jsonl}";  OUT=runs/async_dp${SUFFIX}; SAVE="${SAVE:-50}"
fi

EXTRA_ARGS=""
[ "$DTYPE" = "bfloat16" ] && EXTRA_ARGS="$EXTRA_ARGS --dtype bfloat16"
if [ "$GSPO" = "1" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --gspo --gspo-norm $GSPO_NORM --gspo-eps-low $GSPO_EPS_LOW --gspo-eps-high $GSPO_EPS_HIGH"
fi
[ "$MAX_STEPS" != "0" ] && EXTRA_ARGS="$EXTRA_ARGS --max-steps $MAX_STEPS"
# Must exceed the worst-case single request: max_completion tokens at the per-sequence rate
# implied by MAX_SEQS. At TRL's 120 s default an 8k completion cannot finish when the server
# runs ~1000 sequences, so every request times out and the trainer never sees a first batch.
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"
EXTRA_ARGS="$EXTRA_ARGS --request-timeout $REQUEST_TIMEOUT"

# Per-reward-call task split (mcq vs ranking counts) lands next to the run so the training
# reward column can be decomposed after the fact. Default ON (was opt-in, and the one run
# that needed it did not have it).
export RLFORGE_TASK_LOG="${RLFORGE_TASK_LOG:-$OUT/task_split.jsonl}"
mkdir -p "$OUT"

echo "[dp] mode=$MODE data=$DATA out=$OUT gspo=$GSPO norm=${GSPO_NORM:-} eps=${GSPO_EPS_LOW:-}/${GSPO_EPS_HIGH:-} cps=$CPS ngen=$NGEN stale=$STALE inflight=$INFLIGHT lr=$LR"
echo "[dp] rollout GPUs=$SERVER_GPUS TP=$TP | trainer GPUs=$TRAINER_GPUS ranks=$NUM_TRAINER | max_completion=$MAX_COMPLETION"
echo "[dp] epochs=$EPOCHS max_steps=${MAX_STEPS:-0} save_steps=${SAVE:-50} request_timeout=$REQUEST_TIMEOUT task_log=$RLFORGE_TASK_LOG"

# Throughput knobs. On this workload vLLM's stock ceilings throttle the rollout:
# ~900 sequences run concurrently, but CUDA graphs are only captured up to
# max_cudagraph_capture_size (stock 512) and max_num_batched_tokens defaults to
# 8192, so big-batch steps fall back to eager execution.
VLLM_EXTRA=""
[ -n "${MAX_SEQS:-}" ] && VLLM_EXTRA="$VLLM_EXTRA --max-num-seqs $MAX_SEQS"
[ -n "${MAX_BATCHED:-}" ] && VLLM_EXTRA="$VLLM_EXTRA --max-num-batched-tokens $MAX_BATCHED"
[ -n "${MAX_CG:-}" ] && VLLM_EXTRA="$VLLM_EXTRA --max-cudagraph-capture-size $MAX_CG"
[ "${ASYNC_SCHED:-0}" = "1" ] && VLLM_EXTRA="$VLLM_EXTRA --async-scheduling"
echo "[dp] vllm extra:${VLLM_EXTRA:- (stock)}"

# ---- run manifest + code snapshot (contract A/H): written BEFORE anything starts, so a
# dead-on-arrival run still leaves a record of what was attempted. ----------------------
SNAP="$OUT/code_snapshot"; mkdir -p "$SNAP"
RLFORGE_PKG=$("$VENV/bin/python" -c "import rlforge, os; print(os.path.dirname(rlforge.__file__))")
cp "$RLFORGE_PKG"/*.py "$SNAP/" 2>/dev/null || true
cp "$RLFORGE_PKG"/rewards/*.py "$SNAP/" 2>/dev/null || true
cp "${BASH_SOURCE[0]}" "$SNAP/run_async_dp.sh" 2>/dev/null || true
"$VENV/bin/python" - <<PY
import hashlib, json, os, platform, subprocess, time
def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
def lines(p):
    return sum(1 for _ in open(p, "rb")) if os.path.exists(p) else None
import collections
tasks = collections.Counter()
try:
    with open("$DATA") as fh:
        for line in fh:
            r = json.loads(line)
            tasks["ranking" if "<" in r.get("answer", "") else "mcq"] += 1
except Exception as e:
    tasks["unreadable"] = str(e)
def pkg(v):
    try:
        mod = __import__(v); return getattr(mod, "__version__", "?")
    except Exception:
        return None
manifest = {
    "run_id": os.path.basename("$OUT"),
    "mode": "$MODE",
    "node": platform.node(),
    "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    "stop_reason": None,
    "gpu_layout": {"rollout_gpus": "$SERVER_GPUS", "tp": int("$TP"),
                   "trainer_gpus": "$TRAINER_GPUS", "num_trainer": int("$NUM_TRAINER")},
    "code_md5": {os.path.basename(f): md5(f) for f in
                 sorted(__import__("glob").glob(os.path.join("$RLFORGE_PKG", "*.py")))},
    "engine": {"torch": pkg("torch"), "vllm": pkg("vllm"), "trl": pkg("trl"),
               "transformers": pkg("transformers")},
    "model": {"path": "$MODEL", "md5_config": md5(os.path.join("$MODEL", "config.json"))},
    "data": {"path": "$DATA", "md5": md5("$DATA"), "rows": lines("$DATA"),
             "tasks": dict(tasks)},
    "seed": 0,
    "hyperparams": {
        "gspo": "$GSPO" == "1", "gspo_norm": "${GSPO_NORM:-}",
        "gspo_eps_low": "${GSPO_EPS_LOW:-}", "gspo_eps_high": "${GSPO_EPS_HIGH:-}",
        "cps": int("$CPS"), "ngen": int("$NGEN"), "stale": int("$STALE"),
        "inflight": int("$INFLIGHT"), "lr": "$LR", "epochs": "$EPOCHS",
        "max_steps": "${MAX_STEPS:-0}", "max_completion": int("$MAX_COMPLETION"),
        "request_timeout": "$REQUEST_TIMEOUT", "save_steps": "${SAVE:-50}",
        "save_total_limit": 4, "max_seqs": "${MAX_SEQS:-stock}",
        "max_batched": "${MAX_BATCHED:-stock}", "max_cg": "${MAX_CG:-stock}",
    },
    "reward": {"correct": 1.0, "wrong": 0.0, "unparsed": -0.5, "truncated": -2.0,
               "ranking": "exact=+1 else concordant/5-1", "cap": int("$MAX_COMPLETION")},
    "command": "GSPO=$GSPO CPS=$CPS NGEN=$NGEN STALE=$STALE INFLIGHT=$INFLIGHT LR=$LR "
               "MAX_COMPLETION=$MAX_COMPLETION MAX_STEPS=${MAX_STEPS:-0} "
               "bash run_async_dp.sh $MODE $SUFFIX",
}
json.dump(manifest, open(os.path.join("$OUT", "run.json"), "w"), indent=1)
print("[dp] run.json written to $OUT/run.json")
PY

CUDA_VISIBLE_DEVICES=$SERVER_GPUS VLLM_SERVER_DEV_MODE=1 \
  vllm serve "$MODEL" \
    --served-model-name "$MODEL" \
    --dtype bfloat16 \
    --max-model-len 24576 \
    --gpu-memory-utilization 0.90 \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --weight-transfer-config '{"backend":"nccl"}' \
    $VLLM_EXTRA \
    > "logs/vllm_dp${SUFFIX}.log" 2>&1 &
SERVER_PID=$!

STOP_REASON="trainer exited"
record_stop() {
  "$VENV/bin/python" - "$OUT" "$STOP_REASON" <<'PY' 2>/dev/null || true
import json, os, sys, time
out, reason = sys.argv[1], sys.argv[2]
p = os.path.join(out, "run.json")
try:
    m = json.load(open(p))
    m["stop_reason"] = reason
    m["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(m, open(p, "w"), indent=1)
except Exception:
    pass
PY
}
trap 'STOP_REASON="killed (signal)"; record_stop; kill "$SERVER_PID" 2>/dev/null || true' EXIT
trap 'STOP_REASON="killed (SIGTERM)"; record_stop; kill "$SERVER_PID" 2>/dev/null; exit 143' TERM
trap 'STOP_REASON="killed (SIGINT)";  record_stop; kill "$SERVER_PID" 2>/dev/null; exit 130' INT

echo "[dp] waiting for vLLM server (pid $SERVER_PID)..."
for i in $(seq 1 180); do
  if curl -sf "localhost:$PORT/health" > /dev/null; then echo "[dp] server up"; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[dp] server died"; STOP_REASON="vllm server died at startup"; record_stop
    tail -60 "logs/vllm_dp${SUFFIX}.log"; exit 1
  fi
  sleep 5
done
if ! curl -sf "localhost:$PORT/health" > /dev/null; then
  echo "[dp] server never healthy"; STOP_REASON="vllm server never healthy"; record_stop; exit 1
fi

set +e
CUDA_VISIBLE_DEVICES=$TRAINER_GPUS accelerate launch \
    --num_processes "$NUM_TRAINER" --mixed_precision no --dynamo_backend no \
    -m rlforge.trainer \
  --model "$MODEL" \
  --train "$DATA" \
  --out "$OUT" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --completions-per-step "$CPS" \
  --max-completion "$MAX_COMPLETION" \
  --num-generations "$NGEN" \
  --save-steps "$SAVE" \
  --max-staleness "$STALE" \
  --max-inflight-tasks "$INFLIGHT" \
  $EXTRA_ARGS \
  2>&1 | tee "logs/trainer_dp${SUFFIX}.log"
TRAINER_RC=${PIPESTATUS[0]}
set -e
[ "$TRAINER_RC" = "0" ] || STOP_REASON="trainer exited rc=$TRAINER_RC"
record_stop
echo "[dp] done (trainer rc=$TRAINER_RC)"
exit "$TRAINER_RC"
