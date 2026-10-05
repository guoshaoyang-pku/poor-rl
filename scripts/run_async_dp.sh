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
REWARD="${REWARD:-rlforge.rewards.mcq:mcq_reward}"
PORT="${PORT:-8000}"

source "$VENV/bin/activate"
cd "$ROOT"
mkdir -p logs runs evals

export TMPDIR="${TMPDIR:-$ROOT/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/tmp/triton}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-$ROOT/tmp/vllm}"
# FI_SAMPLER=1 enables the batched flashinfer sampler -- needs nvcc (JIT); offline nodes
# keep 0. Respects a pre-set VLLM_USE_FLASHINFER_SAMPLER from the node env file.
export VLLM_USE_FLASHINFER_SAMPLER="${FI_SAMPLER:-${VLLM_USE_FLASHINFER_SAMPLER:-0}}"
export VLLM_ALLREDUCE_USE_FLASHINFER="${VLLM_ALLREDUCE_USE_FLASHINFER:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TRL_EXPERIMENTAL_SILENCE="${TRL_EXPERIMENTAL_SILENCE:-1}"
REPORT_TO="${REPORT_TO:-swanlab}"
case ",$REPORT_TO," in
  *,swanlab,*)
    export SWANLAB_MODE="${SWANLAB_MODE:-local}"
    export SWANLAB_LOGDIR="${SWANLAB_LOGDIR:-$ROOT/swanlog}"
    export SWANLAB_PROJ_NAME="${SWANLAB_PROJ_NAME:-AIQ}"
    mkdir -p "$SWANLAB_LOGDIR"
    ;;
esac
mkdir -p "$TMPDIR"

DTYPE="${DTYPE:-none}"
# Trainer precision recipe (see docs/PRECISION.md):
#   DTYPE=none    + MIXED_PRECISION=no   -> pure fp32 (arm-A style; slowest, safest)
#   DTYPE=bfloat16+ MIXED_PRECISION=no   -> pure bf16 (fast; tiny updates can vanish)
#   DTYPE=none    + MIXED_PRECISION=bf16 -> fp32 master weights + bf16 autocast compute
#     (the recipe for low-precision work: small lr updates land on fp32 masters)
MIXED_PRECISION="${MIXED_PRECISION:-no}"
# Perf knobs (docs/OPTIMIZATION.md). Defaults = validated-safe on H200.
LIGER="${LIGER:-0}"            # 1 = liger base kernels (rmsnorm/rope/swiglu)
GRAD_CKPT="${GRAD_CKPT:-1}"    # 0 = disable grad ckpt (~30%% trainer, needs VRAM)
OPTIM="${OPTIM:-}"             # e.g. adamw_torch_fused
TF32="${TF32:-0}"              # 1 = allow tf32 (fp32-master recipe)
DYNAMO="${DYNAMO:-no}"         # inductor = torch.compile via accelerate (test first)
NO_THINKING="${NO_THINKING:-0}"
# FP8 KV is validated on H200; set KV_DTYPE=auto on unsupported hardware or to compare.
KV_DTYPE="${KV_DTYPE:-fp8}"
ROLLOUT_QUANTIZATION="${ROLLOUT_QUANTIZATION:-none}" # none or fp8; vLLM weight quantization
FP8="${FP8:-off}" # native: matched FP8 trainer and poor_rl_fp8 rollout
FP8_ALIGN="${FP8_ALIGN:-off}"
case "$FP8" in off|native) ;; *) echo "FP8 must be off or native" >&2; exit 2 ;; esac
case "$FP8_ALIGN" in off|vllm) ;; *) echo "FP8_ALIGN must be off or vllm" >&2; exit 2 ;; esac
[ "$FP8_ALIGN" = "off" ] || [ "$FP8" = "native" ] || { echo "FP8_ALIGN=vllm requires FP8=native" >&2; exit 2; }
if [ "$FP8" = "native" ]; then
  [ "$DTYPE" = "none" ] || { echo "FP8=native requires DTYPE=none" >&2; exit 2; }
  [ "${TP:-1}" = "1" ] || { echo "FP8=native requires TP=1" >&2; exit 2; }
  TP=1
  MIXED_PRECISION=bf16
  ROLLOUT_QUANTIZATION=poor_rl_fp8
  KV_DTYPE=auto
  export RLFORGE_FAST_LOGPROB=1
  # vLLM AOT validation misses plugin constructor changes such as embedding dtype.
  export VLLM_DISABLE_COMPILE_CACHE=1
fi
# LoRA (PEFT adapter training). LORA=1 freezes the base model, trains an fp32 adapter
# (PEFT keeps adapters fp32 when the base is bf16) and -- with the server-side flags added
# below -- syncs only the adapter each step (~1% of the bytes). The base can still be served
# under ROLLOUT_QUANTIZATION=fp8; measure the resulting train/rollout logprob gap.
LORA="${LORA:-0}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-0}"   # 0 = 2 x r
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}"
GSPO="${GSPO:-1}"
if [ "$FP8" = "native" ]; then
  [ "$GSPO" = "1" ] && [ "$LORA" = "0" ] \
    || { echo "FP8=native currently requires GSPO=1 and LORA=0" >&2; exit 2; }
fi
GSPO_NORM="${GSPO_NORM:-seq_mean}"
GSPO_EPS_LOW="${GSPO_EPS_LOW:-3e-4}"
GSPO_EPS_HIGH="${GSPO_EPS_HIGH:-4e-4}"
ADAPT_CLIP_LOW_MAX_FRAC="${ADAPT_CLIP_LOW_MAX_FRAC:-}"
ADAPT_CLIP_HIGH_MAX_FRAC="${ADAPT_CLIP_HIGH_MAX_FRAC:-}"
GSPO_EPS_MAX="${GSPO_EPS_MAX:-0.1}"
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
[ "$FP8" = "native" ] && EXTRA_ARGS="$EXTRA_ARGS --fp8 native --prefix-share on --token-budget 0"
[ "$FP8_ALIGN" = "vllm" ] && EXTRA_ARGS="$EXTRA_ARGS --fp8-align vllm"
if [ "$GSPO" = "1" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --gspo --gspo-norm $GSPO_NORM --gspo-eps-low $GSPO_EPS_LOW --gspo-eps-high $GSPO_EPS_HIGH"
  if [ -n "$ADAPT_CLIP_LOW_MAX_FRAC" ] || [ -n "$ADAPT_CLIP_HIGH_MAX_FRAC" ]; then
    if [ -z "$ADAPT_CLIP_LOW_MAX_FRAC" ] || [ -z "$ADAPT_CLIP_HIGH_MAX_FRAC" ]; then
      echo "set both adaptive clip fraction caps" >&2; exit 2
    fi
    EXTRA_ARGS="$EXTRA_ARGS --adaptive-clip-low-max $ADAPT_CLIP_LOW_MAX_FRAC --adaptive-clip-high-max $ADAPT_CLIP_HIGH_MAX_FRAC --gspo-eps-max $GSPO_EPS_MAX"
  fi
fi
[ "$MAX_STEPS" != "0" ] && EXTRA_ARGS="$EXTRA_ARGS --max-steps $MAX_STEPS"
[ "$LIGER" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --use-liger"
[ "$GRAD_CKPT" = "0" ] && EXTRA_ARGS="$EXTRA_ARGS --no-grad-ckpt"
[ -n "$OPTIM" ] && EXTRA_ARGS="$EXTRA_ARGS --optim $OPTIM"
[ "$TF32" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --allow-tf32"
[ "$NO_THINKING" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --no-thinking"
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
echo "[dp] perf: liger=$LIGER grad_ckpt=$GRAD_CKPT optim=${OPTIM:-default} tf32=$TF32 dynamo=$DYNAMO fi_sampler=$VLLM_USE_FLASHINFER_SAMPLER kv_dtype=${KV_DTYPE:-auto} rollout_quantization=$ROLLOUT_QUANTIZATION"

# Throughput knobs. On this workload vLLM's stock ceilings throttle the rollout:
# ~900 sequences run concurrently, but CUDA graphs are only captured up to
# max_cudagraph_capture_size (stock 512) and max_num_batched_tokens defaults to
# 8192, so big-batch steps fall back to eager execution.
# KV cache: KV_DTYPE=fp8 halves KV memory on Hopper/Blackwell (more concurrent
# sequences); GPU_MEM_UTIL caps the rollout server's VRAM share.
VLLM_EXTRA="${VLLM_EXTRA:-}"
FP8_VLLM_ARGS=()
ROLLOUT_DP=1
if [ "$FP8" = "native" ]; then
  IFS=, read -r -a FP8_ROLLOUT_GPUS <<< "$SERVER_GPUS"
  ROLLOUT_DP="${#FP8_ROLLOUT_GPUS[@]}"
  FP8_VLLM_ARGS=(--data-parallel-size "$ROLLOUT_DP" --data-parallel-size-local "$ROLLOUT_DP" --api-server-count "$ROLLOUT_DP")
fi
[ "$FP8_ALIGN" = "vllm" ] && FP8_VLLM_ARGS+=(--additional-config '{"gdn_prefill_backend":"triton"}')
[ -n "${MAX_SEQS:-}" ] && VLLM_EXTRA="$VLLM_EXTRA --max-num-seqs $MAX_SEQS"
[ -n "${MAX_BATCHED:-}" ] && VLLM_EXTRA="$VLLM_EXTRA --max-num-batched-tokens $MAX_BATCHED"
[ -n "${MAX_CG:-}" ] && VLLM_EXTRA="$VLLM_EXTRA --max-cudagraph-capture-size $MAX_CG"
[ "${ASYNC_SCHED:-0}" = "1" ] && VLLM_EXTRA="$VLLM_EXTRA --async-scheduling"
case "$ROLLOUT_QUANTIZATION" in
  none) ;;
  fp8) VLLM_EXTRA="$VLLM_EXTRA --quantization fp8" ;;
  poor_rl_fp8) VLLM_EXTRA="$VLLM_EXTRA --quantization poor_rl_fp8" ;;
  *) echo "ROLLOUT_QUANTIZATION must be none, fp8 or poor_rl_fp8 (got $ROLLOUT_QUANTIZATION)" >&2; exit 2 ;;
esac
[ -n "${KV_DTYPE:-}" ] && [ "$KV_DTYPE" != "auto" ] && VLLM_EXTRA="$VLLM_EXTRA --kv-cache-dtype $KV_DTYPE"
if [ "$LORA" = "1" ]; then
  # Adapter-only sync: the trainer saves each policy version under $OUT/.vllm_lora and the
  # server loads it over the HTTP API, so the server must allow runtime adapter updates and
  # hold every version an in-flight request can still name (max_staleness + 2, per TRL).
  # --max-lora-rank must be one of vLLM's stacked-buffer ranks and >= the adapter's rank.
  export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
  LORA_RANK="$LORA_R"
  for cand in 1 8 16 32 64 128 256 320 512; do
    if [ "$cand" -ge "$LORA_R" ]; then LORA_RANK="$cand"; break; fi
  done
  [ "$LORA_RANK" -ge "$LORA_R" ] || { echo "LORA_R $LORA_R exceeds vLLM's max rank 512" >&2; exit 2; }
  LORA_SLOTS=$((STALE + 2))
  VLLM_EXTRA="$VLLM_EXTRA --enable-lora --max-lora-rank $LORA_RANK --max-loras $LORA_SLOTS"
  EXTRA_ARGS="$EXTRA_ARGS --lora --lora-r $LORA_R --lora-dropout $LORA_DROPOUT --lora-target-modules $LORA_TARGET_MODULES"
  [ "$LORA_ALPHA" != "0" ] && EXTRA_ARGS="$EXTRA_ARGS --lora-alpha $LORA_ALPHA"
  echo "[dp] lora: r=$LORA_R alpha=${LORA_ALPHA:-auto} rank_cap=$LORA_RANK slots=$LORA_SLOTS targets=$LORA_TARGET_MODULES"
  # Coverage preflight: a target list that is correct for a pure transformer (q/k/v/o + MLP)
  # matches NOTHING on the GatedDeltaNet token mixers of a 3:1 hybrid Qwen3.5/3.8 model, so
  # the run would train a partial adapter while reporting a plausible-looking size.
  "$VENV/bin/python" - "$MODEL" "$LORA_TARGET_MODULES" <<'PY' || exit 2
import sys

from transformers import AutoConfig

model, targets = sys.argv[1], {t.strip() for t in sys.argv[2].split(",") if t.strip()}
cfg = AutoConfig.from_pretrained(model)
tc = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
kinds = set(getattr(tc, "layer_types", None) or [])
gaps = []
if "linear_attention" in kinds and "in_proj_qkv" not in targets:
    gaps.append("linear_attention (GatedDeltaNet) token mixers are uncovered: add "
                "in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj")
if "full_attention" in kinds and not {"q_proj", "k_proj", "v_proj", "o_proj"} <= targets:
    gaps.append("full_attention projections are uncovered: add q_proj,k_proj,v_proj,o_proj")
if "linear_attention" in kinds and "in_proj_qkv" in targets and "out_proj" not in targets:
    gaps.append("in_proj_* is covered but out_proj (the DeltaNet output projection) is not")
if gaps:
    print("[dp] LORA COVERAGE ERROR:")
    for g in gaps:
        print("[dp]   - " + g)
    sys.exit(1)

print(f"[dp] lora coverage OK for layer kinds {sorted(kinds) or ['attention-only']}")
PY
fi
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
    "gpu_layout": {"rollout_gpus": "$SERVER_GPUS", "tp": int("$TP"), "dp": int("$ROLLOUT_DP"),
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
        "adaptive_clip_low_max_frac": "${ADAPT_CLIP_LOW_MAX_FRAC:-}",
        "adaptive_clip_high_max_frac": "${ADAPT_CLIP_HIGH_MAX_FRAC:-}",
        "gspo_eps_max": "${GSPO_EPS_MAX:-0.1}",
        "cps": int("$CPS"), "ngen": int("$NGEN"), "stale": int("$STALE"),
        "inflight": int("$INFLIGHT"), "lr": "$LR", "epochs": "$EPOCHS",
        "max_steps": "${MAX_STEPS:-0}", "max_completion": int("$MAX_COMPLETION"),
        "request_timeout": "$REQUEST_TIMEOUT", "save_steps": "${SAVE:-50}",
        "save_total_limit": 4, "max_seqs": "${MAX_SEQS:-stock}",
        "max_batched": "${MAX_BATCHED:-stock}", "max_cg": "${MAX_CG:-stock}",
        "trainer_weight_dtype": "$DTYPE", "trainer_mixed_precision": "$MIXED_PRECISION",
        "trainer_fp8": "$FP8",
        "trainer_fp8_alignment": "$FP8_ALIGN",
        "fp8_graphs": os.environ.get("RLFORGE_FP8_GRAPHS", "0"),
        "fp8_graph_max_mb": os.environ.get("RLFORGE_FP8_GRAPH_MAX_MB", "1024"),
        "fp8_head_graph": os.environ.get("RLFORGE_FP8_HEAD_GRAPH", "0"),
        "fp8_fuse_mlp": os.environ.get("RLFORGE_FP8_FUSE_MLP", "1"),
        "fp8_align_compile": os.environ.get("RLFORGE_FP8_ALIGN_COMPILE", "0"),
        "fp8_align_pointwise": os.environ.get("RLFORGE_FP8_ALIGN_POINTWISE", "0"),
        "fp8_align_backward": os.environ.get("RLFORGE_FP8_ALIGN_BACKWARD", "0"),
        "vllm_disable_compile_cache": os.environ.get("VLLM_DISABLE_COMPILE_CACHE", "0"),
        "sb_activation_budget_gib": os.environ.get("RLFORGE_SB_ACT_GB", "80"),
        "no_thinking": "$NO_THINKING" == "1",
        "tracker_backend": "$REPORT_TO", "swanlab_mode": "${SWANLAB_MODE:-disabled}",
        "rollout_dtype": "bfloat16", "rollout_quantization": "$ROLLOUT_QUANTIZATION",
        "kv_cache_dtype": "${KV_DTYPE:-auto}",
        "lora": "$LORA" == "1", "lora_r": int("$LORA_R"),
        "lora_alpha": "$LORA_ALPHA", "lora_dropout": "$LORA_DROPOUT",
        "lora_target_modules": "$LORA_TARGET_MODULES",
    },
    "reward": {"correct": 1.0, "wrong": 0.0, "unparsed": -0.5, "truncated": -2.0,
               "ranking": "exact=+1 else concordant/5-1", "cap": int("$MAX_COMPLETION")},
    "command": "FP8=$FP8 FP8_ALIGN=$FP8_ALIGN DTYPE=$DTYPE TP=$TP "
               "SERVER_GPUS=$SERVER_GPUS TRAINER_GPUS=$TRAINER_GPUS NUM_TRAINER=$NUM_TRAINER "
               "RLFORGE_FP8_GRAPHS=${RLFORGE_FP8_GRAPHS:-0} RLFORGE_FP8_GRAPH_MAX_MB=${RLFORGE_FP8_GRAPH_MAX_MB:-1024} "
               "RLFORGE_FP8_HEAD_GRAPH=${RLFORGE_FP8_HEAD_GRAPH:-0} RLFORGE_FP8_FUSE_MLP=${RLFORGE_FP8_FUSE_MLP:-1} "
               "RLFORGE_FP8_ALIGN_COMPILE=${RLFORGE_FP8_ALIGN_COMPILE:-0} "
               "RLFORGE_FP8_ALIGN_POINTWISE=${RLFORGE_FP8_ALIGN_POINTWISE:-0} RLFORGE_FP8_ALIGN_BACKWARD=${RLFORGE_FP8_ALIGN_BACKWARD:-0} "
               "RLFORGE_SB_ACT_GB=${RLFORGE_SB_ACT_GB:-80} "
               "GSPO=$GSPO CPS=$CPS NGEN=$NGEN STALE=$STALE INFLIGHT=$INFLIGHT LR=$LR "
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
    "${FP8_VLLM_ARGS[@]}" \
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
# REPORT_TO defaults to local SwanLab; override with tensorboard/wandb/mlflow/none as needed.
# ACCELERATE_CONFIG: when set, replaces the inline accelerate flags entirely (e.g.
# an FSDP config from examples/accelerate/).
if [ -n "${ACCELERATE_CONFIG:-}" ]; then
  ACC_LAUNCH=(accelerate launch --config_file "$ACCELERATE_CONFIG" --num_processes "$NUM_TRAINER")
else
  ACC_LAUNCH=(accelerate launch --num_processes "$NUM_TRAINER" --mixed_precision "$MIXED_PRECISION" --dynamo_backend "$DYNAMO")
fi
CUDA_VISIBLE_DEVICES=$TRAINER_GPUS "${ACC_LAUNCH[@]}" \
    -m rlforge.trainer \
  --model "$MODEL" \
  --server-url "http://localhost:$PORT" \
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
  --report-to "${REPORT_TO:-swanlab}" \
  $EXTRA_ARGS \
  2>&1 | tee "logs/trainer_dp${SUFFIX}.log"
TRAINER_RC=${PIPESTATUS[0]}
set -e
[ "$TRAINER_RC" = "0" ] || STOP_REASON="trainer exited rc=$TRAINER_RC"
record_stop
echo "[dp] done (trainer rc=$TRAINER_RC)"
exit "$TRAINER_RC"
