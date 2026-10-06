#!/bin/bash
# rlforge async GSPO launcher -- v3.2 (2026-10-03): a COPY of run_g32_v3_1.sh (md5 da30395d, the launcher of the
# live v3.1e run; left untouched) + the v3_2 trainer-throughput options. Algorithm and every v3.1 default are
# unchanged; the differences are trainer-side compute only (aiq_rl/docs/reports/V3_2_THROUGHPUT_2026-10-03.md):
#   * code tree rlforge_v3_2 (= rlforge_v3_1 + prefix_share.plan_subbatches/subbatch_logprobs/BalancedGroupRowBatcher
#     + rlforge/fast_logprob.py + trainer._compute_loss_subbatched). PREFIX_SHARE_MD5 = the v3_2 file (gate:
#     rlforge_v3_2/scripts/v32_gate.py, results in the report).
#   * FAST_LOGPROB=1 (default): fused log-prob/entropy, bf16 GEMM logits + TF32 backward GEMMs (RLFORGE_LOGPROB_BWD).
#   * SUBBATCH_TOKENS=N (default 131072): forward+backward per token-budgeted sub-batch of one prompt group;
#     SB_CKPT=auto (default) keeps the activations of as many layers as fit RLFORGE_SB_ACT_GB.
#   * BALANCE_ROWS=on (default): rank load balancing inside each micro-batch (BalancedGroupRowBatcher; same samples
#     per micro-batch and per step, groups may be split over ranks; loss = exact per-step seq mean).
#   * V32=0 turns all three off -> the v3_1 trainer path (same code tree).
# rlforge async GSPO launcher -- v3.1 (2026-10-03): a COPY of run_g32_v3.sh (md5 996889f1, the launcher of
# the live v3 run; left untouched) + the drop-audit fixes and the non-blocking judge scorer. With its defaults
# (judge off: AIQ_HALLUC=0, SCORE_CONC=0, REWARD=aiq_think_reward) the trained algorithm is the v3 one; the
# differences are logging/guards only:
#   * code tree rlforge_v3_1 (= rlforge_v3_prod + drop-audit fixes from rlforge_v3_next + rlforge/score_loop.py;
#     prefix_share.py unchanged, gate md5 ede53a03 still enforced).
#   * drop audit ON by default (DROP_AUDIT=on -> --drop-audit on; v3 needed RLFORGE_DROP_AUDIT=on by hand).
#   * v3 relaunch values are the defaults: INFLIGHT=640, QUEUE_MAXSIZE=512 (v3 file: 4096 / TRL 1024).
#   * RUN_NAME has no default (v3's default is the live run's name) and an existing $OUT/run.json is refused.
#   * NUM_TRAINER follows TRAINER_GPUS; optional GROUPS_PER_STEP=G pins the step size (guard, below).
#   * scorer knobs SCORE_CONC / JUDGED_STALE / EARLY_HOOKS / SCORE_TASK_MAX_S and every AIQ_HALLUC_* value are
#     passed through and recorded in run.json (infra.nonblocking_scorer / infra.drop_audit / infra.v3_1).
#   * DRY=1 reports busy ports like busy GPUs (prints the plan while another run holds 8011/13411).
# v3 header (unchanged below):
# rlforge async GSPO launcher -- v3 infra line, 2026-10-03.
# Derived from run_g32_think.sh (left untouched) and launch_g32_v1.sh (v2 one-click defaults).
# Self-contained: sources env_360_2.sh and carries the v3 defaults, so a launch needs only RUN_NAME
# (and MAX_STEPS/SAVE for a smoke). Differences vs v2 (run_g32_think.sh + launch_g32_v1.sh):
#   * code: rlforge_v3/src first on PYTHONPATH (prefix_share.py + dp_route.py + trainer.py with
#     --prefix-share/--token-budget/--dp-route/--queue-maxsize); preflight asserts rlforge resolves there.
#   * trainer: --prefix-share on --token-budget 0 (one whole group per row, GroupRowBatcher), 4 ranks
#     on 360-2 GPUs 4-7, CPS=256 -> gas 8 -> 32 groups = 1024 completions/step exactly.
#   * rollout: ONE vLLM DP server spanning hosts. Head on 360-2 GPUs 0,1,3 (DP ranks 0-2, HTTP API);
#     headless ranks on 360-1 GPUs 2,4 (DP ranks 3-4) started over ssh with setsid. TP=1, DP=5.
#     /get_world_size = 5, so TRL's NCCL weight-transfer group = 6 (trainer rank 0 + 5 engines).
#   * cross-node env: VLLM_HOST_IP, NCCL/GLOO_SOCKET_IFNAME=bond4, NCCL_IB_HCA=^mlx5_bond_0.
#   * built-in watchdog: head pid, remote headless pid (ssh kill -0, every WD_INTERVAL s) and head
#     /health. On a failure it SIGTERMs this launcher, whose trap tears down trainer + head + remote
#     ranks and writes stop_reason to run.json. Trainer exit is caught by `wait` (same teardown).
#   * algorithm (coordinator v3): STALE=3, eps 3e-3/3e-3, KL 0.05, lr 2e-6, max_steps 600, bf16 KV.
#   * judge: AIQ_HALLUC=0 by default. With the judge on, use the non-blocking scorer (SCORE_CONC=32,
#     rlforge_v3_1/src/rlforge/score_loop.py) and the thread-safe reward REWARD=aiq_think_reward_v3:think_reward;
#     see aiq_rl/docs/reports/NONBLOCKING_SCORING_2026-10-03.md (Review fixes, How to enable).
#   * audits (trainer): per-rank micro-batch counts ("[rlforge][mb_audit]", first RLFORGE_MB_AUDIT_STEPS
#     steps then every 50) for the DDP equal-count check; POSLOG_STEPS=N writes gate-5 per-position
#     log-ratio jsonl (runs/<run>/poslog/poslog_rank*.jsonl) for the first N steps.
#
# Usage (on 360-2, as one session so the whole run is one process group):
#   cd /data/home/guoshaoyang/aiq_rl
#   RUN_NAME=<run> setsid nohup bash run_g32_v3_1.sh full > logs/<run>.driver.log 2>&1 < /dev/null &
#   DRY=1 RUN_NAME=<run> bash run_g32_v3_1.sh full     # preflight + print plan/commands, start nothing
# Stop: kill -TERM <launcher pid>   (its trap stops the trainer, the head and the 360-1 ranks)
set -euo pipefail
SELF="$(readlink -f "${BASH_SOURCE[0]}")"   # v3_1: absolute path of this launcher (md5 in run.json)

MODE="${1:-full}"
source /data/home/guoshaoyang/aiq_rl/env_360_2.sh
# v3_1: no default run name (v3's default is the live v3 run; reusing it would overwrite that run's files).
export RUN_NAME="${RUN_NAME:?set RUN_NAME (run_g32_v3_2.sh has no default run name)}"
SUFFIX="${2:-_${RUN_NAME}}"
ROOT="${ROOT:?set ROOT}"
VENV="${VENV:?set VENV}"
RLFORGE_V3="${RLFORGE_V3:-/data/home/guoshaoyang/rlforge_v3_2}"
LAUNCHER_BASE_MD5=da30395d7eafe696d84cc51d5f916f51   # run_g32_v3_1.sh this file was copied from (v3.2)
export MODEL="${INIT_MODEL:-/data/shared/guoshaoyang/aiq_rl_store/models/sft_rc_ckpt57_20261003}"
export AIQ_BASE_MODEL=/data/home/guoshaoyang/models/Qwen3.5-0.8B-ms RLFORGE_BASE_MODEL=/data/home/guoshaoyang/models/Qwen3.5-0.8B-ms
REWARD="${REWARD:-aiq_think_reward:think_reward}"
PORT="${PORT:-8011}"
DRY="${DRY:-0}"

source "$VENV/bin/activate"
cd "$ROOT"
export PYTHONPATH="$RLFORGE_V3/src:$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs runs evals

export TMPDIR="${TMPDIR:-$ROOT/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/tmp/triton}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-$ROOT/tmp/vllm}"
export VLLM_USE_FLASHINFER_SAMPLER="${FI_SAMPLER:-${VLLM_USE_FLASHINFER_SAMPLER:-0}}"
export VLLM_ALLREDUCE_USE_FLASHINFER="${VLLM_ALLREDUCE_USE_FLASHINFER:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TRL_EXPERIMENTAL_SILENCE="${TRL_EXPERIMENTAL_SILENCE:-1}"
mkdir -p "$TMPDIR"

# ---- layout (coordinator 13:3x: one group per row => groups/step must be a multiple of the trainer
# ranks; 4 ranks x 8 micro-batches = 32 groups exactly; 5 ranks would force 30/35 groups/step) ----
HEAD_HOST_LABEL=360-2
HEAD_IP="${HEAD_IP:-10.234.161.3}"
REMOTE_HOST_LABEL=360-1
REMOTE_IP="${REMOTE_IP:-10.234.161.2}"
REMOTE_SSH="${REMOTE_SSH:-$REMOTE_IP}"      # 360-2 resolves 360-1 only by its bond4 IP
SERVER_GPUS="${SERVER_GPUS:-0,1,3}"          # head-local DP ranks 0..DP_LOCAL-1
REMOTE_GPUS="${REMOTE_GPUS:-2,4}"            # headless DP ranks DP_LOCAL..DP_TOTAL-1
DP_LOCAL=$(awk -F, '{print NF}' <<< "$SERVER_GPUS")
DP_REMOTE=$(awk -F, '{print NF}' <<< "$REMOTE_GPUS")
DP_TOTAL=$((DP_LOCAL + DP_REMOTE))
# One API server per DP rank (vLLM default). A single API server pegs one CPU core at ~32k gen tok/s
# across 5 replicas: HTTP queueing made /pause take 44 s and dropped connections (smoke2 2026-10-03).
API_SERVER_COUNT="${API_SERVER_COUNT:-$DP_TOTAL}"
# uvicorn closes idle keep-alive connections after 5 s; aiohttp then reuses a dead socket and the
# request fails with ServerDisconnectedError (439 retries per step boundary in smoke3). Keep them open.
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE="${VLLM_HTTP_TIMEOUT_KEEP_ALIVE:-3600}"
TP=1
RPC_PORT="${RPC_PORT:-13411}"
TRAINER_GPUS="${TRAINER_GPUS:-4,5,6,7}"
N_TRAINER_GPUS=$(awk -F, '{print NF}' <<< "$TRAINER_GPUS")
NUM_TRAINER="${NUM_TRAINER:-$N_TRAINER_GPUS}"   # v3_1: follows TRAINER_GPUS (v3: fixed default 4 = same value)
EVAL_GPU="${EVAL_GPU:-2}"                    # recorded only; the eval watcher is started separately
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"
REMOTE_DIR="$ROOT/runs/${RUN_NAME}_remote"   # path on 360-1
export VLLM_HOST_IP="$HEAD_IP"               # TRL's weight-transfer master address (trainer) + head
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond4}" GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond4}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-^mlx5_bond_0}"

DTYPE="${DTYPE:-none}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
LIGER="${LIGER:-0}"
GRAD_CKPT="${GRAD_CKPT:-1}"
OPTIM="${OPTIM:-}"
TF32="${TF32:-0}"
DYNAMO="${DYNAMO:-no}"
NO_THINKING="${NO_THINKING:-0}"
GSPO="${GSPO:-1}"
GSPO_NORM="${GSPO_NORM:-seq_mean}"
GSPO_EPS_LOW="${GSPO_EPS_LOW:-3e-3}"
GSPO_EPS_HIGH="${GSPO_EPS_HIGH:-3e-3}"
GSPO_DYNAMIC_LOW_FRAC="${GSPO_DYNAMIC_LOW_FRAC:-0}"
KL_BETA="${KL_BETA:-0.05}"
NGEN="${NGEN:-32}"
# Groups-per-step guard (v3_1). With one group per row (--token-budget 0) a micro-batch is NUM_TRAINER groups,
# so groups/step = (CPS/NGEN) x NUM_TRAINER. GROUPS_PER_STEP=G (optional) pins the step size when the trainer
# layout changes: CPS defaults to G x NGEN / NUM_TRAINER, and the launcher refuses to start unless G is a
# multiple of the trainer ranks and CPS gives exactly G (checked below, before anything starts).
# GPS_GUARD=1 (default) also refuses NUM_TRAINER != number of TRAINER_GPUS. GPS_GUARD=0 skips both checks.
GPS_TARGET="${GROUPS_PER_STEP:-}"
GPS_GUARD="${GPS_GUARD:-1}"
[[ -z "$GPS_TARGET" || "$GPS_TARGET" =~ ^[1-9][0-9]*$ ]] || { echo "[dp] GROUPS_PER_STEP=$GPS_TARGET: expected a positive integer"; exit 1; }
[[ "$NUM_TRAINER" =~ ^[1-9][0-9]*$ ]] || { echo "[dp] NUM_TRAINER=$NUM_TRAINER: expected a positive integer"; exit 1; }
if [ -n "$GPS_TARGET" ]; then
  if [ "$GPS_GUARD" = "1" ] && [ "${TOKEN_BUDGET:-0}" = "0" ] && [ $(( GPS_TARGET % NUM_TRAINER )) -ne 0 ]; then
    echo "[dp] guard: GROUPS_PER_STEP=$GPS_TARGET is not a multiple of the $NUM_TRAINER trainer ranks (TRAINER_GPUS=$TRAINER_GPUS; one group per row under --token-budget 0: groups/step = micro-batches x ranks); nothing started"
    exit 1
  fi
  [ -n "${CPS:-}" ] || CPS=$(( GPS_TARGET * NGEN / NUM_TRAINER ))   # exactness is checked by the guard below
fi
CPS="${CPS:-256}"
STALE="${STALE:-3}"
INFLIGHT="${INFLIGHT:-640}"                  # v3 relaunch value (15:23); the v3 file's default is 4096
PER_SEQ_FWD="${PER_SEQ_FWD:-auto}"
PREFIX_SHARE="${PREFIX_SHARE:-on}"
TOKEN_BUDGET="${TOKEN_BUDGET:-0}"
DP_ROUTE="${DP_ROUTE:-on}"
PREFIX_SHARE_MD5="${PREFIX_SHARE_MD5:-9686c54b5fe771a2ab5b73c307f53440}"   # v3_2 prefix_share.py (v3 gate path unchanged; v3_2 gate: scripts/v32_gate.py)
# ---- v3_2 trainer throughput (defaults = the measured production setting; V32=0 -> v3_1 trainer path) ----
V32="${V32:-1}"
if [ "$V32" = "1" ]; then
  FAST_LOGPROB="${FAST_LOGPROB:-1}"; SUBBATCH_TOKENS="${SUBBATCH_TOKENS:-131072}"; SB_CKPT="${SB_CKPT:-auto}"; BALANCE_ROWS="${BALANCE_ROWS:-on}"
else
  FAST_LOGPROB="${FAST_LOGPROB:-0}"; SUBBATCH_TOKENS="${SUBBATCH_TOKENS:-0}"; SB_CKPT="${SB_CKPT:-all}"; BALANCE_ROWS="${BALANCE_ROWS:-off}"
fi
export RLFORGE_FAST_LOGPROB="$FAST_LOGPROB"
FP8="${FP8:-off}"
FP8_ALIGN="${FP8_ALIGN:-off}"
case "$FP8" in off|native) ;; *) echo "FP8 must be off or native" >&2; exit 2 ;; esac
case "$FP8_ALIGN" in off|vllm) ;; *) echo "FP8_ALIGN must be off or vllm" >&2; exit 2 ;; esac
[ "$FP8_ALIGN" = "off" ] || [ "$FP8" = "native" ] || { echo "FP8_ALIGN=vllm requires FP8=native" >&2; exit 2; }
if [ "$FP8" = "native" ]; then
  [ "$DTYPE" = "none" ] && [ "$FAST_LOGPROB" = "1" ] && [ "$PREFIX_SHARE" = "on" ] \
    || { echo "FP8=native requires FP32 masters, FAST_LOGPROB=1 and PREFIX_SHARE=on" >&2; exit 2; }
  KV_DTYPE=auto
  MIXED_PRECISION=bf16
  export RLFORGE_FP8_FORWARD="${RLFORGE_FP8_FORWARD:-native}"
  case "$RLFORGE_FP8_FORWARD" in native|torch) ;; *) echo "RLFORGE_FP8_FORWARD must be native or torch" >&2; exit 2 ;; esac
  # The plugin changes embedding dtype before load; stale vLLM AOT caches miss it.
  export VLLM_DISABLE_COMPILE_CACHE=1
fi
export RLFORGE_LOGPROB_BWD="${RLFORGE_LOGPROB_BWD:-tf32}"
export RLFORGE_SB_ACT_GB="${RLFORGE_SB_ACT_GB:-80}"
export RLFORGE_SB_MEM_FULL_KB="${RLFORGE_SB_MEM_FULL_KB:-128}" RLFORGE_SB_MEM_CKPT_KB="${RLFORGE_SB_MEM_CKPT_KB:-4.7}" RLFORGE_SB_MEM_BASE_KB="${RLFORGE_SB_MEM_BASE_KB:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
[[ "$SUBBATCH_TOKENS" =~ ^[0-9]+$ ]] || { echo "[dp] SUBBATCH_TOKENS=$SUBBATCH_TOKENS: expected an integer >= 0"; exit 1; }
case "$SB_CKPT" in all|none|auto) ;; *) echo "[dp] SB_CKPT=$SB_CKPT: expected all|none|auto"; exit 1 ;; esac
case "$BALANCE_ROWS" in on|off) ;; *) echo "[dp] BALANCE_ROWS=$BALANCE_ROWS: expected on/off"; exit 1 ;; esac
[ "$BALANCE_ROWS" = "off" ] || [ "$SUBBATCH_TOKENS" != "0" ] || { echo "[dp] BALANCE_ROWS=on needs SUBBATCH_TOKENS>0"; exit 1; }
QUEUE_MAXSIZE="${QUEUE_MAXSIZE-512}"         # v3 relaunch value; set QUEUE_MAXSIZE= (empty) for TRL's default 1024
# Drop audit (rlforge.drop_audit, observe-only, rank 0): ON by default in v3_1 -> --drop-audit on, jsonl at
# $RLFORGE_DROP_AUDIT_PATH (default <out>/drop_audit.jsonl). DROP_AUDIT=off disables it.
DROP_AUDIT="${DROP_AUDIT:-${RLFORGE_DROP_AUDIT:-on}}"
case "$DROP_AUDIT" in on|1|true) DROP_AUDIT=on ;; off|0|false) DROP_AUDIT=off ;;
  *) echo "[dp] DROP_AUDIT=$DROP_AUDIT: expected on/off"; exit 1 ;; esac
export RLFORGE_DROP_AUDIT="$DROP_AUDIT"     # the trainer validates this env var; keep it consistent with the flag
# Non-blocking scoring (rlforge score_loop; all off by default = stock TRL serial score loop).
#   SCORE_CONC=N        score N groups at once (a judged group then delays only itself); 0 = stock
#   JUDGED_STALE=S      cap on judged samples' extra staleness; empty = auto (STALE+2 when SCORE_CONC>1); -1 = off
#   EARLY_HOOKS=1       EXPERIMENTAL: judge calls at rollout end; length-biased under judge saturation -> keep 0
#   SCORE_TASK_MAX_S=T  fail the run if one group's scoring exceeds T s (default max(600, 3 x AIQ_HALLUC_TIMEOUT_S))
SCORE_CONC="${SCORE_CONC:-0}"
JUDGED_STALE="${JUDGED_STALE:-}"
EARLY_HOOKS="${EARLY_HOOKS:-0}"
SCORE_TASK_MAX_S="${SCORE_TASK_MAX_S:-}"
[[ "$SCORE_CONC" =~ ^[0-9]+$ ]] || { echo "[dp] SCORE_CONC=$SCORE_CONC: expected an integer >= 0"; exit 1; }
[ -z "$JUDGED_STALE" ] || [[ "$JUDGED_STALE" =~ ^-?[0-9]+$ ]] || { echo "[dp] JUDGED_STALE=$JUDGED_STALE: expected an integer"; exit 1; }
# trainer audits (rlforge_v3 trainer): per-rank micro-batch counts for the first N steps then every 50
# (DDP equal-count check); gate-5 per-position log-ratio jsonl for the first POSLOG_STEPS steps (0 = off).
export RLFORGE_MB_AUDIT_STEPS="${RLFORGE_MB_AUDIT_STEPS:-20}"
POSLOG_STEPS="${POSLOG_STEPS:-0}"
export RLFORGE_DP_SIZE="${RLFORGE_DP_SIZE:-$DP_TOTAL}"   # dp_route: skip the /get_world_size probe
# Hallucination hook (aiq_think_reward[_v3]): OFF by default; turn it on only with SCORE_CONC>1 + reward v3.
export AIQ_HALLUC="${AIQ_HALLUC:-0}"
export AIQ_HALLUC_K="${AIQ_HALLUC_K:-2}"
export AIQ_HALLUC_LUNA_REWARD="${AIQ_HALLUC_LUNA_REWARD:-0}"
export AIQ_HALLUC_PROVIDER="${AIQ_HALLUC_PROVIDER:-cctq}"
export AIQ_HALLUC_GROUPS_PER_STEP="${AIQ_HALLUC_GROUPS_PER_STEP:-$(( ${CPS} * ${NUM_TRAINER} / ${NGEN} ))}"
if [ "$AIQ_HALLUC" = "1" ] && [ -z "${AIQ_EVAL_KEYS:-}" ] && [ -z "${AIQ_HALLUC_BASE_URL:-}" ]; then
  echo "[dp] hook on: set AIQ_HALLUC_BASE_URL (judge relay) or AIQ_EVAL_KEYS"; exit 1; fi
EPOCHS="${EPOCHS:-40}"
LR="${LR:-2e-6}"
MAX_COMPLETION="${MAX_COMPLETION:-16384}"
MAX_STEPS="${MAX_STEPS:-600}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
KV_DTYPE="${KV_DTYPE:-auto}"
MAX_SEQS="${MAX_SEQS:-2048}"
MAX_BATCHED="${MAX_BATCHED:-32768}"
MAX_CG="${MAX_CG:-2048}"
ASYNC_SCHED="${ASYNC_SCHED:-1}"
WD_INTERVAL="${WD_INTERVAL:-30}"
WD_SSH_MISSES="${WD_SSH_MISSES:-10}"         # consecutive ssh failures (not "process dead") tolerated
WD_HEALTH_FAILS="${WD_HEALTH_FAILS:-6}"      # consecutive head /health failures tolerated
GPU_WAIT_S="${GPU_WAIT_S:-1800}"             # wait for busy GPUs (poll 30 s) before giving up
GPU_BUSY_MIB="${GPU_BUSY_MIB:-1000}"
CKPT_SHA_CHECK="${CKPT_SHA_CHECK:-1}"

if [ $(( CPS % NGEN )) -ne 0 ]; then echo "[dp] CPS=$CPS must be a multiple of NGEN=$NGEN"; exit 1; fi
GAS=$(( CPS / NGEN ))
GROUPS_PER_STEP=$(( GAS * NUM_TRAINER ))
SAMPLES_PER_STEP=$(( CPS * NUM_TRAINER ))
if [ "$GPS_GUARD" = "1" ]; then
  [ "$NUM_TRAINER" = "$N_TRAINER_GPUS" ] \
    || { echo "[dp] guard: NUM_TRAINER=$NUM_TRAINER but TRAINER_GPUS=$TRAINER_GPUS has $N_TRAINER_GPUS GPUs (GPS_GUARD=0 skips)"; exit 1; }
  if [ "${TOKEN_BUDGET:-0}" = "0" ] && [ -n "$GPS_TARGET" ]; then
    [ "$GROUPS_PER_STEP" = "$GPS_TARGET" ] \
      || { echo "[dp] guard: CPS=$CPS x $NUM_TRAINER ranks / NGEN=$NGEN = $GROUPS_PER_STEP groups/step != GROUPS_PER_STEP=$GPS_TARGET (unset CPS to derive it); nothing started"; exit 1; }
  fi
  GPS_GUARD_MSG="on: ranks=$NUM_TRAINER (TRAINER_GPUS=$TRAINER_GPUS) groups/step=$GROUPS_PER_STEP target=${GPS_TARGET:-none}"
else
  GPS_GUARD_MSG="off (GPS_GUARD=0)"
fi
[ "$GROUPS_PER_STEP" = "32" ] || echo "[dp] NOTE: groups/step=$GROUPS_PER_STEP (v3 coordinator value: 32)"

POOL=/data/shared/guoshaoyang/aiq_rl_store/data/rl_pool_b_v0_think
if [ "$MODE" = "smoke" ]; then
  DATA="${DATA:-work_g32think_20261003/data/smoke64.jsonl}"; OUT=runs/g32smoke${SUFFIX}; SAVE="${SAVE:-1000}"
  [ "$MAX_STEPS" = "0" ] && MAX_STEPS=8
else
  DATA="${DATA:-$POOL/train_think.jsonl}";  OUT=runs/${RUN_NAME}; SAVE="${SAVE:-25}"
fi
RUN_ID=$(basename "$OUT")
if [ "$POSLOG_STEPS" != "0" ]; then
  export RLFORGE_POSLOG_STEPS="$POSLOG_STEPS" RLFORGE_POSLOG_DIR="${RLFORGE_POSLOG_DIR:-$ROOT/$OUT/poslog}"
fi

EXTRA_ARGS=""
[ "$DTYPE" = "bfloat16" ] && EXTRA_ARGS="$EXTRA_ARGS --dtype bfloat16"
[ "$FP8" = "native" ] && EXTRA_ARGS="$EXTRA_ARGS --fp8 native"
[ "$FP8_ALIGN" = "vllm" ] && EXTRA_ARGS="$EXTRA_ARGS --fp8-align vllm"
if [ "$GSPO" = "1" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --gspo --gspo-norm $GSPO_NORM --gspo-eps-low $GSPO_EPS_LOW --gspo-eps-high $GSPO_EPS_HIGH --gspo-dynamic-low-frac $GSPO_DYNAMIC_LOW_FRAC --per-seq-forward $PER_SEQ_FWD --kl-beta $KL_BETA"
fi
[ "$MAX_STEPS" != "0" ] && EXTRA_ARGS="$EXTRA_ARGS --max-steps $MAX_STEPS"
[ "$LIGER" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --use-liger"
[ "$GRAD_CKPT" = "0" ] && EXTRA_ARGS="$EXTRA_ARGS --no-grad-ckpt"
[ -n "$OPTIM" ] && EXTRA_ARGS="$EXTRA_ARGS --optim $OPTIM"
[ "$TF32" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --allow-tf32"
[ "$NO_THINKING" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --no-thinking"
# v3 infra flags (rlforge_v3 trainer)
[ "${PREFIX_SHARE:-off}" = "on" ] && EXTRA_ARGS="$EXTRA_ARGS --prefix-share on --token-budget ${TOKEN_BUDGET:-0}"
# v3_2 flags
[ "$SUBBATCH_TOKENS" != "0" ] && EXTRA_ARGS="$EXTRA_ARGS --subbatch-tokens $SUBBATCH_TOKENS --sb-ckpt $SB_CKPT"
[ "$BALANCE_ROWS" = "on" ] && EXTRA_ARGS="$EXTRA_ARGS --balance-rows on"
[ "$DP_ROUTE" = "on" ] && EXTRA_ARGS="$EXTRA_ARGS --dp-route on"
[ -n "$QUEUE_MAXSIZE" ] && EXTRA_ARGS="$EXTRA_ARGS --queue-maxsize $QUEUE_MAXSIZE"
[ "$SCORE_CONC" != "0" ] && EXTRA_ARGS="$EXTRA_ARGS --score-concurrency $SCORE_CONC"
[ -n "$JUDGED_STALE" ] && EXTRA_ARGS="$EXTRA_ARGS --judged-max-staleness $JUDGED_STALE"
[ "$EARLY_HOOKS" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --reward-early-hooks"
[ -n "$SCORE_TASK_MAX_S" ] && EXTRA_ARGS="$EXTRA_ARGS --score-task-max-s $SCORE_TASK_MAX_S"
EXTRA_ARGS="$EXTRA_ARGS --drop-audit $DROP_AUDIT"   # v3_1: explicit (v3 relied on RLFORGE_DROP_AUDIT in the env)
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-5400}"
EXTRA_ARGS="$EXTRA_ARGS --request-timeout $REQUEST_TIMEOUT"

export RLFORGE_TASK_LOG="${RLFORGE_TASK_LOG:-$OUT/task_split.jsonl}"
export AIQ_SAMPLE_LOG="${AIQ_SAMPLE_LOG:-$OUT/rollout_samples.jsonl}"

VLLM_EXTRA="${VLLM_EXTRA:-}"
[ "$FP8" = "native" ] && VLLM_EXTRA="$VLLM_EXTRA --quantization poor_rl_fp8"
FP8_VLLM_ARGS=()
FP8_REMOTE_VLLM_ARGS=""
if [ "$FP8_ALIGN" = "vllm" ]; then
  FP8_VLLM_ARGS=(--additional-config '{"gdn_prefill_backend":"triton"}')
  FP8_REMOTE_VLLM_ARGS="--additional-config '{\"gdn_prefill_backend\":\"triton\"}'"
fi
[ -n "${MAX_SEQS:-}" ] && VLLM_EXTRA="$VLLM_EXTRA --max-num-seqs $MAX_SEQS"
[ -n "${MAX_BATCHED:-}" ] && VLLM_EXTRA="$VLLM_EXTRA --max-num-batched-tokens $MAX_BATCHED"
[ -n "${MAX_CG:-}" ] && VLLM_EXTRA="$VLLM_EXTRA --max-cudagraph-capture-size $MAX_CG"
[ "${ASYNC_SCHED:-0}" = "1" ] && VLLM_EXTRA="$VLLM_EXTRA --async-scheduling"
[ -n "${KV_DTYPE:-}" ] && [ "$KV_DTYPE" != "auto" ] && VLLM_EXTRA="$VLLM_EXTRA --kv-cache-dtype $KV_DTYPE"
VLLM_COMMON="--served-model-name $MODEL --dtype bfloat16 --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization $GPU_MEM_UTIL --tensor-parallel-size $TP --data-parallel-size $DP_TOTAL --data-parallel-address $HEAD_IP --data-parallel-rpc-port $RPC_PORT$VLLM_EXTRA"

echo "[dp] v3_2 (code $RLFORGE_V3; copy of run_g32_v3_1.sh ${LAUNCHER_BASE_MD5:0:8}) drop_audit=$DROP_AUDIT groups_guard=$GPS_GUARD_MSG"
echo "[dp] v3_2 trainer: V32=$V32 fast_logprob=$FAST_LOGPROB (bwd $RLFORGE_LOGPROB_BWD) subbatch_tokens=$SUBBATCH_TOKENS sb_ckpt=$SB_CKPT (act_gb=$RLFORGE_SB_ACT_GB) balance_rows=$BALANCE_ROWS alloc=$PYTORCH_CUDA_ALLOC_CONF"
echo "[dp] v3 mode=$MODE run=$RUN_ID data=$DATA out=$OUT gspo=$GSPO norm=$GSPO_NORM eps=$GSPO_EPS_LOW/$GSPO_EPS_HIGH kl=$KL_BETA cps=$CPS ngen=$NGEN stale=$STALE inflight=$INFLIGHT lr=$LR"
echo "[dp] rollout: DP=$DP_TOTAL TP=$TP head $HEAD_HOST_LABEL GPUs=$SERVER_GPUS (ranks 0-$((DP_LOCAL-1))) + headless $REMOTE_HOST_LABEL($REMOTE_IP) GPUs=$REMOTE_GPUS (ranks $DP_LOCAL-$((DP_TOTAL-1))) port=$PORT rpc=$RPC_PORT"
echo "[dp] trainer: GPUs=$TRAINER_GPUS ranks=$NUM_TRAINER gas=$GAS groups/step=$GROUPS_PER_STEP samples/step=$SAMPLES_PER_STEP prefix_share=$PREFIX_SHARE token_budget=$TOKEN_BUDGET dp_route=$DP_ROUTE queue_maxsize=${QUEUE_MAXSIZE:-trl-default} mb_audit=$RLFORGE_MB_AUDIT_STEPS poslog_steps=$POSLOG_STEPS"
echo "[dp] epochs=$EPOCHS max_steps=$MAX_STEPS save_steps=$SAVE request_timeout=$REQUEST_TIMEOUT max_completion=$MAX_COMPLETION task_log=$RLFORGE_TASK_LOG"
echo "[dp] halluc: hook=$AIQ_HALLUC k=$AIQ_HALLUC_K k_file=${AIQ_HALLUC_K_FILE:-} luna_reward=$AIQ_HALLUC_LUNA_REWARD base_url=${AIQ_HALLUC_BASE_URL:-}"
echo "[dp] scoring: score_conc=$SCORE_CONC judged_stale=${JUDGED_STALE:-auto} early_hooks=$EARLY_HOOKS score_task_max_s=${SCORE_TASK_MAX_S:-default} reward=$REWARD judge_frac=${AIQ_HALLUC_JUDGE_FRAC:-1} rpm=${AIQ_HALLUC_RPM:-0} burst=${AIQ_HALLUC_BURST:-rpm/4} threads=${AIQ_HALLUC_THREADS:-64} tail_s=${AIQ_HALLUC_TAIL_S:-0}"
echo "[dp] perf: liger=$LIGER grad_ckpt=$GRAD_CKPT optim=${OPTIM:-default} tf32=$TF32 dynamo=$DYNAMO mixed_precision=$MIXED_PRECISION fi_sampler=$VLLM_USE_FLASHINFER_SAMPLER kv_dtype=$KV_DTYPE"
echo "[dp] vllm common:$VLLM_COMMON"
echo "[dp] net: VLLM_HOST_IP=$VLLM_HOST_IP NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME NCCL_IB_HCA=$NCCL_IB_HCA"

# ---------------------------------------------------------------- preflight (read-only)
fail() { echo "[preflight] FAIL: $*"; exit 1; }
RLFORGE_PKG=$("$VENV/bin/python" -c "import rlforge, os; print(os.path.dirname(rlforge.__file__))")
[ "$RLFORGE_PKG" = "$RLFORGE_V3/src/rlforge" ] || fail "rlforge resolves to $RLFORGE_PKG, expected $RLFORGE_V3/src/rlforge"
for f in prefix_share.py dp_route.py trainer.py; do [ -f "$RLFORGE_PKG/$f" ] || fail "missing $RLFORGE_PKG/$f"; done
grep -q -- '"--drop-audit"' "$RLFORGE_PKG/trainer.py" || fail "$RLFORGE_PKG/trainer.py has no --drop-audit flag (v3_1 always passes it)"
grep -q -- '"--subbatch-tokens"' "$RLFORGE_PKG/trainer.py" || fail "$RLFORGE_PKG/trainer.py has no --subbatch-tokens flag (not the v3_2 trainer)"
[ "$FAST_LOGPROB" = "0" ] || [ -f "$RLFORGE_PKG/fast_logprob.py" ] || fail "FAST_LOGPROB=1: $RLFORGE_PKG/fast_logprob.py missing"
[ "$DROP_AUDIT" = "off" ] || [ -f "$RLFORGE_PKG/drop_audit.py" ] || fail "DROP_AUDIT=on: $RLFORGE_PKG/drop_audit.py missing"
md5_8() { [ -f "$1" ] && md5sum "$1" | cut -c1-8 || echo absent; }
echo "[preflight] code md5: trainer $(md5_8 "$RLFORGE_PKG/trainer.py") drop_audit $(md5_8 "$RLFORGE_PKG/drop_audit.py") score_loop $(md5_8 "$RLFORGE_PKG/score_loop.py") prefix_share $(md5_8 "$RLFORGE_PKG/prefix_share.py") dp_route $(md5_8 "$RLFORGE_PKG/dp_route.py")"
if [ "$PREFIX_SHARE" = "on" ]; then
  # The gate passed for prefix_share ede53a03 with FA3 attention, split ALIGN=64, SDPA fallback off.
  PS_MD5=$(md5sum "$RLFORGE_PKG/prefix_share.py" | cut -c1-32)
  [ "$PS_MD5" = "$PREFIX_SHARE_MD5" ] || fail "prefix_share.py md5 $PS_MD5 != gate-passed $PREFIX_SHARE_MD5"
  [ -z "${RLFORGE_PREFIX_ALIGN:-}" ] || [ "$RLFORGE_PREFIX_ALIGN" = "64" ] || fail "RLFORGE_PREFIX_ALIGN=$RLFORGE_PREFIX_ALIGN (gate: unset/64)"
  [ -z "${RLFORGE_PREFIX_SDPA:-}" ] || [ "$RLFORGE_PREFIX_SDPA" = "0" ] || fail "RLFORGE_PREFIX_SDPA=$RLFORGE_PREFIX_SDPA (gate: unset)"
  case "${LOCAL_KERNELS:-}" in *kernels-community/flash-attn3=*) ;; *) fail "LOCAL_KERNELS lacks the local flash-attn3 repo (env_360_2.sh)";; esac
  [ -d "${LOCAL_KERNELS#*=}" ] || fail "flash-attn3 local kernel dir ${LOCAL_KERNELS#*=} missing"
  echo "[preflight] prefix_share $PS_MD5 (gate-passed), FA3 from ${LOCAL_KERNELS#*=}, align=${RLFORGE_PREFIX_ALIGN:-64}"
fi
REWARD_MODULE="${REWARD%%:*}"
REWARD_FILE="$ROOT/scripts/$REWARD_MODULE.py"
[ -f "$REWARD_FILE" ] || fail "reward module $REWARD_FILE missing"
grep -q "return 0.5 / inv" "$REWARD_FILE" || fail "$REWARD_FILE lacks 'return 0.5 / inv' (inverse ranking reward)"
if [ "$SCORE_CONC" != "0" ] || [ -n "$JUDGED_STALE" ] || [ "$EARLY_HOOKS" = "1" ]; then
  [ -f "$RLFORGE_V3/src/rlforge/score_loop.py" ] || fail "SCORE_CONC=$SCORE_CONC: $RLFORGE_V3/src/rlforge/score_loop.py missing"
  grep -q -- "--score-concurrency" "$RLFORGE_V3/src/rlforge/trainer.py" || fail "rlforge_v3 trainer.py lacks --score-concurrency (apply trainer.py.score_loop.diff)"
  echo "[preflight] score_loop $(md5sum < "$RLFORGE_V3/src/rlforge/score_loop.py" | cut -c1-8) concurrency=$SCORE_CONC"
fi
if [ "$AIQ_HALLUC" = "1" ]; then
  if [ "$SCORE_CONC" -le 1 ]; then
    echo "[preflight] WARNING AIQ_HALLUC=1 with SCORE_CONC=$SCORE_CONC: each judged group stalls the serial scorer (and then generation) for up to AIQ_HALLUC_TIMEOUT_S"
  elif ! grep -q "_close_group" "$REWARD_FILE"; then
    fail "SCORE_CONC=$SCORE_CONC with AIQ_HALLUC=1 needs the thread-safe reward (REWARD=aiq_think_reward_v3:think_reward); $REWARD_FILE is the stock module"
  fi
  # The RPM bucket refuses (never waits): its burst must cover one judged group's calls (~NGEN x JUDGE_FRAC),
  # else the burst, not JUDGE_FRAC, caps every judged group (NONBLOCKING_SCORING_2026-10-03.md section 5).
  if [ "${AIQ_HALLUC_RPM:-0}" != "0" ]; then
    _calls=$(awk -v n="$NGEN" -v f="${AIQ_HALLUC_JUDGE_FRAC:-1}" 'BEGIN{printf "%d", n * f + 0.999}')
    _burst="${AIQ_HALLUC_BURST:-$(awk -v r="$AIQ_HALLUC_RPM" 'BEGIN{printf "%d", r / 4}')}"
    [ "$_burst" -ge "$_calls" ] 2>/dev/null \
      || echo "[preflight] WARNING AIQ_HALLUC_BURST=$_burst < ~$_calls calls per judged group (NGEN=$NGEN x JUDGE_FRAC=${AIQ_HALLUC_JUDGE_FRAC:-1}): the bucket caps each judged group at $_burst calls"
  fi
fi
echo "[preflight] reward $REWARD -> $REWARD_FILE md5 $(md5_8 "$REWARD_FILE")"
[ -f "$DATA" ] || fail "data $DATA missing"
[ -f "$MODEL/model.safetensors" ] || fail "model $MODEL missing"
for p in "$PORT" "$RPC_PORT"; do
  if ss -ltnH "sport = :$p" | grep -q .; then
    # v3_1: DRY only reports it (like busy GPUs), so the plan can be printed while another run holds the ports
    if [ "$DRY" = "1" ]; then echo "[preflight] (dry) port $p already listening on $(hostname -s): a real launch would FAIL here"
    else fail "port $p already listening on $(hostname -s)"; fi
  fi
done
if [ -e "$OUT/run.json" ]; then
  if [ "$DRY" = "1" ]; then echo "[preflight] (dry) $OUT/run.json exists: a real launch would FAIL here (choose a new RUN_NAME)"
  elif [ "${ALLOW_EXISTING_OUT:-0}" != "1" ]; then fail "$OUT/run.json exists (another run's directory; choose a new RUN_NAME or ALLOW_EXISTING_OUT=1)"; fi
fi
ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_SSH" \
  "test -x $VENV/bin/vllm && test -f $MODEL/model.safetensors && ip -4 -o addr show bond4 | grep -q 'inet $REMOTE_IP/'" \
  || fail "remote $REMOTE_SSH: venv/model/bond4 check failed"
FP8_SOURCE_LOCAL=""
if [ "$FP8" = "native" ]; then
  FP8_SOURCE_CHECK='import hashlib, importlib.metadata as metadata, json, os, pathlib, rlforge
p = pathlib.Path(rlforge.__file__).parent
names = ("fp8.py", "fp8_serving.py", "fp8_alignment.py", "fast_logprob.py", "prefix_share.py", "serving_logprobs.py", "serving_decode.py", "serving_rope.py")
missing = [n for n in names if not (p / n).is_file()]
if missing: raise RuntimeError("FP8 source files missing: " + str(missing))
plugins = [ep.value for ep in metadata.entry_points(group="vllm.general_plugins") if ep.name == "poor_rl_fp8"]
if plugins != ["rlforge.fp8_serving:register"]: raise RuntimeError("Install the selected poor-rl checkout in this venv with pip install --no-deps -e <checkout> (poor_rl_fp8 entry point missing or conflicting)")
allowed = os.environ.get("VLLM_PLUGINS")
if allowed is not None and "poor_rl_fp8" not in allowed.split(","): raise RuntimeError("VLLM_PLUGINS disables poor_rl_fp8")
print(json.dumps({"package_root": str(p), "sha256": {n: hashlib.sha256((p / n).read_bytes()).hexdigest() if (p / n).is_file() else None for n in names}, "fp8_plugin": plugins, "versions": {n: metadata.version(n) for n in ("torch", "vllm", "transformers", "triton")}}, sort_keys=True))'
  FP8_SOURCE_LOCAL=$("$VENV/bin/python" -c "$FP8_SOURCE_CHECK") || fail "local FP8 source fingerprint failed"
  _fp8_plugin_env="unset VLLM_PLUGINS;"
  if [ "${VLLM_PLUGINS+x}" = "x" ]; then printf -v _fp8_plugin_env 'export VLLM_PLUGINS=%q;' "$VLLM_PLUGINS"; fi
  printf -v _fp8_source_cmd '%s PYTHONPATH=%q %q -c %q' "$_fp8_plugin_env" "$PYTHONPATH" "$VENV/bin/python" "$FP8_SOURCE_CHECK"
  FP8_SOURCE_REMOTE=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_SSH" "$_fp8_source_cmd") \
    || fail "remote FP8 source fingerprint failed"
  [ "$FP8_SOURCE_LOCAL" = "$FP8_SOURCE_REMOTE" ] || fail "FP8 source differs between trainer/local and remote rollout; synchronize the selected checkout"
  export FP8_SOURCE_LOCAL
  echo "[preflight] FP8 source identical on trainer/local and remote rollout"
fi
if ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_SSH" \
  "test -s $REMOTE_DIR/vllm_headless.pid && kill -0 \$(cat $REMOTE_DIR/vllm_headless.pid) 2>/dev/null"; then
  fail "headless ranks of a previous $RUN_NAME launch are still alive on $REMOTE_HOST_LABEL ($REMOTE_DIR/vllm_headless.pid)"
fi
if [ "$CKPT_SHA_CHECK" = "1" ]; then
  echo "[preflight] sha256 of model.safetensors on both hosts..."
  SHA_REMOTE_F=$(mktemp); ssh -o BatchMode=yes "$REMOTE_SSH" "sha256sum $MODEL/model.safetensors" > "$SHA_REMOTE_F" &
  SHA_PID=$!
  SHA_LOCAL=$(sha256sum "$MODEL/model.safetensors" | awk '{print $1}')
  wait "$SHA_PID" || fail "remote sha256sum failed"
  SHA_REMOTE=$(awk '{print $1}' "$SHA_REMOTE_F"); rm -f "$SHA_REMOTE_F"
  [ "$SHA_LOCAL" = "$SHA_REMOTE" ] || fail "checkpoint differs between hosts ($SHA_LOCAL vs $SHA_REMOTE)"
  echo "[preflight] checkpoint identical on both hosts: $SHA_LOCAL"
else
  SHA_LOCAL=unchecked; SHA_REMOTE=unchecked
fi
gpu_busy() {  # $1 = local|remote, $2 = comma list -> prints busy GPUs ("idx:MiB ")
  local q="nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits" out
  if [ "$1" = local ]; then out=$($q); else out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_SSH" "$q") || { echo "ssh-failed"; return; }; fi
  echo "$out" | awk -F', *' -v list=",$2," -v thr="$GPU_BUSY_MIB" 'index(list, "," $1 ",") && $2+0 > thr {printf "%s:%sMiB ", $1, $2}'
}
waited=0
while true; do
  BUSY_L=$(gpu_busy local "$SERVER_GPUS,$TRAINER_GPUS"); BUSY_R=$(gpu_busy remote "$REMOTE_GPUS")
  [ -z "$BUSY_L$BUSY_R" ] && { echo "[preflight] GPUs free: $HEAD_HOST_LABEL $SERVER_GPUS,$TRAINER_GPUS + $REMOTE_HOST_LABEL $REMOTE_GPUS"; break; }
  echo "[preflight] $(date +%T) busy: $HEAD_HOST_LABEL[${BUSY_L:-none}] $REMOTE_HOST_LABEL[${BUSY_R:-none}] (waited ${waited}s / ${GPU_WAIT_S}s)"
  [ "$DRY" = "1" ] && break
  [ "$waited" -ge "$GPU_WAIT_S" ] && fail "GPUs still busy after ${GPU_WAIT_S}s; nothing started"
  sleep 30; waited=$((waited + 30))
done

ACC_LAUNCH=(accelerate launch --num_processes "$NUM_TRAINER" --mixed_precision "$MIXED_PRECISION" --dynamo_backend "$DYNAMO")
[ -n "${ACCELERATE_CONFIG:-}" ] && ACC_LAUNCH=(accelerate launch --config_file "$ACCELERATE_CONFIG" --num_processes "$NUM_TRAINER")
TRAINER_ARGS=(-m rlforge.trainer --model "$MODEL" --train "$DATA" --out "$OUT" --epochs "$EPOCHS" --lr "$LR"
  --completions-per-step "$CPS" --max-completion "$MAX_COMPLETION" --num-generations "$NGEN" --save-steps "$SAVE"
  --max-staleness "$STALE" --max-inflight-tasks "$INFLIGHT" --report-to "${REPORT_TO:-none}" --reward "$REWARD"
  --server-url "http://localhost:$PORT")
# Remote headless ranks: one setsid session on 360-1 (pid == pgid), so stop = one group kill.
# TMPDIR must stay short: vLLM binds zmq ipc sockets at \$TMPDIR/<uuid4> and the unix-socket path
# limit is 107 chars (smoke 2026-10-03 14:04: \$REMOTE_DIR/tmp/<uuid> was 112+ chars -> ZMQError).
FP8_SHARED_ENV_NAMES=(RLFORGE_FP8_FORWARD VLLM_DISABLE_COMPILE_CACHE VLLM_PLUGINS
  RLFORGE_FP8_ROPE_BF16
  RLFORGE_FP8_ALIGN_COMPILE RLFORGE_FP8_ALIGN_POINTWISE RLFORGE_FP8_ALIGN_BACKWARD
  TRITON_F32_DEFAULT FLA_DISABLE_BACKEND_DISPATCH FLA_CACHE_MODE FLA_USE_FAST_OPS
  VLLM_BATCH_INVARIANT VLLM_USE_V2_MODEL_RUNNER VLLM_GDN_DECODE_KERNEL
  VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE RLFORGE_SERVING_DECODE)
FP8_REMOTE_ENV=""
if [ "$FP8" = "native" ]; then
  printf -v FP8_REMOTE_ENV 'export PYTHONPATH=%q\n' "$PYTHONPATH"
  for _fp8_name in "${FP8_SHARED_ENV_NAMES[@]}"; do
    if [ "${!_fp8_name+x}" = "x" ]; then
      export "$_fp8_name"
      printf -v _fp8_export 'export %s=%q\n' "$_fp8_name" "${!_fp8_name}"
    else
      printf -v _fp8_export 'unset %s\n' "$_fp8_name"
    fi
    FP8_REMOTE_ENV+="$_fp8_export"
  done
fi
REMOTE_SCRIPT="set -e
$FP8_REMOTE_ENV
mkdir -p $REMOTE_DIR $ROOT/tmp
rm -f $REMOTE_DIR/vllm_headless.pid
cd $ROOT
export VLLM_HOST_IP=$REMOTE_IP NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME NCCL_IB_HCA='$NCCL_IB_HCA'
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0 VLLM_SERVER_DEV_MODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=$ROOT/tmp TRITON_CACHE_DIR=$ROOT/tmp/triton VLLM_CACHE_DIR=$ROOT/tmp/vllm CUDA_VISIBLE_DEVICES=$REMOTE_GPUS
setsid nohup $VENV/bin/vllm serve $MODEL --headless --data-parallel-size-local $DP_REMOTE --data-parallel-start-rank $DP_LOCAL $VLLM_COMMON $FP8_REMOTE_VLLM_ARGS --weight-transfer-config '{\"backend\":\"nccl\"}' > $REMOTE_DIR/vllm_headless.log 2>&1 < /dev/null &
P=\$!
echo \$P > $REMOTE_DIR/vllm_headless.pid
sleep 2
echo \"\$P \$(ps -o pgid= -p \$P | tr -d ' ')\""

if [ "$DRY" = "1" ]; then
  FP8_DRY_VLLM_ARGS=""
  [ "${#FP8_VLLM_ARGS[@]}" = "0" ] || printf -v FP8_DRY_VLLM_ARGS '%q ' "${FP8_VLLM_ARGS[@]}"
  echo "[dry] head ($HEAD_HOST_LABEL): CUDA_VISIBLE_DEVICES=$SERVER_GPUS VLLM_SERVER_DEV_MODE=1 vllm serve $MODEL --port $PORT --api-server-count $API_SERVER_COUNT --data-parallel-size-local $DP_LOCAL $VLLM_COMMON $FP8_DRY_VLLM_ARGS --weight-transfer-config '{\"backend\":\"nccl\"}' > logs/vllm_dp${SUFFIX}.log"
  echo "[dry] remote ($REMOTE_SSH) script:"; echo "$REMOTE_SCRIPT" | sed 's/^/[dry]   /'
  echo "[dry] trainer: CUDA_VISIBLE_DEVICES=$TRAINER_GPUS ${ACC_LAUNCH[*]} ${TRAINER_ARGS[*]} $EXTRA_ARGS"
  echo "[dry] code: $RLFORGE_PKG trainer md5 $(md5sum "$RLFORGE_PKG/trainer.py" | cut -c1-32)"
  echo "[dry] v3_2: launcher $SELF md5 $(md5sum "$SELF" | cut -c1-32) (copied from run_g32_v3_1.sh $LAUNCHER_BASE_MD5); out $OUT; drop_audit=$DROP_AUDIT; scorer: score_conc=$SCORE_CONC judged_stale=${JUDGED_STALE:-auto} early_hooks=$EARLY_HOOKS; reward $REWARD"
  echo "[dry] nothing started"; exit 0
fi

mkdir -p "$OUT"
# ---- run manifest + code snapshot: written BEFORE anything starts ----
SNAP="$OUT/code_snapshot"; mkdir -p "$SNAP"
cp "$RLFORGE_PKG"/*.py "$SNAP/" 2>/dev/null || true
cp "$RLFORGE_PKG"/rewards/*.py "$SNAP/" 2>/dev/null || true
cp "${BASH_SOURCE[0]}" "$SNAP/$(basename "${BASH_SOURCE[0]}")" 2>/dev/null || true
cp "$REWARD_FILE" "$SNAP/$REWARD_MODULE.py"
"$VENV/bin/python" - <<PY
import hashlib, json, os, platform, re, subprocess, time
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
                   "trainer_gpus": "$TRAINER_GPUS", "num_trainer": int("$NUM_TRAINER"),
                   "rollout": {"dp_size": int("$DP_TOTAL"), "tp": int("$TP"), "api_server_count": int("$API_SERVER_COUNT"),
                               "head": {"host": "$HEAD_HOST_LABEL", "ip": "$HEAD_IP", "gpus": "$SERVER_GPUS",
                                        "dp_ranks": list(range(0, int("$DP_LOCAL"))), "port": int("$PORT"),
                                        "rpc_port": int("$RPC_PORT")},
                               "headless": {"host": "$REMOTE_HOST_LABEL", "ip": "$REMOTE_IP", "gpus": "$REMOTE_GPUS",
                                            "dp_ranks": list(range(int("$DP_LOCAL"), int("$DP_TOTAL"))),
                                            "log_dir": "$REMOTE_DIR", "pid": None, "pgid": None}},
                   "trainer": {"host": "$HEAD_HOST_LABEL", "gpus": "$TRAINER_GPUS", "ranks": int("$NUM_TRAINER")},
                   "eval_watcher": {"host": "$HEAD_HOST_LABEL", "gpu": "$EVAL_GPU", "started_by": "operator, after launch"},
                   "rationale": "one group per row (prefix-share, token_budget 0) => micro-batch = N groups; "
                                "4 trainer ranks x 8 micro-batches = 32 groups/step exactly (5 ranks would force 30/35); "
                                "trainer-bound (~4.2 samples/s/rank) so the 5th GPU goes to rollout (DP=5)"},
    "code_root": "$RLFORGE_V3",
    "fp8_source_preflight": json.loads(os.environ["FP8_SOURCE_LOCAL"]) if "$FP8" == "native" else None,
    "code_md5": {os.path.basename(f): md5(f) for f in
                 sorted(__import__("glob").glob(os.path.join("$RLFORGE_PKG", "*.py")))},
    "engine": {"torch": pkg("torch"), "vllm": pkg("vllm"), "trl": pkg("trl"),
               "transformers": pkg("transformers")},
    "model": {"path": "$MODEL", "md5_config": md5(os.path.join("$MODEL", "config.json")),
              "sha256_safetensors": {"$HEAD_HOST_LABEL": "$SHA_LOCAL", "$REMOTE_HOST_LABEL": "$SHA_REMOTE"}},
    "data": {"path": "$DATA", "md5": md5("$DATA"), "rows": lines("$DATA"),
             "tasks": dict(tasks)},
    "seed": 0,
    "hyperparams": {
        "gspo": "$GSPO" == "1", "gspo_norm": "$GSPO_NORM",
        "gspo_eps_low": "$GSPO_EPS_LOW", "gspo_eps_high": "$GSPO_EPS_HIGH",
        "gspo_dynamic_low_frac": "$GSPO_DYNAMIC_LOW_FRAC", "per_seq_forward": "$PER_SEQ_FWD", "kl_beta": "$KL_BETA",
        "kl_reference": os.environ.get("RLFORGE_REF_MODEL") or "frozen bf16 copy of init",
        "halluc_hook": "$AIQ_HALLUC", "halluc_k": "$AIQ_HALLUC_K", "halluc_luna_reward": "$AIQ_HALLUC_LUNA_REWARD",
        "halluc_k_file": "${AIQ_HALLUC_K_FILE:-}", "halluc_rules_reward": "${AIQ_HALLUC_RULES_REWARD:-0}",
        "halluc_budget_yuan": "${AIQ_HALLUC_BUDGET_YUAN:-250}", "halluc_base_url": "${AIQ_HALLUC_BASE_URL:-}",
        "cps": int("$CPS"), "ngen": int("$NGEN"), "stale": int("$STALE"),
        "gas": int("$GAS"), "groups_per_step": int("$GROUPS_PER_STEP"), "samples_per_step_expected": int("$SAMPLES_PER_STEP"),
        "inflight": int("$INFLIGHT"), "lr": "$LR", "epochs": "$EPOCHS",
        "max_steps": "$MAX_STEPS", "max_completion": int("$MAX_COMPLETION"),
        "request_timeout": "$REQUEST_TIMEOUT", "save_steps": "$SAVE",
        "save_total_limit": 5, "max_seqs": "${MAX_SEQS:-stock}",
        "max_batched": "${MAX_BATCHED:-stock}", "max_cg": "${MAX_CG:-stock}", "async_scheduling": "$ASYNC_SCHED" == "1",
        "max_model_len": int("$MAX_MODEL_LEN"),
        "no_thinking": "$NO_THINKING" == "1", "mixed_precision": "$MIXED_PRECISION",
        "trainer_dtype": "$DTYPE", "rollout_dtype": "bfloat16",
        "kv_cache_dtype": "$KV_DTYPE", "rollout_quantization": "poor_rl_fp8" if "$FP8" == "native" else "none",
        "trainer_fp8": "$FP8",
        "trainer_fp8_alignment": "$FP8_ALIGN",
        "fp8_forward": os.environ.get("RLFORGE_FP8_FORWARD", "native"),
        "fp8_shared_env": {k: os.environ.get(k) for k in "${FP8_SHARED_ENV_NAMES[*]}".split()} if "$FP8" == "native" else {},
        "fp8_graphs": os.environ.get("RLFORGE_FP8_GRAPHS", "0"),
        "fp8_graph_max_mb": os.environ.get("RLFORGE_FP8_GRAPH_MAX_MB", "1024"),
        "fp8_head_graph": os.environ.get("RLFORGE_FP8_HEAD_GRAPH", "0"),
        "fp8_fuse_mlp": os.environ.get("RLFORGE_FP8_FUSE_MLP", "1"),
        "fp8_align_compile": os.environ.get("RLFORGE_FP8_ALIGN_COMPILE", "0"),
        "gpu_memory_utilization": "$GPU_MEM_UTIL",
    },
    "infra": {
        "prefix_share": "$PREFIX_SHARE", "token_budget": int("$TOKEN_BUDGET"),
        "planner": ("BalancedGroupRowBatcher (v3_2: groups split over ranks for load balance)" if "$BALANCE_ROWS" == "on" else "GroupRowBatcher (one whole group per row)") if "$PREFIX_SHARE" == "on" and int("$TOKEN_BUDGET") == 0 else "see token_budget",
        "prefix_max_row_tokens": os.environ.get("RLFORGE_PREFIX_MAX_ROW_TOKENS", "196608 (default)"),
        "prefix_attn_rows": "8 (prefix_share.ATTN_ROWS module default; no runtime knob in this trainer)",
        "prefix_share_md5": md5(os.path.join("$RLFORGE_PKG", "prefix_share.py")),
        "prefix_share_gate": "v3 gate PASSED 2026-10-03 with ede53a03: bit-exact vs production FA3 per-seq (policy + KL ref, 8 ckpt57 groups x 32), 6 criteria + 2-rank DDP, 2.10x/group",
        "prefix_align": os.environ.get("RLFORGE_PREFIX_ALIGN", "64 (default)"),
        "prefix_sdpa": os.environ.get("RLFORGE_PREFIX_SDPA", "unset (FA path)"),
        "trainer_attention": "kernels-community/flash-attn3 (TRL default; LOCAL_KERNELS=" + os.environ.get("LOCAL_KERNELS", "") + ")",
        "dp_route": "$DP_ROUTE", "dp_route_size": os.environ.get("RLFORGE_DP_SIZE"),
        "queue_maxsize": "${QUEUE_MAXSIZE:-1024 (TRL default)}",
        "mb_audit_steps": os.environ.get("RLFORGE_MB_AUDIT_STEPS"),
        "poslog": {"steps": os.environ.get("RLFORGE_POSLOG_STEPS", "0"), "dir": os.environ.get("RLFORGE_POSLOG_DIR")},
        "vllm_common_flags": "$VLLM_COMMON".strip(),
        "cross_node_env": {"VLLM_HOST_IP": {"$HEAD_HOST_LABEL": "$HEAD_IP", "$REMOTE_HOST_LABEL": "$REMOTE_IP"},
                           "NCCL_SOCKET_IFNAME": "$NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME": "$GLOO_SOCKET_IFNAME",
                           "NCCL_IB_HCA": "$NCCL_IB_HCA"},
        "watchdog": {"interval_s": int("$WD_INTERVAL"), "ssh_misses": int("$WD_SSH_MISSES"),
                     "health_fails": int("$WD_HEALTH_FAILS"),
                     "checks": "head pid alive; remote headless pid alive (ssh kill -0); head /health"},
        "nonblocking_scorer": {"score_concurrency": int("$SCORE_CONC"), "judged_max_staleness": "${JUDGED_STALE:-auto}",
                               "reward_early_hooks": "$EARLY_HOOKS" == "1", "score_task_max_s": "${SCORE_TASK_MAX_S:-default}",
                               "score_loop_md5": md5("$RLFORGE_V3/src/rlforge/score_loop.py"),
                               "judge_env": {k: re.sub(r"://[^/@\\s]+@", "://***@", os.environ[k])
                                             for k in sorted(os.environ) if k.startswith("AIQ_HALLUC")},
                               "trainer_flags": [a for a in "$EXTRA_ARGS".split() if a.startswith("--score") or a.startswith("--judged") or a.startswith("--reward-early")],
                               "doc": "aiq_rl/docs/reports/NONBLOCKING_SCORING_2026-10-03.md"},
        "drop_audit": {"enabled": "$DROP_AUDIT", "flag": "--drop-audit $DROP_AUDIT",
                       "path": os.environ.get("RLFORGE_DROP_AUDIT_PATH") or os.path.join("$OUT", "drop_audit.jsonl"),
                       "drop_audit_md5": md5(os.path.join("$RLFORGE_PKG", "drop_audit.py")),
                       "report_script": "$RLFORGE_V3/scripts/drop_audit_report.py",
                       "report_script_md5": md5("$RLFORGE_V3/scripts/drop_audit_report.py"),
                       "note": "observe-only; it predicts TRL's drop from the base max_staleness, so with the judged "
                               "staleness compensation on, anomalies.decision_mismatch == judged samples kept by the "
                               "allowance (sample/judged_kept_by_allowance_total), by construction"},
        "v3_2": {"launcher": "$SELF", "launcher_md5": md5("$SELF"), "launcher_base": "run_g32_v3_1.sh",
                 "launcher_base_md5": "$LAUNCHER_BASE_MD5", "V32": "$V32",
                 "fast_logprob": "$FAST_LOGPROB", "logprob_bwd": "$RLFORGE_LOGPROB_BWD",
                 "fast_logprob_md5": md5(os.path.join("$RLFORGE_PKG", "fast_logprob.py")),
                 "subbatch_tokens": int("$SUBBATCH_TOKENS"), "sb_ckpt": "$SB_CKPT", "balance_rows": "$BALANCE_ROWS",
                 "sb_mem_env": {k: os.environ.get(k) for k in ("RLFORGE_SB_ACT_GB", "RLFORGE_SB_MEM_FULL_KB",
                                "RLFORGE_SB_MEM_CKPT_KB", "RLFORGE_SB_MEM_BASE_KB", "RLFORGE_SB_BUCKET_LAM",
                                "RLFORGE_BAL_ATTN_CTX", "PYTORCH_CUDA_ALLOC_CONF")},
                 "ref_model": os.environ.get("RLFORGE_REF_MODEL"),
                 "judged_stale_fixed": os.environ.get("RLFORGE_JUDGED_STALE_FIXED"),
                 "doc": "aiq_rl/docs/reports/V3_2_THROUGHPUT_2026-10-03.md"},
        "v3_1": {"launcher": "$SELF", "launcher_md5": md5("$SELF"),
                 "launcher_base": "run_g32_v3.sh", "launcher_base_md5": "$LAUNCHER_BASE_MD5",
                 "code_root": "$RLFORGE_V3",
                 "trainer_md5": md5(os.path.join("$RLFORGE_PKG", "trainer.py")),
                 "score_loop_md5": md5(os.path.join("$RLFORGE_PKG", "score_loop.py")),
                 "prefix_share_md5_expected": "$PREFIX_SHARE_MD5",
                 "trainer_merge": "rlforge_v3_next trainer f80dd387 (= prod 5ec9b09f + drop-audit fixes) + "
                                  "trainer.py.score_loop.diff (made against bf0db39b)",
                 "groups_guard": "$GPS_GUARD_MSG", "groups_per_step_target": "${GPS_TARGET:-}" or None,
                 "groups_per_step": int("$GROUPS_PER_STEP"), "num_trainer": int("$NUM_TRAINER"),
                 "trainer_gpus": "$TRAINER_GPUS", "n_trainer_gpus": int("$N_TRAINER_GPUS"),
                 "defaults_vs_v3_file": {"INFLIGHT": "640 (v3 file 4096; v3 run launched with 640)",
                                         "QUEUE_MAXSIZE": "512 (v3 file: TRL 1024; v3 run launched with 512)",
                                         "DROP_AUDIT": "on (v3 run launched with RLFORGE_DROP_AUDIT=on)",
                                         "RUN_NAME": "required", "NUM_TRAINER": "= count(TRAINER_GPUS)"},
                 "knobs": {"INFLIGHT": "$INFLIGHT", "QUEUE_MAXSIZE": "$QUEUE_MAXSIZE", "DROP_AUDIT": "$DROP_AUDIT",
                           "RLFORGE_DROP_AUDIT_PATH": os.environ.get("RLFORGE_DROP_AUDIT_PATH"),
                           "GROUPS_PER_STEP": "${GPS_TARGET:-}", "GPS_GUARD": "$GPS_GUARD",
                           "TRAINER_GPUS": "$TRAINER_GPUS", "NUM_TRAINER": "$NUM_TRAINER", "CPS": "$CPS",
                           "SCORE_CONC": "$SCORE_CONC", "JUDGED_STALE": "$JUDGED_STALE", "EARLY_HOOKS": "$EARLY_HOOKS",
                           "SCORE_TASK_MAX_S": "$SCORE_TASK_MAX_S",
                           "RLFORGE_SCORE_TASK_MAX_S": os.environ.get("RLFORGE_SCORE_TASK_MAX_S"),
                           "RLFORGE_SCORE_LOOP_UNVETTED": os.environ.get("RLFORGE_SCORE_LOOP_UNVETTED"),
                           "REWARD": "$REWARD", "AIQ_HALLUC": "$AIQ_HALLUC", "API_SERVER_COUNT": "$API_SERVER_COUNT",
                           "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "$VLLM_HTTP_TIMEOUT_KEEP_ALIVE"}},
    },
    "reward": {"spec": "$REWARD", "code_md5": md5("$REWARD_FILE"), "correct": 1.0, "wrong": 0.0, "unparsed": -0.5, "truncated": -2.0,
               "ranking": "exact=+1 else 0.5/inv", "parse": "last <answer> after last </think>", "cap": int("$MAX_COMPLETION")},
    "command": "FP8=$FP8 FP8_ALIGN=$FP8_ALIGN DTYPE=$DTYPE "
               "RLFORGE_FP8_FORWARD=${RLFORGE_FP8_FORWARD:-native} VLLM_DISABLE_COMPILE_CACHE=${VLLM_DISABLE_COMPILE_CACHE:-0} "
               "RLFORGE_FP8_GRAPHS=${RLFORGE_FP8_GRAPHS:-0} RLFORGE_FP8_GRAPH_MAX_MB=${RLFORGE_FP8_GRAPH_MAX_MB:-1024} "
               "RLFORGE_FP8_HEAD_GRAPH=${RLFORGE_FP8_HEAD_GRAPH:-0} RLFORGE_FP8_FUSE_MLP=${RLFORGE_FP8_FUSE_MLP:-1} "
               "RLFORGE_FP8_ALIGN_COMPILE=${RLFORGE_FP8_ALIGN_COMPILE:-0} "
               "RUN_NAME=$RUN_NAME MAX_STEPS=$MAX_STEPS SAVE=$SAVE CPS=$CPS NGEN=$NGEN STALE=$STALE INFLIGHT=$INFLIGHT LR=$LR "
               "KL_BETA=$KL_BETA GSPO_EPS_LOW=$GSPO_EPS_LOW GSPO_EPS_HIGH=$GSPO_EPS_HIGH PREFIX_SHARE=$PREFIX_SHARE "
               "TOKEN_BUDGET=$TOKEN_BUDGET DP_ROUTE=$DP_ROUTE POSLOG_STEPS=$POSLOG_STEPS AIQ_HALLUC=$AIQ_HALLUC "
               "SCORE_CONC=$SCORE_CONC JUDGED_STALE=$JUDGED_STALE EARLY_HOOKS=$EARLY_HOOKS SCORE_TASK_MAX_S=$SCORE_TASK_MAX_S "
               "REWARD=$REWARD QUEUE_MAXSIZE=$QUEUE_MAXSIZE DROP_AUDIT=$DROP_AUDIT GROUPS_PER_STEP=${GPS_TARGET:-} "
               "RLFORGE_V3=$RLFORGE_V3 V32=$V32 FAST_LOGPROB=$FAST_LOGPROB SUBBATCH_TOKENS=$SUBBATCH_TOKENS SB_CKPT=$SB_CKPT "
               "BALANCE_ROWS=$BALANCE_ROWS INIT_MODEL=$MODEL RLFORGE_REF_MODEL=${RLFORGE_REF_MODEL:-} bash run_g32_v3_2.sh $MODE $SUFFIX",
}
json.dump(manifest, open(os.path.join("$OUT", "run.json"), "w"), indent=1)
print("[dp] run.json written to $OUT/run.json")
PY

# ---------------------------------------------------------------- teardown machinery
STOP_REASON="trainer exited"
SERVER_PID=""; TRAINER_PID=""; WD_PID=""; REMOTE_PID=""; REMOTE_PGID=""; REMOTE_LAUNCHED=0
LAUNCHER_PID=$$
WD_REASON_FILE="$OUT/.watchdog_reason"
rm -f "$WD_REASON_FILE"
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
descendants() { local c; for c in $(ps -o pid= --ppid "$1" 2>/dev/null); do echo "$c"; descendants "$c"; done; }
stop_tree() {  # TERM a local pid and all its descendants, KILL leftovers after 30 s
  local root="$1" pids p i alive
  [[ "$root" =~ ^[0-9]+$ ]] && [ "$root" -gt 1 ] || return 0
  pids="$root $(descendants "$root" | tr '\n' ' ')"
  kill -TERM $pids 2>/dev/null || true
  for i in $(seq 30); do
    alive=""; for p in $pids; do kill -0 "$p" 2>/dev/null && alive="$alive $p"; done
    [ -z "$alive" ] && return 0
    sleep 1
  done
  echo "[teardown] SIGKILL leftovers:$alive"; kill -KILL $alive 2>/dev/null || true
}
stop_remote() {  # one group kill of OUR headless session on 360-1 (pgid recorded at launch, or pidfile)
  local pg="$REMOTE_PGID"
  [ -z "$pg" ] && [ "$REMOTE_LAUNCHED" = "1" ] && pg=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_SSH" "cat $REMOTE_DIR/vllm_headless.pid 2>/dev/null" || true)
  [[ "$pg" =~ ^[0-9]+$ ]] && [ "$pg" -gt 1 ] || { echo "[teardown] no remote pgid; nothing to stop on $REMOTE_HOST_LABEL"; return 0; }
  echo "[teardown] stopping headless ranks on $REMOTE_HOST_LABEL (pgid $pg)"
  timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_SSH" "P=$pg; [ \"\$P\" -gt 1 ] || exit 0
    kill -TERM -- -\$P 2>/dev/null || exit 0
    for i in \$(seq 30); do kill -0 -- -\$P 2>/dev/null || exit 0; sleep 2; done
    echo 'SIGKILL remote group'; kill -KILL -- -\$P 2>/dev/null; exit 0" \
    || echo "[teardown] WARNING: ssh to $REMOTE_SSH failed; check 360-1 GPUs $REMOTE_GPUS by hand (pgid $pg)"
}
TORN=0
teardown() {
  [ "$TORN" = "1" ] && return 0
  TORN=1
  trap '' TERM INT
  [ -n "$WD_PID" ] && kill "$WD_PID" 2>/dev/null || true
  [ -s "$WD_REASON_FILE" ] && STOP_REASON="watchdog: $(cat "$WD_REASON_FILE")"
  echo "[teardown] $(date +%T) stop_reason=$STOP_REASON"
  record_stop
  [ -n "$TRAINER_PID" ] && stop_tree "$TRAINER_PID"
  [ -n "$SERVER_PID" ] && stop_tree "$SERVER_PID"
  stop_remote
  echo "[teardown] $(date +%T) done"
}
trap 'teardown' EXIT
trap 'STOP_REASON="killed (SIGTERM)"; teardown; exit 143' TERM
trap 'STOP_REASON="killed (SIGINT)";  teardown; exit 130' INT

# ---------------------------------------------------------------- start rollout (head, then headless)
CUDA_VISIBLE_DEVICES=$SERVER_GPUS VLLM_SERVER_DEV_MODE=1 \
  vllm serve "$MODEL" --port "$PORT" --api-server-count "$API_SERVER_COUNT" --data-parallel-size-local "$DP_LOCAL" \
    $VLLM_COMMON --weight-transfer-config '{"backend":"nccl"}' \
    "${FP8_VLLM_ARGS[@]}" \
    > "logs/vllm_dp${SUFFIX}.log" 2>&1 &
SERVER_PID=$!
echo "[dp] head vLLM pid $SERVER_PID (log logs/vllm_dp${SUFFIX}.log)"
sleep 5
REMOTE_LAUNCHED=1
REMOTE_OUT=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_SSH" "bash -s" <<< "$REMOTE_SCRIPT") \
  || { STOP_REASON="failed to start headless ranks on $REMOTE_HOST_LABEL"; exit 1; }
read -r REMOTE_PID REMOTE_PGID <<< "$(tail -1 <<< "$REMOTE_OUT")"
[[ "$REMOTE_PID" =~ ^[0-9]+$ ]] || { STOP_REASON="bad remote pid '$REMOTE_OUT'"; exit 1; }
[[ "$REMOTE_PGID" =~ ^[0-9]+$ ]] || REMOTE_PGID="$REMOTE_PID"
[ "$REMOTE_PGID" = "$REMOTE_PID" ] || echo "[dp] WARNING: remote pid $REMOTE_PID != pgid $REMOTE_PGID (group kill uses pgid)"
echo "[dp] headless ranks on $REMOTE_HOST_LABEL: pid $REMOTE_PID pgid $REMOTE_PGID (log $REMOTE_HOST_LABEL:$REMOTE_DIR/vllm_headless.log)"
"$VENV/bin/python" - "$OUT/run.json" "$REMOTE_PID" "$REMOTE_PGID" "$SERVER_PID" "$LAUNCHER_PID" <<'PY' || true
import json, sys
p, rpid, rpg, spid, lpid = sys.argv[1:]
m = json.load(open(p))
h = m["gpu_layout"]["rollout"]["headless"]; h["pid"], h["pgid"] = int(rpid), int(rpg)
m["gpu_layout"]["rollout"]["head"]["pid"] = int(spid); m["launcher_pid"] = int(lpid)
json.dump(m, open(p, "w"), indent=1)
PY

remote_alive() {  # 0 alive, 1 dead, 255 ssh failure
  local rc=0
  timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_SSH" "kill -0 $REMOTE_PID 2>/dev/null && exit 0 || exit 1" 2>/dev/null || rc=$?
  case "$rc" in 0|1) return "$rc" ;; *) return 255 ;; esac   # 124 (timeout) / 255 (ssh) = unknown
}
echo "[dp] waiting for vLLM DP server (head pid $SERVER_PID, $DP_TOTAL ranks)..."
for i in $(seq 1 180); do
  if curl -sf "localhost:$PORT/health" > /dev/null; then echo "[dp] server up"; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[dp] head died"; STOP_REASON="vllm head died at startup"; tail -60 "logs/vllm_dp${SUFFIX}.log"; exit 1
  fi
  if [ $((i % 6)) -eq 0 ]; then
    rc=0; remote_alive || rc=$?
    if [ "$rc" = "1" ]; then
      echo "[dp] headless ranks died"; STOP_REASON="headless ranks on $REMOTE_HOST_LABEL died at startup"
      ssh -o BatchMode=yes "$REMOTE_SSH" "tail -60 $REMOTE_DIR/vllm_headless.log" || true; exit 1
    fi
  fi
  sleep 5
done
if ! curl -sf "localhost:$PORT/health" > /dev/null; then
  echo "[dp] server never healthy"; STOP_REASON="vllm DP server never healthy"; exit 1
fi
WS=$(curl -sf "localhost:$PORT/get_world_size" | "$VENV/bin/python" -c "import json,sys; print(json.load(sys.stdin)['world_size'])" || echo "?")
echo "[dp] /get_world_size = $WS (expected $((DP_TOTAL * TP)))"
if [ "$WS" != "$((DP_TOTAL * TP))" ]; then STOP_REASON="world size $WS != $((DP_TOTAL * TP))"; exit 1; fi
NENG=$(curl -sf "localhost:$PORT/metrics" | grep -c '^vllm:generation_tokens_total{' || true)
echo "[dp] engines reporting in /metrics: $NENG"

# ---------------------------------------------------------------- watchdog
watchdog() {
  local misses=0 hfails=0 rc
  fire() { echo "$1" > "$WD_REASON_FILE"; echo "[watchdog] $(date +%T) $1 -> SIGTERM launcher $LAUNCHER_PID"; kill -TERM "$LAUNCHER_PID"; exit 0; }
  while true; do
    sleep "$WD_INTERVAL"
    kill -0 "$SERVER_PID" 2>/dev/null || fire "vllm head (pid $SERVER_PID) died"
    rc=0; remote_alive || rc=$?
    if [ "$rc" = "0" ]; then misses=0
    elif [ "$rc" = "1" ]; then
      sleep 5; rc=0; remote_alive || rc=$?
      [ "$rc" = "1" ] && fire "headless DP ranks on $REMOTE_HOST_LABEL (pid $REMOTE_PID) died"
    else
      misses=$((misses + 1)); echo "[watchdog] $(date +%T) ssh to $REMOTE_SSH failed ($misses/$WD_SSH_MISSES)"
      [ "$misses" -ge "$WD_SSH_MISSES" ] && fire "lost ssh to $REMOTE_HOST_LABEL for $misses checks"
    fi
    if curl -sf -m 20 "localhost:$PORT/health" > /dev/null; then hfails=0
    else
      hfails=$((hfails + 1)); echo "[watchdog] $(date +%T) head /health failed ($hfails/$WD_HEALTH_FAILS)"
      [ "$hfails" -ge "$WD_HEALTH_FAILS" ] && fire "head /health failing for $hfails checks"
    fi
  done
}
watchdog &
WD_PID=$!
echo "[dp] watchdog pid $WD_PID (every ${WD_INTERVAL}s: head pid, $REMOTE_HOST_LABEL pid $REMOTE_PID, /health)"

# ---------------------------------------------------------------- trainer
# Runs in the background and is reaped with `wait`, so a SIGTERM (operator or watchdog) reaches the
# trap immediately instead of after the foreground pipeline ends.
set +e
( CUDA_VISIBLE_DEVICES=$TRAINER_GPUS "${ACC_LAUNCH[@]}" "${TRAINER_ARGS[@]}" $EXTRA_ARGS \
    2>&1 | tee "logs/trainer_dp${SUFFIX}.log" ) &
TRAINER_PID=$!
echo "[dp] trainer pid $TRAINER_PID (log logs/trainer_dp${SUFFIX}.log)"
wait "$TRAINER_PID"
TRAINER_RC=$?
TRAINER_PID=""
set -e
if [ "$TRAINER_RC" = "0" ]; then STOP_REASON="completed"; else STOP_REASON="trainer exited rc=$TRAINER_RC"; fi
echo "[dp] done (trainer rc=$TRAINER_RC)"
exit "$TRAINER_RC"
