#!/usr/bin/env bash
# CPU-only end-to-end test of the integrated launcher: tiny Qwen3.8 + 2 fake vLLM backends (real HF sampling)
# + real lora_agent + real router + e2e_gspo.py under torchrun (2 ranks, gloo, FSDP2), 5 steps.
#   run A: max_stale 1 (async, production-like)   run B: max_stale 0 (on-policy every step)
# Pass criteria (checked at the end):
#   * |log rho| ~ 0 (fp32) wherever the trainer weights equal the serving adapter (A: steps 1-2, B: every step)
#     -> adapter naming, gather/save, router push to the agent, backend load and token/logprob alignment are right
#   * |log rho| >> 0 where they differ (A: stale steps after the first lr>0 update) -> updates move the policy
#   * prefix-shared logp == per-seq logp (verify_prefix), grad_norm > 0 after the lr-0 step, 5 versions loaded
# Usage: CODE=<dir with src/ and tools/> [PY=python] [SRC_CFG=<dir with the 27B config.json>]
#        [PROMPTS_SRC=prompts_27b_tok.jsonl | ANSWERS_SRC=train_think.jsonl + TOKENIZER=<dir>] bash cpu_e2e_test.sh
# Works on a Tione host (sources env.sh) or locally (macOS: perl setsid, no GNU timeout needed).
set -u
ENV_SH=${ENV_SH:-/home/tione/guoshaoyang/a100_rl/env.sh}
[ -f "$ENV_SH" ] && source "$ENV_SH"
PY=${PY:-python}
CODE=${CODE:-$(cd "$(dirname "$0")/../.." && pwd)}
W=${W:-${TMPDIR:-/tmp}/e2e_cpu_test/$(date +%m%d_%H%M%S)}
SRC_CFG=${SRC_CFG:-${MODELS:-/home/tione/guoshaoyang/models}/Qwen3.8-27B}
PROMPTS_SRC=${PROMPTS_SRC:-/home/tione/guoshaoyang/a100_rl/data/prompts_27b_tok.jsonl}
ANSWERS_SRC=${ANSWERS_SRC:-/home/tione/guoshaoyang/a100_rl/data/src/train_think.jsonl}
TOKENIZER=${TOKENIZER:-$SRC_CFG}
mkdir -p $W
export PYTHONPATH=$CODE/src:${PYTHONPATH:-}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 FAKE_THREADS=4 CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false
TO=$(command -v timeout || command -v gtimeout || true); TO=${TO:+$TO 1500}
if command -v setsid >/dev/null; then SETSID=(setsid); else SETSID=(perl -e 'use POSIX qw(setsid); setsid(); exec @ARGV or die'); fi
start() { local name=$1; shift; nohup "${SETSID[@]}" "$@" > $W/$name.log 2>&1 < /dev/null & local pid=$!; sleep 0.5
  echo "$pid $(ps -o pgid= -p $pid | tr -d ' ') $name" >> $W/pids.txt; }
teardown() { while read -r pid pgid name; do kill -TERM -- -$pgid 2>/dev/null; done < $W/pids.txt; sleep 2
  while read -r pid pgid name; do kill -KILL -- -$pgid 2>/dev/null; done < $W/pids.txt; }
trap teardown EXIT
freeport() { $PY -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1])"; }

$PY $CODE/tools/e2e/make_tiny.py --src $SRC_CFG --out $W/tiny > $W/make_tiny.log 2>&1 || { cat $W/make_tiny.log; exit 1; }
tail -1 $W/make_tiny.log
# short real prompts: first 160..400 tokens of real AIQ prompts (same question_id -> real answers)
$PY - "$W" "$PROMPTS_SRC" "$ANSWERS_SRC" "$TOKENIZER" <<'EOF'
import json, os, sys, random
w, psrc, asrc, tok = sys.argv[1:5]; rnd = random.Random(0); out = open(f"{w}/prompts.jsonl", "w")
if os.path.exists(psrc):
    rows = ((json.loads(l)["question_id"], json.loads(l)["input_ids"]) for l in open(psrc))
else:  # render real AIQ prompts with the model's chat template (thinking on, as in the 27B data)
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(tok)
    def gen():
        for l in open(asrc):
            r = json.loads(l)
            yield r["question_id"], tk.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=True,
                                                          return_dict=False)
    rows = gen()
for i, (qid, ids) in enumerate(rows):
    ids = list(ids["input_ids"] if isinstance(ids, dict) else ids); n = rnd.randint(160, 400)
    out.write(json.dumps({"question_id": qid, "input_ids": ids[:n], "n_tok": n}) + "\n")
    if i >= 63: break
out.close(); print("prompts", i + 1)
EOF

P1=$(freeport); P2=$(freeport); PA=$(freeport); PR=$(freeport)
start fake0 $PY $CODE/tools/e2e/fake_vllm.py --model $W/tiny --port $P1
start fake1 $PY $CODE/tools/e2e/fake_vllm.py --model $W/tiny --port $P2
start agent $PY -m rlforge.rollout.lora_agent --host 127.0.0.1 --port $PA --root $W/agent_recv
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:$P1/health >/dev/null && curl -sf http://127.0.0.1:$P2/health >/dev/null \
    && curl -sf http://127.0.0.1:$PA/health >/dev/null && break; sleep 1; done
# SHIP=keep: the fp32 fake serves the fp32 adapter exactly (the |log rho|~0 check); SHIP=bf16 (production default:
# vLLM serves LoRA in the model dtype) adds the bf16 rounding of the LoRA weights to the mismatch.
start router $PY -m rlforge.rollout.router --host 127.0.0.1 --port $PR --stage-root $W/stage --log $W/router.jsonl \
  --ship-dtype ${SHIP:-keep} \
  --backend http://127.0.0.1:$P1 --backend http://127.0.0.1:$P2,agent=http://127.0.0.1:$PA --health-s 2
for i in $(seq 1 60); do curl -sf http://127.0.0.1:$PR/health >/dev/null && break; sleep 1; done
curl -s http://127.0.0.1:$PR/router/state | head -c 600; echo

COMMON="--device cpu --model $W/tiny --prompts $W/prompts.jsonl --answers $ANSWERS_SRC --router http://127.0.0.1:$PR
  --steps 5 --G 4 --max-tokens 12 --chunk 3 --lr 2e-2 --lr0-steps 1 --backend-precision 0=fake0,1=fake1
  --rollout-workers 4 --verify-prefix 2 --request-timeout 600"
RC=0
# name:max_stale:nproc:replicate:groups   (C = HSDP 2x2: replicate 2 x shard 2, gloo)
for RUN in ${RUNS:-A:1:2:1:2 B:0:2:1:2 C:1:4:2:4}; do
  IFS=: read -r name st np rep ng <<< "$RUN"
  t0=$(date +%s)
  $TO $PY -m torch.distributed.run --nproc_per_node $np --master_port $(freeport) $CODE/tools/e2e/e2e_gspo.py $COMMON \
     --max-stale $st --replicate $rep --groups $ng --adapter-name pol$name --out $W/run$name > $W/run$name.log 2>&1
  rc=$?; echo "run$name rc=$rc wall=$(( $(date +%s) - t0 ))s"; [ $rc -ne 0 ] && { RC=$rc; tail -40 $W/run$name.log; }
done

$PY - "$W" <<'EOF'
import json, os, sys
w = sys.argv[1]; res = {"dir": w, "checks": {}}; ok = True
def steps(n):
    try: return [json.loads(l) for l in open(f"{w}/run{n}/steps.jsonl")]
    except FileNotFoundError: return []
def p90(r): return max(v["tok_abs_p90"] for v in r["abs_log_rho"].values())
A, B, C = steps("A"), steps("B"), steps("C")
c = res["checks"]
c["A_steps"] = len(A); c["B_steps"] = len(B)
c["A_rho_p90"] = [round(p90(r), 7) for r in A]; c["B_rho_p90"] = [round(p90(r), 7) for r in B]
c["A_staleness"] = [r["staleness"] for r in A]; c["B_staleness"] = [r["staleness"] for r in B]
c["A_grad_norm"] = [r["grad_norm"] for r in A]
c["verify_prefix_max"] = max([v["max_abs_logp_diff"] for r in A + B for v in r["verify_prefix"]] or [None])
c["A_backends"] = sorted({b for r in A for b in r["backends"]})
c["C_steps"] = len(C); c["C_rho_p90"] = [round(p90(r), 7) for r in C]; c["C_grad_norm"] = [r["grad_norm"] for r in C]
def zero_expected(r):  # trainer weights v_{s-1} == serving v_b iff no lr>0 step in (b, s-1]  (lr0_steps = 1)
    return r["staleness"] == 0 or r["step"] - 1 <= 1
c["A_zero_expected"] = [zero_expected(r) for r in A]
tests = {
  "5_steps_each": len(A) == 5 and len(B) == 5,
  "rho_zero_where_weights_equal": all(p90(r) < 1e-4 for r in A + B + C if zero_expected(r)),
  "rho_visible_where_weights_differ": all(p90(r) > 1e-3 for r in A + B + C if not zero_expected(r)),
  "C_hsdp_5_steps": len(C) == 5 or "C" not in os.environ.get("RUNS", "A B C"),
  "B_staleness_zero": len(B) > 0 and all(r["staleness"] == 0 for r in B),
  "A_staleness_le1": all(r["staleness"] <= 1 for r in A),
  "grad_nonzero": len(A) >= 2 and all(r["grad_norm"] > 0 for r in A[1:]),
  "prefix_vs_perseq": c["verify_prefix_max"] is not None and c["verify_prefix_max"] < 1e-4,
  "both_backends_used": len(c["A_backends"]) == 2,
}
res["pass"] = tests; res["ok"] = all(tests.values())
for n in "ABC":
    try: res[f"summary_{n}"] = {k: v for k, v in json.load(open(f"{w}/run{n}/summary.json")).items() if k != "router_state"}
    except FileNotFoundError: pass
json.dump(res, open(f"{w}/cpu_e2e_result.json", "w"), indent=1)
print(json.dumps(res, indent=1))
EOF
exit $RC
