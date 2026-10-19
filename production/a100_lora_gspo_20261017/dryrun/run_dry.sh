#!/bin/bash
# No-GPU failure drill for resume_watch.py, run on a100_t1. Uses the real head.py, router, worker.sh and make_plan.py
# (through kit/make_plan.py) with mock vLLM backends and the fake trainer/worker/eval in fake_trainer.py. Both
# watchers run on this host; the node watcher plays a100_t1_2. Scenarios: scenarios.json. Touches nothing outside
# its own dir, ports PORT..PORT+3 (mock backends) and PORT+100 (router).
# Usage: run_dry.sh A|B|C PORT          e.g. run_dry.sh A 18200
set -u
D=$(cd "$(dirname "$0")" && pwd); KIT=$(dirname "$D")
source /home/tione/guoshaoyang/a100_rl/env.sh
S=$1; P=$2; IP=$(hostname -I | awk '{print $1}')
ROOT=/home/tione/guoshaoyang/aiq_rl/q35_resume_dry/$S-$(date +%m%d_%H%M%S); mkdir -p "$ROOT/runs"
python -c "import json,sys;json.dump(json.load(open(sys.argv[1]))[sys.argv[2]],open(sys.argv[3],'w'),indent=1)" \
  "$D/scenarios.json" "$S" "$ROOT/scenario.json"
setsid nohup python "$D/mock_vllm.py" "$ROOT/mock.jsonl" "$P" 4 > "$ROOT/mock.log" 2>&1 < /dev/null &
MOCK=$!
export DRY_ROOT=$ROOT DRY_ROUTER_PORT=$((P + 100)) DRY_BACKENDS=$(seq -s, -f "http://127.0.0.1:%g" "$P" $((P + 3)))
# r1 as the operator starts it (like r5): plan, head, then the node-1 worker by hand
TRAINER_NODES=$IP,$IP CPS_PER_RANK=128 SB_TOKENS=16384 SB_ACT_GB=10 python "$D/kit/make_plan.py" r1 > "$ROOT/r1_plan.json"
bash "$KIT/start_head.sh" "$ROOT/r1_plan.json"; sleep 5
bash "$KIT/worker.sh" "$ROOT/runs/r1/plan.json"
W=(--kit "$D/kit" --poll-s 3 --kill-after-s 20 --worker-exit-wait-s 120 --head-exit-wait-s 120 --gpu-wait-s 30
   --node-wait-s 60 --gpu-mib -1)
setsid nohup python "$KIT/resume_watch.py" head "$ROOT/runs/r1" "${W[@]}" > "$ROOT/head_watch.log" 2>&1 < /dev/null &
setsid nohup python "$KIT/resume_watch.py" node "$ROOT/runs/r1" "${W[@]}" > "$ROOT/node_watch.log" 2>&1 < /dev/null &
for _ in $(seq 240); do   # up to 20 min
  st=$(python -c "import json,sys;print(json.load(open(sys.argv[1])).get('status'))" "$ROOT/resume_watch/head_watch.json" 2>/dev/null)
  case $st in done|blocked) break ;; esac
  sleep 5
done
sleep 20   # node watcher and fake evals notice the end
python "$D/check_dry.py" "$ROOT"; rc=$?
kill "$MOCK"; pkill -f "$ROOT/" 2>/dev/null   # every drill process carries ROOT in its argv
echo "dry $S rc=$rc root $ROOT"
exit $rc
