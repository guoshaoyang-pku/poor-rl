#!/bin/bash
# Trainer node r>0 of a multi-node run: runs this node's accelerate launcher from the plan, records its exit.
# Usage: worker.sh PLAN_JSON   (the head must be started first; this node's IP must be in plan['worker_argvs'])
set -u
source /home/tione/guoshaoyang/a100_rl/env.sh
IP=$(hostname -I | awk '{print $1}')
RUN=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['run_dir'])" "$1")
setsid nohup bash -c '
  eval "$(python - "$1" "$2" <<PY
import json, shlex, sys
p = json.load(open(sys.argv[1]))
for k, v in {**p["env"], **p["trainer_env"]}.items():
    print(f"export {k}={shlex.quote(str(v))}")
print("ARGV=(" + " ".join(shlex.quote(a) for a in p["worker_argvs"][sys.argv[2]]) + ")")
PY
)"
  cd "$3"; "${ARGV[@]}" >> "$3/trainer_node_$2.log" 2>&1; rc=$?
  echo "{\"utc\": \"$(date -u +%FT%TZ)\", \"rc\": $rc}" > "$3/worker_$2_exit.json"
' worker "$1" "$IP" "$RUN" > /dev/null 2>&1 < /dev/null &
echo "worker $IP pid $! run $RUN"
