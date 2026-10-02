#!/bin/bash
# Pull rollout samples + run meta + companion-eval sample files from the cluster
# into artifacts/samples/<run_id>/ for the hub viewer. Usage: sync_samples.sh <run_id> [remote_run_dir]
set -u
RUN="${1:?run id}"
R2="${R2:-360-2}"
REMOTE_RUN="${2:-/data/home/guoshaoyang/aiq_rl/runs/$RUN}"
STORE=/data/shared/guoshaoyang/aiq_rl_store
DST="$(cd "$(dirname "$0")/.." && pwd)/artifacts/samples/$RUN"
mkdir -p "$DST"
scp -q "$R2:$REMOTE_RUN/rollout_samples.jsonl" "$DST/train.jsonl" 2>/dev/null && echo "[sync] train samples: $(wc -l < "$DST/train.jsonl") rows"
scp -q "$R2:$REMOTE_RUN/run.json" "$DST/meta.json" 2>/dev/null && python3 - "$DST/meta.json" <<'PY'
import json,sys,datetime
p=sys.argv[1]; m=json.load(open(p))
try: m['started_ts']=datetime.datetime.strptime(m['started'],'%Y-%m-%d %H:%M:%S').timestamp()
except Exception: pass
json.dump(m,open(p,'w'),ensure_ascii=False,indent=1)
PY
# companion eval results + their display samples (named <run>-<ckpt>-eval300.json / -samples.jsonl)
ssh "$R2" "ls $STORE/evals/ 2>/dev/null" | grep -F "$RUN" | grep -E "samples\.jsonl$|eval300\.json$" | while read -r f; do
  case "$f" in
    *samples.jsonl) out="eval_$(echo "$f" | sed "s/.*-\(checkpoint-[0-9]*\|final\)-samples.jsonl/\1/").jsonl";;
    *) continue;;
  esac
  scp -q "$R2:$STORE/evals/$f" "$DST/$out" 2>/dev/null && echo "[sync] $f -> $out"
done
echo "[sync] done -> $DST"
