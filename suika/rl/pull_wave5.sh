#!/bin/bash
# Pull wave5 (w5_k220_fixed on node_a, w5_var on node_b) into dqn_runs/.
# Raw per-actor actions_a*.jsonl stay on the nodes (~GB/day); each node first
# digests them into top-K replayable action sequences + 10-min score bins, and
# only those digests, metrics/eval logs and eval_actions.jsonl come back.
# Local work is tar extraction only. Usage: pull_wave5.sh [TOPK]
set -u
cd "$(dirname "$0")/.."
TOPK=${1:-100}
ROOT=/path/to/suika-dqn/runs/wave5_20260929
DIGEST=suika_dqn/digest_actions.py

pull_arm() {  # HOST ARM
  local H=$1 A=$2 D=dqn_runs/$1/wave5_20260929/$2
  mkdir -p "$D"
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$H" "python3 - $ROOT/$A $TOPK" < $DIGEST
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$H" \
    "cd $ROOT/$A && tar cf - --ignore-failed-read metrics.jsonl eval.jsonl eval_actions.jsonl digest_top_actions.jsonl digest_episode_bins.jsonl run.log learner.log evaluator.log 2>/dev/null" \
    | tar xf - -C "$D"
  du -sh "$D" | tr '\n' ' '; echo
}

pull_arm node_a w5_k220_fixed
pull_arm node_b w5_var

for H in node_a node_b; do
  N=$(ssh -o BatchMode=yes -o ConnectTimeout=20 $H "ps aux | grep -c '[r]un_arm.py'" 2>/dev/null || echo '?')
  echo "[$(date '+%m-%d %H:%M')] $H orchestrators=$N"
done
