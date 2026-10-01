#!/bin/bash
# Pull wave6 (w6_h720_fixed + w6_debug on node_a, w6_var on node_b) into
# dqn_runs/. Same discipline as wave5: raw actions_a*.jsonl stay on the nodes;
# each node digests them first, only digests + metrics/eval logs come back.
# Usage: pull_wave6.sh [TOPK]
set -u
cd "$(dirname "$0")/.."
TOPK=${1:-100}
ROOT=/path/to/suika-dqn/runs/wave6_20260930
DIGEST=suika_dqn/digest_actions.py

pull_arm() {  # HOST ARM
  local H=$1 A=$2 D=dqn_runs/$1/wave6_20260930/$2
  mkdir -p "$D"
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$H" "python3 - $ROOT/$A $TOPK" < $DIGEST 2>/dev/null
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$H" \
    "cd $ROOT/$A && tar cf - --ignore-failed-read metrics.jsonl eval.jsonl eval_actions.jsonl digest_top_actions.jsonl digest_episode_bins.jsonl policy_best.json run.log learner.log evaluator.log 2>/dev/null" \
    | tar xf - -C "$D"
  du -sh "$D" | tr '\n' ' '; echo
}

pull_arm node_a w6_debug
pull_arm node_a w6_h720_fixed
pull_arm node_b w6_var
pull_arm node_b w6_max_fixed
ROOT_360=/data/shared/suika-dqn/runs/wave6_20260930
D=dqn_runs/node_f/wave6_20260930/w6_var_xl
mkdir -p "$D"
ssh -o BatchMode=yes -o ConnectTimeout=20 node_f "/data/shared/suika-dqn/suika-venv/bin/python - $ROOT_360/w6_var_xl $TOPK" < $DIGEST 2>/dev/null
ssh -o BatchMode=yes -o ConnectTimeout=20 node_f \
  "cd $ROOT_360/w6_var_xl && tar cf - --ignore-failed-read metrics.jsonl eval.jsonl eval_actions.jsonl digest_top_actions.jsonl digest_episode_bins.jsonl policy_best.json run.log learner.log evaluator.log supervisor.log 2>/dev/null" \
  | tar xf - -C "$D"
du -sh "$D" | tr '\n' ' '; echo

for H in node_a node_b node_f; do
  N=$(ssh -o BatchMode=yes -o ConnectTimeout=20 $H "ps aux | grep -c '[r]un_arm.py'" 2>/dev/null || echo '?')
  echo "[$(date '+%m-%d %H:%M')] $H orchestrators=$N"
done
