#!/bin/bash
# Pull latest wave3b (perm) / wave3c (t1) jsonl mirrors and replot curves.
# Lightweight: ssh + tar/scp only, no local compute beyond a few-second plot.
set -u
cd "$(dirname "$0")/.."
PY=python3

# perm has rsync; t1 does not (use tar of recently-changed files)
rsync -az --include='*.jsonl' --exclude='*' -e "ssh -o BatchMode=yes -o ConnectTimeout=20" \
  node_d:/data/user/suika-dqn/runs/wave3b_20260928/w3b_mlp_deep/ \
  dqn_runs/node_d/wave3b_20260928/w3b_mlp_deep/ 2>/dev/null

mkdir -p dqn_runs/node_d/wave4_20260928/w4_qwen_text
rsync -az --include='*.jsonl' --exclude='*' -e "ssh -o BatchMode=yes -o ConnectTimeout=20" \
  node_d:/data/user/suika-dqn/runs/w4_20260928/w4_qwen_text/ \
  dqn_runs/node_d/wave4_20260928/w4_qwen_text/ 2>/dev/null

mkdir -p dqn_runs/node_c/wave3c_20260929/w3c_tf_xl
ssh -o BatchMode=yes -o ConnectTimeout=20 node_c \
  "cd /path/to/suika-dqn/runs/wave3c_20260929/w3c_tf_xl && tar cf - --ignore-failed-read --newer-mtime='40 minutes ago' *.jsonl 2>/dev/null" \
  | tar xf - -C dqn_runs/node_c/wave3c_20260929/w3c_tf_xl/ 2>/dev/null

$PY suika_dqn/plot_wave2.py

# one-line health report
P_PERM=$(ssh -o BatchMode=yes -o ConnectTimeout=20 node_c "ps aux | grep -c w3c_tf_x[l]" 2>/dev/null || echo '?')
P_MLP=$(ssh -o BatchMode=yes -o ConnectTimeout=20 node_d "ps aux | grep -c w3b_mlp_dee[p]" 2>/dev/null || echo '?')
echo "[$(date '+%m-%d %H:%M')] pull+plot done; procs tf_xl=$P_PERM mlp=$P_MLP"
