#!/bin/bash
# Pull the wave4 kill-line ablation jsonl mirrors and replot.
# t1 and t1_3 both lack rsync for inbound pushes into our tree here, so pull
# with tar-over-ssh (recently changed files only). Local compute is a few
# seconds of matplotlib.
set -u
cd "$(dirname "$0")/.."
PY=python3

pull_wave3c() {
  mkdir -p dqn_runs/node_c/wave3c_20260929/w3c_tf_xl
  ssh -o BatchMode=yes -o ConnectTimeout=20 node_c \
    "cd /path/to/suika-dqn/runs/wave3c_20260929/w3c_tf_xl && tar cf - --ignore-failed-read --newer-mtime='90 minutes ago' *.jsonl 2>/dev/null" \
    | tar xf - -C dqn_runs/node_c/wave3c_20260929/w3c_tf_xl/ 2>/dev/null
}

pull_cont() {
  mkdir -p dqn_runs/node_c/wave4_20260929/w4_k200_cont
  ssh -o BatchMode=yes -o ConnectTimeout=20 node_c \
    "cd /path/to/suika-dqn/runs/wave4_20260929/w4_k200_cont && tar cf - --ignore-failed-read *.jsonl 2>/dev/null" \
    | tar xf - -C dqn_runs/node_c/wave4_20260929/w4_k200_cont/ 2>/dev/null
}

pull_scratch() {
  mkdir -p dqn_runs/node_a/wave4_20260929/w4_k200_scratch
  ssh -o BatchMode=yes -o ConnectTimeout=20 node_a \
    "cd /path/to/suika-dqn/runs/wave4_20260929/w4_k200_scratch && tar cf - --ignore-failed-read *.jsonl 2>/dev/null" \
    | tar xf - -C dqn_runs/node_a/wave4_20260929/w4_k200_scratch/ 2>/dev/null
}

pull_wave3c
pull_cont
pull_scratch

$PY suika_dqn/plot_wave4.py

for H in node_c node_a; do
  N=$(ssh -o BatchMode=yes -o ConnectTimeout=20 $H \
      "ps aux | grep -c '[r]un_arm.py'" 2>/dev/null || echo '?')
  echo "[$(date '+%m-%d %H:%M')] $H orchestrators=$N"
done
