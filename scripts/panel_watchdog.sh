#!/bin/bash
# Keepalive watchdog for rlforge local panels (hub 63400 / AIQ 63401 / Suika 63402 / SFT 63403)
# and the panel mirror daemon. Restarts whatever stops responding.
ROOT="/Users/guoshaoyang/Desktop/workdir/rlforge"
VENV="/tmp/rlforge-swanlab-preview-venv"
SESSION="rlforge-panels"

start_panel() {
  local project="$1" port="$2" win
  win=$(echo "$project" | tr 'A-Z' 'a-z')
  # skip if a window with this name already exists and its pane is alive
  if tmux list-windows -t "$SESSION" -F '#{window_name} #{pane_dead}' 2>/dev/null | grep -q "^$win 0"; then
    return
  fi
  tmux kill-window -t "$SESSION:$win" 2>/dev/null || true
  tmux new-window -t "$SESSION" -n "$win" \
    "cd '$ROOT' && ROOT='$ROOT' PROJECT='$project' PORT='$port' VENV='$VENV' bash scripts/panel.sh 2>&1 | tee /tmp/panel_${win}.log"
  echo "[watchdog] $(date '+%F %T') restarted $project on :$port" >> /tmp/panel_watchdog.log
}

start_home() {
  if tmux list-windows -t "$SESSION" -F '#{window_name} #{pane_dead}' 2>/dev/null | grep -q "^home 0"; then
    return
  fi
  tmux kill-window -t "$SESSION:home" 2>/dev/null || true
  tmux new-window -t "$SESSION" -n "home" \
    "cd '$ROOT/scripts' && python3 home.py 2>&1 | tee /tmp/home.log"
  echo "[watchdog] $(date '+%F %T') restarted hub on :63400" >> /tmp/panel_watchdog.log
}

start_mirror() {
  if pgrep -f "panel_mirror.py --loop" >/dev/null 2>&1; then
    return
  fi
  nohup "$VENV/bin/python" "$ROOT/scripts/panel_mirror.py" --loop 300 >> /tmp/panel_mirror.log 2>&1 &
  echo "[watchdog] $(date '+%F %T') restarted panel mirror daemon" >> /tmp/panel_watchdog.log
}

while true; do
  if ! curl -s -o /dev/null --max-time 5 "http://127.0.0.1:63400/"; then
    start_home
  fi
  for spec in "AIQ 63401" "Suika 63402" "SFT 63403"; do
    set -- $spec
    if ! curl -s -o /dev/null --max-time 5 "http://127.0.0.1:$2/"; then
      start_panel "$1" "$2"
    fi
  done
  start_mirror
  sleep 60
done
