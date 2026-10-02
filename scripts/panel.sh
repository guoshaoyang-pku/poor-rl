#!/bin/bash
set -euo pipefail
ROOT="${ROOT:?set ROOT}"
PORT="${PORT:-5092}"
HOST="${HOST:-127.0.0.1}"
PROJECT="${PROJECT:-AIQ}"
case "$PROJECT" in
  AIQ|Suika|SFT) ;;
  *) echo "PROJECT must be AIQ, Suika, or SFT (got $PROJECT)" >&2; exit 2 ;;
esac
LOGDIR="${SWANLAB_LOGDIR:-$ROOT/artifacts/swanlab/$PROJECT/swanlog}"
VENV="${VENV:-}"
SWANLAB="${VENV:+$VENV/bin/}swanlab"

if [ ! -x "$SWANLAB" ]; then
  SWANLAB="$(command -v swanlab || true)"
fi
if [ -z "$SWANLAB" ]; then
  echo "SwanLab CLI not found. Install with: pip install -e '.[panel]'" >&2
  exit 127
fi

# Best-effort: give each panel a "back to experiment home" button (no-op if already injected).
INJECTOR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/inject_home_link.py"
PYBIN="${VENV:+$VENV/bin/python}"
if [ -n "$PYBIN" ] && [ -x "$PYBIN" ] && [ -f "$INJECTOR" ]; then
  "$PYBIN" "$INJECTOR" >/dev/null 2>&1 || echo "[panel] home-link injection skipped" >&2
fi

mkdir -p "$LOGDIR"
exec "$SWANLAB" watch "$LOGDIR" --host "$HOST" --port "$PORT"
