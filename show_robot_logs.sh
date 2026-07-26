#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/programas/myrobot}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs}"
LOG_DIR="${1:-$LOG_ROOT/latest}"

if [[ ! -e "$LOG_DIR" ]]; then
    echo "[robot-logs] ERROR: log folder not found: $LOG_DIR" >&2
    exit 1
fi

if [[ -L "$LOG_DIR" ]]; then
    LOG_DIR="$(readlink -f "$LOG_DIR")"
fi

echo "[robot-logs] folder: $LOG_DIR"
echo "[robot-logs] Ctrl+C to stop"
echo

touch "$LOG_DIR/launcher.log" "$LOG_DIR/server.log" "$LOG_DIR/video.log" "$LOG_DIR/speech.log"

tail -n 20 -F \
    "$LOG_DIR/launcher.log" \
    "$LOG_DIR/server.log" \
    "$LOG_DIR/video.log" \
    "$LOG_DIR/speech.log"
