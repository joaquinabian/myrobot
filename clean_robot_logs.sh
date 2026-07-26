#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/programas/myrobot}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs}"
KEEP="${KEEP:-10}"

if [[ ! -d "$LOG_ROOT" ]]; then
    echo "[robot-clean-logs] no logs folder: $LOG_ROOT"
    exit 0
fi

echo "[robot-clean-logs] keeping newest $KEEP log runs in $LOG_ROOT"

# Remove old timestamped log folders, keep the newest $KEEP.
find "$LOG_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
    | sort -rn \
    | awk -v keep="$KEEP" 'NR > keep {print $2}' \
    | while read -r olddir; do
        echo "[robot-clean-logs] removing $olddir"
        rm -rf "$olddir"
    done

# Recreate latest link if missing.
latest_dir="$(find "$LOG_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
    | sort -rn | awk 'NR == 1 {print $2}')"

if [[ -n "${latest_dir:-}" ]]; then
    ln -sfn "$latest_dir" "$LOG_ROOT/latest"
fi
