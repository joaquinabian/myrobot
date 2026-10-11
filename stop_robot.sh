#!/usr/bin/env bash
# Stop myrobot processes. ASCII only.

set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROC_ROOT="${PROC_ROOT:-/proc}"
KILL_COMMAND="${KILL_COMMAND:-kill}"
readonly STOP_SCRIPT_PID="$$"
export STOP_SCRIPT_PID
PROCESS_NAMES=(
    robot_server.py
    robot_video_client.py
    robot_speech_client.py
)
STOP_TIMEOUT="${STOP_TIMEOUT:-10}"

is_project_process() {
    local pid="$1"
    local process_name="$2"
    local process_cwd argument

    [[ "$pid" != "$STOP_SCRIPT_PID" ]] || return 1
    process_cwd="$(readlink -f "$PROC_ROOT/$pid/cwd" 2>/dev/null)" \
        || return 1

    while IFS= read -r -d '' argument; do
        if [[ "$argument" == "$PROJECT_DIR/$process_name" ]]; then
            return 0
        fi
        if [[ "$process_cwd" == "$PROJECT_DIR" \
              && ( "$argument" == "$process_name" \
                   || "$argument" == "./$process_name" ) ]]; then
            return 0
        fi
    done < "$PROC_ROOT/$pid/cmdline"

    return 1
}

PIDS=()
for process_name in "${PROCESS_NAMES[@]}"; do
    while IFS= read -r pid; do
        if is_project_process "$pid" "$process_name"; then
            PIDS+=("$pid")
        fi
    done < <(pgrep -f "$process_name" 2>/dev/null || true)
done

for pid in "${PIDS[@]}"; do
    "$KILL_COMMAND" -TERM "$pid" 2>/dev/null || true
done

deadline=$((SECONDS + STOP_TIMEOUT))
while (( SECONDS < deadline )); do
    running=0
    for pid in "${PIDS[@]}"; do
        if "$KILL_COMMAND" -0 "$pid" 2>/dev/null; then
            running=1
            break
        fi
    done
    (( running == 0 )) && break
    sleep 0.2
done

remaining=()
for pid in "${PIDS[@]}"; do
    if "$KILL_COMMAND" -0 "$pid" 2>/dev/null; then
        remaining+=("$pid")
    fi
done

if (( ${#remaining[@]} > 0 )); then
    printf 'robot processes still running after SIGTERM (PIDs): %s\n' \
        "${remaining[*]}" >&2
    exit 1
fi

echo "robot processes stopped"
