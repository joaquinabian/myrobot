#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/programas/myrobot}"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_DIR/config_robot.sh}"

if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "[robot] ERROR: project folder not found: $PROJECT_DIR" >&2
    exit 1
fi

cd "$PROJECT_DIR"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[robot] ERROR: config file not found: $CONFIG_FILE" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$CONFIG_FILE"

VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON="${PYTHON:-$VENV_DIR/bin/python}"

SERVER_SCRIPT="${SERVER_SCRIPT:-robot_server.py}"
VIDEO_SCRIPT="${VIDEO_SCRIPT:-robot_video_client.py}"
SPEECH_SCRIPT="${SPEECH_SCRIPT:-robot_speech_client.py}"

HOST="${HOST:-127.0.0.1}"
VIDEO_PORT="${VIDEO_PORT:-65000}"
SPEECH_PORT="${SPEECH_PORT:-65001}"

NO_LIRC="${NO_LIRC:-1}"
NO_BORED="${NO_BORED:-1}"
NO_TTS="${NO_TTS:-0}"
NO_VIDEO="${NO_VIDEO:-0}"
NO_SPEECH="${NO_SPEECH:-0}"
DRY_RUN="${DRY_RUN:-0}"
DEBUG="${DEBUG:-0}"

MIC_INDEX="${MIC_INDEX:-1}"

TTS_ENGINE="${TTS_ENGINE:-piper}"
TTS_MODEL="${TTS_MODEL:-voices/es_ES-davefx-medium.onnx}"
TTS_LEADING_SILENCE_MS="${TTS_LEADING_SILENCE_MS:-350}"

VIDEO_INTERVAL="${VIDEO_INTERVAL:-0.18}"
VIDEO_DEADBAND_X="${VIDEO_DEADBAND_X:-30}"
VIDEO_DEADBAND_Y="${VIDEO_DEADBAND_Y:-30}"
VIDEO_PIXELS_PER_DEGREE_X="${VIDEO_PIXELS_PER_DEGREE_X:-35}"
VIDEO_PIXELS_PER_DEGREE_Y="${VIDEO_PIXELS_PER_DEGREE_Y:-35}"
VIDEO_MAX_STEP="${VIDEO_MAX_STEP:-3}"

VIDEO_EXTRA_ARGS="${VIDEO_EXTRA_ARGS:---send-unknown}"
SPEECH_EXTRA_ARGS="${SPEECH_EXTRA_ARGS:---print-raw --pause-after-command 0.2 --min-command-interval 0.35}"

LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$LOG_ROOT/$RUN_ID"
LATEST_LINK="$LOG_ROOT/latest"

mkdir -p "$LOG_DIR"
ln -sfn "$LOG_DIR" "$LATEST_LINK"

LAUNCHER_LOG="$LOG_DIR/launcher.log"
SERVER_LOG="$LOG_DIR/server.log"
VIDEO_LOG="$LOG_DIR/video.log"
SPEECH_LOG="$LOG_DIR/speech.log"

PIDS=()

log() {
    local msg="$*"
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$msg" | tee -a "$LAUNCHER_LOG"
}

die() {
    log "ERROR: $*"
    stop_all
    exit 1
}

stop_all() {
    if [[ "${#PIDS[@]}" -gt 0 ]]; then
        log "stopping components"
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
            fi
        done

        sleep 1

        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
    fi
}

on_exit() {
    stop_all
}
trap on_exit EXIT INT TERM

check_file() {
    local f="$1"
    if [[ ! -f "$f" ]]; then
        die "missing file: $f"
    fi
}

check_executable() {
    local f="$1"
    if [[ ! -x "$f" ]]; then
        die "not executable: $f"
    fi
}

wait_for_port() {
    local host="$1"
    local port="$2"
    local name="$3"
    local timeout_s="${4:-15}"
    local start
    start="$(date +%s)"

    while true; do
        if ss -ltn | awk '{print $4}' | grep -qE "(${host}|127\.0\.0\.1|localhost):${port}$"; then
            log "$name port ready: $host:$port"
            return 0
        fi

        if (( "$(date +%s)" - start >= timeout_s )); then
            return 1
        fi

        sleep 0.25
    done
}

start_component() {
    local name="$1"
    local logfile="$2"
    shift 2

    log "starting $name"
    if [[ "$DEBUG" == "1" ]]; then
        log "$name command: $*"
    fi

    # stdbuf keeps logs line-buffered. stderr is merged into the component log.
    stdbuf -oL -eL "$@" >>"$logfile" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    log "$name pid=$pid log=$logfile"
}

check_component_alive() {
    local pid="$1"
    local name="$2"

    if ! kill -0 "$pid" 2>/dev/null; then
        return 1
    fi

    return 0
}

monitor_components() {
    log "robot running"
    log "logs: $LOG_DIR"
    log "press Ctrl+C to stop"
    echo
    echo "[robot] clean console mode"
    echo "[robot] detailed logs: $LOG_DIR"
    echo "[robot] show live logs with:"
    echo "        ./show_robot_logs.sh"
    echo

    while true; do
        for pid in "${PIDS[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                log "ERROR: component stopped unexpectedly, pid=$pid"
                log "last launcher lines:"
                tail -20 "$LAUNCHER_LOG" || true
                log "last server lines:"
                tail -30 "$SERVER_LOG" || true
                log "last video lines:"
                tail -30 "$VIDEO_LOG" || true
                log "last speech lines:"
                tail -30 "$SPEECH_LOG" || true
                exit 1
            fi
        done
        sleep 1
    done
}

if [[ ! -x "$PYTHON" ]]; then
    die "python not found or not executable: $PYTHON"
fi

check_file "$SERVER_SCRIPT"
check_file "$VIDEO_SCRIPT"
check_file "$SPEECH_SCRIPT"

log "project: $PROJECT_DIR"
log "run id: $RUN_ID"
log "python: $PYTHON"

server_args=(
    "$SERVER_SCRIPT"
    "--host" "$HOST"
    "--video-port" "$VIDEO_PORT"
    "--speech-port" "$SPEECH_PORT"
    "--video-interval" "$VIDEO_INTERVAL"
    "--video-deadband-x" "$VIDEO_DEADBAND_X"
    "--video-deadband-y" "$VIDEO_DEADBAND_Y"
    "--video-pixels-per-degree-x" "$VIDEO_PIXELS_PER_DEGREE_X"
    "--video-pixels-per-degree-y" "$VIDEO_PIXELS_PER_DEGREE_Y"
    "--video-max-step" "$VIDEO_MAX_STEP"
)

if [[ "$NO_LIRC" == "1" ]]; then
    server_args+=("--no-lirc")
fi

if [[ "$NO_BORED" == "1" ]]; then
    server_args+=("--no-bored")
fi

if [[ "$NO_TTS" == "1" ]]; then
    server_args+=("--no-say")
else
    server_args+=(
        "--tts-engine" "$TTS_ENGINE"
        "--tts-model" "$TTS_MODEL"
        "--tts-leading-silence-ms" "$TTS_LEADING_SILENCE_MS"
    )
fi

if [[ "$DRY_RUN" == "1" ]]; then
    server_args+=("--dry-run")
fi

start_component "server" "$SERVER_LOG" "$PYTHON" "${server_args[@]}"

if ! wait_for_port "$HOST" "$VIDEO_PORT" "video" 20; then
    die "video port did not become ready"
fi

if ! wait_for_port "$HOST" "$SPEECH_PORT" "speech" 20; then
    die "speech port did not become ready"
fi

if [[ "$NO_VIDEO" != "1" ]]; then
    # shellcheck disable=SC2206
    video_extra_array=($VIDEO_EXTRA_ARGS)
    video_args=(
        "$VIDEO_SCRIPT"
        "--host" "$HOST"
        "--port" "$VIDEO_PORT"
    )
    video_args+=("${video_extra_array[@]}")
    start_component "video" "$VIDEO_LOG" "$PYTHON" "${video_args[@]}"
else
    log "video disabled"
fi

if [[ "$NO_SPEECH" != "1" ]]; then
    # shellcheck disable=SC2206
    speech_extra_array=($SPEECH_EXTRA_ARGS)
    speech_args=(
        "$SPEECH_SCRIPT"
        "--host" "$HOST"
        "--port" "$SPEECH_PORT"
        "--mic-index" "$MIC_INDEX"
    )
    speech_args+=("${speech_extra_array[@]}")
    start_component "speech" "$SPEECH_LOG" "$PYTHON" "${speech_args[@]}"
else
    log "speech disabled"
fi

monitor_components
