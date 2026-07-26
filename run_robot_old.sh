#!/usr/bin/env bash
# Robot launcher for myrobot_03.
# Starts server, video client, and speech client in a controlled order.
# ASCII only.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/programas/myrobot_03}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"

SERVER_SCRIPT="${SERVER_SCRIPT:-robot_server_tracking_tts_v3.py}"
VIDEO_SCRIPT="${VIDEO_SCRIPT:-robot_recognize_video_client.py}"
SPEECH_SCRIPT="${SPEECH_SCRIPT:-robot_speech_client_v2.py}"

HOST="${HOST:-127.0.0.1}"
VIDEO_PORT="${VIDEO_PORT:-65000}"
SPEECH_PORT="${SPEECH_PORT:-65001}"

MIC_INDEX="${MIC_INDEX:-1}"
PIPER_MODEL="${PIPER_MODEL:-voices/es_ES-davefx-medium.onnx}"
TTS_LEADING_SILENCE_MS="${TTS_LEADING_SILENCE_MS:-350}"

VIDEO_INTERVAL="${VIDEO_INTERVAL:-0.18}"
VIDEO_DEADBAND_X="${VIDEO_DEADBAND_X:-30}"
VIDEO_DEADBAND_Y="${VIDEO_DEADBAND_Y:-30}"
VIDEO_PIXELS_PER_DEGREE_X="${VIDEO_PIXELS_PER_DEGREE_X:-35}"
VIDEO_PIXELS_PER_DEGREE_Y="${VIDEO_PIXELS_PER_DEGREE_Y:-35}"
VIDEO_MAX_STEP="${VIDEO_MAX_STEP:-3}"

# Set to 1 to disable one component.
NO_VIDEO="${NO_VIDEO:-0}"
NO_SPEECH="${NO_SPEECH:-0}"
NO_TTS="${NO_TTS:-0}"
NO_LIRC="${NO_LIRC:-1}"
NO_BORED="${NO_BORED:-1}"

# Add extra args without editing this file.
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"
VIDEO_EXTRA_ARGS="${VIDEO_EXTRA_ARGS:---send-unknown}"
SPEECH_EXTRA_ARGS="${SPEECH_EXTRA_ARGS:---print-raw}"

PIDS=()

log() {
    printf '[robot] %s\n' "$*"
}

fail() {
    printf '[robot] ERROR: %s\n' "$*" >&2
    exit 1
}

cleanup() {
    log "stopping components"
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    sleep 0.5
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
}

wait_for_port() {
    local port="$1"
    local name="$2"
    local tries=40

    for _ in $(seq 1 "$tries"); do
        if ss -ltn | grep -q "${HOST}:${port}"; then
            log "$name port is ready: ${HOST}:${port}"
            return 0
        fi
        sleep 0.25
    done

    fail "$name port did not become ready: ${HOST}:${port}"
}

require_file() {
    local path="$1"
    [[ -e "$path" ]] || fail "missing file: $path"
}

main() {
    cd "$PROJECT_DIR" || fail "cannot cd to $PROJECT_DIR"

    require_file "$VENV_DIR/bin/activate"
    require_file "$SERVER_SCRIPT"
    require_file "$VIDEO_SCRIPT"
    require_file "$SPEECH_SCRIPT"

    if [[ "$NO_TTS" != "1" ]]; then
        require_file "$PIPER_MODEL"
    fi

    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"

    trap cleanup EXIT INT TERM

    log "project: $PROJECT_DIR"
    log "python: $(python --version 2>&1)"

    local server_args=()
    server_args+=("--host" "$HOST")
    server_args+=("--video-port" "$VIDEO_PORT")
    server_args+=("--speech-port" "$SPEECH_PORT")
    server_args+=("--video-interval" "$VIDEO_INTERVAL")
    server_args+=("--video-deadband-x" "$VIDEO_DEADBAND_X")
    server_args+=("--video-deadband-y" "$VIDEO_DEADBAND_Y")
    server_args+=("--video-pixels-per-degree-x" "$VIDEO_PIXELS_PER_DEGREE_X")
    server_args+=("--video-pixels-per-degree-y" "$VIDEO_PIXELS_PER_DEGREE_Y")
    server_args+=("--video-max-step" "$VIDEO_MAX_STEP")

    if [[ "$NO_LIRC" == "1" ]]; then
        server_args+=("--no-lirc")
    fi

    if [[ "$NO_BORED" == "1" ]]; then
        server_args+=("--no-bored")
    fi

    if [[ "$NO_TTS" == "1" ]]; then
        server_args+=("--no-say")
    else
        server_args+=("--tts-engine" "piper")
        server_args+=("--tts-model" "$PIPER_MODEL")
        server_args+=("--tts-leading-silence-ms" "$TTS_LEADING_SILENCE_MS")
    fi

    if [[ -n "$SERVER_EXTRA_ARGS" ]]; then
        # Intentionally use word splitting for optional user args.
        # shellcheck disable=SC2206
        extra=( $SERVER_EXTRA_ARGS )
        server_args+=("${extra[@]}")
    fi

    log "starting server"
    python "$SERVER_SCRIPT" "${server_args[@]}" &
    PIDS+=("$!")

    wait_for_port "$VIDEO_PORT" "video"
    wait_for_port "$SPEECH_PORT" "speech"

    if [[ "$NO_VIDEO" != "1" ]]; then
        local video_args=()
        video_args+=("--host" "$HOST")
        video_args+=("--port" "$VIDEO_PORT")

        if [[ -n "$VIDEO_EXTRA_ARGS" ]]; then
            # shellcheck disable=SC2206
            extra=( $VIDEO_EXTRA_ARGS )
            video_args+=("${extra[@]}")
        fi

        log "starting video client"
        python "$VIDEO_SCRIPT" "${video_args[@]}" &
        PIDS+=("$!")
    else
        log "video client disabled"
    fi

    if [[ "$NO_SPEECH" != "1" ]]; then
        local speech_args=()
        speech_args+=("--host" "$HOST")
        speech_args+=("--port" "$SPEECH_PORT")
        speech_args+=("--mic-index" "$MIC_INDEX")

        if [[ -n "$SPEECH_EXTRA_ARGS" ]]; then
            # shellcheck disable=SC2206
            extra=( $SPEECH_EXTRA_ARGS )
            speech_args+=("${extra[@]}")
        fi

        log "starting speech client"
        python "$SPEECH_SCRIPT" "${speech_args[@]}" 2>/tmp/robot_speech_audio_warnings.txt &
        PIDS+=("$!")
    else
        log "speech client disabled"
    fi

    log "robot is running"
    log "press Ctrl+C to stop"

    while true; do
        for pid in "${PIDS[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                fail "component stopped unexpectedly, pid=$pid"
            fi
        done
        sleep 1
    done
}

main "$@"
