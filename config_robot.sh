# Runtime configuration for myrobot.
# This file is sourced by run_robot.sh.
# Values can still be overridden from the command line:
#   VIDEO_MAX_STEP=4 ./run_robot.sh

# Project paths
PROJECT_DIR="/home/quim/programas/myrobot"
VENV_DIR="$PROJECT_DIR/.venv"

# Final program names
SERVER_SCRIPT="robot_server.py"
VIDEO_SCRIPT="robot_video_client.py"
SPEECH_SCRIPT="robot_speech_client.py"
TTS_ENGINE_SCRIPT="tts_engine.py"

# Network ports
HOST="127.0.0.1"
VIDEO_PORT="65000"
SPEECH_PORT="65001"

# Hardware / feature switches
NO_LIRC="1"
NO_BORED="1"
NO_VIDEO="0"
NO_SPEECH="0"
NO_TTS="0"
DRY_RUN="0"

# TTS
TTS_ENGINE="piper"
TTS_MODEL="voices/es_ES-davefx-medium.onnx"
TTS_LEADING_SILENCE_MS="350"
TTS_DEBUG="0"

# Video tracking, validated on RPi5 + C920 + Crickit
VIDEO_INTERVAL="0.08"
VIDEO_DEADBAND_X="35"
VIDEO_DEADBAND_Y="35"
VIDEO_PIXELS_PER_DEGREE_X="45"
VIDEO_PIXELS_PER_DEGREE_Y="45"
VIDEO_MAX_STEP="2"
INVERT_VIDEO_X="0"
INVERT_VIDEO_Y="0"

# Video client
VIDEO_EXTRA_ARGS="--send-unknown"

# Speech client
MIC_INDEX=""
SPEECH_EXTRA_ARGS="--print-raw --pause-after-command 0.2 --min-command-interval 0.35"

# Log directory
LOG_DIR="$PROJECT_DIR/logs"
