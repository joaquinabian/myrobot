#!/usr/bin/env bash
# Stop myrobot_03 processes. ASCII only.

set -euo pipefail

pkill -f robot_server_tracking_tts_v3.py 2>/dev/null || true
pkill -f robot_server_tracking_v2.py 2>/dev/null || true
pkill -f robot_server.py 2>/dev/null || true
pkill -f robot_recognize_video_client.py 2>/dev/null || true
pkill -f robot_speech_client_v2.py 2>/dev/null || true

echo "robot processes stopped"
