#!/bin/sh
set -eu

sudo apt update
sudo apt install -y python3-pyaudio portaudio19-dev flac espeak-ng

. .venv/bin/activate
python -m pip install SpeechRecognition
