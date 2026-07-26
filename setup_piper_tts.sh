#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/programas/myrobot_03"
source .venv/bin/activate

sudo apt update
sudo apt install -y espeak-ng alsa-utils ffmpeg

python -m pip install --upgrade pip
python -m pip install piper-tts

mkdir -p voices

echo "Installed piper-tts."
echo "Next, list available voices with:"
echo "  python -m piper.download_voices --help"
echo "Then download one voice, for example:"
echo "  python -m piper.download_voices --data-dir voices es_ES-davefx-medium"
