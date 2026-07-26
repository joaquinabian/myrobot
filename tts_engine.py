"""Small TTS abstraction for myrobot_04.

Engines:
- espeak: very light fallback, low quality.
- piper: local neural TTS using a Piper ONNX voice.

Notes:
- All messages and comments are ASCII-only.
- Piper receives text through stdin.
- sys.executable is used for "python -m piper" fallback, so the current venv is used.
- leading_silence_ms adds silence at the beginning of the generated WAV.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TtsConfig:
    engine: str = "espeak"
    voice: str = "es"
    speed: int = 145
    volume: int = 100
    piper_model: str = ""
    piper_speaker: int | None = None
    piper_length_scale: float | None = None
    piper_noise_scale: float | None = None
    piper_noise_w: float | None = None
    piper_command: str = "piper"
    player_command: str = "aplay -q"
    leading_silence_ms: int = 350
    enabled: bool = True
    debug: bool = False


class TtsEngine:
    def __init__(self, config: TtsConfig | None = None):
        self.config = config or TtsConfig()

    def say(self, text: str) -> bool:
        if not self.config.enabled:
            print(f"say disabled: {text}")
            return True

        engine = self.config.engine.lower()

        if engine == "none":
            print(f"say none: {text}")
            return True
        if engine == "espeak":
            return self._say_espeak(text)
        if engine == "piper":
            return self._say_piper(text)

        raise ValueError(f"Unknown TTS engine: {self.config.engine}")

    def _say_espeak(self, text: str) -> bool:
        command = [
            "espeak-ng",
            "-v", self.config.voice,
            "-s", str(self.config.speed),
            "-a", str(self.config.volume),
            text,
        ]

        try:
            subprocess.run(command, check=True)
            return True
        except FileNotFoundError:
            fallback = [
                "espeak",
                "-v", self.config.voice,
                "-s", str(self.config.speed),
                "-a", str(self.config.volume),
                text,
            ]
            subprocess.run(fallback, check=True)
            return True
        except subprocess.CalledProcessError as exc:
            print(f"espeak failed: {exc}")
            return False

    def _make_piper_command(self, executable: list[str], wav_path: str) -> list[str]:
        model_path = Path(self.config.piper_model).expanduser()

        command = list(executable)
        command += ["--model", str(model_path), "--output_file", wav_path]

        if self.config.piper_speaker is not None:
            command += ["--speaker", str(self.config.piper_speaker)]
        if self.config.piper_length_scale is not None:
            command += ["--length-scale", str(self.config.piper_length_scale)]
        if self.config.piper_noise_scale is not None:
            command += ["--noise-scale", str(self.config.piper_noise_scale)]
        if self.config.piper_noise_w is not None:
            command += ["--noise-w", str(self.config.piper_noise_w)]

        return command

    def _run_piper_command(self, command: list[str], text: str) -> subprocess.CompletedProcess:
        input_text = text.strip() + "\n"

        if self.config.debug:
            print("piper command:", " ".join(shlex.quote(x) for x in command))
            return subprocess.run(command, input=input_text, text=True, check=True)

        result = subprocess.run(
            command,
            input=input_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            print("piper command failed")
            if result.stdout:
                print(result.stdout.strip())
            if result.stderr:
                print(result.stderr.strip())
            result.check_returncode()

        return result

    def _say_piper(self, text: str) -> bool:
        if not self.config.piper_model:
            raise ValueError("piper_model is required when engine is piper")

        model_path = Path(self.config.piper_model).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"Piper model not found: {model_path}")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        padded_path = ""

        try:
            piper_executable = shlex.split(self.config.piper_command)
            command = self._make_piper_command(piper_executable, wav_path)

            try:
                self._run_piper_command(command, text)
            except FileNotFoundError:
                command = self._make_piper_command([sys.executable, "-m", "piper"], wav_path)
                self._run_piper_command(command, text)

            if self.config.leading_silence_ms > 0:
                padded_path = self._add_leading_silence(
                    wav_path, self.config.leading_silence_ms
                )
                play_path = padded_path
            else:
                play_path = wav_path

            player = shlex.split(self.config.player_command)
            player += [play_path]
            subprocess.run(player, check=True)
            return True
        except subprocess.CalledProcessError as exc:
            print(f"piper failed: {exc}")
            return False
        finally:
            for path in (wav_path, padded_path):
                if path:
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    def _add_leading_silence(self, wav_path: str, silence_ms: int) -> str:
        with wave.open(wav_path, "rb") as src:
            params = src.getparams()
            frames = src.readframes(src.getnframes())

        sample_rate = params.framerate
        channels = params.nchannels
        sample_width = params.sampwidth
        silence_frames = int(sample_rate * silence_ms / 1000.0)
        silence_bytes = b"\x00" * silence_frames * channels * sample_width

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            padded_path = tmp.name

        with wave.open(padded_path, "wb") as dst:
            dst.setparams(params)
            dst.writeframes(silence_bytes)
            dst.writeframes(frames)

        return padded_path


def say(text: str, config: TtsConfig | None = None) -> bool:
    return TtsEngine(config).say(text)
