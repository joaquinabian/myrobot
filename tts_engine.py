"""Small TTS abstraction for myrobot_04.

Engines:
- espeak: very light fallback, low quality.
- piper: local neural TTS using a Piper ONNX voice.

Notes:
- All messages and comments are ASCII-only.
- The Piper model remains loaded between calls.
- leading_silence_ms adds silence at the beginning of the generated WAV.
"""

from __future__ import annotations

import os
import shlex
import subprocess
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
    player_command: str = "aplay -q"
    leading_silence_ms: int = 350
    enabled: bool = True
    debug: bool = False


class TtsEngine:
    def __init__(self, config: TtsConfig | None = None):
        self.config = config or TtsConfig()
        self.piper_voice = None

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

    def _say_piper(self, text: str) -> bool:
        from piper import PiperVoice, SynthesisConfig

        if not self.config.piper_model:
            raise ValueError("piper_model is required when engine is piper")

        model_path = Path(self.config.piper_model).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"Piper model not found: {model_path}")

        if self.piper_voice is None:
            self.piper_voice = PiperVoice.load(model_path)

        synthesis_config = SynthesisConfig(
            speaker_id=self.config.piper_speaker,
            length_scale=self.config.piper_length_scale,
            noise_scale=self.config.piper_noise_scale,
            noise_w_scale=self.config.piper_noise_w,
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        try:
            with wave.open(wav_path, "wb") as wav_file:
                first_chunk = True
                for chunk in self.piper_voice.synthesize(
                    text, synthesis_config
                ):
                    if first_chunk:
                        wav_file.setframerate(chunk.sample_rate)
                        wav_file.setsampwidth(chunk.sample_width)
                        wav_file.setnchannels(chunk.sample_channels)
                        silence_frames = int(
                            chunk.sample_rate
                            * self.config.leading_silence_ms
                            / 1000
                        )
                        wav_file.writeframes(
                            b"\x00"
                            * silence_frames
                            * chunk.sample_channels
                            * chunk.sample_width
                        )
                        first_chunk = False
                    wav_file.writeframes(chunk.audio_int16_bytes)

            player = shlex.split(self.config.player_command)
            player += [wav_path]
            subprocess.run(player, check=True)
            return True
        except subprocess.CalledProcessError as exc:
            print(f"piper failed: {exc}")
            return False
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass


def say(text: str, config: TtsConfig | None = None) -> bool:
    return TtsEngine(config).say(text)
