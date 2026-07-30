# Speech client for myrobot_03.
#
# Version v3.
# Main difference from v2:
# - The microphone stream is opened once and kept open.
# - This avoids repeated open/close cycles in PyAudio/PortAudio.
# - It is intended to reduce native segmentation faults on Raspberry/Linux.
#
# Other features:
# - Lists microphone devices with --list-mics.
# - Uses configurable --mic-index.
# - Reconnects to the server for later commands.
# - Sends commands terminated with "\n".
# - Normalizes Spanish phrases into simple robot commands.
# - Speaks recognition and calibration messages locally.
#
# ASCII only.

import argparse
import re
import select
import socket
import time
import unicodedata

import speech_recognition as sr

from tts_engine import TtsConfig, TtsEngine


HOST = "127.0.0.1"
PORT = 65001
DEFAULT_TTS_MODEL = "voices/es_ES-davefx-medium.onnx"
VALID_COMMANDS = {"arriba", "abajo", "izquierda", "derecha"}


COMMAND_PATTERNS = [
    ("derecha", [
        "derecha", "a la derecha", "hacia la derecha",
        "mueve derecha", "mover derecha", "gira derecha",
    ]),
    ("izquierda", [
        "izquierda", "a la izquierda", "hacia la izquierda",
        "mueve izquierda", "mover izquierda", "gira izquierda",
    ]),
    ("arriba", [
        "arriba", "hacia arriba", "sube", "subir",
        "mueve arriba", "mover arriba",
    ]),
    ("abajo", [
        "abajo", "hacia abajo", "baja", "bajar",
        "mueve abajo", "mover abajo",
    ]),
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default=HOST,
                        help="robot server host")
    parser.add_argument("--port", type=int, default=PORT,
                        help="robot server speech port")
    parser.add_argument("--no-server", action="store_true",
                        help="listen and recognize, but do not connect")
    parser.add_argument("--require-server", action="store_true",
                        help="exit if the server cannot be reached")
    parser.add_argument("--retry-interval", type=float, default=2.0,
                        help="seconds between server reconnect attempts")

    parser.add_argument("--list-mics", action="store_true",
                        help="list microphone device names and exit")
    parser.add_argument("--mic-index", type=int, default=None,
                        help="microphone device index")
    parser.add_argument("--language", default="es-ES",
                        help="Google recognition language")
    parser.add_argument("--ambient-duration", type=float, default=2.0,
                        help="seconds used to calibrate ambient noise")
    parser.add_argument("--energy-offset", type=float, default=800.0,
                        help="extra value added after ambient calibration")
    parser.add_argument("--dynamic-energy", action="store_true",
                        help="enable dynamic energy threshold")
    parser.add_argument("--timeout", type=float, default=2.0,
                        help="seconds waiting for speech to start")
    parser.add_argument("--phrase-time-limit", type=float, default=4.0,
                        help="max seconds per phrase")
    parser.add_argument("--pause-after-command", type=float, default=0.5,
                        help="seconds to pause after a valid command")
    parser.add_argument("--pause-after-error", type=float, default=0.2,
                        help="seconds to pause after not understood/request errors")
    parser.add_argument("--min-command-interval", type=float, default=0.8,
                        help="minimum seconds between sent commands")

    parser.set_defaults(echo=True)
    parser.add_argument("--echo", dest="echo", action="store_true",
                        help="enable local speech (default)")
    parser.add_argument("--no-echo", dest="echo", action="store_false",
                        help="disable local speech")
    parser.add_argument("--tts-engine", choices=["none", "espeak", "piper"],
                        default="piper", help="local speech engine")
    parser.add_argument("--tts-model", default=DEFAULT_TTS_MODEL,
                        help="Piper ONNX model path")
    parser.add_argument("--tts-voice", default="es",
                        help="espeak/espeak-ng voice")
    parser.add_argument("--tts-speed", type=int, default=145,
                        help="espeak speech speed")
    parser.add_argument("--tts-volume", type=int, default=100,
                        help="espeak speech volume")
    parser.add_argument("--tts-leading-silence-ms", type=int, default=350,
                        help="silence added before Piper audio")
    parser.add_argument("--player-command", default="aplay",
                        help="audio player command")
    parser.add_argument("--print-raw", action="store_true",
                        help="print every recognized raw phrase")

    return parser.parse_args()


def list_microphones():
    names = sr.Microphone.list_microphone_names()
    for index, name in enumerate(names):
        print(f"{index}: {name}")


def strip_accents(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text


def normalize_text(text):
    text = strip_accents(text).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_command(text):
    normalized = normalize_text(text)

    if normalized in VALID_COMMANDS:
        return normalized

    padded = f" {normalized} "
    for command, patterns in COMMAND_PATTERNS:
        for pattern in patterns:
            pattern = normalize_text(pattern)
            if f" {pattern} " in padded:
                return command

    words = set(normalized.split())
    matches = sorted(words.intersection(VALID_COMMANDS))
    if len(matches) == 1:
        return matches[0]

    return ""


class ServerConnection:
    def __init__(self, host, port, retry_interval=2.0, no_server=False,
                 require_server=False):
        self.host = host
        self.port = port
        self.retry_interval = retry_interval
        self.no_server = no_server
        self.require_server = require_server
        self.sock = None
        self.last_try = 0.0
        self.buffer = b""

    def connect(self, force=False):
        if self.no_server:
            return False

        now = time.time()
        if not force and now - self.last_try < self.retry_interval:
            return self.sock is not None

        self.last_try = now
        self.close()

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.host, self.port))
            self.sock = sock
            print("speech client connected")
            return True
        except OSError as exc:
            print(f"speech server not available: {exc}")
            if self.require_server:
                raise
            return False

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
            self.buffer = b""

    def send(self, command):
        if self.no_server:
            print(f"no-server command: {command}")
            return False

        if self.sock is None:
            print(f"command not sent: {command}")
            return False

        message = f"{command}\n".encode("utf-8")
        try:
            self.sock.sendall(message)
            print(f"sent command: {command}")
            return True
        except OSError as exc:
            print(f"speech server disconnected: {exc}")
            self.close()
            return False

    def receive_speech(self):
        if self.sock is None:
            return []

        readable, _, _ = select.select([self.sock], [], [], 0)
        if readable:
            try:
                data = self.sock.recv(4096)
            except OSError as exc:
                print(f"speech server disconnected: {exc}")
                self.close()
                return []
            if not data:
                self.close()
                return []
            self.buffer += data

        messages = []
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            text = line.decode("utf-8", errors="replace")
            if text.startswith("say\t"):
                messages.append(text[4:])
        return messages


def say_local(speaker, text, source=None):
    input_stream = None
    if source is not None and source.stream is not None:
        input_stream = source.stream.pyaudio_stream
        input_stream.stop_stream()

    try:
        speaker.say(text)
    finally:
        if input_stream is not None:
            input_stream.start_stream()


def play_server_speech(connection, speaker, source):
    for text in connection.receive_speech():
        print(f"server says: {text}")
        say_local(speaker, text, source)


def make_microphone(mic_index):
    if mic_index is None:
        print("Using default microphone")
        return sr.Microphone()

    print(f"Using microphone index {mic_index}")
    return sr.Microphone(device_index=mic_index)


def calibrate_source(recognizer, source, speaker, args):
    print("Calibrating ambient noise")
    print("Please keep silent")
    say_local(speaker, "Voy a calcular el ruido de fondo. Silencio por favor",
              source)

    recognizer.adjust_for_ambient_noise(source, duration=args.ambient_duration)
    recognizer.energy_threshold += args.energy_offset
    recognizer.dynamic_energy_threshold = args.dynamic_energy

    print(f"energy_threshold: {recognizer.energy_threshold}")
    print(f"dynamic_energy_threshold: {recognizer.dynamic_energy_threshold}")
    say_local(speaker, "Ruido de fondo calculado", source)


def listen_loop(recognizer, source, connection, speaker, args):
    last_command_time = 0.0

    while True:
        play_server_speech(connection, speaker, source)
        try:
            audio = recognizer.listen(source, timeout=args.timeout,
                                      phrase_time_limit=args.phrase_time_limit)
            text = recognizer.recognize_google(audio, language=args.language)
            normalized = normalize_text(text)
            command = normalize_command(text)

            if args.print_raw or command:
                print(f"heard: {text}")
                print(f"normalized: {normalized}")

            say_local(speaker, f"Has dicho {text}", source)

            if not command:
                print("no valid command")
                continue

            now = time.time()
            if now - last_command_time < args.min_command_interval:
                print(f"command ignored by interval: {command}")
                continue

            last_command_time = now
            print(f"command: {command}")
            connection.send(command)

            if args.pause_after_command > 0:
                time.sleep(args.pause_after_command)

        except sr.WaitTimeoutError:
            print("silent")
        except sr.UnknownValueError:
            print("speech not understood")
            say_local(speaker, "¿Qué?", source)
            if args.pause_after_error > 0:
                time.sleep(args.pause_after_error)
        except sr.RequestError as exc:
            print(f"Google speech request error: {exc}")
            time.sleep(2)

        if connection.sock is None and not args.no_server:
            connection.connect()


def main():
    args = parse_args()

    if args.list_mics:
        list_microphones()
        return

    connection = ServerConnection(args.host, args.port,
                                  retry_interval=args.retry_interval,
                                  no_server=args.no_server,
                                  require_server=args.require_server)
    connection.connect(force=True)

    recognizer = sr.Recognizer()
    microphone = make_microphone(args.mic_index)
    speaker = TtsEngine(TtsConfig(
        engine=args.tts_engine,
        voice=args.tts_voice,
        speed=args.tts_speed,
        volume=args.tts_volume,
        piper_model=args.tts_model,
        player_command=args.player_command,
        leading_silence_ms=args.tts_leading_silence_ms,
        enabled=args.echo,
    ))

    print("Opening microphone stream")

    try:
        with microphone as source:
            calibrate_source(recognizer, source, speaker, args)
            print("speech client ready")
            listen_loop(recognizer, source, connection, speaker, args)

    except KeyboardInterrupt:
        print("\nSpeech client stopped")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
