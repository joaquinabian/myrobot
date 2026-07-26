# Robot server for myrobot_03.
#
# Main functional changes:
# - Continuous accept loops for video and speech clients.
# - Video tracking is decoupled from detection speed.
# - Only the latest video dx/dy is used; old messages do not accumulate.
# - Video servo updates have deadband, rate limit and max step limit.
# - Video tracking does not use smooth_move delays.
# - The server can run without LIRC.
# - The server can run without Crickit hardware using --dry-run.
# - Servo access is centralized in one shared controller and one lock.
# - The speech command "abajo" is clamped correctly.
# - TTS is routed through tts_engine_v3: espeak, piper or none.
# - Piper TTS runs in a background queue so tracking is not blocked.
#
# All comments and messages are ASCII-only to avoid VNC/editor encoding issues.

import argparse
import queue
import socket
import threading
import time

from tts_engine import TtsConfig, TtsEngine


HOST = "127.0.0.1"
PORT_VIDEO = 65000
PORT_SPEECH = 65001
LIRC_PATH = "/var/run/lirc/lircd"

MIN_ANGLE = 1
MAX_ANGLE = 179
START_ANGLE_1 = 90
START_ANGLE_2 = 90

SPEECH_STEP = 20
IR_STEP = 2


class DummyServo:
    def __init__(self, name):
        self.name = name
        self._angle = None

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = value
        print(f"dry-run {self.name}.angle = {value}")

    def set_pulse_width_range(self, min_pulse, max_pulse):
        print(f"dry-run {self.name}.set_pulse_width_range({min_pulse}, {max_pulse})")


class ServoController:
    def __init__(self, servo_1, servo_2, min_angle=MIN_ANGLE,
                 max_angle=MAX_ANGLE):
        self.s1 = servo_1
        self.s2 = servo_2
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.angle_1 = START_ANGLE_1
        self.angle_2 = START_ANGLE_2
        self.lock = threading.Lock()
        self.last_activity = time.time()
        self.set_angles(self.angle_1, self.angle_2, smooth=False)

    def clamp(self, value):
        return max(self.min_angle, min(self.max_angle, int(round(value))))

    def set_angles(self, angle_1=None, angle_2=None, smooth=True):
        with self.lock:
            target_1 = self.angle_1 if angle_1 is None else self.clamp(angle_1)
            target_2 = self.angle_2 if angle_2 is None else self.clamp(angle_2)

            old_1 = self.angle_1
            old_2 = self.angle_2
            self.angle_1 = target_1
            self.angle_2 = target_2
            self.last_activity = time.time()

            if smooth:
                self._smooth_move(old_1, old_2, target_1, target_2)
            else:
                self.s1.angle = target_1
                self.s2.angle = target_2

            print(f"servo angles: {self.angle_1}, {self.angle_2}")

    def move_by(self, delta_1=0, delta_2=0, smooth=True):
        if delta_1 == 0 and delta_2 == 0:
            return
        self.set_angles(self.angle_1 + delta_1, self.angle_2 + delta_2,
                        smooth=smooth)

    def _smooth_move(self, old_1, old_2, target_1, target_2):
        steps = max(abs(target_1 - old_1), abs(target_2 - old_2)) // 5
        steps = max(1, min(steps, 10))

        for n in range(1, steps + 1):
            a1 = round(old_1 + (target_1 - old_1) * n / steps)
            a2 = round(old_2 + (target_2 - old_2) * n / steps)
            self.s1.angle = a1
            self.s2.angle = a2
            time.sleep(0.05)


class LineReceiver:
    def __init__(self, conn, max_bytes=4096):
        self.conn = conn
        self.buffer = b""
        self.max_bytes = max_bytes

    def recv_line(self):
        while b"\n" not in self.buffer:
            data = self.conn.recv(128)
            if not data:
                return None
            self.buffer += data
            if len(self.buffer) > self.max_bytes:
                self.buffer = b""
                return None

        line, self.buffer = self.buffer.split(b"\n", 1)
        return line.decode("utf-8", errors="replace").strip()


class LatestVideoState:
    def __init__(self):
        self.lock = threading.Lock()
        self.seq = 0
        self.dx = 0
        self.dy = 0
        self.name = ""
        self.time = 0.0

    def update(self, dx, dy, name):
        with self.lock:
            self.seq += 1
            self.dx = dx
            self.dy = dy
            self.name = name
            self.time = time.time()

    def snapshot(self):
        with self.lock:
            return self.seq, self.dx, self.dy, self.name, self.time


class StopState:
    def __init__(self):
        self.event = threading.Event()

    def stop(self):
        self.event.set()

    def stopped(self):
        return self.event.is_set()


class SayHelper:
    def __init__(self, args):
        self.enabled = not args.no_say and args.tts_engine != "none"
        self.sync = args.tts_sync
        self.stop_event = threading.Event()
        self.queue = queue.Queue(maxsize=args.tts_queue_size)
        self.thread = None
        self.engine = None

        engine_name = "none" if not self.enabled else args.tts_engine
        config = TtsConfig(
            engine=engine_name,
            voice=args.tts_voice,
            speed=args.tts_speed,
            volume=args.tts_volume,
            piper_model=args.tts_model,
            piper_command=args.piper_command,
            player_command=args.player_command,
            leading_silence_ms=args.tts_leading_silence_ms,
            debug=args.tts_debug,
            enabled=self.enabled,
        )
        self.engine = TtsEngine(config)

        if self.enabled and not self.sync:
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()

        print(f"TTS engine: {engine_name}")

    def say(self, text, wait=0):
        print(f"say: {text}")

        if self.enabled and self.engine is not None:
            if self.sync:
                self._say_now(text)
            else:
                try:
                    self.queue.put_nowait(text)
                except queue.Full:
                    print("TTS queue full; dropping message")

        if wait:
            time.sleep(wait)

    def _say_now(self, text):
        try:
            self.engine.say(text)
        except Exception as exc:
            print(f"TTS error: {exc}")

    def _worker(self):
        print("TTS worker running")
        while not self.stop_event.is_set():
            try:
                text = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._say_now(text)
            self.queue.task_done()

    def stop(self):
        self.stop_event.set()


class VideoClientThread(threading.Thread):
    def __init__(self, conn, address, video_state, stop_state):
        super().__init__(daemon=True)
        self.conn = conn
        self.address = address
        self.video_state = video_state
        self.stop_state = stop_state
        self.reader = LineReceiver(conn)

    def parse_message(self, line):
        parts = line.split(",", 2)
        if len(parts) != 3:
            raise ValueError(f"bad video message: {line}")

        dx = int(float(parts[0]))
        dy = int(float(parts[1]))
        name = parts[2].strip()
        return dx, dy, name

    def run(self):
        print(f"video client connected: {self.address}")
        try:
            while not self.stop_state.stopped():
                line = self.reader.recv_line()
                if line is None:
                    break
                try:
                    dx, dy, name = self.parse_message(line)
                except ValueError as exc:
                    print(exc)
                    continue
                self.video_state.update(dx, dy, name)
        except OSError as exc:
            print(f"video client socket error: {exc}")
        finally:
            print("video client disconnected")
            try:
                self.conn.close()
            except OSError:
                pass


class VideoAcceptThread(threading.Thread):
    def __init__(self, server_sock, video_state, stop_state):
        super().__init__(daemon=True)
        self.server_sock = server_sock
        self.video_state = video_state
        self.stop_state = stop_state

    def run(self):
        print("video accept thread running")
        while not self.stop_state.stopped():
            try:
                conn, address = self.server_sock.accept()
            except OSError:
                break
            thread = VideoClientThread(conn, address, self.video_state,
                                       self.stop_state)
            thread.start()


class VideoControlThread(threading.Thread):
    def __init__(self, video_state, servos, say_helper, args, stop_state):
        super().__init__(daemon=True)
        self.video_state = video_state
        self.servos = servos
        self.say_helper = say_helper
        self.args = args
        self.stop_state = stop_state
        self.last_seq = 0
        self.last_move_time = 0.0
        self.last_print_time = 0.0
        self.names = {}

    def maybe_greet(self, name):
        if not name or name == "unknown":
            return

        now = time.time()
        last = self.names.get(name)
        if last is None or now - last > self.args.greet_interval:
            print(f"seen: {name}")
            self.say_helper.say(f"Hola {name}")
            self.names[name] = now

    def step_from_error(self, error, deadband, pixels_per_degree, max_step):
        abs_error = abs(error)
        if abs_error <= deadband:
            return 0

        effective_error = abs_error - deadband
        step = int(round(effective_error / pixels_per_degree))
        step = max(1, min(max_step, step))

        if error < 0:
            step = -step
        return step

    def run(self):
        print("video control thread running")
        interval = self.args.video_interval
        stale_timeout = self.args.video_stale_timeout

        while not self.stop_state.stopped():
            time.sleep(interval)

            seq, dx, dy, name, msg_time = self.video_state.snapshot()
            if seq == 0:
                continue
            if seq == self.last_seq:
                continue
            if time.time() - msg_time > stale_timeout:
                continue

            self.last_seq = seq
            self.maybe_greet(name)

            step_x = self.step_from_error(dx, self.args.video_deadband_x,
                                          self.args.video_pixels_per_degree_x,
                                          self.args.video_max_step)
            step_y = self.step_from_error(dy, self.args.video_deadband_y,
                                          self.args.video_pixels_per_degree_y,
                                          self.args.video_max_step)

            # Mapping used by the original project:
            # servo_1 is vertical. Positive dy means face is below center,
            # so the original direction was negative servo_1 movement.
            # servo_2 is horizontal. Positive dx used positive servo_2 movement.
            delta_1 = -step_y
            delta_2 = step_x

            if self.args.invert_video_y:
                delta_1 = -delta_1
            if self.args.invert_video_x:
                delta_2 = -delta_2

            now = time.time()
            if now - self.last_print_time > self.args.video_print_interval:
                print(f"video latest: dx={dx}, dy={dy}, name={name}, "
                      f"d1={delta_1}, d2={delta_2}")
                self.last_print_time = now

            self.servos.move_by(delta_1=delta_1, delta_2=delta_2,
                                smooth=False)


class SpeechClientThread(threading.Thread):
    def __init__(self, conn, address, servos, stop_state):
        super().__init__(daemon=True)
        self.conn = conn
        self.address = address
        self.servos = servos
        self.stop_state = stop_state

    def recv_message(self):
        data = self.conn.recv(64)
        if not data:
            return None
        return data.decode("utf-8", errors="replace").strip().lower()

    def run(self):
        print(f"speech client connected: {self.address}")
        try:
            while not self.stop_state.stopped():
                message = self.recv_message()
                if message is None:
                    break

                print(f"speech message: {message}")

                if message == "derecha":
                    self.servos.move_by(delta_2=SPEECH_STEP, smooth=False)
                elif message == "izquierda":
                    self.servos.move_by(delta_2=-SPEECH_STEP, smooth=False)
                elif message == "arriba":
                    self.servos.move_by(delta_1=SPEECH_STEP, smooth=False)
                elif message == "abajo":
                    self.servos.move_by(delta_1=-SPEECH_STEP, smooth=False)
                else:
                    print(f"unknown speech command: {message}")
        except OSError as exc:
            print(f"speech client socket error: {exc}")
        finally:
            print("speech client disconnected")
            try:
                self.conn.close()
            except OSError:
                pass


class SpeechAcceptThread(threading.Thread):
    def __init__(self, server_sock, servos, stop_state):
        super().__init__(daemon=True)
        self.server_sock = server_sock
        self.servos = servos
        self.stop_state = stop_state

    def run(self):
        print("speech accept thread running")
        while not self.stop_state.stopped():
            try:
                conn, address = self.server_sock.accept()
            except OSError:
                break
            thread = SpeechClientThread(conn, address, self.servos,
                                        self.stop_state)
            thread.start()


class LircThread(threading.Thread):
    def __init__(self, conn, servos, stop_state):
        super().__init__(daemon=True)
        self.conn = conn
        self.servos = servos
        self.stop_state = stop_state

    def recv_key(self):
        data = self.conn.recv(128)
        if not data:
            return None, None

        words = data.strip().split()
        if len(words) < 3:
            return None, None

        key = words[2].decode("utf-8", errors="replace")
        repeat = words[1].decode("utf-8", errors="replace")
        return key, repeat

    def run(self):
        print("LIRC thread running")
        while not self.stop_state.stopped():
            key, repeat = self.recv_key()
            if key is None:
                print("IR daemon disconnected or bad message")
                break

            print(f"IR key: {key}, repeat: {repeat}")

            if key == "KEY_2":
                self.servos.move_by(delta_1=IR_STEP, smooth=False)
            elif key == "KEY_8":
                self.servos.move_by(delta_1=-IR_STEP, smooth=False)
            elif key == "KEY_4":
                self.servos.move_by(delta_2=IR_STEP, smooth=False)
            elif key == "KEY_6":
                self.servos.move_by(delta_2=-IR_STEP, smooth=False)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default=HOST)
    parser.add_argument("--video-port", type=int, default=PORT_VIDEO)
    parser.add_argument("--speech-port", type=int, default=PORT_SPEECH)

    parser.add_argument("--dry-run", action="store_true",
                        help="run without Crickit hardware")
    parser.add_argument("--no-lirc", action="store_true",
                        help="do not connect to LIRC")
    parser.add_argument("--lirc-path", default=LIRC_PATH)
    parser.add_argument("--no-say", action="store_true",
                        help="disable speech output")
    parser.add_argument("--tts-engine", choices=["none", "espeak", "piper"],
                        default="espeak", help="speech output engine")
    parser.add_argument("--tts-model", default="voices/es_ES-davefx-medium.onnx",
                        help="Piper ONNX model path")
    parser.add_argument("--tts-voice", default="es",
                        help="espeak/espeak-ng voice")
    parser.add_argument("--tts-speed", type=int, default=145,
                        help="espeak speech speed")
    parser.add_argument("--tts-volume", type=int, default=100,
                        help="espeak speech volume")
    parser.add_argument("--tts-leading-silence-ms", type=int, default=350,
                        help="silence added before Piper audio")
    parser.add_argument("--piper-command", default="piper",
                        help="Piper command")
    parser.add_argument("--player-command", default="aplay",
                        help="audio player command")
    parser.add_argument("--tts-debug", action="store_true",
                        help="show Piper/ONNX runtime warnings")
    parser.add_argument("--tts-sync", action="store_true",
                        help="run TTS synchronously instead of background queue")
    parser.add_argument("--tts-queue-size", type=int, default=5,
                        help="max queued TTS messages")
    parser.add_argument("--no-bored", action="store_true",
                        help="disable periodic bored message")
    parser.add_argument("--bored-interval", type=float, default=120.0)

    parser.add_argument("--video-interval", type=float, default=0.25,
                        help="seconds between video servo updates")
    parser.add_argument("--video-stale-timeout", type=float, default=1.0,
                        help="ignore video data older than this many seconds")
    parser.add_argument("--video-deadband-x", type=int, default=35,
                        help="horizontal deadband in pixels")
    parser.add_argument("--video-deadband-y", type=int, default=35,
                        help="vertical deadband in pixels")
    parser.add_argument("--video-pixels-per-degree-x", type=float, default=45.0,
                        help="horizontal pixels per servo degree")
    parser.add_argument("--video-pixels-per-degree-y", type=float, default=45.0,
                        help="vertical pixels per servo degree")
    parser.add_argument("--video-max-step", type=int, default=2,
                        help="max servo degrees per video update")
    parser.add_argument("--video-print-interval", type=float, default=0.5,
                        help="minimum seconds between video debug prints")
    parser.add_argument("--invert-video-x", action="store_true",
                        help="invert horizontal video tracking")
    parser.add_argument("--invert-video-y", action="store_true",
                        help="invert vertical video tracking")
    parser.add_argument("--greet-interval", type=float, default=60.0,
                        help="minimum seconds between greetings per name")

    return parser.parse_args()


def init_servos(dry_run=False):
    if dry_run:
        s1 = DummyServo("servo_1")
        s2 = DummyServo("servo_2")
        return s1, s2

    from adafruit_crickit import crickit as cr

    s1 = cr.servo_1
    s2 = cr.servo_2

    # HS-645MG pulse width range used in the original project.
    s1.set_pulse_width_range(min_pulse=900, max_pulse=2100)
    s2.set_pulse_width_range(min_pulse=900, max_pulse=2100)

    return s1, s2


def make_tcp_server(host, port, label):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen()
    print(f"listening for {label} clients on {host}:{port}")
    return sock


def connect_lirc(path):
    ir_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    ir_sock.connect(path)
    ir_sock.settimeout(None)
    print("IR daemon connected")
    return ir_sock


def main():
    args = parse_args()
    stop_state = StopState()
    say_helper = SayHelper(args)

    s1, s2 = init_servos(dry_run=args.dry_run)
    servos = ServoController(s1, s2)

    video_state = LatestVideoState()

    video_server = make_tcp_server(args.host, args.video_port, "video")
    speech_server = make_tcp_server(args.host, args.speech_port, "speech")

    threads = []

    video_accept = VideoAcceptThread(video_server, video_state, stop_state)
    video_accept.start()
    threads.append(video_accept)

    video_control = VideoControlThread(video_state, servos, say_helper, args,
                                       stop_state)
    video_control.start()
    threads.append(video_control)

    speech_accept = SpeechAcceptThread(speech_server, servos, stop_state)
    speech_accept.start()
    threads.append(speech_accept)

    ir_conn = None
    if not args.no_lirc:
        try:
            ir_conn = connect_lirc(args.lirc_path)
            ir_thread = LircThread(ir_conn, servos, stop_state)
            ir_thread.start()
            threads.append(ir_thread)
        except OSError as exc:
            print(f"No LIRC available: {exc}")

    print("robot_server ready")

    try:
        while not stop_state.stopped():
            if not args.no_bored:
                idle_time = time.time() - servos.last_activity
                if idle_time > args.bored_interval:
                    say_helper.say("Me aburro")
                    servos.last_activity = time.time()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_state.stop()
        say_helper.stop()
        for sock in (video_server, speech_server, ir_conn):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        print("Done")


if __name__ == "__main__":
    main()
