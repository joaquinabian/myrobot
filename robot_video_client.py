# https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/
#
# Face recognition video client for myrobot_03.
#
# Functional changes compared with the original version:
# - Can run without the central server.
# - Uses explicit cv2.VideoCapture with V4L2, MJPG, resolution and FPS.
# - Computes dx/dy using the real frame center.
# - Adds a recognition threshold: below threshold -> unknown.
# - Sends line-delimited TCP messages: "dx,dy,name\n".
# - Useful for testing video recognition before testing servos, speech or IR.

import argparse
import atexit
import pickle
import select
import signal
import socket
import sys
import termios
import time
import tty

import cv2
import numpy as np


HOST = "localhost"
PORT = 65000

DEFAULT_DETECTOR_DIR = "face_detection_model"
DEFAULT_EMBEDDER = "nn4.small2.v1.t7"
DEFAULT_RECOGNIZER = "output/recognizer.pickle"
DEFAULT_LABEL_ENCODER = "output/label.pickle"
DEFAULT_SFACE_RECOGNIZER = "output/candidates/sface/recognizer.pickle"
DEFAULT_SFACE_LABEL_ENCODER = "output/candidates/sface/label.pickle"
DEFAULT_YUNET_MODEL = "models/face_detection_yunet_2023mar.onnx"
DEFAULT_SFACE_MODEL = "models/face_recognition_sface_2021dec.onnx"

FACTOR = 1.0 / 255.0


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--face-backend", choices=["sface", "legacy"],
                        default="sface",
                        help="face detection and recognition backend")
    parser.add_argument("-d", "--detector", default=DEFAULT_DETECTOR_DIR,
                        help="OpenCV Caffe face detector directory")
    parser.add_argument("-m", "--embedding-model", default=DEFAULT_EMBEDDER,
                        help="Torch face embedding model")
    parser.add_argument("-r", "--recognizer", default=None,
                        help="trained face classifier")
    parser.add_argument("-l", "--le", default=None,
                        help="label encoder file")
    parser.add_argument("--yunet-model", default=DEFAULT_YUNET_MODEL,
                        help="YuNet detector model")
    parser.add_argument("--sface-model", default=DEFAULT_SFACE_MODEL,
                        help="SFace embedding model")

    parser.add_argument("--camera", type=int, default=0,
                        help="camera index")
    parser.add_argument("--width", type=int, default=640,
                        help="capture width")
    parser.add_argument("--height", type=int, default=480,
                        help="capture height")
    parser.add_argument("--fps", type=int, default=30,
                        help="capture FPS")
    parser.add_argument("--display-width", type=int, default=600,
                        help="frame width used for processing and display")
    parser.add_argument("--tracking-width", type=int, default=320,
                        help="frame width used by YuNet for tracking")
    parser.add_argument("--recognition-interval", type=float, default=1.0,
                        help="seconds between full-resolution recognition")

    parser.add_argument("-c", "--confidence", type=float, default=0.5,
                        help="minimum face detector confidence")
    parser.add_argument("--yunet-confidence", type=float, default=0.7,
                        help="minimum YuNet detector score")
    parser.add_argument("--recognition-threshold", type=float, default=0.80,
                        help="minimum probability to accept an identity")
    parser.add_argument("--send-unknown", action="store_true",
                        help="also send faces classified as unknown")
    parser.add_argument("--target-max-distance", type=float, default=80.0,
                        help="maximum center distance for retaining a target")
    parser.add_argument("--target-hold-frames", type=int, default=8,
                        help="frames to wait before switching a lost target")

    parser.add_argument("--host", default=HOST,
                        help="video server host")
    parser.add_argument("--port", type=int, default=PORT,
                        help="video server port")
    parser.add_argument("--no-server", action="store_true",
                        help="do not connect to the central server")
    parser.add_argument("--require-server", action="store_true",
                        help="exit if the central server is not available")

    parser.add_argument("--no-display", action="store_true",
                        help="do not show OpenCV display window")
    parser.add_argument("--show-all-faces", action="store_true",
                        help="also draw faces which are not the selected target")
    parser.add_argument("--save-frame", default="",
                        help="save last processed frame to this file")

    return parser.parse_args()


def connect_server(host, port, require_server=False):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    for attempt in range(10):
        try:
            sock.connect((host, port))
            print("video client connected")
            return sock
        except OSError:
            print("video client retrying server connection", attempt + 1)
            time.sleep(2)

    sock.close()

    if require_server:
        raise RuntimeError("Could not connect to the video server")

    print("video client running without server")
    return None


def open_camera(camera_index, width, height, fps):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    print("Camera opened")
    print("Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("FPS:", cap.get(cv2.CAP_PROP_FPS))

    return cap


def resize_to_width(frame, width):
    h, w = frame.shape[:2]
    scale = width / float(w)
    new_height = int(h * scale)
    return cv2.resize(frame, (width, new_height))


def send_face_message(sock, dx, dy, name):
    if sock is None:
        return

    message = f"{int(dx)},{int(dy)},{name}\n"
    sock.sendall(message.encode("utf-8"))


def is_exit_key(key):
    return key & 0xFF in (ord("q"), ord("Q"), 27)


class TerminalKeyReader:
    def __init__(self):
        self.enabled = sys.stdin.isatty()
        self.settings = None
        if self.enabled:
            self.settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

    def read(self):
        if not self.enabled:
            return -1
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return -1
        return ord(sys.stdin.read(1))

    def restore(self):
        if self.settings is not None:
            previous_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK, {signal.SIGTTOU}
            )
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            except (OSError, termios.error):
                pass
            finally:
                self.settings = None
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


class TargetSelector:
    def __init__(self, max_distance, hold_frames):
        self.max_distance = max_distance
        self.hold_frames = hold_frames
        self.target = None
        self.missed_frames = 0

    @staticmethod
    def _distance(candidate, target):
        dx = candidate["center_x"] - target["center_x"]
        dy = candidate["center_y"] - target["center_y"]
        return (dx * dx + dy * dy) ** 0.5

    @staticmethod
    def _initial_key(candidate):
        recognized = candidate["name"] != "unknown"
        return recognized, candidate["area"], candidate["confidence"]

    def select(self, candidates):
        if not candidates:
            if self.target is not None:
                self.missed_frames += 1
                if self.missed_frames > self.hold_frames:
                    self.target = None
            return None

        selected = None

        if self.target is not None:
            target_name = self.target["name"]
            continuity_distance = min(
                self.max_distance,
                max(30.0, self.target["area"] ** 0.5 * 0.75)
            )
            nearby = [
                candidate for candidate in candidates
                if self._distance(candidate, self.target) <= continuity_distance
            ]

            if target_name != "unknown":
                same_name = [
                    candidate for candidate in candidates
                    if candidate["name"] == target_name
                ]
                if same_name:
                    selected = min(
                        same_name,
                        key=lambda candidate: self._distance(
                            candidate, self.target
                        )
                    )
                else:
                    nearby = [
                        candidate for candidate in nearby
                        if candidate["name"] == "unknown"
                    ]

            if selected is None and nearby:
                selected = min(
                    nearby,
                    key=lambda candidate: self._distance(candidate, self.target)
                )
                if (target_name != "unknown"
                        and selected["name"] == "unknown"):
                    for key in ("name", "raw_name", "proba"):
                        selected[key] = self.target[key]
            elif selected is None and self.missed_frames < self.hold_frames:
                self.missed_frames += 1
                return None

        if selected is None:
            selected = max(candidates, key=self._initial_key)

        self.target = selected.copy()
        self.missed_frames = 0
        return selected


def scale_yunet_face(face, source_shape, destination_shape):
    source_height, source_width = source_shape[:2]
    destination_height, destination_width = destination_shape[:2]
    scale_x = destination_width / float(source_width)
    scale_y = destination_height / float(source_height)
    scaled = np.asarray(face, dtype=np.float32).copy()
    for index in (0, 2, 4, 6, 8, 10, 12):
        scaled[index] *= scale_x
    for index in (1, 3, 5, 7, 9, 11, 13):
        scaled[index] *= scale_y
    return scaled


def yunet_candidates(detector, tracking_frame, output_frame,
                     frame_center_x, frame_center_y):
    height, width = tracking_frame.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(tracking_frame)
    if faces is None:
        return []

    candidates = []
    for face in faces:
        scaled = scale_yunet_face(
            face, tracking_frame.shape, output_frame.shape
        )
        x, y, box_width, box_height = scaled[:4]
        start_x = max(0, int(round(x)))
        start_y = max(0, int(round(y)))
        end_x = min(output_frame.shape[1],
                    int(round(x + box_width)))
        end_y = min(output_frame.shape[0],
                    int(round(y + box_height)))
        if end_x - start_x < 20 or end_y - start_y < 20:
            continue
        center_x = (start_x + end_x) / 2.0
        center_y = (start_y + end_y) / 2.0
        candidates.append({
            "name": "unknown",
            "raw_name": "unknown",
            "proba": 0.0,
            "confidence": float(face[-1]),
            "center_x": center_x,
            "center_y": center_y,
            "dx": center_x - frame_center_x,
            "dy": center_y - frame_center_y,
            "area": (end_x - start_x) * (end_y - start_y),
            "box": (start_x, start_y, end_x, end_y),
        })
    return candidates


def recognize_sface_faces(detector, embedder, recognizer, le, full_frame,
                          output_frame, threshold):
    height, width = full_frame.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(full_frame)
    if faces is None:
        return []

    recognized = []
    for face in faces:
        aligned = embedder.alignCrop(full_frame, face)
        feature = embedder.feature(aligned).reshape(1, -1)
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature /= norm
        predictions = recognizer.predict_proba(feature)[0]
        index = int(np.argmax(predictions))
        probability = float(predictions[index])
        raw_name = str(le.classes_[index])
        name = raw_name if probability >= threshold else "unknown"
        scaled = scale_yunet_face(face, full_frame.shape, output_frame.shape)
        x, y, box_width, box_height = scaled[:4]
        recognized.append({
            "name": name,
            "raw_name": raw_name,
            "proba": probability,
            "center_x": x + box_width / 2.0,
            "center_y": y + box_height / 2.0,
            "area": box_width * box_height,
        })
    return recognized


def assign_recognized_identities(candidates, recognized):
    remaining = list(range(len(candidates)))
    for identity in recognized:
        if not remaining:
            break
        candidate_index = min(
            remaining,
            key=lambda index: TargetSelector._distance(
                candidates[index], identity
            )
        )
        candidate = candidates[candidate_index]
        max_distance = max(
            30.0,
            candidate["area"] ** 0.5,
            identity["area"] ** 0.5,
        )
        if TargetSelector._distance(candidate, identity) <= max_distance:
            for key in ("name", "raw_name", "proba"):
                candidate[key] = identity[key]
            remaining.remove(candidate_index)


def main():
    args = parse_args()

    sock = None
    if not args.no_server:
        sock = connect_server(args.host, args.port, args.require_server)

    if args.face_backend == "sface":
        recognizer_path = args.recognizer or DEFAULT_SFACE_RECOGNIZER
        label_encoder_path = args.le or DEFAULT_SFACE_LABEL_ENCODER
        print("[INFO] loading YuNet detector...")
        detector = cv2.FaceDetectorYN.create(
            args.yunet_model, "", (args.tracking_width,
                                   int(args.tracking_width * 3 / 4)),
            args.yunet_confidence, 0.3, 5000
        )
        print("[INFO] loading SFace recognizer...")
        embedder = cv2.FaceRecognizerSF.create(args.sface_model, "")
    else:
        recognizer_path = args.recognizer or DEFAULT_RECOGNIZER
        label_encoder_path = args.le or DEFAULT_LABEL_ENCODER
        proto_path = f"{args.detector}/deploy.prototxt"
        model_path = (
            f"{args.detector}/res10_300x300_ssd_iter_140000.caffemodel"
        )
        print("[INFO] loading face detector...")
        detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        print("[INFO] loading face recognizer...")
        embedder = cv2.dnn.readNetFromTorch(args.embedding_model)

    with open(recognizer_path, "rb") as recognizer_file:
        recognizer = pickle.load(recognizer_file)
    with open(label_encoder_path, "rb") as label_encoder_file:
        le = pickle.load(label_encoder_file)

    print("[INFO] starting video stream...")
    cap = open_camera(args.camera, args.width, args.height, args.fps)
    atexit.register(cap.release)
    atexit.register(cv2.destroyAllWindows)
    terminal_keys = TerminalKeyReader()
    atexit.register(terminal_keys.restore)

    def request_exit(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, request_exit)
    target_selector = TargetSelector(args.target_max_distance,
                                     args.target_hold_frames)
    window_name = "Face target"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    t0 = time.time()
    last_frame = None
    last_recognition = None

    while True:
        if is_exit_key(terminal_keys.read()):
            print("Exit requested from terminal")
            break

        ret, capture_frame = cap.read()

        if not ret:
            print("No frame")
            continue

        frame = resize_to_width(capture_frame, args.display_width)
        last_frame = frame.copy()

        h, w = frame.shape[:2]
        frame_center_x = w / 2.0
        frame_center_y = h / 2.0

        if args.face_backend == "sface":
            tracking_frame = resize_to_width(
                capture_frame, args.tracking_width
            )
            candidates = yunet_candidates(
                detector, tracking_frame, frame,
                frame_center_x, frame_center_y
            )
            now = time.monotonic()
            current_target = target_selector.target
            recognition_due = (
                candidates
                and (
                    last_recognition is None
                    or current_target is None
                    or current_target["name"] == "unknown"
                    or now - last_recognition >= args.recognition_interval
                )
            )
            if recognition_due:
                identities = recognize_sface_faces(
                    detector, embedder, recognizer, le,
                    capture_frame, frame, args.recognition_threshold
                )
                assign_recognized_identities(candidates, identities)
                last_recognition = now
        else:
            image_blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)

            detector.setInput(image_blob)
            detections = detector.forward()
            candidates = []

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence < args.confidence:
                    continue

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                startX, startY, endX, endY = box.astype("int")

                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                face = frame[startY:endY, startX:endX]
                fH, fW = face.shape[:2]

                if fW < 20 or fH < 20:
                    continue

                face_blob = cv2.dnn.blobFromImage(
                    face, FACTOR, (96, 96), (0, 0, 0),
                    swapRB=True, crop=False)

                embedder.setInput(face_blob)
                vec = embedder.forward()

                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                raw_name = str(le.classes_[j])

                if proba >= args.recognition_threshold:
                    name = raw_name
                else:
                    name = "unknown"

                face_center_x = (startX + endX) / 2.0
                face_center_y = (startY + endY) / 2.0

                candidates.append({
                    "name": name,
                    "raw_name": raw_name,
                    "proba": float(proba),
                    "confidence": float(confidence),
                    "center_x": face_center_x,
                    "center_y": face_center_y,
                    "dx": face_center_x - frame_center_x,
                    "dy": face_center_y - frame_center_y,
                    "area": (endX - startX) * (endY - startY),
                    "box": (startX, startY, endX, endY),
                })

        if args.show_all_faces:
            for candidate in candidates:
                startX, startY, endX, endY = candidate["box"]
                color = ((0, 255, 0)
                         if candidate["name"] != "unknown"
                         else (0, 0, 255))
                text = (
                    f"{candidate['name']}: "
                    f"{candidate['proba'] * 100:.1f}%"
                )
                y = startY - 10 if startY > 20 else startY + 20
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              color, 1)
                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        target = target_selector.select(candidates)
        if target is not None:
            startX, startY, endX, endY = target["box"]
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (255, 255, 0), 3)
            target_text = (
                f"TARGET {target['name']}: {target['proba'] * 100:.1f}%"
            )
            target_y = startY - 10 if startY > 20 else startY + 20
            cv2.putText(frame, target_text, (startX, target_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            print(f"target -> dx={target['dx']:.0f}, "
                  f"dy={target['dy']:.0f}, name={target['name']}, "
                  f"raw={target['raw_name']}, "
                  f"proba={target['proba']:.3f}, "
                  f"detector={target['confidence']:.3f}")

            if target["name"] != "unknown" or args.send_unknown:
                send_face_message(sock, target["dx"], target["dy"],
                                  target["name"])

        frame_count += 1

        if not args.no_display:
            cv2.imshow(window_name, frame)
            key = cv2.waitKeyEx(10)

            if is_exit_key(key):
                print("Exit requested from video window")
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Video window closed")
                break

    elapsed = time.time() - t0
    fps = frame_count / elapsed if elapsed > 0 else 0

    print("[INFO] elapsed time: {:.2f}".format(elapsed))
    print("[INFO] approx. FPS: {:.2f}".format(fps))

    if args.save_frame and last_frame is not None:
        ok = cv2.imwrite(args.save_frame, last_frame)
        print("Saved frame:", ok, args.save_frame)

    cap.release()

    if sock is not None:
        sock.close()

    terminal_keys.restore()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
