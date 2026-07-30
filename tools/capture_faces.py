#!/usr/bin/env python3
"""Capture face images for myrobot_04 training.

This script does not modify the active recognizer in output/.
It only creates or extends:

    dataset/<name>/

Controls:
    s      toggle auto-save on/off
    space  save current detected face once
    q      quit

Recommended first use:

    .venv/bin/python -m tools.capture_faces --name joaquin --samples 150

Move your head slowly:
- center
- left/right
- up/down
- nearer/farther
- different light angles
- with and without glasses, if relevant
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


DEFAULT_DETECTOR_DIR = "face_detection_model"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", required=True,
                        help="person name, used as dataset folder name")
    parser.add_argument("--dataset", default="dataset",
                        help="dataset root folder")
    parser.add_argument("--detector", default=DEFAULT_DETECTOR_DIR,
                        help="folder with deploy.prototxt and caffemodel")

    parser.add_argument("--camera", type=int, default=0,
                        help="camera index")
    parser.add_argument("--width", type=int, default=640,
                        help="capture width")
    parser.add_argument("--height", type=int, default=480,
                        help="capture height")
    parser.add_argument("--fps", type=int, default=30,
                        help="capture FPS")
    parser.add_argument("--display-width", type=int, default=600,
                        help="display/process width")

    parser.add_argument("--confidence", type=float, default=0.5,
                        help="minimum face detector confidence")
    parser.add_argument("--samples", type=int, default=150,
                        help="number of face images to save")
    parser.add_argument("--interval", type=float, default=0.25,
                        help="minimum seconds between auto-saved samples")
    parser.add_argument("--margin", type=float, default=0.25,
                        help="extra margin around detected face")
    parser.add_argument("--min-face-size", type=int, default=50,
                        help="minimum accepted face width/height in pixels")

    parser.add_argument("--no-window", action="store_true",
                        help="run without OpenCV window")

    return parser.parse_args()


def open_camera(camera_index, width, height, fps):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    print("camera opened")
    print("width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("fps:", cap.get(cv2.CAP_PROP_FPS))

    return cap


def resize_to_width(frame, width):
    h, w = frame.shape[:2]
    scale = width / float(w)
    new_height = int(h * scale)
    return cv2.resize(frame, (width, new_height))


def load_detector(detector_dir):
    detector_dir = Path(detector_dir)
    proto = detector_dir / "deploy.prototxt"
    model = detector_dir / "res10_300x300_ssd_iter_140000.caffemodel"

    if not proto.exists():
        raise FileNotFoundError(proto)
    if not model.exists():
        raise FileNotFoundError(model)

    return cv2.dnn.readNetFromCaffe(str(proto), str(model))


def find_best_face(detector, frame, min_confidence):
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False
    )

    detector.setInput(blob)
    detections = detector.forward()

    best = None
    best_area = 0

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])

        if confidence < min_confidence:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        start_x, start_y, end_x, end_y = box.astype("int")

        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(w, end_x)
        end_y = min(h, end_y)

        face_w = end_x - start_x
        face_h = end_y - start_y
        area = face_w * face_h

        if area > best_area:
            best_area = area
            best = (start_x, start_y, end_x, end_y, confidence)

    return best


def expand_box(box, frame_shape, margin):
    start_x, start_y, end_x, end_y, confidence = box
    h, w = frame_shape[:2]

    face_w = end_x - start_x
    face_h = end_y - start_y
    mx = int(face_w * margin)
    my = int(face_h * margin)

    start_x = max(0, start_x - mx)
    start_y = max(0, start_y - my)
    end_x = min(w, end_x + mx)
    end_y = min(h, end_y + my)

    return start_x, start_y, end_x, end_y, confidence


def next_index(person_dir):
    existing = sorted(person_dir.glob("*.jpg"))
    max_index = 0

    for path in existing:
        stem = path.stem
        parts = stem.split("_")
        if parts and parts[-1].isdigit():
            max_index = max(max_index, int(parts[-1]))

    return max_index + 1


def save_face(frame, box, person_dir, name, index):
    start_x, start_y, end_x, end_y, confidence = box
    face = frame[start_y:end_y, start_x:end_x]

    filename = f"{name}_{index:04d}.jpg"
    output_path = person_dir / filename

    ok = cv2.imwrite(str(output_path), face)
    if not ok:
        raise RuntimeError(f"Could not save image: {output_path}")

    print(f"saved {output_path} confidence={confidence:.3f}")
    return output_path


def draw_status(frame, box, saved_count, target_count, auto_save):
    status = "AUTO ON" if auto_save else "AUTO OFF"
    text = f"{status}  saved {saved_count}/{target_count}"

    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)

    if box is None:
        cv2.putText(frame, "no face", (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 2)
        return

    start_x, start_y, end_x, end_y, confidence = box
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    conf_text = f"face {confidence:.2f}"
    y = start_y - 10 if start_y > 20 else start_y + 20
    cv2.putText(frame, conf_text, (start_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)


def main():
    args = parse_args()

    person_dir = Path(args.dataset) / args.name
    person_dir.mkdir(parents=True, exist_ok=True)

    detector = load_detector(args.detector)
    cap = open_camera(args.camera, args.width, args.height, args.fps)

    auto_save = False
    saved_count = 0
    sample_index = next_index(person_dir)
    last_save_time = 0.0

    print("dataset folder:", person_dir)
    print("press s to toggle auto-save")
    print("press space to save one face")
    print("press q to quit")

    try:
        while saved_count < args.samples:
            ret, frame = cap.read()
            if not ret:
                print("no frame")
                continue

            frame = resize_to_width(frame, args.display_width)
            box = find_best_face(detector, frame, args.confidence)

            valid_face = False
            expanded_box = None

            if box is not None:
                expanded_box = expand_box(box, frame.shape, args.margin)
                start_x, start_y, end_x, end_y, confidence = expanded_box
                face_w = end_x - start_x
                face_h = end_y - start_y
                valid_face = (
                    face_w >= args.min_face_size
                    and face_h >= args.min_face_size
                )

            now = time.time()
            should_save = (
                auto_save
                and valid_face
                and (now - last_save_time) >= args.interval
            )

            if should_save:
                save_face(frame, expanded_box, person_dir, args.name, sample_index)
                saved_count += 1
                sample_index += 1
                last_save_time = now

            display = frame.copy()
            draw_status(display, expanded_box if valid_face else None,
                        saved_count, args.samples, auto_save)

            if not args.no_window:
                cv2.imshow("capture_faces", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                if key == ord("s"):
                    auto_save = not auto_save
                    print("auto-save:", auto_save)
                if key == ord(" ") and valid_face:
                    save_face(frame, expanded_box, person_dir,
                              args.name, sample_index)
                    saved_count += 1
                    sample_index += 1
                    last_save_time = now

        print("capture done")
        print("saved in:", person_dir)
        print("new images:", saved_count)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
