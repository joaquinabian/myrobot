#!/usr/bin/env python3
"""Benchmark one face backend on the real camera without robot hardware."""

from __future__ import annotations

import argparse
import json
import pickle
import resource
import time
from pathlib import Path

import cv2
import numpy as np

from tools.benchmark_face_backends import percentile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["legacy", "sface"],
                        required=True)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--process-width", type=int, default=600)
    parser.add_argument("--recognition-threshold", type=float, default=0.8)
    parser.add_argument("--report", default="")
    return parser.parse_args()


def temperature_c():
    path = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        return int(path.read_text(encoding="ascii").strip()) / 1000.0
    except (OSError, ValueError):
        return None


def open_camera(args):
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")
    return cap


def resize_to_width(frame, width):
    height, current_width = frame.shape[:2]
    if current_width == width:
        return frame
    scale = width / float(current_width)
    return cv2.resize(frame, (width, int(height * scale)))


def load_classifier(backend):
    directory = Path("output/candidates") / backend
    classifier = pickle.loads(
        (directory / "recognizer.pickle").read_bytes()
    )
    encoder = pickle.loads((directory / "label.pickle").read_bytes())
    return classifier, encoder


class LegacyCameraBackend:
    def __init__(self):
        self.detector = cv2.dnn.readNetFromCaffe(
            "face_detection_model/deploy.prototxt",
            "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel",
        )
        self.embedder = cv2.dnn.readNetFromTorch("nn4.small2.v1.t7")

    def features(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        self.detector.setInput(blob)
        detections = self.detector.forward()
        features = []

        for index in range(detections.shape[2]):
            if float(detections[0, 0, index, 2]) < 0.5:
                continue
            box = detections[0, 0, index, 3:7] * np.array(
                [width, height, width, height]
            )
            start_x, start_y, end_x, end_y = box.astype(int)
            start_x, start_y = max(0, start_x), max(0, start_y)
            end_x, end_y = min(width, end_x), min(height, end_y)
            face = frame[start_y:end_y, start_x:end_x]
            if face.shape[0] < 20 or face.shape[1] < 20:
                continue
            face_blob = cv2.dnn.blobFromImage(
                face, 1.0 / 255.0, (96, 96), (0, 0, 0),
                swapRB=True, crop=False
            )
            self.embedder.setInput(face_blob)
            features.append(self.embedder.forward().reshape(-1))
        return features


class SFaceCameraBackend:
    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create(
            "models/face_detection_yunet_2023mar.onnx",
            "", (600, 450), 0.7, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            "models/face_recognition_sface_2021dec.onnx", ""
        )

    def features(self, frame):
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return []

        features = []
        for face in faces:
            aligned = self.recognizer.alignCrop(frame, face)
            feature = self.recognizer.feature(aligned).reshape(-1)
            norm = np.linalg.norm(feature)
            features.append(feature / norm if norm > 0 else feature)
        return features


def create_backend(name):
    if name == "legacy":
        return LegacyCameraBackend()
    return SFaceCameraBackend()


def main():
    args = parse_args()
    classifier, encoder = load_classifier(args.backend)
    backend = create_backend(args.backend)
    cap = open_camera(args)

    frames = 0
    read_failures = 0
    faces = 0
    frames_with_face = 0
    accepted = 0
    names = {}
    processing_ms = []
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    start_temp = temperature_c()

    try:
        while time.perf_counter() - wall_started < args.duration:
            ok, frame = cap.read()
            if not ok:
                read_failures += 1
                continue
            frame = resize_to_width(frame, args.process_width)

            processing_started = time.perf_counter()
            frame_features = backend.features(frame)
            for feature in frame_features:
                probabilities = classifier.predict_proba([feature])[0]
                best = int(np.argmax(probabilities))
                probability = float(probabilities[best])
                if probability >= args.recognition_threshold:
                    name = str(encoder.classes_[best])
                    accepted += 1
                    names[name] = names.get(name, 0) + 1
            processing_ms.append(
                (time.perf_counter() - processing_started) * 1000.0
            )
            faces += len(frame_features)
            if frame_features:
                frames_with_face += 1
            frames += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    wall_seconds = time.perf_counter() - wall_started
    cpu_seconds = time.process_time() - cpu_started
    report = {
        "backend": args.backend,
        "duration_requested_seconds": args.duration,
        "wall_seconds": wall_seconds,
        "frames": frames,
        "camera_fps": frames / wall_seconds if wall_seconds else 0.0,
        "read_failures": read_failures,
        "faces": faces,
        "frames_with_face": frames_with_face,
        "frames_with_face_percent": (
            frames_with_face * 100.0 / frames if frames else 0.0
        ),
        "accepted_faces": accepted,
        "accepted_face_percent": (
            accepted * 100.0 / faces if faces else 0.0
        ),
        "accepted_names": names,
        "processing_ms": {
            "mean": sum(processing_ms) / len(processing_ms),
            "median": percentile(processing_ms, 50),
            "p95": percentile(processing_ms, 95),
        },
        "cpu": {
            "process_seconds": cpu_seconds,
            "equivalent_cores": cpu_seconds / wall_seconds,
        },
        "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "temperature_c": {
            "start": start_temp,
            "end": temperature_c(),
        },
    }

    report_path = (
        Path(args.report) if args.report
        else Path(f"output/face_camera_{args.backend}.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
