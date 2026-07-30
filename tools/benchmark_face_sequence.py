#!/usr/bin/env python3
"""Process one recorded sequence with one face backend configuration."""

from __future__ import annotations

import argparse
import json
import pickle
import resource
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from tools.benchmark_face_backends import percentile
from tools.benchmark_face_camera import temperature_c


CONFIGURATIONS = {
    "legacy-600": ("legacy", 600, False),
    "legacy-320": ("legacy", 320, False),
    "sface-600": ("sface", 600, False),
    "sface-320": ("sface", 320, False),
    "sface-hybrid": ("sface", 320, True),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/face_sequence.avi")
    parser.add_argument("--configuration", choices=CONFIGURATIONS,
                        required=True)
    parser.add_argument("--report", default="")
    return parser.parse_args()


def resize_to_width(frame, width):
    height, current_width = frame.shape[:2]
    if current_width == width:
        return frame
    scale = width / float(current_width)
    return cv2.resize(frame, (width, int(round(height * scale))))


def load_classifier(backend):
    directory = Path("output/candidates") / backend
    classifier = pickle.loads(
        (directory / "recognizer.pickle").read_bytes()
    )
    encoder = pickle.loads((directory / "label.pickle").read_bytes())
    return classifier, encoder


class LegacySequenceBackend:
    def __init__(self, process_width):
        self.process_width = process_width
        self.detector = cv2.dnn.readNetFromCaffe(
            "face_detection_model/deploy.prototxt",
            "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel",
        )
        self.embedder = cv2.dnn.readNetFromTorch("nn4.small2.v1.t7")

    def observations(self, original):
        frame = resize_to_width(original, self.process_width)
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        self.detector.setInput(blob)
        detections = self.detector.forward()
        observations = []

        for index in range(detections.shape[2]):
            confidence = float(detections[0, 0, index, 2])
            if confidence < 0.5:
                continue
            box = detections[0, 0, index, 3:7] * np.array(
                [width, height, width, height]
            )
            start_x, start_y, end_x, end_y = box.astype(int)
            start_x, start_y = max(0, start_x), max(0, start_y)
            end_x, end_y = min(width, end_x), min(height, end_y)
            face = frame[start_y:end_y, start_x:end_x]
            face_width = end_x - start_x
            face_height = end_y - start_y
            if face_width < 20 or face_height < 20:
                continue
            face_blob = cv2.dnn.blobFromImage(
                face, 1.0 / 255.0, (96, 96), (0, 0, 0),
                swapRB=True, crop=False
            )
            self.embedder.setInput(face_blob)
            observations.append({
                "feature": self.embedder.forward().reshape(-1),
                "detector_score": confidence,
                "face_width": face_width,
                "face_height": face_height,
                "frame_width": width,
                "frame_height": height,
            })
        return observations


class SFaceSequenceBackend:
    def __init__(self, detect_width, hybrid):
        self.detect_width = detect_width
        self.hybrid = hybrid
        self.detector = cv2.FaceDetectorYN.create(
            "models/face_detection_yunet_2023mar.onnx",
            "", (detect_width, int(detect_width * 0.75)), 0.7, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            "models/face_recognition_sface_2021dec.onnx", ""
        )

    @staticmethod
    def scale_face(face, scale_x, scale_y):
        scaled = face.copy()
        for index in (0, 2, 4, 6, 8, 10, 12):
            scaled[index] *= scale_x
        for index in (1, 3, 5, 7, 9, 11, 13):
            scaled[index] *= scale_y
        return scaled

    def observations(self, original):
        detection_frame = resize_to_width(original, self.detect_width)
        detect_height, detect_width = detection_frame.shape[:2]
        self.detector.setInputSize((detect_width, detect_height))
        _, faces = self.detector.detect(detection_frame)
        if faces is None:
            return []

        observations = []
        for face in faces:
            align_frame = detection_frame
            align_face = face
            if self.hybrid:
                original_height, original_width = original.shape[:2]
                align_frame = original
                align_face = self.scale_face(
                    face,
                    original_width / detect_width,
                    original_height / detect_height,
                )

            aligned = self.recognizer.alignCrop(align_frame, align_face)
            feature = self.recognizer.feature(aligned).reshape(-1)
            norm = np.linalg.norm(feature)
            if norm > 0:
                feature = feature / norm
            observations.append({
                "feature": feature,
                "detector_score": float(face[-1]),
                "face_width": float(face[2]),
                "face_height": float(face[3]),
                "frame_width": detect_width,
                "frame_height": detect_height,
            })
        return observations


def create_backend(configuration):
    backend_name, width, hybrid = CONFIGURATIONS[configuration]
    if backend_name == "legacy":
        return backend_name, LegacySequenceBackend(width)
    return backend_name, SFaceSequenceBackend(width, hybrid)


def summarize(values):
    if not values:
        return {"mean": None, "median": None, "p95": None}
    return {
        "mean": float(sum(values) / len(values)),
        "median": float(percentile(values, 50)),
        "p95": float(percentile(values, 95)),
    }


def load_phases(input_path):
    metadata_path = input_path.with_suffix(".json")
    if not metadata_path.is_file():
        return []
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    phases = []
    start = 0.0
    for phase in metadata.get("phases", []):
        end = start + float(phase["duration_seconds"])
        phases.append((phase["name"], start, end))
        start = end
    return phases


def phase_name_at(phases, elapsed):
    for name, start, end in phases:
        if start <= elapsed < end:
            return name
    return "unassigned"


def empty_phase_stats():
    return {
        "frames": 0,
        "frames_with_face": 0,
        "faces": 0,
        "predicted_names": Counter(),
        "top_probability": [],
        "top_two_margin": [],
    }


def main():
    args = parse_args()
    input_path = Path(args.input)
    backend_name, backend = create_backend(args.configuration)
    classifier, encoder = load_classifier(backend_name)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")
    video_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    phases = load_phases(input_path)
    phase_stats = {
        name: empty_phase_stats()
        for name, _, _ in phases
    }

    frames = 0
    frames_with_face = 0
    faces = 0
    processing_ms = []
    detector_scores = []
    face_widths = []
    face_area_ratios = []
    top_probabilities = []
    top_two_margins = []
    predicted_names = []
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    start_temp = temperature_c()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            phase_name = phase_name_at(phases, frames / video_fps)
            current_phase = phase_stats.setdefault(
                phase_name, empty_phase_stats()
            )
            current_phase["frames"] += 1
            processing_started = time.perf_counter()
            observations = backend.observations(frame)
            processing_ms.append(
                (time.perf_counter() - processing_started) * 1000.0
            )
            frames += 1
            if observations:
                frames_with_face += 1
                current_phase["frames_with_face"] += 1

            for observation in observations:
                probabilities = classifier.predict_proba(
                    [observation["feature"]]
                )[0]
                order = np.argsort(probabilities)
                best = int(order[-1])
                second = int(order[-2])
                top_probability = float(probabilities[best])
                margin = top_probability - float(probabilities[second])

                faces += 1
                current_phase["faces"] += 1
                detector_scores.append(observation["detector_score"])
                face_widths.append(observation["face_width"])
                face_area_ratios.append(
                    observation["face_width"] * observation["face_height"]
                    / (
                        observation["frame_width"]
                        * observation["frame_height"]
                    )
                )
                top_probabilities.append(top_probability)
                top_two_margins.append(margin)
                predicted_name = str(encoder.classes_[best])
                predicted_names.append(predicted_name)
                current_phase["predicted_names"][predicted_name] += 1
                current_phase["top_probability"].append(top_probability)
                current_phase["top_two_margin"].append(margin)
    finally:
        capture.release()

    wall_seconds = time.perf_counter() - wall_started
    cpu_seconds = time.process_time() - cpu_started
    phase_report = {
        name: {
            "frames": stats["frames"],
            "frames_with_face": stats["frames_with_face"],
            "frames_with_face_percent": (
                stats["frames_with_face"] * 100.0 / stats["frames"]
                if stats["frames"] else 0.0
            ),
            "faces": stats["faces"],
            "predicted_names": dict(stats["predicted_names"]),
            "top_probability": summarize(stats["top_probability"]),
            "top_two_margin": summarize(stats["top_two_margin"]),
        }
        for name, stats in phase_stats.items()
    }
    report = {
        "configuration": args.configuration,
        "input": str(input_path),
        "frames": frames,
        "frames_with_face": frames_with_face,
        "frames_with_face_percent": (
            frames_with_face * 100.0 / frames if frames else 0.0
        ),
        "faces": faces,
        "predicted_names": dict(Counter(predicted_names)),
        "phases": phase_report,
        "processing_fps": frames / wall_seconds if wall_seconds else 0.0,
        "processing_ms": summarize(processing_ms),
        "detector_score": summarize(detector_scores),
        "face_width_pixels": summarize(face_widths),
        "face_area_ratio": summarize(face_area_ratios),
        "top_probability": summarize(top_probabilities),
        "top_two_margin": summarize(top_two_margins),
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
        else Path(
            f"output/face_sequence_{args.configuration}.json"
        )
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
