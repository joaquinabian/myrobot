#!/usr/bin/env python3
"""Benchmark legacy OpenFace and YuNet/SFace on a blocked dataset split.

The script is offline-only: it does not open the camera, connect to the robot
server or access servos. Source images and the active models in output/ are
never modified.
"""

from __future__ import annotations

import argparse
import json
import pickle
import resource
import statistics
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


DEFAULT_LEGACY_MODEL = "nn4.small2.v1.t7"
DEFAULT_LEGACY_DETECTOR = "face_detection_model"
DEFAULT_YUNET_MODEL = "models/face_detection_yunet_2023mar.onnx"
DEFAULT_SFACE_MODEL = "models/face_recognition_sface_2021dec.onnx"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--backend", choices=["legacy", "sface", "both"],
                        default="both")
    parser.add_argument("--validation-count", type=int, default=30,
                        help="last readable images per identity used for validation")
    parser.add_argument("--legacy-model", default=DEFAULT_LEGACY_MODEL)
    parser.add_argument("--legacy-detector", default=DEFAULT_LEGACY_DETECTOR)
    parser.add_argument("--yunet-model", default=DEFAULT_YUNET_MODEL)
    parser.add_argument("--sface-model", default=DEFAULT_SFACE_MODEL)
    parser.add_argument("--yunet-score-threshold", type=float, default=0.7)
    parser.add_argument("--report", default="output/face_benchmark.json")
    parser.add_argument("--save-candidates", action="store_true",
                        help="save candidate recognizers outside active output files")
    return parser.parse_args()


def image_number(path):
    suffix = path.stem.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else path.name


def scan_dataset(dataset_dir, validation_count):
    dataset_dir = Path(dataset_dir)
    identities = {}
    unreadable = []

    for person_dir in sorted(path for path in dataset_dir.iterdir()
                             if path.is_dir()):
        readable = []
        paths = sorted(person_dir.glob("*.jpg"), key=image_number)
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                unreadable.append(str(path))
            else:
                readable.append(path)

        if len(readable) <= validation_count:
            raise ValueError(
                f"{person_dir.name} has {len(readable)} readable images; "
                f"need more than validation-count={validation_count}"
            )
        identities[person_dir.name] = {
            "train": readable[:-validation_count],
            "validation": readable[-validation_count:],
        }

    if len(identities) < 2:
        raise ValueError("At least two identity folders are required")
    return identities, unreadable


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percent / 100.0))
    return ordered[index]


def prepare_dataset_crop(image):
    height, width = image.shape[:2]
    border_y = max(8, int(height * 0.20))
    border_x = max(8, int(width * 0.20))
    padded = cv2.copyMakeBorder(
        image, border_y, border_y, border_x, border_x,
        cv2.BORDER_REPLICATE
    )
    padded_height, padded_width = padded.shape[:2]
    scale = max(1.0, 240.0 / min(padded_height, padded_width))
    if scale > 1.0:
        padded = cv2.resize(
            padded, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC
        )
    return padded


class LegacyEmbedder:
    name = "legacy"

    def __init__(self, model_path, detector_dir):
        self.net = cv2.dnn.readNetFromTorch(str(model_path))
        detector_dir = Path(detector_dir)
        self.detector = cv2.dnn.readNetFromCaffe(
            str(detector_dir / "deploy.prototxt"),
            str(detector_dir / "res10_300x300_ssd_iter_140000.caffemodel"),
        )

    def extract(self, image):
        prepared = prepare_dataset_crop(image)
        height, width = prepared.shape[:2]
        detector_blob = cv2.dnn.blobFromImage(
            cv2.resize(prepared, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        self.detector.setInput(detector_blob)
        detections = self.detector.forward()
        best = None
        for index in range(detections.shape[2]):
            confidence = float(detections[0, 0, index, 2])
            if best is None or confidence > best[0]:
                best = confidence, detections[0, 0, index, 3:7]
        if best is None or best[0] < 0.5:
            return None

        box = best[1] * np.array([width, height, width, height])
        start_x, start_y, end_x, end_y = box.astype(int)
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(width, end_x)
        end_y = min(height, end_y)
        face = prepared[start_y:end_y, start_x:end_x]
        if face.shape[0] < 20 or face.shape[1] < 20:
            return None

        blob = cv2.dnn.blobFromImage(
            face, 1.0 / 255.0, (96, 96), (0, 0, 0),
            swapRB=True, crop=False
        )
        self.net.setInput(blob)
        return self.net.forward().reshape(-1)


class SFaceEmbedder:
    name = "sface"

    def __init__(self, detector_path, recognizer_path, score_threshold):
        self.detector = cv2.FaceDetectorYN.create(
            str(detector_path), "", (320, 320), score_threshold, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            str(recognizer_path), ""
        )

    def extract(self, image):
        prepared = prepare_dataset_crop(image)
        height, width = prepared.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(prepared)
        if faces is None or len(faces) == 0:
            return None
        face = max(faces, key=lambda item: float(item[-1]))
        aligned = self.recognizer.alignCrop(prepared, face)
        feature = self.recognizer.feature(aligned).reshape(-1)
        norm = np.linalg.norm(feature)
        return feature / norm if norm > 0 else feature


def extract_dataset(embedder, identities):
    features = {"train": [], "validation": []}
    labels = {"train": [], "validation": []}
    sample_paths = {"train": [], "validation": []}
    failed = []
    timings_ms = []

    for person, split_paths in identities.items():
        for split, paths in split_paths.items():
            for path in paths:
                image = cv2.imread(str(path))
                started = time.perf_counter()
                feature = embedder.extract(image)
                timings_ms.append((time.perf_counter() - started) * 1000.0)
                if feature is None:
                    failed.append(str(path))
                    continue
                features[split].append(feature)
                labels[split].append(person)
                sample_paths[split].append(str(path))

    return features, labels, sample_paths, failed, timings_ms


def train_and_evaluate(features, labels, sample_paths):
    encoder = LabelEncoder()
    train_y = encoder.fit_transform(labels["train"])
    validation_y = encoder.transform(labels["validation"])

    classifier = SVC(C=1.0, kernel="linear", probability=True,
                     random_state=42)
    classifier.fit(np.asarray(features["train"]), train_y)
    predictions = classifier.predict(np.asarray(features["validation"]))
    predicted_names = encoder.inverse_transform(predictions)
    errors = [
        {
            "path": path,
            "expected": expected,
            "predicted": predicted,
        }
        for path, expected, predicted in zip(
            sample_paths["validation"],
            labels["validation"],
            predicted_names,
        )
        if expected != predicted
    ]

    return classifier, encoder, {
        "accuracy": float(accuracy_score(validation_y, predictions)),
        "labels": encoder.classes_.tolist(),
        "confusion_matrix": confusion_matrix(
            validation_y, predictions,
            labels=list(range(len(encoder.classes_)))
        ).tolist(),
        "validation_predictions": Counter(
            predicted_names
        ),
        "classification_errors": errors,
    }


def candidate_paths(backend):
    directory = Path("output/candidates") / backend
    return directory, directory / "recognizer.pickle", directory / "label.pickle"


def save_candidate(backend, classifier, encoder, metadata):
    directory, recognizer_path, label_path = candidate_paths(backend)
    directory.mkdir(parents=True, exist_ok=True)
    recognizer_path.write_bytes(pickle.dumps(classifier))
    label_path.write_bytes(pickle.dumps(encoder))
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )


def benchmark_backend(embedder, identities, unreadable, save_candidates):
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    features, labels, sample_paths, failed, timings_ms = extract_dataset(
        embedder, identities
    )
    classifier, encoder, evaluation = train_and_evaluate(
        features, labels, sample_paths
    )
    total_wall_seconds = time.perf_counter() - wall_started
    total_cpu_seconds = time.process_time() - cpu_started

    report = {
        "backend": embedder.name,
        "readable_used": {
            split: len(features[split])
            for split in ("train", "validation")
        },
        "embedding_failures": failed,
        "unreadable_source_images": unreadable,
        "timing_ms": {
            "mean": statistics.fmean(timings_ms),
            "median": statistics.median(timings_ms),
            "p95": percentile(timings_ms, 95),
            "samples": len(timings_ms),
        },
        "process": {
            "total_wall_seconds": total_wall_seconds,
            "total_cpu_seconds": total_cpu_seconds,
            "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        **evaluation,
    }

    if save_candidates:
        save_candidate(embedder.name, classifier, encoder, report)
    return report


def require_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main():
    args = parse_args()
    identities, unreadable = scan_dataset(
        args.dataset, args.validation_count
    )

    backends = []
    if args.backend in ("legacy", "both"):
        backends.append(LegacyEmbedder(
            require_file(args.legacy_model),
            args.legacy_detector,
        ))
    if args.backend in ("sface", "both"):
        backends.append(SFaceEmbedder(
            require_file(args.yunet_model),
            require_file(args.sface_model),
            args.yunet_score_threshold,
        ))

    dataset_summary = {
        person: {
            split: len(paths)
            for split, paths in split_paths.items()
        }
        for person, split_paths in identities.items()
    }
    report = {
        "dataset": str(Path(args.dataset)),
        "split": "last readable images per identity",
        "validation_count": args.validation_count,
        "dataset_summary": dataset_summary,
        "unreadable_count": len(unreadable),
        "results": [
            benchmark_backend(backend, identities, unreadable,
                              args.save_candidates)
            for backend in backends
        ],
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=dict) + "\n",
        encoding="utf-8"
    )

    print(f"report: {report_path}")
    for result in report["results"]:
        timing = result["timing_ms"]
        print(
            f"{result['backend']}: accuracy={result['accuracy']:.3f} "
            f"failures={len(result['embedding_failures'])} "
            f"median={timing['median']:.2f}ms p95={timing['p95']:.2f}ms "
            f"rss={result['process']['max_rss_kb'] / 1024:.1f}MiB"
        )


if __name__ == "__main__":
    main()
