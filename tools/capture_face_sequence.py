#!/usr/bin/env python3
"""Capture one camera sequence for repeatable face backend benchmarks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2


DEFAULT_PHASES = [
    ("center", 5.0),
    ("turn_left", 5.0),
    ("turn_right", 5.0),
    ("look_up_down", 5.0),
    ("move_near_far", 10.0),
    ("leave_and_return", 10.0),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output/face_sequence.avi")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def is_exit_key(key):
    return key & 0xFF in (ord("q"), ord("Q"), 27)


def is_start_key(key):
    return key & 0xFF in (ord("s"), ord("S"), 13, 32)


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


def phase_at(elapsed):
    phase_start = 0.0
    for name, duration in DEFAULT_PHASES:
        phase_end = phase_start + duration
        if elapsed < phase_end:
            return name, phase_start, phase_end
        phase_start = phase_end
    return None


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = open_camera(args)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_duration = sum(duration for _, duration in DEFAULT_PHASES)
    frames = 0
    read_failures = 0
    window_name = "Face sequence capture"
    writer = None

    print("Capture phases:")
    for name, duration in DEFAULT_PHASES:
        print(f"  {name}: {duration:.0f}s")

    try:
        if not args.no_display:
            print("Press SPACE, Enter or s to start; q or Esc to cancel")
            while True:
                ok, frame = cap.read()
                if not ok:
                    read_failures += 1
                    continue

                display = frame.copy()
                cv2.putText(
                    display, "FRAME PREVIEW - SPACE/ENTER/S to start",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 255), 2
                )
                cv2.putText(
                    display, "Q/ESC to cancel", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
                )
                cv2.imshow(window_name, display)
                key = cv2.waitKeyEx(10)
                if is_start_key(key):
                    break
                if is_exit_key(key):
                    print("Capture cancelled by user")
                    return
                if cv2.getWindowProperty(
                    window_name, cv2.WND_PROP_VISIBLE
                ) < 1:
                    print("Capture window closed")
                    return

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            args.fps,
            (actual_width, actual_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {output_path}")

        started = time.perf_counter()
        while True:
            elapsed = time.perf_counter() - started
            phase = phase_at(elapsed)
            if phase is None:
                break

            ok, frame = cap.read()
            if not ok:
                read_failures += 1
                continue

            writer.write(frame)
            frames += 1

            if not args.no_display:
                name, _, phase_end = phase
                display = frame.copy()
                text = f"{name}  {phase_end - elapsed:.1f}s"
                cv2.putText(
                    display, text, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
                )
                cv2.imshow(window_name, display)
                if is_exit_key(cv2.waitKeyEx(1)):
                    print("Capture stopped by user")
                    break
                if cv2.getWindowProperty(
                    window_name, cv2.WND_PROP_VISIBLE
                ) < 1:
                    print("Capture window closed")
                    break
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()

    wall_seconds = time.perf_counter() - started
    metadata = {
        "video": str(output_path),
        "width": actual_width,
        "height": actual_height,
        "camera_fps_reported": actual_fps,
        "frames": frames,
        "read_failures": read_failures,
        "wall_seconds": wall_seconds,
        "capture_fps": frames / wall_seconds if wall_seconds else 0.0,
        "phases": [
            {"name": name, "duration_seconds": duration}
            for name, duration in DEFAULT_PHASES
        ],
        "planned_duration_seconds": total_duration,
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"video: {output_path}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
