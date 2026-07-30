#!/usr/bin/env python3
"""Manually inspect camera framing and servo travel."""

import argparse

import cv2


LEFT_KEYS = (ord("a"), ord("A"), 65361)
UP_KEYS = (ord("w"), ord("W"), 65362)
RIGHT_KEYS = (ord("d"), ord("D"), 65363)
DOWN_KEYS = (ord("s"), ord("S"), 65364)
EXIT_KEYS = (ord("q"), ord("Q"), 27)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--vertical-angle", type=int, default=88)
    parser.add_argument("--horizontal-angle", type=int, default=90)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--min-angle", type=int, default=1)
    parser.add_argument("--max-angle", type=int, default=179)
    return parser.parse_args()


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def updated_angles(key, vertical, horizontal, step, minimum, maximum):
    if key in LEFT_KEYS:
        horizontal = clamp(horizontal - step, minimum, maximum)
    elif key in RIGHT_KEYS:
        horizontal = clamp(horizontal + step, minimum, maximum)
    elif key in UP_KEYS:
        vertical = clamp(vertical - step, minimum, maximum)
    elif key in DOWN_KEYS:
        vertical = clamp(vertical + step, minimum, maximum)
    return vertical, horizontal


def open_camera(args):
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")
    return cap


def main():
    args = parse_args()

    from adafruit_crickit import crickit as cr

    vertical_servo = cr.servo_1
    horizontal_servo = cr.servo_2
    vertical_servo.set_pulse_width_range(min_pulse=900, max_pulse=2100)
    horizontal_servo.set_pulse_width_range(min_pulse=900, max_pulse=2100)

    vertical = clamp(
        args.vertical_angle, args.min_angle, args.max_angle
    )
    horizontal = clamp(
        args.horizontal_angle, args.min_angle, args.max_angle
    )
    vertical_servo.angle = vertical
    horizontal_servo.angle = horizontal

    cap = open_camera(args)
    window_name = "Servo calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("Arrows or WASD move 2 degrees; q or Esc exits")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            cv2.putText(
                frame,
                f"vertical={vertical}  horizontal={horizontal}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "arrows/WASD: move   Q/ESC: exit",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            cv2.imshow(window_name, frame)
            key = cv2.waitKeyEx(20)
            if key in EXIT_KEYS:
                break
            if cv2.getWindowProperty(
                window_name, cv2.WND_PROP_VISIBLE
            ) < 1:
                break

            new_vertical, new_horizontal = updated_angles(
                key, vertical, horizontal, args.step,
                args.min_angle, args.max_angle
            )
            if new_vertical != vertical:
                vertical = new_vertical
                vertical_servo.angle = vertical
                print(f"vertical={vertical}")
            if new_horizontal != horizontal:
                horizontal = new_horizontal
                horizontal_servo.angle = horizontal
                print(f"horizontal={horizontal}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"final vertical={vertical} horizontal={horizontal}")


if __name__ == "__main__":
    main()
