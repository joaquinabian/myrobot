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
import pickle
import socket
import time

import cv2
import numpy as np


HOST = "localhost"
PORT = 65000

DEFAULT_DETECTOR_DIR = "face_detection_model"
DEFAULT_EMBEDDER = "nn4.small2.v1.t7"
DEFAULT_RECOGNIZER = "output/recognizer.pickle"
DEFAULT_LABEL_ENCODER = "output/label.pickle"

FACTOR = 1.0 / 255.0


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--detector", default=DEFAULT_DETECTOR_DIR,
                        help="OpenCV Caffe face detector directory")
    parser.add_argument("-m", "--embedding-model", default=DEFAULT_EMBEDDER,
                        help="Torch face embedding model")
    parser.add_argument("-r", "--recognizer", default=DEFAULT_RECOGNIZER,
                        help="trained face classifier")
    parser.add_argument("-l", "--le", default=DEFAULT_LABEL_ENCODER,
                        help="label encoder file")

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

    parser.add_argument("-c", "--confidence", type=float, default=0.5,
                        help="minimum face detector confidence")
    parser.add_argument("--recognition-threshold", type=float, default=0.80,
                        help="minimum probability to accept an identity")
    parser.add_argument("--send-unknown", action="store_true",
                        help="also send faces classified as unknown")

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


def main():
    args = parse_args()

    sock = None
    if not args.no_server:
        sock = connect_server(args.host, args.port, args.require_server)

    proto_path = f"{args.detector}/deploy.prototxt"
    model_path = f"{args.detector}/res10_300x300_ssd_iter_140000.caffemodel"

    print("[INFO] loading face detector...")
    detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)

    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(args.embedding_model)

    recognizer = pickle.loads(open(args.recognizer, "rb").read())
    le = pickle.loads(open(args.le, "rb").read())

    print("[INFO] starting video stream...")
    cap = open_camera(args.camera, args.width, args.height, args.fps)

    frame_count = 0
    t0 = time.time()
    last_frame = None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("No frame")
            continue

        frame = resize_to_width(frame, args.display_width)
        last_frame = frame.copy()

        h, w = frame.shape[:2]
        frame_center_x = w / 2.0
        frame_center_y = h / 2.0

        image_blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(image_blob)
        detections = detector.forward()

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
                color = (0, 255, 0)
            else:
                name = "unknown"
                color = (0, 0, 255)

            face_center_x = (startX + endX) / 2.0
            face_center_y = (startY + endY) / 2.0

            dx = face_center_x - frame_center_x
            dy = face_center_y - frame_center_y

            text = f"{name}: {proba * 100:.1f}%"
            y = startY - 10 if startY > 20 else startY + 20

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            print(f"face -> dx={dx:.0f}, dy={dy:.0f}, "
                  f"name={name}, raw={raw_name}, proba={proba:.3f}, "
                  f"detector={confidence:.3f}")

            if name != "unknown" or args.send_unknown:
                send_face_message(sock, dx, dy, name)

        frame_count += 1

        if not args.no_display:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
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

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
