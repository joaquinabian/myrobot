# myrobot
Robot with facial recognition, face tracking and some conversation

## Video client face backend

`robot_video_client.py` uses YuNet/SFace by default. YuNet detects faces at
320 pixels for tracking. Detection, alignment and recognition run on the
capture frame when a target is acquired and every second thereafter:

```bash
.venv/bin/python -u robot_video_client.py
```

The output frame remains 600 pixels wide by default, so switching the tracking
detector to 320 pixels does not change the scale of `dx` and `dy`.

Change the recognition period with `--recognition-interval`. Use the previous
Caffe/OpenFace backend without changing files:

```bash
.venv/bin/python -u robot_video_client.py --face-backend legacy
```

## Face motion control

`robot_server.py` filters the latest face error before converting it to servo
increments. Defaults:

- calibrated start position: vertical `121`, horizontal `73`;
- update interval: `0.08` seconds;
- error filter: EMA with `--video-filter-alpha 0.35`;
- deadband: `35` pixels per axis;
- conversion: `45` pixels per degree;
- maximum increment: `3` horizontal and `2` vertical degrees per update;
- target recovery: continue the last movement for `2.0` seconds.

The maximum increment can be set per axis with `--video-max-step-x` and
`--video-max-step-y`. An omitted axis uses `--video-max-step`.
Target recovery duration is set with `--video-recovery-timeout`; use `0` to
disable it.

Start positions and travel limits are configurable per servo with
`--servo-1-start-angle`, `--servo-1-min-angle`, `--servo-1-max-angle` and the
corresponding `--servo-2-*` options. The vertical limits remain `1–179` until
the camera is installed in its final location.

The servos retain their position after video data stops. Optional return to
the configured rest angles is enabled by setting a nonzero timeout:

```bash
.venv/bin/python -u robot_server.py \
  --video-rest-timeout 5 \
  --video-rest-angle-1 90 \
  --video-rest-angle-2 90
```

Use `--dry-run --no-lirc --no-say --no-bored` to test the server without
Crickit, IR or speech output.

## Offline face backend benchmark

`tools/benchmark_face_backends.py` compares the current Caffe/OpenFace pipeline with
YuNet/SFace without opening the camera or connecting to the robot server.
Source images and the active recognizer files in `output/` are not modified.

The split is sequential: the last 30 readable images of every identity are
validation data and all earlier images are training data.

Run each backend separately when comparing memory use:

```bash
.venv/bin/python -u -m tools.benchmark_face_backends \
  --backend legacy \
  --report output/face_benchmark_legacy.json

.venv/bin/python -u -m tools.benchmark_face_backends \
  --backend sface \
  --report output/face_benchmark_sface.json
```

Run both on the same split and save candidate classifiers:

```bash
.venv/bin/python -u -m tools.benchmark_face_backends \
  --backend both \
  --save-candidates
```

Candidate files are written below `output/candidates/`; the active
`output/recognizer.pickle` and `output/label.pickle` are left unchanged.

This same-session validation is useful for comparing backends, but it does not
measure robustness across different capture days or unknown people.

### Camera benchmark

The real-camera benchmark opens the C920 for a limited time but never connects
to the robot server or accesses servos:

```bash
.venv/bin/python -u -m tools.benchmark_face_camera \
  --backend legacy \
  --duration 20

.venv/bin/python -u -m tools.benchmark_face_camera \
  --backend sface \
  --duration 20 \
  --process-width 320
```

Reports are written below `output/face_camera_*.json`. Run one backend at a
time and keep the same face, distance and lighting.

### Recorded sequence comparison

Capture one 40-second sequence:

```bash
.venv/bin/python -u -m tools.capture_face_sequence
```

Process the same frames with each configuration:

```bash
for configuration in \
  legacy-600 legacy-320 \
  sface-600 sface-320 sface-hybrid
do
  .venv/bin/python -u -m tools.benchmark_face_sequence \
    --configuration "$configuration"
done
```

`sface-hybrid` detects at 320 pixels and rescales the bounding box and five
landmarks to align the face from the original 640-pixel frame. Reports include
global and per-phase measurements.

## Tests

Run all tests from the project root:

```bash
.venv/bin/python -m unittest discover -s tests
```

Tests do not activate the camera, microphone, IR, GPIO or real servos.

## Local data

`dataset/`, `output/candidates/`, `output/face_sequence.avi` and `voices/`
remain local and are not committed. They contain face data, derived
classifiers, recorded video or the installed Piper voice.
