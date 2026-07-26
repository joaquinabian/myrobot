# AGENTS.md

## Project context

- This project runs directly on a Raspberry Pi 5 with Debian 13 and Python 3.13.
- The files in this repository are the authoritative current version.
- Do not use previous myrobot versions as authoritative unless explicitly requested.
- The hardware includes a Logitech C920 camera, Adafruit Crickit, servos,
  microphone and Piper text-to-speech.
- The local directory `voices/` is required at runtime but must remain untracked.

## Working rules

- Inspect the existing code before proposing or applying changes.
- Keep changes small, focused and easy to review.
- Do not rename or move files unless explicitly requested.
- Do not reorganize the project structure unless explicitly requested.
- Do not invent modules, classes, APIs or hardware behaviour.
- Do not run `sudo`, install packages, remove packages, modify services or
  change system configuration without explicit approval.
- Do not automatically execute programs that move servos or activate hardware.
- Ask before tests involving servos, motors, camera, microphone, IR or GPIO.
- Never modify or commit `voices/`, credentials, tokens, passwords or secrets.
- Before finishing, show the Git diff.
- Run safe syntax checks when appropriate.
- Clearly state which changes have not been tested with real hardware.
