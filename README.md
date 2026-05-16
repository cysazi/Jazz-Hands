# Jazz Hands

Jazz Hands is a glove-based musical instrument. The active runtime is now the
two-camera version:

- 2-camera IR mocap for hand position
- ESP-NOW IMU packets for glove rotation and button state
- MIDI output for note playback
- optional haptic pulses on note changes
- a VisPy visualizer for live feedback

## Running The System

From the repo root:

```powershell
python .\JazzHands.py
```

`JazzHands.py` is intentionally small. It is the one-button launcher for the
organized package code in `jazzhands/`.

## Repo Layout

```text
JazzHands.py
  One-button runtime launcher.

jazzhands/
  Active importable app code.

Test_Debug_Scripts/
  Manual scripts for hardware checks, calibration runs, and visualizer testing.
  These are entry points, not modules imported by the main app.

Archive/
  Preserved old experiments, four-camera work, older firmware, and legacy
  monolithic scripts. Nothing here is part of the active runtime flow.
```

## Active Runtime Modules

```text
jazzhands/app.py
  Main two-camera playing app.

jazzhands/mocap/
  Two-camera camera settings, calibration, tracking, movement alignment, and
  calibration JSON files.

jazzhands/visualizer/
  FL Studio / MIDI plane visualizer logic.

jazzhands/haptics/
  Python serial haptics driver.

jazzhands/assets/
  Hand meshes used by the visualizer.

jazzhands/firmware/espnow_imu/
  Current ESP-NOW receiver and hand-module firmware.
```

## Manual Test And Debug Scripts

Useful entry points:

```powershell
python .\Test_Debug_Scripts\debug_fl_studio_visualizer.py
python .\Test_Debug_Scripts\apply_camera_settings.py --help
python .\Test_Debug_Scripts\calibrate_mocap_cameras.py
python .\Test_Debug_Scripts\align_mocap_movement.py
python .\Test_Debug_Scripts\full_mocap_run.py
python .\Test_Debug_Scripts\reflective_ball_playing_test.py
```

## Calibration Files

The active two-camera calibration files live in:

- `jazzhands/mocap/mocap_calibration.json`
- `jazzhands/mocap/mocap_calibration_aligned.json`
- `jazzhands/mocap/camera_uvc_settings_values.json`

The runtime prefers the aligned calibration file.

## Haptics

Haptics are controlled in Python by `jazzhands/haptics/controller.py`.

The current ESP-NOW firmware pair lives under `jazzhands/firmware/espnow_imu/`:

- `receiver_espnow_imu/receiver_espnow_imu.ino`
  - forwards IMU packets to Python
  - accepts binary haptics commands from Python over USB serial
  - relays those commands back to the gloves over ESP-NOW
- `hand_module_espnow_imu/hand_module_espnow_imu.ino`
  - sends IMU packets to the receiver
  - receives haptics commands from the receiver
  - drives the local haptics motor on the glove

## Dependencies

At minimum, the current runtime expects:

- Python
- `numpy`
- `pyserial`
- `vispy`
- a VisPy GUI backend such as `PyQt6`
- `mido`
- an RTMidi backend for MIDI output
- `opencv-python` for camera mocap
