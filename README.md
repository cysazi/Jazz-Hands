# Jazz Hands

Jazz Hands is a glove-based musical instrument. The current runtime is centered on:

- 2-camera IR mocap for hand position
- ESP32 IMU packets for glove rotation and button state
- MIDI output for note playback
- optional haptic pulses on note changes
- a VisPy visualizer for live feedback

The current goal is independence: if you run `JazzHandsKalman.py`, the system should guide startup from one entrypoint.

## Current Architecture

Right now, the main runtime is [JazzHandsKalman.py](./JazzHandsKalman.py).

It does the following:

1. Applies saved UVC settings for cameras 1 and 2.
2. Prompts: `Do you want to initiate calibration?`
3. If you answer `yes`, it launches:
   - 2-camera pose calibration
   - movement-based world-axis alignment
4. Opens the serial IMU receiver.
   - if `COM_PORTS` is left empty, Jazz Hands auto-detects likely ESP32 serial ports
5. Starts the 2-camera mocap feed.
6. Maps glove motion to note and control data.
7. Sends MIDI and optional haptic pulses.
8. Opens the live visualizer.

## Hand Responsibilities

- Right hand:
  - controls the played note
  - controls attack
  - controls stereo position
  - controls instrument cycling gestures
  - triggers note-change haptics
- Left hand:
  - controls volume
  - controls reverb mode
  - can move the octave down
- Right-hand button:
  - moves the octave up

## Running The System

From the repo root:

```powershell
python .\JazzHandsKalman.py
```

Startup behavior:

- Saved 2-camera settings are applied automatically from `camera_tests/camera_uvc_settings_values.json`.
- You will be asked whether to initiate calibration.
- If you answer `no`, Jazz Hands will reuse the latest saved calibration files.
- If you answer `yes`, Jazz Hands will walk through the full 2-camera calibration flow before starting the runtime.

## Calibration Flow

When calibration is requested, Jazz Hands launches the existing camera tools in this order:

1. `camera_tests/camera_uvc_settings.py --apply`
2. `camera_tests/calibrate_mocap_cameras.py`
3. `camera_tests/mocap_movement_alignment.py`

The saved files are:

- raw camera calibration: `camera_tests/mocap_calibration.json`
- aligned runtime calibration: `camera_tests/mocap_calibration_aligned.json`

The runtime prefers the aligned calibration file.

## Haptics

Haptics are controlled in Python by [haptics_controller.py](./haptics_controller.py) and on the ESP32 by [Haptics1.cpp](./Haptics1.cpp).

Behavior:

- A short pulse is triggered when the right-hand note changes.
- This now uses the receiver path: `Python -> receiver ESP32 -> ESP-NOW -> target hand module`.
- The pulse is available for note preview too, not only while MIDI is actively playing.
- This is controlled by the boolean in `JazzHandsKalman.py`:

```python
ENABLE_NOTE_HAPTICS = True
```

If you want the system to run without note pulses, set that value to `False`.

### Haptics Wiring

Current haptics firmware assumes:

- `GPIO25` -> SDA
- `GPIO26` -> SCL

This is intentional because the glove currently uses:

- `GPIO21` and `GPIO22` for the IMU I2C bus
- `GPIO23` for the glove button

### Receiver And Hand Firmware

The newer ESP-NOW firmware pair under `camera_tests/espnow_imu/` is now the intended path for haptics integration:

- `receiver_espnow_imu/receiver_espnow_imu.ino`
  - forwards IMU packets to Python
  - accepts binary haptics commands from Python over USB serial
  - relays those commands back to the gloves over ESP-NOW
- `hand_module_espnow_imu/hand_module_espnow_imu.ino`
  - sends IMU packets to the receiver
  - receives haptics commands from the receiver
  - drives the local haptics motor on the glove

The receiver currently includes placeholder left/right glove MAC addresses that should be replaced before flashing real hardware.

## Important Files

- `JazzHandsKalman.py`
  - main entrypoint
- `haptics_controller.py`
  - Python serial haptics driver
- `Haptics1.cpp`
  - ESP32 haptics firmware
- `camera_tests/camera_uvc_settings.py`
  - applies saved UVC camera settings
- `camera_tests/calibrate_mocap_cameras.py`
  - 2-camera pose calibration
- `camera_tests/mocap_movement_alignment.py`
  - world-axis alignment

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

## Current Focus

The repo still contains older experiments for UWB, sound, visualization, and 4-camera work, but the active focus is:

- make `JazzHandsKalman.py` the single launch point
- keep the runtime solid on 2 cameras first
- keep haptics tied to played-note feedback
- return to 4-camera support after the 2-camera path is stable
