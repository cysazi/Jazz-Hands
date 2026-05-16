"""
Mocap playing test.

This script keeps the camera/mocap tracking code here, but imports the FL Studio
debug visualizer for the hand meshes, plane writing, play-side gate, and MIDI.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import ctypes
import math
import struct
import sys
import threading
import time

import numpy as np

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from jazzhands.haptics.controller import HapticsController
from jazzhands.mocap import tracker as mocap
from jazzhands.mocap import tracker_combined_vispy as combined
from jazzhands.visualizer import fl_studio_debug_visualizer as fl_debug


DEFAULT_PREVIEW_HZ = 15.0
DEFAULT_VISUALIZER_HZ = 60
HAND_LABELS = ("LEFT", "RIGHT")
HAND_SIDE_AXIS = "y"
RIGHT_HAND_HIGHER_ON_SIDE_AXIS = False
CONTROLLED_HAND_LABEL = "RIGHT"
DEFAULT_PLAYING_TEST_CALIBRATION_PATH = combined.ALIGNED_MOVEMENT_CALIBRATION_PATH
USE_SERIAL_IMU_ROTATION = True
IMU_PLANE_DRAW_HAND_LABEL = "RIGHT"
IMU_CHANNEL_CYCLE_HAND_LABEL = "LEFT"
IMU_CHANNEL_CYCLE_TARGET_HAND_LABEL = "CONTROLLED"
IMU_SERIAL_PORT: str | None = None
IMU_SERIAL_BAUD = 115200
IMU_PACKET_STALE_SECONDS = 0.25

IMU_PACKET_HEADER = 0xAAAA
IMU_HEADER_BYTES = b"\xAA\xAA"
IMU_PACKET_STRUCT = struct.Struct("<HBIIBB7fB")
IMU_PACKET_SIZE = IMU_PACKET_STRUCT.size
HAPTICS_COMMAND_STRUCT = struct.Struct("<HBBH")
HAPTICS_COMMAND_HEADER = 0xCC33
IMU_PACKET_HAS_ACCEL = 0b00000001
IMU_PACKET_HAS_QUAT = 0b00000010
IMU_PACKET_HAS_BUTTON = 0b00000100
IMU_PACKET_HAS_ERROR = 0b10000000
IMU_DEVICE_ID_TO_HAND = {
    1: "LEFT",
    2: "RIGHT",
}
IMU_QUATERNION_COMPONENT_SIGNS = {
    "LEFT": np.array([1.0, -1.0, 1.0, 1.0], dtype=np.float64),
    "RIGHT": np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64),
}
IMU_MOUNTING_QUATERNION_OFFSETS = {
    "LEFT": fl_debug.quaternion_multiply(
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64),
    ),
    "RIGHT": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
}
ENABLE_NOTE_HAPTICS = True
BUTTON_HOLD_CLEAR_SECONDS = 5.0
MIN_OCTAVE_OFFSET = -4
MAX_OCTAVE_OFFSET = 4
STACKED_PLANE_OCTAVES = (-1, 0, 1)

NOTE_TO_SEMITONE = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "FB": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
    "CB": 11,
}
SEMITONE_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
SCALE_INTERVALS = {
    "chromatic": tuple(range(12)),
    "major": (0, 2, 4, 5, 7, 9, 11),
    "ionian": (0, 2, 4, 5, 7, 9, 11),
    "natural-minor": (0, 2, 3, 5, 7, 8, 10),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "aeolian": (0, 2, 3, 5, 7, 8, 10),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
    "harmonic-minor": (0, 2, 3, 5, 7, 8, 11),
    "melodic-minor": (0, 2, 3, 5, 7, 9, 11),
    "harmonic-major": (0, 2, 4, 5, 7, 8, 11),
    "double-harmonic": (0, 1, 4, 5, 7, 8, 11),
    "hungarian-minor": (0, 2, 3, 6, 7, 8, 11),
    "ukrainian-dorian": (0, 2, 3, 6, 7, 9, 10),
    "neapolitan-minor": (0, 1, 3, 5, 7, 8, 11),
    "neapolitan-major": (0, 1, 3, 5, 7, 9, 11),
    "enigmatic": (0, 1, 4, 6, 8, 10, 11),
    "persian": (0, 1, 4, 5, 6, 8, 11),
    "romanian-minor": (0, 2, 3, 6, 7, 9, 10),
    "spanish-gypsy": (0, 1, 4, 5, 7, 8, 10),
    "altered": (0, 1, 3, 4, 6, 8, 10),
    "whole-tone-plus": (0, 2, 4, 6, 8, 10, 11),
}


@dataclass
class CameraWorkerSnapshot:
    camera_id: int
    frame: np.ndarray | None = None
    mask: np.ndarray | None = None
    observations: list[mocap.MarkerObservation] | None = None
    timestamp: float = 0.0
    frame_number: int = 0
    delivered_fps: float = 0.0
    read_ms: float = 0.0
    detect_ms: float = 0.0
    error: str | None = None


@dataclass
class ImuPacketSnapshot:
    device_id: int
    hand_label: str
    timestamp_us: int
    sequence: int
    packet_type: int
    button_pressed: bool
    accel: np.ndarray | None
    quaternion: np.ndarray | None
    error_handler: int
    receive_time: float
    accel_receive_time: float | None = None
    quat_receive_time: float | None = None

    @property
    def has_quat(self) -> bool:
        return self.quaternion is not None

    @property
    def packet_has_quat(self) -> bool:
        return bool(self.packet_type & IMU_PACKET_HAS_QUAT)

    @property
    def has_accel(self) -> bool:
        return self.accel is not None

    @property
    def packet_has_accel(self) -> bool:
        return bool(self.packet_type & IMU_PACKET_HAS_ACCEL)

    def quat_age_seconds(self, now: float | None = None) -> float | None:
        if self.quat_receive_time is None:
            return None
        current_time = time.time() if now is None else now
        return max(current_time - self.quat_receive_time, 0.0)

    def has_fresh_quat(self, stale_seconds: float, now: float | None = None) -> bool:
        age = self.quat_age_seconds(now)
        return age is not None and age <= max(float(stale_seconds), 0.001)

    def accel_age_seconds(self, now: float | None = None) -> float | None:
        if self.accel_receive_time is None:
            return None
        current_time = time.time() if now is None else now
        return max(current_time - self.accel_receive_time, 0.0)


@dataclass
class ProcessedSceneState:
    timestamp: float
    frames: dict[int, np.ndarray]
    observations_by_camera: dict[int, list[mocap.MarkerObservation]]
    camera_snapshots: dict[int, CameraWorkerSnapshot]
    tracks: list[mocap.MarkerTrack]
    selected_tracks: list[mocap.MarkerTrack]
    tracked_hand_labels: set[str]
    hand_assignments: dict[str, int]
    hand_display_positions: dict[str, np.ndarray]
    measurement_count: int
    live_track_count: int
    used_exclusive_pairing: bool


class SerialImuReader(threading.Thread):
    def __init__(
        self,
        port: str | None,
        baud: int,
        stale_seconds: float,
        stop_event: threading.Event,
    ):
        super().__init__(daemon=True)
        self.requested_port = port
        self.baud = int(baud)
        self.stale_seconds = max(float(stale_seconds), 0.001)
        self.stop_event = stop_event
        self.lock = threading.Lock()
        self.snapshots_by_label: dict[str, ImuPacketSnapshot] = {}
        self.port_name: str | None = None
        self.serial_port = None
        self.error: str | None = None
        self.packet_count = 0

    def run(self) -> None:
        if serial is None:
            self.error = "pyserial is not installed"
            print("[playing test IMU] pyserial is not installed; serial IMU disabled")
            return

        port = self.requested_port or self._auto_detect_port()
        if port is None:
            self.error = "no serial port found"
            print("[playing test IMU] no serial port found; pass --imu-serial-port COMx")
            return

        self.port_name = port
        try:
            with serial.Serial(port, self.baud, timeout=0.02) as serial_port:
                self.serial_port = serial_port
                print(f"[playing test IMU] reading ESP-NOW receiver on {port} @ {self.baud}")
                self._read_loop(serial_port)
        except Exception as error:
            self.error = str(error)
            print(f"[playing test IMU] serial reader stopped: {error}")
        finally:
            self.serial_port = None

    def _auto_detect_port(self) -> str | None:
        if list_ports is None:
            return None

        ports = list(list_ports.comports())
        if not ports:
            return None

        preferred_keywords = (
            "usb",
            "uart",
            "cp210",
            "ch340",
            "wch",
            "silicon labs",
            "esp32",
        )
        for port_info in ports:
            text = " ".join(
                str(value).lower()
                for value in (
                    port_info.device,
                    port_info.description,
                    port_info.manufacturer,
                    port_info.hwid,
                )
                if value
            )
            if any(keyword in text for keyword in preferred_keywords):
                return str(port_info.device)

        return str(ports[0].device)

    def _read_loop(self, serial_port) -> None:
        buffer = bytearray()
        while not self.stop_event.is_set():
            chunk = serial_port.read(256)
            if chunk:
                buffer.extend(chunk)
                self._consume_buffer(buffer)

    def _consume_buffer(self, buffer: bytearray) -> None:
        while True:
            header_index = buffer.find(IMU_HEADER_BYTES)
            if header_index < 0:
                if buffer[-1:] == IMU_HEADER_BYTES[:1]:
                    del buffer[:-1]
                else:
                    buffer.clear()
                return
            if header_index > 0:
                del buffer[:header_index]
            if len(buffer) < IMU_PACKET_SIZE:
                return

            packet_bytes = bytes(buffer[:IMU_PACKET_SIZE])
            del buffer[:IMU_PACKET_SIZE]
            self._store_packet(packet_bytes)

    def _store_packet(self, packet_bytes: bytes) -> None:
        unpacked = IMU_PACKET_STRUCT.unpack(packet_bytes)
        (
            header,
            device_id,
            timestamp_us,
            sequence,
            packet_type,
            button_pressed,
            accel_x,
            accel_y,
            accel_z,
            quat_w,
            quat_i,
            quat_j,
            quat_k,
            error_handler,
        ) = unpacked
        if header != IMU_PACKET_HEADER:
            return

        hand_label = IMU_DEVICE_ID_TO_HAND.get(device_id)
        if hand_label is None:
            return

        receive_time = time.time()
        previous = self.snapshots_by_label.get(hand_label)
        accel = previous.accel if previous is not None else None
        quaternion = previous.quaternion if previous is not None else None
        accel_receive_time = previous.accel_receive_time if previous is not None else None
        quat_receive_time = previous.quat_receive_time if previous is not None else None
        if packet_type & IMU_PACKET_HAS_ACCEL:
            accel = np.array([accel_x, accel_y, accel_z], dtype=np.float64)
            accel_receive_time = receive_time
        if packet_type & IMU_PACKET_HAS_QUAT:
            quaternion = fl_debug.normalize_quat(
                np.array([quat_w, quat_i, quat_j, quat_k], dtype=np.float64)
            )
            quat_receive_time = receive_time

        snapshot = ImuPacketSnapshot(
            device_id=int(device_id),
            hand_label=hand_label,
            timestamp_us=int(timestamp_us),
            sequence=int(sequence),
            packet_type=int(packet_type),
            button_pressed=bool(button_pressed),
            accel=accel,
            quaternion=quaternion,
            error_handler=int(error_handler),
            receive_time=receive_time,
            accel_receive_time=accel_receive_time,
            quat_receive_time=quat_receive_time,
        )
        with self.lock:
            self.snapshots_by_label[hand_label] = snapshot
            self.packet_count += 1

    def snapshot_for_hand(self, hand_label: str) -> ImuPacketSnapshot | None:
        label = str(hand_label).upper()
        with self.lock:
            snapshot = self.snapshots_by_label.get(label)
        if snapshot is None:
            return None
        if time.time() - snapshot.receive_time > self.stale_seconds:
            return None
        return snapshot

    def _snapshot_status_part(self, snapshot: ImuPacketSnapshot, now: float) -> str:
        age_ms = (now - snapshot.receive_time) * 1000.0
        quat_age = snapshot.quat_age_seconds(now)
        if quat_age is None:
            quat_text = "none"
        elif quat_age <= self.stale_seconds:
            quat_text = f"LIVE {quat_age * 1000.0:.0f}ms"
        else:
            quat_text = f"STALE {quat_age * 1000.0:.0f}ms"

        accel_age = snapshot.accel_age_seconds(now)
        if accel_age is None:
            accel_text = "none"
        elif accel_age <= self.stale_seconds:
            accel_text = f"LIVE {accel_age * 1000.0:.0f}ms"
        else:
            accel_text = f"STALE {accel_age * 1000.0:.0f}ms"

        return (
            f"seq={snapshot.sequence} age={age_ms:.0f}ms "
            f"pkt_quat={'yes' if snapshot.packet_has_quat else 'no'} "
            f"quat={quat_text} "
            f"pkt_accel={'yes' if snapshot.packet_has_accel else 'no'} "
            f"accel={accel_text} "
            f"button={'down' if snapshot.button_pressed else 'up'} "
            f"err=0x{snapshot.error_handler:02X}"
        )

    def status_lines(self) -> list[str]:
        with self.lock:
            snapshots_by_label = dict(self.snapshots_by_label)
            packet_count = self.packet_count
        if self.error:
            return [f"imu {label}: error:{self.error}" for label in HAND_LABELS]
        if self.port_name is None:
            return [f"imu {label}: starting" for label in HAND_LABELS]

        lines = []
        now = time.time()
        for label in HAND_LABELS:
            snapshot = snapshots_by_label.get(label)
            if snapshot is None:
                detail = "waiting"
            elif now - snapshot.receive_time > self.stale_seconds:
                detail = f"STALE packet age={(now - snapshot.receive_time) * 1000.0:.0f}ms"
            else:
                detail = self._snapshot_status_part(snapshot, now)
            lines.append(f"imu={self.port_name} packets={packet_count} {label}: {detail}")
        return lines

    def status_text(self) -> str:
        return " | ".join(self.status_lines())

    def send_haptics_command(self, hand_label: str, intensity: int, duration_ms: int) -> bool:
        device_id = 1 if str(hand_label).upper() == "LEFT" else 2
        payload = HAPTICS_COMMAND_STRUCT.pack(
            HAPTICS_COMMAND_HEADER,
            int(device_id),
            int(np.clip(intensity, 0, 255)),
            int(np.clip(duration_ms, 1, 1000)),
        )
        with self.lock:
            serial_port = self.serial_port
            if serial_port is None or not serial_port.is_open:
                return False
            serial_port.write(payload)
        return True


class CameraProcessingWorker(threading.Thread):
    def __init__(
        self,
        source: mocap.CameraSource,
        detector: mocap.ReflectiveMarkerDetector,
        stop_event: threading.Event,
        build_preview_mask: bool,
    ):
        super().__init__(daemon=True)
        self.source = source
        self.detector = detector
        self.stop_event = stop_event
        self.build_preview_mask = build_preview_mask
        self.lock = threading.Lock()
        self.frame_intervals: deque[float] = deque(maxlen=120)
        self.snapshot_data = CameraWorkerSnapshot(
            camera_id=source.camera_id,
            observations=[],
        )

    def run(self) -> None:
        last_frame_time: float | None = None
        while not self.stop_event.is_set():
            read_start = time.perf_counter()
            ok, frame = self.source.read()
            read_end = time.perf_counter()
            timestamp = time.time()

            if not ok or frame is None:
                self._store_snapshot(
                    frame=None,
                    mask=None,
                    observations=[],
                    timestamp=timestamp,
                    read_ms=(read_end - read_start) * 1000.0,
                    detect_ms=0.0,
                    error="read failed",
                )
                time.sleep(0.001)
                continue

            detect_start = time.perf_counter()
            if self.build_preview_mask:
                observations, mask = detect_markers_and_mask(
                    frame,
                    self.detector.settings,
                    self.source.camera_id,
                    timestamp,
                )
            else:
                observations = self.detector.detect(frame, self.source.camera_id, timestamp)
                mask = None
            detect_end = time.perf_counter()

            if last_frame_time is not None:
                interval = timestamp - last_frame_time
                if interval > 0:
                    self.frame_intervals.append(interval)
            last_frame_time = timestamp

            self._store_snapshot(
                frame=frame,
                mask=mask,
                observations=observations,
                timestamp=timestamp,
                read_ms=(read_end - read_start) * 1000.0,
                detect_ms=(detect_end - detect_start) * 1000.0,
                error=None,
            )

    def _store_snapshot(
        self,
        frame: np.ndarray | None,
        mask: np.ndarray | None,
        observations: list[mocap.MarkerObservation],
        timestamp: float,
        read_ms: float,
        detect_ms: float,
        error: str | None,
    ) -> None:
        with self.lock:
            frame_number = self.snapshot_data.frame_number + (1 if frame is not None else 0)
            delivered_fps = 0.0
            if self.frame_intervals:
                delivered_fps = len(self.frame_intervals) / sum(self.frame_intervals)

            self.snapshot_data = CameraWorkerSnapshot(
                camera_id=self.source.camera_id,
                frame=frame,
                mask=mask,
                observations=list(observations),
                timestamp=timestamp,
                frame_number=frame_number,
                delivered_fps=delivered_fps,
                read_ms=read_ms,
                detect_ms=detect_ms,
                error=error,
            )

    def snapshot(self) -> CameraWorkerSnapshot:
        with self.lock:
            return self.snapshot_data


class KeyboardPoller:
    KEY_CODES = {
        "SPACE": 0x20,
        "C": 0x43,
        "R": 0x52,
        "ESCAPE": 0x1B,
    }

    def __init__(self):
        self.enabled = sys.platform == "win32"
        self.previous_states = {name: False for name in self.KEY_CODES}
        self.user32 = None
        if self.enabled:
            try:
                self.user32 = ctypes.windll.user32
            except Exception:
                self.enabled = False

    def update(self) -> tuple[set[str], set[str]]:
        if not self.enabled or self.user32 is None:
            return set(), set()

        pressed: set[str] = set()
        released: set[str] = set()
        for name, code in self.KEY_CODES.items():
            is_down = bool(self.user32.GetAsyncKeyState(code) & 0x8000)
            was_down = self.previous_states.get(name, False)
            if is_down and not was_down:
                pressed.add(name)
            elif was_down and not is_down:
                released.add(name)
            self.previous_states[name] = is_down
        return pressed, released


def normalize_scale_name(name: str) -> str:
    normalized = str(name).strip().lower().replace("_", "-").replace(" ", "-")
    if normalized not in SCALE_INTERVALS:
        available = ", ".join(sorted(SCALE_INTERVALS))
        raise argparse.ArgumentTypeError(f"Unknown scale {name!r}. Available: {available}")
    return normalized


def parse_midi_note(value: str | int) -> int:
    if isinstance(value, int):
        return int(np.clip(value, 0, 127))

    text = str(value).strip()
    if text.isdigit():
        return int(np.clip(int(text), 0, 127))

    note_part = text[:-1].upper()
    octave_part = text[-1:]
    if len(text) >= 3 and text[-2] == "-":
        note_part = text[:-2].upper()
        octave_part = text[-2:]
    if note_part not in NOTE_TO_SEMITONE:
        raise argparse.ArgumentTypeError(
            f"Expected MIDI note number or note name like C4, F#3, Bb4; got {value!r}"
        )
    try:
        octave = int(octave_part)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Expected octave number in note name {value!r}") from error
    return int(np.clip((octave + 1) * 12 + NOTE_TO_SEMITONE[note_part], 0, 127))


def midi_note_name(midi_note: int) -> str:
    note = int(np.clip(midi_note, 0, 127))
    return f"{SEMITONE_NAMES[note % 12]}{(note // 12) - 1}"


class GloveScalePlayingVisualizer(fl_debug.DualHandFLStudioVisualizer):
    def __init__(
        self,
        *args,
        scale_name: str = "chromatic",
        middle_note: int = fl_debug.MIDI_BASE_NOTE,
        button_hold_clear_seconds: float = BUTTON_HOLD_CLEAR_SECONDS,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scale_name = normalize_scale_name(scale_name)
        self.scale_intervals = SCALE_INTERVALS[self.scale_name]
        self.initial_midi_base_note = int(np.clip(middle_note, 0, 127))
        self.midi_base_note = self.initial_midi_base_note
        self.octave_offset = 0
        self.button_hold_clear_seconds = max(float(button_hold_clear_seconds), 0.25)
        self.button_press_times: dict[str, float | None] = {label: None for label in self.hand_labels}
        self.button_clear_consumed: dict[str, bool] = {label: False for label in self.hand_labels}
        self.selected_note_offset: int | None = None
        self.selected_note_name = "none"
        self.stacked_plane_meshes: dict[tuple[str, int], object] = {}
        self.stacked_plane_outlines: dict[tuple[str, int], object] = {}
        self.stacked_plane_section_lines: dict[tuple[str, int], list[object]] = {}

    @property
    def section_count(self) -> int:
        return len(self.scale_intervals)

    def scale_status_text(self) -> str:
        return (
            f"{self.scale_name} middle={midi_note_name(self.midi_base_note)} "
            f"octave={self.octave_offset:+d} selected={self.selected_note_name}"
        )

    def set_scale(self, scale_name: str) -> None:
        self._all_midi_notes_off()
        self.scale_name = normalize_scale_name(scale_name)
        self.scale_intervals = SCALE_INTERVALS[self.scale_name]
        for label in self.hand_labels:
            state = self.hands[label]
            if state.plane is not None:
                self._update_plane_visuals(label, state.plane, preview=False)
        print(f"[playing test scale] scale={self.scale_name} sections={self.section_count}")

    def set_middle_note(self, middle_note: int) -> None:
        self._all_midi_notes_off()
        self.initial_midi_base_note = int(np.clip(middle_note, 0, 127))
        self.octave_offset = 0
        self.midi_base_note = self.initial_midi_base_note
        print(f"[playing test scale] middle={midi_note_name(self.midi_base_note)} ({self.midi_base_note})")

    def adjust_octave(self, delta: int) -> None:
        next_offset = int(np.clip(self.octave_offset + int(delta), MIN_OCTAVE_OFFSET, MAX_OCTAVE_OFFSET))
        if next_offset == self.octave_offset:
            return
        self._update_midi_note("RIGHT", None)
        self.octave_offset = next_offset
        self.midi_base_note = int(np.clip(self.initial_midi_base_note + self.octave_offset * 12, 0, 127))
        print(
            f"[playing test scale] octave={self.octave_offset:+d} "
            f"middle={midi_note_name(self.midi_base_note)}"
        )

    def clear_plane(self, label: str | None = None) -> None:
        super().clear_plane(label)
        labels = self.hand_labels if label is None else (self._normalize_label(label),)
        for hand_label in labels:
            for octave_index in STACKED_PLANE_OCTAVES:
                key = (hand_label, octave_index)
                mesh = self.stacked_plane_meshes.get(key)
                outline = self.stacked_plane_outlines.get(key)
                if mesh is not None:
                    mesh.visible = False
                if outline is not None:
                    outline.visible = False
                for line in self.stacked_plane_section_lines.get(key, []):
                    line.visible = False
            if hand_label == "LEFT":
                self.selected_note_offset = None
                self.selected_note_name = "none"
            if hand_label == "RIGHT":
                self._update_midi_note("RIGHT", None)

    def _update_plane_state(self, label: str) -> None:
        state = self.hands[label]
        pressed_edge = state.button_pressed and not state.last_button_pressed
        released_edge = (not state.button_pressed) and state.last_button_pressed
        now = time.time()

        if state.plane is None:
            if pressed_edge:
                state.drawing = True
                state.draw_origin = state.position.copy()
                state.is_on_play_side = False
                state.active_note_index = None
                self._update_midi_note(label, None)
                print(f"[{label}] plane origin captured: {state.draw_origin}")

            if state.drawing and state.button_pressed and state.draw_origin is not None:
                definition = self._compute_debug_plane(state.draw_origin, state.position)
                if definition is not None:
                    self._update_plane_visuals(label, definition, preview=True)

            if released_edge and state.drawing:
                if state.draw_origin is not None:
                    definition = self._compute_debug_plane(state.draw_origin, state.position)
                    if definition is not None:
                        state.plane = definition
                        state.no_play_side_sign = -1.0
                        state.play_side_sign = 1.0
                        state.is_on_play_side = False
                        state.active_note_index = None
                        self._update_plane_visuals(label, definition, preview=False)
                        print(
                            f"[{label}] plane committed: center={definition.center}, "
                            f"normal={fl_debug.AXIS_NAMES[definition.normal_axis]}"
                        )
                    else:
                        print(f"[{label}] plane draw ignored: move farther before releasing button")
                state.drawing = False
                state.draw_origin = None
            state.last_button_pressed = state.button_pressed
            return

        if pressed_edge:
            self.button_press_times[label] = now
            self.button_clear_consumed[label] = False

        press_time = self.button_press_times.get(label)
        if (
            state.button_pressed
            and press_time is not None
            and not self.button_clear_consumed[label]
            and now - press_time >= self.button_hold_clear_seconds
        ):
            self.clear_plane(label)
            self.button_clear_consumed[label] = True
            state.drawing = False
            state.draw_origin = None
            print(f"[{label}] plane cleared after {self.button_hold_clear_seconds:.1f}s hold")

        if released_edge:
            if not self.button_clear_consumed[label]:
                self.adjust_octave(1 if label == "LEFT" else -1)
            self.button_press_times[label] = None
            self.button_clear_consumed[label] = False

        state.last_button_pressed = state.button_pressed

    def _stacked_plane_definition(
        self,
        definition: fl_debug.PlaneDefinition,
        octave_index: int,
    ) -> fl_debug.PlaneDefinition:
        note_axis, note_half, _other_axis, _other_half = self._note_axis_and_half_extent(definition)
        center = definition.center.copy()
        center[note_axis] += float(octave_index) * (2.0 * note_half)
        return fl_debug.PlaneDefinition(
            center=center,
            axis_u=definition.axis_u,
            axis_v=definition.axis_v,
            normal_axis=definition.normal_axis,
            half_u=definition.half_u,
            half_v=definition.half_v,
        )

    def _ensure_stacked_plane_visuals(self, label: str, octave_index: int) -> None:
        key = (label, octave_index)
        if key not in self.stacked_plane_meshes:
            mesh = fl_debug.scene.visuals.Mesh(
                vertices=np.zeros((4, 3), dtype=np.float32),
                faces=fl_debug.PLANE_FACES,
                color=(0.25, 0.85, 1.0, 0.0),
                shading=None,
                parent=self.view.scene,
            )
            mesh.visible = False
            self.stacked_plane_meshes[key] = mesh

        if key not in self.stacked_plane_outlines:
            outline = fl_debug.scene.visuals.Line(
                pos=np.zeros((2, 3), dtype=np.float32),
                color=(0.75, 0.95, 1.0, 0.0),
                width=2,
                method="gl",
                parent=self.view.scene,
            )
            outline.visible = False
            self.stacked_plane_outlines[key] = outline

        lines = self.stacked_plane_section_lines.setdefault(key, [])
        while len(lines) < self.section_count - 1:
            line = fl_debug.scene.visuals.Line(
                pos=np.zeros((2, 3), dtype=np.float32),
                color=(0.75, 0.95, 1.0, 0.0),
                width=1,
                method="gl",
                parent=self.view.scene,
            )
            line.visible = False
            lines.append(line)

    def _section_line_segments_for_count(
        self,
        definition: fl_debug.PlaneDefinition,
        section_count: int,
    ) -> list[np.ndarray]:
        segments = []
        note_axis, note_half, other_axis, other_half = self._note_axis_and_half_extent(definition)
        axis_min = definition.center[note_axis] - note_half
        for split_index in range(1, section_count):
            split_ratio = split_index / section_count
            split_coord = axis_min + split_ratio * (2.0 * note_half)
            p0 = definition.center.copy()
            p1 = definition.center.copy()
            p0[note_axis] = split_coord
            p1[note_axis] = split_coord
            p0[other_axis] = definition.center[other_axis] - other_half
            p1[other_axis] = definition.center[other_axis] + other_half
            segments.append(np.array([p0, p1], dtype=np.float32))
        return segments

    def _update_plane_visuals(self, label: str, definition: fl_debug.PlaneDefinition, preview: bool) -> None:
        colors_by_label = {
            "LEFT": {
                "fill": (0.30, 0.80, 1.00, 0.08 if preview else 0.16),
                "edge": (0.45, 0.88, 1.00, 0.55 if preview else 0.9),
                "split": (0.72, 0.93, 1.00, 0.35 if preview else 0.65),
            },
            "RIGHT": {
                "fill": (1.00, 0.48, 0.32, 0.08 if preview else 0.16),
                "edge": (1.00, 0.62, 0.45, 0.55 if preview else 0.9),
                "split": (1.00, 0.84, 0.70, 0.35 if preview else 0.65),
            },
        }
        colors = colors_by_label[label]

        for octave_index in STACKED_PLANE_OCTAVES:
            stacked = self._stacked_plane_definition(definition, octave_index)
            self._ensure_stacked_plane_visuals(label, octave_index)
            key = (label, octave_index)
            alpha_scale = 1.0 if octave_index == 0 else 0.55
            fill = (*colors["fill"][:3], colors["fill"][3] * alpha_scale)
            edge = (*colors["edge"][:3], colors["edge"][3] * alpha_scale)
            split = (*colors["split"][:3], colors["split"][3] * alpha_scale)

            vertices = self._plane_vertices(stacked) * fl_debug.POSITION_SCALE
            outline = np.vstack([vertices, vertices[0]])
            self.stacked_plane_meshes[key].set_data(vertices=vertices, faces=fl_debug.PLANE_FACES, color=fill)
            self.stacked_plane_meshes[key].visible = True
            self.stacked_plane_outlines[key].set_data(pos=outline, color=edge)
            self.stacked_plane_outlines[key].visible = True

            segments = self._section_line_segments_for_count(stacked, self.section_count)
            lines = self.stacked_plane_section_lines[key]
            for index, line in enumerate(lines):
                if index < len(segments):
                    line.set_data(pos=segments[index] * fl_debug.POSITION_SCALE, color=split)
                    line.visible = True
                else:
                    line.visible = False

    def _stacked_plane_position_metrics(
        self,
        position: np.ndarray,
        definition: fl_debug.PlaneDefinition,
    ) -> tuple[float, float, bool, int]:
        note_axis, note_half, other_axis, other_half = self._note_axis_and_half_extent(definition)
        axis_min = definition.center[note_axis] - note_half
        normalized = (position[note_axis] - axis_min) / (2.0 * note_half)
        if normalized < 0.0:
            octave_index = -1
            local_note = normalized + 1.0
        elif normalized >= 1.0:
            octave_index = 1
            local_note = normalized - 1.0
        else:
            octave_index = 0
            local_note = normalized

        other_min = definition.center[other_axis] - other_half
        other_normalized = (position[other_axis] - other_min) / (2.0 * other_half)
        inside = (
            octave_index in STACKED_PLANE_OCTAVES
            and 0.0 <= local_note <= 1.0
            and 0.0 <= other_normalized <= 1.0
        )
        return (
            float(np.clip(local_note, 0.0, 1.0) * 100.0),
            float(np.clip(other_normalized, 0.0, 1.0) * 100.0),
            inside,
            octave_index,
        )

    def _note_offset_from_left_position(self) -> tuple[int | None, str, float, int]:
        state = self.hands["LEFT"]
        if not state.tracking_active or state.plane is None:
            return None, "none", 0.0, 0
        note_pct, _other_pct, inside, octave_index = self._stacked_plane_position_metrics(state.position, state.plane)
        note_axis, _note_half, _other_axis, _other_half = self._note_axis_and_half_extent(state.plane)
        section_index = min(int((note_pct / 100.0) * self.section_count), self.section_count - 1)
        semitone_offset = self.scale_intervals[section_index] + (12 * octave_index)
        midi_note = int(np.clip(self.midi_base_note + semitone_offset, 0, 127))
        note_name = midi_note_name(midi_note)
        if not inside and self.require_inside_plane_to_play:
            return None, f"outside {note_name}", note_pct, octave_index
        return semitone_offset, f"{note_name} {fl_debug.AXIS_NAMES[note_axis]}={note_pct:5.1f}%", note_pct, octave_index

    def _update_note_state(self, label: str) -> None:
        if label == "LEFT":
            note_offset, note_name, note_pct, octave_index = self._note_offset_from_left_position()
            state = self.hands["LEFT"]
            if note_offset is not None and state.preview_note_index != note_offset:
                self.haptics.pulse("LEFT", fl_debug.HAPTICS_NOTE_INTENSITY, fl_debug.HAPTICS_NOTE_DURATION_MS)
            self.selected_note_offset = note_offset
            self.selected_note_name = note_name
            state.active_note_index = None
            state.preview_note_index = note_offset
            state.last_note_name = f"select={note_name} oct={octave_index:+d}"
            state.last_inside = note_offset is not None
            state.last_u_pct = note_pct
            self._update_midi_note("LEFT", None)
            return

        state = self.hands["RIGHT"]
        if not state.tracking_active:
            state.active_note_index = None
            state.preview_note_index = None
            state.last_note_name = "tracking lost"
            state.is_on_play_side = False
            self._update_midi_note("RIGHT", None)
            return

        if state.plane is None:
            state.active_note_index = None
            state.preview_note_index = None
            state.last_note_name = "inactive"
            self._update_midi_note("RIGHT", None)
            return

        _note_pct, other_pct, inside, octave_index = self._stacked_plane_position_metrics(state.position, state.plane)
        play_offset = float(state.position[0] - state.plane.center[0])
        if state.is_on_play_side:
            if play_offset <= fl_debug.PLAY_EXIT_THRESHOLD:
                state.is_on_play_side = False
        elif play_offset >= fl_debug.PLAY_ENTER_THRESHOLD:
            state.is_on_play_side = True

        can_play = (
            self.selected_note_offset is not None
            and state.is_on_play_side
            and (inside or not self.require_inside_plane_to_play)
        )
        state.active_note_index = self.selected_note_offset if can_play else None
        state.preview_note_index = self.selected_note_offset
        state.last_note_name = (
            f"PLAY {self.selected_note_name}"
            if can_play
            else f"ready={self.selected_note_name} trigger_oct={octave_index:+d}"
        )
        state.last_inside = inside
        state.last_offset = play_offset
        state.last_u_pct = other_pct
        self._update_midi_note("RIGHT", state.active_note_index)

    def _update_left_controls(self) -> None:
        state = self.hands["LEFT"]
        if not state.tracking_active:
            return

        volume_roll = float(np.clip(state.rotation_euler[0], 0.0, 180.0))
        volume_value = int(np.clip(round((volume_roll / 180.0) * 127.0), 0, 127))
        state.volume_value = volume_value
        self.current_attack_value = self.midi_velocity

        right_channel = self.midi_channels["RIGHT"]
        if state.volume_value != self.current_volume_value:
            self.current_volume_value = state.volume_value
            self._send_control_change(101, self.current_volume_value, right_channel)
            self._send_control_change(7, self.current_volume_value, right_channel)


def detect_markers_and_mask(
    frame: np.ndarray,
    settings: mocap.DetectionSettings,
    camera_id: int,
    timestamp: float,
) -> tuple[list[mocap.MarkerObservation], np.ndarray]:
    gray = frame if len(frame.shape) == 2 else mocap.cv2.cvtColor(frame, mocap.cv2.COLOR_BGR2GRAY)
    mask = build_mask_from_gray(gray, settings)
    contours, _hierarchy = mocap.cv2.findContours(
        mask,
        mocap.cv2.RETR_EXTERNAL,
        mocap.cv2.CHAIN_APPROX_SIMPLE,
    )
    observations: list[mocap.MarkerObservation] = []

    for contour in contours:
        area = float(mocap.cv2.contourArea(contour))
        if area < settings.min_area or area > settings.max_area:
            continue

        perimeter = float(mocap.cv2.arcLength(contour, True))
        if perimeter <= 1e-6:
            continue

        circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
        if circularity < settings.min_circularity:
            continue

        (x, y), radius = mocap.cv2.minEnclosingCircle(contour)
        if radius <= 1e-6:
            continue
        if radius < settings.min_radius_px or radius > settings.max_radius_px:
            continue

        _bx, _by, width, height = mocap.cv2.boundingRect(contour)
        aspect_ratio = max(width, height) / max(min(width, height), 1)
        if aspect_ratio > settings.max_aspect_ratio:
            continue

        fill_ratio = float(area / (math.pi * radius * radius))
        if fill_ratio < settings.min_fill_ratio:
            continue

        moments = mocap.cv2.moments(contour)
        if abs(moments["m00"]) <= 1e-6:
            center = np.array([x, y], dtype=np.float64)
        else:
            center = np.array(
                [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
                dtype=np.float64,
            )

        contour_mask = np.zeros(gray.shape, dtype=np.uint8)
        mocap.cv2.drawContours(contour_mask, [contour], -1, 255, thickness=mocap.cv2.FILLED)
        brightness = float(mocap.cv2.mean(gray, mask=contour_mask)[0])
        if brightness < settings.min_brightness:
            continue

        score = brightness * circularity * min(fill_ratio, 1.0) * math.sqrt(area)
        observations.append(
            mocap.MarkerObservation(
                camera_id=camera_id,
                pixel=center,
                radius_px=float(radius),
                area_px=area,
                circularity=circularity,
                brightness=brightness,
                score=float(score),
                timestamp=timestamp,
            )
        )

    observations.sort(key=lambda obs: obs.score, reverse=True)
    return observations[: settings.max_markers_per_camera], mask


def build_mask_from_gray(gray: np.ndarray, settings: mocap.DetectionSettings) -> np.ndarray:
    processed = mocap.preprocess_gray(gray, settings.blur_kernel)
    threshold = mocap.choose_threshold(processed, settings)
    _ok, mask = mocap.cv2.threshold(processed, threshold, 255, mocap.cv2.THRESH_BINARY)

    kernel_size = settings.morphology_kernel
    if kernel_size > 1:
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = mocap.cv2.morphologyEx(mask, mocap.cv2.MORPH_OPEN, kernel)
        mask = mocap.cv2.morphologyEx(mask, mocap.cv2.MORPH_CLOSE, kernel)

    return mask


def binary_preview_from_mask(
    mask: np.ndarray | None,
    observations: list[mocap.MarkerObservation],
    camera_id: int,
    width: int,
    height: int,
) -> np.ndarray:
    if mask is None:
        return combined.blank_panel(f"camera {camera_id} binary", width, height)

    preview = mocap.cv2.cvtColor(mask, mocap.cv2.COLOR_GRAY2BGR)
    for index, observation in enumerate(observations, start=1):
        center = tuple(int(round(value)) for value in observation.pixel)
        radius = max(3, int(round(observation.radius_px)))
        mocap.cv2.circle(preview, center, radius, (0, 255, 0), 2)
        mocap.cv2.putText(
            preview,
            str(index),
            (center[0] + 8, center[1] - 8),
            mocap.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            mocap.cv2.LINE_AA,
        )
    combined.draw_panel_title(preview, f"camera {camera_id} binary")
    return combined.resize_panel(preview, width, height)


def build_threaded_combined_preview(
    preview_camera_ids: list[int],
    frames: dict[int, np.ndarray],
    observations_by_camera: dict[int, list[mocap.MarkerObservation]],
    tracks: list[mocap.MarkerTrack],
    camera_snapshots: dict[int, CameraWorkerSnapshot],
    track_memory_pixels: float,
    panel_width: int,
    panel_height: int,
) -> np.ndarray:
    panels: list[np.ndarray] = []
    for camera_id in preview_camera_ids[:2]:
        frame = frames.get(camera_id)
        observations = observations_by_camera.get(camera_id, [])
        snapshot = camera_snapshots.get(camera_id)
        if frame is None:
            raw_panel = combined.blank_panel(f"camera {camera_id} tracked", panel_width, panel_height)
        else:
            raw_panel = mocap.draw_preview(
                frame,
                observations,
                tracks,
                camera_id,
                track_memory_pixels,
            )
            combined.draw_panel_title(raw_panel, f"camera {camera_id} tracked")
            raw_panel = combined.resize_panel(raw_panel, panel_width, panel_height)

        binary_panel = binary_preview_from_mask(
            snapshot.mask if snapshot is not None else None,
            observations,
            camera_id,
            panel_width,
            panel_height,
        )
        panels.append(raw_panel)
        panels.append(binary_panel)

    while len(panels) < 4:
        panels.append(combined.blank_panel("unused", panel_width, panel_height))

    top_row = mocap.cv2.hconcat([panels[0], panels[1]])
    bottom_row = mocap.cv2.hconcat([panels[2], panels[3]])
    return mocap.cv2.vconcat([top_row, bottom_row])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = combined.build_arg_parser()
    parser.description = (
        "Track two mocap markers and feed their positions into the FL Studio "
        "debug visualizer hand/plane/MIDI logic."
    )
    parser.set_defaults(
        calibration=str(DEFAULT_PLAYING_TEST_CALIBRATION_PATH),
        tracked_point_count=2,
        update_hz=120.0,
    )
    parser.add_argument("--preview-hz", type=float, default=DEFAULT_PREVIEW_HZ)
    parser.add_argument("--visualizer-hz", type=float, default=DEFAULT_VISUALIZER_HZ)
    parser.add_argument(
        "--hand-side-axis",
        choices=("x", "y", "z"),
        default=HAND_SIDE_AXIS,
        help="Mocap axis used to decide left hand vs right hand.",
    )
    parser.add_argument(
        "--right-hand-higher-on-axis",
        action="store_true",
        dest="right_hand_higher_on_axis",
        default=RIGHT_HAND_HIGHER_ON_SIDE_AXIS,
        help="Treat the larger value on --hand-side-axis as the right hand.",
    )
    parser.add_argument(
        "--right-hand-lower-on-axis",
        action="store_false",
        dest="right_hand_higher_on_axis",
        help="Treat the smaller value on --hand-side-axis as the right hand.",
    )
    parser.add_argument(
        "--controlled-hand",
        choices=HAND_LABELS,
        default=CONTROLLED_HAND_LABEL,
        help="Only this hand responds to keyboard drawing/clear controls for now.",
    )
    parser.add_argument(
        "--use-serial-imu-rotation",
        action="store_true",
        default=USE_SERIAL_IMU_ROTATION,
        help="Read receiver ESP32 binary packets and use IMU quaternions/buttons for hand rotation and plane drawing.",
    )
    parser.add_argument(
        "--no-serial-imu-rotation",
        action="store_false",
        dest="use_serial_imu_rotation",
        help="Disable receiver ESP32 serial IMU input.",
    )
    parser.add_argument(
        "--imu-serial-port",
        default=IMU_SERIAL_PORT,
        help="Receiver ESP32 serial port, for example COM5. Default tries to auto-detect.",
    )
    parser.add_argument("--imu-serial-baud", type=int, default=IMU_SERIAL_BAUD)
    parser.add_argument(
        "--imu-packet-stale-seconds",
        type=float,
        default=IMU_PACKET_STALE_SECONDS,
        help="Ignore IMU packets older than this many seconds.",
    )
    parser.add_argument(
        "--imu-plane-draw-hand",
        choices=HAND_LABELS,
        default=IMU_PLANE_DRAW_HAND_LABEL,
        help="Legacy option; glove buttons now draw their own hand's plane.",
    )
    parser.add_argument(
        "--imu-channel-cycle-hand",
        choices=HAND_LABELS,
        default=IMU_CHANNEL_CYCLE_HAND_LABEL,
        help="Legacy option; glove buttons now control octave after planes exist.",
    )
    parser.add_argument(
        "--imu-channel-cycle-target-hand",
        choices=(*HAND_LABELS, "CONTROLLED"),
        default=IMU_CHANNEL_CYCLE_TARGET_HAND_LABEL,
        help="Legacy option retained for CLI compatibility.",
    )
    parser.add_argument("--no-midi", action="store_true")
    parser.add_argument("--midi-output-hint", default=fl_debug.MIDI_OUTPUT_HINT)
    parser.add_argument("--midi-base-note", type=int, default=fl_debug.MIDI_BASE_NOTE)
    parser.add_argument(
        "--middle-note",
        default=None,
        help="Middle stacked plane base note as MIDI number or note name, for example C4, D3, F#4.",
    )
    parser.add_argument(
        "--scale",
        type=normalize_scale_name,
        default="chromatic",
        help="Scale for note selection. Use --list-scales to see options.",
    )
    parser.add_argument("--list-scales", action="store_true", help="Print available scales and exit.")
    parser.add_argument("--midi-velocity", type=int, default=fl_debug.MIDI_VELOCITY)
    parser.add_argument(
        "--left-midi-channel",
        type=int,
        default=fl_debug.HAND_MIDI_CHANNELS_1_BASED["LEFT"],
    )
    parser.add_argument(
        "--right-midi-channel",
        type=int,
        default=fl_debug.HAND_MIDI_CHANNELS_1_BASED["RIGHT"],
    )
    parser.add_argument("--max-midi-channel", type=int, default=fl_debug.DEFAULT_MAX_MIDI_CHANNEL)
    parser.add_argument("--no-haptics", action="store_true")
    parser.add_argument(
        "--require-inside-plane",
        action="store_true",
        help="Require the marker to stay inside the drawn plane bounds before notes play.",
    )
    return parser


class MocapPlayingTestApp:
    def __init__(
        self,
        args: argparse.Namespace,
        sources: list[mocap.CameraSource],
        calibrations: dict[int, mocap.CameraCalibration],
    ):
        self.args = args
        self.sources = sources
        self.calibrations = calibrations
        self.preview_camera_ids = [source.camera_id for source in sources[:2]]
        self.settings_by_camera = {
            source.camera_id: mocap.build_detection_settings(args, source.camera_id)
            for source in sources
        }
        self.detectors = {
            camera_id: mocap.ReflectiveMarkerDetector(settings)
            for camera_id, settings in self.settings_by_camera.items()
        }
        self.triangulator = mocap.MultiCameraTriangulator(
            calibrations=calibrations,
            max_pair_error_px=args.max_reprojection_error,
            cluster_distance_m=args.cluster_distance,
            room_bounds=args.room_bounds,
        )
        self.tracker = mocap.MarkerTracker(
            max_match_distance_m=args.track_distance,
            max_missing_frames=args.max_missing_frames,
            min_confirmed_hits=args.track_confirmation_hits,
            max_tentative_missing_frames=args.tentative_max_missing_frames,
            duplicate_track_distance_m=args.duplicate_track_distance,
            velocity_damping=args.velocity_damping,
            stationary_distance_m=args.stationary_distance,
            max_prediction_dt=args.max_prediction_dt,
        )
        self.display_track_ids: list[int] = []
        self.last_measurement_count = 0
        self.last_live_track_count = 0
        self.last_used_exclusive_pairing = False
        self.last_print_time = 0.0
        self.last_preview_draw_time = 0.0
        self.last_visualizer_update_time = 0.0
        self.last_processed_frame_numbers: dict[int, int] = {}
        self.track_id_to_hand_label: dict[int, str] = {}
        self.last_hand_assignments: dict[str, int] = {}
        self.last_tracked_hand_labels: set[str] = set()
        self.display_positions_by_track: dict[int, np.ndarray] = {}
        self.cv2_space_button = False
        self.last_imu_channel_cycle_button_pressed = False
        self.closed = False
        self.processing_lock = threading.Lock()
        self.latest_scene_state: ProcessedSceneState | None = None

        self.keyboard = KeyboardPoller()
        self.stop_event = threading.Event()
        self.imu_reader: SerialImuReader | None = None
        self.camera_workers = [
            CameraProcessingWorker(
                source=source,
                detector=self.detectors[source.camera_id],
                stop_event=self.stop_event,
                build_preview_mask=not args.no_preview,
            )
            for source in self.sources
        ]

        middle_note = parse_midi_note(args.middle_note) if args.middle_note is not None else int(args.midi_base_note)
        self.visualizer = GloveScalePlayingVisualizer(
            keyboard_controlled=False,
            keyboard_buttons_enabled=True,
            enable_midi=not args.no_midi,
            midi_output_hint=args.midi_output_hint,
            midi_base_note=middle_note,
            midi_velocity=args.midi_velocity,
            midi_channels_1_based={"LEFT": args.left_midi_channel, "RIGHT": args.right_midi_channel},
            max_midi_channel=args.max_midi_channel,
            enable_haptics=False,
            require_inside_plane_to_play=args.require_inside_plane,
            controlled_hand_label=args.controlled_hand,
            allow_hand_switching=False,
            update_hz=args.visualizer_hz,
            start_timer=False,
            show=True,
            scale_name=args.scale,
            middle_note=middle_note,
        )
        self.visualizer.canvas.events.close.connect(self.close)

        self._setup_combined_preview_window()
        for worker in self.camera_workers:
            worker.start()
        print(f"[playing test] started {len(self.camera_workers)} camera processing threads")
        if args.use_serial_imu_rotation:
            self.imu_reader = SerialImuReader(
                args.imu_serial_port,
                args.imu_serial_baud,
                args.imu_packet_stale_seconds,
                self.stop_event,
            )
            self.imu_reader.start()
            self.visualizer.haptics = HapticsController(
                enabled=(ENABLE_NOTE_HAPTICS and not args.no_haptics),
                send_func=self.imu_reader.send_haptics_command,
            )
            print(f"[playing test] IMU plane draw button is mapped to {args.imu_plane_draw_hand}")
            print(
                "[playing test] IMU channel cycle button is mapped to "
                f"{args.imu_channel_cycle_hand} -> {args.imu_channel_cycle_target_hand}"
            )

        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            name="playing-test-processing",
            daemon=True,
        )
        self.processing_thread.start()

        ui_update_hz = max(
            float(args.visualizer_hz),
            60.0,
            0.0 if args.no_preview else float(args.preview_hz),
        )
        self.timer = fl_debug.app.Timer(
            interval=max(1.0 / max(ui_update_hz, 1.0), 0.001),
            connect=self.update,
            start=True,
        )

    def _setup_combined_preview_window(self) -> None:
        if self.args.no_preview:
            return
        mocap.cv2.namedWindow(combined.DEFAULT_COMBINED_WINDOW_NAME, mocap.cv2.WINDOW_NORMAL)
        mocap.cv2.resizeWindow(
            combined.DEFAULT_COMBINED_WINDOW_NAME,
            int(self.args.panel_width) * 2,
            int(self.args.panel_height) * 2,
        )

    def update(self, _event) -> None:
        if self.closed:
            return

        timestamp = time.time()
        self._handle_polled_keyboard()
        scene_state = self._latest_scene_state_snapshot()
        if scene_state is not None:
            self._apply_scene_state_to_visualizer(scene_state)
        self._update_visualizer_if_due(timestamp)
        if scene_state is not None:
            self._print_status_from_scene_state(scene_state)

        if not self.args.no_preview and scene_state is not None:
            preview_interval = 1.0 / max(float(self.args.preview_hz), 1.0)
            if timestamp - self.last_preview_draw_time >= preview_interval:
                self.last_preview_draw_time = timestamp
                preview = build_threaded_combined_preview(
                    self.preview_camera_ids,
                    scene_state.frames,
                    scene_state.observations_by_camera,
                    scene_state.tracks,
                    scene_state.camera_snapshots,
                    self.args.track_memory_pixels,
                    int(self.args.panel_width),
                    int(self.args.panel_height),
                )
                mocap.cv2.imshow(combined.DEFAULT_COMBINED_WINDOW_NAME, preview)
        self._pump_preview_keyboard()

    def _processing_loop(self) -> None:
        interval = 1.0 / max(float(self.args.update_hz), 1.0)
        while not self.stop_event.is_set():
            loop_start = time.perf_counter()
            timestamp = time.time()
            frames, observations_by_camera, camera_snapshots = self._collect_camera_snapshots()
            if self._has_new_camera_data(camera_snapshots):
                self._mark_camera_data_processed(camera_snapshots)

                mocap.lock_observations_to_existing_tracks(
                    frames,
                    observations_by_camera,
                    self.tracker.tracks,
                    self.settings_by_camera,
                    self.args.track_memory_pixels,
                    timestamp,
                )
                measurements = self._triangulate_measurements(observations_by_camera, timestamp)
                tracks = self.tracker.update(measurements, timestamp)
                live_tracks = [
                    track for track in tracks if track.confirmed and track.missing_frames == 0
                ]
                selected_tracks = self._select_display_tracks(live_tracks)
                self._store_scene_state(
                    timestamp,
                    frames,
                    observations_by_camera,
                    camera_snapshots,
                    tracks,
                    selected_tracks,
                    len(measurements),
                    len(live_tracks),
                    self.last_used_exclusive_pairing,
                )

            elapsed = time.perf_counter() - loop_start
            remaining = interval - elapsed
            if remaining > 0.0:
                self.stop_event.wait(min(remaining, 0.01))
            else:
                time.sleep(0.001)

    def _collect_camera_snapshots(
        self,
    ) -> tuple[
        dict[int, np.ndarray],
        dict[int, list[mocap.MarkerObservation]],
        dict[int, CameraWorkerSnapshot],
    ]:
        frames: dict[int, np.ndarray] = {}
        observations_by_camera: dict[int, list[mocap.MarkerObservation]] = {}
        snapshots: dict[int, CameraWorkerSnapshot] = {}

        for worker in self.camera_workers:
            snapshot = worker.snapshot()
            snapshots[snapshot.camera_id] = snapshot
            observations_by_camera[snapshot.camera_id] = snapshot.observations or []
            if snapshot.frame is not None:
                frames[snapshot.camera_id] = snapshot.frame

        return frames, observations_by_camera, snapshots

    def _triangulate_measurements(
        self,
        observations_by_camera: dict[int, list[mocap.MarkerObservation]],
        timestamp: float,
    ) -> list[mocap.MarkerMeasurement]:
        self.last_used_exclusive_pairing = False
        visible_calibrated_camera_ids = [
            source.camera_id
            for source in self.sources
            if source.camera_id in self.calibrations
            and observations_by_camera.get(source.camera_id)
        ]
        if self.args.exclusive_pairing and len(visible_calibrated_camera_ids) == 2:
            expected_count = self._expected_measurement_count(observations_by_camera)
            exclusive_measurements = combined.triangulate_exclusive_two_camera_pairs(
                observations_by_camera,
                self.calibrations,
                [source.camera_id for source in self.sources],
                self.args.room_bounds,
                self.args.max_reprojection_error,
                expected_count,
                self.args.min_measurement_separation,
                reference_positions=self._pairing_reference_positions(timestamp),
                track_bias_distance_m=self.args.pairing_track_bias_distance,
            )
            if exclusive_measurements:
                self.last_used_exclusive_pairing = True
                return exclusive_measurements

        measurements = self.triangulator.triangulate(observations_by_camera)
        if not self.args.exclusive_pairing:
            return measurements

        expected_count = self._expected_measurement_count(observations_by_camera)
        if len(measurements) >= expected_count:
            return measurements

        exclusive_measurements = combined.triangulate_exclusive_two_camera_pairs(
            observations_by_camera,
            self.calibrations,
            [source.camera_id for source in self.sources],
            self.args.room_bounds,
            self.args.max_reprojection_error,
            expected_count,
            self.args.min_measurement_separation,
            reference_positions=self._pairing_reference_positions(timestamp),
            track_bias_distance_m=self.args.pairing_track_bias_distance,
        )
        if len(exclusive_measurements) > len(measurements):
            self.last_used_exclusive_pairing = True
            return exclusive_measurements
        return measurements

    def _pairing_reference_positions(self, timestamp: float) -> list[np.ndarray]:
        max_tracks = max(min(int(self.args.tracked_point_count), len(HAND_LABELS)), 1)
        live_reference_tracks = [
            track
            for track in self.tracker.tracks
            if track.confirmed and track.missing_frames <= self.args.max_missing_frames
        ]
        tracks_by_id = {track.track_id: track for track in live_reference_tracks}
        ordered_tracks = [
            tracks_by_id[track_id]
            for track_id in self.display_track_ids
            if track_id in tracks_by_id
        ]
        ordered_track_ids = {track.track_id for track in ordered_tracks}
        ordered_tracks.extend(
            track
            for track in sorted(live_reference_tracks, key=lambda item: item.track_id)
            if track.track_id not in ordered_track_ids
        )
        return [
            self.tracker._predicted_position(track, timestamp)
            for track in ordered_tracks[:max_tracks]
        ]

    def _expected_measurement_count(
        self,
        observations_by_camera: dict[int, list[mocap.MarkerObservation]],
    ) -> int:
        calibrated_counts = [
            len(observations_by_camera.get(source.camera_id, []))
            for source in self.sources
            if source.camera_id in self.calibrations
        ]
        if len(calibrated_counts) < 2:
            return 1
        return max(1, min(int(self.args.tracked_point_count), *calibrated_counts[:2]))

    def _select_display_tracks(
        self,
        live_tracks: list[mocap.MarkerTrack],
    ) -> list[mocap.MarkerTrack]:
        max_tracks = max(min(int(self.args.tracked_point_count), len(HAND_LABELS)), 1)
        live_by_id = {track.track_id: track for track in live_tracks}
        selected_ids = [
            track_id
            for track_id in self.display_track_ids
            if track_id in live_by_id
        ][:max_tracks]

        if len(selected_ids) < max_tracks:
            for track in sorted(live_tracks, key=lambda item: item.track_id):
                if track.track_id in selected_ids:
                    continue
                selected_ids.append(track.track_id)
                if len(selected_ids) >= max_tracks:
                    break

        self.display_track_ids = selected_ids
        return [live_by_id[track_id] for track_id in selected_ids]

    def _store_scene_state(
        self,
        timestamp: float,
        frames: dict[int, np.ndarray],
        observations_by_camera: dict[int, list[mocap.MarkerObservation]],
        camera_snapshots: dict[int, CameraWorkerSnapshot],
        tracks: list[mocap.MarkerTrack],
        selected_tracks: list[mocap.MarkerTrack],
        measurement_count: int,
        live_track_count: int,
        used_exclusive_pairing: bool,
    ) -> None:
        scale = np.array(
            [
                float(self.args.x_scaling_factor),
                float(self.args.y_scaling_factor),
                float(self.args.z_scaling_factor),
            ],
            dtype=np.float64,
        )
        assigned_tracks = self._assign_tracks_to_hands(selected_tracks)
        self._prune_display_positions({track.track_id for track in selected_tracks})
        hand_assignments = {
            label: track.track_id
            for label, track in assigned_tracks.items()
        }
        tracked_hand_labels = set(assigned_tracks)
        hand_display_positions = {
            label: self._smoothed_display_position(track) * scale
            for label, track in assigned_tracks.items()
        }

        scene_state = ProcessedSceneState(
            timestamp=timestamp,
            frames=frames,
            observations_by_camera=observations_by_camera,
            camera_snapshots=camera_snapshots,
            tracks=tracks,
            selected_tracks=selected_tracks,
            tracked_hand_labels=tracked_hand_labels,
            hand_assignments=hand_assignments,
            hand_display_positions=hand_display_positions,
            measurement_count=measurement_count,
            live_track_count=live_track_count,
            used_exclusive_pairing=used_exclusive_pairing,
        )
        with self.processing_lock:
            self.latest_scene_state = scene_state

    def _latest_scene_state_snapshot(self) -> ProcessedSceneState | None:
        with self.processing_lock:
            return self.latest_scene_state

    def _apply_scene_state_to_visualizer(self, scene_state: ProcessedSceneState) -> None:
        self.last_measurement_count = scene_state.measurement_count
        self.last_live_track_count = scene_state.live_track_count
        self.last_used_exclusive_pairing = scene_state.used_exclusive_pairing
        self.last_hand_assignments = dict(scene_state.hand_assignments)
        self.last_tracked_hand_labels = set(scene_state.tracked_hand_labels)

        for label in HAND_LABELS:
            display_position = scene_state.hand_display_positions.get(label)
            if display_position is None:
                self.visualizer.set_hand_tracking_active(label, False)
                continue
            self.visualizer.set_hand_pose(
                label,
                display_position,
                rotation_quaternion=None,
                button_pressed=None,
            )
            self.visualizer.set_hand_tracking_active(label, True)
        self._feed_imu_to_visualizer(scene_state.tracked_hand_labels)

    def _feed_tracks_to_visualizer(self, selected_tracks: list[mocap.MarkerTrack]) -> None:
        scale = np.array(
            [
                float(self.args.x_scaling_factor),
                float(self.args.y_scaling_factor),
                float(self.args.z_scaling_factor),
            ],
            dtype=np.float64,
        )
        assigned_tracks = self._assign_tracks_to_hands(selected_tracks)
        self._prune_display_positions({track.track_id for track in selected_tracks})
        self.last_hand_assignments = {
            label: track.track_id
            for label, track in assigned_tracks.items()
        }
        self.last_tracked_hand_labels = set(assigned_tracks)

        for label in HAND_LABELS:
            track = assigned_tracks.get(label)
            if track is None:
                self.visualizer.set_hand_tracking_active(label, False)
                continue
            display_position = self._smoothed_display_position(track)
            self.visualizer.set_hand_pose(
                label,
                display_position * scale,
                rotation_quaternion=None,
                button_pressed=None,
            )
            self.visualizer.set_hand_tracking_active(label, True)
        self._feed_imu_to_visualizer(set(assigned_tracks))

    def _feed_imu_to_visualizer(self, tracked_hand_labels: set[str]) -> None:
        now = time.time()
        for label in HAND_LABELS:
            imu_snapshot = self._imu_snapshot_for_hand(label)
            if imu_snapshot is None:
                self.visualizer.set_hand_imu_state(label, button_pressed=False)
                continue

            rotation_quaternion = (
                self._correct_imu_quaternion(label, imu_snapshot.quaternion)
                if imu_snapshot.has_fresh_quat(self.args.imu_packet_stale_seconds)
                else None
            )
            button_pressed = self._fresh_imu_button_pressed(
                label,
                imu_snapshot,
                now,
                tracked_hand_labels,
                require_tracking=True,
            )
            self.visualizer.set_hand_imu_state(
                label,
                rotation_quaternion=rotation_quaternion,
                acceleration=imu_snapshot.accel if imu_snapshot.has_accel else None,
                button_pressed=button_pressed,
            )

    def _fresh_imu_button_pressed(
        self,
        label: str,
        imu_snapshot: ImuPacketSnapshot,
        now: float,
        tracked_hand_labels: set[str],
        require_tracking: bool,
    ) -> bool:
        button_is_fresh = (now - imu_snapshot.receive_time) <= self.args.imu_packet_stale_seconds
        if require_tracking and label not in tracked_hand_labels:
            return False
        return bool(
            button_is_fresh
            and (imu_snapshot.packet_type & IMU_PACKET_HAS_BUTTON)
            and imu_snapshot.button_pressed
        )

    def _channel_cycle_target_hand(self) -> str:
        target = str(self.args.imu_channel_cycle_target_hand).upper()
        if target == "CONTROLLED":
            return self.visualizer.controlled_hand_label
        return target

    def _correct_imu_quaternion(self, label: str, quaternion: np.ndarray) -> np.ndarray:
        signs = IMU_QUATERNION_COMPONENT_SIGNS.get(label, np.ones(4, dtype=np.float64))
        corrected = fl_debug.normalize_quat(np.asarray(quaternion, dtype=np.float64) * signs)
        offset = IMU_MOUNTING_QUATERNION_OFFSETS.get(
            label,
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )
        return fl_debug.normalize_quat(fl_debug.quaternion_multiply(corrected, offset))

    def _imu_snapshot_for_hand(self, label: str) -> ImuPacketSnapshot | None:
        if self.imu_reader is None:
            return None
        return self.imu_reader.snapshot_for_hand(label)

    def _smoothed_display_position(self, track: mocap.MarkerTrack) -> np.ndarray:
        raw_position = np.asarray(track.position, dtype=np.float64)
        smoothing = float(np.clip(self.args.visual_smoothing, 0.0, 0.98))
        previous_position = self.display_positions_by_track.get(track.track_id)
        if previous_position is None or smoothing <= 0.0:
            display_position = raw_position.copy()
        else:
            display_position = smoothing * previous_position + (1.0 - smoothing) * raw_position
        self.display_positions_by_track[track.track_id] = display_position
        return display_position

    def _prune_display_positions(self, valid_track_ids: set[int]) -> None:
        for track_id in list(self.display_positions_by_track):
            if track_id not in valid_track_ids:
                self.display_positions_by_track.pop(track_id, None)

    def _assign_tracks_to_hands(
        self,
        selected_tracks: list[mocap.MarkerTrack],
    ) -> dict[str, mocap.MarkerTrack]:
        if not selected_tracks:
            self.track_id_to_hand_label = {}
            return {}

        axis_index = self._hand_side_axis_index()
        right_is_higher = bool(self.args.right_hand_higher_on_axis)
        assignments: dict[str, mocap.MarkerTrack] = {}

        if len(selected_tracks) >= 2:
            ordered = sorted(selected_tracks, key=lambda track: float(track.position[axis_index]))
            lower_track = ordered[0]
            higher_track = ordered[-1]
            if right_is_higher:
                assignments["RIGHT"] = higher_track
                assignments["LEFT"] = lower_track
            else:
                assignments["RIGHT"] = lower_track
                assignments["LEFT"] = higher_track
        else:
            track = selected_tracks[0]
            previous_label = self.track_id_to_hand_label.get(track.track_id)
            if previous_label in HAND_LABELS:
                assignments[previous_label] = track
            else:
                axis_value = float(track.position[axis_index])
                if (axis_value >= 0.0 and right_is_higher) or (axis_value < 0.0 and not right_is_higher):
                    assignments["RIGHT"] = track
                else:
                    assignments["LEFT"] = track

        self.track_id_to_hand_label = {
            track.track_id: label
            for label, track in assignments.items()
        }
        return assignments

    def _hand_side_axis_index(self) -> int:
        return {"x": 0, "y": 1, "z": 2}[str(self.args.hand_side_axis).lower()]

    def _has_new_camera_data(self, camera_snapshots: dict[int, CameraWorkerSnapshot]) -> bool:
        for camera_id, snapshot in camera_snapshots.items():
            if snapshot.frame_number <= 0:
                continue
            if self.last_processed_frame_numbers.get(camera_id) != snapshot.frame_number:
                return True
        return False

    def _mark_camera_data_processed(
        self,
        camera_snapshots: dict[int, CameraWorkerSnapshot],
    ) -> None:
        for camera_id, snapshot in camera_snapshots.items():
            self.last_processed_frame_numbers[camera_id] = snapshot.frame_number

    def _handle_polled_keyboard(self) -> None:
        pressed, released = self.keyboard.update()
        if "ESCAPE" in pressed:
            self.visualizer.canvas.close()
            return
        if "C" in pressed or "R" in pressed:
            self.visualizer.clear_plane(self.visualizer.controlled_hand_label)
        if "SPACE" in pressed:
            self.visualizer.set_hand_button(self.visualizer.controlled_hand_label, True)
        if "SPACE" in released:
            self.visualizer.set_hand_button(self.visualizer.controlled_hand_label, False)

    def _pump_preview_keyboard(self) -> None:
        if self.args.no_preview:
            return
        key = mocap.cv2.waitKey(1) & 0xFF
        self._handle_cv2_key(key)

    def _handle_cv2_key(self, key: int) -> None:
        if key in (255, -1):
            return
        if key == 27:
            self.visualizer.canvas.close()
        elif key in (ord("c"), ord("r")):
            self.visualizer.clear_plane(self.visualizer.controlled_hand_label)
        elif key == ord(" ") and not self.keyboard.enabled:
            self.cv2_space_button = not self.cv2_space_button
            self.visualizer.set_hand_button(
                self.visualizer.controlled_hand_label,
                self.cv2_space_button,
            )

    def _update_visualizer_if_due(self, timestamp: float, force: bool = False) -> None:
        interval = 1.0 / max(float(self.args.visualizer_hz), 1.0)
        if force or timestamp - self.last_visualizer_update_time >= interval:
            self.last_visualizer_update_time = timestamp
            self._feed_imu_to_visualizer(self.last_tracked_hand_labels)
            self.visualizer.set_external_status(
                [f"scale: {self.visualizer.scale_status_text()}", *self._imu_status_lines()]
            )
            self.visualizer.update(None)

    def _print_status(
        self,
        timestamp: float,
        tracks: list[mocap.MarkerTrack],
        observations_by_camera: dict[int, list[mocap.MarkerObservation]],
        camera_snapshots: dict[int, CameraWorkerSnapshot],
        selected_tracks: list[mocap.MarkerTrack],
    ) -> None:
        if timestamp - self.last_print_time < self.args.print_interval:
            return
        self.last_print_time = timestamp
        mocap.print_status(
            tracks,
            observations_by_camera,
            calibrated_camera_count=len(set(self.calibrations) & {source.camera_id for source in self.sources}),
            triangulator=self.triangulator,
        )
        thread_parts = []
        for camera_id, snapshot in sorted(camera_snapshots.items()):
            if snapshot.error:
                thread_parts.append(f"cam {camera_id}: {snapshot.error}")
            else:
                thread_parts.append(
                    f"cam {camera_id}: {snapshot.delivered_fps:.1f}fps "
                    f"read={snapshot.read_ms:.1f}ms detect={snapshot.detect_ms:.1f}ms"
                )
        mapping = ", ".join(
            f"{label}=track {track_id}"
            for label, track_id in sorted(self.last_hand_assignments.items())
        ) or "none"
        print(
            "[playing test] "
            f"measurements={self.last_measurement_count}, "
            f"live_tracks={self.last_live_track_count}, "
            f"selected={len(selected_tracks)}, "
            f"exclusive_pairing={'yes' if self.last_used_exclusive_pairing else 'no'}, "
            f"hands={mapping}, "
            f"{self._imu_status_text()}, "
            + " | ".join(thread_parts)
        )

    def _print_status_from_scene_state(self, scene_state: ProcessedSceneState) -> None:
        self._print_status(
            scene_state.timestamp,
            scene_state.tracks,
            scene_state.observations_by_camera,
            scene_state.camera_snapshots,
            scene_state.selected_tracks,
        )

    def _imu_status_text(self) -> str:
        return " | ".join(self._imu_status_lines())

    def _imu_status_lines(self) -> list[str]:
        if self.imu_reader is None:
            return [f"imu {label}: off" for label in HAND_LABELS]
        return self.imu_reader.status_lines()

    def close(self, _event=None) -> None:
        if self.closed:
            return
        self.closed = True
        if hasattr(self, "timer"):
            self.timer.stop()
        self.stop_event.set()
        if hasattr(self, "processing_thread"):
            self.processing_thread.join(timeout=0.5)
        for worker in self.camera_workers:
            worker.join(timeout=0.5)
        if self.imu_reader is not None:
            self.imu_reader.join(timeout=0.5)
        for source in self.sources:
            source.close()
        self.visualizer.close()
        if mocap.cv2 is not None:
            mocap.cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.list_scales:
        print("Available scales:")
        for scale_name in sorted(SCALE_INTERVALS):
            print(f"  {scale_name}")
        return 0
    combined.apply_scaling_defaults(args)

    if mocap.cv2 is None:
        print("OpenCV is required for camera mocap. Install it with: python -m pip install opencv-python")
        return 1

    calibrations = combined.load_calibrations(args)
    sources = mocap.open_available_cameras(
        args.cameras,
        args.width,
        args.height,
        args.fps,
        args.exposure,
        args.auto_exposure,
        args.gain,
    )
    if not sources:
        print("[playing test] no cameras opened")
        return 1

    connected_ids = {source.camera_id for source in sources}
    calibrated_connected_ids = connected_ids & set(calibrations)
    if len(calibrated_connected_ids) < 2:
        print(
            "[playing test] fewer than two connected cameras have calibration. "
            "The playing test needs at least two calibrated camera views."
        )

    visualizer: MocapPlayingTestApp | None = None
    try:
        visualizer = MocapPlayingTestApp(args, sources, calibrations)
        fl_debug.app.run()
    except KeyboardInterrupt:
        print("\n[playing test] stopped")
    finally:
        if visualizer is not None:
            visualizer.close()
        else:
            for source in sources:
                source.close()
            if mocap.cv2 is not None:
                mocap.cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
