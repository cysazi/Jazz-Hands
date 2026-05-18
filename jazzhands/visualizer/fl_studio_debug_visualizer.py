"""
Single-window two-hand visualizer for Jazz Hands plane writing and FL Studio MIDI.

This module is intentionally independent from the camera/mocap scripts. Run it
directly for keyboard testing, or import DualHandFLStudioVisualizer and feed it
external hand positions from mocap.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import math
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from vispy import app, scene
from vispy.app import Timer
from vispy.io import read_mesh
from vispy.visuals.transforms import MatrixTransform

from jazzhands.haptics.controller import HapticsController
from jazzhands.music.scales import NOTE_TO_SEMITONE, SCALE_INTERVALS, SEMITONE_NAMES, normalize_scale_key

try:
    import mido
except ImportError:
    mido = None


PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_DIR = os.path.join(PACKAGE_DIR, "assets")
RIGHT_HAND_OBJ_PATH = os.path.join(ASSET_DIR, "hand.obj")
LEFT_HAND_OBJ_PATH = os.path.join(ASSET_DIR, "lefthand.obj")
HAND_OBJ_PATH = RIGHT_HAND_OBJ_PATH
HAND_OBJ_PATHS = {
    "LEFT": LEFT_HAND_OBJ_PATH,
    "RIGHT": RIGHT_HAND_OBJ_PATH,
}


@contextlib.contextmanager
def suppress_native_stderr():
    original_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(original_stderr_fd, 2)
        os.close(original_stderr_fd)
        os.close(devnull_fd)

KEYBOARD_CONTROLLED = False
ENABLE_MIDI = True
MIDI_OUTPUT_HINT = "JazzHands (A)"
MIDI_BASE_NOTE = 60
MIDI_VELOCITY = 127
HAND_MIDI_CHANNELS_1_BASED = {"LEFT": 1, "RIGHT": 1}
DEFAULT_MAX_MIDI_CHANNEL = 4
ENABLE_HAPTICS = True
HAPTICS_PORT: str | None = None
HAPTICS_BAUD = 115200
HAPTICS_NOTE_INTENSITY = 150
HAPTICS_NOTE_DURATION_MS = 55

POSITION_SCALE = 1.0
MODEL_SCALE = 0.02
MIDI_PAN_CONTROL = 10
PAN_CENTER_VALUE = 64
RIGHT_STEREO_SECTION_COUNT = 5
RIGHT_STEREO_CENTER_DEADZONE_PERCENT = 24.0
RIGHT_STEREO_MAX_PAN_PERCENT = 30.0
MIN_PLANE_HALF_EXTENT = 0.03
MIN_DRAW_DISTANCE = 0.02
NOTE_SECTION_COUNT = 12
NOTE_NAMES = ("C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B")
MIN_OCTAVE_OFFSET = -4
MAX_OCTAVE_OFFSET = 4
STACKED_PLANE_OCTAVES = (-1, 0, 1)
BUTTON_HOLD_CLEAR_SECONDS = 5.0
LEFT_VELOCITY_AXIS_DEFAULT = "roll"
LEFT_VELOCITY_MIN_DEG = -60.0
LEFT_VELOCITY_MAX_DEG = 60.0
ROTATION_AXIS_TO_EULER_INDEX = {
    "roll": 0,
    "pitch": 1,
    "yaw": 2,
}
ROTATION_AXIS_KEY_HINTS = {
    "roll": "U/O",
    "pitch": "I/K",
    "yaw": "J/L",
}
AXIS_NAMES = ("X", "Y", "Z")
WORLD_UP_AXIS = 2
PLAY_ENTER_THRESHOLD = 0.02
PLAY_EXIT_THRESHOLD = 0.01
SIDE_INFER_OFFSET_EPS = 0.01
SIDE_INFER_VELOCITY_EPS = 0.03
PLANE_FACES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

VELOCITY_STEP_DELTA = 0.1
VELOCITY_STEP_MIN = 0.1
VELOCITY_STEP_MAX = 5.0
ANGULAR_STEP_DELTA = 0.1
ANGULAR_STEP_MIN = 0.1
ANGULAR_STEP_MAX = 10.0
STATUS_UPDATE_INTERVAL_SECONDS = 0.10
INSTRUMENT_CYCLE = (
    "Synth",
    "Violin",
    "Horn",
    "Instrument4",
    "Instrument5",
    "Instrument6",
    "Instrument7",
    "Instrument8",
    "Instrument9",
)
GESTURE_ROLL_ANGLE_THRESHOLD_DEG = 100.0
GESTURE_ACCEL_SPIKE_THRESHOLD_MPS2 = 3.0
GESTURE_ROLL_DELTA_THRESHOLD_DEG = 40.0
GESTURE_ROLL_RATE_THRESHOLD_DPS = 360.0
GESTURE_WINDOW_MS = 220
GESTURE_COOLDOWN_MS = 300
NOTE_GLOW_DURATION_SECONDS = 0.55
NOTE_GLOW_HALF_SIZE = 0.15
NOTE_GLOW_SEGMENTS = 40

FRAME_MAP = np.array(
    [
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
    ],
    dtype=np.float32,
)


def rot_x(deg: float) -> np.ndarray:
    angle = np.deg2rad(deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def rot_z(deg: float) -> np.ndarray:
    angle = np.deg2rad(deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def center_mesh_vertices(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float32)
    bounds_center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    return vertices - bounds_center


MODEL_OFFSET = rot_x(-90.0)
HAND_VISUAL_OFFSETS = {
    "LEFT": np.eye(3, dtype=np.float32),
    "RIGHT": rot_x(180.0) @ rot_z(180.0),
}
HAND_MODEL_OFFSETS = {
    "LEFT": np.eye(3, dtype=np.float32),
    "RIGHT": np.eye(3, dtype=np.float32),
}


@dataclass
class PlaneDefinition:
    center: np.ndarray
    axis_u: int
    axis_v: int
    normal_axis: int
    half_u: float
    half_v: float


@dataclass
class HandRuntimeState:
    label: str
    position: np.ndarray
    rotation_quaternion: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    rotation_euler: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    button_pressed: bool = False
    last_button_pressed: bool = False
    drawing: bool = False
    draw_origin: np.ndarray | None = None
    plane: PlaneDefinition | None = None
    no_play_side_sign: float = 1.0
    play_side_sign: float = -1.0
    is_on_play_side: bool = False
    active_note_index: int | None = None
    preview_note_index: int | None = None
    last_note_name: str = "inactive"
    last_inside: bool = False
    last_offset: float = 0.0
    last_u_pct: float = 0.0
    last_v_pct: float = 0.0
    tracking_active: bool = True
    volume_value: int = 0
    attack_value: int = 0
    reverb_value: int = 0


def configure_vispy_backend() -> str:
    for backend in ("pyqt6", "pyside6", "tkinter"):
        try:
            with suppress_native_stderr():
                app.use_app(backend)
            return backend
        except Exception:
            continue
    with suppress_native_stderr():
        return app.use_app().backend_name


def clamp_midi_channel_1_based(value: int, max_channel: int = 16) -> int:
    upper = int(np.clip(int(max_channel), 1, 16))
    return int(np.clip(int(value), 1, upper))


def normalize_scale_name(name: str) -> str:
    normalized = normalize_scale_key(name)
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


def normalize_midi_port_name(name: str) -> str:
    return re.sub(r"\s+\d+$", "", str(name).strip()).lower()


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw - az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = normalize_quat(q)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def quat_to_euler_deg(q: np.ndarray) -> tuple[float, float, float]:
    roll, pitch, yaw = quat_to_euler(q)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def euler_deg_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return normalize_quat(
        np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=np.float64,
        )
    )


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


class DualHandFLStudioVisualizer:
    def __init__(
        self,
        keyboard_controlled: bool = KEYBOARD_CONTROLLED,
        enable_midi: bool = ENABLE_MIDI,
        midi_output_hint: str = MIDI_OUTPUT_HINT,
        midi_base_note: int = MIDI_BASE_NOTE,
        midi_velocity: int = MIDI_VELOCITY,
        midi_channels_1_based: dict[str, int] | None = None,
        max_midi_channel: int = DEFAULT_MAX_MIDI_CHANNEL,
        enable_haptics: bool = ENABLE_HAPTICS,
        haptics_port: str | None = HAPTICS_PORT,
        haptics_baud: int = HAPTICS_BAUD,
        require_inside_plane_to_play: bool = False,
        scale_name: str = "chromatic",
        removed_notes: str | list[str] | tuple[str, ...] | None = None,
        notes_per_plane: int | None = None,
        button_hold_clear_seconds: float = BUTTON_HOLD_CLEAR_SECONDS,
        left_velocity_axis: str = LEFT_VELOCITY_AXIS_DEFAULT,
        left_velocity_min_degrees: float = LEFT_VELOCITY_MIN_DEG,
        left_velocity_max_degrees: float = LEFT_VELOCITY_MAX_DEG,
        invert_left_velocity: bool = False,
        keyboard_buttons_enabled: bool = True,
        controlled_hand_label: str = "LEFT",
        allow_hand_switching: bool = True,
        verbose_overlay: bool = False,
        performance_mode: bool = False,
        update_hz: float = 120.0,
        start_timer: bool = True,
        show: bool = True,
    ):
        configure_vispy_backend()
        self.keyboard_controlled = bool(keyboard_controlled)
        self.enable_midi = bool(enable_midi)
        self.midi_output_hint = str(midi_output_hint)
        self.midi_base_note = int(np.clip(midi_base_note, 0, 127))
        self.midi_velocity = int(np.clip(midi_velocity, 1, 127))
        self.haptics = HapticsController(haptics_port, haptics_baud, enable_haptics)
        self.scale_name = normalize_scale_name(scale_name)
        self.scale_intervals = SCALE_INTERVALS[self.scale_name]
        self.initial_midi_base_note = self.midi_base_note
        self.octave_offset = 0
        self.removed_note_names = self._normalize_removed_note_names(removed_notes)
        self.removed_note_pitch_classes = {
            NOTE_TO_SEMITONE[note_name] for note_name in self.removed_note_names
        }
        self.notes_per_plane_override = int(notes_per_plane) if notes_per_plane is not None else None
        self.button_hold_clear_seconds = max(float(button_hold_clear_seconds), 0.25)
        self.left_velocity_axis = str(left_velocity_axis).lower()
        self.left_velocity_min_degrees = float(left_velocity_min_degrees)
        self.left_velocity_max_degrees = float(left_velocity_max_degrees)
        self.invert_left_velocity = bool(invert_left_velocity)
        if self.left_velocity_axis not in ROTATION_AXIS_TO_EULER_INDEX:
            self.left_velocity_axis = LEFT_VELOCITY_AXIS_DEFAULT
        self.max_midi_channel = int(np.clip(int(max_midi_channel), 1, 16))
        channels = midi_channels_1_based or HAND_MIDI_CHANNELS_1_BASED
        self.midi_channels = {
            "LEFT": clamp_midi_channel_1_based(channels.get("LEFT", HAND_MIDI_CHANNELS_1_BASED["LEFT"]), self.max_midi_channel) - 1,
            "RIGHT": clamp_midi_channel_1_based(channels.get("RIGHT", HAND_MIDI_CHANNELS_1_BASED["RIGHT"]), self.max_midi_channel) - 1,
        }
        self.require_inside_plane_to_play = bool(require_inside_plane_to_play)
        self.keyboard_buttons_enabled = bool(keyboard_buttons_enabled)
        self.allow_hand_switching = bool(allow_hand_switching)
        self.verbose_overlay = bool(verbose_overlay)
        self.performance_mode = bool(performance_mode)
        self.start_timer = bool(start_timer)

        self.hand_labels = ("LEFT", "RIGHT")
        self.hands = {
            "LEFT": HandRuntimeState("LEFT", np.array([-0.25, 0.0, 0.0], dtype=np.float64)),
            "RIGHT": HandRuntimeState("RIGHT", np.array([0.25, 0.0, 0.0], dtype=np.float64)),
        }
        self.controlled_hand_label = self._normalize_label(controlled_hand_label)
        self.velocity_step = 0.5
        self.rotation_step = 1.5
        self.keys_down: dict[str, bool] = {}
        self.external_status_lines: list[str] = []
        self.angular_velocity_vectors = {
            "LEFT": np.zeros(3, dtype=np.float64),
            "RIGHT": np.zeros(3, dtype=np.float64),
        }
        self.last_status_update_time = 0.0

        with suppress_native_stderr():
            self.canvas = scene.SceneCanvas(
                keys="interactive",
                show=show,
                bgcolor="black",
                size=(1280, 800),
                title="Jazz Hands FL Studio Debug Visualizer",
            )
        self.view = self.canvas.central_widget.add_view(border_color=(0.2, 0.2, 0.2, 1.0))
        self.view.camera = scene.cameras.TurntableCamera(
            fov=35,
            distance=3.3,
            elevation=42.0,
            azimuth=-90.0,
            center=(0, 0, 0),
        )
        self.world_scene = scene.Node(parent=self.view.scene)
        self.world_scene.transform = MatrixTransform()
        if self.performance_mode:
            transform = np.eye(4, dtype=np.float32)
            transform[0, 0] = -1.0
            self.world_scene.transform.matrix = transform
        scene.visuals.GridLines(scale=(0.25, 0.25), color=(0.25, 0.25, 0.25, 1.0), parent=self.world_scene)
        scene.visuals.XYZAxis(width=2, parent=self.world_scene)
        self._add_axis_labels(self.world_scene)
        self.hand_meshes = {}
        self._setup_hand_meshes()
        self.plane_meshes: dict[str, object] = {}
        self.plane_outlines: dict[str, object] = {}
        self.plane_section_lines: dict[str, list[object]] = {label: [] for label in self.hand_labels}
        self.stacked_plane_meshes: dict[tuple[str, int], object] = {}
        self.stacked_plane_outlines: dict[tuple[str, int], object] = {}
        self.stacked_plane_section_lines: dict[tuple[str, int], list[object]] = {}
        self.status_text = scene.visuals.Text(
            "",
            color=(1.0, 0.96, 0.78, 1.0),
            font_size=34,
            bold=True,
            pos=(640, 58),
            anchor_x="center",
            anchor_y="bottom",
            parent=self.canvas.scene,
        )
        self.note_shadow_text = scene.visuals.Text(
            "",
            color=(0.3, 0.85, 1.0, 0.22),
            font_size=42,
            bold=True,
            pos=(640, 56),
            anchor_x="center",
            anchor_y="bottom",
            parent=self.canvas.scene,
        )
        self.hit_glow_mesh = scene.visuals.Line(
            pos=np.zeros((NOTE_GLOW_SEGMENTS + 1, 3), dtype=np.float32),
            color=(0.6, 0.95, 1.0, 0.0),
            width=4,
            method="gl",
            parent=self.world_scene,
        )
        self.hit_glow_mesh.visible = False
        self.last_right_can_play = False
        self.hit_glow_start_time: float | None = None
        self.hit_glow_definition: PlaneDefinition | None = None

        self.midi_out = None
        self.midi_output_name: str | None = None
        self.active_midi_notes: dict[str, int | None] = {"LEFT": None, "RIGHT": None}
        self.current_volume_value = 0
        self.current_attack_value = 127
        self.current_reverb_value = 0
        self.current_pan_value = PAN_CENTER_VALUE
        self.current_pan_section_index: int | None = None
        self.current_pan_section_name = "center"
        self.button_press_times: dict[str, float | None] = {label: None for label in self.hand_labels}
        self.button_clear_consumed: dict[str, bool] = {label: False for label in self.hand_labels}
        self.selected_note_offset: int | None = None
        self.selected_note_name = "none"
        self.right_roll_samples: deque[tuple[int, float]] = deque(maxlen=10)
        self.last_instrument_gesture_ms = -10_000
        self.current_instrument = INSTRUMENT_CYCLE[0]
        self._refresh_note_layout()
        self._setup_midi_output()
        atexit.register(self.close_midi)

        self.canvas.events.key_press.connect(self.on_key_press)
        self.canvas.events.key_release.connect(self.on_key_release)
        self.canvas.events.close.connect(self.on_close)
        if self.start_timer:
            self.timer = Timer(
                interval=max(1.0 / max(float(update_hz), 1.0), 0.001),
                connect=self.update,
                start=True,
            )

    def set_keyboard_controlled(self, enabled: bool) -> None:
        self.keyboard_controlled = bool(enabled)

    def set_external_status(self, text: str | list[str] | tuple[str, ...] | None) -> None:
        if text is None:
            self.external_status_lines = []
        elif isinstance(text, str):
            self.external_status_lines = [line for line in text.splitlines() if line]
        else:
            self.external_status_lines = [str(line) for line in text if str(line)]

    @property
    def section_count(self) -> int:
        return max(min(self.notes_per_plane, len(self.note_layout_offsets)), 1)

    @property
    def half_scale_section_count(self) -> int:
        return max(int(math.ceil(len(self.scale_intervals) / 2.0)), 1)

    def _scale_degree_offset_for_section(self, section_index: int, page_offset: int = 0) -> int:
        absolute_index = self.note_layout_page_zero_start + int(page_offset) * self.section_count + int(section_index)
        absolute_index = int(np.clip(absolute_index, 0, len(self.note_layout_offsets) - 1))
        return int(self.note_layout_offsets[absolute_index])

    @staticmethod
    def _normalize_removed_note_names(removed_notes: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
        if removed_notes is None:
            return ()
        if isinstance(removed_notes, str):
            raw_values = removed_notes.replace(";", ",").split(",")
        else:
            raw_values = list(removed_notes)
        normalized = []
        for raw_value in raw_values:
            text = str(raw_value).strip().upper().replace("♯", "#").replace("♭", "B")
            text = text.replace(" NATURAL", "").replace("-NATURAL", "").replace("_NATURAL", "")
            if not text:
                continue
            if text not in NOTE_TO_SEMITONE:
                raise ValueError(f"Unknown note to remove: {raw_value!r}")
            normalized.append(text)
        return tuple(dict.fromkeys(normalized))

    def _allowed_pitch_classes(self) -> set[int]:
        pitch_classes = {
            (self.midi_base_note + int(interval)) % 12
            for interval in self.scale_intervals
        }
        return pitch_classes - self.removed_note_pitch_classes

    def _refresh_note_layout(self) -> None:
        allowed_pitch_classes = self._allowed_pitch_classes()
        if not allowed_pitch_classes:
            allowed_pitch_classes = {self.midi_base_note % 12}

        offsets = [
            midi_note - self.midi_base_note
            for midi_note in range(128)
            if midi_note % 12 in allowed_pitch_classes
        ]
        if not offsets:
            offsets = [0]
        self.note_layout_offsets = tuple(offsets)
        lower_or_equal_indices = [
            index for index, offset in enumerate(self.note_layout_offsets) if offset <= 0
        ]
        middle_index = lower_or_equal_indices[-1] if lower_or_equal_indices else 0
        default_notes_per_plane = max(int(math.ceil(len(allowed_pitch_classes) / 2.0)), 1)
        requested_notes_per_plane = self.notes_per_plane_override or default_notes_per_plane
        self.notes_per_plane = int(np.clip(requested_notes_per_plane, 1, len(self.note_layout_offsets)))
        lower_count = (self.notes_per_plane - 1) // 2
        self.note_layout_page_zero_start = int(np.clip(middle_index - lower_count, 0, len(self.note_layout_offsets) - 1))

    def scale_status_text(self) -> str:
        axis_index = ROTATION_AXIS_TO_EULER_INDEX[self.left_velocity_axis]
        left_state = self.hands["LEFT"]
        left_angle = float(left_state.rotation_euler[axis_index])
        roll, pitch, yaw = (float(value) for value in left_state.rotation_euler)
        return (
            f"{self.scale_name} middle={midi_note_name(self.midi_base_note)} "
            f"page={self.octave_offset:+d} selected={self.selected_note_name} "
            f"notes/plane={self.section_count} "
            f"velocity={self.current_attack_value} {self.left_velocity_axis}={left_angle:+.1f}deg "
            f"left_rpy=({roll:+.1f},{pitch:+.1f},{yaw:+.1f}) "
            f"keys={ROTATION_AXIS_KEY_HINTS[self.left_velocity_axis]}"
        )

    def set_scale(self, scale_name: str) -> None:
        self._all_midi_notes_off()
        self.scale_name = normalize_scale_name(scale_name)
        self.scale_intervals = SCALE_INTERVALS[self.scale_name]
        self._refresh_note_layout()
        for label in self.hand_labels:
            state = self.hands[label]
            if state.plane is not None:
                self._update_plane_visuals(label, state.plane, preview=False)
        print(f"[FL visualizer scale] scale={self.scale_name} sections={self.section_count}")

    def set_middle_note(self, middle_note: int) -> None:
        self._all_midi_notes_off()
        self.initial_midi_base_note = int(np.clip(middle_note, 0, 127))
        self.octave_offset = 0
        self.midi_base_note = self.initial_midi_base_note
        self._refresh_note_layout()
        print(f"[FL visualizer scale] middle={midi_note_name(self.midi_base_note)} ({self.midi_base_note})")

    def adjust_octave(self, delta: int) -> None:
        next_offset = int(np.clip(self.octave_offset + int(delta), MIN_OCTAVE_OFFSET, MAX_OCTAVE_OFFSET))
        if next_offset == self.octave_offset:
            return
        self._update_midi_note("RIGHT", None)
        self.octave_offset = next_offset
        print(
            f"[FL visualizer scale] half-scale page={self.octave_offset:+d} "
            f"middle={midi_note_name(self.midi_base_note)}"
        )

    def set_hand_pose(
        self,
        label: str,
        position: np.ndarray | tuple[float, float, float],
        rotation_quaternion: np.ndarray | tuple[float, float, float, float] | None = None,
        button_pressed: bool | None = None,
        timestamp: float | None = None,
    ) -> None:
        _ = timestamp
        state = self.hands[self._normalize_label(label)]
        new_position = np.asarray(position, dtype=np.float64).reshape(3)
        state.velocity = new_position - state.position
        state.position = new_position
        if rotation_quaternion is not None:
            state.rotation_quaternion = normalize_quat(np.asarray(rotation_quaternion, dtype=np.float64))
            state.rotation_euler = np.array(quat_to_euler_deg(state.rotation_quaternion), dtype=np.float64)
        if button_pressed is not None:
            state.button_pressed = bool(button_pressed)
        state.tracking_active = True

    def set_hand_imu_state(
        self,
        label: str,
        rotation_quaternion: np.ndarray | tuple[float, float, float, float] | None = None,
        acceleration: np.ndarray | tuple[float, float, float] | None = None,
        button_pressed: bool | None = None,
    ) -> None:
        state = self.hands[self._normalize_label(label)]
        if rotation_quaternion is not None:
            state.rotation_quaternion = normalize_quat(np.asarray(rotation_quaternion, dtype=np.float64))
            state.rotation_euler = np.array(quat_to_euler_deg(state.rotation_quaternion), dtype=np.float64)
        if acceleration is not None:
            state.acceleration = np.asarray(acceleration, dtype=np.float64).reshape(3)
        if button_pressed is not None:
            state.button_pressed = bool(button_pressed)

    def set_hand_button(self, label: str, pressed: bool) -> None:
        self.hands[self._normalize_label(label)].button_pressed = bool(pressed)

    def set_hand_tracking_active(self, label: str, active: bool) -> None:
        state = self.hands[self._normalize_label(label)]
        state.tracking_active = bool(active)
        if not state.tracking_active:
            state.active_note_index = None
            state.last_note_name = "tracking lost"
            self._update_midi_note(state.label, None)

    def clear_plane(self, label: str | None = None) -> None:
        labels = self.hand_labels if label is None else (self._normalize_label(label),)
        for hand_label in labels:
            state = self.hands[hand_label]
            state.plane = None
            state.drawing = False
            state.draw_origin = None
            state.is_on_play_side = False
            state.active_note_index = None
            state.last_note_name = "inactive"
            self._update_midi_note(hand_label, None)
            mesh = self.plane_meshes.get(hand_label)
            outline = self.plane_outlines.get(hand_label)
            if mesh is not None:
                mesh.visible = False
            if outline is not None:
                outline.visible = False
            for line in self.plane_section_lines[hand_label]:
                line.visible = False
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

    def _normalize_label(self, label: str) -> str:
        normalized = str(label).upper()
        if normalized not in self.hands:
            raise ValueError(f"unknown hand label: {label!r}")
        return normalized

    def _add_axis_labels(self, parent) -> None:
        scene.visuals.Text("X", color="red", font_size=18, pos=(1.25, 0, 0), parent=parent)
        scene.visuals.Text("Y", color="green", font_size=18, pos=(0, 1.25, 0), parent=parent)
        scene.visuals.Text("Z", color="blue", font_size=18, pos=(0, 0, 1.25), parent=parent)

    def _setup_hand_meshes(self) -> None:
        colors = {
            "LEFT": (0.25, 0.75, 0.95, 0.82),
            "RIGHT": (0.95, 0.45, 0.25, 0.82),
        }
        for label in self.hand_labels:
            model_path = HAND_OBJ_PATHS[label]
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Could not find {label.lower()} hand model: {model_path}")
            with suppress_native_stderr():
                vertices, faces, _normals, _texcoords = read_mesh(model_path)
            vertices = center_mesh_vertices(vertices)
            mesh = scene.visuals.Mesh(
                vertices=vertices,
                faces=faces,
                color=colors[label],
                shading=None,
                parent=self.world_scene,
            )
            mesh.transform = MatrixTransform()
            self.hand_meshes[label] = mesh

    def _setup_midi_output(self) -> None:
        if not self.enable_midi:
            print("[FL visualizer MIDI] disabled")
            return
        if mido is None:
            print("[FL visualizer MIDI] mido is not installed; MIDI disabled")
            return
        try:
            output_names = mido.get_output_names()
        except Exception as error:
            print(f"[FL visualizer MIDI] could not list MIDI outputs: {error}")
            return

        hint = str(self.midi_output_hint).strip()
        normalized_hint = normalize_midi_port_name(hint)
        target = next(
            (
                name
                for name in output_names
                if normalized_hint
                and (
                    normalized_hint in normalize_midi_port_name(name)
                    or hint.lower() in str(name).lower()
                )
            ),
            None,
        )
        if target is None:
            print(f"[FL visualizer MIDI] output containing '{self.midi_output_hint}' not found. Available: {output_names}")
            return
        try:
            self.midi_out = mido.open_output(target)
            self.midi_output_name = target
            print(f"[FL visualizer MIDI] connected: {target}")
        except Exception as error:
            print(f"[FL visualizer MIDI] failed to open '{target}': {error}")

    def _send_note_on(self, note: int, channel: int) -> None:
        if self.midi_out is None:
            return
        self.midi_out.send(
            mido.Message(
                "note_on",
                note=int(note),
                velocity=int(np.clip(self.current_attack_value, 1, 127)),
                channel=int(channel),
            )
        )

    def _send_note_off(self, note: int, channel: int) -> None:
        if self.midi_out is None:
            return
        self.midi_out.send(mido.Message("note_off", note=int(note), velocity=0, channel=int(channel)))

    def _send_control_change(self, control: int, value: int, channel: int) -> None:
        if self.midi_out is None:
            return
        self.midi_out.send(
            mido.Message("control_change", control=int(control), value=int(np.clip(value, 0, 127)), channel=int(channel))
        )

    def _update_midi_note(self, label: str, note_index: int | None) -> None:
        desired_note = None
        if note_index is not None:
            desired_note = int(np.clip(self.midi_base_note + int(note_index), 0, 127))
        previous_note = self.active_midi_notes[label]
        if desired_note == previous_note:
            return

        channel = self.midi_channels[label]
        if previous_note is not None:
            self._send_note_off(previous_note, channel)
        if desired_note is not None:
            self._send_note_on(desired_note, channel)
        self.active_midi_notes[label] = desired_note

    def _all_midi_notes_off(self) -> None:
        if not hasattr(self, "active_midi_notes"):
            return
        for label in self.hand_labels:
            self._update_midi_note(label, None)

    def midi_channel_text(self) -> str:
        left_channel = self.midi_channels["LEFT"] + 1
        right_channel = self.midi_channels["RIGHT"] + 1
        return f"channels LEFT={left_channel} RIGHT={right_channel} max={self.max_midi_channel}"

    def set_hand_midi_channel(self, label: str, channel_1_based: int) -> None:
        hand_label = self._normalize_label(label)
        next_channel = clamp_midi_channel_1_based(channel_1_based, self.max_midi_channel) - 1
        if next_channel == self.midi_channels[hand_label]:
            return

        self._update_midi_note(hand_label, None)
        self.midi_channels[hand_label] = next_channel
        print(f"[FL visualizer MIDI] {hand_label} channel {next_channel + 1}")

    def cycle_hand_midi_channel(self, label: str, delta: int = 1) -> None:
        hand_label = self._normalize_label(label)
        current_channel = self.midi_channels[hand_label] + 1
        next_channel = ((current_channel - 1 + int(delta)) % self.max_midi_channel) + 1
        self.set_hand_midi_channel(hand_label, next_channel)

    def close_midi(self) -> None:
        self._all_midi_notes_off()
        if self.midi_out is not None:
            self.midi_out.close()
            self.midi_out = None
            self.midi_output_name = None

    def _normalize_key_name(self, raw_name) -> str | None:
        if raw_name is None:
            return None
        key_name = str(raw_name)
        if key_name.strip() == "" or key_name.lower() == "space":
            return "SPACE"
        return key_name.upper()

    def on_key_press(self, event) -> None:
        if not event.key:
            return
        key_name = self._normalize_key_name(event.key.name)
        if key_name is None:
            return
        self.keys_down[key_name] = True

        if key_name == "ESCAPE":
            self.close()
            self.canvas.close()
            return
        if key_name == "TAB" and self.allow_hand_switching:
            self.controlled_hand_label = "RIGHT" if self.controlled_hand_label == "LEFT" else "LEFT"
        elif key_name == "C":
            self.clear_plane(self.controlled_hand_label)
        elif key_name == "F":
            self._flip_play_side(self.controlled_hand_label)
        elif key_name == "UP":
            self.velocity_step = float(np.clip(self.velocity_step + VELOCITY_STEP_DELTA, VELOCITY_STEP_MIN, VELOCITY_STEP_MAX))
        elif key_name == "DOWN":
            self.velocity_step = float(np.clip(self.velocity_step - VELOCITY_STEP_DELTA, VELOCITY_STEP_MIN, VELOCITY_STEP_MAX))
        elif key_name == "RIGHT":
            self.rotation_step = float(np.clip(self.rotation_step + ANGULAR_STEP_DELTA, ANGULAR_STEP_MIN, ANGULAR_STEP_MAX))
        elif key_name == "LEFT":
            self.rotation_step = float(np.clip(self.rotation_step - ANGULAR_STEP_DELTA, ANGULAR_STEP_MIN, ANGULAR_STEP_MAX))
        elif key_name.isdigit() and key_name != "0":
            channel = int(key_name)
            if channel <= self.max_midi_channel:
                self.set_hand_midi_channel(self.controlled_hand_label, channel)
        elif key_name.startswith("DIGIT") and key_name[5:].isdigit():
            channel = int(key_name[5:])
            if 1 <= channel <= self.max_midi_channel:
                self.set_hand_midi_channel(self.controlled_hand_label, channel)

        self._update_keyboard_motion()

    def on_key_release(self, event) -> None:
        if not event.key:
            return
        key_name = self._normalize_key_name(event.key.name)
        if key_name is None:
            return
        self.keys_down[key_name] = False
        if key_name == "SPACE":
            self.keys_down.pop(" ", None)
            self.keys_down.pop("Space", None)
        self._update_keyboard_motion()

    def _update_keyboard_motion(self) -> None:
        for label in self.hand_labels:
            self.angular_velocity_vectors[label] = np.zeros(3, dtype=np.float64)
            if self.keyboard_buttons_enabled and label != self.controlled_hand_label:
                self.hands[label].button_pressed = False

        if self.keyboard_buttons_enabled:
            self.hands[self.controlled_hand_label].button_pressed = self.keys_down.get("SPACE", False)

        if not self.keyboard_controlled:
            return

        state = self.hands[self.controlled_hand_label]
        velocity = np.zeros(3, dtype=np.float64)
        if self.keys_down.get("W", False):
            velocity[0] += self.velocity_step
        if self.keys_down.get("S", False):
            velocity[0] -= self.velocity_step
        if self.keys_down.get("A", False):
            velocity[1] += self.velocity_step
        if self.keys_down.get("D", False):
            velocity[1] -= self.velocity_step
        if self.keys_down.get("Q", False):
            velocity[2] -= self.velocity_step
        if self.keys_down.get("E", False):
            velocity[2] += self.velocity_step
        state.velocity = velocity

        angular_velocity = np.zeros(3, dtype=np.float64)
        if self.keys_down.get("I", False):
            angular_velocity[1] += self.rotation_step
        if self.keys_down.get("K", False):
            angular_velocity[1] -= self.rotation_step
        if self.keys_down.get("J", False):
            angular_velocity[2] += self.rotation_step
        if self.keys_down.get("L", False):
            angular_velocity[2] -= self.rotation_step
        if self.keys_down.get("U", False):
            angular_velocity[0] += self.rotation_step
        if self.keys_down.get("O", False):
            angular_velocity[0] -= self.rotation_step
        self.angular_velocity_vectors[self.controlled_hand_label] = angular_velocity

    def update(self, event) -> None:
        dt = float(event.dt) if event is not None and event.dt is not None else 0.0
        if self.keyboard_controlled:
            self._apply_keyboard_motion(dt)

        for label in self.hand_labels:
            self._update_plane_state(label)
            self._update_note_state(label)
        self._update_left_controls()
        self._update_right_stereo_control()
        self._update_right_instrument_gesture()
        self._update_hit_glow()
        for label in self.hand_labels:
            self._update_hand_visual(label)

        now = time.time()
        if now - self.last_status_update_time >= STATUS_UPDATE_INTERVAL_SECONDS:
            self.last_status_update_time = now
            self._update_status_text()
        self.canvas.update()

    def _apply_keyboard_motion(self, dt: float) -> None:
        for label in self.hand_labels:
            state = self.hands[label]
            if label == self.controlled_hand_label:
                state.position = state.position + state.velocity * dt

                angular_velocity = self.angular_velocity_vectors[label]
                if float(np.linalg.norm(angular_velocity)) > 1e-9 and dt > 0.0:
                    state.rotation_euler = state.rotation_euler + np.degrees(angular_velocity * dt)
                    state.rotation_euler = ((state.rotation_euler + 180.0) % 360.0) - 180.0
                    state.rotation_quaternion = euler_deg_to_quat(*state.rotation_euler)
            else:
                state.velocity = np.zeros(3, dtype=np.float64)

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
                definition = self._compute_plane_for_hand(label, state.draw_origin, state.position)
                if definition is not None:
                    self._update_plane_visuals(label, definition, preview=True)

            if released_edge and state.drawing:
                if state.draw_origin is not None:
                    definition = self._compute_plane_for_hand(label, state.draw_origin, state.position)
                    if definition is not None:
                        state.plane = definition
                        state.no_play_side_sign = -1.0
                        state.play_side_sign = 1.0
                        state.is_on_play_side = False
                        state.active_note_index = None
                        self._update_plane_visuals(label, definition, preview=False)
                        print(
                            f"[{label}] plane committed: center={definition.center}, "
                            f"normal={AXIS_NAMES[definition.normal_axis]}, "
                            f"play side=+{AXIS_NAMES[definition.normal_axis]}"
                        )
                    else:
                        print(f"[{label}] plane draw ignored: move farther before releasing SPACE")
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
                self.adjust_octave(-1 if label == "LEFT" else 1)
            self.button_press_times[label] = None
            self.button_clear_consumed[label] = False

        state.last_button_pressed = state.button_pressed

    def _compute_debug_plane(self, origin: np.ndarray, current_position: np.ndarray) -> PlaneDefinition | None:
        delta = current_position - origin
        if float(np.linalg.norm(delta)) < MIN_DRAW_DISTANCE:
            return None

        abs_delta = np.abs(delta)
        normal_axis = int(np.argmin(abs_delta))
        plane_axes = [axis for axis in range(3) if axis != normal_axis]
        half_u = max(float(abs_delta[plane_axes[0]]), MIN_PLANE_HALF_EXTENT)
        half_v = max(float(abs_delta[plane_axes[1]]), MIN_PLANE_HALF_EXTENT)

        return PlaneDefinition(
            center=origin.copy(),
            axis_u=plane_axes[0],
            axis_v=plane_axes[1],
            normal_axis=normal_axis,
            half_u=half_u,
            half_v=half_v,
        )

    def _compute_plane_for_hand(self, label: str, origin: np.ndarray, current_position: np.ndarray) -> PlaneDefinition | None:
        _ = label
        return self._compute_debug_plane(origin, current_position)

    def _infer_no_play_side_sign(
        self,
        definition: PlaneDefinition,
        position: np.ndarray,
        velocity: np.ndarray,
    ) -> float:
        offset = float(position[definition.normal_axis] - definition.center[definition.normal_axis])
        if abs(offset) >= SIDE_INFER_OFFSET_EPS:
            return 1.0 if offset > 0.0 else -1.0

        velocity_normal = float(velocity[definition.normal_axis])
        if abs(velocity_normal) >= SIDE_INFER_VELOCITY_EPS:
            return 1.0 if velocity_normal > 0.0 else -1.0

        return 1.0

    def _flip_play_side(self, label: str) -> None:
        _ = label
        print("[FL visualizer] play side is fixed to the plane's positive normal axis")

    def _ensure_plane_visuals(self, label: str) -> None:
        if label not in self.plane_meshes:
            mesh = scene.visuals.Mesh(
                vertices=np.zeros((4, 3), dtype=np.float32),
                faces=PLANE_FACES,
                color=(0.25, 0.85, 1.0, 0.0),
                shading=None,
                parent=self.world_scene,
            )
            mesh.visible = False
            self.plane_meshes[label] = mesh

        if label not in self.plane_outlines:
            outline = scene.visuals.Line(
                pos=np.zeros((2, 3), dtype=np.float32),
                color=(0.75, 0.95, 1.0, 0.0),
                width=2,
                method="gl",
                parent=self.world_scene,
            )
            outline.visible = False
            self.plane_outlines[label] = outline

        while len(self.plane_section_lines[label]) < NOTE_SECTION_COUNT - 1:
            line = scene.visuals.Line(
                pos=np.zeros((2, 3), dtype=np.float32),
                color=(0.75, 0.95, 1.0, 0.0),
                width=1,
                method="gl",
                parent=self.world_scene,
            )
            line.visible = False
            self.plane_section_lines[label].append(line)

    def _hide_stacked_plane_visuals(self, label: str) -> None:
        for octave_index in STACKED_PLANE_OCTAVES:
            key = (label, octave_index)
            mesh = self.stacked_plane_meshes.get(key)
            outline = self.stacked_plane_outlines.get(key)
            if mesh is not None:
                mesh.visible = False
            if outline is not None:
                outline.visible = False
            for line in self.stacked_plane_section_lines.get(key, []):
                line.visible = False

    def _hide_regular_plane_visuals(self, label: str) -> None:
        mesh = self.plane_meshes.get(label)
        outline = self.plane_outlines.get(label)
        if mesh is not None:
            mesh.visible = False
        if outline is not None:
            outline.visible = False
        for line in self.plane_section_lines.get(label, []):
            line.visible = False

    def _plane_vertices(self, definition: PlaneDefinition) -> np.ndarray:
        center = definition.center
        vertices = np.tile(center.astype(np.float32), (4, 1))
        for idx, (u_sign, v_sign) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
            vertices[idx, definition.axis_u] = center[definition.axis_u] + u_sign * definition.half_u
            vertices[idx, definition.axis_v] = center[definition.axis_v] + v_sign * definition.half_v
            vertices[idx, definition.normal_axis] = center[definition.normal_axis]
        return vertices

    def _glow_circle_vertices(self, definition: PlaneDefinition) -> np.ndarray:
        vertices = np.tile(definition.center.astype(np.float32), (NOTE_GLOW_SEGMENTS + 1, 1))
        radius = float(definition.half_u)
        for index, angle in enumerate(np.linspace(0.0, 2.0 * math.pi, NOTE_GLOW_SEGMENTS, endpoint=False)):
            vertices[index, definition.axis_u] += math.cos(angle) * radius
            vertices[index, definition.axis_v] += math.sin(angle) * radius
        vertices[-1] = vertices[0]
        return vertices

    def _note_axis_and_half_extent(self, definition: PlaneDefinition) -> tuple[int, float, int, float]:
        if definition.axis_u == WORLD_UP_AXIS:
            return definition.axis_u, definition.half_u, definition.axis_v, definition.half_v
        if definition.axis_v == WORLD_UP_AXIS:
            return definition.axis_v, definition.half_v, definition.axis_u, definition.half_u
        if definition.half_u >= definition.half_v:
            return definition.axis_u, definition.half_u, definition.axis_v, definition.half_v
        return definition.axis_v, definition.half_v, definition.axis_u, definition.half_u

    def _section_line_segments(self, definition: PlaneDefinition) -> list[np.ndarray]:
        return self._section_line_segments_for_count(definition, self.section_count)

    def _stacked_plane_definition(
        self,
        definition: PlaneDefinition,
        octave_index: int,
    ) -> PlaneDefinition:
        note_axis, note_half, _other_axis, _other_half = self._note_axis_and_half_extent(definition)
        center = definition.center.copy()
        center[note_axis] += float(octave_index) * (2.0 * note_half)
        return PlaneDefinition(
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
            mesh = scene.visuals.Mesh(
                vertices=np.zeros((4, 3), dtype=np.float32),
                faces=PLANE_FACES,
                color=(0.25, 0.85, 1.0, 0.0),
                shading=None,
                parent=self.world_scene,
            )
            mesh.visible = False
            self.stacked_plane_meshes[key] = mesh

        if key not in self.stacked_plane_outlines:
            outline = scene.visuals.Line(
                pos=np.zeros((2, 3), dtype=np.float32),
                color=(0.75, 0.95, 1.0, 0.0),
                width=2,
                method="gl",
                parent=self.world_scene,
            )
            outline.visible = False
            self.stacked_plane_outlines[key] = outline

        lines = self.stacked_plane_section_lines.setdefault(key, [])
        while len(lines) < self.section_count - 1:
            line = scene.visuals.Line(
                pos=np.zeros((2, 3), dtype=np.float32),
                color=(0.75, 0.95, 1.0, 0.0),
                width=1,
                method="gl",
                parent=self.world_scene,
            )
            line.visible = False
            lines.append(line)

    def _section_line_segments_for_count(
        self,
        definition: PlaneDefinition,
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

    def _right_stereo_axis_and_half_extent(self, definition: PlaneDefinition) -> tuple[int, float, int, float]:
        if definition.axis_u == 0:
            return definition.axis_u, definition.half_u, definition.axis_v, definition.half_v
        if definition.axis_v == 0:
            return definition.axis_v, definition.half_v, definition.axis_u, definition.half_u
        if definition.axis_u != WORLD_UP_AXIS:
            return definition.axis_u, definition.half_u, definition.axis_v, definition.half_v
        if definition.axis_v != WORLD_UP_AXIS:
            return definition.axis_v, definition.half_v, definition.axis_u, definition.half_u
        return definition.axis_u, definition.half_u, definition.axis_v, definition.half_v

    def _right_stereo_section_line_segments(self, definition: PlaneDefinition) -> list[np.ndarray]:
        segments = []
        stereo_axis, stereo_half, other_axis, other_half = self._right_stereo_axis_and_half_extent(definition)
        stereo_min = definition.center[stereo_axis] - stereo_half
        other_min = definition.center[other_axis] - other_half
        for split_index in range(1, RIGHT_STEREO_SECTION_COUNT):
            split_coord = stereo_min + (split_index / RIGHT_STEREO_SECTION_COUNT) * (2.0 * stereo_half)
            p0 = definition.center.copy()
            p1 = definition.center.copy()
            p0[stereo_axis] = split_coord
            p1[stereo_axis] = split_coord
            p0[other_axis] = definition.center[other_axis] - other_half
            p1[other_axis] = definition.center[other_axis] + other_half
            segments.append(np.array([p0, p1], dtype=np.float32))
        for split_index in range(1, RIGHT_STEREO_SECTION_COUNT):
            split_coord = other_min + (split_index / RIGHT_STEREO_SECTION_COUNT) * (2.0 * other_half)
            p0 = definition.center.copy()
            p1 = definition.center.copy()
            p0[other_axis] = split_coord
            p1[other_axis] = split_coord
            p0[stereo_axis] = definition.center[stereo_axis] - stereo_half
            p1[stereo_axis] = definition.center[stereo_axis] + stereo_half
            segments.append(np.array([p0, p1], dtype=np.float32))
        return segments

    def _update_right_plane_visuals(self, definition: PlaneDefinition, preview: bool) -> None:
        self._hide_stacked_plane_visuals("RIGHT")
        self._ensure_plane_visuals("RIGHT")
        fill = (1.00, 0.48, 0.32, 0.08 if preview else 0.16)
        edge = (1.00, 0.62, 0.45, 0.55 if preview else 0.9)
        split = (1.00, 0.84, 0.70, 0.35 if preview else 0.65)

        vertices = self._plane_vertices(definition) * POSITION_SCALE
        outline = np.vstack([vertices, vertices[0]])
        self.plane_meshes["RIGHT"].set_data(vertices=vertices, faces=PLANE_FACES, color=fill)
        self.plane_meshes["RIGHT"].visible = True
        self.plane_outlines["RIGHT"].set_data(pos=outline, color=edge)
        self.plane_outlines["RIGHT"].visible = True

        segments = self._right_stereo_section_line_segments(definition)
        for index, line in enumerate(self.plane_section_lines["RIGHT"]):
            if index < len(segments):
                line.set_data(pos=segments[index] * POSITION_SCALE, color=split)
                line.visible = True
            else:
                line.visible = False

    def _update_plane_visuals(self, label: str, definition: PlaneDefinition, preview: bool) -> None:
        if label == "RIGHT":
            self._update_right_plane_visuals(definition, preview)
            return

        self._hide_regular_plane_visuals(label)
        colors = {
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
        }[label]

        for octave_index in STACKED_PLANE_OCTAVES:
            stacked = self._stacked_plane_definition(definition, octave_index)
            self._ensure_stacked_plane_visuals(label, octave_index)
            key = (label, octave_index)
            alpha_scale = 1.0 if octave_index == 0 else 0.55
            fill = (*colors["fill"][:3], colors["fill"][3] * alpha_scale)
            edge = (*colors["edge"][:3], colors["edge"][3] * alpha_scale)
            split = (*colors["split"][:3], colors["split"][3] * alpha_scale)

            vertices = self._plane_vertices(stacked) * POSITION_SCALE
            outline = np.vstack([vertices, vertices[0]])
            self.stacked_plane_meshes[key].set_data(vertices=vertices, faces=PLANE_FACES, color=fill)
            self.stacked_plane_meshes[key].visible = True
            self.stacked_plane_outlines[key].set_data(pos=outline, color=edge)
            self.stacked_plane_outlines[key].visible = True

            segments = self._section_line_segments_for_count(stacked, self.section_count)
            lines = self.stacked_plane_section_lines[key]
            for index, line in enumerate(lines):
                if index < len(segments):
                    line.set_data(pos=segments[index] * POSITION_SCALE, color=split)
                    line.visible = True
                else:
                    line.visible = False

    def _plane_position_metrics(
        self,
        position: np.ndarray,
        definition: PlaneDefinition,
    ) -> tuple[float, float, bool, float]:
        u_min = definition.center[definition.axis_u] - definition.half_u
        v_min = definition.center[definition.axis_v] - definition.half_v
        u = (position[definition.axis_u] - u_min) / (2.0 * definition.half_u)
        v = (position[definition.axis_v] - v_min) / (2.0 * definition.half_v)
        inside = (0.0 <= u <= 1.0) and (0.0 <= v <= 1.0)
        offset = float(position[definition.normal_axis] - definition.center[definition.normal_axis])
        return (
            float(np.clip(u, 0.0, 1.0) * 100.0),
            float(np.clip(v, 0.0, 1.0) * 100.0),
            inside,
            offset,
        )

    def _note_section_from_position(self, position: np.ndarray, definition: PlaneDefinition) -> tuple[int, str, str, float]:
        note_axis, note_half, _other_axis, _other_half = self._note_axis_and_half_extent(definition)
        axis_min = definition.center[note_axis] - note_half
        normalized = (position[note_axis] - axis_min) / (2.0 * note_half)
        clamped = float(np.clip(normalized, 0.0, 1.0))
        section_index = min(int(clamped * NOTE_SECTION_COUNT), NOTE_SECTION_COUNT - 1)
        return section_index, NOTE_NAMES[section_index], AXIS_NAMES[note_axis], clamped * 100.0

    def _stacked_plane_position_metrics(
        self,
        position: np.ndarray,
        definition: PlaneDefinition,
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

    def _left_note_plane_position_metrics(
        self,
        position: np.ndarray,
        definition: PlaneDefinition,
    ) -> tuple[float, float, bool, int]:
        note_pct, other_pct, inside, ghost_page_offset = self._stacked_plane_position_metrics(
            position,
            definition,
        )
        return note_pct, other_pct, inside, self.octave_offset + ghost_page_offset

    def _right_stereo_position_metrics(self, position: np.ndarray, definition: PlaneDefinition) -> tuple[float, int, str, int]:
        stereo_axis, stereo_half, _other_axis, _other_half = self._right_stereo_axis_and_half_extent(definition)
        axis_min = definition.center[stereo_axis] - stereo_half
        normalized = (position[stereo_axis] - axis_min) / (2.0 * stereo_half)
        stereo_pct = float(np.clip(normalized, 0.0, 1.0) * 100.0)
        pan_percent = float(np.clip((stereo_pct - 50.0) * 2.0, -100.0, 100.0))
        center_half_width = float(np.clip(RIGHT_STEREO_CENTER_DEADZONE_PERCENT, 0.0, 95.0))
        if abs(pan_percent) <= center_half_width:
            pan_value = PAN_CENTER_VALUE
            pan_name = "center"
        else:
            side_amount = (abs(pan_percent) - center_half_width) / (100.0 - center_half_width)
            capped_side_amount = side_amount * float(np.clip(RIGHT_STEREO_MAX_PAN_PERCENT, 0.0, 100.0)) / 100.0
            if pan_percent < 0.0:
                pan_value = int(round(PAN_CENTER_VALUE * (1.0 - capped_side_amount)))
                pan_name = f"L {capped_side_amount * 100.0:.0f}%"
            else:
                pan_value = int(round(PAN_CENTER_VALUE + (127 - PAN_CENTER_VALUE) * capped_side_amount))
                pan_name = f"R {capped_side_amount * 100.0:.0f}%"
        section_index = min(
            int((stereo_pct / 100.0) * RIGHT_STEREO_SECTION_COUNT),
            RIGHT_STEREO_SECTION_COUNT - 1,
        )
        return (
            stereo_pct,
            int(np.clip(pan_value, 0, 127)),
            pan_name,
            section_index,
        )

    @staticmethod
    def _section_index_from_percent(percent: float, section_count: int) -> int:
        section_count = max(int(section_count), 1)
        normalized = float(np.clip(percent / 100.0, 0.0, 1.0))
        if normalized <= 0.0:
            return 0
        if normalized >= 1.0:
            return section_count - 1
        return int(np.floor(np.nextafter(normalized * section_count, -np.inf)))

    def _note_offset_from_left_position(self) -> tuple[int | None, str, float, int]:
        state = self.hands["LEFT"]
        if not state.tracking_active or state.plane is None:
            return None, "none", 0.0, 0
        note_pct, _other_pct, inside, page_offset = self._left_note_plane_position_metrics(state.position, state.plane)
        note_axis, _note_half, _other_axis, _other_half = self._note_axis_and_half_extent(state.plane)
        section_index = self._section_index_from_percent(note_pct, self.section_count)
        semitone_offset = self._scale_degree_offset_for_section(section_index, page_offset)
        midi_note = int(np.clip(self.midi_base_note + semitone_offset, 0, 127))
        note_name = midi_note_name(midi_note)
        if not inside and self.require_inside_plane_to_play:
            return None, f"outside {note_name}", note_pct, page_offset
        return semitone_offset, f"{note_name} {AXIS_NAMES[note_axis]}={note_pct:5.1f}%", note_pct, page_offset

    def _continuous_plane_percentages(self, state: HandRuntimeState) -> tuple[float, float, bool]:
        if state.plane is None:
            return 0.0, 0.0, False
        u_pct, v_pct, inside, _offset = self._plane_position_metrics(state.position, state.plane)
        return u_pct, v_pct, inside

    def _rotation_pct(self, value: float, min_degrees: float, max_degrees: float) -> float:
        span = max(float(max_degrees) - float(min_degrees), 1e-6)
        return float(np.clip((float(value) - float(min_degrees)) / span, 0.0, 1.0) * 100.0)

    def _update_left_controls(self) -> None:
        state = self.hands["LEFT"]
        if not state.tracking_active:
            return

        state.volume_value = 127
        self.current_attack_value = 127

        right_channel = self.midi_channels["RIGHT"]
        if state.volume_value != self.current_volume_value:
            self.current_volume_value = state.volume_value
            self._send_control_change(101, self.current_volume_value, right_channel)
            self._send_control_change(7, self.current_volume_value, right_channel)

    def _update_right_stereo_control(self) -> None:
        state = self.hands["RIGHT"]
        if not state.tracking_active or state.plane is None:
            return

        stereo_pct, pan_value, pan_name, section_index = self._right_stereo_position_metrics(state.position, state.plane)
        state.last_v_pct = stereo_pct
        self.current_pan_section_index = section_index
        self.current_pan_section_name = pan_name
        if pan_value == self.current_pan_value:
            return

        self.current_pan_value = pan_value
        self._send_control_change(MIDI_PAN_CONTROL, self.current_pan_value, self.midi_channels["RIGHT"])

    @staticmethod
    def _roll_delta_deg(current_roll: float, previous_roll: float) -> float:
        delta = float(current_roll - previous_roll)
        while delta > 180.0:
            delta -= 360.0
        while delta < -180.0:
            delta += 360.0
        return delta

    def _update_right_instrument_gesture(self) -> None:
        state = self.hands["RIGHT"]
        if not state.tracking_active:
            self.right_roll_samples.clear()
            return

        timestamp_ms = int(time.time() * 1000.0)
        roll_deg = float(state.rotation_euler[0])
        if self.right_roll_samples and timestamp_ms <= self.right_roll_samples[-1][0]:
            return
        self.right_roll_samples.append((timestamp_ms, roll_deg))
        if len(self.right_roll_samples) < 2:
            return

        current_t, current_roll = self.right_roll_samples[-1]
        window_start_t = current_t - GESTURE_WINDOW_MS
        oldest_t, oldest_roll = self.right_roll_samples[-1]
        for sample_t, sample_roll in self.right_roll_samples:
            if sample_t >= window_start_t:
                oldest_t, oldest_roll = sample_t, sample_roll
                break

        dt_ms = current_t - oldest_t
        if dt_ms <= 0:
            return

        delta_roll = self._roll_delta_deg(current_roll, oldest_roll)
        roll_rate_dps = delta_roll / (dt_ms / 1000.0)
        accel_norm = float(np.linalg.norm(state.acceleration))
        if (
            abs(current_roll) >= GESTURE_ROLL_ANGLE_THRESHOLD_DEG
            and accel_norm >= GESTURE_ACCEL_SPIKE_THRESHOLD_MPS2
            and abs(delta_roll) >= GESTURE_ROLL_DELTA_THRESHOLD_DEG
            and abs(roll_rate_dps) >= GESTURE_ROLL_RATE_THRESHOLD_DPS
            and (current_t - self.last_instrument_gesture_ms) >= GESTURE_COOLDOWN_MS
        ):
            direction = 1 if roll_rate_dps > 0.0 else -1
            current_idx = INSTRUMENT_CYCLE.index(self.current_instrument)
            self.current_instrument = INSTRUMENT_CYCLE[(current_idx + direction) % len(INSTRUMENT_CYCLE)]
            print(f"[FL visualizer gesture] instrument set to {self.current_instrument}")
            self.last_instrument_gesture_ms = current_t

    def _update_note_state(self, label: str) -> None:
        if label == "LEFT":
            self._update_left_controls()
            note_offset, note_name, note_pct, octave_index = self._note_offset_from_left_position()
            state = self.hands["LEFT"]
            if note_offset is not None and state.preview_note_index != note_offset:
                self.haptics.pulse("LEFT", HAPTICS_NOTE_INTENSITY, HAPTICS_NOTE_DURATION_MS)
            self.selected_note_offset = note_offset
            self.selected_note_name = note_name
            state.active_note_index = None
            state.preview_note_index = note_offset
            state.last_note_name = f"select={note_name} page={octave_index:+d}"
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
            self.last_right_can_play = False
            self._update_midi_note("RIGHT", None)
            return

        if state.plane is None:
            state.active_note_index = None
            state.preview_note_index = None
            state.last_note_name = "inactive"
            self.last_right_can_play = False
            self._update_midi_note("RIGHT", None)
            return

        _u_pct, _v_pct, inside, _offset = self._plane_position_metrics(state.position, state.plane)
        stereo_pct, pan_value, pan_name, section_index = self._right_stereo_position_metrics(state.position, state.plane)
        play_offset = float(state.position[state.plane.normal_axis] - state.plane.center[state.plane.normal_axis])
        if state.is_on_play_side:
            if play_offset <= PLAY_EXIT_THRESHOLD:
                state.is_on_play_side = False
        elif play_offset >= PLAY_ENTER_THRESHOLD:
            state.is_on_play_side = True

        can_play = (
            self.selected_note_offset is not None
            and state.is_on_play_side
            and (inside or not self.require_inside_plane_to_play)
        )
        if can_play and not self.last_right_can_play:
            self._trigger_note_glow(state.plane, state.position)
        self.last_right_can_play = can_play
        state.active_note_index = self.selected_note_offset if can_play else None
        if self.selected_note_offset is not None and state.preview_note_index != self.selected_note_offset:
            self.haptics.pulse("RIGHT", HAPTICS_NOTE_INTENSITY, HAPTICS_NOTE_DURATION_MS)
        state.preview_note_index = self.selected_note_offset
        state.last_note_name = (
            f"PLAY {self.selected_note_name} pan={pan_name}"
            if can_play
            else f"ready={self.selected_note_name} pan={pan_name}"
        )
        state.last_inside = inside
        state.last_offset = play_offset
        state.last_u_pct = stereo_pct
        state.last_v_pct = float(section_index + 1)
        self.current_pan_section_index = section_index
        self.current_pan_section_name = pan_name
        self._update_midi_note("RIGHT", state.active_note_index)

    def _trigger_note_glow(self, definition: PlaneDefinition | None, position: np.ndarray) -> None:
        if definition is None:
            return
        center = np.asarray(position, dtype=np.float64).copy()
        center[definition.normal_axis] = definition.center[definition.normal_axis]
        self.hit_glow_definition = PlaneDefinition(
            center=center,
            axis_u=definition.axis_u,
            axis_v=definition.axis_v,
            normal_axis=definition.normal_axis,
            half_u=NOTE_GLOW_HALF_SIZE,
            half_v=NOTE_GLOW_HALF_SIZE,
        )
        self.hit_glow_start_time = time.time()
        self._update_hit_glow()

    def _update_hit_glow(self) -> None:
        if self.hit_glow_start_time is None or self.hit_glow_definition is None:
            self.hit_glow_mesh.visible = False
            return

        elapsed = time.time() - self.hit_glow_start_time
        progress = elapsed / NOTE_GLOW_DURATION_SECONDS
        if progress >= 1.0:
            self.hit_glow_mesh.visible = False
            self.hit_glow_start_time = None
            return

        definition = self.hit_glow_definition
        expanded = PlaneDefinition(
            center=definition.center.copy(),
            axis_u=definition.axis_u,
            axis_v=definition.axis_v,
            normal_axis=definition.normal_axis,
            half_u=definition.half_u * (1.08 + 0.18 * progress),
            half_v=definition.half_v * (1.08 + 0.18 * progress),
        )
        vertices = self._glow_circle_vertices(expanded) * POSITION_SCALE
        alpha = 0.42 * (1.0 - progress) ** 1.4
        self.hit_glow_mesh.set_data(pos=vertices, color=(0.62, 0.95, 1.0, alpha))
        self.hit_glow_mesh.visible = True

    def _update_hand_visual(self, label: str) -> None:
        state = self.hands[label]
        rotation = quaternion_to_rotation_matrix(state.rotation_quaternion)
        rotation = FRAME_MAP @ rotation @ FRAME_MAP.T
        rotation = rotation @ MODEL_OFFSET
        rotation = rotation @ HAND_MODEL_OFFSETS[label]
        rotation = HAND_VISUAL_OFFSETS[label] @ rotation

        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rotation * MODEL_SCALE
        transform[3, :3] = state.position.astype(np.float32) * POSITION_SCALE
        self.hand_meshes[label].transform.matrix = transform

    def _update_status_text(self) -> None:
        if not self.verbose_overlay:
            self._update_compact_status_text()
            return

        lines = []
        keyboard_channel_max = min(self.max_midi_channel, 9)
        channel_key_text = (
            "1 set selected channel"
            if keyboard_channel_max <= 1
            else f"1-{keyboard_channel_max} set selected channel"
        )
        for label in self.hand_labels:
            state = self.hands[label]
            plane_text = "plane=not set"
            if state.drawing:
                plane_text = "plane=drawing"
            elif state.plane is not None:
                if label == "RIGHT":
                    plane_text = (
                        f"plane=stereo x={state.last_u_pct:5.1f}% "
                        f"section={self.current_pan_section_name} pan={self.current_pan_value} "
                        f"play_x={state.last_offset:+.3f}"
                    )
                else:
                    plane_text = (
                        f"plane={AXIS_NAMES[state.plane.axis_u]}/{AXIS_NAMES[state.plane.axis_v]} "
                        f"normal={AXIS_NAMES[state.plane.normal_axis]} "
                        f"uv=({state.last_u_pct:5.1f}%, {state.last_v_pct:5.1f}%) "
                        f"play_x={state.last_offset:+.3f}"
                    )
            lines.append(
                f"{label} ch={self.midi_channels[label] + 1} "
                f"pos=({state.position[0]:+.2f}, {state.position[1]:+.2f}, {state.position[2]:+.2f}) "
                f"button={'DOWN' if state.button_pressed else 'UP'} "
                f"zone={'PLAY' if state.is_on_play_side else 'NO-PLAY'} "
                f"inside={'yes' if state.last_inside else 'no'} "
                f"note={state.last_note_name} | {plane_text}"
            )
        if self.keyboard_controlled:
            controls = (
                "Keyboard ON: TAB hand, WASDQE move, IJKLUO rotate, SPACE draw, "
                f"C clear, F flip play side, arrows adjust steps, {channel_key_text} | "
                f"velocity uses LEFT {self.left_velocity_axis} ({ROTATION_AXIS_KEY_HINTS[self.left_velocity_axis]})"
            )
        else:
            controls = (
                "Keyboard OFF: drive with set_hand_pose(label, position, rotation_quaternion, button_pressed) | "
                f"{channel_key_text}"
            )
        if not self.allow_hand_switching:
            controls = f"{controls} | controls locked to {self.controlled_hand_label}"
        midi_text = self.midi_output_name if self.midi_output_name else "disconnected"
        right_channel_1_based = self.midi_channels["RIGHT"] + 1
        left_channel_1_based = self.midi_channels["LEFT"] + 1
        self.status_text.text = "\n".join(
            [
                f"Controlling: {self.controlled_hand_label} | MIDI: {midi_text} | {self.midi_channel_text()} | instrument={self.current_instrument}",
                f"scale: {self.scale_status_text()}",
                *self.external_status_lines,
                *lines,
                (
                    f"MIDI OUT: LEFT selects notes, RIGHT plays selected note on ch={right_channel_1_based} "
                    f"base_note={midi_note_name(self.midi_base_note)} velocity={self.current_attack_value}"
                ),
                (
                    f"MIDI OUT: LEFT rot -> CC101/CC7={self.current_volume_value}, RIGHT x -> CC10 pan={self.current_pan_value} on RIGHT ch={right_channel_1_based} | "
                    f"LEFT ch setting={left_channel_1_based}"
                ),
                f"move_step={self.velocity_step:.2f} rotate_step={self.rotation_step:.2f} volume={self.current_volume_value} attack={self.current_attack_value} pan={self.current_pan_value} reverb={self.current_reverb_value}",
                controls,
            ]
        )

    def _update_compact_status_text(self) -> None:
        active_note = self.active_midi_notes["RIGHT"]
        note_text = midi_note_name(active_note) if active_note is not None else ""
        self.status_text.text = note_text
        self.note_shadow_text.text = note_text

    def on_close(self, _event=None) -> None:
        self.close()

    def close(self) -> None:
        if hasattr(self, "timer"):
            self.timer.stop()
        self.close_midi()
        if hasattr(self, "haptics"):
            self.haptics.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FL Studio two-hand plane visualizer.")
    parser.add_argument("--external-control", action="store_true", help="Disable keyboard motion for import-style testing.")
    parser.add_argument("--no-midi", action="store_true", help="Disable MIDI output.")
    parser.add_argument("--midi-output-hint", default=MIDI_OUTPUT_HINT)
    parser.add_argument("--midi-base-note", "--middle-note", dest="midi_base_note", type=parse_midi_note, default=MIDI_BASE_NOTE)
    parser.add_argument(
        "--scale",
        type=normalize_scale_name,
        default="chromatic",
        help="Scale for note selection. Use --list-scales to see options.",
    )
    parser.add_argument("--list-scales", action="store_true", help="Print available scales and exit.")
    parser.add_argument("--midi-velocity", type=int, default=MIDI_VELOCITY)
    parser.add_argument(
        "--left-velocity-axis",
        choices=tuple(ROTATION_AXIS_TO_EULER_INDEX),
        default=LEFT_VELOCITY_AXIS_DEFAULT,
        help="Left-hand rotation axis used for MIDI velocity/volume.",
    )
    parser.add_argument(
        "--left-velocity-min-degrees",
        type=float,
        default=LEFT_VELOCITY_MIN_DEG,
        help="Left-hand angle that maps to velocity 1.",
    )
    parser.add_argument(
        "--left-velocity-max-degrees",
        type=float,
        default=LEFT_VELOCITY_MAX_DEG,
        help="Left-hand angle that maps to velocity 127.",
    )
    parser.add_argument(
        "--invert-left-velocity",
        action="store_true",
        help="Invert the left-hand rotation to velocity mapping.",
    )
    parser.add_argument("--left-midi-channel", type=int, default=HAND_MIDI_CHANNELS_1_BASED["LEFT"])
    parser.add_argument("--right-midi-channel", type=int, default=HAND_MIDI_CHANNELS_1_BASED["RIGHT"])
    parser.add_argument("--max-midi-channel", type=int, default=DEFAULT_MAX_MIDI_CHANNEL)
    parser.add_argument("--haptics-port", default=HAPTICS_PORT)
    parser.add_argument("--haptics-baud", type=int, default=HAPTICS_BAUD)
    parser.add_argument("--no-haptics", action="store_true")
    parser.add_argument("--require-inside-plane", action="store_true")
    parser.add_argument("--verbose-overlay", action="store_true", help="Show the full diagnostic VisPy text overlay.")
    parser.add_argument(
        "--performance-mode",
        action="store_true",
        help="Mirror the 3D visualizer for a TV/projector placed behind the performers.",
    )
    parser.add_argument("--update-hz", type=float, default=120.0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.list_scales:
        print("Available scales:")
        for scale_name in sorted(SCALE_INTERVALS):
            print(f"  {scale_name}")
        return 0

    visualizer = DualHandFLStudioVisualizer(
        keyboard_controlled=not args.external_control,
        enable_midi=not args.no_midi,
        midi_output_hint=args.midi_output_hint,
        midi_base_note=args.midi_base_note,
        scale_name=args.scale,
        midi_velocity=args.midi_velocity,
        left_velocity_axis=args.left_velocity_axis,
        left_velocity_min_degrees=args.left_velocity_min_degrees,
        left_velocity_max_degrees=args.left_velocity_max_degrees,
        invert_left_velocity=args.invert_left_velocity,
        midi_channels_1_based={"LEFT": args.left_midi_channel, "RIGHT": args.right_midi_channel},
        max_midi_channel=args.max_midi_channel,
        enable_haptics=not args.no_haptics,
        haptics_port=args.haptics_port,
        haptics_baud=args.haptics_baud,
        require_inside_plane_to_play=args.require_inside_plane,
        verbose_overlay=args.verbose_overlay,
        performance_mode=args.performance_mode,
        update_hz=args.update_hz,
    )

    print("Starting FL Studio Debug Visualizer...")
    print("Controls: TAB switches hand, WASDQE moves, IJKLUO rotates, SPACE draws, C clears, F flips play side.")
    print("Scale logic: LEFT hand selects notes; RIGHT hand plays the selected note on the +X side.")
    print(f"Scale: {args.scale} middle_note={midi_note_name(args.midi_base_note)}")
    print("MIDI channels: number keys set the selected hand's channel.")
    try:
        app.run()
    finally:
        visualizer.close()
    print("FL Studio Debug Visualizer closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
