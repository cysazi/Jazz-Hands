"""
Single-window two-hand visualizer for Jazz Hands plane writing and FL Studio MIDI.

This module is intentionally independent from the camera/mocap scripts. Run it
directly for keyboard testing, or import DualHandFLStudioVisualizer and feed it
external hand positions from mocap.
"""

from __future__ import annotations

import argparse
import atexit
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
from vispy import app, scene
from vispy.app import Timer
from vispy.io import read_mesh
from vispy.visuals.transforms import MatrixTransform

try:
    import mido
except ImportError:
    mido = None


CURRENT_FILEPATH = os.path.dirname(os.path.abspath(__file__))
RIGHT_HAND_OBJ_PATH = os.path.join(CURRENT_FILEPATH, "Visualization_Tests", "hand.obj")
LEFT_HAND_OBJ_PATH = os.path.join(CURRENT_FILEPATH, "Visualization_Tests", "lefthand.obj")
HAND_OBJ_PATH = RIGHT_HAND_OBJ_PATH
HAND_OBJ_PATHS = {
    "LEFT": LEFT_HAND_OBJ_PATH,
    "RIGHT": RIGHT_HAND_OBJ_PATH,
}

KEYBOARD_CONTROLLED = False
ENABLE_MIDI = True
MIDI_OUTPUT_HINT = "JazzHands (A)"
MIDI_BASE_NOTE = 60
MIDI_VELOCITY = 96
HAND_MIDI_CHANNELS_1_BASED = {"LEFT": 1, "RIGHT": 2}

POSITION_SCALE = 1.0
MODEL_SCALE = 0.02
MIN_PLANE_HALF_EXTENT = 0.03
MIN_DRAW_DISTANCE = 0.02
NOTE_SECTION_COUNT = 12
NOTE_NAMES = ("C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B")
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
    "RIGHT": rot_z(180.0),
}
HAND_MODEL_OFFSETS = {
    "LEFT": rot_z(180.0),
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
    button_pressed: bool = False
    last_button_pressed: bool = False
    drawing: bool = False
    draw_origin: np.ndarray | None = None
    plane: PlaneDefinition | None = None
    no_play_side_sign: float = 1.0
    play_side_sign: float = -1.0
    is_on_play_side: bool = False
    active_note_index: int | None = None
    last_note_name: str = "inactive"
    last_inside: bool = False
    last_offset: float = 0.0
    last_u_pct: float = 0.0
    last_v_pct: float = 0.0
    tracking_active: bool = True


def configure_vispy_backend() -> str:
    for backend in ("pyqt6", "pyside6", "tkinter"):
        try:
            app.use_app(backend)
            return backend
        except Exception:
            continue
    return app.use_app().backend_name


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
        require_inside_plane_to_play: bool = False,
        keyboard_buttons_enabled: bool = True,
        controlled_hand_label: str = "LEFT",
        allow_hand_switching: bool = True,
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
        channels = midi_channels_1_based or HAND_MIDI_CHANNELS_1_BASED
        self.midi_channels = {
            "LEFT": int(np.clip(int(channels.get("LEFT", 1)), 1, 16)) - 1,
            "RIGHT": int(np.clip(int(channels.get("RIGHT", 2)), 1, 16)) - 1,
        }
        self.require_inside_plane_to_play = bool(require_inside_plane_to_play)
        self.keyboard_buttons_enabled = bool(keyboard_buttons_enabled)
        self.allow_hand_switching = bool(allow_hand_switching)
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

        self.canvas = scene.SceneCanvas(
            keys="interactive",
            show=show,
            bgcolor="black",
            size=(1100, 800),
            title="Jazz Hands FL Studio Debug Visualizer",
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(
            fov=45,
            distance=3.0,
            elevation=25.0,
            azimuth=-35.0,
            center=(0, 0, 0),
        )
        scene.visuals.GridLines(scale=(0.25, 0.25), color=(0.25, 0.25, 0.25, 1.0), parent=self.view.scene)
        scene.visuals.XYZAxis(width=2, parent=self.view.scene)
        self._add_axis_labels()
        self.hand_meshes = {}
        self._setup_hand_meshes()
        self.plane_meshes: dict[str, object] = {}
        self.plane_outlines: dict[str, object] = {}
        self.plane_section_lines: dict[str, list[object]] = {label: [] for label in self.hand_labels}
        self.status_text = scene.visuals.Text(
            "",
            color="white",
            font_size=8,
            pos=(10, 10),
            anchor_x="left",
            anchor_y="bottom",
            parent=self.canvas.scene,
        )

        self.midi_out = None
        self.midi_output_name: str | None = None
        self.active_midi_notes: dict[str, int | None] = {"LEFT": None, "RIGHT": None}
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
        if button_pressed is not None:
            state.button_pressed = bool(button_pressed)
        state.tracking_active = True

    def set_hand_imu_state(
        self,
        label: str,
        rotation_quaternion: np.ndarray | tuple[float, float, float, float] | None = None,
        button_pressed: bool | None = None,
    ) -> None:
        state = self.hands[self._normalize_label(label)]
        if rotation_quaternion is not None:
            state.rotation_quaternion = normalize_quat(np.asarray(rotation_quaternion, dtype=np.float64))
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

    def _normalize_label(self, label: str) -> str:
        normalized = str(label).upper()
        if normalized not in self.hands:
            raise ValueError(f"unknown hand label: {label!r}")
        return normalized

    def _add_axis_labels(self) -> None:
        scene.visuals.Text("X", color="red", font_size=18, pos=(1.25, 0, 0), parent=self.view.scene)
        scene.visuals.Text("Y", color="green", font_size=18, pos=(0, 1.25, 0), parent=self.view.scene)
        scene.visuals.Text("Z", color="blue", font_size=18, pos=(0, 0, 1.25), parent=self.view.scene)

    def _setup_hand_meshes(self) -> None:
        colors = {
            "LEFT": (0.25, 0.75, 0.95, 0.82),
            "RIGHT": (0.95, 0.45, 0.25, 0.82),
        }
        for label in self.hand_labels:
            model_path = HAND_OBJ_PATHS[label]
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Could not find {label.lower()} hand model: {model_path}")
            vertices, faces, _normals, _texcoords = read_mesh(model_path)
            vertices = center_mesh_vertices(vertices)
            mesh = scene.visuals.Mesh(
                vertices=vertices,
                faces=faces,
                color=colors[label],
                shading="smooth",
                parent=self.view.scene,
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

        target = next((name for name in output_names if self.midi_output_hint.lower() in name.lower()), None)
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
        self.midi_out.send(mido.Message("note_on", note=int(note), velocity=self.midi_velocity, channel=int(channel)))

    def _send_note_off(self, note: int, channel: int) -> None:
        if self.midi_out is None:
            return
        self.midi_out.send(mido.Message("note_off", note=int(note), velocity=0, channel=int(channel)))

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

    def close_midi(self) -> None:
        for label in self.hand_labels:
            self._update_midi_note(label, None)
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
            velocity[1] += self.velocity_step
        if self.keys_down.get("S", False):
            velocity[1] -= self.velocity_step
        if self.keys_down.get("A", False):
            velocity[0] -= self.velocity_step
        if self.keys_down.get("D", False):
            velocity[0] += self.velocity_step
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
            self._update_hand_visual(label)

        self._update_status_text()
        self.canvas.update()

    def _apply_keyboard_motion(self, dt: float) -> None:
        for label in self.hand_labels:
            state = self.hands[label]
            if label == self.controlled_hand_label:
                state.position = state.position + state.velocity * dt

                angular_velocity = self.angular_velocity_vectors[label]
                rotation_angle = float(np.linalg.norm(angular_velocity) * dt)
                if rotation_angle > 1e-9 and dt > 0.0:
                    rotation_axis = angular_velocity / (rotation_angle / dt)
                    half_angle = rotation_angle / 2.0
                    delta_q = np.array(
                        [
                            math.cos(half_angle),
                            *(rotation_axis * math.sin(half_angle)),
                        ],
                        dtype=np.float64,
                    )
                    state.rotation_quaternion = normalize_quat(
                        quaternion_multiply(delta_q, state.rotation_quaternion)
                    )
            else:
                state.velocity = np.zeros(3, dtype=np.float64)

    def _update_plane_state(self, label: str) -> None:
        state = self.hands[label]
        pressed_edge = state.button_pressed and not state.last_button_pressed
        released_edge = (not state.button_pressed) and state.last_button_pressed

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
                        f"normal={AXIS_NAMES[definition.normal_axis]}, play side=+X"
                    )
                else:
                    print(f"[{label}] plane draw ignored: move farther before releasing SPACE")
            state.drawing = False
            state.draw_origin = None

        state.last_button_pressed = state.button_pressed

    def _compute_debug_plane(self, origin: np.ndarray, current_position: np.ndarray) -> PlaneDefinition | None:
        delta = current_position - origin
        if float(np.linalg.norm(delta)) < MIN_DRAW_DISTANCE:
            return None

        abs_delta = np.abs(delta)
        normal_axis = int(np.argmin(abs_delta))
        plane_axes = [axis for axis in range(3) if axis != normal_axis]
        max_extent = max(float(abs_delta[plane_axes[0]]), float(abs_delta[plane_axes[1]]), MIN_PLANE_HALF_EXTENT)

        return PlaneDefinition(
            center=origin.copy(),
            axis_u=plane_axes[0],
            axis_v=plane_axes[1],
            normal_axis=normal_axis,
            half_u=max_extent,
            half_v=max_extent,
        )

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
        print("[FL visualizer] play side is fixed to positive X")

    def _ensure_plane_visuals(self, label: str) -> None:
        if label not in self.plane_meshes:
            mesh = scene.visuals.Mesh(
                vertices=np.zeros((4, 3), dtype=np.float32),
                faces=PLANE_FACES,
                color=(0.25, 0.85, 1.0, 0.0),
                shading=None,
                parent=self.view.scene,
            )
            mesh.visible = False
            self.plane_meshes[label] = mesh

        if label not in self.plane_outlines:
            outline = scene.visuals.Line(
                pos=np.zeros((2, 3), dtype=np.float32),
                color=(0.75, 0.95, 1.0, 0.0),
                width=2,
                method="gl",
                parent=self.view.scene,
            )
            outline.visible = False
            self.plane_outlines[label] = outline

        while len(self.plane_section_lines[label]) < NOTE_SECTION_COUNT - 1:
            line = scene.visuals.Line(
                pos=np.zeros((2, 3), dtype=np.float32),
                color=(0.75, 0.95, 1.0, 0.0),
                width=1,
                method="gl",
                parent=self.view.scene,
            )
            line.visible = False
            self.plane_section_lines[label].append(line)

    def _plane_vertices(self, definition: PlaneDefinition) -> np.ndarray:
        center = definition.center
        vertices = np.tile(center.astype(np.float32), (4, 1))
        for idx, (u_sign, v_sign) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
            vertices[idx, definition.axis_u] = center[definition.axis_u] + u_sign * definition.half_u
            vertices[idx, definition.axis_v] = center[definition.axis_v] + v_sign * definition.half_v
            vertices[idx, definition.normal_axis] = center[definition.normal_axis]
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
        segments = []
        note_axis, note_half, other_axis, other_half = self._note_axis_and_half_extent(definition)
        axis_min = definition.center[note_axis] - note_half
        for split_index in range(1, NOTE_SECTION_COUNT):
            split_ratio = split_index / NOTE_SECTION_COUNT
            split_coord = axis_min + split_ratio * (2.0 * note_half)
            p0 = definition.center.copy()
            p1 = definition.center.copy()
            p0[note_axis] = split_coord
            p1[note_axis] = split_coord
            p0[other_axis] = definition.center[other_axis] - other_half
            p1[other_axis] = definition.center[other_axis] + other_half
            segments.append(np.array([p0, p1], dtype=np.float32))
        return segments

    def _update_plane_visuals(self, label: str, definition: PlaneDefinition, preview: bool) -> None:
        self._ensure_plane_visuals(label)
        colors = {
            "LEFT": {
                "fill": (0.30, 0.80, 1.00, 0.12 if preview else 0.24),
                "edge": (0.45, 0.88, 1.00, 0.75 if preview else 1.0),
                "split": (0.72, 0.93, 1.00, 0.45 if preview else 0.75),
            },
            "RIGHT": {
                "fill": (1.00, 0.48, 0.32, 0.12 if preview else 0.24),
                "edge": (1.00, 0.62, 0.45, 0.75 if preview else 1.0),
                "split": (1.00, 0.84, 0.70, 0.45 if preview else 0.75),
            },
        }[label]

        vertices = self._plane_vertices(definition) * POSITION_SCALE
        outline = np.vstack([vertices, vertices[0]])
        self.plane_meshes[label].set_data(vertices=vertices, faces=PLANE_FACES, color=colors["fill"])
        self.plane_meshes[label].visible = True
        self.plane_outlines[label].set_data(pos=outline, color=colors["edge"])
        self.plane_outlines[label].visible = True

        for line, segment in zip(self.plane_section_lines[label], self._section_line_segments(definition)):
            line.set_data(pos=segment * POSITION_SCALE, color=colors["split"])
            line.visible = True

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

    def _update_note_state(self, label: str) -> None:
        state = self.hands[label]
        if not state.tracking_active:
            state.active_note_index = None
            state.last_note_name = "tracking lost"
            state.is_on_play_side = False
            self._update_midi_note(label, None)
            return

        if state.plane is None:
            state.active_note_index = None
            state.last_note_name = "inactive"
            self._update_midi_note(label, None)
            return

        u_pct, v_pct, inside, _plane_normal_offset = self._plane_position_metrics(state.position, state.plane)
        play_offset = float(state.position[0] - state.plane.center[0])
        play_metric = play_offset
        if state.is_on_play_side:
            if play_metric <= PLAY_EXIT_THRESHOLD:
                state.is_on_play_side = False
        elif play_metric >= PLAY_ENTER_THRESHOLD:
            state.is_on_play_side = True

        note_index, note_name, note_axis, note_pct = self._note_section_from_position(state.position, state.plane)
        can_play = state.is_on_play_side and (inside or not self.require_inside_plane_to_play)
        state.active_note_index = note_index if can_play else None
        state.last_note_name = (
            f"{note_name} axis={note_axis} pos={note_pct:5.1f}%"
            if can_play
            else "inactive"
        )
        state.last_inside = inside
        state.last_offset = play_offset
        state.last_u_pct = u_pct
        state.last_v_pct = v_pct
        self._update_midi_note(label, state.active_note_index)

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
        lines = []
        for label in self.hand_labels:
            state = self.hands[label]
            plane_text = "plane=not set"
            if state.drawing:
                plane_text = "plane=drawing"
            elif state.plane is not None:
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
        controls = (
            "Keyboard ON: TAB hand, WASDQE move, IJKLUO rotate, SPACE draw, "
            "C clear, F flip play side, arrows adjust steps"
            if self.keyboard_controlled
            else "Keyboard OFF: drive with set_hand_pose(label, position, rotation_quaternion, button_pressed)"
        )
        if not self.allow_hand_switching:
            controls = f"{controls} | controls locked to {self.controlled_hand_label}"
        midi_text = self.midi_output_name if self.midi_output_name else "disconnected"
        self.status_text.text = "\n".join(
            [
                f"Controlling: {self.controlled_hand_label} | MIDI: {midi_text}",
                *self.external_status_lines,
                *lines,
                f"move_step={self.velocity_step:.2f} rotate_step={self.rotation_step:.2f}",
                controls,
            ]
        )

    def on_close(self, _event=None) -> None:
        self.close()

    def close(self) -> None:
        if hasattr(self, "timer"):
            self.timer.stop()
        self.close_midi()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FL Studio two-hand plane visualizer.")
    parser.add_argument("--external-control", action="store_true", help="Disable keyboard motion for import-style testing.")
    parser.add_argument("--no-midi", action="store_true", help="Disable MIDI output.")
    parser.add_argument("--midi-output-hint", default=MIDI_OUTPUT_HINT)
    parser.add_argument("--midi-base-note", type=int, default=MIDI_BASE_NOTE)
    parser.add_argument("--midi-velocity", type=int, default=MIDI_VELOCITY)
    parser.add_argument("--left-midi-channel", type=int, default=HAND_MIDI_CHANNELS_1_BASED["LEFT"])
    parser.add_argument("--right-midi-channel", type=int, default=HAND_MIDI_CHANNELS_1_BASED["RIGHT"])
    parser.add_argument("--require-inside-plane", action="store_true")
    parser.add_argument("--update-hz", type=float, default=120.0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    visualizer = DualHandFLStudioVisualizer(
        keyboard_controlled=not args.external_control,
        enable_midi=not args.no_midi,
        midi_output_hint=args.midi_output_hint,
        midi_base_note=args.midi_base_note,
        midi_velocity=args.midi_velocity,
        midi_channels_1_based={
            "LEFT": args.left_midi_channel,
            "RIGHT": args.right_midi_channel,
        },
        require_inside_plane_to_play=args.require_inside_plane,
        update_hz=args.update_hz,
    )

    print("Starting FL Studio Debug Visualizer...")
    print("Controls: TAB switches hand, WASDQE moves, IJKLUO rotates, SPACE draws, C clears, F flips play side.")
    try:
        app.run()
    finally:
        visualizer.close()
    print("FL Studio Debug Visualizer closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
