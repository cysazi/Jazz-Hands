from __future__ import annotations

import atexit
import math
import os
import queue
from dataclasses import dataclass

import numpy as np
from vispy import app, scene
from vispy.app import Timer
from vispy.io import read_mesh
from vispy.visuals.transforms import MatrixTransform

import JazzHandsKalman as jhk
from Visualization_Tests.Visualization_Helper_Functions import setup_canvas


CURRENT_FILEPATH = os.path.dirname(os.path.abspath(__file__))
HAND_OBJ_PATH = os.path.join(CURRENT_FILEPATH, "Visualization_Tests", "hand.obj")

HAND_MOVE_KEYS = {
    "LEFT": {"forward": "W", "backward": "S", "left": "A", "right": "D", "up": "Q", "down": "E"},
    "RIGHT": {"forward": "T", "backward": "G", "left": "F", "right": "H", "up": "R", "down": "Y"},
}
SETTINGS_ROTATION_KEYS = {
    "pitch_pos": "I",
    "pitch_neg": "K",
    "yaw_pos": "J",
    "yaw_neg": "L",
    "roll_pos": "U",
    "roll_neg": "O",
}
DRAW_KEY = "SPACE"
CLEAR_KEY = "C"
RESET_ROTATION_KEY = "V"
MIN_PLANE_HALF_EXTENT = 0.03
MIDI_OUTPUT_HINT = "JazzHands (A)"
MIDI_PRINT_THROTTLE_S = 0.25

PLANE_FACES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
AXIS_NAMES = ("X", "Y", "Z")


@dataclass
class PlaneDefinition:
    origin: np.ndarray
    normal_axis: int
    half_extent: float

    @property
    def plane_axes(self) -> list[int]:
        return [axis for axis in range(3) if axis != self.normal_axis]


class _NullReader:
    def send_correction(self, relay_id, device_id, correction):
        return

    def stop(self):
        return


class _ConsoleDawInterface:
    def __init__(self):
        self.previous_note = jhk.NoteData.blank_note()
        self._last_print_time = 0.0

    def play_note(self, note: jhk.NoteData):
        if note == self.previous_note:
            return
        print(
            "[MIDI DEBUG] "
            f"note={note.note} attack={note.attack} instrument={note.instrument} "
            f"volume={note.volume} stereo={note.stereo} reverb={note.reverb_mode}"
        )
        self.previous_note = note

    def stop_notes(self):
        if self.previous_note == jhk.NoteData.blank_note():
            return
        print("[MIDI DEBUG] stop")
        self.previous_note = jhk.NoteData.blank_note()


def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def _open_midi_port():
    try:
        output_names = jhk.mido.get_output_names()
    except Exception as exc:
        print(f"[MIDI] Could not enumerate MIDI outputs: {exc}")
        return None

    if not output_names:
        print("[MIDI] No MIDI output devices found. Open Logic or a virtual MIDI port first.")
        return None

    target = next((name for name in output_names if MIDI_OUTPUT_HINT.lower() in name.lower()), None)
    if target is None:
        target = output_names[0]
        print(f"[MIDI] Output containing '{MIDI_OUTPUT_HINT}' not found; using '{target}'.")
    else:
        print(f"[MIDI] Using '{target}'.")

    try:
        return jhk.mido.open_output(target)
    except Exception as exc:
        print(f"[MIDI] Failed to open '{target}': {exc}")
        return None


def _make_daw_interface():
    midi_port = _open_midi_port()
    if midi_port is None:
        print("[MIDI] Running with console note logging only.")
        return _ConsoleDawInterface()
    return jhk.DawInterface(port=midi_port)


class DualHandMusicDebugger:
    def __init__(self, glove_pair: jhk.GlovePair):
        self.glove_pair = glove_pair
        self.left_hand = glove_pair.left_hand
        self.right_hand = glove_pair.right_hand
        self.hand_labels = ("LEFT", "RIGHT")
        self.hands = {"LEFT": self.left_hand, "RIGHT": self.right_hand}

        self.canvas, self.view, _grid, _axes, _x_label, _y_label, _z_label = setup_canvas()
        self.view.camera.distance = 3.0
        self.view.camera.center = (0.0, 0.0, 0.0)
        self.canvas.events.key_press.connect(self.on_key_press)
        self.canvas.events.key_release.connect(self.on_key_release)
        self.canvas.events.close.connect(self.on_close)

        self.manual_positions = {
            "LEFT": np.array([-0.25, 0.0, 0.0], dtype=np.float64),
            "RIGHT": np.array([0.25, 0.0, 0.0], dtype=np.float64),
        }
        self.plane_definitions: dict[str, PlaneDefinition | None] = {label: None for label in self.hand_labels}
        self.draw_origins: dict[str, np.ndarray | None] = {label: None for label in self.hand_labels}
        self.draw_key_owner_label: str | None = None
        self.keys_down: dict[str, bool] = {}
        self.settings_hand_label = "LEFT"
        self.velocity_step = 0.5
        self.rotation_step = 1.5
        self.angular_velocity_vectors = {
            "LEFT": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "RIGHT": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        }

        self.hand_meshes: dict[str, scene.visuals.Mesh] = {}
        self.plane_meshes: dict[str, scene.visuals.Mesh] = {}
        self.plane_outlines: dict[str, scene.visuals.Line] = {}
        self.plane_section_lines: dict[str, list[scene.visuals.Line]] = {label: [] for label in self.hand_labels}
        self._setup_hand_meshes()
        self._setup_plane_visuals()

        self.status_text = scene.visuals.Text(
            "",
            color="white",
            font_size=7,
            pos=(10, 10),
            anchor_x="left",
            anchor_y="bottom",
            parent=self.canvas.scene,
        )

        self.timer = Timer("auto", connect=self.update, start=True)

    def _setup_hand_meshes(self):
        if not os.path.exists(HAND_OBJ_PATH):
            raise FileNotFoundError(f"Could not find model: {HAND_OBJ_PATH}")

        vertices, faces, _normals, _texcoords = read_mesh(HAND_OBJ_PATH)
        colors = {
            "LEFT": (0.25, 0.75, 0.95, 0.85),
            "RIGHT": (0.95, 0.45, 0.25, 0.85),
        }
        for label in self.hand_labels:
            mesh = scene.visuals.Mesh(
                vertices=vertices,
                faces=faces,
                color=colors[label],
                shading="smooth",
                parent=self.view.scene,
            )
            mesh.transform = MatrixTransform()
            self.hand_meshes[label] = mesh

    def _setup_plane_visuals(self):
        for label in self.hand_labels:
            self.plane_meshes[label] = scene.visuals.Mesh(
                vertices=np.zeros((4, 3), dtype=np.float32),
                faces=PLANE_FACES,
                color=(0.0, 0.0, 0.0, 0.0),
                shading=None,
                parent=self.view.scene,
            )
            self.plane_meshes[label].visible = False
            self.plane_outlines[label] = scene.visuals.Line(
                pos=np.zeros((5, 3), dtype=np.float32),
                color=(1.0, 1.0, 1.0, 0.0),
                width=2,
                method="gl",
                parent=self.view.scene,
            )
            self.plane_outlines[label].visible = False

    def _ensure_section_lines(self, label: str, count: int):
        lines = self.plane_section_lines[label]
        while len(lines) < count:
            line = scene.visuals.Line(
                pos=np.zeros((2, 3), dtype=np.float32),
                color=(1.0, 1.0, 1.0, 0.0),
                width=1,
                method="gl",
                parent=self.view.scene,
            )
            line.visible = False
            lines.append(line)
        for line in lines[count:]:
            line.visible = False

    def _plane_vertices(self, definition: PlaneDefinition) -> np.ndarray:
        axes = definition.plane_axes
        vertices = np.tile(definition.origin.astype(np.float32), (4, 1))
        corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        for idx, (u_sign, v_sign) in enumerate(corners):
            vertices[idx, axes[0]] = definition.origin[axes[0]] + u_sign * definition.half_extent
            vertices[idx, axes[1]] = definition.origin[axes[1]] + v_sign * definition.half_extent
            vertices[idx, definition.normal_axis] = definition.origin[definition.normal_axis]
        return vertices

    def _section_segments(self, label: str, definition: PlaneDefinition) -> list[np.ndarray]:
        hand = self.hands[label]
        axes = definition.plane_axes
        segments: list[np.ndarray] = []

        for split_index in range(1, hand.active_area_x_subsections):
            value = -definition.half_extent + split_index * (2.0 * definition.half_extent / hand.active_area_x_subsections)
            p0 = definition.origin.copy()
            p1 = definition.origin.copy()
            p0[axes[0]] += value
            p1[axes[0]] += value
            p0[axes[1]] -= definition.half_extent
            p1[axes[1]] += definition.half_extent
            segments.append(np.array([p0, p1], dtype=np.float32))

        for split_index in range(1, hand.active_area_y_subsections):
            value = -definition.half_extent + split_index * (2.0 * definition.half_extent / hand.active_area_y_subsections)
            p0 = definition.origin.copy()
            p1 = definition.origin.copy()
            p0[axes[1]] += value
            p1[axes[1]] += value
            p0[axes[0]] -= definition.half_extent
            p1[axes[0]] += definition.half_extent
            segments.append(np.array([p0, p1], dtype=np.float32))

        return segments

    def _update_plane_visual(self, label: str, definition: PlaneDefinition, preview: bool):
        vertices = self._plane_vertices(definition)
        outline = np.vstack([vertices, vertices[0]])
        if label == "LEFT":
            fill = (0.25, 0.75, 0.95, 0.12 if preview else 0.22)
            edge = (0.35, 0.85, 1.0, 0.65 if preview else 1.0)
            split = (0.75, 0.95, 1.0, 0.45 if preview else 0.75)
        else:
            fill = (0.95, 0.45, 0.25, 0.12 if preview else 0.22)
            edge = (1.0, 0.62, 0.45, 0.65 if preview else 1.0)
            split = (1.0, 0.84, 0.70, 0.45 if preview else 0.75)

        self.plane_meshes[label].set_data(vertices=vertices, faces=PLANE_FACES, color=fill)
        self.plane_meshes[label].visible = True
        self.plane_outlines[label].set_data(pos=outline, color=edge)
        self.plane_outlines[label].visible = True

        segments = self._section_segments(label, definition)
        self._ensure_section_lines(label, len(segments))
        for line, segment in zip(self.plane_section_lines[label], segments):
            line.set_data(pos=segment, color=split)
            line.visible = True

    def _hide_plane_visual(self, label: str):
        self.plane_meshes[label].visible = False
        self.plane_outlines[label].visible = False
        for line in self.plane_section_lines[label]:
            line.visible = False

    def _camera_relative_basis(self) -> tuple[np.ndarray, np.ndarray]:
        azimuth_deg = float(getattr(self.view.camera, "azimuth", 0.0))
        azimuth_rad = np.deg2rad(azimuth_deg)
        forward = np.array([-np.sin(azimuth_rad), np.cos(azimuth_rad), 0.0], dtype=np.float64)
        right = np.array([np.cos(azimuth_rad), np.sin(azimuth_rad), 0.0], dtype=np.float64)
        return forward, right

    def _compute_linear_velocity(self, label: str) -> np.ndarray:
        keys = HAND_MOVE_KEYS[label]
        forward, right = self._camera_relative_basis()
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        if self.keys_down.get(keys["forward"], False):
            velocity += forward
        if self.keys_down.get(keys["backward"], False):
            velocity -= forward
        if self.keys_down.get(keys["left"], False):
            velocity -= right
        if self.keys_down.get(keys["right"], False):
            velocity += right
        if self.keys_down.get(keys["up"], False):
            velocity += up
        if self.keys_down.get(keys["down"], False):
            velocity -= up

        norm = float(np.linalg.norm(velocity))
        if norm > 1e-9:
            velocity = velocity / norm * self.velocity_step
        return velocity

    def _switch_settings_hand(self, direction: int):
        current_index = self.hand_labels.index(self.settings_hand_label)
        self.settings_hand_label = self.hand_labels[(current_index + direction) % len(self.hand_labels)]
        print(f"Settings hand switched to {self.settings_hand_label}.")

    def _adjust_octave(self, delta: int):
        current = int(self.glove_pair.current_octave)
        self.glove_pair.current_octave = int(np.clip(current + delta, 2, 7))
        print(f"Current octave: {self.glove_pair.current_octave}")

    def _cycle_instrument(self, direction: int):
        cycle = jhk.INSTRUMENT_CYCLE
        current = self.right_hand.instrument if self.right_hand.instrument in cycle else cycle[0]
        next_index = (cycle.index(current) + direction) % len(cycle)
        self.right_hand.instrument = cycle[next_index]
        print(f"Right hand instrument: {self.right_hand.instrument}")

    def _reset_rotation(self, label: str):
        hand = self.hands[label]
        hand.rotation_quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        hand.rotation_euler = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        print(f"{label} rotation reset.")

    def _relative_position(self, label: str) -> np.ndarray:
        definition = self.plane_definitions[label]
        origin = self.draw_origins[label]
        if definition is not None:
            return self.manual_positions[label] - definition.origin
        if origin is not None:
            return self.manual_positions[label] - origin
        return self.manual_positions[label].copy()

    def _compute_plane_from_relative_position(self, label: str, relative_position: np.ndarray) -> PlaneDefinition:
        abs_position = np.abs(relative_position)
        normal_axis = int(np.argmin(abs_position))
        plane_axes = [axis for axis in range(3) if axis != normal_axis]
        half_extent = max(
            float(abs_position[plane_axes[0]]),
            float(abs_position[plane_axes[1]]),
            MIN_PLANE_HALF_EXTENT,
        )
        origin = self.draw_origins[label]
        if origin is None:
            origin = self.manual_positions[label].copy()
        return PlaneDefinition(origin=origin.copy(), normal_axis=normal_axis, half_extent=half_extent)

    def _begin_draw(self, label: str):
        hand = self.hands[label]
        self.draw_origins[label] = self.manual_positions[label].copy()
        self.plane_definitions[label] = None
        hand.glove_state = 1
        hand.is_UWB_calibrated = True
        hand.button_pressed = True
        hand.in_active_area = False
        hand.position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        hand.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self._stop_current_note()
        print(f"{label} plane origin captured: {self.draw_origins[label]}")

    def _finish_draw(self, label: str):
        hand = self.hands[label]
        hand.button_pressed = False
        origin = self.draw_origins[label]
        if origin is None:
            return

        relative_position = self.manual_positions[label] - origin
        definition = self._compute_plane_from_relative_position(label, relative_position)
        self.plane_definitions[label] = definition
        hand.plane_normal_axis = definition.normal_axis
        hand.active_area_half_extents = np.full(3, definition.half_extent, dtype=np.float64)
        hand.active_area_half_extents[definition.normal_axis] = 0.0
        hand.position = relative_position.astype(np.float64)
        hand.glove_state = 2
        hand.in_active_area = False
        self._update_plane_visual(label, definition, preview=False)
        self._stop_current_note()
        print(
            f"{label} plane committed: normal={AXIS_NAMES[definition.normal_axis]} "
            f"size={definition.half_extent * 2.0:.2f}m"
        )

    def _clear_hand_plane(self, label: str):
        hand = self.hands[label]
        hand.glove_state = 0
        hand.is_UWB_calibrated = False
        hand.button_pressed = False
        hand.in_active_area = False
        hand.x_section = 0
        hand.y_section = 0
        hand.active_area_half_extents = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.plane_definitions[label] = None
        self.draw_origins[label] = None
        self._hide_plane_visual(label)
        self._stop_current_note()
        print(f"{label} plane cleared.")

    def _update_motion(self):
        for label in self.hand_labels:
            self.angular_velocity_vectors[label] = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        keys = SETTINGS_ROTATION_KEYS
        angular_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        if self.keys_down.get(keys["pitch_pos"], False):
            angular_velocity[1] += self.rotation_step
        if self.keys_down.get(keys["pitch_neg"], False):
            angular_velocity[1] -= self.rotation_step
        if self.keys_down.get(keys["yaw_pos"], False):
            angular_velocity[2] += self.rotation_step
        if self.keys_down.get(keys["yaw_neg"], False):
            angular_velocity[2] -= self.rotation_step
        if self.keys_down.get(keys["roll_pos"], False):
            angular_velocity[0] += self.rotation_step
        if self.keys_down.get(keys["roll_neg"], False):
            angular_velocity[0] -= self.rotation_step
        self.angular_velocity_vectors[self.settings_hand_label] = angular_velocity

    def on_key_press(self, event):
        if not event.key:
            return
        key_name = event.key.name.upper()
        was_held = self.keys_down.get(key_name, False)
        self.keys_down[key_name] = True
        self._update_motion()

        if key_name == "LEFT":
            if not was_held:
                self._switch_settings_hand(-1)
            return
        if key_name == "RIGHT":
            if not was_held:
                self._switch_settings_hand(1)
            return
        if key_name == DRAW_KEY and not was_held:
            self.draw_key_owner_label = self.settings_hand_label
            self._begin_draw(self.draw_key_owner_label)
            return
        if key_name == CLEAR_KEY and not was_held:
            self._clear_hand_plane(self.settings_hand_label)
            return
        if key_name == RESET_ROTATION_KEY and not was_held:
            self._reset_rotation(self.settings_hand_label)
            return
        if key_name == "UP" and not was_held:
            self._adjust_octave(1)
            return
        if key_name == "DOWN" and not was_held:
            self._adjust_octave(-1)
            return
        if key_name == "Z" and not was_held:
            self._cycle_instrument(-1)
            return
        if key_name == "X" and not was_held:
            self._cycle_instrument(1)
            return

    def on_key_release(self, event):
        if not event.key:
            return
        key_name = event.key.name.upper()
        self.keys_down[key_name] = False
        self._update_motion()

        if key_name == DRAW_KEY and self.draw_key_owner_label is not None:
            self._finish_draw(self.draw_key_owner_label)
            self.draw_key_owner_label = None

    def _apply_rotation(self, label: str, dt: float):
        hand = self.hands[label]
        angular_velocity = self.angular_velocity_vectors[label]
        rotation_angle = float(np.linalg.norm(angular_velocity) * dt)
        if rotation_angle <= 1e-9 or dt <= 0.0:
            return

        rotation_axis = angular_velocity / (rotation_angle / dt)
        half_angle = rotation_angle / 2.0
        delta_q = np.array(
            [
                math.cos(half_angle),
                *(rotation_axis * math.sin(half_angle)),
            ],
            dtype=np.float64,
        )
        hand.rotation_quaternion = _normalize_quaternion(
            jhk.quaternion_multiply(delta_q, np.asarray(hand.rotation_quaternion, dtype=np.float64))
        )
        hand.rotation_euler = np.array(jhk.quat_to_euler_deg(hand.rotation_quaternion), dtype=np.float64)

    def _update_hand_logic_position(self, label: str):
        hand = self.hands[label]
        relative_position = self._relative_position(label)
        hand.position = relative_position.astype(np.float64)

        if hand.glove_state == 1 and self.draw_origins[label] is not None:
            preview = self._compute_plane_from_relative_position(label, relative_position)
            hand.plane_normal_axis = preview.normal_axis
            self._update_plane_visual(label, preview, preview=True)

        if hand.glove_state == 2:
            hand.get_section_from_position()
        else:
            hand.in_active_area = False

    def _mesh_transform_matrix(self, label: str) -> np.ndarray:
        hand = self.hands[label]
        q_current = np.asarray(hand.rotation_quaternion, dtype=np.float64)
        rotation_matrix = jhk.quaternion_to_transform_matrix(q_current, np.array([0.0, 0.0, 0.0]))[:3, :3]
        rotation_matrix = jhk.FRAME_MAP @ rotation_matrix @ jhk.FRAME_MAP.T
        rotation_matrix = rotation_matrix @ jhk.MODEL_OFFSET

        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rotation_matrix * jhk.MODEL_SCALE
        transform[3, :3] = np.asarray(self.manual_positions[label], dtype=np.float32) * jhk.POSITION_SCALE
        return transform

    def _update_visuals(self):
        for label in self.hand_labels:
            self.hand_meshes[label].transform.matrix = self._mesh_transform_matrix(label)

    def _stop_current_note(self):
        blank_note = jhk.NoteData.blank_note()
        if self.glove_pair.daw_interface.previous_note == blank_note:
            return
        self.glove_pair.daw_interface.stop_notes()
        self.glove_pair.daw_interface.previous_note = blank_note

    def _run_music_logic(self):
        both_ready = self.left_hand.glove_state == 2 and self.right_hand.glove_state == 2
        both_active = self.left_hand.in_active_area and self.right_hand.in_active_area
        if both_ready and both_active:
            self.glove_pair.section_to_note()
            self.glove_pair.play_note()
        else:
            self._stop_current_note()

    def _status_for_hand(self, label: str) -> str:
        hand = self.hands[label]
        definition = self.plane_definitions[label]
        if definition is None:
            plane = "plane=not set"
        else:
            plane = (
                f"plane={AXIS_NAMES[definition.normal_axis]} normal "
                f"size={definition.half_extent * 2.0:.2f}m"
            )
        return (
            f"{label}: state={hand.glove_state} active={hand.in_active_area} "
            f"section=({hand.x_section},{hand.y_section}) "
            f"pos={np.array2string(hand.position, precision=2, suppress_small=True)} {plane}"
        )

    def _update_status(self):
        note = self.glove_pair.daw_interface.previous_note
        self.status_text.text = "\n".join(
            [
                self._status_for_hand("LEFT"),
                self._status_for_hand("RIGHT"),
                (
                    f"Music: note={note.note} volume={note.volume} reverb={note.reverb_mode} "
                    f"pan={note.stereo} attack={note.attack} instrument={self.right_hand.instrument} "
                    f"octave={self.glove_pair.current_octave}"
                ),
                f"Settings hand: {self.settings_hand_label} (Left/Right arrows)",
                "Controls: LEFT move=WASDQE, RIGHT move=TFGHRY, IJKLUO rotate selected hand",
                "SPACE draw/release plane, C clear, V reset rotation, Up/Down octave, Z/X instrument",
            ]
        )

    def update(self, event):
        dt = float(event.dt) if event is not None and event.dt is not None else 0.0

        for label in self.hand_labels:
            hand = self.hands[label]
            hand.velocity = self._compute_linear_velocity(label)
            self.manual_positions[label] += hand.velocity * dt
            self._apply_rotation(label, dt)
            self._update_hand_logic_position(label)

        self._run_music_logic()
        self._update_visuals()
        self._update_status()
        self.canvas.update()

    def on_close(self, event):
        self._stop_current_note()
        self.timer.stop()
        app.quit()


def build_glove_pair() -> jhk.GlovePair:
    daw_interface = _make_daw_interface()
    reader = _NullReader()
    relay_queue: queue.Queue = queue.Queue()
    left_hand = jhk.LeftHand(
        device_id=1,
        active_area_x_subsections=3,
        active_area_y_subsections=12,
    )
    right_hand = jhk.RightHand(
        device_id=2,
        active_area_x_subsections=5,
        active_area_y_subsections=8,
        instrument="Synth",
    )
    return jhk.GlovePair(
        left_hand=left_hand,
        right_hand=right_hand,
        relay_id=1,
        reader=reader,
        relay_queue=relay_queue,
        daw_interface=daw_interface,
    )


def main():
    jhk.DEBUG_LOGGING = False
    jhk.MIDI_DEBUG_LOGGING = False

    glove_pair = build_glove_pair()
    atexit.register(
        lambda: glove_pair.daw_interface.stop_notes()
        if glove_pair.daw_interface.previous_note != jhk.NoteData.blank_note()
        else None
    )

    print("Starting dual-hand current-architecture music debugger...")
    print("Controls:")
    print("  - LEFT hand move: WASDQE")
    print("  - RIGHT hand move: TFGHRY")
    print("  - Settings hand select: Left/Right arrows")
    print("  - Selected-hand controls: IJKLUO rotate, SPACE draw/release plane, C clear, V reset rotation")
    print("  - Music controls: Up/Down octave, Z/X instrument")
    print("  - Draw both hand planes, then move both hands to the positive side of their planes to play.")

    DualHandMusicDebugger(glove_pair)
    app.run()
    print("Dual-hand music debugger closed.")


if __name__ == "__main__":
    main()
