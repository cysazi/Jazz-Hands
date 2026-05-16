import numpy as np
from vispy import app, scene
from vispy.app import Timer
import os
from vispy.io import read_mesh
from vispy.visuals.transforms import MatrixTransform
import math
import time
from dataclasses import dataclass
import atexit

# Import necessary components from your main script and helpers
import JazzHandsKalman as jhk
from Visualization_Tests.Visualization_Helper_Functions import setup_canvas

try:
    import mido
except ImportError:
    mido = None

# region Constants from Accel_Filtering_Test.py
CURRENT_FILEPATH = os.path.dirname(os.path.abspath(__file__))
HAND_OBJ_PATH = os.path.join(CURRENT_FILEPATH, "Visualization_Tests", "hand.obj")

POSITION_SCALE = 1.0
MODEL_SCALE = 0.02

FRAME_MAP = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
], dtype=np.float32)

MIN_PLANE_HALF_EXTENT = 0.03
PREVIEW_UPDATE_HZ = 45
PREVIEW_MIN_INTERVAL_SECONDS = 1.0 / PREVIEW_UPDATE_HZ
NOTE_SECTION_COUNT = 12
NOTE_NAMES = ("C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B")
AXIS_NAMES = ("X", "Y", "Z")
WORLD_UP_AXIS = 2  # Z
SIDE_INFER_OFFSET_EPS = 0.01
SIDE_INFER_VELOCITY_EPS = 0.03
PLAY_ENTER_THRESHOLD = 0.02
PLAY_EXIT_THRESHOLD = 0.01
PLANE_FACES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
MIDI_OUTPUT_HINT = "JazzHands (A)"
MIDI_BASE_NOTE = 60  # C4
MIDI_VELOCITY_MAX = 100
MIDI_VELOCITY_MIN = 0
MIDI_NOTE_MIN = 0
MIDI_NOTE_MAX = 127
PALM_NORMAL_LOCAL = np.array([0.0, 0.0, 1.0], dtype=np.float64)
MIDI_EXPRESSION_CC = 11  # Expression (continuous dynamics)
MIDI_VOLUME_CC = 7       # Channel volume
MIDI_MODWHEEL_CC = 1     # Mod wheel
MIDI_DEFAULT_CHANNEL = 0  # MIDI channels are 0-15 (0 == channel 1)
MIDI_MIN_OCTAVE_OFFSET = math.ceil((MIDI_NOTE_MIN - MIDI_BASE_NOTE) / 12.0)
MIDI_MAX_OCTAVE_OFFSET = math.floor((MIDI_NOTE_MAX - (MIDI_BASE_NOTE + NOTE_SECTION_COUNT - 1)) / 12.0)
CHORD_INTERVALS: dict[str, tuple[int, ...]] = {
    "Single": (0,),
    "Major": (0, 4, 7),
    "Minor": (0, 3, 7),
    "Sus2": (0, 2, 7),
    "Sus4": (0, 5, 7),
    "Maj7": (0, 4, 7, 11),
}
CHORD_TYPE_ORDER = tuple(CHORD_INTERVALS.keys())

HAND_MOVE_KEYS = {
    "LEFT": {"forward": "W", "backward": "S", "left": "A", "right": "D", "up": "Q", "down": "E"},
    "RIGHT": {"forward": "T", "backward": "G", "left": "F", "right": "H", "up": "R", "down": "Y"},
}
SETTINGS_ROTATION_KEYS = {"pitch_pos": "I", "pitch_neg": "K", "yaw_pos": "J", "yaw_neg": "L", "roll_pos": "U", "roll_neg": "O"}
SETTINGS_DRAW_KEY = "SPACE"
SETTINGS_CLEAR_KEY = "C"
SETTINGS_CALIBRATE_KEY = "V"


def rot_x(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ], dtype=np.float64)


MODEL_OFFSET = rot_x(-90.0)
# endregion


@dataclass
class PlaneDefinition:
    center: np.ndarray
    axis_u: int
    axis_v: int
    normal_axis: int
    half_u: float
    half_v: float


class DualHandDebugVisualizer:
    def __init__(self, glove_pair):
        self.glove_pair = glove_pair
        self.left_hand = self.glove_pair.left_hand
        self.right_hand = self.glove_pair.right_hand
        self.hand_labels = ("LEFT", "RIGHT")
        self.hands = {
            "LEFT": self.left_hand,
            "RIGHT": self.right_hand,
        }
        self.settings_hand_label = "LEFT"
        self.draw_key_owner_label: str | None = None

        self.canvas, self.view, _, _, _, _, _ = setup_canvas()
        self.view.camera.distance = 3

        # --- Load Hand Mesh ---
        if not os.path.exists(HAND_OBJ_PATH):
            raise FileNotFoundError(f"Could not find model: {HAND_OBJ_PATH}")
        vertices, faces, _normals, _texcoords = read_mesh(HAND_OBJ_PATH)
        self.hand_meshes: dict[str, scene.visuals.Mesh] = {}
        hand_colors = {
            "LEFT": (0.25, 0.75, 0.95, 0.80),
            "RIGHT": (0.95, 0.45, 0.25, 0.80),
        }
        for label in self.hand_labels:
            mesh = scene.visuals.Mesh(
                vertices=vertices,
                faces=faces,
                color=hand_colors[label],
                shading="smooth",
                parent=self.view.scene,
            )
            mesh.transform = MatrixTransform()
            self.hand_meshes[label] = mesh
        # --- End Hand Mesh ---

        self.canvas.events.key_press.connect(self.on_key_press)
        self.canvas.events.key_release.connect(self.on_key_release)
        self.canvas.events.close.connect(self.on_close)

        self.timer = Timer('auto', connect=self.update, start=True)

        # --- State tracking for keyboard input ---
        self.velocity_step = 0.5
        self.rotation_step = 1.5  # Radians per second
        self.keys_down = {}
        self.angular_velocity_vectors = {
            "LEFT": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "RIGHT": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        }

        # --- Plane drawing state ---
        self.last_button_states = {label: False for label in self.hand_labels}
        self.button_transition_counts = {label: 0 for label in self.hand_labels}
        self.press_centers: dict[str, np.ndarray | None] = {label: None for label in self.hand_labels}
        self.last_preview_update_times: dict[str, float | None] = {label: None for label in self.hand_labels}

        self.plane_definitions: dict[str, PlaneDefinition | None] = {label: None for label in self.hand_labels}
        self.plane_mesh_visuals: dict[str, scene.visuals.Mesh | None] = {label: None for label in self.hand_labels}
        self.plane_outline_visuals: dict[str, scene.visuals.Line | None] = {label: None for label in self.hand_labels}
        self.plane_section_lines: dict[str, list[scene.visuals.Line]] = {label: [] for label in self.hand_labels}
        self.last_plane_sizes = {label: np.array([0.0, 0.0], dtype=np.float32) for label in self.hand_labels}
        self.last_plane_axes = {label: ("X", "Y") for label in self.hand_labels}
        self.no_play_side_signs = {label: 1.0 for label in self.hand_labels}
        self.play_side_signs = {label: -1.0 for label in self.hand_labels}
        self.is_on_play_sides = {label: False for label in self.hand_labels}
        self.active_note_indices: dict[str, int | None] = {label: None for label in self.hand_labels}
        self.plane_color_theme = {
            "LEFT": {
                "fill_preview": (0.30, 0.80, 1.00, 0.12),
                "fill_set": (0.30, 0.80, 1.00, 0.22),
                "edge_preview": (0.45, 0.88, 1.00, 0.85),
                "edge_set": (0.30, 0.80, 1.00, 1.00),
                "split_preview": (0.72, 0.93, 1.00, 0.50),
                "split_set": (0.72, 0.93, 1.00, 0.80),
            },
            "RIGHT": {
                "fill_preview": (1.00, 0.48, 0.32, 0.12),
                "fill_set": (1.00, 0.48, 0.32, 0.22),
                "edge_preview": (1.00, 0.62, 0.45, 0.85),
                "edge_set": (1.00, 0.48, 0.32, 1.00),
                "split_preview": (1.00, 0.84, 0.70, 0.50),
                "split_set": (1.00, 0.84, 0.70, 0.80),
            },
        }
        self.midi_out = None
        self.midi_output_name: str | None = None
        self.hand_midi_channels = {"LEFT": MIDI_DEFAULT_CHANNEL, "RIGHT": min(MIDI_DEFAULT_CHANNEL + 1, 15)}
        self.current_octave_offset: int = 0
        self.chord_mode_enabled = False
        self.chord_type_index = 1  # Start at Major when chord mode is enabled.
        self.last_midi_notes_by_hand = {label: tuple() for label in self.hand_labels}
        self.last_midi_channels_by_hand = {label: None for label in self.hand_labels}
        self.last_midi_velocities = {label: 0 for label in self.hand_labels}
        self.current_midi_velocities = {label: 0 for label in self.hand_labels}
        self.last_expression_values: dict[int, int] = {}
        self.last_channel_pressure_values: dict[int, int] = {}
        self.last_cc_values: dict[tuple[int, int], int] = {}
        self.velocity_reference_quaternions = {
            label: normalize_quat(np.asarray(self.hands[label].rotation_quaternion, dtype=np.float64))
            for label in self.hand_labels
        }
        self._setup_midi_output()
        atexit.register(self._close_midi_output)

        self.status_text = scene.visuals.Text(
            "Plane: not set",
            color="white",
            font_size=5,
            pos=(10, 10),
            anchor_x="left",
            anchor_y="bottom",
            parent=self.canvas.scene,
        )

    def _setup_midi_output(self):
        if mido is None:
            print("[MIDI] mido not installed; MIDI output disabled.")
            return
        try:
            output_names = mido.get_output_names()
        except Exception as exc:
            print(f"[MIDI] Could not enumerate MIDI outputs: {exc}")
            return

        target = next((name for name in output_names if MIDI_OUTPUT_HINT.lower() in name.lower()), None)
        if target is None:
            print(f"[MIDI] Output containing '{MIDI_OUTPUT_HINT}' not found. Available: {output_names}")
            return
        try:
            self.midi_out = mido.open_output(target)
            self.midi_output_name = target
            print(f"[MIDI] Connected output: {target}")
        except Exception as exc:
            print(f"[MIDI] Failed to open output '{target}': {exc}")
            self.midi_out = None
            self.midi_output_name = None

    def _send_note_off(self, note: int, channel: int):
        if self.midi_out is None:
            return
        channel_out = int(channel)
        self.midi_out.send(mido.Message("note_off", note=note, velocity=0, channel=channel_out))

    def _send_note_on(self, note: int, velocity: int, channel: int):
        if self.midi_out is None:
            return
        channel_out = int(channel)
        note_velocity = max(1, int(velocity))
        self.midi_out.send(mido.Message("note_on", note=note, velocity=note_velocity, channel=channel_out))

    def _send_cc_if_changed(self, control: int, value: int, channel: int):
        if self.midi_out is None:
            return
        channel_out = int(channel)
        value_i = int(np.clip(value, 0, 127))
        key = (channel_out, int(control))
        previous = self.last_cc_values.get(key)
        if previous == value_i:
            return
        self.midi_out.send(
            mido.Message("control_change", channel=channel_out, control=control, value=value_i)
        )
        self.last_cc_values[key] = value_i

    def _send_channel_pressure_if_changed(self, value: int, channel: int):
        if self.midi_out is None:
            return
        channel_out = int(channel)
        value_i = int(np.clip(value, 0, 127))
        previous = self.last_channel_pressure_values.get(channel_out)
        if previous == value_i:
            return
        self.midi_out.send(mido.Message("aftertouch", channel=channel_out, value=value_i))
        self.last_channel_pressure_values[channel_out] = value_i

    def _selected_chord_name(self) -> str:
        return CHORD_TYPE_ORDER[self.chord_type_index]

    def _build_output_notes(self, note_index: int | None) -> tuple[int, ...]:
        if note_index is None:
            return ()

        root = MIDI_BASE_NOTE + (self.current_octave_offset * 12) + note_index
        intervals = CHORD_INTERVALS[self._selected_chord_name()] if self.chord_mode_enabled else CHORD_INTERVALS["Single"]

        note_set = {
            int(np.clip(root + interval, MIDI_NOTE_MIN, MIDI_NOTE_MAX))
            for interval in intervals
        }
        return tuple(sorted(note_set))

    def _send_notes_off(self, notes: tuple[int, ...], channel: int):
        for note in notes:
            self._send_note_off(note, channel=channel)

    def _update_midi_note(self, hand_label: str, note_index: int | None, note_velocity: int = 0):
        if self.midi_out is None:
            return
        channel = self.hand_midi_channels[hand_label]
        desired_notes = self._build_output_notes(note_index)
        last_notes = self.last_midi_notes_by_hand[hand_label]
        last_channel = self.last_midi_channels_by_hand[hand_label]
        if desired_notes == last_notes and last_channel == channel:
            return

        if last_notes:
            off_channel = last_channel if last_channel is not None else channel
            self._send_notes_off(last_notes, channel=off_channel)
        if desired_notes:
            for note in desired_notes:
                self._send_note_on(note, note_velocity, channel=channel)
            self._update_live_expression(hand_label, True, note_velocity)
            self.last_midi_velocities[hand_label] = note_velocity
            self.last_midi_channels_by_hand[hand_label] = channel
        else:
            self.last_midi_velocities[hand_label] = 0
            self.last_midi_channels_by_hand[hand_label] = None
        self.last_midi_notes_by_hand[hand_label] = desired_notes

    def _close_midi_output(self):
        if self.midi_out is not None:
            for label in self.hand_labels:
                last_notes = self.last_midi_notes_by_hand[label]
                if last_notes:
                    off_channel = self.last_midi_channels_by_hand[label]
                    if off_channel is None:
                        off_channel = self.hand_midi_channels[label]
                    self._send_notes_off(last_notes, channel=off_channel)
                    self.last_midi_notes_by_hand[label] = ()
                    self.last_midi_channels_by_hand[label] = None
            self.midi_out.close()
            self.midi_out = None

    def _update_live_expression(self, hand_label: str, note_active: bool, velocity_0_to_100: int):
        if self.midi_out is None:
            return

        if not note_active:
            return

        channel = self.hand_midi_channels[hand_label]
        expression = int(round(np.clip(velocity_0_to_100, 0, 100) * 127.0 / 100.0))
        if expression == self.last_expression_values.get(channel):
            # Keep CC / aftertouch in sync even when legacy expression field did not change.
            pass

        # Compatibility bundle: different plugins/hosts honor different live controllers.
        self._send_cc_if_changed(MIDI_EXPRESSION_CC, expression, channel=channel)
        self._send_cc_if_changed(MIDI_VOLUME_CC, expression, channel=channel)
        self._send_cc_if_changed(MIDI_MODWHEEL_CC, expression, channel=channel)
        self._send_channel_pressure_if_changed(expression, channel=channel)
        self.last_expression_values[channel] = expression

    def _select_midi_channel(self, hand_label: str, channel_1_based: int):
        channel_zero_based = max(0, min(15, int(channel_1_based) - 1))
        if channel_zero_based == self.hand_midi_channels[hand_label]:
            return

        last_notes = self.last_midi_notes_by_hand[hand_label]
        if last_notes:
            off_channel = self.last_midi_channels_by_hand[hand_label]
            if off_channel is None:
                off_channel = self.hand_midi_channels[hand_label]
            self._send_notes_off(last_notes, channel=off_channel)
            self.last_midi_notes_by_hand[hand_label] = ()
            self.last_midi_channels_by_hand[hand_label] = None
            self.last_midi_velocities[hand_label] = 0

        self.hand_midi_channels[hand_label] = channel_zero_based
        print(f"Switched {hand_label} instrument MIDI channel to {self.hand_midi_channels[hand_label] + 1}.")

    def _midi_channel_from_row_key(self, key_name: str) -> int | None:
        key_to_channel = {
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "0": 10,
        }
        return key_to_channel.get(key_name)

    def _adjust_octave(self, delta: int):
        proposed = self.current_octave_offset + int(delta)
        clamped = max(MIDI_MIN_OCTAVE_OFFSET, min(MIDI_MAX_OCTAVE_OFFSET, proposed))
        if clamped == self.current_octave_offset:
            print(
                f"Octave already at limit ({self.current_octave_offset:+d}). "
                f"Range: {MIDI_MIN_OCTAVE_OFFSET:+d}..{MIDI_MAX_OCTAVE_OFFSET:+d}"
            )
            return
        self.current_octave_offset = clamped
        print(
            f"Octave shift set to {self.current_octave_offset:+d} "
            f"(effective base note: {MIDI_BASE_NOTE + self.current_octave_offset * 12})."
        )

    def _toggle_chord_mode(self):
        self.chord_mode_enabled = not self.chord_mode_enabled
        mode_label = "ON" if self.chord_mode_enabled else "OFF"
        print(f"Chord mode: {mode_label} ({self._selected_chord_name()})")
        self._all_notes_off()

    def _cycle_chord_type(self):
        self.chord_type_index = (self.chord_type_index + 1) % len(CHORD_TYPE_ORDER)
        print(f"Chord type: {self._selected_chord_name()}")
        self._all_notes_off()

    def _all_notes_off(self):
        for label in self.hand_labels:
            self._update_midi_note(label, None)

    def _calibrate_velocity_baseline(self, hand_label: str):
        hand = self.hands[hand_label]
        self.velocity_reference_quaternions[hand_label] = normalize_quat(
            np.asarray(hand.rotation_quaternion, dtype=np.float64)
        )
        print(f"Velocity baseline calibrated (V) for {hand_label} hand.")

    def _switch_settings_hand(self, direction: int):
        current_index = self.hand_labels.index(self.settings_hand_label)
        next_index = (current_index + direction) % len(self.hand_labels)
        self.settings_hand_label = self.hand_labels[next_index]
        print(f"Settings hand switched to {self.settings_hand_label}.")

    def _compute_rotation_velocity(self, q_current: np.ndarray, q_reference: np.ndarray) -> int:
        # Compare palm-normal direction against baseline; opposite normal (180 deg flip) => max velocity.
        ref_rot = quaternion_to_rotation_matrix(q_reference)
        cur_rot = quaternion_to_rotation_matrix(q_current)

        ref_normal = ref_rot @ PALM_NORMAL_LOCAL
        cur_normal = cur_rot @ PALM_NORMAL_LOCAL

        dot = float(np.clip(np.dot(ref_normal, cur_normal), -1.0, 1.0))
        angle_rad = math.acos(dot)
        angle_ratio = angle_rad / math.pi  # 0..1 maps to 0..180 degrees
        velocity = int(round(MIDI_VELOCITY_MIN + angle_ratio * (MIDI_VELOCITY_MAX - MIDI_VELOCITY_MIN)))
        return int(np.clip(velocity, MIDI_VELOCITY_MIN, MIDI_VELOCITY_MAX))

    def _set_button_state_for_hand(self, hand, is_pressed: bool):
        if hasattr(hand, "button_state"):
            hand.button_state = is_pressed
        elif hasattr(hand, "button_pressed"):
            hand.button_pressed = is_pressed

    def _get_button_state_for_hand(self, hand) -> bool:
        if hasattr(hand, "button_state"):
            return bool(hand.button_state)
        if hasattr(hand, "button_pressed"):
            return bool(hand.button_pressed)
        return False

    def _clear_plane_visual(self, hand_label: str):
        plane_mesh_visual = self.plane_mesh_visuals[hand_label]
        plane_outline_visual = self.plane_outline_visuals[hand_label]
        section_lines = self.plane_section_lines[hand_label]
        if plane_mesh_visual is not None:
            plane_mesh_visual.visible = False
        if plane_outline_visual is not None:
            plane_outline_visual.visible = False
        for line in section_lines:
            line.visible = False

    def _clear_hand_plane(self, hand_label: str):
        self._clear_plane_visual(hand_label)
        self.plane_definitions[hand_label] = None
        self.press_centers[hand_label] = None
        self.last_plane_sizes[hand_label] = np.array([0.0, 0.0], dtype=np.float32)
        self.last_plane_axes[hand_label] = ("X", "Y")
        self.last_preview_update_times[hand_label] = None
        self.no_play_side_signs[hand_label] = 1.0
        self.play_side_signs[hand_label] = -1.0
        self.is_on_play_sides[hand_label] = False
        self.active_note_indices[hand_label] = None
        self._update_midi_note(hand_label, None)
        print(f"Cleared {hand_label} plane.")

    def _ensure_plane_visuals(self, hand_label: str):
        if self.plane_mesh_visuals[hand_label] is None:
            self.plane_mesh_visuals[hand_label] = scene.visuals.Mesh(
                vertices=np.zeros((4, 3), dtype=np.float32),
                faces=PLANE_FACES,
                color=(1.0, 0.8, 0.2, 0.0),
                shading=None,
                parent=self.view.scene,
            )
            self.plane_mesh_visuals[hand_label].visible = False

        if self.plane_outline_visuals[hand_label] is None:
            self.plane_outline_visuals[hand_label] = scene.visuals.Line(
                pos=np.zeros((2, 3), dtype=np.float32),
                color=(1.0, 0.8, 0.2, 0.0),
                width=2,
                method="gl",
                parent=self.view.scene,
            )
            self.plane_outline_visuals[hand_label].visible = False

        expected = NOTE_SECTION_COUNT - 1
        while len(self.plane_section_lines[hand_label]) < expected:
            line = scene.visuals.Line(
                pos=np.zeros((2, 3), dtype=np.float32),
                color=(1.0, 0.95, 0.55, 0.0),
                width=1,
                method="gl",
                parent=self.view.scene,
            )
            line.visible = False
            self.plane_section_lines[hand_label].append(line)

    def _compute_plane_definition(self, center: np.ndarray, corner: np.ndarray) -> PlaneDefinition:
        delta = corner - center
        abs_delta = np.abs(delta)

        # The smallest movement axis becomes plane normal.
        normal_axis = int(np.argmin(abs_delta))
        plane_axes = [axis for axis in range(3) if axis != normal_axis]

        half_u = max(float(abs_delta[plane_axes[0]]), MIN_PLANE_HALF_EXTENT)
        half_v = max(float(abs_delta[plane_axes[1]]), MIN_PLANE_HALF_EXTENT)

        return PlaneDefinition(
            center=center.copy(),
            axis_u=plane_axes[0],
            axis_v=plane_axes[1],
            normal_axis=normal_axis,
            half_u=half_u,
            half_v=half_v,
        )

    def _camera_relative_basis(self) -> tuple[np.ndarray, np.ndarray]:
        camera = self.view.camera
        azimuth_deg = float(getattr(camera, "azimuth", 0.0))
        azimuth_rad = np.deg2rad(azimuth_deg)

        # Turntable camera looks along +Y at azimuth=0; rotate around world Z.
        forward = np.array([-np.sin(azimuth_rad), np.cos(azimuth_rad), 0.0], dtype=np.float64)
        right = np.array([np.cos(azimuth_rad), np.sin(azimuth_rad), 0.0], dtype=np.float64)
        return forward, right

    def _compute_linear_velocity(self, hand_label: str) -> np.ndarray:
        keys = HAND_MOVE_KEYS[hand_label]
        forward, right = self._camera_relative_basis()
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        if self.keys_down.get(keys["forward"], False): vel += forward
        if self.keys_down.get(keys["backward"], False): vel -= forward
        if self.keys_down.get(keys["left"], False): vel -= right
        if self.keys_down.get(keys["right"], False): vel += right
        if self.keys_down.get(keys["up"], False): vel += up
        if self.keys_down.get(keys["down"], False): vel -= up

        norm = np.linalg.norm(vel)
        if norm > 1e-9:
            vel = (vel / norm) * self.velocity_step
        return vel

    def _plane_position_metrics(self, pos: np.ndarray, definition: PlaneDefinition) -> tuple[float, float, bool, float]:
        u_min = definition.center[definition.axis_u] - definition.half_u
        u_span = definition.half_u * 2.0
        v_min = definition.center[definition.axis_v] - definition.half_v
        v_span = definition.half_v * 2.0

        u = (pos[definition.axis_u] - u_min) / u_span
        v = (pos[definition.axis_v] - v_min) / v_span

        inside = (0.0 <= u <= 1.0) and (0.0 <= v <= 1.0)
        u_pct = float(np.clip(u, 0.0, 1.0) * 100.0)
        v_pct = float(np.clip(v, 0.0, 1.0) * 100.0)
        plane_offset = float(pos[definition.normal_axis] - definition.center[definition.normal_axis])
        return u_pct, v_pct, inside, plane_offset

    def _infer_no_play_side_sign(self, definition: PlaneDefinition, pos: np.ndarray, velocity: np.ndarray) -> float:
        offset = float(pos[definition.normal_axis] - definition.center[definition.normal_axis])
        if abs(offset) >= SIDE_INFER_OFFSET_EPS:
            return 1.0 if offset > 0.0 else -1.0

        vel_normal = float(velocity[definition.normal_axis])
        if abs(vel_normal) >= SIDE_INFER_VELOCITY_EPS:
            return 1.0 if vel_normal > 0.0 else -1.0

        return 1.0

    def _note_axis_and_half_extent(self, definition: PlaneDefinition) -> tuple[int, float, int, float]:
        # For vertical planes, prefer world-up axis so note changes happen moving up/down.
        if definition.axis_u == WORLD_UP_AXIS:
            return definition.axis_u, definition.half_u, definition.axis_v, definition.half_v
        if definition.axis_v == WORLD_UP_AXIS:
            return definition.axis_v, definition.half_v, definition.axis_u, definition.half_u

        # For horizontal planes, use whichever axis spans farther.
        if definition.half_u >= definition.half_v:
            return definition.axis_u, definition.half_u, definition.axis_v, definition.half_v
        return definition.axis_v, definition.half_v, definition.axis_u, definition.half_u

    def _note_section_from_position(self, pos: np.ndarray, definition: PlaneDefinition) -> tuple[int, str, str, float]:
        note_axis, note_half, _other_axis, _other_half = self._note_axis_and_half_extent(definition)
        axis_min = definition.center[note_axis] - note_half
        axis_span = note_half * 2.0
        normalized = (pos[note_axis] - axis_min) / axis_span
        clamped = float(np.clip(normalized, 0.0, 1.0))

        section_index = min(int(clamped * NOTE_SECTION_COUNT), NOTE_SECTION_COUNT - 1)
        note_name = NOTE_NAMES[section_index]
        section_percent = clamped * 100.0
        return section_index, note_name, AXIS_NAMES[note_axis], section_percent

    def _plane_vertices(self, definition: PlaneDefinition) -> np.ndarray:
        center = definition.center
        vertices = np.tile(center.astype(np.float32), (4, 1))

        corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        for idx, (u_sign, v_sign) in enumerate(corners):
            vertices[idx, definition.axis_u] = center[definition.axis_u] + (u_sign * definition.half_u)
            vertices[idx, definition.axis_v] = center[definition.axis_v] + (v_sign * definition.half_v)
            vertices[idx, definition.normal_axis] = center[definition.normal_axis]

        return vertices

    def _section_line_segments(self, definition: PlaneDefinition) -> list[np.ndarray]:
        segments: list[np.ndarray] = []

        note_axis, note_half, other_axis, other_half = self._note_axis_and_half_extent(definition)
        axis_min = definition.center[note_axis] - note_half
        for split_index in range(1, NOTE_SECTION_COUNT):
            split_ratio = split_index / NOTE_SECTION_COUNT
            split_coord = axis_min + (split_ratio * (2.0 * note_half))

            p0 = definition.center.copy()
            p1 = definition.center.copy()
            p0[note_axis] = split_coord
            p1[note_axis] = split_coord
            p0[other_axis] = definition.center[other_axis] - other_half
            p1[other_axis] = definition.center[other_axis] + other_half
            segments.append(np.array([p0, p1], dtype=np.float32))

        return segments

    def _update_plane_visuals(self, hand_label: str, definition: PlaneDefinition, preview: bool):
        self._ensure_plane_visuals(hand_label)

        vertices = self._plane_vertices(definition)
        outline_vertices = np.vstack([vertices, vertices[0]])
        segments = self._section_line_segments(definition)

        colors = self.plane_color_theme[hand_label]
        plane_color = colors["fill_preview"] if preview else colors["fill_set"]
        edge_color = colors["edge_preview"] if preview else colors["edge_set"]
        split_line_color = colors["split_preview"] if preview else colors["split_set"]

        plane_mesh_visual = self.plane_mesh_visuals[hand_label]
        plane_outline_visual = self.plane_outline_visuals[hand_label]
        assert plane_mesh_visual is not None
        assert plane_outline_visual is not None

        plane_mesh_visual.set_data(vertices=vertices, faces=PLANE_FACES, color=plane_color)
        plane_mesh_visual.visible = True

        plane_outline_visual.set_data(pos=outline_vertices, color=edge_color)
        plane_outline_visual.visible = True

        for line, segment in zip(self.plane_section_lines[hand_label], segments):
            line.set_data(pos=segment, color=split_line_color)
            line.visible = True

    def _set_plane_from_center_corner(
        self,
        hand_label: str,
        center: np.ndarray,
        corner: np.ndarray,
        preview: bool = False,
    ) -> PlaneDefinition:
        definition = self._compute_plane_definition(center, corner)
        self._update_plane_visuals(hand_label, definition, preview=preview)
        self.last_plane_sizes[hand_label] = np.array([definition.half_u * 2.0, definition.half_v * 2.0], dtype=np.float32)
        self.last_plane_axes[hand_label] = (AXIS_NAMES[definition.axis_u], AXIS_NAMES[definition.axis_v])

        if not preview:
            self.plane_definitions[hand_label] = definition
        return definition

    def on_key_press(self, event):
        """Record that a key is being held down and update motion vectors."""
        if not event.key: return
        key_name = event.key.name.upper()
        was_held = self.keys_down.get(key_name, False)
        self.keys_down[key_name] = True
        self._update_motion()

        if key_name == 'LEFT':
            if not was_held:
                self._switch_settings_hand(-1)
            self._update_motion()
            return
        elif key_name == 'RIGHT':
            if not was_held:
                self._switch_settings_hand(1)
            self._update_motion()
            return

        if key_name == SETTINGS_DRAW_KEY:
            self._set_button_state_for_hand(self.hands[self.settings_hand_label], True)
            self.draw_key_owner_label = self.settings_hand_label
        elif key_name == SETTINGS_CLEAR_KEY:
            self._clear_hand_plane(self.settings_hand_label)
        elif key_name == SETTINGS_CALIBRATE_KEY:
            self._calibrate_velocity_baseline(self.settings_hand_label)
        elif key_name == 'UP':
            if not was_held:
                self._adjust_octave(1)
        elif key_name == 'DOWN':
            if not was_held:
                self._adjust_octave(-1)
        elif key_name == 'Z':
            if not was_held:
                self._toggle_chord_mode()
        elif key_name == 'X':
            if not was_held:
                self._cycle_chord_type()
        else:
            selected_channel = self._midi_channel_from_row_key(key_name)
            if selected_channel is not None:
                self._select_midi_channel(self.settings_hand_label, selected_channel)
                return

    def on_key_release(self, event):
        """Record that a key is no longer held down and update motion vectors."""
        if not event.key: return
        key_name = event.key.name.upper()
        self.keys_down[key_name] = False
        self._update_motion()

        if key_name == SETTINGS_DRAW_KEY and self.draw_key_owner_label is not None:
            self._set_button_state_for_hand(self.hands[self.draw_key_owner_label], False)
            self.draw_key_owner_label = None

    def _update_motion(self):
        """Apply shared rotation keys (IJKLUO) to currently selected settings hand."""
        for hand_label in self.hand_labels:
            self.angular_velocity_vectors[hand_label] = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        keys = SETTINGS_ROTATION_KEYS
        ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        if self.keys_down.get(keys["pitch_pos"], False): ang_vel[1] += self.rotation_step  # Pitch
        if self.keys_down.get(keys["pitch_neg"], False): ang_vel[1] -= self.rotation_step
        if self.keys_down.get(keys["yaw_pos"], False): ang_vel[2] += self.rotation_step   # Yaw
        if self.keys_down.get(keys["yaw_neg"], False): ang_vel[2] -= self.rotation_step
        if self.keys_down.get(keys["roll_pos"], False): ang_vel[0] += self.rotation_step  # Roll
        if self.keys_down.get(keys["roll_neg"], False): ang_vel[0] -= self.rotation_step
        self.angular_velocity_vectors[self.settings_hand_label] = ang_vel

    def update(self, event):
        """Called by the timer to update hand position, rotation, and logic."""
        dt = float(event.dt) if (event is not None and event.dt is not None) else 0.0
        now = time.perf_counter()

        # --- Update Position + Rotation (independently per hand) ---
        for label in self.hand_labels:
            hand = self.hands[label]
            hand.velocity = self._compute_linear_velocity(label)
            hand.position += hand.velocity * dt

            angular_velocity = self.angular_velocity_vectors[label]
            rotation_angle = np.linalg.norm(angular_velocity) * dt
            if rotation_angle > 1e-9:
                rotation_axis = angular_velocity / (rotation_angle / dt)

                angle_rad_half = rotation_angle / 2.0
                w = math.cos(angle_rad_half)
                sin_half = math.sin(angle_rad_half)
                x, y, z = rotation_axis * sin_half

                delta_q = np.array([w, x, y, z])
                hand.rotation_quaternion = quaternion_multiply(
                    delta_q,
                    np.asarray(hand.rotation_quaternion, dtype=np.float64),
                )
                hand.rotation_quaternion = normalize_quat(hand.rotation_quaternion)

        # --- Run Logic ---
        for hand in self.hands.values():
            if hasattr(hand, "interpret_position"):
                hand.interpret_position()

        # --- Plane draw logic (independent per hand) ---
        button_states: dict[str, bool] = {}
        draw_states: dict[str, str] = {}
        for label in self.hand_labels:
            hand = self.hands[label]
            current_button_state = self._get_button_state_for_hand(hand)
            button_states[label] = current_button_state
            last_button_state = self.last_button_states[label]
            pressed_edge = (not last_button_state) and current_button_state
            released_edge = last_button_state and (not current_button_state)
            press_center = self.press_centers[label]
            pos = np.array(hand.position, dtype=np.float32)

            if pressed_edge:
                self.button_transition_counts[label] += 1
                self.press_centers[label] = pos.copy()
                self.last_preview_update_times[label] = None
                print(f"{label} plane center captured: {self.press_centers[label]}")
                press_center = self.press_centers[label]

            if press_center is not None and current_button_state:
                last_preview = self.last_preview_update_times[label]
                if last_preview is None or (now - last_preview) >= PREVIEW_MIN_INTERVAL_SECONDS:
                    self._set_plane_from_center_corner(label, press_center, pos, preview=True)
                    self.last_preview_update_times[label] = now

            if released_edge and press_center is not None:
                self.button_transition_counts[label] += 1
                committed_plane = self._set_plane_from_center_corner(label, press_center, pos, preview=False)
                self.last_preview_update_times[label] = None
                self.no_play_side_signs[label] = self._infer_no_play_side_sign(committed_plane, pos, hand.velocity)
                self.play_side_signs[label] = -self.no_play_side_signs[label]
                self.is_on_play_sides[label] = False
                self.active_note_indices[label] = None
                self._update_midi_note(label, None)
                print(f"{label} plane corner captured: {pos}")
                self.press_centers[label] = None

            self.last_button_states[label] = current_button_state
            draw_states[label] = "dragging corner" if self.press_centers[label] is not None else "idle"

        # --- Update Visual Transform + velocity estimate per hand ---
        for label in self.hand_labels:
            hand = self.hands[label]
            q_current = normalize_quat(np.asarray(hand.rotation_quaternion, dtype=np.float64))
            q_reference = self.velocity_reference_quaternions[label]
            self.current_midi_velocities[label] = self._compute_rotation_velocity(q_current, q_reference)

            hand_pos = np.array(hand.position, dtype=np.float32)
            R = quaternion_to_rotation_matrix(q_current)
            R = FRAME_MAP @ R @ FRAME_MAP.T
            R = R @ MODEL_OFFSET

            M = np.eye(4, dtype=np.float32)
            M[:3, :3] = R * MODEL_SCALE
            M[3, :3] = hand_pos * POSITION_SCALE
            self.hand_meshes[label].transform.matrix = M
        # --- End Visual Transform ---

        # --- Note/gate logic (independent per hand) ---
        plane_lines: list[str] = []
        zone_note_lines: list[str] = []
        hand_perf_lines: list[str] = []
        for label in self.hand_labels:
            hand = self.hands[label]
            pos = np.array(hand.position, dtype=np.float32)
            plane_definition = self.plane_definitions[label]
            note_active = False
            if plane_definition is not None:
                u_pct, v_pct, inside, plane_offset = self._plane_position_metrics(pos, plane_definition)
                play_metric = self.play_side_signs[label] * plane_offset
                if self.is_on_play_sides[label]:
                    if play_metric <= PLAY_EXIT_THRESHOLD:
                        self.is_on_play_sides[label] = False
                else:
                    if play_metric >= PLAY_ENTER_THRESHOLD:
                        self.is_on_play_sides[label] = True

                gate_status = f"{label} Zone: {'PLAY' if self.is_on_play_sides[label] else 'NO-PLAY'}"
                current_note_index: int | None = None
                if inside and self.is_on_play_sides[label]:
                    note_index, note_name, note_axis_name, note_axis_pct = self._note_section_from_position(pos, plane_definition)
                    current_note_index = note_index
                    note_active = True
                    note_status = (
                        f"Note: {note_name} ({note_index + 1}/{NOTE_SECTION_COUNT}) "
                        f"axis={note_axis_name} pos={note_axis_pct:5.1f}%"
                    )
                else:
                    note_status = "Note: inactive (need inside plane + play zone)"

                self.active_note_indices[label] = current_note_index
                self._update_midi_note(label, current_note_index, self.current_midi_velocities[label])
                plane_lines.append(
                    f"{label} Plane UV({self.last_plane_axes[label][0]}/{self.last_plane_axes[label][1]}): "
                    f"[{self.last_plane_sizes[label][0]:.2f}, {self.last_plane_sizes[label][1]:.2f}]  "
                    f"inside_uv=[{u_pct:5.1f}%, {v_pct:5.1f}%]  offset={plane_offset:+.3f}"
                )
                zone_note_lines.append(f"{gate_status}  {note_status}")
            else:
                self.active_note_indices[label] = None
                self._update_midi_note(label, None)
                plane_lines.append(f"{label} Plane: not set")
                zone_note_lines.append(f"{label} Zone: N/A  Note: inactive")

            self._update_live_expression(label, note_active, self.current_midi_velocities[label])
            hand_perf_lines.append(
                f"{label}: CH={self.hand_midi_channels[label] + 1} "
                f"Vel(rot)={self.current_midi_velocities[label]:3d} "
                f"Button={'PRESSED' if button_states[label] else 'RELEASED'} "
                f"trans={self.button_transition_counts[label]} draw={draw_states[label]}"
            )

        if self.midi_output_name:
            midi_status = f"MIDI: {self.midi_output_name}"
        else:
            midi_status = "MIDI: disconnected"
        chord_status = f"Chord: {'ON' if self.chord_mode_enabled else 'OFF'} ({self._selected_chord_name()})"
        settings_status = f"Settings Hand: {self.settings_hand_label} (Left/Right arrows)"
        controls_status = (
            "Controls: LEFT move=WASDQE, RIGHT move=TFGHRY, "
            "IJKLUO rotate selected hand, SPACE draw, C clear, V calibrate, "
            "1-0 set channel on selected hand, Up/Down octave, Z chord on/off, X chord type"
        )
        self.status_text.text = "\n".join(
            hand_perf_lines
            + plane_lines
            + zone_note_lines
            + [settings_status, chord_status, f"Octave offset: {self.current_octave_offset:+d}  (Up/Down arrows)", midi_status, controls_status]
        )

        self.canvas.update()

    def on_close(self, event):
        """Ensure the application exits cleanly."""
        self._close_midi_output()
        self.timer.stop()
        app.quit()

def main():
    """Initializes a mock GlovePair and runs the dual-hand debug visualizer."""
    mock_reader = None
    mock_queue = None

    glove_pair = jhk.GlovePair(
        device_ids=(1, 2),
        relay_id=1,
        reader=mock_reader,
        relay_queue=mock_queue
    )

    print("Starting Dual-Hand Debug Visualizer...")
    print("Controls:")
    print("  - LEFT hand move: WASDQE")
    print("  - RIGHT hand move: TFGHRY")
    print("  - Settings hand select: Left/Right arrows")
    print("  - Selected-hand controls: IJKLUO rotate, SPACE draw, C clear, V calibrate, 1-0 channel")
    print("  - MIDI Octave: Up/Down arrows to shift octave.")
    print("  - Chords: Z toggles chord mode, X cycles chord type.")
    
    visualizer = DualHandDebugVisualizer(glove_pair)
    
    app.run()

    print("Dual-Hand Debug Visualizer closed.")

if __name__ == "__main__":
    main()
