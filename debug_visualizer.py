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
MIDI_VELOCITY = 100


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


class DebugVisualizer:
    def __init__(self, glove_pair):
        self.glove_pair = glove_pair
        self.left_hand = self.glove_pair.left_hand

        self.canvas, self.view, _, _, _, _, _ = setup_canvas()
        self.view.camera.distance = 3

        # --- Load Hand Mesh ---
        if not os.path.exists(HAND_OBJ_PATH):
            raise FileNotFoundError(f"Could not find model: {HAND_OBJ_PATH}")
        vertices, faces, _normals, _texcoords = read_mesh(HAND_OBJ_PATH)
        self.hand_mesh = scene.visuals.Mesh(
            vertices=vertices,
            faces=faces,
            color=(0.25, 0.75, 0.95, 0.80),
            shading="smooth",
            parent=self.view.scene,
        )
        self.hand_mesh.transform = MatrixTransform()
        # --- End Hand Mesh ---

        self.canvas.events.key_press.connect(self.on_key_press)
        self.canvas.events.key_release.connect(self.on_key_release)
        self.canvas.events.close.connect(self.on_close)

        self.timer = Timer('auto', connect=self.update, start=True)

        # --- State tracking for keyboard input ---
        self.velocity_step = 0.5
        self.rotation_step = 1.5  # Radians per second
        self.keys_down = {}
        self.angular_velocity_vector = np.array([0.0, 0.0, 0.0]) # Pitch, Yaw, Roll

        # --- Plane drawing state ---
        self.last_button_state = False
        self.current_button_state = False
        self.button_transition_count = 0
        self.press_center: np.ndarray | None = None
        self.last_preview_update_time: float | None = None

        self.plane_definition: PlaneDefinition | None = None
        self.plane_mesh_visual: scene.visuals.Mesh | None = None
        self.plane_outline_visual: scene.visuals.Line | None = None
        self.plane_section_lines: list[scene.visuals.Line] = []
        self.last_plane_size = np.array([0.0, 0.0], dtype=np.float32)
        self.last_plane_axes = ("X", "Y")
        self.no_play_side_sign = 1.0
        self.play_side_sign = -1.0
        self.is_on_play_side = False
        self.active_note_index: int | None = None
        self.midi_out = None
        self.midi_output_name: str | None = None
        self.last_midi_note: int | None = None
        self._setup_midi_output()
        atexit.register(self._close_midi_output)

        self.status_text = scene.visuals.Text(
            "Plane: not set",
            color="white",
            font_size=10,
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

    def _send_note_off(self, note: int):
        if self.midi_out is None:
            return
        self.midi_out.send(mido.Message("note_off", note=note, velocity=0))

    def _send_note_on(self, note: int):
        if self.midi_out is None:
            return
        self.midi_out.send(mido.Message("note_on", note=note, velocity=MIDI_VELOCITY))

    def _update_midi_note(self, note_index: int | None):
        if self.midi_out is None:
            return
        desired_note = (MIDI_BASE_NOTE + note_index) if note_index is not None else None
        if desired_note == self.last_midi_note:
            return

        if self.last_midi_note is not None:
            self._send_note_off(self.last_midi_note)
        if desired_note is not None:
            self._send_note_on(desired_note)
        self.last_midi_note = desired_note

    def _close_midi_output(self):
        if self.midi_out is not None:
            if self.last_midi_note is not None:
                self._send_note_off(self.last_midi_note)
                self.last_midi_note = None
            self.midi_out.close()
            self.midi_out = None

    def _set_button_state(self, is_pressed: bool):
        if hasattr(self.left_hand, "button_state"):
            self.left_hand.button_state = is_pressed
        elif hasattr(self.left_hand, "button_pressed"):
            self.left_hand.button_pressed = is_pressed

    def _get_button_state(self) -> bool:
        if hasattr(self.left_hand, "button_state"):
            return bool(self.left_hand.button_state)
        if hasattr(self.left_hand, "button_pressed"):
            return bool(self.left_hand.button_pressed)
        return False

    def _clear_plane_visual(self):
        if self.plane_mesh_visual is not None:
            self.plane_mesh_visual.visible = False
        if self.plane_outline_visual is not None:
            self.plane_outline_visual.visible = False
        for line in self.plane_section_lines:
            line.visible = False

    def _ensure_plane_visuals(self):
        if self.plane_mesh_visual is None:
            self.plane_mesh_visual = scene.visuals.Mesh(
                vertices=np.zeros((4, 3), dtype=np.float32),
                faces=PLANE_FACES,
                color=(1.0, 0.8, 0.2, 0.0),
                shading=None,
                parent=self.view.scene,
            )
            self.plane_mesh_visual.visible = False

        if self.plane_outline_visual is None:
            self.plane_outline_visual = scene.visuals.Line(
                pos=np.zeros((2, 3), dtype=np.float32),
                color=(1.0, 0.8, 0.2, 0.0),
                width=2,
                method="gl",
                parent=self.view.scene,
            )
            self.plane_outline_visual.visible = False

        expected = NOTE_SECTION_COUNT - 1
        while len(self.plane_section_lines) < expected:
            line = scene.visuals.Line(
                pos=np.zeros((2, 3), dtype=np.float32),
                color=(1.0, 0.95, 0.55, 0.0),
                width=1,
                method="gl",
                parent=self.view.scene,
            )
            line.visible = False
            self.plane_section_lines.append(line)

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

    def _compute_linear_velocity(self) -> np.ndarray:
        forward, right = self._camera_relative_basis()
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        if self.keys_down.get('W', False): vel += forward
        if self.keys_down.get('S', False): vel -= forward
        if self.keys_down.get('A', False): vel -= right
        if self.keys_down.get('D', False): vel += right
        if self.keys_down.get('Q', False): vel += up
        if self.keys_down.get('E', False): vel -= up

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

    def _update_plane_visuals(self, definition: PlaneDefinition, preview: bool):
        self._ensure_plane_visuals()

        vertices = self._plane_vertices(definition)
        outline_vertices = np.vstack([vertices, vertices[0]])
        segments = self._section_line_segments(definition)

        plane_color = (1.0, 0.8, 0.2, 0.12) if preview else (1.0, 0.8, 0.2, 0.22)
        edge_color = (1.0, 0.9, 0.4, 0.85) if preview else (1.0, 0.8, 0.2, 1.0)
        split_line_color = (1.0, 0.95, 0.55, 0.5) if preview else (1.0, 0.95, 0.55, 0.8)

        assert self.plane_mesh_visual is not None
        assert self.plane_outline_visual is not None

        self.plane_mesh_visual.set_data(vertices=vertices, faces=PLANE_FACES, color=plane_color)
        self.plane_mesh_visual.visible = True

        self.plane_outline_visual.set_data(pos=outline_vertices, color=edge_color)
        self.plane_outline_visual.visible = True

        for line, segment in zip(self.plane_section_lines, segments):
            line.set_data(pos=segment, color=split_line_color)
            line.visible = True

    def _set_plane_from_center_corner(self, center: np.ndarray, corner: np.ndarray, preview: bool = False) -> PlaneDefinition:
        definition = self._compute_plane_definition(center, corner)
        self._update_plane_visuals(definition, preview=preview)
        self.last_plane_size = np.array([definition.half_u * 2.0, definition.half_v * 2.0], dtype=np.float32)
        self.last_plane_axes = (AXIS_NAMES[definition.axis_u], AXIS_NAMES[definition.axis_v])

        if not preview:
            self.plane_definition = definition
        return definition

    def on_key_press(self, event):
        """Record that a key is being held down and update motion vectors."""
        if not event.key: return
        self.keys_down[event.key.name.upper()] = True
        self._update_motion()

        key_name = event.key.name.upper()
        if key_name == 'SPACE':
            self._set_button_state(True)
        elif key_name == 'C':
            self._clear_plane_visual()
            self.plane_definition = None
            self.press_center = None
            self.last_plane_size = np.array([0.0, 0.0], dtype=np.float32)
            self.last_plane_axes = ("X", "Y")
            self.last_preview_update_time = None
            self.no_play_side_sign = 1.0
            self.play_side_sign = -1.0
            self.is_on_play_side = False
            self.active_note_index = None
            self._update_midi_note(None)
            print("Cleared plane.")

    def on_key_release(self, event):
        """Record that a key is no longer held down and update motion vectors."""
        if not event.key: return
        self.keys_down[event.key.name.upper()] = False
        self._update_motion()

        if event.key.name.upper() == 'SPACE':
            self._set_button_state(False)

    def _update_motion(self):
        """Calculate the final velocity and angular velocity vectors from the keys_down dictionary."""
        # Angular velocity (Pitch, Yaw, Roll)
        ang_vel = np.array([0.0, 0.0, 0.0])
        if self.keys_down.get('I', False): ang_vel[1] += self.rotation_step  # Pitch
        if self.keys_down.get('K', False): ang_vel[1] -= self.rotation_step
        if self.keys_down.get('J', False): ang_vel[2] += self.rotation_step  # Yaw
        if self.keys_down.get('L', False): ang_vel[2] -= self.rotation_step
        if self.keys_down.get('U', False): ang_vel[0] += self.rotation_step  # Roll
        if self.keys_down.get('O', False): ang_vel[0] -= self.rotation_step
        self.angular_velocity_vector = ang_vel

    def update(self, event):
        """Called by the timer to update hand position, rotation, and logic."""
        dt = float(event.dt) if (event is not None and event.dt is not None) else 0.0
        now = time.perf_counter()

        self.left_hand.velocity = self._compute_linear_velocity()

        # --- Update Position ---
        self.left_hand.position += self.left_hand.velocity * dt

        # --- Update Rotation ---
        rotation_angle = np.linalg.norm(self.angular_velocity_vector) * dt
        if rotation_angle > 1e-9:
            rotation_axis = self.angular_velocity_vector / (rotation_angle / dt)
            
            angle_rad_half = rotation_angle / 2.0
            w = math.cos(angle_rad_half)
            sin_half = math.sin(angle_rad_half)
            x, y, z = rotation_axis * sin_half
            
            delta_q = np.array([w, x, y, z])
            
            # Apply rotation in world frame
            self.left_hand.rotation_quaternion = quaternion_multiply(
                delta_q,
                np.asarray(self.left_hand.rotation_quaternion, dtype=np.float64),
            )
            self.left_hand.rotation_quaternion = normalize_quat(self.left_hand.rotation_quaternion)

        # --- Run Logic ---
        if hasattr(self.left_hand, "interpret_position"):
            self.left_hand.interpret_position()

        # --- Plane draw logic ---
        self.current_button_state = self._get_button_state()
        pressed_edge = (not self.last_button_state) and self.current_button_state
        released_edge = self.last_button_state and (not self.current_button_state)

        pos = np.array(self.left_hand.position, dtype=np.float32)

        if pressed_edge:
            self.button_transition_count += 1
            self.press_center = pos.copy()
            self.last_preview_update_time = None
            print(f"Plane center captured: {self.press_center}")

        if self.press_center is not None and self.current_button_state:
            if self.last_preview_update_time is None or (now - self.last_preview_update_time) >= PREVIEW_MIN_INTERVAL_SECONDS:
                self._set_plane_from_center_corner(self.press_center, pos, preview=True)
                self.last_preview_update_time = now

        if released_edge and self.press_center is not None:
            self.button_transition_count += 1
            committed_plane = self._set_plane_from_center_corner(self.press_center, pos, preview=False)
            self.last_preview_update_time = None
            self.no_play_side_sign = self._infer_no_play_side_sign(committed_plane, pos, self.left_hand.velocity)
            self.play_side_sign = -self.no_play_side_sign
            self.is_on_play_side = False
            self.active_note_index = None
            self._update_midi_note(None)
            print(f"Plane corner captured: {pos}")
            self.press_center = None

        self.last_button_state = self.current_button_state

        # --- Update Visual Transform ---
        q_current = normalize_quat(np.asarray(self.left_hand.rotation_quaternion, dtype=np.float64))
        R = quaternion_to_rotation_matrix(q_current)
        R = FRAME_MAP @ R @ FRAME_MAP.T
        R = R @ MODEL_OFFSET

        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = R * MODEL_SCALE
        M[3, :3] = pos * POSITION_SCALE
        self.hand_mesh.transform.matrix = M
        # --- End Visual Transform ---

        draw_state = "dragging corner" if self.press_center is not None else "idle"
        plane_status = "Plane: not set"
        gate_status = "Zone: N/A"
        note_status = "Note: inactive"
        if self.plane_definition is not None:
            u_pct, v_pct, inside, plane_offset = self._plane_position_metrics(pos, self.plane_definition)
            play_metric = self.play_side_sign * plane_offset
            if self.is_on_play_side:
                if play_metric <= PLAY_EXIT_THRESHOLD:
                    self.is_on_play_side = False
            else:
                if play_metric >= PLAY_ENTER_THRESHOLD:
                    self.is_on_play_side = True

            gate_status = f"Zone: {'PLAY' if self.is_on_play_side else 'NO-PLAY'}"
            current_note_index: int | None = None
            if inside and self.is_on_play_side:
                note_index, note_name, note_axis_name, note_axis_pct = self._note_section_from_position(pos, self.plane_definition)
                current_note_index = note_index
                note_status = f"Note: {note_name} ({note_index + 1}/{NOTE_SECTION_COUNT}) axis={note_axis_name} pos={note_axis_pct:5.1f}%"
            else:
                note_status = "Note: inactive (need inside plane + play zone)"

            self.active_note_index = current_note_index
            self._update_midi_note(current_note_index)
            plane_status = (
                f"Plane UV({self.last_plane_axes[0]}/{self.last_plane_axes[1]}): "
                f"[{self.last_plane_size[0]:.2f}, {self.last_plane_size[1]:.2f}]  "
                f"inside_uv=[{u_pct:5.1f}%, {v_pct:5.1f}%]  offset={plane_offset:+.3f}"
            )
        else:
            self._update_midi_note(None)

        midi_status = f"MIDI: {self.midi_output_name}" if self.midi_output_name else "MIDI: disconnected"
        self.status_text.text = (
            f"Button: {'PRESSED' if self.current_button_state else 'RELEASED'}  "
            f"transitions={self.button_transition_count}  Draw: {draw_state}\n"
            f"{plane_status}\n"
            f"{gate_status}  {note_status}\n"
            f"{midi_status}\n"
            "Controls: WASD move with view, Q/E up/down, IJKLUO rotate, SPACE draw plane, C clear"
        )

        self.canvas.update()

    def on_close(self, event):
        """Ensure the application exits cleanly."""
        self._close_midi_output()
        self.timer.stop()
        app.quit()

def main():
    """Initializes a mock GlovePair and runs the debug visualizer."""
    mock_reader = None
    mock_queue = None

    glove_pair = jhk.GlovePair(
        device_ids=(1, 2),
        relay_id=1,
        reader=mock_reader,
        relay_queue=mock_queue
    )

    print("Starting Debug Visualizer...")
    print("Controls:")
    print("  - Movement: WASD (camera-relative), Q/E (up/down)")
    print("  - Rotation: IJKL (Pitch/Yaw), UO (Roll)")
    print("  - Action: Space press/hold/release to draw plane, C to clear.")
    
    visualizer = DebugVisualizer(glove_pair)
    
    app.run()

    print("Debug Visualizer closed.")

if __name__ == "__main__":
    main()
