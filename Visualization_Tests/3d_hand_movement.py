import os
import sys
import time
from dataclasses import dataclass

import numpy as np
from vispy import app, scene  # type: ignore[import-untyped]
from vispy.io import read_mesh  # type: ignore[import-untyped]
from vispy.visuals.transforms import MatrixTransform  # type: ignore[import-untyped]

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from JazzHands import (
    ANCHOR_SURVEY_DEFAULT_STEP_MS,
    COM_PORTS,
    GlovePair,
    PACKET_SIZE,
    ThreadedMultiDeviceReader,
    start_startup_anchor_survey,
    stop_startup_anchor_survey,
)

PACKET_HAS_UWB_1 = 0b00000100
PACKET_HAS_UWB_2 = 0b00001000
PACKET_HAS_UWB_3 = 0b00010000
PACKET_HAS_UWB_4 = 0b00100000
PACKET_HAS_QUAT = 0b00000010


def configure_vispy_backend() -> str:
    for backend in ("pyqt6", "pyside6", "tkinter"):
        try:
            app.use_app(backend)
            return backend
        except Exception:
            continue
    raise RuntimeError("VisPy backend missing. Install PyQt6 or PySide6.")


BACKEND = configure_vispy_backend()

# --------------------------- Config ---------------------------
RELAY_ID = 1
SERIAL_BAUD = 921600
UPDATE_HZ = 100
POSITION_SCALE = 10.0
MODEL_SCALE = 0.02
HUD_FONT_SIZE = 7
AXIS_LABEL_FONT_SIZE = 12
AXIS_LABEL_DISTANCE = 1.35
POSITION_SMOOTH_TAU_SECONDS = 0.14
POSITION_VISUAL_DEADBAND = 0.03
RX_HEARTBEAT_INTERVAL_SECONDS = 1.0
HAND_OBJ_PATH = os.path.join(THIS_DIR, "hand.obj")
AUTO_SURVEY_STEP_MS = ANCHOR_SURVEY_DEFAULT_STEP_MS

MIN_PLANE_HALF_EXTENT = 0.03
PREVIEW_UPDATE_HZ = 45
PREVIEW_MIN_INTERVAL_SECONDS = 1.0 / PREVIEW_UPDATE_HZ

NOTE_SECTION_COUNT = 12
NOTE_NAMES = ("C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B")
WORLD_UP_AXIS = 2  # Z
SIDE_INFER_OFFSET_EPS = 0.01
SIDE_INFER_VELOCITY_EPS = 0.03
PLAY_ENTER_THRESHOLD = 0.02
PLAY_EXIT_THRESHOLD = 0.01

AXIS_NAMES = ("X", "Y", "Z")
PLANE_FACES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

if len(NOTE_NAMES) != NOTE_SECTION_COUNT:
    raise ValueError("NOTE_NAMES must have exactly NOTE_SECTION_COUNT items.")


@dataclass
class PlaneDefinition:
    center: np.ndarray
    axis_u: int
    axis_v: int
    normal_axis: int
    half_u: float
    half_v: float

# IMU -> VisPy frame mapping
FRAME_MAP = np.array(
    [
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
    ],
    dtype=np.float32,
)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float32,
    )


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def rot_x(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array(
        [
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ],
        dtype=np.float32,
    )


MODEL_OFFSET = rot_x(-90.0)


def alpha_from_tau(dt: float, tau_seconds: float) -> float:
    if tau_seconds <= 1e-6:
        return 1.0
    return float(np.clip(dt / (tau_seconds + dt), 0.0, 1.0))


# --------------------------- Scene setup ---------------------------
canvas = scene.SceneCanvas(keys="interactive", show=True, bgcolor="black", size=(1000, 750))
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(fov=45, distance=8.0, center=(0, 0, 0))

scene.visuals.GridLines(scale=(1, 1), color=(0.3, 0.3, 0.3, 1.0), parent=view.scene)
scene.visuals.XYZAxis(width=2, parent=view.scene)
scene.visuals.Text(
    "X",
    color=(1.0, 0.3, 0.3, 1.0),
    font_size=AXIS_LABEL_FONT_SIZE,
    pos=(AXIS_LABEL_DISTANCE, 0.0, 0.0),
    anchor_x="center",
    anchor_y="center",
    parent=view.scene,
)
scene.visuals.Text(
    "Y",
    color=(0.3, 1.0, 0.3, 1.0),
    font_size=AXIS_LABEL_FONT_SIZE,
    pos=(0.0, AXIS_LABEL_DISTANCE, 0.0),
    anchor_x="center",
    anchor_y="center",
    parent=view.scene,
)
scene.visuals.Text(
    "Z",
    color=(0.3, 0.5, 1.0, 1.0),
    font_size=AXIS_LABEL_FONT_SIZE,
    pos=(0.0, 0.0, AXIS_LABEL_DISTANCE),
    anchor_x="center",
    anchor_y="center",
    parent=view.scene,
)

if not os.path.exists(HAND_OBJ_PATH):
    raise FileNotFoundError(f"Could not find model: {HAND_OBJ_PATH}")

vertices, faces, _normals, _texcoords = read_mesh(HAND_OBJ_PATH)
hand = scene.visuals.Mesh(
    vertices=vertices,
    faces=faces,
    color=(0.25, 0.75, 0.95, 1.0),
    shading="smooth",
    parent=view.scene,
)
hand_transform = MatrixTransform()
hand.transform = hand_transform

status_text = scene.visuals.Text(
    "Waiting for packets...",
    color="white",
    font_size=HUD_FONT_SIZE,
    pos=(10, 10),
    anchor_x="left",
    anchor_y="bottom",
    parent=canvas.scene,
)

button_text = scene.visuals.Text(
    "BUTTON: UNKNOWN",
    color="white",
    font_size=HUD_FONT_SIZE,
    pos=(10, 170),
    anchor_x="left",
    anchor_y="bottom",
    parent=canvas.scene,
)


# --------------------------- Reader setup ---------------------------
if len(COM_PORTS) < RELAY_ID:
    raise RuntimeError(
        f"COM_PORTS has {len(COM_PORTS)} entries, but RELAY_ID={RELAY_ID}. "
        f"Update COM_PORTS in JazzHands.py first."
    )

reader = ThreadedMultiDeviceReader()
selected_port = COM_PORTS[RELAY_ID - 1]
if not reader.add_device(relay_id=RELAY_ID, port=selected_port, baudrate=SERIAL_BAUD):
    raise RuntimeError(
        f"Failed to open relay {RELAY_ID} on {selected_port}. "
        "Check COM port and close Arduino Serial Monitor."
    )

glove_pair = GlovePair(
    device_ids=(1, 2),
    relay_id=RELAY_ID,
    instrument_type="melody",
    relay_queue=reader.processing_queues[RELAY_ID],
)


def print_startup_banner() -> None:
    print("=== JazzHands 3D Visualizer Boot ===")
    print(f"[PY] Backend={BACKEND}")
    print(f"[PY] Relay={RELAY_ID} Port={selected_port} Baud={SERIAL_BAUD}")
    print(f"[PY] Expected packet bytes={PACKET_SIZE}")
    print(f"[PY] Update Hz={UPDATE_HZ} Position scale={POSITION_SCALE}")
    print(f"[PY] Startup auto-survey step_ms={AUTO_SURVEY_STEP_MS}")
    print("[PY] Controls: R = reset visual origin, T = reset rotation baseline, C = clear plane, F = flip play/no-play side")
    print("[PY] Position source: UWB-only translation, quaternion rotation.")


# --------------------------- Runtime state ---------------------------
zero_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
view_origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
origin_initialized = False

smoothed_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
has_smoothed_pos = False
last_update_time: float | None = None

last_valid_uwb_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
has_valid_uwb_pos = False

last_rx_heartbeat_time: float = 0.0
last_rx_packet_count: int = 0
startup_survey_session_id: int | None = None
startup_survey_stop_sent: bool = False

last_button_state = True  # INPUT_PULLUP: 1 idle, 0 pressed
current_button_state = True
button_transition_count = 0
press_center: np.ndarray | None = None

plane_definition: PlaneDefinition | None = None
plane_mesh_visual: scene.visuals.Mesh | None = None
plane_outline_visual: scene.visuals.Line | None = None
plane_section_lines: list[scene.visuals.Line] = []
last_plane_size = np.array([0.0, 0.0], dtype=np.float32)
last_plane_axes = ("X", "Y")
last_preview_update_time: float | None = None

no_play_side_sign = 1.0
play_side_sign = -1.0
is_on_play_side = False
active_note_index: int | None = None


def active_hand_state():
    left = glove_pair.left_hand
    right = glove_pair.right_hand
    return left if left.current_packet_timestamp >= right.current_packet_timestamp else right


def get_uwb_pos_scaled(hand_state) -> tuple[bool, np.ndarray]:
    global has_valid_uwb_pos
    global last_valid_uwb_pos

    uwb_pose_valid = bool(getattr(hand_state, "uwb_pose_valid", False))
    uwb_relative_pose = getattr(hand_state, "uwb_relative_pose", None)
    if uwb_pose_valid and uwb_relative_pose is not None:
        pos = np.array(uwb_relative_pose, dtype=np.float32) * POSITION_SCALE
        last_valid_uwb_pos = pos.copy()
        has_valid_uwb_pos = True
        return True, pos

    if has_valid_uwb_pos:
        return False, last_valid_uwb_pos.copy()

    # UWB-only translation mode: do not fall back to Kalman/inertial position.
    return False, np.array([0.0, 0.0, 0.0], dtype=np.float32)


def reset_translation_to_current_pose() -> None:
    global view_origin
    global origin_initialized
    global smoothed_pos
    global has_smoothed_pos

    hand_state = active_hand_state()
    _valid, uwb_pos = get_uwb_pos_scaled(hand_state)
    view_origin = uwb_pos.copy()
    origin_initialized = True
    smoothed_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    has_smoothed_pos = False
    print("Reset translation origin (R).")


def reset_rotation_to_current_pose() -> None:
    global zero_quat

    hand_state = active_hand_state()
    zero_quat = normalize_quat(hand_state.rotation_quaternion.astype(np.float32))
    print("Reset rotation baseline (T).")


def axis_side_label(sign: float, axis_index: int) -> str:
    prefix = "+" if sign >= 0.0 else "-"
    return f"{prefix}{AXIS_NAMES[axis_index]}"


def plane_orientation_label(definition: PlaneDefinition) -> str:
    return "horizontal" if definition.normal_axis == WORLD_UP_AXIS else "vertical"


def clear_plane_visual() -> None:
    if plane_mesh_visual is not None:
        plane_mesh_visual.visible = False
    if plane_outline_visual is not None:
        plane_outline_visual.visible = False
    for line in plane_section_lines:
        line.visible = False


def ensure_plane_visuals() -> None:
    global plane_mesh_visual
    global plane_outline_visual

    if plane_mesh_visual is None:
        plane_mesh_visual = scene.visuals.Mesh(
            vertices=np.zeros((4, 3), dtype=np.float32),
            faces=PLANE_FACES,
            color=(1.0, 0.8, 0.2, 0.0),
            shading=None,
            parent=view.scene,
        )
        plane_mesh_visual.visible = False

    if plane_outline_visual is None:
        plane_outline_visual = scene.visuals.Line(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=(1.0, 0.8, 0.2, 0.0),
            width=2,
            method="gl",
            parent=view.scene,
        )
        plane_outline_visual.visible = False

    expected = NOTE_SECTION_COUNT - 1
    while len(plane_section_lines) < expected:
        line = scene.visuals.Line(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=(1.0, 0.95, 0.55, 0.0),
            width=1,
            method="gl",
            parent=view.scene,
        )
        line.visible = False
        plane_section_lines.append(line)


def compute_plane_definition(center: np.ndarray, corner: np.ndarray) -> PlaneDefinition:
    delta = corner - center
    abs_delta = np.abs(delta)

    # The smallest movement axis becomes plane normal.
    # If smallest is Z => horizontal plane; otherwise vertical plane.
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


def note_axis_and_half_extent(definition: PlaneDefinition) -> tuple[int, float, int, float]:
    # Prefer vertical pitch mapping when the plane includes world-up axis.
    if definition.axis_u == WORLD_UP_AXIS:
        return definition.axis_u, definition.half_u, definition.axis_v, definition.half_v
    if definition.axis_v == WORLD_UP_AXIS:
        return definition.axis_v, definition.half_v, definition.axis_u, definition.half_u

    # Otherwise use longer axis for note slicing.
    if definition.half_u >= definition.half_v:
        return definition.axis_u, definition.half_u, definition.axis_v, definition.half_v
    return definition.axis_v, definition.half_v, definition.axis_u, definition.half_u


def plane_vertices(definition: PlaneDefinition) -> np.ndarray:
    center = definition.center
    vertices = np.tile(center.astype(np.float32), (4, 1))

    corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    for idx, (u_sign, v_sign) in enumerate(corners):
        vertices[idx, definition.axis_u] = center[definition.axis_u] + (u_sign * definition.half_u)
        vertices[idx, definition.axis_v] = center[definition.axis_v] + (v_sign * definition.half_v)
        vertices[idx, definition.normal_axis] = center[definition.normal_axis]

    return vertices


def section_line_segments(definition: PlaneDefinition) -> list[np.ndarray]:
    segments: list[np.ndarray] = []

    note_axis, note_half, other_axis, other_half = note_axis_and_half_extent(definition)
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


def update_plane_visuals(definition: PlaneDefinition, preview: bool) -> None:
    ensure_plane_visuals()

    vertices = plane_vertices(definition)
    outline_vertices = np.vstack([vertices, vertices[0]])
    segments = section_line_segments(definition)

    plane_color = (1.0, 0.8, 0.2, 0.12) if preview else (1.0, 0.8, 0.2, 0.22)
    edge_color = (1.0, 0.9, 0.4, 0.85) if preview else (1.0, 0.8, 0.2, 1.0)
    split_line_color = (1.0, 0.95, 0.55, 0.5) if preview else (1.0, 0.95, 0.55, 0.8)

    assert plane_mesh_visual is not None
    assert plane_outline_visual is not None

    plane_mesh_visual.set_data(vertices=vertices, faces=PLANE_FACES, color=plane_color)
    plane_mesh_visual.visible = True

    plane_outline_visual.set_data(pos=outline_vertices, color=edge_color)
    plane_outline_visual.visible = True

    for line, segment in zip(plane_section_lines, segments):
        line.set_data(pos=segment, color=split_line_color)
        line.visible = True


def set_plane_from_center_corner(center: np.ndarray, corner: np.ndarray, preview: bool = False) -> PlaneDefinition:
    global plane_definition
    global last_plane_size
    global last_plane_axes

    definition = compute_plane_definition(center, corner)
    update_plane_visuals(definition, preview=preview)
    last_plane_size = np.array([definition.half_u * 2.0, definition.half_v * 2.0], dtype=np.float32)
    last_plane_axes = (AXIS_NAMES[definition.axis_u], AXIS_NAMES[definition.axis_v])

    if not preview:
        plane_definition = definition
    return definition


def plane_position_metrics(pos: np.ndarray, definition: PlaneDefinition) -> tuple[float, float, bool, float]:
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


def infer_no_play_side_sign(definition: PlaneDefinition, pos: np.ndarray, velocity: np.ndarray) -> float:
    offset = float(pos[definition.normal_axis] - definition.center[definition.normal_axis])
    if abs(offset) >= SIDE_INFER_OFFSET_EPS:
        return 1.0 if offset > 0.0 else -1.0

    vel_normal = float(velocity[definition.normal_axis])
    if abs(vel_normal) >= SIDE_INFER_VELOCITY_EPS:
        return 1.0 if vel_normal > 0.0 else -1.0

    return 1.0


def note_section_from_position(pos: np.ndarray, definition: PlaneDefinition) -> tuple[int, str, str, float]:
    note_axis, note_half, _other_axis, _other_half = note_axis_and_half_extent(definition)
    axis_min = definition.center[note_axis] - note_half
    axis_span = note_half * 2.0
    normalized = (pos[note_axis] - axis_min) / axis_span
    clamped = float(np.clip(normalized, 0.0, 1.0))

    section_index = min(int(clamped * NOTE_SECTION_COUNT), NOTE_SECTION_COUNT - 1)
    note_name = NOTE_NAMES[section_index]
    section_percent = clamped * 100.0
    return section_index, note_name, AXIS_NAMES[note_axis], section_percent


def print_rx_heartbeat(now: float, active_hand, left_hand, right_hand) -> None:
    global last_rx_heartbeat_time
    global last_rx_packet_count

    if (now - last_rx_heartbeat_time) < RX_HEARTBEAT_INTERVAL_SECONDS:
        return
    last_rx_heartbeat_time = now

    snapshot = reader.get_device_snapshot(RELAY_ID)
    if snapshot is None:
        print("[PY RX] NO DATA - relay not found.")
        return

    total = int(snapshot["packet_count"])
    delta = total - last_rx_packet_count
    last_rx_packet_count = total

    if delta > 0:
        print(
            "[PY RX] RECEIVING "
            f"+{delta}/s total={total} queue={snapshot['queue_size']} pkt={snapshot['packet_size']} "
            f"dev={active_hand.device_id} flags=0b{active_hand.last_packet_flags:08b} "
            f"uwb=[{active_hand.UWB_distance_1},{active_hand.UWB_distance_2},{active_hand.UWB_distance_3},{active_hand.UWB_distance_4}] "
            f"uwb_pose={getattr(active_hand, 'uwb_pose_valid', False)} "
            f"survey={getattr(active_hand, 'uwb_survey_done', False)} "
            f"left_ts={left_hand.current_packet_timestamp} right_ts={right_hand.current_packet_timestamp}"
        )
    else:
        print(
            "[PY RX] NO NEW PACKETS "
            f"total={total} queue={snapshot['queue_size']} pkt={snapshot['packet_size']} "
            f"left_ts={left_hand.current_packet_timestamp} right_ts={right_hand.current_packet_timestamp}"
        )


def update(_event) -> None:
    global origin_initialized
    global view_origin
    global smoothed_pos
    global has_smoothed_pos
    global last_update_time
    global startup_survey_stop_sent
    global last_button_state
    global current_button_state
    global button_transition_count
    global press_center
    global no_play_side_sign
    global play_side_sign
    global is_on_play_side
    global active_note_index
    global last_preview_update_time

    now = time.perf_counter()
    if last_update_time is None:
        dt = 1.0 / UPDATE_HZ
    else:
        dt = now - last_update_time
    last_update_time = now
    dt = float(np.clip(dt, 1e-4, 0.2))

    left_hand = glove_pair.left_hand
    right_hand = glove_pair.right_hand
    hand_state = left_hand if left_hand.current_packet_timestamp >= right_hand.current_packet_timestamp else right_hand
    survey = glove_pair.survey_manager

    print_rx_heartbeat(now, hand_state, left_hand, right_hand)

    if survey.locked and (not startup_survey_stop_sent) and (startup_survey_session_id is not None):
        if stop_startup_anchor_survey(reader, RELAY_ID, startup_survey_session_id):
            startup_survey_stop_sent = True
            survey.stop()
            print(
                "[PY SURVEY] LOCKED "
                f"L12/L13/L14/L23/L24/L34=[{survey.l12_m:.3f}, {survey.l13_m:.3f}, {survey.l14_m:.3f}, "
                f"{survey.l23_m:.3f}, {survey.l24_m:.3f}, {survey.l34_m:.3f}] m. "
                "STOP sent."
            )

    if hand_state.current_packet_timestamp <= 0:
        snap = survey.status_snapshot()
        status_text.text = (
            f"Backend: {BACKEND}  Port: {COM_PORTS[RELAY_ID - 1]}  Relay: {RELAY_ID}\n"
            "Waiting for packets...\n"
            f"Survey: {'LOCKED' if snap['locked'] else 'collecting'}  "
            f"pairs={snap['counts']}\n"
            "If this persists: check receiver relay flash, MAC/channel, and that Arduino Serial Monitor is closed."
        )
        return

    uwb_valid, raw_uwb_pos = get_uwb_pos_scaled(hand_state)
    if not origin_initialized:
        view_origin = raw_uwb_pos.copy()
        origin_initialized = True

    raw_pos = raw_uwb_pos - view_origin

    if not has_smoothed_pos:
        smoothed_pos = raw_pos.copy()
        has_smoothed_pos = True
    else:
        pos_alpha = alpha_from_tau(dt, POSITION_SMOOTH_TAU_SECONDS)
        delta = raw_pos - smoothed_pos
        delta[np.abs(delta) < POSITION_VISUAL_DEADBAND] = 0.0
        smoothed_pos += pos_alpha * delta

    pos = smoothed_pos
    current_velocity = np.array(getattr(hand_state, "kalman_velocity", hand_state.velocity), dtype=np.float32) * POSITION_SCALE

    current_button_state = bool(hand_state.button_state)
    pressed_edge = last_button_state and not current_button_state
    released_edge = (not last_button_state) and current_button_state

    if pressed_edge:
        button_transition_count += 1
        press_center = pos.copy()
        last_preview_update_time = None
        print(f"Plane center captured: {press_center}")

    if released_edge and press_center is not None:
        button_transition_count += 1
        committed_plane = set_plane_from_center_corner(press_center, pos, preview=False)
        no_play_side_sign = infer_no_play_side_sign(committed_plane, pos, current_velocity)
        play_side_sign = -no_play_side_sign
        is_on_play_side = False
        active_note_index = None
        last_preview_update_time = None
        print(f"Plane corner captured: {pos}")
        print(
            "Plane gate set: "
            f"NO-PLAY={axis_side_label(no_play_side_sign, committed_plane.normal_axis)}  "
            f"PLAY={axis_side_label(play_side_sign, committed_plane.normal_axis)}"
        )
        press_center = None

    if press_center is not None and not current_button_state:
        if last_preview_update_time is None or (now - last_preview_update_time) >= PREVIEW_MIN_INTERVAL_SECONDS:
            set_plane_from_center_corner(press_center, pos, preview=True)
            last_preview_update_time = now

    last_button_state = current_button_state

    q_current = normalize_quat(hand_state.rotation_quaternion.astype(np.float32))
    q_rel = quat_mul(quat_conj(zero_quat), q_current)
    R = quat_to_rotmat(q_rel)
    R = FRAME_MAP @ R @ FRAME_MAP.T
    R = R @ MODEL_OFFSET

    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R * MODEL_SCALE
    M[3, :3] = pos
    hand_transform.matrix = M

    quat_dbg = normalize_quat(hand_state.rotation_quaternion.astype(np.float32))
    kalman_pos = np.array(getattr(hand_state, "kalman_position", hand_state.position), dtype=np.float32)
    kalman_vel = np.array(getattr(hand_state, "kalman_velocity", hand_state.velocity), dtype=np.float32)
    uwb_rel = np.array(getattr(hand_state, "uwb_relative_pose", np.array([0.0, 0.0, 0.0])), dtype=np.float32)
    local_acc = np.array(hand_state.local_acceleration, dtype=np.float32)
    flags = int(hand_state.last_packet_flags)
    uwb_flag_bits = (
        f"d1={'Y' if (flags & PACKET_HAS_UWB_1) else 'N'} "
        f"d2={'Y' if (flags & PACKET_HAS_UWB_2) else 'N'} "
        f"d3={'Y' if (flags & PACKET_HAS_UWB_3) else 'N'} "
        f"d4={'Y' if (flags & PACKET_HAS_UWB_4) else 'N'}"
    )
    quat_flag = "Y" if (flags & PACKET_HAS_QUAT) else "N"
    survey_snap = survey.status_snapshot()
    pair_counts = survey_snap["counts"]
    survey_state = "LOCKED" if survey_snap["locked"] else ("TIMEOUT" if survey_snap["timed_out"] else "collecting")
    survey_line = (
        f"Anchor survey: {survey_state} "
        f"pairs12/13/23=[{pair_counts[(1, 2)]}/{pair_counts[(1, 3)]}/{pair_counts[(2, 3)]}] "
        f"pairs14/24/34=[{pair_counts[(1, 4)]}/{pair_counts[(2, 4)]}/{pair_counts[(3, 4)]}] "
        f"L12/L13/L14/L23/L24/L34=[{survey_snap['l12_m']:.3f}, {survey_snap['l13_m']:.3f}, {survey_snap['l14_m']:.3f}, "
        f"{survey_snap['l23_m']:.3f}, {survey_snap['l24_m']:.3f}, {survey_snap['l34_m']:.3f}]"
    )
    uwb_health_line = (
        f"UWB health: pose_valid={getattr(hand_state, 'uwb_pose_valid', False)} "
        f"d=[{hand_state.UWB_distance_1}, {hand_state.UWB_distance_2}, {hand_state.UWB_distance_3}, {hand_state.UWB_distance_4}]"
    )

    draw_state = "dragging corner" if press_center is not None else "idle"
    button_text.text = f"BUTTON: {'PRESSED' if not current_button_state else 'RELEASED'}   transitions={button_transition_count}"
    button_text.color = (1.0, 0.25, 0.25, 1.0) if not current_button_state else (0.25, 1.0, 0.25, 1.0)

    note_status = "Note section: inactive"
    if plane_definition is None:
        plane_status = "Plane: not set (press+hold to preview, release to commit)"
        gate_status = "Gate: N/A"
    else:
        u_pct, v_pct, inside, plane_offset = plane_position_metrics(pos, plane_definition)
        uv_axes = f"{AXIS_NAMES[plane_definition.axis_u]}/{AXIS_NAMES[plane_definition.axis_v]}"
        normal_axis_name = AXIS_NAMES[plane_definition.normal_axis]
        orientation = plane_orientation_label(plane_definition)

        play_metric = play_side_sign * plane_offset
        if is_on_play_side:
            if play_metric <= PLAY_EXIT_THRESHOLD:
                is_on_play_side = False
        else:
            if play_metric >= PLAY_ENTER_THRESHOLD:
                is_on_play_side = True

        gate_side_text = "PLAY" if is_on_play_side else "NO-PLAY"
        gate_status = (
            f"Gate: {gate_side_text}  "
            f"NO-PLAY={axis_side_label(no_play_side_sign, plane_definition.normal_axis)}  "
            f"PLAY={axis_side_label(play_side_sign, plane_definition.normal_axis)}"
        )

        current_note_index: int | None = None
        if inside and is_on_play_side:
            note_index, note_name, note_axis_name, note_axis_pct = note_section_from_position(pos, plane_definition)
            current_note_index = note_index
            note_status = (
                f"Note section: {note_index + 1}/{NOTE_SECTION_COUNT} ({note_name})  "
                f"axis={note_axis_name}  axis_pos={note_axis_pct:5.1f}%"
            )
        else:
            note_status = "Note section: inactive (need inside plane + play side)"

        if current_note_index != active_note_index:
            if current_note_index is None:
                if active_note_index is not None:
                    print("Note section cleared.")
            else:
                print(
                    f"Active note section -> {NOTE_NAMES[current_note_index]} "
                    f"({current_note_index + 1}/{NOTE_SECTION_COUNT})"
                )
            active_note_index = current_note_index

        plane_status = (
            f"Plane ({orientation}) UV({uv_axes}): [{u_pct:5.1f}%, {v_pct:5.1f}%]  "
            f"offset {normal_axis_name}: {plane_offset:+.3f}"
        )

    status_text.text = (
        f"Backend: {BACKEND}  Port: {COM_PORTS[RELAY_ID - 1]}  Relay: {RELAY_ID}\n"
        f"Device: {hand_state.device_id}  Packet: {hand_state.last_packet_size} bytes  flags=0b{hand_state.last_packet_flags:08b}\n"
        f"Quat flag: {quat_flag}\n"
        f"UWB new-sample bits: {uwb_flag_bits}\n"
        f"{survey_line}\n"
        f"{uwb_health_line}\n"
        f"Button(raw): {int(current_button_state)} (0=pressed)  Draw: {draw_state}\n"
        f"UWB Pose: {'valid' if uwb_valid else 'invalid'}  Survey: {'LOCKED' if getattr(hand_state, 'uwb_survey_done', False) else 'collecting'}\n"
        f"Render Pos (UWB only): [{pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}]\n"
        f"UWB rel raw: [{uwb_rel[0]:+.2f}, {uwb_rel[1]:+.2f}, {uwb_rel[2]:+.2f}] m\n"
        f"Kalman Pos: [{kalman_pos[0]:+.2f}, {kalman_pos[1]:+.2f}, {kalman_pos[2]:+.2f}] m\n"
        f"Kalman Vel: [{kalman_vel[0]:+.2f}, {kalman_vel[1]:+.2f}, {kalman_vel[2]:+.2f}] m/s\n"
        f"Local Acc: [{local_acc[0]:+.2f}, {local_acc[1]:+.2f}, {local_acc[2]:+.2f}] m/s^2\n"
        f"Last plane size ({last_plane_axes[0]}/{last_plane_axes[1]}): [{last_plane_size[0]:.2f}, {last_plane_size[1]:.2f}]\n"
        f"{plane_status}\n"
        f"{gate_status}\n"
        f"{note_status}\n"
        f"Quat: [{quat_dbg[0]:+.3f}, {quat_dbg[1]:+.3f}, {quat_dbg[2]:+.3f}, {quat_dbg[3]:+.3f}]\n"
        f"UWB dists: d1={hand_state.UWB_distance_1} d2={hand_state.UWB_distance_2} d3={hand_state.UWB_distance_3} d4={hand_state.UWB_distance_4}"
    )


@canvas.events.key_press.connect
def on_key_press(event) -> None:
    global press_center
    global plane_definition
    global last_plane_size
    global last_plane_axes
    global no_play_side_sign
    global play_side_sign
    global is_on_play_side
    global active_note_index
    global last_preview_update_time

    key = str(event.key).upper()

    if key == "R":
        reset_translation_to_current_pose()
    elif key == "T":
        reset_rotation_to_current_pose()
    elif key == "C":
        clear_plane_visual()
        plane_definition = None
        press_center = None
        last_plane_size = np.array([0.0, 0.0], dtype=np.float32)
        last_plane_axes = ("X", "Y")
        no_play_side_sign = 1.0
        play_side_sign = -1.0
        is_on_play_side = False
        active_note_index = None
        last_preview_update_time = None
        print("Cleared plane.")
    elif key == "F" and plane_definition is not None:
        no_play_side_sign *= -1.0
        play_side_sign *= -1.0
        is_on_play_side = False
        active_note_index = None
        print(
            "Flipped gate side: "
            f"NO-PLAY={axis_side_label(no_play_side_sign, plane_definition.normal_axis)}  "
            f"PLAY={axis_side_label(play_side_sign, plane_definition.normal_axis)}"
        )


print_startup_banner()
reader.start()
glove_pair.start()

startup_survey_session_id = start_startup_anchor_survey(
    reader,
    RELAY_ID,
    step_ms=AUTO_SURVEY_STEP_MS,
)
if startup_survey_session_id is None:
    print("[PY SURVEY] Failed to send START_SURVEY command to receiver.")
else:
    glove_pair.survey_manager.reset(startup_survey_session_id)
    print(
        f"[PY SURVEY] START sent session={startup_survey_session_id} "
        f"step_ms={AUTO_SURVEY_STEP_MS}"
    )

timer = app.Timer(interval=1 / UPDATE_HZ, connect=update, start=True)

try:
    app.run()
finally:
    if startup_survey_session_id is not None:
        stop_startup_anchor_survey(reader, RELAY_ID, startup_survey_session_id)
    timer.stop()
    glove_pair.stop()
    reader.stop()
