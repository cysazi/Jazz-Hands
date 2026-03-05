import os
import sys

import numpy as np
from vispy import app, scene  # type: ignore[import-untyped]
from vispy.io import read_mesh  # type: ignore[import-untyped]
from vispy.visuals.transforms import MatrixTransform  # type: ignore[import-untyped]

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from JazzHands import COM_PORTS, GlovePair, ThreadedMultiDeviceReader


# --------------------------- Backend setup ---------------------------
def configure_vispy_backend() -> str:
    for backend in ("pyqt6", "pyside6", "tkinter"):
        try:
            app.use_app(backend)
            return backend
        except Exception:
            continue
    raise RuntimeError(
        "VisPy could not load a GUI backend. Install one with: "
        "python -m pip install PyQt6"
    )


BACKEND = configure_vispy_backend()


# --------------------------- Config ---------------------------
RELAY_ID = 1
POSITION_SCALE = 1.5
MODEL_SCALE = 0.02
UPDATE_HZ = 100
HAND_OBJ_PATH = os.path.join(THIS_DIR, "hand.obj")
AXIS_LABEL_DISTANCE = 1.2
MIN_BOX_HALF_EXTENT = 0.03

# IMU -> VisPy frame mapping borrowed from rotating color changing hand.py
FRAME_MAP = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
], dtype=np.float32)


def rot_x(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ], dtype=np.float32)


MODEL_OFFSET = rot_x(-90.0)


# --------------------------- Quaternion helpers ---------------------------
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
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float32)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ], dtype=np.float32)


# --------------------------- Scene setup ---------------------------
canvas = scene.SceneCanvas(keys="interactive", show=True, bgcolor="black", size=(1000, 750))
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(fov=45, distance=8.0, center=(0, 0, 0))

scene.visuals.GridLines(scale=(1, 1), color=(0.3, 0.3, 0.3, 1.0), parent=view.scene)
scene.visuals.XYZAxis(width=2, parent=view.scene)

scene.visuals.Text(
    "X",
    color="red",
    font_size=24,
    pos=(AXIS_LABEL_DISTANCE, 0, 0),
    parent=view.scene,
)
scene.visuals.Text(
    "Y",
    color="green",
    font_size=24,
    pos=(0, AXIS_LABEL_DISTANCE, 0),
    parent=view.scene,
)
scene.visuals.Text(
    "Z",
    color="blue",
    font_size=24,
    pos=(0, 0, AXIS_LABEL_DISTANCE),
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
    font_size=8,
    pos=(10, 10),
    anchor_x="left",
    anchor_y="bottom",
    parent=canvas.scene,
)

button_text = scene.visuals.Text(
    "BUTTON: UNKNOWN",
    color="white",
    font_size=8,
    pos=(10, 400),
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
reader.add_device(relay_id=RELAY_ID, port=COM_PORTS[RELAY_ID - 1], baudrate=115200)

glove_pair = GlovePair(
    device_ids=(1, 2),
    relay_id=RELAY_ID,
    instrument_type="melody",
    relay_queue=reader.processing_queues[RELAY_ID],
)

# Visual reset state (keyboard R)
zero_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
position_origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)

# Button draw state: INPUT_PULLUP means 1=idle, 0=pressed
last_button_state = True
button_transition_count = 0
press_center: np.ndarray | None = None
box_visual: scene.visuals.Box | None = None
last_box_size = np.array([0.0, 0.0, 0.0], dtype=np.float32)


def reset_view_to_current_pose() -> None:
    global zero_quat, position_origin
    left = glove_pair.left_hand
    zero_quat = normalize_quat(left.rotation_quaternion.astype(np.float32))
    position_origin = left.position.astype(np.float32).copy()
    print("View reset (keyboard R).")


def current_visual_pos() -> np.ndarray:
    left = glove_pair.left_hand
    # Match Accel_Filtering_Test.py behavior: use integrated position directly.
    return (left.position.astype(np.float32) - position_origin) * POSITION_SCALE


def set_box_from_center_corner(center: np.ndarray, corner: np.ndarray, preview: bool = False) -> None:
    global box_visual, last_box_size

    # Center is fixed at button-down; corner is sampled at button-up/current preview.
    delta = corner - center
    half_extents = np.abs(delta)

    # Square cross-section: largest-motion axis is "length", other two are equal.
    primary_axis = int(np.argmax(half_extents))
    cross_axes = [i for i in range(3) if i != primary_axis]
    cross_half = max(float(half_extents[cross_axes[0]]), float(half_extents[cross_axes[1]]), MIN_BOX_HALF_EXTENT)
    half_extents[cross_axes[0]] = cross_half
    half_extents[cross_axes[1]] = cross_half
    half_extents[primary_axis] = max(float(half_extents[primary_axis]), MIN_BOX_HALF_EXTENT)

    box_size = half_extents * 2.0
    last_box_size = box_size.copy()

    if box_visual is not None:
        box_visual.parent = None

    box_color = (1.0, 0.8, 0.2, 0.10) if preview else (1.0, 0.8, 0.2, 0.18)
    box_edge = (1.0, 0.9, 0.4, 0.9) if preview else (1.0, 0.8, 0.2, 1.0)

    box_visual = scene.visuals.Box(
        width=float(box_size[0]),
        height=float(box_size[1]),
        depth=float(box_size[2]),
        color=box_color,
        edge_color=box_edge,
        parent=view.scene,
    )

    box_tf = MatrixTransform()
    box_tf.translate((float(center[0]), float(center[1]), float(center[2])))
    box_visual.transform = box_tf


def update(_event) -> None:
    global last_button_state, press_center, button_transition_count

    left = glove_pair.left_hand
    pos = current_visual_pos()

    # Button press/release for drawing a 3D box.
    current_button_state = bool(left.button_state)
    pressed_edge = last_button_state and not current_button_state
    released_edge = (not last_button_state) and current_button_state

    if pressed_edge:
        button_transition_count += 1
        press_center = pos.copy()
        print(f"Box center captured: {press_center}")

    # Show preview box while held.
    if press_center is not None and not current_button_state:
        set_box_from_center_corner(press_center, pos, preview=True)

    if released_edge and press_center is not None:
        button_transition_count += 1
        set_box_from_center_corner(press_center, pos, preview=False)
        print(f"Box corner captured: {pos}")
        press_center = None

    last_button_state = current_button_state

    # Orientation mapping pipeline from rotating color changing hand.py
    q_current = normalize_quat(left.rotation_quaternion.astype(np.float32))
    q_rel = quat_mul(quat_conj(zero_quat), q_current)

    # Empirical sign fix from rotating color changing hand.py
    q_rel = np.array([q_rel[0], q_rel[1], -q_rel[2], q_rel[3]], dtype=np.float32)
    R = quat_to_rotmat(q_rel)
    R = FRAME_MAP @ R @ FRAME_MAP.T
    R = R @ MODEL_OFFSET

    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R * MODEL_SCALE
    M[3, :3] = pos
    hand_transform.matrix = M

    vel = left.velocity
    local_acc = left.local_acceleration
    global_acc = left.global_acceleration
    draw_state = "holding center" if press_center is not None else "idle"

    button_text.text = f"BUTTON: {'PRESSED' if not current_button_state else 'RELEASED'}   transitions={button_transition_count}"
    button_text.color = (1.0, 0.25, 0.25, 1.0) if not current_button_state else (0.25, 1.0, 0.25, 1.0)

    status_text.text = (
        f"Backend: {BACKEND}  Port: {COM_PORTS[RELAY_ID - 1]}  Relay: {RELAY_ID}\n"
        f"Button(raw): {int(current_button_state)} (0=pressed)  Draw: {draw_state}\n"
        f"Pos: [{pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}]\n"
        f"Vel: [{vel[0]:+.2f}, {vel[1]:+.2f}, {vel[2]:+.2f}]\n"
        f"Local Acc: [{local_acc[0]:+.2f}, {local_acc[1]:+.2f}, {local_acc[2]:+.2f}]\n"
        f"Global Acc: [{global_acc[0]:+.2f}, {global_acc[1]:+.2f}, {global_acc[2]:+.2f}]\n"
        f"Last box size: [{last_box_size[0]:.2f}, {last_box_size[1]:.2f}, {last_box_size[2]:.2f}]"
    )


@canvas.events.key_press.connect
def on_key_press(event) -> None:
    global box_visual, press_center, last_box_size

    if event.key == "R":
        reset_view_to_current_pose()
    elif event.key == "C":
        if box_visual is not None:
            box_visual.parent = None
            box_visual = None
        press_center = None
        last_box_size = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        print("Cleared box.")


reader.start()
glove_pair.start()
timer = app.Timer(interval=1 / UPDATE_HZ, connect=update, start=True)

try:
    app.run()
finally:
    timer.stop()
    glove_pair.stop()
    reader.stop()

