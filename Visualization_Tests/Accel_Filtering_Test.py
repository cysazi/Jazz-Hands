import csv
from vispy.geometry import create_sphere
from vispy.visuals.transforms import MatrixTransform  # type: ignore[import-untyped]
from vispy import app, scene  # type: ignore[import-untyped]
from vispy.io import read_mesh
import numpy as np
import time
import os
import Visualization_Helper_Functions as vp  # type: ignore[import-not-found]
from JazzHands import DevicePacket, ThreadedMultiDeviceReader, GlovePair, COM_PORTS

CURRENT_FILEPATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_FILEPATH)
HAND_OBJ_PATH = os.path.join(CURRENT_FILEPATH, "hand.obj")

# region Config and Helpers from 3d_hand_movement.py
POSITION_SCALE = 1.5
MODEL_SCALE = 0.02

# IMU -> VisPy frame mapping
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


# Quaternion helpers
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

# endregion


def log_packet_to_csv(packet: DevicePacket):
    """Callback function that writes every received packet to the CSV."""
    global melody_glove
    print("callback triggered: ", packet)
    csv_writer.writerow([
        packet.data.device_number, packet.data.timestamp,
        packet.data.accel_x, packet.data.accel_y, packet.data.accel_z,
        packet.data.UWB_distance_1, packet.data.UWB_distance_2,
        packet.data.button_state, packet.data.quat_w, packet.data.quat_i,
        packet.data.quat_j, packet.data.quat_k,
        melody_glove.left_hand.velocity[0], melody_glove.left_hand.velocity[1], melody_glove.left_hand.velocity[2],
        melody_glove.left_hand.position[0], melody_glove.left_hand.position[1], melody_glove.left_hand.position[2]
    ])


# 1. Setup CSV Logging

csv_file = open('glove_data.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
# Headers based on PacketData class in JazzHands.py
csv_writer.writerow(['device_id', 'timestamp', 'accel_x', 'accel_y', 'accel_z',
                     'uwb_1', 'uwb_2', 'button', 'quat_w', 'quat_i', 'quat_j',
                     'quat_k', 'vel_x', 'vel_y', 'vel_z', 'pos_x', 'pos_y', 'pos_z' ])
time.sleep(1)

# 2. Initialize the Reader
reader = ThreadedMultiDeviceReader()
reader.packet_callback_function = log_packet_to_csv

# Add devices using the ports defined in your configuration
reader.add_device(relay_id=1, port=COM_PORTS[0])
melody_glove = GlovePair(device_ids=(1, 2), relay_id=1, instrument_type="melody",
                         relay_queue=reader.processing_queues[1])

# Visual reset state (keyboard R)
zero_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
position_origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)

# region VisPy Visualization Setup

# General canvas setup
canvas, view, grid, axes, x_label, y_label, z_label = vp.setup_canvas()

# Set camera properties
view.camera = scene.cameras.TurntableCamera(fov=45, distance=8.0, center=(0, 0, 0))

# Specific setup for this test
# create the line for mapping acceleration
acc_line = scene.visuals.Arrow(
    pos=np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32),
    color=(1, 0, 0, 1),  # red
    arrow_size=3,
    width=4,
    method="gl",
    parent=view.scene
)

vel_line = scene.visuals.Arrow(
    pos=np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32),
    color=(0, 1, 0, 1),  # green
    arrow_size=3,
    width=4,
    method="gl",
    parent=view.scene
)

# ── Ellipsoid Semi-Axes ────────────────────────────────────────────────────────
a = 2.0   # semi-axis along X
b = 1.0   # semi-axis along Y
c = 0.5   # semi-axis along Z
# ── Build Ellipsoid Mesh ───────────────────────────────────────────────────────
# Start from a unit sphere (centered at origin) and scale each axis
sphere_data = create_sphere(rows=60, cols=60, radius=1.0)
vertices = sphere_data.get_vertices().copy()   # shape: (N, 3)
faces    = sphere_data.get_faces()             # shape: (M, 3)

# Scale each axis to transform the sphere into an ellipsoid:
#   x = a·sin(φ)·cos(θ),  y = b·sin(φ)·sin(θ),  z = c·cos(φ)
vertices[:, 0] *= a
vertices[:, 1] *= b
vertices[:, 2] *= c

# ── Solid Mesh Visual ──────────────────────────────────────────────────────────

if not os.path.exists(HAND_OBJ_PATH):
    raise FileNotFoundError(f"Could not find model: {HAND_OBJ_PATH}")

vertices, faces, _normals, _texcoords = read_mesh(HAND_OBJ_PATH)
hand = scene.visuals.Mesh(
    vertices=vertices,
    faces=faces,
    color=(0.25, 0.75, 0.95, 0.80),
    shading="smooth",
    parent=view.scene,
)

hand_transform = MatrixTransform()
hand.transform = hand_transform

# show text on the screen
on_screen_text = scene.visuals.Text(
    "",
    color="white",
    font_size=12,
    pos=(10, 10),
    anchor_x="left",
    anchor_y="bottom",
    parent=canvas.scene
)


@canvas.events.key_press.connect
def on_key_press(event):
    if event.key == 'Space':
        on_space_key_pressed()

    elif event.key == 'A':
        on_a_key_pressed()

    elif event.key == 'R':
        reset_view_to_current_pose()


def on_a_key_pressed():
    pass


def reset_view_to_current_pose() -> None:
    global zero_quat, position_origin
    left = melody_glove.left_hand
    zero_quat = normalize_quat(left.rotation_quaternion.astype(np.float32))
    position_origin = left.position.astype(np.float32).copy()
    print("View reset (keyboard R).")


def on_space_key_pressed():
    pass


# endregion

alpha = 0.05

def update(event):
    """Fetch up-to-date data from the glove, update the physical object, and the"""
    # Get updated data from the glove
    global alpha
    VECTOR_SCALE = 3

    if melody_glove.left_hand.error:
        alpha = 0.05 * melody_glove.left_hand.error
    # melody_glove.left_hand.local_acceleration = np.array([1,1,0.2], dtype=np.float32)
    # melody_glove.left_hand.rotation_quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    melody_glove.left_hand.integrate_function()
    left = melody_glove.left_hand
    pos = (left.position.astype(np.float32) - position_origin) * POSITION_SCALE
    vel = left.velocity.astype(np.float32)
    acc = left.global_acceleration.astype(np.float32)
    on_screen_text.text = (f"Pos: {pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f} \n"
                           f"Vel: {vel[0]:+.3f}, {vel[1]:+.3f}, {vel[2]:+.3f} ; |v| = {np.linalg.norm(vel):+.3f} \n"
                           f"Acc: {acc[0]:+.3f}, {acc[1]:+.3f}, {acc[2]:+.3f} ; |a| = {np.linalg.norm(acc):+.3f} \n"
                           f"α = {alpha:.2f}    ")

    # Orientation mapping pipeline from 3d_hand_movement.py
    q_current = normalize_quat(left.rotation_quaternion.astype(np.float32))
    q_rel = quat_mul(quat_conj(zero_quat), q_current)

    # Empirical sign fix from 3d_hand_movement.py
    q_rel = np.array([q_rel[0], q_rel[1], -q_rel[2], q_rel[3]], dtype=np.float32)
    R = quat_to_rotmat(q_rel)
    R = FRAME_MAP @ R @ FRAME_MAP.T
    R = R @ MODEL_OFFSET

    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R * MODEL_SCALE
    M[3, :3] = pos
    hand_transform.matrix = M

    # Add Vel and Acc vectors coming off of the object.
    acc_line.set_data(pos=np.array([pos, (pos + acc) * VECTOR_SCALE], dtype=np.float32))
    vel_line.set_data(pos=np.array([pos, (pos + vel) * VECTOR_SCALE], dtype=np.float32))


# Start systems
reader.start()  # Start the serial threads
melody_glove.start()  # Start packet processing for
timer = app.Timer(interval=(1 / 100), connect=update, start=True)

try:
    app.run()


finally:
    # Ensure resources are closed properly when the window is closed
    reader.stop()
    melody_glove.stop()
    csv_file.close()
