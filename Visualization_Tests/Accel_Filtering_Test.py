import csv
import math
from vispy.geometry import create_sphere
from vispy.visuals.transforms import MatrixTransform  # type: ignore[import-untyped]
from vispy import app, scene  # type: ignore[import-untyped]
from vispy.io import read_mesh
import numpy as np
import os
import Visualization_Helper_Functions as vp  # type: ignore[import-not-found]
from JazzHands import DevicePacket, ThreadedMultiDeviceReader, GlovePair, quat_to_euler, COM_PORTS

CURRENT_FILEPATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_FILEPATH)
HAND_OBJ_PATH = os.path.join(CURRENT_FILEPATH, "hand.obj")



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
csv_writer.writerow({'device_id', 'timestamp', 'accel_x', 'accel_y', 'accel_z',
                     'uwb_1', 'uwb_2', 'button', 'quat_w', 'quat_i', 'quat_j',
                     'quat_k', 'vel_x', 'vel_y', 'vel_z', 'pos_x', })

# 2. Initialize the Reader
reader = ThreadedMultiDeviceReader()
reader.packet_callback_function = log_packet_to_csv

# Add devices using the ports defined in your configuration
reader.add_device(relay_id=1, port=COM_PORTS[0])
melody_glove = GlovePair(device_ids=(1, 2), relay_id=1, instrument_type="melody",
                         relay_queue=reader.processing_queues[1])

# region VisPy Visualization Setup

# General canvas setup
canvas, view, grid, axes, x_label, y_label, z_label = vp.setup_canvas()

# Specific setup for this test
# create the line for mapping acceleration
acc_line = scene.visuals.Arrow(
    pos=np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32),
    color=(1, 1, 0, 1),  # yellow
    arrow_size=3,
    width=4,
    method="gl",
    parent=view.scene
)

vel_line = scene.visuals.Arrow(
    pos=np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32),
    color=(1, 1, 0, 1),  # yellow
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
    color=(0.25, 0.75, 0.95, 0.75),
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


def on_a_key_pressed():
    pass


def on_space_key_pressed():
    pass


# endregion
def quat_to_axis_angle(q):
    """
    Quaternion (w, x, y, z) → (angle_deg, axis).
    angle_deg: rotation angle in degrees
    axis: (ax, ay, az) unit vector
    """
    w, x, y, z = q
    # Clamp w to the valid range for acos
    w_clamped = max(-1.0, min(1.0, w))
    half_angle_rad = math.acos(w_clamped)
    angle_deg = math.degrees(2.0 * half_angle_rad)

    sin_half_angle = math.sin(half_angle_rad)

    # If angle is close to 0, axis is not well-defined.
    # In this case, we can return any unit vector, like (1, 0, 0).
    if abs(sin_half_angle) < 1e-8:
        return 0.0, (1.0, 0.0, 0.0)

    # Normalize the vector part of the quaternion to get the rotation axis.
    axis = np.array([x, y, z], dtype=np.float32) / sin_half_angle
    return angle_deg, axis


def quat_to_euler_deg(q):
    """Same as quat_to_euler but returns degrees."""
    r, p, y = quat_to_euler(q)
    return math.degrees(r), math.degrees(p), math.degrees(y)

alpha = 0.05

def update(event):
    """Fetch up-to-date data from the glove, update the physical object, and the"""
    # Get updated data from the glove
    global alpha
    if melody_glove.left_hand.error:
        alpha = 0.05 * melody_glove.left_hand.error

    pos = melody_glove.left_hand.position.astype(np.float32)
    vel = melody_glove.left_hand.velocity.astype(np.float32)
    acc = melody_glove.left_hand.global_acceleration.astype(np.float32)
    on_screen_text.text = (f"Pos: {pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f} \n"
                           f"Vel: {vel[0]:+.3f}, {vel[1]:+.3f}, {vel[2]:+.3f} ; |v| = {np.linalg.norm(vel):+.3f} \n"
                           f"Acc: {acc[0]:+.3f}, {acc[1]:+.3f}, {acc[2]:+.3f} ; |a| = {np.linalg.norm(acc):+.3f} \n"
                           f"α = {alpha:.2f}")

    # Transform the ellipsoid to reflect this new data.
    hand_transform.reset()  # reset transformation matrix so we don't accumulate matrix operations from previous frames
    hand_transform.rotate(*quat_to_axis_angle(melody_glove.left_hand.rotation_quaternion))
    hand_transform.translate(pos)  # 2) applied last → moves in world space

    # Add Vel and Acc vectors coming off of the object.
    acc_line.set_data(pos=np.array([pos, pos + acc], dtype=np.float32))
    vel_line.set_data(pos=np.array([pos, pos + vel], dtype=np.float32))


# Start systems
reader.start()  # Start the serial threads
melody_glove.start()  # Start packet processing for
timer = app.Timer(interval=(1 / 100), connect=update, start=True)

try:
    app.run()


finally:
    # Ensure resources are closed properly when the window is closed
    reader.stop()
    csv_file.close()
