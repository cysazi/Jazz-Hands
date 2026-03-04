import csv
import math
import time

from vispy.visuals.transforms import MatrixTransform  # type: ignore[import-untyped]
from vispy import app, scene  # type: ignore[import-untyped]
import numpy as np
import os
import sys
import Visualization_Helper_Functions  # type: ignore[import-not-found]

here = os.path.dirname('/Users/cyrus/Coding/PycharmProjects/Jazz-Hands')
sys.path.append(os.path.join(here, '..'))

from JazzHands import DevicePacket, ThreadedMultiDeviceReader, GlovePair, quat_to_euler, COM_PORTS


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
canvas, view, grid, axes, x_label, y_label, z_label = setup_canvas()

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

hand_object = scene.visuals.Sphere(radius=0.35, method='latitude', parent=view.scene,
                                   edge_color='white')
ellipsoid_transform = MatrixTransform()
hand_object.transform = ellipsoid_transform
ellipsoid_scale = np.array([.2, .1, 0.05])
ellipsoid_transform.scale(ellipsoid_scale)

# show text on screen (this is actually pretty cool I didn't know you could do this)
on_screen_text = scene.visuals.Text(
    "",
    color="white",
    font_size=14,
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
    Quaternion (x, y, z, w) → (axis, angle_deg).
    axis: (ax, ay, az) unit vector
    angle_deg: rotation angle in degrees
    """
    x, y, z, w = q
    half = math.acos(max(-1.0, min(1.0, w)))
    angle = 2.0 * half
    s = math.sin(half)
    if s < 1e-10:
        return (1.0, 0.0, 0.0), 0.0
    return math.degrees(angle), np.array([x, y, z], dtype=np.float32)



def quat_to_euler_deg(q):
    """Same as quat_to_euler but returns degrees."""
    r, p, y = quat_to_euler(q)
    return math.degrees(r), math.degrees(p), math.degrees(y)


def update(event):
    """Fetch up-to-date data from the glove, update the physical object, and the"""
    # Get updated data from the glove
    pos = melody_glove.left_hand.position.astype(np.float32)
    vel = melody_glove.left_hand.velocity.astype(np.float32)
    acc = melody_glove.left_hand.global_acceleration.astype(np.float32)
    rot = quat_to_euler_deg(melody_glove.left_hand.rotation_quaternion)
    on_screen_text.text = (f"Pos: {pos[0]:+.2}, {pos[1]:+.2}, {pos[2]:+.2} \n"
                           f"Vel: {vel[0]:+.2}, {vel[1]:+.2}, {vel[2]:+.2} \n"
                           f"")

    # Transform the ellipsoid to reflect this new data.
    ellipsoid_transform.reset()  # reset transformation matrix so we don't accumulate matrix operations from previous frames
    ellipsoid_transform.translate(pos)  # 3) applied last → moves in world space
    apply_euler(ellipsoid_transform, rot[0], rot[1], rot[2])  # 2) applied second → rotates around origin
    ellipsoid_transform.scale(ellipsoid_scale)  # 1) applied first → shapes the ellipsoid

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
