import csv
import time
from JazzHands import ThreadedMultiDeviceReader, COM_PORTS, DevicePacket
from vispy import app, scene  # type: ignore[import-untyped]
import numpy as np

# 1. Setup CSV Logging
csv_file = open('glove_data.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
# Headers based on PacketData class in JazzHands.py
csv_writer.writerow(['device_id', 'timestamp', 'accel_x', 'accel_y', 'accel_z',
                     'uwb_1', 'uwb_2', 'button', 'quat_w', 'quat_i', 'quat_j', 'quat_k'])

def log_packet_to_csv(packet:DevicePacket):
    """Callback function that writes every received packet to the CSV."""
    print("callback triggered: ", packet)
    csv_writer.writerow([
        packet.data.device_number, packet.data.timestamp,
        packet.data.accel_x, packet.data.accel_y, packet.data.accel_z,
        packet.data.UWB_distance_1, packet.data.UWB_distance_2,
        packet.data.button_state, packet.data.quat_w, packet.data.quat_i,
        packet.data.quat_j, packet.data.quat_k
    ])

# 2. Initialize the Reader
reader = ThreadedMultiDeviceReader()
reader.packet_callback_function = log_packet_to_csv

# Add devices using the ports defined in your configuration
reader.add_device(relay_id=1, port=COM_PORTS[0])
# 3. VisPy Visualization Setup (from vispy_test.py)

# set the canvas
canvas = scene.SceneCanvas(
    keys="interactive",
    show=True,
    bgcolor="black",
    size=(900, 700)
)
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(
    fov=45,
    distance=6.0,
    center=(0, 0, 0)
)

# Grid + axes
grid = scene.visuals.GridLines(
    scale=(1, 1),
    color=(0.3, 0.3, 0.3, 1.0)
)
view.add(grid)

axes = scene.visuals.XYZAxis(width=2)
view.add(axes)

#
x_label = scene.visuals.Text(
                            "X",
                             color="red",
                             font_size=18,
                             pos=(1.2, 0, 0),
                             parent=view.scene
                             )
y_label = scene.visuals.Text(
                            "Y",
                             color="green",
                             font_size=18,
                             pos=(0, 1.2, 0),
                             parent=view.scene
                             )
z_label = scene.visuals.Text(
                             "Z",
                             color="blue",
                             font_size=18,
                             pos=(0, 0, 1.2),
                             parent=view.scene
                             )

# create the line for mapping acceleration
acc_line = scene.visuals.Line(
    pos=np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32),
    color=(1, 1, 0, 1),  # yellow
    width=4,
    method="gl",
    parent=view.scene
)

# show text on screen (this is actually pretty cool i didn't know you could do this)
mag_text = scene.visuals.Text(
    "",
    color="white",
    font_size=14,
    pos=(10, 10),
    anchor_x="left",
    anchor_y="bottom",
    parent=canvas.scene
)


def update(event):
    latest_packet = None
    acc_scale:int = 1
    # "Drain" the queue to skip past all the old, lagged data
    print(reader.processing_queues[1].qsize())
    while not reader.processing_queues[1].empty():
        try:
            # get_nowait() prevents the script from hanging if the queue becomes empty exactly while we are checking it.
            latest_packet = reader.processing_queues[1].get_nowait()
        except Exception as e:
            print(e)
            break
    if latest_packet:
        print("Latest packet received: ", latest_packet)
        ax, ay, az = latest_packet.data.accel_x, latest_packet.data.accel_y, latest_packet.data.accel_z
        tip = np.array([ax, ay, az], dtype=np.float32) * acc_scale
        alpha = latest_packet.error_handler * 0.05 % 1

        acc_line.set_data(pos=np.array([[0, 0, 0], tip], dtype=np.float32))

        mag = float(np.linalg.norm([ax, ay, az]))
        mag_text.text = f"ax, ay, az = {ax:.3f}, {ay:.3f}, {az:.3f}  |a|={mag:.3f} m/s²  |   α = {alpha:.2f}"

# Start systems
reader.start() # Start the serial threads
timer = app.Timer(interval=1/100, connect=update, start=True)

try:
    app.run()

finally:
    # Ensure resources are closed properly when the window is closed
    reader.stop()
    csv_file.close()