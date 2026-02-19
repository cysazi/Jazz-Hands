from vispy import app, scene
import numpy as np
import serial

# serial settings
PORT = "COM5"
BAUD = 115200

# refresh rate
UPDATE_INTERVAL = 1/100

# optional scaling (isnt needed)
ACC_SCALE = 1


ser = serial.Serial(PORT, BAUD, timeout=0.0)  # non-blocking

# storing variable
latest_accel = np.array([0.0, 0.0, 0.0], dtype=np.float32)


def read_latest_packet():
    # function to read the latest packet
    global latest_accel

    while True:
        raw = ser.readline()
        if not raw:
            break  # no data to read

        line = raw.decode("utf-8", errors="ignore").strip()
        if not line:
            continue

        parts = line.split(",")
        if len(parts) != 7:
            continue

        try:
            # last three are ax, ay, az
            ax = float(parts[-3])
            ay = float(parts[-2])
            az = float(parts[-1])
            latest_accel[:] = (ax, ay, az)
        except ValueError:
            continue


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


def update_visuals():
    ax, ay, az = latest_accel
    tip = np.array([ax, ay, az], dtype=np.float32) * ACC_SCALE

    acc_line.set_data(pos=np.array([[0, 0, 0], tip], dtype=np.float32))

    mag = float(np.linalg.norm([ax, ay, az]))
    mag_text.text = f"ax, ay, az = {ax:.3f}, {ay:.3f}, {az:.3f}  |a|={mag:.3f} m/s²"


def on_timer(event):
    read_latest_packet()
    update_visuals()

# timer is setting the refresh rate of the function
timer = app.Timer(interval=UPDATE_INTERVAL, connect=on_timer, start=True)

app.run()
