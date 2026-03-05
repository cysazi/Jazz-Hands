import numpy as np
import serial
from vispy import app, scene

# ----------------------------
# Serial settings
# ----------------------------
PORT = "COM5"
BAUD = 115200
ser = serial.Serial(PORT, BAUD, timeout=0.0)  # non-blocking

# ----------------------------
# Integration state
# ----------------------------
pos = np.zeros(3, dtype=np.float32)  # meters (ish)
vel = np.zeros(3, dtype=np.float32)  # m/s
last_t = None

# Tuning knobs (drift control)
VEL_DAMPING = 0.98      # 0.95-0.995 typical (lower = more damping)
ACC_DEADZONE = 0.08     # m/s^2 ignore tiny accel noise

# Drawing state
drawing = False
draw_points = []  # list of 3D points

# Latest sensor values
latest_btn = 0
latest_acc = np.zeros(3, dtype=np.float32)

def parse_line(line: str):
    """
    Expected: t_ms,ax,ay,az,btn
    Returns (t_seconds, acc_xyz, btn) or None if parse fails.
    """
    parts = line.split(",")
    if len(parts) != 5:
        return None

    try:
        t_ms = int(parts[0])
        ax = float(parts[1])
        ay = float(parts[2])
        az = float(parts[3])
        btn = int(parts[4])
    except ValueError:
        return None

    return (t_ms * 0.001, np.array([ax, ay, az], dtype=np.float32), btn)

def read_latest_packet():
    """
    Drain serial buffer and keep only the most recent valid packet.
    """
    global latest_acc, latest_btn, last_t

    newest = None

    while True:
        raw = ser.readline()
        if not raw:
            break

        s = raw.decode("utf-8", errors="ignore").strip()
        if not s:
            continue

        parsed = parse_line(s)
        if parsed is not None:
            newest = parsed

    if newest is None:
        return None

    t_s, acc_xyz, btn = newest
    latest_acc[:] = acc_xyz
    latest_btn = btn
    return newest

def deadzone(v: np.ndarray, dz: float):
    """
    If |component| < dz, set it to 0. Helps reduce integrating noise.
    """
    out = v.copy()
    out[np.abs(out) < dz] = 0.0
    return out

# ----------------------------
# VisPy scene setup
# ----------------------------
canvas = scene.SceneCanvas(keys="interactive", bgcolor="black", size=(1000, 700), show=True)
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(fov=45, distance=2.5, up='+z')

# World axes
axis = scene.visuals.XYZAxis(parent=view.scene)

# Moving “pen tip”
pen_marker = scene.visuals.Markers(parent=view.scene)
pen_marker.set_data(np.array([[0, 0, 0]], dtype=np.float32), size=12)

# Drawn line
line = scene.visuals.Line(parent=view.scene, method="gl", width=3)
line.set_data(np.zeros((0, 3), dtype=np.float32))

# Optional: show a faint grid
grid = scene.visuals.GridLines(parent=view.scene)

# Scale helper: exaggerate position so you can see it easily
VIS_SCALE = 1.0  # try 5.0 or 20.0 if motion is too small

# ----------------------------
# Update loop
# ----------------------------
def update(_event):
    global last_t, pos, vel, drawing, draw_points

    packet = read_latest_packet()
    if packet is None:
        # Still update visuals with current state
        pen_marker.set_data(np.array([pos * VIS_SCALE], dtype=np.float32), size=12)
        return

    t_s, acc_xyz, btn = packet

    if last_t is None:
        last_t = t_s
        return

    dt = t_s - last_t
    last_t = t_s

    # Clamp dt in case of serial hiccups
    if dt <= 0 or dt > 0.1:
        return

    # 1) (Prototype) Use accel directly.
    # NOTE: If your IMU includes gravity (most raw accel does),
    # you MUST remove gravity or use "linear acceleration" output,
    # otherwise position will explode.
    acc = deadzone(acc_xyz, ACC_DEADZONE)

    # 2) Integrate: a -> v -> p
    vel = VEL_DAMPING * (vel + acc * dt)
    pos = pos + vel * dt

    # 3) Button controls drawing
    drawing = (btn == 1)

    # Update pen marker
    pen_marker.set_data(np.array([pos * VIS_SCALE], dtype=np.float32), size=12)

    # If drawing, append points and update line
    if drawing:
        draw_points.append((pos * VIS_SCALE).copy())
        if len(draw_points) >= 2:
            pts = np.vstack(draw_points).astype(np.float32)
            line.set_data(pts)

timer = app.Timer(interval=1/100, connect=update, start=True)

if __name__ == "__main__":
    app.run()