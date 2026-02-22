from vispy import app, scene
import numpy as np
import serial

# ---------- Quaternion -> 3x3 rotation matrix ----------
def quat_to_rotmat(w, x, y, z):
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)]
    ], dtype=np.float32)

# ---------- Serial ----------
ser = serial.Serial("COM5", 115200, timeout=0.01)

latest_quat = (1.0, 0.0, 0.0, 0.0)  # w,x,y,z fallback

# ---------- VisPy scene ----------
canvas = scene.SceneCanvas(keys="interactive", show=True, bgcolor="black", size=(900, 700))
view = canvas.central_widget.add_view()

view.camera = scene.cameras.TurntableCamera(
    fov=45,
    distance=6.0,
    center=(0, 0, 0)
)

grid = scene.visuals.GridLines(scale=(1, 1), color=(0.3, 0.3, 0.3, 1.0))
view.add(grid)

axes = scene.visuals.XYZAxis(width=2)
view.add(axes)

# ---------- Shared transform (rotation lives here) ----------
cube_tf = scene.transforms.MatrixTransform()

# ---------- Color cycling ----------
colors = [
    (0.3, 0.6, 1.0, 1.0),  # blue
    (1.0, 0.2, 0.2, 1.0),  # red
    (0.2, 1.0, 0.2, 1.0),  # green
    (1.0, 1.0, 0.2, 1.0),  # yellow
    (1.0, 0.2, 1.0, 1.0),  # magenta
]
color_idx = 0

def make_cube(rgba):
    """Create a Box, attach the shared transform, add to view, return it."""
    c = scene.visuals.Box(
        width=1,
        height=1,
        depth=1,
        color=rgba,
        edge_color="white"
    )
    c.transform = cube_tf
    view.add(c)
    return c

cube = make_cube(colors[color_idx])

def cycle_cube_color():
    """Remove current cube and recreate it with the next color (keeps same transform)."""
    global cube, color_idx

    color_idx = (color_idx + 1) % len(colors)
    rgba = colors[color_idx]
    print("Applying:", rgba)

    # remove old cube from scenegraph
    cube.parent = None

    # create new cube with new color
    cube = make_cube(rgba)

    canvas.update()

def update(event):
    global latest_quat

    # 1) Read all available serial lines quickly
    while True:
        raw = ser.readline()
        if not raw:
            break

        s = raw.decode("utf-8", errors="ignore").strip()
        if not s:
            continue

        # A) Button event line
        if s == "BUTTON:1":
            cycle_cube_color()
            continue

        # B) Quaternion line: expects at least 4 comma-separated floats
        parts = s.split(",")
        if len(parts) < 4:
            continue

        try:
            w, x, y, z = map(float, parts[:4])
            norm = (w*w + x*x + y*y + z*z) ** 0.5
            if norm > 1e-6:
                latest_quat = (w/norm, x/norm, y/norm, z/norm)
        except ValueError:
            continue

    # 2) Convert quaternion to rotation matrix
    w, x, y, z = latest_quat
    R = quat_to_rotmat(w, x, y, z)

    # 3) Apply rotation to cube transform
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    cube_tf.matrix = M

    canvas.update()

timer = app.Timer(interval=1/60, connect=update, start=True)
app.run()
