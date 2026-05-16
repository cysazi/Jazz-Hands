"""
Hand IMU Visualizer (VisPy + Trimesh + Serial)

GOALS (what this script does):
1) Load a static hand mesh from "hand.obj"
2) Rotate the hand using quaternion data coming from your ESP32 over Serial
3) Let you "re-zero" the orientation (set current pose to be the new 0) when you press the button
4) Apply a fixed 90-degree model offset so the *model's* default pose becomes "palm down"
5) (Optional) Apply a simple axis fix matrix to correct mirrored left/right behavior

EXPECTED SERIAL INPUT (from your ESP32):
- Button event line exactly:           BUTTON:1
- Quaternion line (at least 4 floats): w,x,y,z, ... (can include accel after)

Example quaternion line:
0.999832,0.001234,-0.017890,0.003210,0.12,-0.03,9.81

If your format differs, the parser section is the only thing to change.
"""

from vispy import app, scene
import numpy as np
import serial
import trimesh


# ============================================================
# 1) Quaternion math (core rotation logic)
# ============================================================

def normalize_quat(q):
    """
    Make quaternion unit-length so it represents a pure rotation.
    This prevents scaling / weirdness due to tiny numeric errors.
    """
    w, x, y, z = q
    n = (w*w + x*x + y*y + z*z) ** 0.5
    if n < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    return (w/n, x/n, y/n, z/n)


def quat_conj(q):
    """
    Quaternion conjugate:
      conj(w, x, y, z) = (w, -x, -y, -z)

    For a *unit quaternion*, conjugate == inverse rotation.
    (Important: DO NOT negate w!)
    """
    w, x, y, z = q
    return (w, -x, -y, -z)


def quat_mul(a, b):
    """
    Quaternion multiplication (Hamilton product).
    Combines rotations.

    If a and b are rotations, then:
      a ⊗ b  means apply b first, then apply a.
    """
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw
    )


def quat_to_rotmat(w, x, y, z):
    """
    Convert a UNIT quaternion (w, x, y, z) to a 3x3 rotation matrix.
    """
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)]
    ], dtype=np.float32)


# ============================================================
# 2) Fixed model offset (rotate the OBJ model 90 degrees)
# ============================================================

def rot_x(deg):
    """3x3 rotation matrix around X axis by 'deg' degrees."""
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
    ], dtype=np.float32)


def rot_y(deg):
    """3x3 rotation matrix around Y axis by 'deg' degrees."""
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ], dtype=np.float32)


def rot_z(deg):
    """3x3 rotation matrix around Z axis by 'deg' degrees."""
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=np.float32)


# Choose ONE model offset. Start with +90° about X (what you tried).
# If the model looks wrong, switch to -90, or try Y/Z.
MODEL_OFFSET = rot_x(-90)


# ============================================================
# 3) Optional axis fix (mirror / coordinate correction)
# ============================================================
"""
If turning the IMU right makes the model go left, you likely need a flip.
Try these one at a time:

AXIS_FIX = diag([-1,  1,  1])  # flip X
AXIS_FIX = diag([ 1, -1,  1])  # flip Y
AXIS_FIX = diag([ 1,  1, -1])  # flip Z

Leave as identity to start if you're unsure.
"""
AXIS_FIX = np.diag([1, 1, 1]).astype(np.float32)   # start neutral (no flips)


# ============================================================
# 4) Serial setup
# ============================================================

COM_PORT = "COM5"
BAUD = 115200

# Small timeout so readline() doesn't block for long (keeps the UI responsive)
ser = serial.Serial(COM_PORT, BAUD, timeout=0.01)

# These hold our orientation state
latest_quat = (1.0, 0.0, 0.0, 0.0)  # latest from IMU
zero_quat = None                   # reference "0 pose" (set at start or on button)


# ============================================================
# 5) VisPy scene setup
# ============================================================

canvas = scene.SceneCanvas(keys="interactive", show=True, bgcolor="black", size=(900, 700))
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(fov=45, distance=6.0, center=(0, 0, 0))

# Helpful visuals
view.add(scene.visuals.GridLines(scale=(1, 1), color=(0.3, 0.3, 0.3, 1.0)))
view.add(scene.visuals.XYZAxis(width=2))


# ============================================================
# 6) Load OBJ mesh with trimesh (robust handling)
# ============================================================

OBJ_PATH = "hand.obj"
loaded = trimesh.load(OBJ_PATH)

# Sometimes trimesh returns a Scene (multiple submeshes). Merge them into one mesh.
if isinstance(loaded, trimesh.Scene):
    if len(loaded.geometry) == 0:
        raise ValueError("hand.obj loaded as a Scene but contains no geometry.")
    loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))

mesh = loaded  # now it's a Trimesh object with .vertices and .faces

vertices = mesh.vertices.astype(np.float32)
faces = mesh.faces.astype(np.uint32)


# ============================================================
# 7) Create ONE Mesh visual (DO NOT recreate it)
# ============================================================

hand_tf = scene.transforms.MatrixTransform()

# Scale the model so it fits in view (change if needed)
hand_tf.scale((0.1, 0.1, 0.1))

hand = scene.visuals.Mesh(
    vertices=vertices,
    faces=faces,
    color=(0.8, 0.6, 0.5, 1.0),
    shading="smooth"
)

hand.transform = hand_tf
view.add(hand)


# ============================================================
# 8) Serial parsing + rotation update loop
# ============================================================

def update(event):
    """
    This runs ~100 times per second (because we attach it to a Timer).
    Each frame:
    1) Drain all serial lines currently available
    2) If we saw BUTTON:1 -> set zero_quat (recalibrate)
    3) If we saw quaternion values -> update latest_quat
    4) Compute relative rotation (zero -> current)
    5) Convert to rotation matrix
    6) Apply axis fix + model offset
    7) Put into a 4x4 transform matrix and apply to the mesh
    """
    global latest_quat, zero_quat

    # --------------------------------------------------------
    # A) Read / drain all available serial messages
    # --------------------------------------------------------
    while True:
        raw = ser.readline()
        if not raw:
            break  # nothing else currently in the buffer

        s = raw.decode("utf-8", errors="ignore").strip()
        if not s:
            continue

        # 1) Button press event
        if s == "BUTTON:1":
            # Set the current orientation as our new "zero"
            zero_quat = latest_quat
            print("Calibrated zero orientation (BUTTON:1)")
            continue

        # 2) Quaternion line (first 4 comma-separated values)
        parts = s.split(",")
        if len(parts) < 4:
            continue

        try:
            w, x, y, z = map(float, parts[:4])
            latest_quat = normalize_quat((w, x, y, z))
        except ValueError:
            continue

    # --------------------------------------------------------
    # B) If zero isn't set yet, use the first received quaternion
    # --------------------------------------------------------
    if zero_quat is None:
        zero_quat = latest_quat

    # --------------------------------------------------------
    # C) Compute relative rotation from zero to current
    #     q_rel = inverse(zero) ⊗ current
    # --------------------------------------------------------
    q_rel = quat_mul(quat_conj(zero_quat), latest_quat)

    # --------------------------------------------------------
    # D) Quaternion -> rotation matrix
    # --------------------------------------------------------
    w, x, y, z = q_rel
    x = x
    y = -y
    z = z
    R = quat_to_rotmat(w, x, y, z)

    # --------------------------------------------------------
    # E) Apply coordinate fixes (optional) + model offset (90 deg)
    # --------------------------------------------------------
    # Axis fix corrects mirrored / axis mismatch between sensor and model
    FRAME_MAP = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
    ], dtype=np.float32)

    R = FRAME_MAP @ R @ FRAME_MAP.T

    # Model offset rotates the imported mesh so your "neutral pose" is palm down
    # This applies in the model's local space:
    R = R @ MODEL_OFFSET

    # --------------------------------------------------------
    # F) Put rotation into a 4x4 matrix and apply to the transform
    # --------------------------------------------------------
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    hand_tf.matrix = M

    canvas.update()


# ============================================================
# 9) Timer: this is what makes update() run repeatedly
# ============================================================

timer = app.Timer(interval=1/100, connect=update, start=True)

# Start the app loop
app.run()
