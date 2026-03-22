import numpy as np
from vispy import app, scene
from vispy.app import Timer
import os
from vispy.io import read_mesh
from vispy.visuals.transforms import MatrixTransform
import math

# Import necessary components from your main script and helpers
import JazzHandsKalman as jhk
from Visualization_Tests.Visualization_Helper_Functions import setup_canvas

# region Constants from Accel_Filtering_Test.py
CURRENT_FILEPATH = os.path.dirname(os.path.abspath(__file__))
HAND_OBJ_PATH = os.path.join(CURRENT_FILEPATH, "Visualization_Tests", "hand.obj")

POSITION_SCALE = 1.0
MODEL_SCALE = 0.02

FRAME_MAP = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
], dtype=np.float32)

def rot_x(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)

MODEL_OFFSET = rot_x(-90.0)
# endregion

class DebugVisualizer:
    def __init__(self, glove_pair):
        self.glove_pair = glove_pair
        self.left_hand = self.glove_pair.left_hand

        self.canvas, self.view, _, _, _, _, _ = setup_canvas()
        self.view.camera.distance = 3

        # --- Load Hand Mesh ---
        if not os.path.exists(HAND_OBJ_PATH):
            raise FileNotFoundError(f"Could not find model: {HAND_OBJ_PATH}")
        vertices, faces, _normals, _texcoords = read_mesh(HAND_OBJ_PATH)
        self.hand_mesh = scene.visuals.Mesh(
            vertices=vertices,
            faces=faces,
            color=(0.25, 0.75, 0.95, 0.80),
            shading="smooth",
            parent=self.view.scene,
        )
        self.hand_mesh.transform = MatrixTransform()
        # --- End Hand Mesh ---

        self.canvas.events.key_press.connect(self.on_key_press)
        self.canvas.events.key_release.connect(self.on_key_release)
        self.canvas.events.close.connect(self.on_close)

        self.timer = Timer('auto', connect=self.update, start=True)

        # --- State tracking for keyboard input ---
        self.velocity_step = 0.5
        self.rotation_step = 1.5  # Radians per second
        self.keys_down = {}
        self.angular_velocity_vector = np.array([0.0, 0.0, 0.0]) # Pitch, Yaw, Roll

    def on_key_press(self, event):
        """Record that a key is being held down and update motion vectors."""
        if not event.key: return
        self.keys_down[event.key.name.upper()] = True
        self._update_motion()

        if event.key.name.upper() == 'SPACE':
            if not self.left_hand.button_pressed:
                self.left_hand.button_pressed = True

    def on_key_release(self, event):
        """Record that a key is no longer held down and update motion vectors."""
        if not event.key: return
        self.keys_down[event.key.name.upper()] = False
        self._update_motion()

        if event.key.name.upper() == 'SPACE':
            if self.left_hand.button_pressed:
                self.left_hand.button_pressed = False

    def _update_motion(self):
        """Calculate the final velocity and angular velocity vectors from the keys_down dictionary."""
        # Linear velocity
        vel = np.array([0.0, 0.0, 0.0])
        if self.keys_down.get('W', False): vel[1] += self.velocity_step
        if self.keys_down.get('S', False): vel[1] -= self.velocity_step
        if self.keys_down.get('A', False): vel[0] -= self.velocity_step
        if self.keys_down.get('D', False): vel[0] += self.velocity_step
        if self.keys_down.get('Q', False): vel[2] += self.velocity_step
        if self.keys_down.get('E', False): vel[2] -= self.velocity_step
        self.left_hand.velocity = vel

        # Angular velocity (Pitch, Yaw, Roll)
        ang_vel = np.array([0.0, 0.0, 0.0])
        if self.keys_down.get('I', False): ang_vel[1] += self.rotation_step  # Pitch
        if self.keys_down.get('K', False): ang_vel[1] -= self.rotation_step
        if self.keys_down.get('J', False): ang_vel[2] += self.rotation_step  # Yaw
        if self.keys_down.get('L', False): ang_vel[2] -= self.rotation_step
        if self.keys_down.get('U', False): ang_vel[0] += self.rotation_step  # Roll
        if self.keys_down.get('O', False): ang_vel[0] -= self.rotation_step
        self.angular_velocity_vector = ang_vel

    def update(self, event):
        """Called by the timer to update hand position, rotation, and logic."""
        # --- Update Position ---
        self.left_hand.position += self.left_hand.velocity * event.dt

        # --- Update Rotation ---
        rotation_angle = np.linalg.norm(self.angular_velocity_vector) * event.dt
        if rotation_angle > 1e-9:
            rotation_axis = self.angular_velocity_vector / (rotation_angle / event.dt)
            
            angle_rad_half = rotation_angle / 2.0
            w = math.cos(angle_rad_half)
            sin_half = math.sin(angle_rad_half)
            x, y, z = rotation_axis * sin_half
            
            delta_q = np.array([w, x, y, z])
            
            # Apply rotation in world frame
            self.left_hand.rotation_quaternion = jhk.quaternion_multiply(delta_q, self.left_hand.rotation_quaternion)
            
            # Normalize to prevent drift
            norm = np.linalg.norm(self.left_hand.rotation_quaternion)
            if norm > 0:
                self.left_hand.rotation_quaternion /= norm

        # --- Run Logic ---
        self.left_hand.interpret_position()

        # --- Update Visual Transform ---
        pos = self.left_hand.position
        q_current = self.left_hand.rotation_quaternion
        R = jhk.quaternion_to_transform_matrix(q_current, np.array([0,0,0]))[:3, :3]
        R = FRAME_MAP @ R @ FRAME_MAP.T
        R = R @ MODEL_OFFSET

        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = R * MODEL_SCALE
        M[3, :3] = pos * POSITION_SCALE
        self.hand_mesh.transform.matrix = M
        # --- End Visual Transform ---

        self.canvas.update()

    def on_close(self, event):
        """Ensure the application exits cleanly."""
        self.timer.stop()
        app.quit()

def main():
    """Initializes a mock GlovePair and runs the debug visualizer."""
    mock_reader = None
    mock_queue = None

    glove_pair = jhk.GlovePair(
        device_ids=(1, 2),
        relay_id=1,
        reader=mock_reader,
        relay_queue=mock_queue
    )

    print("Starting Debug Visualizer...")
    print("Controls:")
    print("  - Movement: WASDQE")
    print("  - Rotation: IJKL (Pitch/Yaw), UO (Roll)")
    print("  - Action: Space to press/hold button.")
    
    visualizer = DebugVisualizer(glove_pair)
    
    app.run()

    print("Debug Visualizer closed.")

if __name__ == "__main__":
    main()
