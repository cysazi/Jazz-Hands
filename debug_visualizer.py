import numpy as np
from vispy import app, scene
from vispy.io import read_mesh
from vispy.visuals.transforms import MatrixTransform
import math
import os
import queue
import time

import JazzHandsKalman as jhk
from JazzHandsKalman import DawInterface

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
VELOCITY_STEP_DELTA = 0.1
VELOCITY_STEP_MIN = 0.1
VELOCITY_STEP_MAX = 5.0
ANGULAR_STEP_DELTA = 0.1
ANGULAR_STEP_MIN = 0.1
ANGULAR_STEP_MAX = 10.0


class _NullReader:
    def send_correction(self, relay_id, correction):
        return


class DebugVisualizer(jhk.Visualizer):
    def __init__(self, glove_pair):
        self.glove_pair = glove_pair
        self.left_hand = self.glove_pair.left_hand
        self.right_hand = self.glove_pair.right_hand
        self._ensure_debug_runtime_dependencies()

        super().__init__([glove_pair])
        for lh_view, rh_view in self.view_pairs:
            lh_view.camera.distance = 3
            rh_view.camera.distance = 3

        self.canvas.events.key_press.connect(self.on_key_press)
        self.canvas.events.key_release.connect(self.on_key_release)

        # --- State tracking for keyboard input ---
        self.velocity_step = 0.5
        self.rotation_step = 1.5  # Radians per second
        self.keys_down = {}
        self.angular_velocity_vectors = {
            "LEFT": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "RIGHT": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        }
        self.controlled_hand_label = "LEFT"
        self.hand_meshes = {}
        self.debug_print_interval_seconds = 0.5
        self.last_debug_print_time = 0.0

        self._setup_hand_meshes()

        # --- Status Text ---
        self.status_text = scene.visuals.Text(
            "Initializing...",
            color="white",
            font_size=10,
            pos=(10, 10),
            anchor_x="left",
            anchor_y="bottom",
            parent=self.canvas.scene,
        )

    def _ensure_debug_runtime_dependencies(self):
        if self.glove_pair.reader is None:
            self.glove_pair.reader = _NullReader()
        if self.glove_pair.relay_queue is None:
            self.glove_pair.relay_queue = queue.Queue()
        if not self.right_hand.instrument:
            self.right_hand.instrument = "Synth"

        if not hasattr(self.glove_pair.daw_interface, "previous_note"):
            self.glove_pair.daw_interface.previous_note = jhk.NoteData(
                note=0,
                attack=0,
                instrument="Synth",
                volume=0,
                stereo=0.0,
                reverb_mode=0,
            )

    def _setup_hand_meshes(self):
        if not os.path.exists(HAND_OBJ_PATH):
            raise FileNotFoundError(f"Could not find model: {HAND_OBJ_PATH}")

        vertices, faces, _normals, _texcoords = read_mesh(HAND_OBJ_PATH)
        lh_view, rh_view = self.view_pairs[0]
        self.hand_meshes["LEFT"] = scene.visuals.Mesh(
            vertices=vertices,
            faces=faces,
            color=(0.25, 0.75, 0.95, 0.80),
            shading="smooth",
            parent=lh_view.scene,
        )
        self.hand_meshes["LEFT"].transform = MatrixTransform()
        self.hand_meshes["RIGHT"] = scene.visuals.Mesh(
            vertices=vertices,
            faces=faces,
            color=(0.95, 0.45, 0.25, 0.80),
            shading="smooth",
            parent=rh_view.scene,
        )
        self.hand_meshes["RIGHT"].transform = MatrixTransform()

    def _mesh_transform_matrix(self, hand) -> np.ndarray:
        q_current = np.asarray(hand.rotation_quaternion, dtype=np.float64)
        rotation_matrix = jhk.quaternion_to_transform_matrix(q_current, np.array([0.0, 0.0, 0.0]))[:3, :3]
        rotation_matrix = FRAME_MAP @ rotation_matrix @ FRAME_MAP.T
        rotation_matrix = rotation_matrix @ MODEL_OFFSET

        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rotation_matrix * MODEL_SCALE
        transform[3, :3] = np.asarray(hand.position, dtype=np.float32) * POSITION_SCALE
        return transform

    def _build_debug_packet(self, hand, device_number: int) -> jhk.DevicePacket:
        # Seed synthetic UWB only for the first calibration press.
        should_seed_uwb = hand.glove_state == 0 and bool(hand.button_pressed)

        # update_from_packet() expects raw sensor-frame values, then applies
        # inverse_reference_orientation to produce calibrated/world values.
        raw_position = jhk.rotate_vector_by_quaternion(
            np.asarray(hand.position, dtype=np.float64),
            np.asarray(hand.reference_orientation_quaternion, dtype=np.float64),
        )
        raw_velocity = jhk.rotate_vector_by_quaternion(
            np.asarray(hand.velocity, dtype=np.float64),
            np.asarray(hand.reference_orientation_quaternion, dtype=np.float64),
        )
        raw_quaternion = jhk.quaternion_multiply(
            np.asarray(hand.reference_orientation_quaternion, dtype=np.float64),
            np.asarray(hand.rotation_quaternion, dtype=np.float64),
        )
        p = jhk.PacketData(
            device_number=device_number,
            timestamp=int(time.time() * 1000),
            packet_flags=0,
            button_state=bool(hand.button_pressed),
            pos_x=float(raw_position[0]),
            pos_y=float(raw_position[1]),
            pos_z=float(raw_position[2]),
            vel_x=float(raw_velocity[0]),
            vel_y=float(raw_velocity[1]),
            vel_z=float(raw_velocity[2]),
            UWB_distance_1=float(jhk.DISTANCE_BETWEEN_UWB_ANCHORS) if should_seed_uwb else None,
            UWB_distance_2=float(jhk.DISTANCE_BETWEEN_UWB_ANCHORS) if should_seed_uwb else None,
            UWB_distance_3=float(jhk.DISTANCE_BETWEEN_UWB_ANCHORS) if should_seed_uwb else None,
            UWB_distance_4=float(jhk.DISTANCE_BETWEEN_UWB_ANCHORS) if should_seed_uwb else None,
            UWB_distance_5=float(jhk.DISTANCE_BETWEEN_UWB_ANCHORS) if should_seed_uwb else None,
            quat_w=float(raw_quaternion[0]),
            quat_i=float(raw_quaternion[1]),
            quat_j=float(raw_quaternion[2]),
            quat_k=float(raw_quaternion[3]),
            error_handler=None,
        )
        return jhk.DevicePacket(relay_id=self.glove_pair.relay_id, data=p)

    def on_key_press(self, event):
        """Record that a key is being held down and update motion vectors."""
        if not event.key: return
        key_name = event.key.name.upper()
        self.keys_down[key_name] = True

        if key_name == 'TAB':
            self.controlled_hand_label = "RIGHT" if self.controlled_hand_label == "LEFT" else "LEFT"
            self._update_motion()
            return
        if key_name == 'UP':
            self.velocity_step = float(np.clip(
                self.velocity_step + VELOCITY_STEP_DELTA,
                VELOCITY_STEP_MIN,
                VELOCITY_STEP_MAX,
            ))
            print(f"[DEBUG VIS] velocity_step increased to {self.velocity_step:.2f}")
            self._update_motion()
            return
        if key_name == 'DOWN':
            self.velocity_step = float(np.clip(
                self.velocity_step - VELOCITY_STEP_DELTA,
                VELOCITY_STEP_MIN,
                VELOCITY_STEP_MAX,
            ))
            print(f"[DEBUG VIS] velocity_step decreased to {self.velocity_step:.2f}")
            self._update_motion()
            return
        if key_name == 'RIGHT':
            self.rotation_step = float(np.clip(
                self.rotation_step + ANGULAR_STEP_DELTA,
                ANGULAR_STEP_MIN,
                ANGULAR_STEP_MAX,
            ))
            print(f"[DEBUG VIS] rotation_step increased to {self.rotation_step:.2f}")
            self._update_motion()
            return
        if key_name == 'LEFT':
            self.rotation_step = float(np.clip(
                self.rotation_step - ANGULAR_STEP_DELTA,
                ANGULAR_STEP_MIN,
                ANGULAR_STEP_MAX,
            ))
            print(f"[DEBUG VIS] rotation_step decreased to {self.rotation_step:.2f}")
            self._update_motion()
            return

        self._update_motion()

        if key_name == 'SPACE':
            controlled_hand = self._controlled_hand()
            if not controlled_hand.button_pressed:
                controlled_hand.button_pressed = True
            # Hard-stop active rotation as soon as calibration/draw press begins.
            self.angular_velocity_vectors[self.controlled_hand_label] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            for rotate_key in ("I", "J", "K", "L", "U", "O"):
                self.keys_down[rotate_key] = False

    def on_key_release(self, event):
        """Record that a key is no longer held down and update motion vectors."""
        if not event.key: return
        key_name = event.key.name.upper()
        self.keys_down[key_name] = False
        self._update_motion()

        if key_name == 'SPACE':
            controlled_hand = self._controlled_hand()
            if controlled_hand.button_pressed:
                controlled_hand.button_pressed = False

    def _controlled_hand(self):
        return self.left_hand if self.controlled_hand_label == "LEFT" else self.right_hand

    def _update_motion(self):
        """Calculate the final velocity and angular velocity vectors from the keys_down dictionary."""
        # Linear velocity
        vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        if self.keys_down.get('W', False): vel[1] += self.velocity_step
        if self.keys_down.get('S', False): vel[1] -= self.velocity_step
        if self.keys_down.get('A', False): vel[0] -= self.velocity_step
        if self.keys_down.get('D', False): vel[0] += self.velocity_step
        if self.keys_down.get('Q', False): vel[2] -= self.velocity_step
        if self.keys_down.get('E', False): vel[2] += self.velocity_step
        controlled_hand = self._controlled_hand()
        other_hand = self.right_hand if controlled_hand is self.left_hand else self.left_hand
        controlled_hand.velocity = vel
        other_hand.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Angular velocity (Pitch, Yaw, Roll)
        ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        if self.keys_down.get('I', False): ang_vel[1] += self.rotation_step  # Pitch
        if self.keys_down.get('K', False): ang_vel[1] -= self.rotation_step
        if self.keys_down.get('J', False): ang_vel[2] += self.rotation_step  # Yaw
        if self.keys_down.get('L', False): ang_vel[2] -= self.rotation_step
        if self.keys_down.get('U', False): ang_vel[0] += self.rotation_step  # Roll
        if self.keys_down.get('O', False): ang_vel[0] -= self.rotation_step
        self.angular_velocity_vectors[self.controlled_hand_label] = ang_vel
        other_label = "RIGHT" if self.controlled_hand_label == "LEFT" else "LEFT"
        self.angular_velocity_vectors[other_label] = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    def update(self, event):
        """Called by the timer to update hand positions, rotations, and logic."""
        dt = float(event.dt) if (event is not None and event.dt is not None) else 0.0

        for label, hand in (("LEFT", self.left_hand), ("RIGHT", self.right_hand)):
            previous_state = hand.glove_state
            hand.position += hand.velocity * dt
            angular_velocity = self.angular_velocity_vectors[label]
            rotation_angle = np.linalg.norm(angular_velocity) * dt
            if rotation_angle > 1e-9 and dt > 0.0:
                rotation_axis = angular_velocity / (rotation_angle / dt)
                angle_rad_half = rotation_angle / 2.0
                w = math.cos(angle_rad_half)
                sin_half = math.sin(angle_rad_half)
                x, y, z = rotation_axis * sin_half
                delta_q = np.array([w, x, y, z], dtype=np.float64)

                # Apply rotation in world frame
                hand.rotation_quaternion = jhk.quaternion_multiply(delta_q, hand.rotation_quaternion)

                # Normalize to prevent drift
                norm = np.linalg.norm(hand.rotation_quaternion)
                if norm > 0:
                    hand.rotation_quaternion = hand.rotation_quaternion / norm
                    hand.rotation_euler = np.array(jhk.quat_to_euler_deg(hand.rotation_quaternion), dtype=np.float64)

            device_number = hand.device_id
            packet = self._build_debug_packet(hand, device_number=device_number)
            # Reuse GlovePair production callback path for packet handling.
            self.glove_pair._packet_processor(packet)

            # On first calibration press, hard-stop any active translational/rotational motion.
            if previous_state == 0 and hand.glove_state == 1:
                hand.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                self.angular_velocity_vectors[label] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                for key_name in ("I", "J", "K", "L", "U", "O"):
                    self.keys_down[key_name] = False

        if not (self.left_hand.in_active_area and self.right_hand.in_active_area):
            now = time.perf_counter()
            if (now - self.last_debug_print_time) >= self.debug_print_interval_seconds:
                print(
                    "[DEBUG VIS] MIDI not attempted: "
                    f"left_glove_state={self.left_hand.glove_state}, "
                    f"right_glove_state={self.right_hand.glove_state}, "
                    f"left_active={self.left_hand.in_active_area}, "
                    f"right_active={self.right_hand.in_active_area}, "
                    f"left_calibrated={self.left_hand.is_UWB_calibrated}, "
                    f"right_calibrated={self.right_hand.is_UWB_calibrated}"
                )
                self.last_debug_print_time = now

        if self.glove_pair.daw_interface.port is None:
            now = time.perf_counter()
            if (now - self.last_debug_print_time) >= self.debug_print_interval_seconds:
                print(
                    "[DEBUG VIS] MIDI blocked: daw_interface.port is None."
                )
                self.last_debug_print_time = now

        # Let the base class update both glove visuals in the shared window.
        super().update(event)

        # Keep debug axes fixed at each view origin (no hand-driven transform).
        for hand in (self.left_hand, self.right_hand):
            if hand.visual is not None:
                hand.visual.axis.transform = MatrixTransform()
        self.hand_meshes["LEFT"].transform.matrix = self._mesh_transform_matrix(self.left_hand)
        self.hand_meshes["RIGHT"].transform.matrix = self._mesh_transform_matrix(self.right_hand)

        controlled_hand = self._controlled_hand()
        state_names = {
            0: "Uncalibrated (Press SPACE to start drawing)",
            1: "Drawing Plane (Release SPACE to finalize)",
            2: "Plane Active (X-Section: {x_sec}, Y-Section: {y_sec})"
        }
        status_text_content = state_names.get(controlled_hand.glove_state, "Unknown State")

        if controlled_hand.glove_state == 2:
            status_text_content = status_text_content.format(
                x_sec=controlled_hand.x_section,
                y_sec=controlled_hand.y_section
            )
            status_text_content += f"\nPlane Extents: {controlled_hand.active_area_half_extents[:2] * 2}"
            status_text_content += f"\nPlane Normal: Axis {controlled_hand.plane_normal_axis}"

        self.status_text.text = (
            f"Controlling: {self.controlled_hand_label} (TAB to toggle)\n"
            f"Button: {'PRESSED' if controlled_hand.button_pressed else 'RELEASED'}\n"
            f"Euler: {controlled_hand.in_active_area}\n"
            f"State: {status_text_content}\n"
            f"Step Sizes: move={self.velocity_step:.2f}, rotate={self.rotation_step:.2f}\n"
            "Controls: WASDQE move, IJKLUO rotate, SPACE draw/finalize, TAB switch hand, "
            "UP/DOWN move step, LEFT/RIGHT rotate step"
        )
        self.canvas.update()

    def on_close(self, event):
        """Ensure the application exits cleanly."""
        self.timer.stop()
        app.quit()

def main():
    """Initializes a mock GlovePair and runs the debug visualizer."""
    mock_reader = None
    mock_queue = None
    daw_interface = DawInterface(None)

    glove_pair = jhk.GlovePair(
        left_hand=jhk.LeftHand(device_id=1, active_area_x_subsections=3, active_area_y_subsections=12),
        right_hand=jhk.RightHand(device_id=2, active_area_x_subsections=5, active_area_y_subsections=8),
        relay_id=1,
        reader=mock_reader,
        relay_queue=mock_queue,
        daw_interface = daw_interface
    )

    print("Starting Debug Visualizer...")
    print("Controls:")
    print("  - Movement: WASDQE")
    print("  - Rotation: IJKL (Pitch/Yaw), UO (Roll)")
    print("  - Action: Space to press/hold button.")
    print("  - Toggle active hand: TAB")
    
    visualizer = DebugVisualizer(glove_pair)
    
    app.run()

    print("Debug Visualizer closed.")

if __name__ == "__main__":
    main()
