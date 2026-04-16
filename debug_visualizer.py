import numpy as np
from vispy import app, scene
from vispy.io import read_mesh
from vispy.visuals.transforms import MatrixTransform
import math
import os
import queue
import threading
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
    def send_correction(self, relay_id, device_id, correction):
        return


class DebugVisualizer(jhk.Visualizer):
    def __init__(self, glove_pair):
        self.glove_pair = glove_pair
        self.left_hand = self.glove_pair.left_hand
        self.right_hand = self.glove_pair.right_hand
        self._ensure_debug_runtime_dependencies()

        # Initialize the base class, which creates the scenes and GloveVisuals
        super().__init__([glove_pair])

        # Hide the XYZ axes created by the base class to avoid visual clutter
        for pair in self.glove_pairs:
            pair.left_hand.visual.axis.visible = False
            pair.right_hand.visual.axis.visible = False

        # Explicitly link the cameras to ensure they are synchronized
        lh_view, rh_view = self.view_pairs[0]
        lh_view.camera.link(rh_view.camera)

        for view_pair in self.view_pairs:
            view_pair[0].camera.distance = 3
            view_pair[1].camera.distance = 3

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
        self.debug_print_interval_seconds = 0.5
        self.last_debug_print_time = 0.0

        # Per-hand SPACE debounce to ensure seeding runs only once per press
        self.space_consumed = {"LEFT": False, "RIGHT": False}
        # Track whether a calibration or correction thread is running for each hand
        self._calibrating = {"LEFT": False, "RIGHT": False}
        self._correction_running = {"LEFT": False, "RIGHT": False}
        # Do not start the glove_pair processor in debug mode; handle visuals on the UI thread
        # and run heavy tasks (calibration/corrections) in dedicated background threads to avoid blocking.

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

    def _normalize_key_name(self, raw_name):
        """Normalize event.key.name values to consistent uppercase tokens.

        VisPy may represent the spacebar as ' ' or 'Space' depending on platform; normalize
        these to the token 'SPACE'. Other keys are returned as their upper-case name.
        """
        if raw_name is None:
            return None
        s = str(raw_name)
        # Treat empty/whitespace names as the spacebar
        if s.strip() == "":
            return 'SPACE'
        if s.lower() == 'space':
            return 'SPACE'
        return s.upper()

    def _ensure_debug_runtime_dependencies(self):
        if self.glove_pair.reader is None:
            self.glove_pair.reader = _NullReader()
        # No need for a real queue in synchronous debug mode
        if self.glove_pair.relay_queue is None:
            self.glove_pair.relay_queue = queue.Queue()  # The class expects it, but we won't use it
        if not self.right_hand.instrument:
            self.right_hand.instrument = "Synth"

        if not hasattr(self.glove_pair.daw_interface, "previous_note"):
            self.glove_pair.daw_interface.previous_note = jhk.NoteData.blank_note()

    def _build_debug_packet(self, hand, device_number: int) -> jhk.DevicePacket:
        is_controlled_hand = (self.controlled_hand_label == "LEFT" and hand is self.left_hand) or \
                             (self.controlled_hand_label == "RIGHT" and hand is self.right_hand)
        current_button_state = self.keys_down.get('SPACE', False) if is_controlled_hand else False
        # Seed UWB drawing once per press for the controlled hand to avoid repeated heavy operations.
        label = "LEFT" if hand is self.left_hand else "RIGHT"
        should_seed_uwb = False
        if hand.glove_state == 0 and current_button_state:
            if is_controlled_hand:
                should_seed_uwb = True
        else:
            should_seed_uwb = False

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
            button_state=current_button_state,
            pos_x=float(raw_position[0]),
            pos_y=float(raw_position[1]),
            pos_z=float(raw_position[2]),
            # Keep velocities zero for debug packets to avoid extra dead-reckoning in Kalman
            vel_x=0.0,
            vel_y=0.0,
            vel_z=0.0,
            # If seeding UWB (first SPACE press), provide a full set of short distances so calibration can proceed.
            UWB_distance_1=1.0 if should_seed_uwb else None,
            UWB_distance_2=1.0 if should_seed_uwb else None,
            UWB_distance_3=1.0 if should_seed_uwb else None,
            UWB_distance_4=1.0 if should_seed_uwb else None,
            UWB_distance_5=1.0 if should_seed_uwb else None,
            quat_w=float(raw_quaternion[0]),
            quat_i=float(raw_quaternion[1]),
            quat_j=float(raw_quaternion[2]),
            quat_k=float(raw_quaternion[3]),
            error_handler=None,
        )
        return jhk.DevicePacket(relay_id=self.glove_pair.relay_id, data=p)

    def on_key_press(self, event):
        if not event.key:
            return
        key_name = self._normalize_key_name(event.key.name)
        if key_name is None:
            return
        self.keys_down[key_name] = True

        if key_name == 'TAB':
            self.controlled_hand_label = "RIGHT" if self.controlled_hand_label == "LEFT" else "LEFT"
            self._update_motion()
            return
        if key_name == 'UP':
            self.velocity_step = float(np.clip(
                self.velocity_step + VELOCITY_STEP_DELTA, VELOCITY_STEP_MIN, VELOCITY_STEP_MAX))
            print(f"[DEBUG VIS] velocity_step increased to {self.velocity_step:.2f}")
            self._update_motion()
            return
        if key_name == 'DOWN':
            self.velocity_step = float(np.clip(
                self.velocity_step - VELOCITY_STEP_DELTA, VELOCITY_STEP_MIN, VELOCITY_STEP_MAX))
            print(f"[DEBUG VIS] velocity_step decreased to {self.velocity_step:.2f}")
            self._update_motion()
            return
        if key_name == 'RIGHT':
            self.rotation_step = float(np.clip(
                self.rotation_step + ANGULAR_STEP_DELTA, ANGULAR_STEP_MIN, ANGULAR_STEP_MAX))
            print(f"[DEBUG VIS] rotation_step increased to {self.rotation_step:.2f}")
            self._update_motion()
            return
        if key_name == 'LEFT':
            self.rotation_step = float(np.clip(
                self.rotation_step - ANGULAR_STEP_DELTA, ANGULAR_STEP_MIN, ANGULAR_STEP_MAX))
            print(f"[DEBUG VIS] rotation_step decreased to {self.rotation_step:.2f}")
            self._update_motion()
            return

        self._update_motion()

        if key_name == 'SPACE':
            # Reset rotational inputs for the controlled hand while drawing
            self.angular_velocity_vectors[self.controlled_hand_label] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            for rotate_key in ("I", "J", "K", "L", "U", "O"):
                self.keys_down[rotate_key] = False

    def on_key_release(self, event):
        if not event.key:
            return
        key_name = self._normalize_key_name(event.key.name)
        if key_name is None:
            return
        # Ensure both normalized and common variants are cleared to avoid stale state
        self.keys_down[key_name] = False
        # Also clear common raw-space variants if present
        if key_name == 'SPACE':
            self.keys_down.pop(' ', None)
            self.keys_down.pop('Space', None)
            # Reset per-hand space consumption so next press will be handled
            self.space_consumed['LEFT'] = False
            self.space_consumed['RIGHT'] = False
        self._update_motion()

    def _controlled_hand(self):
        return self.left_hand if self.controlled_hand_label == "LEFT" else self.right_hand

    def _update_motion(self):
        vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        if self.keys_down.get('W', False): vel[1] += self.velocity_step
        if self.keys_down.get('S', False): vel[1] -= self.velocity_step
        if self.keys_down.get('A', False): vel[0] -= self.velocity_step
        if self.keys_down.get('D', False): vel[0] += self.velocity_step
        if self.keys_down.get('Q', False): vel[2] -= self.velocity_step
        if self.keys_down.get('E', False): vel[2] += self.velocity_step
        controlled_hand = self._controlled_hand()
        other_hand = self.right_hand if controlled_hand is self.left_hand else self.left_hand
        # Do NOT assign velocities directly to the Glove objects here. Instead, construct a
        # debug PacketData that contains the requested velocity and feed it to the packet processor
        # so all state updates go through the same code path.

        ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        if self.keys_down.get('I', False): ang_vel[1] += self.rotation_step
        if self.keys_down.get('K', False): ang_vel[1] -= self.rotation_step
        if self.keys_down.get('J', False): ang_vel[2] += self.rotation_step
        if self.keys_down.get('L', False): ang_vel[2] -= self.rotation_step
        if self.keys_down.get('U', False): ang_vel[0] += self.rotation_step
        if self.keys_down.get('O', False): ang_vel[0] -= self.rotation_step
        self.angular_velocity_vectors[self.controlled_hand_label] = ang_vel
        other_label = "RIGHT" if self.controlled_hand_label == "LEFT" else "LEFT"
        self.angular_velocity_vectors[other_label] = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Build a debug packet immediately on key changes and feed it to the packet processor
        try:
            packet = self._build_debug_packet(controlled_hand, device_number=controlled_hand.device_id)
            # Override the packet's velocity fields with the velocity requested by the keypress
            # and ensure the packet position reflects the current visual position.
            packet.data.vel_x = float(vel[0])
            packet.data.vel_y = float(vel[1])
            packet.data.vel_z = float(vel[2])
            packet.data.pos_x = float(controlled_hand.position[0])
            packet.data.pos_y = float(controlled_hand.position[1])
            packet.data.pos_z = float(controlled_hand.position[2])


            # Non-blocking enqueue so the UI thread doesn't stall
            try:
                self.glove_pair.relay_queue.put_nowait(packet)
            except Exception:
                # If no queue/queue full, fall back to synchronous processing
                try:
                    self.glove_pair._packet_processor(packet)
                except Exception:
                    pass
        except Exception:
            # Building/feeding debug packet failed; ignore to keep UI responsive
            pass

    def update(self, event):
        dt = float(event.dt) if (event is not None and event.dt is not None) else 0.0

        for label, hand in (("LEFT", self.left_hand), ("RIGHT", self.right_hand)):
            previous_state = hand.glove_state

            # Apply simulated physics
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
                hand.rotation_quaternion = jhk.quaternion_multiply(delta_q, hand.rotation_quaternion)
                norm = np.linalg.norm(hand.rotation_quaternion)
                if norm > 0:
                    hand.rotation_quaternion = hand.rotation_quaternion / norm
                    hand.rotation_euler = np.array(jhk.quat_to_euler_deg(hand.rotation_quaternion), dtype=np.float64)

            # Build and process the packet synchronously
            device_number = hand.device_id
            packet = self._build_debug_packet(hand, device_number=device_number)

            # Immediate visual updates already applied (position/rotation from simulated physics).
            # If this is the first SPACE press that should seed UWB, write short distances directly
            # into the Glove object so calibration can use them later.
            now_ts = time.time()
            # Reflect button state to the Glove object so visuals and status text stay correct
            hand.button_pressed = packet.data.button_state
            # If the debug packet carried UWB seed distances, copy them into the glove's UWB cache.
            if any(getattr(packet.data, f"UWB_distance_{i}") is not None for i in range(1, 6)):
                try:
                    hand._store_uwb_distance("UWB_distance_1", "UWB_timestamp_1", packet.data.UWB_distance_1, now_ts)
                    hand._store_uwb_distance("UWB_distance_2", "UWB_timestamp_2", packet.data.UWB_distance_2, now_ts)
                    hand._store_uwb_distance("UWB_distance_3", "UWB_timestamp_3", packet.data.UWB_distance_3, now_ts)
                    hand._store_uwb_distance("UWB_distance_4", "UWB_timestamp_4", packet.data.UWB_distance_4, now_ts)
                    hand._store_uwb_distance("UWB_distance_5", "UWB_timestamp_5", packet.data.UWB_distance_5, now_ts)
                except Exception:
                    pass

            # Handle calibration start (offload to background so UI stays smooth)
            label = "LEFT" if hand is self.left_hand else "RIGHT"
            if hand.glove_state == 0 and packet.data.button_state and not self._calibrating[
                label] and not hand.is_UWB_calibrated:
                # Start calibration in a daemon thread
                def _do_calibrate(h=hand, lbl=label):
                    try:
                        self._calibrating[lbl] = True
                        success = h.calibrate_zero_frame()
                        if success:
                            h.glove_state = 1
                            h.position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                            if h.visual:
                                h.visual.box.visible = True
                    except Exception as e:
                        print(f"Calibration thread error for {lbl}: {e}")
                    finally:
                        self._calibrating[lbl] = False

                t = threading.Thread(target=_do_calibrate, daemon=True)
                t.start()

            # Finalize plane on SPACE release (handle on UI thread to manipulate visuals safely)
            if hand.glove_state == 1 and not packet.data.button_state and hand.visual:
                # Equivalent to the 'release' path in get_section_from_position
                abs_pos = np.abs(hand.position)
                plane_axes = [i for i in range(3) if i != hand.plane_normal_axis]
                max_extent = max(abs_pos[plane_axes[0]], abs_pos[plane_axes[1]])
                hand.active_area_half_extents = np.full(3, max_extent)
                hand.active_area_half_extents[hand.plane_normal_axis] = 0.0
                # Ensure box transform matches
                transform = np.eye(4)
                transform[0, 0] = abs(hand.active_area_half_extents[0] * 2.0) if abs(
                    hand.active_area_half_extents[0] * 2.0) > 1e-6 else 1e-6
                transform[1, 1] = abs(hand.active_area_half_extents[1] * 2.0) if abs(
                    hand.active_area_half_extents[1] * 2.0) > 1e-6 else 1e-6
                transform[2, 2] = abs(hand.active_area_half_extents[2] * 2.0) if abs(
                    hand.active_area_half_extents[2] * 2.0) > 1e-6 else 1e-6
                transform[:3, 3] = [0.0, 0.0, 0.0]
                hand.visual.box.transform.matrix = transform
                hand.glove_state = 2

            # Offload expensive correction sending to background threads (do not block UI)
            if hand.glove_state == 2 and not self._correction_running[label]:
                def _do_correction(h=hand, lbl=label):
                    try:
                        self._correction_running[lbl] = True
                        # Use the debug visualizer's reader to send corrections (no-op in _NullReader)
                        h.calculate_and_send_correction(self.glove_pair.reader, self.glove_pair.relay_id, h.device_id)
                    except Exception as e:
                        print(f"Correction thread error for {lbl}: {e}")
                    finally:
                        self._correction_running[lbl] = False

                tc = threading.Thread(target=_do_correction, daemon=True)
                tc.start()

            # Finally, enqueue the packet non-blocking for any other background consumers (if present)
            try:
                self.glove_pair.relay_queue.put_nowait(packet)
            except Exception:
                pass

            if previous_state == 0 and hand.glove_state == 1:
                hand.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                self.angular_velocity_vectors[label] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                for key_name in ("I", "J", "K", "L", "U", "O"):
                    self.keys_down[key_name] = False

        # Let the base class update all visuals
        super().update(event)

        # Update status text
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
            f"State: {status_text_content}\n"
            f"Calibrated: {controlled_hand.is_UWB_calibrated}\n"
            f"Step Sizes: move={self.velocity_step:.2f}, rotate={self.rotation_step:.2f}\n"
            "Controls: WASDQE move, IJKLUO rotate, SPACE draw/finalize, TAB switch hand, "
            "UP/DOWN move step, LEFT/RIGHT rotate step"
        )
        self.canvas.update()

    def on_close(self, event):
        self.timer.stop()
        # Stop the glove_pair background processor if it's running
        try:
            self.glove_pair.stop()
        except Exception:
            pass
        app.quit()


def main():
    mock_reader = _NullReader()
    mock_queue = queue.Queue()  # Still needed for class initialization
    daw_interface = DawInterface(None)

    mock_anchor_positions = {
        1: np.array([0.0, 0.0, 0.0]), 2: np.array([1.0, 0.0, 0.0]),
        3: np.array([0.5, 0.866, 0.0]), 4: np.array([0.5, 0.433, 0.866]),
        5: np.array([0.5, 0.433, -0.866]),
    }

    glove_pair = jhk.GlovePair(
        left_hand=jhk.LeftHand(
            device_id=1, active_area_x_subsections=3, active_area_y_subsections=12,
            anchor_positions=mock_anchor_positions, triangulate_func=jhk.multilaterate_3d_wls),
        right_hand=jhk.RightHand(
            device_id=2, active_area_x_subsections=5, active_area_y_subsections=8,
            anchor_positions=mock_anchor_positions, triangulate_func=jhk.multilaterate_3d_wls),
        relay_id=1, reader=mock_reader, relay_queue=mock_queue, daw_interface=daw_interface)

    # Do not start the glove_pair thread in debug mode

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
