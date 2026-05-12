# region ======================= Imports =======================
import math
import os
import sys
# Limit BLAS/OpenMP threads to 1 to avoid periodic thread contention and latency spikes in real-time.
# Must be set before importing numpy.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np

import serial
import struct
import queue
import time
import threading
import subprocess
from serial.tools import list_ports

from dataclasses import dataclass, field
from collections import deque
from pathlib import Path

from vispy import app, scene
from vispy.scene import visuals
from vispy.app import Timer
from vispy.io import read_mesh
from vispy.visuals.transforms import MatrixTransform

import mido

from haptics_controller import HapticsController

# endregion

# region ======================= Constants and Definitions =======================

# --- Serial and Packet Definitions ---
COM_PORTS: list[str] = []
SERIAL_BAUD: int = 921600
PACKET_HEADER: int = 0xAAAA
HEADER_BYTE: bytes = b'\xAA\xAA'
PACKET_STRUCT = struct.Struct('<HBIIBB7fB')
PACKET_SIZE: int = PACKET_STRUCT.size
assert PACKET_SIZE == 42, "ESP-NOW receiver packet must match hand_imu_packet_t"
HAPTICS_COMMAND_STRUCT = struct.Struct("<HBBH")
HAPTICS_COMMAND_HEADER: int = 0xCC33

# --- Bit Flagging Constants ---
PACKET_HAS_ACCEL: int = 0b00000001
PACKET_HAS_QUAT: int = 0b00000010
PACKET_HAS_BUTTON: int = 0b00000100
PACKET_HAS_ERROR: int = 0b10000000
ERROR_ESPNOW_SEND: int = 0b00000001
ERROR_QUAT_STALE: int = 0b00000010
ERROR_ACCEL_STALE: int = 0b00000100

# --- System and Physics Constants ---
BOUNDING_BOX_NORMAL_TOLERANCE: float = 0.15  # meters, ±tolerance in normal direction
MIDI_DEBUG_LOGGING: bool = True
PERFORMANCE_MODE: bool = False  # for mirroring the L and R hands
# Reduce console logging for real-time performance
DEBUG_LOGGING: bool = True
CAMERA_POSITION_STALE_SECONDS: float = 0.35

# Camera integration: set to 2 or 4.
CAMERA_MODE: int = 2
CAMERA_UPDATE_HZ: float = 120.0
CAMERA_HAND_SIDE_AXIS: int = 0  # x axis: lower x is left, higher x is right
CAMERA_POSITION_SCALE = np.array([4.0, 4.0, 4.0], dtype=np.float64)

ENABLE_HAPTICS: bool = True
ENABLE_NOTE_HAPTICS: bool = True
HAPTICS_PORT: str | None = None
HAPTICS_BAUD: int = 115200
HAPTICS_NOTE_INTENSITY: int = 150
HAPTICS_NOTE_DURATION_MS: int = 55

ROOT_DIR = Path(__file__).resolve().parent
CAMERA_TESTS_DIR = ROOT_DIR / "camera_tests"
TWO_CAMERA_IDS: tuple[int, int] = (1, 2)
TWO_CAMERA_IDS_ARG = ",".join(str(camera_id) for camera_id in TWO_CAMERA_IDS)
TWO_CAMERA_SETTINGS_JSON = CAMERA_TESTS_DIR / "camera_uvc_settings_values.json"
TWO_CAMERA_CALIBRATION_JSON = CAMERA_TESTS_DIR / "mocap_calibration.json"
TWO_CAMERA_ALIGNED_CALIBRATION_JSON = CAMERA_TESTS_DIR / "mocap_calibration_aligned.json"

INSTRUMENT_CYCLE: tuple[str, ...] = (
    "Synth",
    "Violin",
    "Horn",
    "Instrument4",
    "Instrument5",
    "Instrument6",
    "Instrument7",
    "Instrument8",
    "Instrument9",
)
GESTURE_ROLL_RATE_THRESHOLD_DPS: float = 360.0
GESTURE_ROLL_DELTA_THRESHOLD_DEG: float = 40.0
GESTURE_ROLL_ANGLE_THRESHOLD_DEG: float = 100.0
GESTURE_ACCEL_SPIKE_THRESHOLD_MPS2: float = 3.0
GESTURE_WINDOW_MS: int = 220
GESTURE_COOLDOWN_MS: int = 300

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

# Helper: validate positions to avoid NaNs or wildly incorrect tracking results
def is_reasonable_position(pos, max_norm=50.0):
    if pos is None:
        return False
    arr = np.asarray(pos, dtype=float)
    if not np.all(np.isfinite(arr)):
        return False
    if np.linalg.norm(arr) > max_norm:
        return False
    return True


def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        response = input(f"{prompt} {suffix} ").strip().lower()
        if not response:
            return default
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def _run_python_step(script_path: Path, args: list[str], description: str) -> None:
    command = [sys.executable, "-u", str(script_path), *args]
    print(f"\n[{description}] starting")
    print(" ".join(command))
    completed = subprocess.run(command, cwd=str(ROOT_DIR), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{description} failed with exit code {completed.returncode}")


def maybe_run_two_camera_calibration() -> None:
    print("\nJazz Hands startup")
    print(f"2-camera mode is active: cameras {TWO_CAMERA_IDS_ARG}")

    _run_python_step(
        CAMERA_TESTS_DIR / "camera_uvc_settings.py",
        [
            "--apply",
            "--camera-ids",
            TWO_CAMERA_IDS_ARG,
            "--settings-json",
            str(TWO_CAMERA_SETTINGS_JSON),
        ],
        "Apply saved camera settings",
    )

    if not prompt_yes_no("Do you want to initiate calibration?", default=False):
        if not TWO_CAMERA_ALIGNED_CALIBRATION_JSON.exists() and not TWO_CAMERA_CALIBRATION_JSON.exists():
            print(
                "Warning: no saved camera calibration file was found. "
                "The tracker may not start until you run calibration."
            )
        return

    _run_python_step(
        CAMERA_TESTS_DIR / "calibrate_mocap_cameras.py",
        [
            "--cameras",
            TWO_CAMERA_IDS_ARG,
            "--output",
            str(TWO_CAMERA_CALIBRATION_JSON),
            "--threaded",
            "--exit-after-save",
        ],
        "2-camera calibration",
    )
    _run_python_step(
        CAMERA_TESTS_DIR / "mocap_movement_alignment.py",
        [
            "--cameras",
            TWO_CAMERA_IDS_ARG,
            "--calibration",
            str(TWO_CAMERA_CALIBRATION_JSON),
            "--output",
            str(TWO_CAMERA_ALIGNED_CALIBRATION_JSON),
            "--threaded",
            "--exit-after-save",
        ],
        "2-camera movement alignment",
    )


def discover_serial_ports() -> list[str]:
    if COM_PORTS:
        return list(COM_PORTS)

    ports = list(list_ports.comports())
    if not ports:
        return []

    preferred_keywords = (
        "usb",
        "uart",
        "cp210",
        "ch340",
        "wch",
        "silicon labs",
        "esp32",
    )
    ranked: list[tuple[int, str]] = []
    for port_info in ports:
        text = " ".join(
            str(value).lower()
            for value in (
                port_info.device,
                port_info.description,
                port_info.manufacturer,
                port_info.hwid,
            )
            if value
        )
        priority = 0 if any(keyword in text for keyword in preferred_keywords) else 1
        ranked.append((priority, str(port_info.device)))

    ranked.sort(key=lambda item: (item[0], item[1]))
    return [device for _priority, device in ranked]



# endregion

# region ======================= Data Classes =======================

@dataclass
class PacketData:
    """Represents one ESP-NOW IMU packet forwarded by the receiver ESP32."""
    device_number: int
    timestamp: int
    timestamp_us: int
    sequence: int
    packet_type: int
    packet_flags: int
    button_pressed: bool
    accel_x: float | None
    accel_y: float | None
    accel_z: float | None
    quat_w: float | None
    quat_i: float | None
    quat_j: float | None
    quat_k: float | None
    error_handler: int

    @classmethod
    def from_bytes(cls, binary_data: bytes) -> "PacketData":
        """Parses a byte array into a PacketData object."""
        if len(binary_data) != PACKET_SIZE:
            raise ValueError(f"Expected packet size {PACKET_SIZE}, got {len(binary_data)}")

        unpacked = PACKET_STRUCT.unpack(binary_data)
        (
            header,
            device_id,
            timestamp_us,
            sequence,
            packet_flags,
            button_pressed,
            accel_x,
            accel_y,
            accel_z,
            quat_w,
            quat_i,
            quat_j,
            quat_k,
            error_handler,
        ) = unpacked
        if header != PACKET_HEADER:
            raise ValueError(f"Unexpected packet header 0x{header:04X}")

        return cls(
            device_number=int(device_id),
            timestamp=int(timestamp_us // 1000),
            timestamp_us=int(timestamp_us),
            sequence=int(sequence),
            packet_type=int(packet_flags),
            packet_flags=int(packet_flags),
            button_pressed=bool(button_pressed) if (packet_flags & PACKET_HAS_BUTTON) else False,
            accel_x=float(accel_x) if (packet_flags & PACKET_HAS_ACCEL) else None,
            accel_y=float(accel_y) if (packet_flags & PACKET_HAS_ACCEL) else None,
            accel_z=float(accel_z) if (packet_flags & PACKET_HAS_ACCEL) else None,
            quat_w=float(quat_w) if (packet_flags & PACKET_HAS_QUAT) else None,
            quat_i=float(quat_i) if (packet_flags & PACKET_HAS_QUAT) else None,
            quat_j=float(quat_j) if (packet_flags & PACKET_HAS_QUAT) else None,
            quat_k=float(quat_k) if (packet_flags & PACKET_HAS_QUAT) else None,
            error_handler=int(error_handler) if (packet_flags & PACKET_HAS_ERROR) else 0,
        )


@dataclass
class DevicePacket:
    """Wrapper for a packet with relay metadata."""
    relay_id: int
    data: PacketData
    timestamp: float = field(default_factory=time.time)


# endregion

# region ======================= Threaded Device Reader =======================

class ThreadedMultiDeviceReader:
    """Manages serial connections and data parsing for multiple devices in separate threads."""

    def __init__(self):
        self.processing_queues: dict[int, queue.Queue] = {}
        self.relays: dict[int, dict] = {}
        self.threads: dict[int, threading.Thread] = {}
        self.running: bool = False
        self.lock = threading.Lock()

    def add_device(self, relay_id: int, port: str, baudrate: int = SERIAL_BAUD) -> bool:
        """Adds a device to be monitored and starts its reading thread."""
        with self.lock:
            if relay_id in self.relays:
                return False
            self.processing_queues[relay_id] = queue.Queue()
            try:
                ser = serial.Serial(port, baudrate, timeout=1)
                self.relays[relay_id] = {'port': port, 'serial': ser, 'buffer': bytearray()}
                thread = threading.Thread(target=self._device_reader_thread, args=(relay_id,))
                thread.daemon = True
                self.threads[relay_id] = thread
                if self.running:
                    thread.start()
                print(f"Added device {relay_id} on {port}")
                return True
            except serial.SerialException as e:
                print(f"Failed to open {port}: {e}")
                return False

    def _device_reader_thread(self, relay_id: int):
        """The core loop for a single device thread."""
        device = self.relays[relay_id]
        ser = device['serial']
        buffer = device['buffer']
        while self.running and relay_id in self.relays:
            try:
                if ser.in_waiting > 0:
                    buffer.extend(ser.read(ser.in_waiting))

                    while len(buffer) >= PACKET_SIZE:
                        header_index = buffer.find(HEADER_BYTE)
                        if header_index == -1:
                            if buffer[-1:] == HEADER_BYTE[:1]:
                                del buffer[:-1]
                            else:
                                buffer.clear()
                            break
                        if header_index > 0:
                            del buffer[:header_index]
                        if len(buffer) >= PACKET_SIZE:
                            packet_bytes = bytes(buffer[:PACKET_SIZE])
                            try:
                                packet_data = PacketData.from_bytes(packet_bytes)
                                wrapped_packet = DevicePacket(relay_id=relay_id, data=packet_data)
                                self.processing_queues[relay_id].put(wrapped_packet)
                            except ValueError as e:
                                print(f"Relay {relay_id} packet error: {e}")
                            finally:
                                del buffer[:PACKET_SIZE]
                else:
                    time.sleep(0.001)
            except Exception as e:
                print(f"Error in device {relay_id} thread: {e}")
                time.sleep(0.1)

    def start(self):
        """Starts all registered device threads."""
        self.running = True
        for thread in self.threads.values():
            if not thread.is_alive():
                thread.start()

    def stop(self):
        """Stops all device threads and closes serial ports."""
        self.running = False
        for thread in self.threads.values():
            thread.join(timeout=2.0)
        with self.lock:
            for device in self.relays.values():
                if device['serial'].is_open:
                    device['serial'].close()

    def send_haptics_command(self, hand_label: str, intensity: int, duration_ms: int) -> bool:
        device_id = 1 if str(hand_label).upper() == "LEFT" else 2
        payload = HAPTICS_COMMAND_STRUCT.pack(
            HAPTICS_COMMAND_HEADER,
            int(device_id),
            int(np.clip(intensity, 0, 255)),
            int(np.clip(duration_ms, 1, 1000)),
        )
        with self.lock:
            if not self.relays:
                return False
            first_relay = self.relays.get(1) or next(iter(self.relays.values()))
            serial_port = first_relay["serial"]
            if not serial_port.is_open:
                return False
            serial_port.write(payload)
        return True


# endregion

# region ======================= Math Helper Functions =======================

def quat_to_euler(q):
    """
    Quaternion (w, x, y, z) → Euler angles (roll, pitch, yaw) in radians.

    Convention: Intrinsic ZYX (Tait-Bryan angles)
      - Roll  = rotation about X
      - Pitch = rotation about Y
      - Yaw   = rotation about Z

    Decomposition: R = Rz(yaw) * Ry(pitch) * Rx(roll)

    Returns: (roll, pitch, yaw) in radians
    """
    w, x, y, z = q

    # Roll (X-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (Y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        # Gimbal lock: clamp to ±90°
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (Z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def quat_to_euler_deg(q):
    """Same as quat_to_euler but returns degrees."""
    r, p, y = quat_to_euler(q)
    return math.degrees(r), math.degrees(p), math.degrees(y)


# endregion

# region ======================= Quaternion Math =======================

def quaternion_multiply(q1, q2):
    """Multiplies two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def quaternion_inverse(q):
    """Calculates the inverse of a quaternion."""
    w, x, y, z = q
    conjugate = np.array([w, -x, -y, -z])
    norm_sq = w * w + x * x + y * y + z * z
    return conjugate / norm_sq


def rotate_vector_by_quaternion(v, q):
    """Rotates a 3D vector by a quaternion and maps it to the visualizer frame."""
    # Quaternion-vector multiplication: treat v as a pure quaternion (0, vx, vy, vz)
    q_vec = np.array([0.0, float(v[0]), float(v[1]), float(v[2])], dtype=np.float64)
    q_inv = quaternion_inverse(q)
    q_rotated = quaternion_multiply(quaternion_multiply(q, q_vec), q_inv)
    rotated = q_rotated[1:]

    # Map into visualizer frame using FRAME_MAP and MODEL_OFFSET (same pipeline as other modules)
    try:
        mapped = FRAME_MAP @ rotated
        mapped = MODEL_OFFSET @ mapped
    except Exception:
        # If mapping fails for any reason, return the rotated vector instead
        mapped = rotated
    return mapped


def quaternion_to_transform_matrix(q, position):
    """Converts a quaternion and position to a 4x4 transformation matrix."""
    w, x, y, z = q

    # Rotation matrix part
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    rotation_matrix = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 0],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), 0],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), 0],
        [0, 0, 0, 1]
    ])

    # Translation part
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = position

    # Combine rotation and translation
    return translation_matrix @ rotation_matrix


# endregion

# region ======================= Glove Classes =======================


@dataclass
class Glove:
    """Represents the state of a single glove, acting as the master reference."""
    device_id: int
    active_area_x_subsections: int
    active_area_y_subsections: int
    button_pressed: bool = False
    glove_state: int = 0
    in_active_area: bool = False
    """
    0: device off, pre-calibration
    1: drawing box
    2: box drawn, fully initialized
    """
    visual: 'GloveVisual | None' = None
    active_area_half_extents: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    plane_normal_axis: int = 2  # Default to XY plane (Z-axis normal)

    # Position comes from camera triangulation; orientation/acceleration come from IMU packets.
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    acceleration: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    world_acceleration: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    rotation_quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    rotation_euler: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    # Position translated into sections
    x_section: int = 0
    y_section: int = 0

    raw_camera_position: np.ndarray | None = None
    camera_position_timestamp: float = 0.0
    reference_camera_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    has_camera_reference: bool = False
    reference_orientation_quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    inverse_reference_orientation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    last_packet_receive_time: float = 0.0

    # History for gesture detection
    position_history: deque = field(default_factory=lambda: deque(maxlen=10))
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=10))
    quat_history: deque = field(default_factory=lambda: deque(maxlen=10))
    euler_history: deque = field(default_factory=lambda: deque(maxlen=10))

    def has_fresh_camera_position(self, now: float | None = None) -> bool:
        if self.raw_camera_position is None:
            return False
        current_time = time.time() if now is None else float(now)
        return (current_time - self.camera_position_timestamp) <= CAMERA_POSITION_STALE_SECONDS

    def calibrate_zero_frame(self) -> bool:
        """Sets the current camera-triangulated position as the local origin."""
        if not self.has_fresh_camera_position() or not is_reasonable_position(self.raw_camera_position, max_norm=50.0):
            print(f"Glove {self.device_id}: camera zero-frame calibration failed. Missing camera position.")
            return False

        self.reference_camera_position = np.asarray(self.raw_camera_position, dtype=np.float64).copy()
        self.reference_orientation_quaternion = self.rotation_quaternion.copy()
        self.inverse_reference_orientation = quaternion_inverse(self.reference_orientation_quaternion)
        self.has_camera_reference = True
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        print(f"Glove {self.device_id} camera origin calibrated at {self.reference_camera_position}")
        return True

    def update_from_camera_position(self, position: np.ndarray | tuple[float, float, float],
                                    timestamp: float | None = None):
        """Updates glove position from camera triangulation."""
        new_raw_position = np.asarray(position, dtype=np.float64).reshape(3)
        if not is_reasonable_position(new_raw_position, max_norm=50.0):
            if DEBUG_LOGGING:
                print(f"Ignoring invalid camera position for device {self.device_id}: {new_raw_position}")
            return

        now = time.time() if timestamp is None else float(timestamp)
        previous_position = self.position.copy()
        previous_timestamp = self.camera_position_timestamp

        self.raw_camera_position = new_raw_position.copy()
        self.camera_position_timestamp = now
        self.position_history.append(previous_position)
        self.velocity_history.append(self.velocity.copy())

        if self.has_camera_reference:
            self.position = new_raw_position - self.reference_camera_position
        else:
            self.position = new_raw_position

        dt = now - previous_timestamp if previous_timestamp > 0.0 else 0.0
        if dt > 1e-6:
            self.velocity = (self.position - previous_position) / dt
        else:
            self.velocity = np.array([0.0, 0.0, 0.0])

    def update_from_packet(self, p: DevicePacket):
        """Updates IMU-derived state from an incoming ESP32 packet."""
        self.button_pressed = p.data.button_pressed
        self.last_packet_receive_time = time.time()

        self.euler_history.append(self.rotation_euler)
        if p.data.quat_w is not None:
            raw_rotation = np.array([p.data.quat_w, p.data.quat_i, p.data.quat_j, p.data.quat_k], dtype=np.float64)
            self.quat_history.append(self.rotation_quaternion.copy())
            self.rotation_quaternion = quaternion_multiply(self.inverse_reference_orientation, raw_rotation)
            self.rotation_euler = np.array(quat_to_euler_deg(self.rotation_quaternion), dtype=np.float64)

        if p.data.accel_x is not None:
            self.acceleration = np.array([p.data.accel_x, p.data.accel_y, p.data.accel_z], dtype=np.float64)
            self.world_acceleration = rotate_vector_by_quaternion(self.acceleration, self.rotation_quaternion)

    @staticmethod
    def _get_section_index(pos_val: float, num_sections: int, area_half_length: float) -> int:
        """
        Calculates the 0-based index for a position value within a subdivided area.
        """
        if area_half_length <= 1e-6:
            return 0

        # Normalize position from -area_half_length to +area_half_length -> 0.0 to 1.0
        normalized_pos = (pos_val + area_half_length) / (2 * area_half_length)

        # Scale to the number of sections and floor to get the index
        section_index = math.floor(normalized_pos * num_sections)

        # Clamp the value to be within the valid range [0, num_sections - 1]
        return max(0, min(num_sections - 1, section_index))

    def _update_section_visuals(self):
        if not self.visual:
            return

        for line in self.visual.section_lines:
            line.parent = None
        self.visual.section_lines.clear()

        plane_axes = [i for i in range(3) if i != self.plane_normal_axis]
        u_axis, v_axis = plane_axes[0], plane_axes[1]

        if self.active_area_x_subsections > 1:
            for i in range(1, self.active_area_x_subsections):
                pos = -0.5 + i / self.active_area_x_subsections
                start, end = np.zeros(3), np.zeros(3)
                start[u_axis] = pos
                start[v_axis] = -0.5
                end[u_axis] = pos
                end[v_axis] = 0.5
                line = visuals.Line(pos=np.array([start, end]), color='cyan', parent=self.visual.box, width=4)
                self.visual.section_lines.append(line)

        if self.active_area_y_subsections > 1:
            for i in range(1, self.active_area_y_subsections):
                pos = -0.5 + i / self.active_area_y_subsections
                start, end = np.zeros(3), np.zeros(3)
                start[u_axis], start[v_axis] = -0.5, pos
                end[u_axis], end[v_axis] = 0.5, pos
                line = visuals.Line(pos=np.array([start, end]), color='cyan', parent=self.visual.box, width=4)
                self.visual.section_lines.append(line)

    def get_section_from_position(self):
        """Convert positions to sections and other instructions"""
        if self.glove_state == 2:
            if not self.has_fresh_camera_position():
                self.in_active_area = False
                return

            plane_axes = [i for i in range(3) if i != self.plane_normal_axis]
            u_axis, v_axis = plane_axes[0], plane_axes[1]

            if self.position[self.plane_normal_axis] > 0:
                self.in_active_area = True
                self.x_section = self._get_section_index(
                    self.position[u_axis], self.active_area_x_subsections, self.active_area_half_extents[u_axis]
                )
                self.y_section = self._get_section_index(
                    self.position[v_axis], self.active_area_y_subsections, self.active_area_half_extents[v_axis]
                )
            else:
                self.in_active_area = False

        elif self.glove_state == 0:
            if self.button_pressed:
                if self.calibrate_zero_frame():
                    self.glove_state = 1
                    self.position = np.array([0.0, 0.0, 0.0])
                    if self.visual:
                        self.visual.box.visible = True

        elif self.glove_state == 1:
            if self.visual:
                if self.button_pressed:
                    abs_pos = np.abs(self.position)
                    self.plane_normal_axis = int(np.argmin(abs_pos))

                    plane_axes = [i for i in range(3) if i != self.plane_normal_axis]
                    half_u = max(float(abs_pos[plane_axes[0]]), 0.005)
                    half_v = max(float(abs_pos[plane_axes[1]]), 0.005)

                    dimensions = np.full(3, 0.005, dtype=np.float64)
                    dimensions[plane_axes[0]] = half_u * 2.0
                    dimensions[plane_axes[1]] = half_v * 2.0
                    dimensions[self.plane_normal_axis] = 0.005

                    transform = np.eye(4)
                    transform[0, 0] = dimensions[0] if dimensions[0] > 1e-6 else 1e-6
                    transform[1, 1] = dimensions[1] if dimensions[1] > 1e-6 else 1e-6
                    transform[2, 2] = dimensions[2] if dimensions[2] > 1e-6 else 1e-6
                    transform[:3, 3] = [0.0, 0.0, 0.0]

                    self.visual.box.transform.matrix = transform
                else:
                    self.glove_state = 2
                    abs_pos = np.abs(self.position)
                    plane_axes = [i for i in range(3) if i != self.plane_normal_axis]
                    self.active_area_half_extents = np.zeros(3, dtype=np.float64)
                    self.active_area_half_extents[plane_axes[0]] = max(float(abs_pos[plane_axes[0]]), 0.005)
                    self.active_area_half_extents[plane_axes[1]] = max(float(abs_pos[plane_axes[1]]), 0.005)
                    self.active_area_half_extents[self.plane_normal_axis] = 0.0

                    self._update_section_visuals()


@dataclass
class LeftHand(Glove):
    """Manages the parameters controlled by the left hand"""
    volume: int = 0
    reverb_amount: int = 0
    attack: int = 0


@dataclass
class RightHand(Glove):
    """Manages the parameters controlled by the right hand"""
    note: int = 0
    instrument: str = "Synth"


@dataclass
class GlovePair:
    """Manages a pair of gloves and their processing thread."""
    left_hand: LeftHand
    right_hand: RightHand
    relay_id: int
    relay_queue: queue.Queue
    reader: ThreadedMultiDeviceReader
    daw_interface: 'DawInterface'
    current_octave: int = 4
    right_roll_samples: deque = field(default_factory=lambda: deque(maxlen=10))
    last_instrument_gesture_ms: int = field(default=-10_000)
    last_preview_note: int | None = field(default=None)
    running: bool = field(init=False, default=False)
    processing_thread: threading.Thread = field(init=False)

    def start(self):
        """Starts the packet processing thread for this glove pair."""
        self.processing_thread = threading.Thread(target=self._packet_processor)
        self.processing_thread.daemon = True
        self.running = True
        self.processing_thread.start()

    def _process_single_packet(self, packet: DevicePacket, from_queue: bool = False):
        glove = self._hand_for_device_id(packet.data.device_number)
        glove.update_from_packet(packet)
        glove.get_section_from_position()

        if from_queue:
            self.relay_queue.task_done()

        self._update_playing_state(glove, packet)

    def _hand_for_device_id(self, device_id: int) -> Glove:
        return self.left_hand if device_id % 2 != 0 else self.right_hand

    def update_camera_position(self, device_id: int, position: np.ndarray | tuple[float, float, float],
                               timestamp: float | None = None):
        """Feeds a camera-triangulated position into the matching hand."""
        glove = self._hand_for_device_id(device_id)
        glove.update_from_camera_position(position, timestamp)
        glove.get_section_from_position()
        self._update_playing_state(glove)

    def _update_playing_state(self, updated_glove: Glove, packet: DevicePacket | None = None):
        self._update_note_preview_haptics()
        if self.left_hand.glove_state == 2 and self.right_hand.glove_state == 2:
            if packet is not None:
                self._update_right_roll_gesture(updated_glove, packet)

            if self.left_hand.in_active_area and self.right_hand.in_active_area:
                self.section_to_note()
                self.play_note()
            else:
                self.daw_interface.stop_notes()
                self.daw_interface.previous_note = NoteData.blank_note()

    def _update_note_preview_haptics(self) -> None:
        preview_note = None
        if self.right_hand.glove_state == 2 and self.right_hand.in_active_area:
            preview_note = int(self.right_hand.y_section + (self.current_octave * 12))

        if (
            ENABLE_NOTE_HAPTICS
            and preview_note is not None
            and preview_note != self.last_preview_note
        ):
            self.daw_interface.haptics.pulse("RIGHT", HAPTICS_NOTE_INTENSITY, HAPTICS_NOTE_DURATION_MS)

        self.last_preview_note = preview_note

    @staticmethod
    def _roll_delta_deg(current_roll: float, previous_roll: float) -> float:
        delta = float(current_roll - previous_roll)
        while delta > 180.0:
            delta -= 360.0
        while delta < -180.0:
            delta += 360.0
        return delta

    def _cycle_instrument(self, direction: int):
        if direction == 0:
            return
        current = self.right_hand.instrument if self.right_hand.instrument else INSTRUMENT_CYCLE[0]
        if current not in INSTRUMENT_CYCLE:
            current = INSTRUMENT_CYCLE[0]
        current_idx = INSTRUMENT_CYCLE.index(current)
        next_idx = (current_idx + direction) % len(INSTRUMENT_CYCLE)
        self.right_hand.instrument = INSTRUMENT_CYCLE[next_idx]
        print(
            f"[GESTURE] Right-hand rapid roll {'RIGHT' if direction > 0 else 'LEFT'} -> "
            f"instrument set to {self.right_hand.instrument}"
        )

    def _update_right_roll_gesture(self, updated_glove: Glove, packet: DevicePacket):
        if updated_glove is not self.right_hand:
            return
        if self.right_hand.glove_state != 2:
            self.right_roll_samples.clear()
            return

        timestamp_ms = int(packet.data.timestamp)
        roll_deg = float(self.right_hand.rotation_euler[0])
        if self.right_roll_samples and timestamp_ms <= self.right_roll_samples[-1][0]:
            return
        self.right_roll_samples.append((timestamp_ms, roll_deg))
        if len(self.right_roll_samples) < 2:
            return

        current_t, current_roll = self.right_roll_samples[-1]
        window_start_t = current_t - GESTURE_WINDOW_MS
        oldest_t, oldest_roll = self.right_roll_samples[-1]
        for sample_t, sample_roll in self.right_roll_samples:
            if sample_t >= window_start_t:
                oldest_t, oldest_roll = sample_t, sample_roll
                break

        dt_ms = current_t - oldest_t
        if dt_ms <= 0:
            return

        delta_roll = self._roll_delta_deg(current_roll, oldest_roll)
        roll_rate_dps = delta_roll / (dt_ms / 1000.0)
        accel_norm = float(np.linalg.norm(self.right_hand.acceleration))
        if (
                abs(current_roll) >= GESTURE_ROLL_ANGLE_THRESHOLD_DEG
                and accel_norm >= GESTURE_ACCEL_SPIKE_THRESHOLD_MPS2
                and abs(delta_roll) >= GESTURE_ROLL_DELTA_THRESHOLD_DEG
                and abs(roll_rate_dps) >= GESTURE_ROLL_RATE_THRESHOLD_DPS
                and (current_t - self.last_instrument_gesture_ms) >= GESTURE_COOLDOWN_MS
        ):
            direction = 1 if roll_rate_dps > 0.0 else -1
            self._cycle_instrument(direction)
            self.last_instrument_gesture_ms = current_t

    def _packet_processor(self, packet: DevicePacket | None = None):
        """Process one packet (when provided) or run the threaded queue loop."""
        if packet is not None:
            try:
                self._process_single_packet(packet, from_queue=False)
            except Exception as e:
                print(f"Error in processor for relay {self.relay_id}: {e}")
            return

        while self.running:
            try:
                packet = self.relay_queue.get(timeout=0.1)
                self._process_single_packet(packet, from_queue=True)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processor for relay {self.relay_id}: {e}")

    def stop(self):
        """Stops the processing thread."""
        self.running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)

    def play_note(self):
        note_packet = NoteData(
            note=self.right_hand.note,
            volume=self.left_hand.volume,
            reverb_mode=self.left_hand.reverb_amount,
            stereo=0.0,
            attack=self.left_hand.attack,
            instrument=self.right_hand.instrument
        )
        if MIDI_DEBUG_LOGGING:
            print(
                "[MIDI DEBUG] GlovePair.play_note NoteData constructed: "
                f"note={note_packet.note}, attack={note_packet.attack}, instrument={note_packet.instrument}, "
                f"volume={note_packet.volume}, stereo={note_packet.stereo}, reverb_mode={note_packet.reverb_mode}, "
                f"left_state={self.left_hand.glove_state}, left_sections=({self.left_hand.x_section},{self.left_hand.y_section}), "
                f"right_sections=({self.right_hand.x_section},{self.right_hand.y_section})"
            )
        self.daw_interface.play_note(note_packet)

    def section_to_note(self):
        """Get notes from the sectional data"""

        for hand in [self.left_hand, self.right_hand]:
            if hand is self.left_hand:
                if hand.button_pressed:
                    self.current_octave -= 1 if self.current_octave > 2 else 0

                volume_roll = float(np.clip(hand.rotation_euler[0], 0.0, 180.0))
                hand.volume = int(np.clip(round((volume_roll / 180.0) * 127.0), 0, 127))

                plane_axes = [i for i in range(3) if i != hand.plane_normal_axis]
                x_axis, y_axis = plane_axes[0], plane_axes[1]
                x_extent = float(hand.active_area_half_extents[x_axis]) if float(hand.active_area_half_extents[x_axis]) > 1e-6 else 1.0
                y_extent = float(hand.active_area_half_extents[y_axis]) if float(hand.active_area_half_extents[y_axis]) > 1e-6 else 1.0
                x_normalized = float(np.clip((hand.position[x_axis] + x_extent) / (2.0 * x_extent), 0.0, 1.0))
                y_normalized = float(np.clip((hand.position[y_axis] + y_extent) / (2.0 * y_extent), 0.0, 1.0))
                hand.reverb_amount = int(np.clip(round(x_normalized * 127.0), 0, 127))
                hand.attack = int(np.clip(round(y_normalized * 127.0), 0, 127))

            if hand is self.right_hand:
                if hand.button_pressed:
                    self.current_octave += 1 if self.current_octave < 7 else 0

                hand.note = hand.y_section + (self.current_octave * 12)


# endregion
# region ======================= Notes and MIDI  =======================

@dataclass
class NoteData:
    # Note data
    note: int | None  # 0-127
    attack: int | None  # 0-127
    instrument: str | None

    # Control data
    volume: int | None  # 0-127
    stereo: float | None  # unused for now
    reverb_mode: int | None  # continuous 0-127

    @classmethod
    def blank_note(cls):
        return cls(
            note=0,
            attack=0,
            instrument="",
            volume=0,
            stereo=0,
            reverb_mode=0,
        )


@dataclass
class DawInterface:
    port: mido.backends.rtmidi.Output
    haptics: HapticsController = field(default_factory=lambda: HapticsController(HAPTICS_PORT, HAPTICS_BAUD, ENABLE_HAPTICS))
    previous_note: NoteData = field(init=False)


    def __post_init__(self):
        self.previous_note = NoteData.blank_note()
        if MIDI_DEBUG_LOGGING:
            print(f"[MIDI DEBUG] DawInterface init: port={self.port}")
        if self.port is None or "":
            out_ports = mido.get_output_names()
            if not out_ports:
                print("No MIDI output devices found. Open a DAW or connect a device.")
                if MIDI_DEBUG_LOGGING:
                    print("[MIDI DEBUG] No output ports found at init.")
                return
            else:
                for i, name in enumerate(out_ports):
                    print(f"{i} | {name}")
                chosen_port = input("Please type the number of the output port to select the device \n>>> ")
                chosen_port = int(chosen_port.strip())
                if chosen_port + 1 > len(out_ports) or chosen_port < 0:
                    print("Invalid port number, please try again.")
                    time.sleep(1)
                    chosen_port = int(input("Please type the number of the output port to select the device\n>>> "))
                    self.port = mido.open_output(out_ports[chosen_port])
                else:
                    self.port = mido.open_output(out_ports[chosen_port])
                    if MIDI_DEBUG_LOGGING:
                        print(f"[MIDI DEBUG] Opened output port: {out_ports[chosen_port]}")

    @staticmethod
    def _get_inst_channel(inst: str):
        match inst:
            case "Synth":
                inst_channel = 0
            case "Violin":
                inst_channel = 1
            case "Horn":
                inst_channel = 2
            case "Trumpet":
                inst_channel = 2
            case "Clarinet":
                inst_channel = 3
            case "Drums":
                inst_channel = 11
            case "Instrument4":
                inst_channel = 4
            case "Instrument5":
                inst_channel = 5
            case "Instrument6":
                inst_channel = 6
            case "Instrument7":
                inst_channel = 7
            case "Instrument8":
                inst_channel = 8
            case "Instrument9":
                inst_channel = 9
            case _:
                inst_channel = 0
        return inst_channel

    def stop_notes(self):
        if self.port is None:
            return
        self.port.send(mido.Message(
            'note_off',
            note=self.previous_note.note,
            velocity=0,
            channel=self._get_inst_channel(self.previous_note.instrument)
        ))

    def prepare_midi_messages(self, note: NoteData) -> list[mido.Message]:
        inst_channel: int
        message_list = []
        note_value = int(np.clip(note.note, 0, 127))
        note_attack = int(np.clip(note.attack, 0, 127))
        note_volume = int(np.clip(note.volume, 0, 127))
        # pan CC expects 0 – 127, where 64 is center
        note_pan = int(np.clip(round((float(note.stereo) + 0.5) * 127.0), 0, 127))
        if MIDI_DEBUG_LOGGING:
            print(
                f"[MIDI DEBUG] prepare_midi_messages input: note={note} "
                f"prev={self.previous_note} normalized(note={note_value}, attack={note_attack}, "
                f"volume={note_volume}, pan={note_pan})"
            )

        if note == self.previous_note:
            if MIDI_DEBUG_LOGGING:
                print("[MIDI DEBUG] No MIDI messages: note equals previous_note.")
            return message_list

        def add_control_msgs(msg_list: list[mido.Message]):
            if note.reverb_mode != self.previous_note.reverb_mode:
                reverb_msg = mido.Message(
                    'control_change',
                    channel=11,
                    control=100,
                    value=int(np.clip(note.reverb_mode, 0, 127))
                )
                msg_list.append(reverb_msg)
            if note.volume != self.previous_note.volume:
                volume_msg = mido.Message(
                    'control_change',
                    control=101,
                    value=note_volume
                )
                msg_list.append(volume_msg)
            if note.stereo != self.previous_note.stereo:
                stereo_msg = mido.Message(
                    'control_change',
                    control=102,
                    value=64
                )
                msg_list.append(stereo_msg)

        if note.note == self.previous_note.note and note.instrument == self.previous_note.instrument:
            add_control_msgs(message_list)
        else:
            off_msg = mido.Message(
                'note_off',
                note=self.previous_note.note,
                velocity=0,
                channel=self._get_inst_channel(self.previous_note.instrument)
            )
            message_list.append(off_msg)
            new_note = mido.Message(
                'note_on',
                note=note_value,
                velocity=note_attack,
                channel=self._get_inst_channel(note.instrument)
            )
            message_list.append(new_note)
            add_control_msgs(message_list)

        if MIDI_DEBUG_LOGGING:
            if message_list:
                for i, msg in enumerate(message_list, start=1):
                    print(f"[MIDI DEBUG] msg[{i}/{len(message_list)}]: {msg}")
            else:
                print("[MIDI DEBUG] No MIDI messages generated after diff checks.")

        return message_list

    def play_note(self, note: NoteData):
        if self.port is None:
            if MIDI_DEBUG_LOGGING:
                print(f"[MIDI DEBUG] Skipping play_note because port is None. note={note}")
            self.previous_note = note
            return
        message_list = self.prepare_midi_messages(note)
        if MIDI_DEBUG_LOGGING:
            print(f"[MIDI DEBUG] Sending {len(message_list)} MIDI message(s).")
        for msg in message_list:
            self.port.send(msg)
        self.previous_note = note
        if MIDI_DEBUG_LOGGING:
            print(f"[MIDI DEBUG] Updated previous_note to: {self.previous_note}")


# endregion

# region ======================= Visualizer Classes =======================


# noinspection PyTypeHints
@dataclass
class GloveVisual:
    """Holds the vispy visuals for a single glove."""
    axis: visuals.XYZAxis
    parent_scene: scene.SceneNode
    box: visuals.Box = field(init=False)
    section_lines: list[visuals.Line] = field(default_factory=list)

    def __post_init__(self):
        self.box = visuals.Box(width=1, height=1, depth=1, color=(1, 0.5, 0.5, 0.14), edge_color=(1, 0.5, 0.5, 1))
        self.box.transform = scene.transforms.MatrixTransform()
        self.box.parent = self.parent_scene
        self.box.visible = False


class Visualizer:
    def __init__(self, glove_pairs, camera_feeder: 'CameraPositionFeeder | None' = None):
        self.glove_pairs = glove_pairs
        self.camera_feeder = camera_feeder
        # Create canvas sized to 75% of the screen (if available) for a larger visualization area
        try:
            import tkinter as _tk
            root = _tk.Tk()
            sw = root.winfo_screenwidth()
            sh = root.winfo_screenheight()
            root.destroy()
            target_size = (int(sw * 0.75), int(sh * 0.75))
        except Exception:
            # Fallback to a reasonable default if tkinter or screen info isn't available
            target_size = (int(0.75 * 1920), int(0.75 * 1280))

        self.canvas = scene.SceneCanvas(keys='interactive', show=True, size=target_size)
        self.view = self.canvas.central_widget.add_view(border_color='white')
        self.view.camera = scene.cameras.TurntableCamera()
        visuals.GridLines(scale=(0.25, 0.25), color=(0.25, 0.25, 0.25, 1.0), parent=self.view.scene)
        visuals.XYZAxis(parent=self.view.scene)
        self.hand_meshes = {}

        for pair in self.glove_pairs:
            pair.left_hand.visual = GloveVisual(
                axis=visuals.XYZAxis(parent=self.view.scene),
                parent_scene=self.view.scene,
            )
            pair.right_hand.visual = GloveVisual(
                axis=visuals.XYZAxis(parent=self.view.scene),
                parent_scene=self.view.scene,
            )

        self._setup_hand_meshes()

        self.timer = Timer('auto', connect=self.update, start=True)
        self.canvas.events.close.connect(self.on_close)

    def _setup_hand_meshes(self):
        if not os.path.exists(HAND_OBJ_PATH):
            print(f"Could not find model: {HAND_OBJ_PATH}")
            return

        vertices, faces, _normals, _texcoords = read_mesh(HAND_OBJ_PATH)
        self.hand_meshes["LEFT"] = scene.visuals.Mesh(
            vertices=vertices,
            faces=faces,
            color=(0.25, 0.75, 0.95, 0.80),
            shading=None,
            parent=self.view.scene,
        )
        self.hand_meshes["LEFT"].transform = MatrixTransform()
        self.hand_meshes["RIGHT"] = scene.visuals.Mesh(
            vertices=vertices,
            faces=faces,
            color=(0.95, 0.45, 0.25, 0.80),
            shading=None,
            parent=self.view.scene,
        )
        self.hand_meshes["RIGHT"].transform = MatrixTransform()

    def _mesh_transform_matrix(self, hand) -> np.ndarray:
        q_current = np.asarray(hand.rotation_quaternion, dtype=np.float64)
        rotation_matrix = quaternion_to_transform_matrix(q_current, np.array([0.0, 0.0, 0.0]))[:3, :3]
        rotation_matrix = FRAME_MAP @ rotation_matrix @ FRAME_MAP.T
        rotation_matrix = rotation_matrix @ MODEL_OFFSET

        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rotation_matrix * MODEL_SCALE
        transform[3, :3] = np.asarray(hand.position, dtype=np.float32) * POSITION_SCALE
        return transform

    def on_close(self, event):
        self.timer.stop()
        if self.camera_feeder is not None:
            self.camera_feeder.stop()
        for pair in self.glove_pairs:
            pair.stop()
        if self.glove_pairs:
            self.glove_pairs[0].reader.stop()
        print("Visualization closed, system stopping.")
        app.quit()

    def update(self, event):
        if self.camera_feeder is not None:
            self.camera_feeder.update()
        for pair in self.glove_pairs:
            left_hand = pair.left_hand
            right_hand = pair.right_hand

            if left_hand.visual:
                transform_matrix = quaternion_to_transform_matrix(left_hand.rotation_quaternion, left_hand.position)

                if "LEFT" in self.hand_meshes:
                    self.hand_meshes["LEFT"].transform.matrix = self._mesh_transform_matrix(left_hand)
                else:
                    left_hand.visual.axis.transform = scene.transforms.MatrixTransform(transform_matrix)

            if right_hand.visual:
                transform_matrix = quaternion_to_transform_matrix(right_hand.rotation_quaternion, right_hand.position)
                if "RIGHT" in self.hand_meshes:
                    self.hand_meshes["RIGHT"].transform.matrix = self._mesh_transform_matrix(right_hand)
                else:
                    right_hand.visual.axis.transform = scene.transforms.MatrixTransform(transform_matrix)

        self.canvas.update()


# endregion

# region ======================= Main Application =======================

def _load_camera_modules():
    root_dir = Path(__file__).resolve().parent
    camera_tests_dir = root_dir / "camera_tests"
    four_camera_dir = camera_tests_dir / "4_camera_tests"
    for path in (camera_tests_dir, four_camera_dir, root_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import mocap_tracker as mocap
    import four_camera_shared as four_shared
    return mocap, four_shared


class CameraPositionFeeder:
    def __init__(self, glove_pair: GlovePair, mode: int = CAMERA_MODE):
        if mode not in (2, 4):
            raise ValueError("CAMERA_MODE must be 2 or 4")
        self.glove_pair = glove_pair
        self.mode = int(mode)
        self.mocap, self.four_shared = _load_camera_modules()
        self.args = self._build_args()
        self.stop_event = None
        self.workers = []
        self.calibrations = {}
        self.tracker = None
        self.triangulator = None
        self.display_track_ids: list[int] = []
        self.last_processed_frame_numbers: dict[int, int] = {}

    def _build_args(self):
        parser = self.mocap.build_arg_parser()
        args = parser.parse_args([])
        if self.mode == 4:
            args.cameras = list(self.four_shared.CAMERA_IDS)
            args.calibration = str(self.four_shared.default_calibration_path())
            args.front_cameras = tuple(self.four_shared.FRONT_CAMERA_IDS)
            args.top_cameras = tuple(self.four_shared.TOP_CAMERA_IDS)
            args.fusion_y_tolerance = self.four_shared.DEFAULT_FUSION_Y_TOLERANCE_M
            args.max_fused_reprojection_error = self.four_shared.DEFAULT_MAX_FUSED_REPROJECTION_ERROR_PX
            args.max_layout_measurements = self.four_shared.DEFAULT_TRACKED_POINT_COUNT
            args.min_measurement_separation = self.four_shared.DEFAULT_MIN_MEASUREMENT_SEPARATION_M
            args.pairing_track_bias_distance = self.four_shared.DEFAULT_PAIRING_TRACK_BIAS_DISTANCE_M
        else:
            args.cameras = list(TWO_CAMERA_IDS)
            calibration_path = TWO_CAMERA_ALIGNED_CALIBRATION_JSON
            if not calibration_path.exists():
                calibration_path = TWO_CAMERA_CALIBRATION_JSON
            args.calibration = str(calibration_path)
        args.no_preview = True
        args.tracked_point_count = 2
        return args

    def start(self) -> bool:
        if self.mocap.cv2 is None:
            print("OpenCV is required for camera mocap. Install it with: python -m pip install opencv-python")
            return False

        self.calibrations = self.four_shared.load_calibrations(self.args, f"{self.mode}cam jazzhands")
        self.stop_event, self.workers = self.four_shared.start_threaded_cameras(
            self.args,
            list(self.args.cameras),
            build_masks=False,
            label=f"{self.mode}cam jazzhands",
        )
        self.four_shared.wait_for_open_attempts(self.workers)
        snapshots = self.four_shared.collect_snapshots(self.workers)
        if not self.four_shared.any_camera_open(snapshots) and self.four_shared.all_open_attempts_done(snapshots):
            print(f"[{self.mode}cam jazzhands] no cameras opened")
            self.stop()
            return False

        self.tracker = self.mocap.MarkerTracker(
            max_match_distance_m=self.args.track_distance,
            max_missing_frames=self.args.max_missing_frames,
            min_confirmed_hits=self.args.track_confirmation_hits,
            max_tentative_missing_frames=self.args.tentative_max_missing_frames,
            duplicate_track_distance_m=self.args.duplicate_track_distance,
            velocity_damping=self.args.velocity_damping,
            stationary_distance_m=self.args.stationary_distance,
            max_prediction_dt=self.args.max_prediction_dt,
        )
        if self.mode == 2:
            self.triangulator = self.mocap.MultiCameraTriangulator(
                self.calibrations,
                self.args.max_reprojection_error,
                self.args.cluster_distance,
                self.args.room_bounds,
            )
        print(f"[{self.mode}cam jazzhands] camera position feeder ready")
        return True

    def stop(self):
        if self.stop_event is not None and self.workers:
            self.four_shared.stop_threaded_cameras(self.stop_event, self.workers)
        self.stop_event = None
        self.workers = []

    def update(self):
        if self.tracker is None:
            return
        timestamp = time.time()
        snapshots = self.four_shared.collect_snapshots(self.workers)
        if not self._has_new_camera_data(snapshots):
            return
        self._mark_camera_data_processed(snapshots)

        observations_by_camera = self.four_shared.observations_from_snapshots(snapshots)
        measurements = self._measurements(observations_by_camera, timestamp)
        tracks = self.tracker.update(measurements, timestamp)
        live_tracks = [track for track in tracks if track.confirmed and track.missing_frames == 0]
        selected_tracks = self._select_tracks(live_tracks)
        self._feed_tracks(selected_tracks, timestamp)

    def _measurements(self, observations_by_camera, timestamp: float):
        if self.mode == 4:
            measurements, _diagnostics = self.four_shared.fuse_layout_measurements(
                observations_by_camera,
                self.calibrations,
                tuple(self.args.front_cameras),
                tuple(self.args.top_cameras),
                self.args.room_bounds,
                self.args.max_reprojection_error,
                self.args.fusion_y_tolerance,
                self.args.max_fused_reprojection_error,
                self.args.max_layout_measurements,
                self.args.min_measurement_separation,
                self._pairing_reference_positions(timestamp),
                self.args.pairing_track_bias_distance,
            )
            return measurements
        return self.triangulator.triangulate(observations_by_camera)

    def _pairing_reference_positions(self, timestamp: float) -> list[np.ndarray]:
        tracks_by_id = {
            track.track_id: track
            for track in self.tracker.tracks
            if track.confirmed and track.missing_frames <= self.args.max_missing_frames
        }
        ordered_tracks = [
            tracks_by_id[track_id]
            for track_id in self.display_track_ids
            if track_id in tracks_by_id
        ]
        ordered_ids = {track.track_id for track in ordered_tracks}
        ordered_tracks.extend(
            track for track in sorted(tracks_by_id.values(), key=lambda item: item.track_id)
            if track.track_id not in ordered_ids
        )
        return [self.tracker._predicted_position(track, timestamp) for track in ordered_tracks[:2]]

    def _select_tracks(self, live_tracks) -> list:
        live_by_id = {track.track_id: track for track in live_tracks}
        selected_ids = [track_id for track_id in self.display_track_ids if track_id in live_by_id][:2]
        for track in sorted(live_tracks, key=lambda item: item.track_id):
            if len(selected_ids) >= 2:
                break
            if track.track_id not in selected_ids:
                selected_ids.append(track.track_id)
        self.display_track_ids = selected_ids
        return [live_by_id[track_id] for track_id in selected_ids]

    def _feed_tracks(self, tracks, timestamp: float):
        if not tracks:
            return
        axis = int(CAMERA_HAND_SIDE_AXIS)
        ordered = sorted(tracks, key=lambda track: float(track.position[axis]))
        assignments = []
        if len(ordered) == 1:
            device_id = 1 if float(ordered[0].position[axis]) < 0.0 else 2
            assignments.append((device_id, ordered[0]))
        else:
            assignments.append((1, ordered[0]))
            assignments.append((2, ordered[-1]))

        for device_id, track in assignments:
            self.glove_pair.update_camera_position(
                device_id,
                np.asarray(track.position, dtype=np.float64) * CAMERA_POSITION_SCALE,
                timestamp,
            )

    def _has_new_camera_data(self, snapshots) -> bool:
        for camera_id, snapshot in snapshots.items():
            if snapshot.frame_number <= 0:
                continue
            if self.last_processed_frame_numbers.get(camera_id) != snapshot.frame_number:
                return True
        return False

    def _mark_camera_data_processed(self, snapshots) -> None:
        for camera_id, snapshot in snapshots.items():
            self.last_processed_frame_numbers[camera_id] = snapshot.frame_number

def main():
    """Initializes and runs the main application."""
    maybe_run_two_camera_calibration()

    # Create MIDI Object
    daw_interface = DawInterface(port=None)

    # Create reader object
    reader = ThreadedMultiDeviceReader()
    serial_ports = discover_serial_ports()
    if not serial_ports:
        print("No serial receiver ports were found. Connect the ESP32 receiver and try again.")
    for i, port in enumerate(serial_ports, start=1):
        if not reader.add_device(relay_id=i, port=port):
            print(f"Could not connect to device on {port}. Please check connection.")

    reader.start()
    daw_interface.haptics = HapticsController(
        enabled=ENABLE_HAPTICS,
        send_func=reader.send_haptics_command,
    )

    # Create glove pairs. Positions should be supplied from camera triangulation
    # through GlovePair.update_camera_position(...).
    glove_pairs = []
    if 1 in reader.processing_queues:
        left_hand = LeftHand(
            device_id=1,
            active_area_x_subsections=3,
            active_area_y_subsections=12
        )
        right_hand = RightHand(
            device_id=2,
            active_area_x_subsections=5,
            active_area_y_subsections=8
        )

        glove_pairs.append(
            GlovePair(
                left_hand=left_hand,
                right_hand=right_hand,
                relay_id=1,
                reader=reader,
                relay_queue=reader.processing_queues[1],
                daw_interface=daw_interface
            )
        )

    for pair in glove_pairs:
        pair.start()

    camera_feeder = None
    if glove_pairs:
        try:
            camera_feeder = CameraPositionFeeder(glove_pairs[0], CAMERA_MODE)
        except ImportError as error:
            print(f"Camera integration disabled because a camera dependency is missing: {error}")
        if camera_feeder is not None and not camera_feeder.start():
            camera_feeder = None

    print("\n" + "=" * 60)
    print("SYSTEM READY")
    print("=" * 60)
    print("System running in 2-camera mode. Close the visualization window to stop.\n")

    visualizer = Visualizer(glove_pairs, camera_feeder=camera_feeder)
    app.run()

    print("System stopped.")


if __name__ == "__main__":
    main()

# endregion
