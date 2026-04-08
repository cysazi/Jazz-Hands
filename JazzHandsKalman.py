# region ======================= Imports =======================
import math
import numpy as np

import serial
import struct
import queue
import time
import threading

from dataclasses import dataclass, field
from collections import deque

from vispy import app, scene
from vispy.scene import visuals
from vispy.app import Timer
from vispy.io import mesh

import mido

# endregion

# region ======================= Constants and Definitions =======================

 # --- UWB Configuration ---
NUM_UWB_ANCHORS: int = 4  # Set to 3, 4, or 5
# Set to None for auto-calibration, or provide positions as [(x,y,z), ...] to skip calibration
# Positions should be in meters. Example: [(0,0,0), (5,0,0), (2.5,4.33,0), (2.5,2.16,3)]
MANUAL_ANCHOR_POSITIONS: list[tuple[float, float, float]] | None = [(0.0,0.0,0.0), (1.66,0.0,0.0), (0.0,0.915, 0.74), (1.36, 0.255, 2.15)]
# Alternative: specify individual distances and let Python calculate positions
# Format: {(anchor1, anchor2): distance, ...} - distances in meters
MANUAL_ANCHOR_DISTANCES: dict[tuple[int, int], float] | None = None
# Example: {(1,2): 5.0, (1,3): 5.0, (2,3): 5.0, (1,4): 6.0, (2,4): 6.0, (3,4): 6.0}

# --- Serial and Packet Definitions ---
COM_PORTS: list[str] = ["/dev/cu.usbserial-023BB305"]
PACKET_SIZE: int = 70  # Updated for 5 UWB distances (4 bytes per extra anchor)
CALIBRATION_PACKET_SIZE: int = 8  # header(2) + source(1) + dest(1) + distance(4)
HEADER_BYTE: bytes = b'\xAA\xAA'
CALIBRATION_HEADER_BYTE: bytes = b'\xBB\xBB'
# Updated format: pos(3f), vel(3f), uwb(5f), quat(4f)
PACKET_FORMAT: str = '<HBIBB 3f 3f 5f 4f B'
CALIBRATION_FORMAT: str = '<H B B f'

# --- Bit Flagging Constants ---
PACKET_HAS_UWB_1: int = 0b00000100
PACKET_HAS_UWB_2: int = 0b00001000
PACKET_HAS_UWB_3: int = 0b00010000
PACKET_HAS_UWB_4: int = 0b00100000
PACKET_HAS_UWB_5: int = 0b01000000
PACKET_HAS_ERROR: int = 0b10000000

# --- System and Physics Constants ---
BOUNDING_BOX_NORMAL_TOLERANCE: float = 0.15  # meters, ±tolerance in normal direction
MIDI_DEBUG_LOGGING: bool = True
PERFORMANCE_MODE: bool = False  # for mirroring the L and R hands
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
GESTURE_ROLL_RATE_THRESHOLD_DPS: float = 220.0
GESTURE_ROLL_DELTA_THRESHOLD_DEG: float = 28.0
GESTURE_WINDOW_MS: int = 220
GESTURE_COOLDOWN_MS: int = 300


# endregion

# region ======================= Data Classes =======================

@dataclass
class PacketData:
    """Represents the data structure of a packet from the ESP32."""
    device_number: int
    timestamp: int
    packet_flags: int
    button_state: bool
    pos_x: float
    pos_y: float
    pos_z: float
    vel_x: float
    vel_y: float
    vel_z: float
    UWB_distance_1: float | None
    UWB_distance_2: float | None
    UWB_distance_3: float | None
    UWB_distance_4: float | None
    UWB_distance_5: float | None
    quat_w: float
    quat_i: float
    quat_j: float
    quat_k: float
    error_handler: int | None

    @classmethod
    def from_bytes(cls, binary_data: bytes) -> "PacketData":
        """Parses a byte array into a PacketData object."""
        if len(binary_data) != PACKET_SIZE:
            raise ValueError(f"Expected packet size {PACKET_SIZE}, got {len(binary_data)}")

        unpacked = struct.unpack(PACKET_FORMAT, binary_data)
        temp_packet_flag = unpacked[3]

        return cls(
            device_number=unpacked[1],
            timestamp=unpacked[2],
            packet_flags=temp_packet_flag,
            button_state=bool(unpacked[4]),
            pos_x=unpacked[5],
            pos_y=unpacked[6],
            pos_z=unpacked[7],
            vel_x=unpacked[8],
            vel_y=unpacked[9],
            vel_z=unpacked[10],
            UWB_distance_1=unpacked[11] if (temp_packet_flag & PACKET_HAS_UWB_1) else None,
            UWB_distance_2=unpacked[12] if (temp_packet_flag & PACKET_HAS_UWB_2) else None,
            UWB_distance_3=unpacked[13] if (temp_packet_flag & PACKET_HAS_UWB_3) else None,
            UWB_distance_4=unpacked[14] if (temp_packet_flag & PACKET_HAS_UWB_4) else None,
            UWB_distance_5=unpacked[15] if (temp_packet_flag & PACKET_HAS_UWB_5) else None,
            quat_w=unpacked[16],
            quat_i=unpacked[17],
            quat_j=unpacked[18],
            quat_k=unpacked[19],
            error_handler=unpacked[20] if (temp_packet_flag & PACKET_HAS_ERROR) else None
        )


@dataclass
class CalibrationData:
    """Represents a calibration packet from the anchors."""
    source_anchor: int
    dest_anchor: int
    distance: float

    @classmethod
    def from_bytes(cls, binary_data: bytes) -> "CalibrationData":
        """Parses a calibration packet."""
        if len(binary_data) != CALIBRATION_PACKET_SIZE:
            raise ValueError(f"Expected calibration packet size {CALIBRATION_PACKET_SIZE}, got {len(binary_data)}")

        unpacked = struct.unpack(CALIBRATION_FORMAT, binary_data)
        return cls(
            source_anchor=unpacked[1],
            dest_anchor=unpacked[2],
            distance=unpacked[3]
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
        self.calibration_queue: queue.Queue = queue.Queue()  # Global calibration queue
        self.relays: dict[int, dict] = {}
        self.threads: dict[int, threading.Thread] = {}
        self.running: bool = False
        self.lock = threading.Lock()
        self.calibration_mode: bool = (MANUAL_ANCHOR_POSITIONS is None and MANUAL_ANCHOR_DISTANCES is None)

    def add_device(self, relay_id: int, port: str, baudrate: int = 115200) -> bool:
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

                    # Check for calibration packets first (smaller, higher priority)
                    if self.calibration_mode and len(buffer) >= CALIBRATION_PACKET_SIZE:
                        cal_header_index = buffer.find(CALIBRATION_HEADER_BYTE)
                        if cal_header_index != -1:
                            if cal_header_index > 0:
                                buffer[:] = buffer[cal_header_index:]
                            if len(buffer) >= CALIBRATION_PACKET_SIZE:
                                cal_packet_bytes = bytes(buffer[:CALIBRATION_PACKET_SIZE])
                                try:
                                    cal_data = CalibrationData.from_bytes(cal_packet_bytes)
                                    self.calibration_queue.put(cal_data)
                                    print(f"Calibration: Anchor {cal_data.source_anchor} -> {cal_data.dest_anchor}: {cal_data.distance:.3f}m")
                                except ValueError as e:
                                    print(f"Calibration packet error: {e}")
                                finally:
                                    buffer[:] = buffer[CALIBRATION_PACKET_SIZE:]
                                continue

                    # Check for normal data packets
                    while len(buffer) >= PACKET_SIZE:
                        header_index = buffer.find(HEADER_BYTE)
                        if header_index == -1:
                            buffer.clear()
                            break
                        if header_index > 0:
                            buffer[:] = buffer[header_index:]
                        if len(buffer) >= PACKET_SIZE:
                            packet_bytes = bytes(buffer[:PACKET_SIZE])
                            try:
                                packet_data = PacketData.from_bytes(packet_bytes)
                                wrapped_packet = DevicePacket(relay_id=relay_id, data=packet_data)
                                self.processing_queues[relay_id].put(wrapped_packet)
                            except ValueError as e:
                                print(f"Relay {relay_id} packet error: {e}")
                            finally:
                                buffer[:] = buffer[PACKET_SIZE:]  # Consume packet
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

    def send_correction(self, relay_id: int, device_id:int, correction: np.ndarray):
        """Sends a position correction vector to the specified device."""
        with self.lock:
            if relay_id in self.relays and self.relays[relay_id]['serial'].is_open:
                # Message format: 'C' for Correction, the device number, followed by 3 floats (dx, dy, dz)
                message = struct.pack('<cb3f', b'C', device_id.to_bytes(), correction[0], correction[1], correction[2])
                self.relays[relay_id]['serial'].write(message)


# endregion

# region ======================= Math Helper Functions =======================

def calculate_anchor_positions_from_distances(distances: dict[tuple[int, int], float]) -> dict[int, np.ndarray]:
    """
    Calculate 3D anchor positions from pairwise distances.
    This function defines a coordinate system based on the first few anchors:
    - Anchor 1 is at the origin (0,0,0).
    - Anchor 2 is on the positive X-axis.
    - Anchor 3 is on the XY plane with a positive Y value.
    - Anchors 4, 5, etc., are placed in 3D space relative to this frame.
    """
    positions = {}
    if NUM_UWB_ANCHORS < 3:
        raise ValueError("At least 3 anchors are required for 3D positioning.")

    # Anchor 1 at origin
    positions[1] = np.array([0.0, 0.0, 0.0])

    # Anchor 2 on X-axis
    d12 = distances.get((1, 2)) or distances.get((2, 1))
    if d12 is None: raise ValueError("Missing distance between anchors 1 and 2")
    positions[2] = np.array([d12, 0.0, 0.0])

    # Anchor 3 in XY plane
    d13 = distances.get((1, 3)) or distances.get((3, 1))
    d23 = distances.get((2, 3)) or distances.get((3, 2))
    if d13 is None or d23 is None: raise ValueError("Missing distances for anchor 3")
    
    # Law of cosines to find x3, y3
    x3 = (d13**2 + d12**2 - d23**2) / (2 * d12)
    y3_squared = d13**2 - x3**2
    y3 = math.sqrt(max(0, y3_squared))
    positions[3] = np.array([x3, y3, 0.0])

    # General trilateration for subsequent anchors (4, 5, ...)
    for i in range(4, NUM_UWB_ANCHORS + 1):
        d1i = distances.get((1, i)) or distances.get((i, 1))
        d2i = distances.get((2, i)) or distances.get((i, 2))
        d3i = distances.get((3, i)) or distances.get((i, 3))
        if d1i is None or d2i is None or d3i is None:
            raise ValueError(f"Missing distances for anchor {i}")

        # Solve for position (xi, yi, zi) using anchors 1, 2, and 3
        p2_x = positions[2][0]
        p3_x, p3_y = positions[3][0], positions[3][1]

        # from sphere equations (x-x_n)^2 + ... = d_n^2
        xi = (d1i**2 - d2i**2 + p2_x**2) / (2 * p2_x)
        yi = (d1i**2 - d3i**2 + p3_x**2 + p3_y**2 - 2 * p3_x * xi) / (2 * p3_y)
        zi_squared = d1i**2 - xi**2 - yi**2
        zi = math.sqrt(max(0, zi_squared))
        positions[i] = np.array([xi, yi, zi])

    return positions


def multilaterate_3d_wls(distances: list[float], anchor_positions: dict[int, np.ndarray]) -> np.ndarray:
    """
    3D multilateration using Weighted Least Squares for N >= 3 anchors.
    Returns (x, y, z) position.
    """
    # Build matrices for WLS: (A^T W A)^-1 A^T W b
    n = len(distances)
    if n < 3:
        raise ValueError("Need at least 3 distances for 3D multilateration.")
        
    A = np.zeros((n - 1, 3))
    b = np.zeros(n - 1)
    W = np.eye(n - 1)

    # Use the last anchor as the reference point to build a system of n-1 linear equations
    ref_pos = anchor_positions[n]
    ref_dist = distances[n-1]

    for i in range(n - 1):
        anchor_id = i + 1
        p_i = anchor_positions[anchor_id]
        d_i = distances[i]

        # Linear least squares formulation by subtracting sphere equations
        # 2(ref_pos - p_i) . X = d_i^2 - ref_dist^2 - |p_i|^2 + |ref_pos|^2
        A[i, :] = 2 * (ref_pos - p_i)
        b[i] = d_i**2 - ref_dist**2 - np.dot(p_i, p_i) + np.dot(ref_pos, ref_pos)

        # Weight inversely proportional to distance (closer = more accurate)
        W[i, i] = 1.0 / max(d_i, 0.1)**2

    try:
        # Weighted least squares solution: pos = (A^T W A)^-1 A^T W b
        AT_W_A = A.T @ W @ A
        AT_W_b = A.T @ W @ b
        pos = np.linalg.solve(AT_W_A, AT_W_b)
        return pos
    except np.linalg.LinAlgError:
        # Fallback to unweighted if singular
        pos, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return pos


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
    """Rotates a 3D vector by a quaternion."""
    q_vec = np.array([0, v[0], v[1], v[2]])
    q_inv = quaternion_inverse(q)
    q_rotated = quaternion_multiply(quaternion_multiply(q, q_vec), q_inv)
    return q_rotated[1:]


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
    anchor_positions: dict[int, np.ndarray] = field(default_factory=dict)  # 3D positions of UWB anchors
    triangulate_func: callable = None  # Will be set based on NUM_UWB_ANCHORS
    is_UWB_calibrated: bool = False
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

    # State updated directly from ESP32's dead-reckoning
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    rotation_quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    rotation_euler: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    # Position translated into sections
    x_section: int = 0
    y_section: int = 0

    # Raw UWB data for correction calculation
    UWB_distance_1: float | None = None
    UWB_distance_2: float | None = None
    UWB_distance_3: float | None = None
    UWB_distance_4: float | None = None
    UWB_distance_5: float | None = None

    # Reference point for UWB coordinate system calibration
    reference_UWB_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    reference_orientation_quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    inverse_reference_orientation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))

    # History for gesture detection
    position_history: deque = field(default_factory=lambda: deque(maxlen=10))
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=10))
    quat_history: deque = field(default_factory=lambda: deque(maxlen=10))
    euler_history: deque = field(default_factory=lambda: deque(maxlen=10))

    def _get_uwb_distances(self) -> list[float | None]:
        """Returns a list of UWB distances based on NUM_UWB_ANCHORS."""
        all_distances = [
            self.UWB_distance_1, self.UWB_distance_2, self.UWB_distance_3,
            self.UWB_distance_4, self.UWB_distance_5
        ]
        return all_distances[:NUM_UWB_ANCHORS]

    def calibrate_zero_frame(self):
        """Sets the current UWB position as the origin of the coordinate system."""
        distances = self._get_uwb_distances()

        if all(d is not None for d in distances):
            # Use 3D multilateration to get absolute position
            uwb_position = self.triangulate_func(distances, self.anchor_positions)
            self.reference_UWB_position = uwb_position
            self.reference_orientation_quaternion = self.rotation_quaternion.copy()
            self.inverse_reference_orientation = quaternion_inverse(self.reference_orientation_quaternion)
            self.is_UWB_calibrated = True
            print(f"Glove {self.device_id} UWB calibrated at {self.reference_UWB_position}")
        else:
            print(f"Glove {self.device_id}: UWB calibration failed. Missing UWB data.")

    def update_from_packet(self, p: DevicePacket):
        """Updates the glove's state from an incoming ESP32 packet."""
        self.button_pressed = p.data.button_state

        # Remember previous states
        self.position_history.append(self.position)
        self.velocity_history.append(self.velocity)
        self.euler_history.append(self.rotation_euler)

        # Raw data from packet
        raw_position = np.array([p.data.pos_x, p.data.pos_y, p.data.pos_z])
        raw_velocity = np.array([p.data.vel_x, p.data.vel_y, p.data.vel_z])
        raw_rotation = np.array([p.data.quat_w, p.data.quat_i, p.data.quat_j, p.data.quat_k])

        # Transform data to calibrated world frame
        self.position = rotate_vector_by_quaternion(raw_position, self.inverse_reference_orientation)
        self.velocity = rotate_vector_by_quaternion(raw_velocity, self.inverse_reference_orientation)
        self.rotation_quaternion = quaternion_multiply(self.inverse_reference_orientation, raw_rotation)

        self.rotation_euler = np.array(quat_to_euler_deg(self.rotation_quaternion), dtype=np.float64)

        if p.data.UWB_distance_1: self.UWB_distance_1 = p.data.UWB_distance_1
        if p.data.UWB_distance_2: self.UWB_distance_2 = p.data.UWB_distance_2
        if p.data.UWB_distance_3: self.UWB_distance_3 = p.data.UWB_distance_3
        if p.data.UWB_distance_4: self.UWB_distance_4 = p.data.UWB_distance_4
        if p.data.UWB_distance_5: self.UWB_distance_5 = p.data.UWB_distance_5

    def calculate_and_send_correction(self, reader: ThreadedMultiDeviceReader, relay_id: int, device_id:int):
        distances = self._get_uwb_distances()

        # Only proceed if we have fresh UWB data
        if all(d is not None for d in distances):
            # 1. Calculate UWB ground truth in the absolute anchor frame
            abs_uwb_pos = self.triangulate_func(distances, self.anchor_positions)

            # 2. Translate to the relative frame based on the calibration origin
            rel_uwb_pos = abs_uwb_pos - self.reference_UWB_position

            # 3. Calculate the correction vector (the drift)
            correction = rel_uwb_pos - self.position

            # 4. Apply bounding box clamping in the normal direction
            if self.glove_state == 2:  # Only apply bounding when box is drawn
                normal_component = correction[self.plane_normal_axis]
                if abs(normal_component) > BOUNDING_BOX_NORMAL_TOLERANCE:
                    correction[self.plane_normal_axis] = np.sign(normal_component) * BOUNDING_BOX_NORMAL_TOLERANCE

            # 5. Send the correction back to the ESP32
            reader.send_correction(relay_id, device_id, correction)

            # 6. Immediately apply the correction locally for smooth visualization
            self.position += correction

            # 7. Invalidate UWB data after use to prevent re-sending the same correction
            self.UWB_distance_1 = None
            self.UWB_distance_2 = None
            self.UWB_distance_3 = None
            self.UWB_distance_4 = None
            self.UWB_distance_5 = None

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
                self.glove_state = 1
                self.calibrate_zero_frame()
                self.position = np.array([0.0, 0.0, 0.0])
                if self.visual:
                    self.visual.box.visible = True

        elif self.glove_state == 1:
            if self.visual:
                if self.button_pressed:
                    abs_pos = np.abs(self.position)
                    self.plane_normal_axis = int(np.argmin(abs_pos))

                    plane_axes = [i for i in range(3) if i != self.plane_normal_axis]
                    max_dim = max(abs_pos[plane_axes[0]], abs_pos[plane_axes[1]])

                    dimensions = np.full(3, max_dim) * 2.0
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
                    max_extent = max(abs_pos[plane_axes[0]], abs_pos[plane_axes[1]])

                    self.active_area_half_extents = np.full(3, max_extent)
                    self.active_area_half_extents[self.plane_normal_axis] = 0.0

                    self._update_section_visuals()


@dataclass
class LeftHand(Glove):
    """Manages the parameters controlled by the left hand"""
    note: int = 0
    volume: int = 0
    reverb_mode: int = 0


@dataclass
class RightHand(Glove):
    """Manages the parameters controlled by the right hand"""
    stereo: float = 0
    attack: int = 0
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
    running: bool = field(init=False, default=False)
    processing_thread: threading.Thread = field(init=False)

    def start(self):
        """Starts the packet processing thread for this glove pair."""
        self.processing_thread = threading.Thread(target=self._packet_processor)
        self.processing_thread.daemon = True
        self.running = True
        self.processing_thread.start()

    def _process_single_packet(self, packet: DevicePacket, from_queue: bool = False):
        glove = self.left_hand if packet.data.device_number % 2 != 0 else self.right_hand
        glove.update_from_packet(packet)
        glove.calculate_and_send_correction(self.reader, self.relay_id, glove.device_id)
        if from_queue:
            self.relay_queue.task_done()
        glove.get_section_from_position()
        self._update_right_roll_gesture(glove, packet)

        if self.left_hand.glove_state == 2 and self.right_hand.glove_state == 2:
            if self.left_hand.in_active_area and self.right_hand.in_active_area:
                self.section_to_note()
                self.play_note()
            else:
                self.daw_interface.stop_notes()
                self.daw_interface.previous_note = NoteData.blank_note()

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
        if (
            abs(delta_roll) >= GESTURE_ROLL_DELTA_THRESHOLD_DEG
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
            note=self.left_hand.note,
            volume=self.left_hand.volume,
            reverb_mode=self.left_hand.reverb_mode,
            stereo=self.right_hand.stereo,
            attack=self.right_hand.attack,
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

                hand.note = hand.y_section + (self.current_octave * 12)

                hand.volume = int(np.clip(round(hand.rotation_euler[0] / 180 * 63 + 64), 0, 127))

                if hand.x_section == 1 or hand.x_section == 2:
                    hand.reverb_mode = 1
                elif hand.x_section == 0:
                    hand.reverb_mode = 0
                else:
                    hand.reverb_mode = 2

            if hand is self.right_hand:
                if hand.button_pressed:
                    self.current_octave += 1 if self.current_octave < 7 else 0

                hand.stereo = float(np.clip((hand.x_section - 2) * 0.25, -0.5, 0.5))  # section to stereo pan
                hand.attack = hand.y_section * 18  # converts section number 0 to 7 to attack int 0 to 127 (approximately)


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
    stereo: float | None  # between -0.5 and +0.5
    reverb_mode: int | None  # 0, 1, or 2

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
    previous_note: NoteData = field(init=False)

    def __post_init__(self):
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
                if note.reverb_mode == 0:
                    reverb = 5
                elif note.reverb_mode == 1:
                    reverb = 32
                else:
                    reverb = 127
                reverb_msg = mido.Message(
                    'control_change',
                    channel=11,
                    control=100,
                    value=reverb
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
                    value=note_pan
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
    def __init__(self, glove_pairs):
        self.glove_pairs = glove_pairs
        self.canvas = scene.SceneCanvas(keys='interactive', show=True)
        self.grid = self.canvas.central_widget.add_grid()
        self.view_pairs = []

        for i, pair in enumerate(self.glove_pairs):
            # Left Hand View
            lh_view = self.grid.add_view(row=i, col=0, border_color='white')
            lh_view.camera = scene.cameras.TurntableCamera()
            pair.left_hand.visual = GloveVisual(axis=visuals.XYZAxis(parent=lh_view.scene),
                                                parent_scene=lh_view.scene)

            # Right Hand View
            rh_view = self.grid.add_view(row=i, col=1, border_color='white')
            rh_view.camera = scene.cameras.TurntableCamera()
            pair.right_hand.visual = GloveVisual(axis=visuals.XYZAxis(parent=rh_view.scene),
                                                 parent_scene=rh_view.scene)
            lh_view.camera.link(rh_view.camera)
            if PERFORMANCE_MODE:
                self.view_pairs.append((rh_view, lh_view))
            else:
                self.view_pairs.append((lh_view, rh_view))

        self.timer = Timer('auto', connect=self.update, start=True)
        self.canvas.events.close.connect(self.on_close)

    def on_close(self, event):
        self.timer.stop()
        for pair in self.glove_pairs:
            pair.stop()
        if self.glove_pairs:
            self.glove_pairs[0].reader.stop()
        print("Visualization closed, system stopping.")
        app.quit()

    def update(self, event):
        for pair in self.glove_pairs:
            for hand in [pair.left_hand, pair.right_hand]:
                if hand.visual:
                    transform_matrix = quaternion_to_transform_matrix(hand.rotation_quaternion, hand.position)
                    hand.visual.axis.transform = scene.transforms.MatrixTransform(transform_matrix)
        self.canvas.update()


# endregion

# region ======================= Main Application =======================

def perform_auto_calibration(reader: ThreadedMultiDeviceReader) -> dict[int, np.ndarray]:
    """
    Waits for calibration packets from anchors and calculates their 3D positions.
    Returns a dictionary of anchor positions {anchor_id: np.ndarray([x, y, z])}.
    """
    print(f"\n{'='*60}")
    print("AUTO-CALIBRATION MODE")
    print(f"Waiting for {NUM_UWB_ANCHORS} UWB anchors to calibrate...")
    print(f"{'='*60}\n")

    # Collect all pairwise distances from calibration packets
    distances = {}
    required_measurements = NUM_UWB_ANCHORS * (NUM_UWB_ANCHORS - 1) // 2

    while len(distances) < required_measurements:
        try:
            cal_data = reader.calibration_queue.get(timeout=30.0)
            # Use a sorted tuple as key to handle (1,2) and (2,1) as the same measurement
            key = tuple(sorted((cal_data.source_anchor, cal_data.dest_anchor)))
            if key not in distances:
                distances[key] = cal_data.distance
                print(f"  [{len(distances)}/{required_measurements}] Anchor {cal_data.source_anchor} -> {cal_data.dest_anchor}: {cal_data.distance:.3f}m")
        except queue.Empty:
            print("Calibration timeout! Make sure all anchors are powered on and sending data.")
            raise TimeoutError("Auto-calibration failed: timeout waiting for anchor data")

    print(f"\n{'='*60}")
    print("Calibration complete! Calculating anchor positions...")
    print(f"{'='*60}\n")

    # Calculate anchor positions from the collected distances
    anchor_positions = calculate_anchor_positions_from_distances(distances)

    for anchor_id, pos in sorted(anchor_positions.items()):
        print(f"  Anchor {anchor_id}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m")

    print(f"\n{'='*60}\n")
    reader.calibration_mode = False
    return anchor_positions


def main():
    """Initializes and runs the main application."""
    # Create MIDI Object
    daw_interface = DawInterface(port=None)

    # Create reader object
    reader = ThreadedMultiDeviceReader()
    for i, port in enumerate(COM_PORTS, start=1):
        if not reader.add_device(relay_id=i, port=port):
            print(f"Could not connect to device on {port}. Please check connection.")

    reader.start()

    # Determine anchor positions
    anchor_positions = {}
    if MANUAL_ANCHOR_POSITIONS is not None:
        print("Using manual anchor positions (skipping auto-calibration)")
        for i, pos in enumerate(MANUAL_ANCHOR_POSITIONS, start=1):
            anchor_positions[i] = np.array(pos)
            print(f"  Anchor {i}: {pos}")
    elif MANUAL_ANCHOR_DISTANCES is not None:
        print("Using manual anchor distances")
        anchor_positions = calculate_anchor_positions_from_distances(MANUAL_ANCHOR_DISTANCES)
        for anchor_id, pos in anchor_positions.items():
            print(f"  Anchor {anchor_id}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m")
    else:
        # Auto-calibration mode
        anchor_positions = perform_auto_calibration(reader)

    # Create glove pairs with anchor positions
    glove_pairs = []
    if 1 in reader.processing_queues:
        left_hand = LeftHand(
            device_id=1,
            active_area_x_subsections=3,
            active_area_y_subsections=12,
            anchor_positions=anchor_positions,
            triangulate_func=multilaterate_3d_wls
        )
        right_hand = RightHand(
            device_id=2,
            active_area_x_subsections=5,
            active_area_y_subsections=8,
            anchor_positions=anchor_positions,
            triangulate_func=multilaterate_3d_wls
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

    print("\n" + "="*60)
    print("SYSTEM READY")
    print("="*60)
    print("System running. Close the visualization window to stop.\n")

    visualizer = Visualizer(glove_pairs)
    app.run()

    print("System stopped.")


if __name__ == "__main__":
    main()

# endregion
