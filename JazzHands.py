# region ======================= Imports, Constants, Enums =======================
import math
import queue
from collections import deque
from enum import IntEnum
from typing import Callable
import serial, struct, time, threading  # type: ignore[import-untyped]
import numpy as np
from dataclasses import dataclass, field
from vispy import app, scene  # type: ignore[import-untyped]

# from Visualization_Tests import Accel_Filtering_Test as vp  # import our own vispy for easier setup

# Definition of constants
COM_PORTS: list[str] = ["/dev/cu.usbserial-023B6AC7",  # Board 3
                        "/dev/cu.usbserial-023B6B29", ]  # Board 4
NUMBER_OF_DEVICES: int = 2
DISTANCE_BETWEEN_UWB_ANCHORS: float = 10  # meters
TIMEOUT: float = 0  # Non-Blocking
"""--------------PACKET DATA-------------- 
    Header (2)                   [H]
    Device Number (1)            [B]
    timestamp (4)                [I]
    packet type flags (1)        [B]
    Button State (1)             [B]
    X,Y,Z Accel (4*3 = 12)      [3f]
    UWB (4x2=8)                 [2f]
    Quaternion (4*4 = 16)       [4f]
    Error Handler (1)            [B]
Total:                       46 bytes """
PACKET_SIZE: int = 46  # Number of bytes per packet:
HEADER_BYTE: bytes = b'\xAA\xAA'  # the header is 0xAAAA
HEADER_SIZE: int = len(HEADER_BYTE)  # header is 2 bytes long
PACKET_FORMAT: str = '<HBIBB3f2f4fB'  # tells the unpacker the order of packet:
OCTAVE: int = 12  # Makes MIDI/Note math look better

# Bit Flagging Constants
PACKET_HAS_ACCEL: int = 0b00000001
PACKET_HAS_QUAT: int = 0b00000010
PACKET_HAS_UWB_1: int = 0b00000100
PACKET_HAS_UWB_2: int = 0b00001000
PACKET_HAS_ERROR: int = 0b10000000


# Note Definitions

class Note(IntEnum):
    # Octave 0
    C0 = 12
    Cs0 = 13
    D0 = 14
    Ds0 = 15
    E0 = 16
    F0 = 17
    Fs0 = 18
    G0 = 19
    Gs0 = 20
    A0 = 21  # Lowest piano key
    As0 = 22
    B0 = 23

    # Octave 1
    C1 = 24
    Cs1 = 25
    D1 = 26
    Ds1 = 27
    E1 = 28
    F1 = 29
    Fs1 = 30
    G1 = 31
    Gs1 = 32
    A1 = 33
    As1 = 34
    B1 = 35

    # Octave 2
    C2 = 36
    Cs2 = 37
    D2 = 38
    Ds2 = 39
    E2 = 40
    F2 = 41
    Fs2 = 42
    G2 = 43
    Gs2 = 44
    A2 = 45
    As2 = 46
    B2 = 47

    # Octave 3
    C3 = 48
    Cs3 = 49
    D3 = 50
    Ds3 = 51
    E3 = 52
    F3 = 53
    Fs3 = 54
    G3 = 55
    Gs3 = 56
    A3 = 57
    As3 = 58
    B3 = 59

    # Octave 4 (Middle C octave)
    C4 = 60  # Middle C
    Cs4 = 61
    D4 = 62
    Ds4 = 63
    E4 = 64
    F4 = 65
    Fs4 = 66
    G4 = 67
    Gs4 = 68
    A4 = 69  # 440 Hz tuning standard
    As4 = 70
    B4 = 71

    # Octave 5
    C5 = 72
    Cs5 = 73
    D5 = 74
    Ds5 = 75
    E5 = 76
    F5 = 77
    Fs5 = 78
    G5 = 79
    Gs5 = 80
    A5 = 81
    As5 = 82
    B5 = 83

    # Octave 6
    C6 = 84
    Cs6 = 85
    D6 = 86
    Ds6 = 87
    E6 = 88
    F6 = 89
    Fs6 = 90
    G6 = 91
    Gs6 = 92
    A6 = 93
    As6 = 94
    B6 = 95

    # Octave 7
    C7 = 96
    Cs7 = 97
    D7 = 98
    Ds7 = 99
    E7 = 100
    F7 = 101
    Fs7 = 102
    G7 = 103
    Gs7 = 104
    A7 = 105
    As7 = 106
    B7 = 107

    # Octave 8
    C8 = 108  # Highest piano key


# Helper functions
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


# endregion

# TODO: determine if these are necessary
def to_freq(note: int) -> float:
    return 440.0 * (2 ** ((note - 69) / 12))


def to_name(note: int) -> str:
    return f"{NOTE_NAMES[note % 12]}{note // 12 - 1}"


# region ======================= Data Classes =======================

@dataclass
class PacketData:
    device_number: int
    timestamp: int
    packet_flags: int
    button_state: bool
    accel_x: float | None
    accel_y: float | None
    accel_z: float | None
    UWB_distance_1: float | None
    UWB_distance_2: float | None
    quat_w: float | None
    quat_i: float | None
    quat_j: float | None
    quat_k: float | None
    error_handler: int | None

    @classmethod
    def from_bytes(cls, binary_data: bytes) -> PacketData:
        """Parse the binary data"""
        if len(binary_data) != PACKET_SIZE:
            raise ValueError(
                f"Expected {PACKET_SIZE}, got {len(binary_data)}")  # raise an error if wrong amount received
        try:
            unpacked = struct.unpack(PACKET_FORMAT, binary_data)
        except struct.error:
            raise struct.error()
        temp_packet_flag: int = unpacked[3]  # a temp var to use the packet flags.
        return cls(
            device_number=unpacked[1],  # unpacked[0] is the header
            timestamp=unpacked[2],
            packet_flags=unpacked[3],
            button_state=bool(unpacked[4]),
            accel_x=unpacked[5] if ((temp_packet_flag & PACKET_HAS_ACCEL) == PACKET_HAS_ACCEL) else None,
            accel_y=unpacked[6] if ((temp_packet_flag & PACKET_HAS_ACCEL) == PACKET_HAS_ACCEL) else None,
            accel_z=unpacked[7] if ((temp_packet_flag & PACKET_HAS_ACCEL) == PACKET_HAS_ACCEL) else None,
            UWB_distance_1=unpacked[8] if ((temp_packet_flag & PACKET_HAS_UWB_1) == PACKET_HAS_UWB_1) else None,
            UWB_distance_2=unpacked[9] if ((temp_packet_flag & PACKET_HAS_UWB_2) == PACKET_HAS_UWB_2) else None,
            quat_w=unpacked[10] if ((temp_packet_flag & PACKET_HAS_QUAT) == PACKET_HAS_QUAT) else None,
            quat_i=unpacked[11] if ((temp_packet_flag & PACKET_HAS_QUAT) == PACKET_HAS_QUAT) else None,
            quat_j=unpacked[12] if ((temp_packet_flag & PACKET_HAS_QUAT) == PACKET_HAS_QUAT) else None,
            quat_k=unpacked[13] if ((temp_packet_flag & PACKET_HAS_QUAT) == PACKET_HAS_QUAT) else None,
            error_handler=unpacked[14] if ((temp_packet_flag & PACKET_HAS_ERROR) == PACKET_HAS_ERROR) else None
        )


@dataclass
class DevicePacket:
    """Wrapper for packet with metadata"""
    relay_id: int
    data: PacketData
    timestamp: float = field(default_factory=time.time)
    sequence_num: int = 0


# endregion

# region ======================= Threaded Device Reader =======================

class ThreadedMultiDeviceReader:
    """Manages multiple serial devices with thread-safe queuing"""

    def __init__(self):
        # Main queue for all devices
        self.processing_queues: dict[int, queue.Queue] = {}

        # Device management
        self.relays: dict[int, dict] = {}  # relay_id -> device info
        self.threads: dict[int, threading.Thread] = {}

        # Control
        self.running: bool = False
        self.stats: dict = {}  # Track packet counts, errors, etc.
        self.lock = threading.Lock()

        # Callbacks
        self.packet_callback_function: Callable | None = None
        self.error_callback_function: Callable | None = None

    def add_device(self, relay_id: int, port: str, baudrate: int = 115200) -> bool:
        """Add a new device to monitor"""
        with self.lock:
            if relay_id in self.relays:
                print(f"Device {relay_id} already exists")
                return False

            # Create dedicated queue for this relay
            self.processing_queues[relay_id] = queue.Queue()
            try:
                # Create serial connection
                ser = serial.Serial(port, baudrate, timeout=1)

                # Store device info
                self.relays[relay_id] = {
                    'port': port,
                    'serial': ser,
                    'buffer': bytearray(),
                    'packet_count': 0,
                    'error_count': 0,
                    'last_packet_time': None
                }

                # Initialize stats
                self.stats[relay_id] = {
                    'packets_received': 0,
                    'bytes_received': 0,
                    'errors': 0,
                    'start_time': time.time()
                }

                # Start thread for this device
                thread = threading.Thread(
                    target=self._device_reader_thread,
                    args=(relay_id,),
                    name=f"Reader-Device-{relay_id}"
                )
                thread.daemon = True
                self.threads[relay_id] = thread

                if self.running:
                    thread.start()

                print(f"Added device {relay_id} on {port}")
                return True

            except serial.SerialException as e:
                print(f"Failed to open {port}: {e}")
                return False

    def __enter__(self):
        """Called when entering 'with' block"""
        self.start()  # Start all threads
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Called when exiting 'with' block (even on error!)"""
        self.stop()  # Stop threads, close ports
        return False  # Don't suppress exceptions

    def remove_device(self, relay_id: int):
        """Remove and stop monitoring a device"""
        with self.lock:
            if relay_id not in self.relays:
                return

            # Close serial port
            self.relays[relay_id]['serial'].close()

            # Remove from tracking
            del self.relays[relay_id]

    def _device_reader_thread(self, relay_id: int):
        """Thread function for reading from one device"""
        device = self.relays[relay_id]
        ser = device['serial']
        buffer = device['buffer']

        print(f"Started thread for device {relay_id}")

        while self.running and relay_id in self.relays:
            try:
                # Read available data
                if ser.in_waiting > 0:
                    new_data = ser.read(ser.in_waiting)
                    buffer.extend(new_data)

                    # Try to parse packets
                    while len(buffer) >= PACKET_SIZE:
                        # Look for header
                        header_index = buffer.find(HEADER_BYTE)

                        if header_index == -1:
                            # No header found, clear buffer
                            buffer.clear()
                            print("Discarded buffer (no packet header found)")
                            break

                        # Remove bytes before header
                        if header_index > 0:
                            buffer[:] = buffer[header_index:]

                        # Check if we have complete packet
                        if len(buffer) >= PACKET_SIZE:
                            packet_bytes = bytes(buffer[:PACKET_SIZE])

                            try:
                                # Parse packet
                                print("got a packet")
                                packet_data = PacketData.from_bytes(packet_bytes)

                                # Update device info
                                with self.lock:
                                    device['packet_count'] += 1
                                    device['last_packet_time'] = time.time()

                                # Create wrapped packet
                                wrapped_packet = DevicePacket(
                                    relay_id=relay_id,
                                    data=packet_data,
                                    sequence_num=device['packet_count']
                                )

                                if self.packet_callback_function:
                                    self.packet_callback_function(wrapped_packet)

                                # Put into processing queue
                                self.processing_queues[relay_id].put(wrapped_packet)

                                # Remove packet from buffer
                                buffer[:] = buffer[PACKET_SIZE:]

                            except Exception as e:
                                print(f"Dropped packet due to parsing error: {e}")
                                # Parse error
                                with self.lock:
                                    device['error_count'] += 1
                                    self.stats[relay_id]['errors'] += 1

                                if self.error_callback_function:
                                    self.error_callback_function(relay_id, e)

                                # Remove bad packet
                                buffer[:] = buffer[1:]  # Skip one byte and try again
                        else:
                            # Need more data
                            break

                else:
                    # No data available, small sleep
                    time.sleep(0.001)

            except Exception as e:
                print(f"Error in device {relay_id} thread: {e}")
                time.sleep(0.1)  # Back off on error

        print(f"Stopped thread for device {relay_id}")

    # TODO: figure out what to do with this lol
    """
    def _callback_processor(self):
        Dedicated thread for callbacks
        while True:
            packet: DevicePacket = self.processing_queue.get()  # type: ignore[annotation-unchecked]
            if packet is None:
                break
            try:
                if self.packet_callback_function:
                    self.packet_callback_function(packet)
            except Exception as e:
                print(f"Callback error: packet for device {packet.relay_id}: {e}")
"""

    def start(self):
        """Start all device threads"""
        self.running = True
        for relay_id, thread in self.threads.items():
            if not thread.is_alive():
                thread.start()
        print(f"Started {len(self.threads)} device threads")

    def stop(self):
        """Stop all device threads"""
        print("Stopping all device threads...")
        self.running = False

        # Wait for threads to stop
        for thread in self.threads.values():
            thread.join(timeout=2.0)

        # Close all serial ports
        with self.lock:
            for device in self.relays.values():
                try:
                    device['serial'].close()
                except serial.SerialException:
                    pass

        print("All threads stopped")

    def get_stats(self) -> dict:
        """Get statistics for all devices"""
        with self.lock:
            stats_copy = {}
            for relay_id, stats in self.stats.items():
                runtime = time.time() - stats['start_time']
                stats_copy[relay_id] = {
                    **stats,
                    'runtime': runtime,
                    'packets_per_sec': stats['packets_received'] / runtime if runtime > 0 else 0,
                    'bytes_per_sec': stats['bytes_received'] / runtime if runtime > 0 else 0
                }
            return stats_copy

    def is_device_active(self, relay_id: int, timeout: float = 1.0) -> bool:
        """Check if device received data recently"""
        with self.lock:
            if relay_id not in self.relays:
                return False

            last_time = self.relays[relay_id]['last_packet_time']
            if last_time is None:
                return False

            return (time.time() - last_time) < timeout


# endregion

# region ======================= Math Helper Functions =======================

def quat_to_euler(q):
    """
    Quaternion (x, y, z, w) → Euler angles (roll, pitch, yaw) in radians.

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


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Returns the inverse of a unit quaternion."""
    # Negate x, y, z (keep w positive)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Fast, unrolled Hamilton quaternion multiplication."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,  # w
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,  # i
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,  # j
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2  # k
    ])


def rotate_vector(quaternion: np.ndarray, local_vector: np.ndarray) -> np.ndarray:
    """Rotates a vector using unrolled multiplication (fastest)"""
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    vx, vy, vz = local_vector[0], local_vector[1], local_vector[2]

    # Pre-calculate repeated multiplications
    qx2 = qx * 2.0
    qy2 = qy * 2.0
    qz2 = qz * 2.0

    # Cross product of q_vector and v
    cx = qy * vz - qz * vy
    cy = qz * vx - qx * vz
    cz = qx * vy - qy * vx

    # q_vector cross (q_vector cross v + qw * v)
    tx = cx + qw * vx
    ty = cy + qw * vy
    tz = cz + qw * vz

    # Final rotation
    return np.array([
        vx + qy2 * tz - qz2 * ty,
        vy + qz2 * tx - qx2 * tz,
        vz + qx2 * ty - qy2 * tx
    ])


def get_dt_seconds(current_us: int, last_us: int) -> float:
    """Calculates dt in seconds, perfectly handling 32-bit microsecond rollovers."""

    # Calculate the raw microsecond difference, masking it to 32 bits.
    # If a rollover occurred, this handles the math wrap-around instantly.
    dt_us = (current_us - last_us) & 0xFFFFFFFF

    # Convert to seconds
    return dt_us / 1000000.0


def triangulate_position(d1, d2, D) -> tuple[float, float]:
    """Given two UWB distances, calculate the coordinates assuming our setup
    Calculates 2D coordinates using UWB distances and the Law of Cosines.
    Assumes Anchor 1 is at (0,0) and Anchor 2 is at (D, 0).
    """
    # 1. Check for triangle inequality violations (sensor noise)
    if d1 + d2 < D:
        # The tag is between the anchors, but distances are too short.
        # Best guess: scale the distances to meet on the line.
        return (d1 / (d1 + d2)) * D, 0.0

    # 2. Calculate the X coordinate
    x = (d1 ** 2 + D ** 2 - d2 ** 2) / (2 * D)

    # 3. Calculate the Y coordinate safely
    # If noise pushes x slightly larger than d1, clamp it to prevent math domain errors
    y_squared = d1 ** 2 - x ** 2
    if y_squared < 0:
        y = 0.0
    else:
        y = math.sqrt(y_squared)

    return x, y


# endregion

# region ======================= Glove Classes=======================

@dataclass
class Glove:
    """Basic Glove data"""
    # Basic Data
    device_id: int
    current_packet_timestamp: int = 0  # in µs
    last_packet_timestamp: int = field(init=False)  # in µs
    UWB_distance_1: float | None = field(init=False)
    UWB_distance_2: float | None = field(init=False)
    UWB_1_timestamp: int | None = None  # in µs
    UWB_2_timestamp: int | None = None  # in µs
    is_initiated: bool = False
    button_state: bool = False

    # Current 3D orientation data
    local_acceleration: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    global_acceleration: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    rotation_quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    rotation_euler: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))

    # History of past 100ms-worth of data
    acceleration_history: deque = field(default_factory=lambda: deque(maxlen=10))
    position_history: deque = field(default_factory=lambda: deque(maxlen=10))
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=10))
    rotation_history: deque = field(default_factory=lambda: deque(maxlen=10))

    # Initial state definitions
    reference_quaternion: np.ndarray = field(default_factory=lambda: np.array(
        [1.0, 0.0, 0.0, 0.0]))  # Stores a Quat to define XYZ relative to the start position
    reference_UWB_coordinates: tuple[float, float] = (0.0, 0.0)

    def calibrate_zero_frame(self):
        """Sets the current orientation as the new global 'zero' frame."""
        # Set current rotation as global reference frame
        self.reference_quaternion = self.rotation_quaternion.copy()
        # Find current UWB coordinates
        self.reference_UWB_coordinates = triangulate_position(self.UWB_distance_1, self.UWB_distance_2,
                                                              DISTANCE_BETWEEN_UWB_ANCHORS)
        print(f"Glove {self.device_id} frame calibrated!")

    def get_dynamic_zupt_threshold(self) -> float:
        """Calculates a dynamic ZUPT threshold based on recent movement variance.
          - (From Gemini)
        """
        # TODO: Need to test dynamic zupt threshold
        # Need a full queue to do meaningful math
        if len(self.acceleration_history) < self.acceleration_history.maxlen:  # type: ignore[operator]
            return 0.5  # Your safe default

        # Calculate the magnitude of the last 10 acceleration vectors
        recent_accel_magnitudes: list[np.floating] = [np.linalg.norm(a) for a in self.acceleration_history]

        # Calculate the variance (how much the signal is wiggling)
        variance: np.floating = np.var(recent_accel_magnitudes)

        # Formula: Absolute Hardware Noise Floor + (Variance * Sensitivity Multiplier)
        # When moving fast/hitting hard, threshold spikes up to ignore ringing.
        # When still, threshold drops to the baseline hardware noise floor.
        base_noise_floor: float = 0.15  # The raw noise of the sensor sitting on a table
        dynamic_scaling: float = 2.0  # How aggressively it blocks post-hit ringing (tune this)

        # Cap the maximum threshold so it doesn't get permanently stuck
        max_threshold: float = 2.5

        calculated_threshold: np.floating = base_noise_floor + (variance * dynamic_scaling)
        return min(calculated_threshold, max_threshold)  # type: ignore[return-value]

    def needs_zupt(self) -> bool:
        """Check if local accel is just signal noise."""
        magnitude = np.linalg.norm(self.local_acceleration)
        # dynamic_threshold = self.get_dynamic_zupt_threshold()
        dynamic_threshold = 0.15
        return bool(magnitude < dynamic_threshold)  # return the True or False that this generates

    def integrate_function(self):
        """Calculate and integrate global-frame accel to find velocity and position"""

        dt = get_dt_seconds(self.current_packet_timestamp, self.last_packet_timestamp)
        # Check for ESP32 Reset:
        # if dt is very large, go back to zero everything.
        if dt > 1.0:
            print(f"[!] ESP32 Reset or major lag detected on Glove {self.device_id}. Re-zeroing...")

            # Reset the physics state
            self.velocity = np.array([0.0, 0.0, 0.0])
            self.position = np.array([0.0, 0.0, 0.0])

            # Re-establish our base coordinate frame
            self.calibrate_zero_frame()

            # Skip integration for this frame to avoid physics explosions
            return

        # ZUPT on the LOCAL acceleration
        if self.needs_zupt():
            self.velocity *= 0.5  # dampen velocity by 1 half
            if np.linalg.norm(self.velocity) < 0.01:
                self.velocity = np.array([0.0, 0.0, 0.0])
        else:
            # Calculate the relative orientation
            q_ref_inv = quat_conjugate(self.reference_quaternion)
            q_relative = quat_multiply(q_ref_inv, self.rotation_quaternion)

            # Rotate local accel into your CUSTOM global frame
            global_accel = rotate_vector(q_relative, self.local_acceleration)

            # Integrate velocity (only if ZUPT ≠ True)
            self.velocity += global_accel * dt
        # Integrate position always (ve.locity could be dampening
        self.position += self.velocity * dt

        # If both UWB values have been populated, pull position closer to UWB position

        # Check to see if they are timely enough to use
        if self.UWB_distance_1 is not None and self.UWB_distance_2 is not None:

            # Calculate the forward distance in BOTH directions
            gap_1_to_2 = (self.UWB_2_timestamp - self.UWB_1_timestamp) & 0xFFFFFFFF  # type: ignore[operator]
            gap_2_to_1 = (self.UWB_1_timestamp - self.UWB_2_timestamp) & 0xFFFFFFFF  # type: ignore[operator]

            # The smaller gap is the true time difference
            if gap_1_to_2 < gap_2_to_1:
                uwb_time_gap = gap_1_to_2  # In µs
                older_is_1 = True
            else:
                uwb_time_gap = gap_2_to_1  # In µs
                older_is_1 = False

            # 10,000 µs = 100 ms threshold
            if uwb_time_gap < 100000:
                triangulate_position(self.UWB_distance_1, self.UWB_distance_2, DISTANCE_BETWEEN_UWB_ANCHORS)
            else:
                # Discard the older reading to fix temporal shearing
                if older_is_1:
                    self.UWB_distance_1 = None
                else:
                    self.UWB_distance_2 = None

            # TODO: finish Triangulation "pulling" system

        # Save history
        self.velocity_history.append(self.velocity.copy())
        self.position_history.append(self.position.copy())

    def update_values(self, p: DevicePacket):
        # Set new data if we have it
        self.last_packet_timestamp = self.current_packet_timestamp if self.current_packet_timestamp else 0
        self.current_packet_timestamp = p.data.timestamp
        self.button_state = p.data.button_state
        # Accel and Quat should always run, therefore not using an If statement
        try:
            self.local_acceleration = np.array([p.data.accel_x, p.data.accel_y, p.data.accel_z])
        except Exception as e:
            print(f"Error with updating Acceleration: {e}")
        try:
            self.rotation_quaternion = np.array([p.data.quat_w, p.data.quat_i, p.data.quat_j, p.data.quat_k])
        except Exception as e:
            print(f"Error with updating Quaternion: {e}")
        # UWB is not always in the packet, so only update the values if needed
        if p.data.UWB_distance_1:
            self.UWB_distance_1 = p.data.UWB_distance_1
            self.UWB_1_timestamp = p.data.timestamp
        if p.data.UWB_distance_2:
            self.UWB_distance_2 = p.data.UWB_distance_2
            self.UWB_2_timestamp = p.data.timestamp

        # After getting the latest data, integrate it
        self.integrate_function()


@dataclass
class LeftHand(Glove):
    """Left-hand specific methods and data in addition to regular glove data"""
    device_id: int
    current_note: int = field(init=False)
    current_octave: int = field(init=False, default=4)
    # pitch: int = self.current_note + self.current_octave * 12


@dataclass
class RightHand(Glove):
    """Right-hand specific methods and data based on regular glove data"""
    device_id: int


@dataclass
class GlovePair:
    """Container for a pair of glove states"""
    device_ids: tuple[int, int]
    relay_id: int  # tied to its relay
    instrument_type: str
    relay_queue: queue.Queue
    integration_thread: threading.Thread = field(init=False)
    left_hand: LeftHand = field(init=False)
    right_hand: RightHand = field(init=False)
    running: bool = field(init=False, default=False)

    def __post_init__(self):
        self.left_hand = LeftHand(device_id=self.device_ids[0])
        self.right_hand = RightHand(device_id=self.device_ids[1])

    def start(self):
        self.integration_thread = threading.Thread(target=self._math_processor)
        self.integration_thread.daemon = True
        self.running = True
        self.integration_thread.start()
        return self

    def _math_processor(self):
        while self.running:
            try:
                packet = self.relay_queue.get(timeout=0.1)
                self.process_incoming_packet(packet)
                self.relay_queue.task_done()
            except queue.Empty:
                # No packet arrived in the last 0.1 seconds.
                # Just ignore it and let the while loop check self.running again.
                pass
            except Exception as e:
                print(f"Error in math processor for reader {self.relay_id}: {e}")

    def stop(self):
        self.running = False
        self.integration_thread.join(timeout=2.0)

    def process_incoming_packet(self, packet: DevicePacket) -> None:
        if packet.relay_id != self.relay_id:
            return
        match packet.data.device_number % 2:
            case 1:
                self.left_hand.update_values(packet)
            case 2:
                self.right_hand.update_values(packet)

    def main_logic(self):  # TODO: see if we should do it this way
        """Does all the main updating, which includes:
            1. Find position within box
            2. Translate into music stuff
            3. Send to DAW
        """
        pass


# endregion
# region ======================= Main Loop =======================

# canvas, view, grid, axes, x_label, y_label, z_label = vp.setup_canvas()

def main():
    # Create the Serial Reader and packet processor
    reader = ThreadedMultiDeviceReader()
    for number, com_port in enumerate(COM_PORTS, start=1):
        reader.add_device(relay_id=number, port=com_port)

    # Set up glove pairs
    glove_pairs: list[GlovePair] = [
        melody_glove := GlovePair(device_ids=(1, 2), relay_id=1, instrument_type="melody",
                                  relay_queue=reader.processing_queues[1]),
        drum_glove := GlovePair(device_ids=(3, 4), relay_id=2, instrument_type="drums",
                                relay_queue=reader.processing_queues[2]),
    ]

    # Vispy Setup (IDK how this is gonna work yet, all I know is that it will just fetch data from the gloves at 60Hz
    def update_visuals(event):
        # left_hand_pos: np.ndarray = melody_glove.left_hand.position
        # left_hand_vel: np.ndarray = melody_glove.left_hand.velocity
        # left_hand_acc: np.ndarray = melody_glove.left_hand.global_acceleration
        # right_hand_pos: np.ndarray = melody_glove.right_hand.position
        # right_hand_vel: np.ndarray = melody_glove.right_hand.velocity
        # right_hand_acc: np.ndarray = melody_glove.right_hand.global_acceleration
        pass

    timer = app.Timer(interval=(1 / 60), connect=update_visuals, start=False)
    app.run()
    # region ======================= Initiation Logic =======================


if __name__ == "__main__":
    main()
