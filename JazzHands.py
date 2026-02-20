import queue
from collections import deque
from enum import IntEnum
from typing import Callable
import serial, struct, time, threading  # type: ignore[import-untyped]
import numpy as np
from dataclasses import dataclass, field

# Definition of constants
COM_PORTS: list[str] = ["/dev/cu.usbserial-023B6AC7",  # Board 3
                        "/dev/cu.usbserial-023B6B29", ]  # Board 4
NUMBER_OF_DEVICES: int = 2
TIMEOUT: float = 0  # Non-Blocking
"""--------Packet details----------
Header (2) [2B]
Device Number (1) [B]
packet type flags (1) [B]
timestamp (4) [L]
Button State (1) [B]
X,Y,Z Accel (4*3 = 12) [3f]
UWB (4x2=8) [2f]
Quaternion (4*4 = 16) [4f]
Total: 46 bytes"""
PACKET_SIZE: int = 46  # Number of bytes per packet:
HEADER_BYTE: bytes = b'\xAA\xAA'  # the header is 0xAAAA
HEADER_SIZE: int = len(HEADER_BYTE)  # header is 2 bytes long
PACKET_FORMAT: str = '<H B I B B 3f 2f 4f'  # tells the unpacker the order of packet:
OCTAVE: int = 12  # Makes MIDI/Note math look better

'''---------Bit Flagging Constants---------'''
PACKET_HAS_ACCEL: int = 0b00000001
PACKET_HAS_QUAT: int = 0b00000010
PACKET_HAS_UWB_1: int = 0b00000100
PACKET_HAS_UWB_2: int = 0b00001000
PACKET_HAS_ERROR: int = 0b10000000

'''---------Note Definitions---------'''


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


def to_freq(note: int) -> float:
    return 440.0 * (2 ** ((note - 69) / 12))


def to_name(note: int) -> str:
    return f"{NOTE_NAMES[note % 12]}{note // 12 - 1}"


@dataclass
class PacketData:
    device_number: int
    timestamp: float
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
        unpacked = struct.unpack(PACKET_FORMAT, binary_data)
        temp_packet_flag: int = unpacked[HEADER_SIZE + 2]  # a temp var to use the packet flags.
        return cls(
            device_number=unpacked[HEADER_SIZE],
            timestamp=unpacked[HEADER_SIZE + 1],
            packet_flags=unpacked[HEADER_SIZE + 2],
            button_state=bool(unpacked[HEADER_SIZE + 3]),
            accel_x=unpacked[HEADER_SIZE + 4] if (temp_packet_flag & PACKET_HAS_ACCEL) else None,
            accel_y=unpacked[HEADER_SIZE + 5] if (temp_packet_flag & PACKET_HAS_ACCEL) else None,
            accel_z=unpacked[HEADER_SIZE + 6] if (temp_packet_flag & PACKET_HAS_ACCEL) else None,
            UWB_distance_1=unpacked[HEADER_SIZE + 7] if (temp_packet_flag & PACKET_HAS_UWB_1) else None,
            UWB_distance_2=unpacked[HEADER_SIZE + 8] if (temp_packet_flag & PACKET_HAS_UWB_2) else None,
            quat_w=unpacked[HEADER_SIZE + 9] if (temp_packet_flag & PACKET_HAS_QUAT) else None,
            quat_i=unpacked[HEADER_SIZE + 10] if (temp_packet_flag & PACKET_HAS_QUAT) else None,
            quat_j=unpacked[HEADER_SIZE + 11] if (temp_packet_flag & PACKET_HAS_QUAT) else None,
            quat_k=unpacked[HEADER_SIZE + 12] if (temp_packet_flag & PACKET_HAS_QUAT) else None,
            error_handler=unpacked[HEADER_SIZE + 13] if (temp_packet_flag & PACKET_HAS_ERROR) else None
        )


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


@dataclass
class DevicePacket:
    """Wrapper for packet with metadata"""
    relay_id: int
    data: PacketData
    timestamp: float = field(default_factory=time.time)
    sequence_num: int = 0


@dataclass
class Glove:
    """Basic Glove data"""
    device_id: int
    is_initiated: bool = field(init=False, default=False)
    last_packet_time: float | None = field(default_factory=lambda: None)
    button_state: bool = field(default_factory=lambda: False)
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    acceleration: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    rotation_quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    rotation_euler: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    acceleration_queue: deque = field(default_factory=lambda: deque(maxlen=10))
    position_history: deque = field(default_factory=lambda: deque(maxlen=10))
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=10))
    rotation_history: deque = field(default_factory=lambda: deque(maxlen=10))

    def needs_zupt(self) -> bool:
        """Check if accel is due to signal noise or an actual value"""
        magnitude = np.linalg.norm(self.acceleration)
        accel_zupt_threshold: float = 0.5  # TODO: TO BE DECIDED WHETHER THIS IS DYNAMIC OR STATIC, Also could be after lo-pass filter
        if magnitude < accel_zupt_threshold:
            return True
        else:
            return False

    def integrate_function(self):
        # integrate accel to find velocity and position
        if self.needs_zupt():
            # Set velocity to zero and ignore accel.
            self.acceleration = np.array([0, 0, 0])
        else:
            pass

    def update_data(self, packet: DevicePacket):
        self.button_state = packet.data.button_state
        self.acceleration_queue.append(np.array([packet.data.accel_x, packet.data.accel_y, packet.data.accel_z]))


@dataclass
class LeftHand(Glove):
    """Left-hand specific methods and data in addition to regular glove data"""
    current_note: int = field(init=False)
    current_octave: int = field(init=False, default=4)
    # pitch: int = current_note + current_octave * 12


@dataclass
class RightHand(Glove):
    """Right-hand specific methods and data based on regular glove data"""


@dataclass
class GlovePair:
    """Container for a pair of glove states"""
    device_ids: tuple[int, int]
    reader_number: int  # tied to its relay
    instrument_type: str
    left_hand: LeftHand = field(init=False)
    right_hand: RightHand = field(init=False)
    running: bool = field(init=False, default=False)

    def start(self):
        threading.Thread(target=self._math_processor).start()

    def __post_init__(self):
        self.left_hand = LeftHand(device_id=self.device_ids[0])
        self.right_hand = RightHand(device_id=self.device_ids[1])

    def _math_processor(self):
        self.main_logic()
        pass

    def stop(self):
        self.running = False
        threading.Thread(target=self._math_processor).join(timeout=1)

    def process_incoming_packet(self, packet: DevicePacket):
        if packet.relay_id != self.reader_number:
            return
        match packet.data.device_number % 2:
            case 1:
                self.left_hand.update_data(packet)
            case 2:
                self.right_hand.update_data(packet)

    def main_logic(self):
        """Does all the main updating, which includes:
            1.
            2.
            3.
            4.
        """
        pass


def main():
    # Create glove pairs
    glove_pairs: list[GlovePair] = [  # type: ignore[annotation-unchecked]
        melody_glove := GlovePair(device_ids=(1, 2), reader_number=1, instrument_type="melody"),
        drum_glove := GlovePair(device_ids=(3, 4), reader_number=2, instrument_type="drums"),
    ]

    # Create reader
    with ThreadedMultiDeviceReader() as reader:
        # Add devices
        for number, com_port in enumerate(COM_PORTS, start=1):
            reader.add_device(relay_id=number, port=com_port)

    def on_packet(packet: DevicePacket):
        for pair in glove_pairs:
            pair.process_incoming_packet(packet)
        try:
            # Main processing loop
            while True:

                # Print stats occasionally (for debugging)
                if int(time.time()) % 10 == 0:
                    stats = reader.get_stats()
                    for device_id, stat in stats.items():
                        print(f"Device {device_id}: {stat['packets_per_sec']:.1f} packets/sec")

                time.sleep(0.01)  # Main loop rate

        except KeyboardInterrupt:
            print("\nShutting down...")

    # packet received

    # packet sent to right place

    # packet


if __name__ == "__main__":
    main()
