# region ======================= Imports, Constants, Enums =======================
import math
import queue
import struct
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable

import numpy as np
import serial
from vispy import app, scene


# endregion

# region ======================= Constants and Definitions =======================

# --- Serial and Packet Definitions ---
COM_PORTS: list[str] = ["/dev/cu.usbserial-023B6AC7", "/dev/cu.usbserial-023B6B29"]
PACKET_SIZE: int = 58  # Updated packet size
HEADER_BYTE: bytes = b'\xAA\xAA'
# Updated format: pos(3f), vel(3f), uwb(2f), quat(4f)
PACKET_FORMAT: str = '<HBIBB 3f 3f 2f 4f B'

# --- Bit Flagging Constants ---
PACKET_HAS_UWB_1: int = 0b00000100
PACKET_HAS_UWB_2: int = 0b00001000
PACKET_HAS_ERROR: int = 0b10000000

# --- System and Physics Constants ---
DISTANCE_BETWEEN_UWB_ANCHORS: float = 10.0  # meters


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
            quat_w=unpacked[13],
            quat_i=unpacked[14],
            quat_j=unpacked[15],
            quat_k=unpacked[16],
            error_handler=unpacked[17] if (temp_packet_flag & PACKET_HAS_ERROR) else None
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
                                buffer[:] = buffer[PACKET_SIZE:] # Consume packet
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

    def send_correction(self, relay_id: int, correction: np.ndarray):
        """Sends a position correction vector to the specified device."""
        with self.lock:
            if relay_id in self.relays and self.relays[relay_id]['serial'].is_open:
                # Message format: 'C' for Correction, followed by 3 floats (dx, dy, dz)
                message = struct.pack('<c3f', b'C', correction[0], correction[1], correction[2])
                self.relays[relay_id]['serial'].write(message)


# endregion

# region ======================= Math Helper Functions =======================

def triangulate_position(d1: float, d2: float, D: float) -> tuple[float, float]:
    """Calculates 2D coordinates from two UWB distances."""
    if d1 + d2 < D:
        return (d1 / (d1 + d2)) * D, 0.0
    x = (d1 ** 2 + D ** 2 - d2 ** 2) / (2 * D)
    y_squared = d1 ** 2 - x ** 2
    y = math.sqrt(y_squared) if y_squared >= 0 else 0.0
    return x, y


# endregion

# region ======================= Glove Classes =======================

@dataclass
class Glove:
    """Represents the state of a single glove, acting as the master reference."""
    device_id: int
    is_UWB_calibrated: bool = False
    button_state: bool = False

    # State updated directly from ESP32's dead-reckoning
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    rotation_quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))

    # Raw UWB data for correction calculation
    UWB_distance_1: float | None = None
    UWB_distance_2: float | None = None

    # Reference point for UWB coordinate system calibration
    reference_UWB_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))

    def calibrate_uwb(self):
        """Sets the current UWB position as the origin of the coordinate system."""
        if self.UWB_distance_1 is not None and self.UWB_distance_2 is not None:
            x, y = triangulate_position(self.UWB_distance_1, self.UWB_distance_2, DISTANCE_BETWEEN_UWB_ANCHORS)
            # Use the ESP32's Z position for the reference, as UWB is 2D
            self.reference_UWB_position = np.array([x, y, self.position[2]])
            self.is_UWB_calibrated = True
            print(f"Glove {self.device_id} UWB calibrated at {self.reference_UWB_position}")
        else:
            print(f"Glove {self.device_id}: UWB calibration failed. No UWB data available.")

    def update_from_packet(self, p: DevicePacket):
        """Updates the glove's state from an incoming ESP32 packet."""
        self.button_state = p.data.button_state
        self.position = np.array([p.data.pos_x, p.data.pos_y, p.data.pos_z])
        self.velocity = np.array([p.data.vel_x, p.data.vel_y, p.data.vel_z])
        self.rotation_quaternion = np.array([p.data.quat_w, p.data.quat_i, p.data.quat_j, p.data.quat_k])

        if p.data.UWB_distance_1:
            self.UWB_distance_1 = p.data.UWB_distance_1
        if p.data.UWB_distance_2:
            self.UWB_distance_2 = p.data.UWB_distance_2

    def calculate_and_send_correction(self, reader: ThreadedMultiDeviceReader, relay_id: int):
        """Calculates and sends a position correction vector if UWB data is available."""
        if not self.is_UWB_calibrated:
            # Use a button press to trigger the initial calibration
            if self.button_state:
                self.calibrate_uwb()
            return

        # Only proceed if we have fresh UWB data
        if self.UWB_distance_1 is not None and self.UWB_distance_2 is not None:
            # 1. Calculate UWB ground truth in the absolute anchor frame
            abs_uwb_x, abs_uwb_y = triangulate_position(self.UWB_distance_1, self.UWB_distance_2,
                                                        DISTANCE_BETWEEN_UWB_ANCHORS)

            # 2. Translate to the relative frame based on the calibration origin
            rel_uwb_x = abs_uwb_x - self.reference_UWB_position[0]
            rel_uwb_y = abs_uwb_y - self.reference_UWB_position[1]

            # 3. Create the ground truth vector (using the ESP32's Z-axis value)
            ground_truth_pos = np.array([rel_uwb_x, rel_uwb_y, self.position[2]])

            # 4. Calculate the correction vector (the drift)
            correction = ground_truth_pos - self.position

            # 5. Send the correction back to the ESP32
            reader.send_correction(relay_id, correction)

            # 6. Immediately apply the correction locally for smooth visualization
            self.position += correction

            # 7. Invalidate UWB data after use to prevent re-sending the same correction
            self.UWB_distance_1 = None
            self.UWB_distance_2 = None


@dataclass
class GlovePair:
    """Manages a pair of gloves and their processing thread."""
    device_ids: tuple[int, int]
    relay_id: int
    relay_queue: queue.Queue
    reader: ThreadedMultiDeviceReader
    left_hand: Glove = field(init=False)
    right_hand: Glove = field(init=False)
    running: bool = field(init=False, default=False)
    processing_thread: threading.Thread = field(init=False)

    def __post_init__(self):
        self.left_hand = Glove(device_id=self.device_ids[0])
        self.right_hand = Glove(device_id=self.device_ids[1])

    def start(self):
        """Starts the packet processing thread for this glove pair."""
        self.processing_thread = threading.Thread(target=self._packet_processor)
        self.processing_thread.daemon = True
        self.running = True
        self.processing_thread.start()

    def _packet_processor(self):
        """The core loop that processes incoming packets for this pair."""
        while self.running:
            try:
                packet = self.relay_queue.get(timeout=0.1)
                glove = self.left_hand if packet.data.device_number % 2 != 0 else self.right_hand
                glove.update_from_packet(packet)
                glove.calculate_and_send_correction(self.reader, self.relay_id)
                self.relay_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processor for relay {self.relay_id}: {e}")

    def stop(self):
        """Stops the processing thread."""
        self.running = False
        self.processing_thread.join(timeout=2.0)


# endregion

# region ======================= Main Application =======================

def main():
    """Initializes and runs the main application."""
    reader = ThreadedMultiDeviceReader()
    for i, port in enumerate(COM_PORTS, start=1):
        if not reader.add_device(relay_id=i, port=port):
            print(f"Could not connect to device on {port}. Please check connection.")
            # Decide if you want to exit or continue with fewer devices
            # return

    # Assuming one pair for now, can be expanded
    glove_pairs = []
    if 1 in reader.processing_queues:
        glove_pairs.append(GlovePair(device_ids=(1, 2), relay_id=1, reader=reader,
                                     relay_queue=reader.processing_queues[1]))
    # if 2 in reader.processing_queues:
    #     glove_pairs.append(GlovePair(device_ids=(3, 4), relay_id=2, reader=reader,
    #                                  relay_queue=reader.processing_queues[2]))

    try:
        reader.start()
        for pair in glove_pairs:
            pair.start()

        print("System running. Press Ctrl+C to stop.")

        # Example of a main loop for visualization or other tasks
        while True:
            if glove_pairs:
                pos = glove_pairs[0].left_hand.position
                print(f"Glove 1 Position: X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f}", end='\r')
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping system...")
    finally:
        for pair in glove_pairs:
            pair.stop()
        reader.stop()
        print("System stopped.")


if __name__ == "__main__":
    main()

# endregion
