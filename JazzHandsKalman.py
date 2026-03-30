# region ======================= Imports, Constants, Enums =======================
import math
import numpy as np

import serial
import struct
import queue
import time
import threading

from dataclasses import dataclass, field

from vispy import app, scene
from vispy.scene import visuals
from vispy.app import Timer

from Visualization_Tests.Visualization_Helper_Functions import setup_canvas


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

    def send_correction(self, relay_id: int, correction: np.ndarray):
        """Sends a position correction vector to the specified device."""
        with self.lock:
            if relay_id in self.relays and self.relays[relay_id]['serial'].is_open:
                # Message format: 'C' for Correction, followed by 3 floats (dx, dy, dz)
                message = struct.pack('<cb3f', b'C', relay_id.to_bytes(), correction[0], correction[1], correction[2])
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
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    rotation_matrix = np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy),     0],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx),     0],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy), 0],
        [0,               0,               0,               1]
    ])
    
    # Translation part
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = position
    
    # Combine rotation and translation
    return translation_matrix @ rotation_matrix


# endregion

# region ======================= Glove Classes =======================

@dataclass
class GloveVisual:
    """Holds the vispy visuals for a single glove."""
    axis: visuals.XYZAxis
    parent_scene: scene.SceneNode
    box: visuals.Box = field(init=False)
    section_lines: list[visuals.Line] = field(default_factory=list)

    def __post_init__(self):
        self.box = visuals.Box(width=1, height=1, depth=1, color=(1, 0.5, 0.5, 0.2), edge_color=(1, 0.5, 0.5, 1))
        self.box.transform = scene.transforms.MatrixTransform()
        self.box.parent = self.parent_scene
        self.box.visible = False

@dataclass
class Glove:
    """Represents the state of a single glove, acting as the master reference."""
    device_id: int
    active_area_x_subsections: int
    active_area_y_subsections: int
    is_UWB_calibrated: bool = False
    button_pressed: bool = False
    glove_state: int = 0
    """
    0: device off, pre-calibration
    1: drawing box
    2: box drawn, fully initialized
    """
    visual: GloveVisual | None = None
    active_area_half_extents: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    plane_normal_axis: int = 2 # Default to XY plane (Z-axis normal)
    

    # State updated directly from ESP32's dead-reckoning
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    rotation_quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))

    # Position translated into sections
    x_section:int = 0
    y_section: int = 0

    # Raw UWB data for correction calculation
    UWB_distance_1: float | None = None
    UWB_distance_2: float | None = None

    # Reference point for UWB coordinate system calibration
    reference_UWB_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    reference_orientation_quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    inverse_reference_orientation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))

    def calibrate_zero_frame(self):
        """Sets the current UWB position as the origin of the coordinate system."""
        if self.UWB_distance_1 is not None and self.UWB_distance_2 is not None:
            x, y = triangulate_position(self.UWB_distance_1, self.UWB_distance_2, DISTANCE_BETWEEN_UWB_ANCHORS)
            # Use the ESP32's Z position for the reference, as UWB is 2D
            self.reference_UWB_position = np.array([x, y, self.position[2]])
            self.reference_orientation_quaternion = self.rotation_quaternion.copy()
            self.inverse_reference_orientation = quaternion_inverse(self.reference_orientation_quaternion)
            self.is_UWB_calibrated = True
            print(f"Glove {self.device_id} UWB calibrated at {self.reference_UWB_position}")
        else:
            print(f"Glove {self.device_id}: UWB calibration failed. No UWB data available.")

    def update_from_packet(self, p: DevicePacket):
        """Updates the glove's state from an incoming ESP32 packet."""
        self.button_pressed = p.data.button_state

        # Raw data from packet
        raw_position = np.array([p.data.pos_x, p.data.pos_y, p.data.pos_z])
        raw_velocity = np.array([p.data.vel_x, p.data.vel_y, p.data.vel_z])
        raw_rotation = np.array([p.data.quat_w, p.data.quat_i, p.data.quat_j, p.data.quat_k])

        # Transform data to calibrated world frame
        self.position = rotate_vector_by_quaternion(raw_position, self.inverse_reference_orientation)
        self.velocity = rotate_vector_by_quaternion(raw_velocity, self.inverse_reference_orientation)
        self.rotation_quaternion = quaternion_multiply(self.inverse_reference_orientation, raw_rotation)

        if p.data.UWB_distance_1:
            self.UWB_distance_1 = p.data.UWB_distance_1
        if p.data.UWB_distance_2:
            self.UWB_distance_2 = p.data.UWB_distance_2

    def calculate_and_send_correction(self, reader: ThreadedMultiDeviceReader, relay_id: int):
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

    def _get_section_index(self, pos_val: float, num_sections: int, area_half_length: float) -> int:
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
                line = visuals.Line(pos=np.array([start, end]), color='cyan', parent=self.visual.parent_scene)
                line.transform = self.visual.box.transform
                self.visual.section_lines.append(line)

        if self.active_area_y_subsections > 1:
            for i in range(1, self.active_area_y_subsections):
                pos = -0.5 + i / self.active_area_y_subsections
                start, end = np.zeros(3), np.zeros(3)
                start[u_axis], start[v_axis] = -0.5, pos
                end[u_axis], end[v_axis] = 0.5, pos
                line = visuals.Line(pos=np.array([start, end]), color='cyan', parent=self.visual.parent_scene)
                line.transform = self.visual.box.transform
                self.visual.section_lines.append(line)


    def get_section_from_position(self):
        """Convert positions to sections and other instructions"""
        if self.glove_state == 2:
            plane_axes = [i for i in range(3) if i != self.plane_normal_axis]
            u_axis, v_axis = plane_axes[0], plane_axes[1]

            if self.position[self.plane_normal_axis] > 0:
                self.x_section = self._get_section_index(
                    self.position[u_axis], self.active_area_x_subsections, self.active_area_half_extents[u_axis]
                )
                self.y_section = self._get_section_index(
                    self.position[v_axis], self.active_area_y_subsections, self.active_area_half_extents[v_axis]
                )

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
                    self.plane_normal_axis = np.argmin(abs_pos)
                    
                    plane_axes = [i for i in range(3) if i != self.plane_normal_axis]
                    max_dim = max(abs_pos[plane_axes[0]], abs_pos[plane_axes[1]])
                    
                    dimensions = np.full(3, max_dim) * 2.0
                    dimensions[self.plane_normal_axis] = 0.01

                    transform = np.eye(4)
                    transform[0, 0] = dimensions[0] if dimensions[0] > 1e-6 else 1e-6
                    transform[1, 1] = dimensions[1] if dimensions[1] > 1e-6 else 1e-6
                    transform[2, 2] = dimensions[2] if dimensions[2] > 1e-6 else 1e-6
                    transform[:3, 3] = np.array([0.0, 0.0, 0.0])
                    
                    self.visual.box.transform.matrix = transform
                else:
                    self.glove_state = 2
                    abs_pos = np.abs(self.position)
                    plane_axes = [i for i in range(3) if i != self.plane_normal_axis]
                    max_extent = max(abs_pos[plane_axes[0]], abs_pos[plane_axes[1]])
                    
                    self.active_area_half_extents = np.full(3, max_extent)
                    self.active_area_half_extents[self.plane_normal_axis] = 0.0
                    
                    self._update_section_visuals()


class GloveVisualizer:
    def __init__(self, glove: Glove, title: str, color='blue'):
        self.glove = glove
        self.canvas, self.view, _, _, _, _, _ = setup_canvas()
        self.canvas.title = title
        self.glove.visual = GloveVisual(axis=visuals.XYZAxis(parent=self.view.scene), parent_scene=self.view.scene)

    def update(self):
        if self.glove.visual:
            transform_matrix = quaternion_to_transform_matrix(self.glove.rotation_quaternion, self.glove.position)
            self.glove.visual.axis.transform = scene.transforms.MatrixTransform(transform_matrix)
        self.canvas.update()

class Visualizer:
    def __init__(self, glove_pairs):
        self.glove_pairs = glove_pairs
        self.visualizers = []

        for i, pair in enumerate(self.glove_pairs):
            self.visualizers.append(GloveVisualizer(pair.left_hand, f"Glove Pair {i+1} - Left Hand"))
            self.visualizers.append(GloveVisualizer(pair.right_hand, f"Glove Pair {i+1} - Right Hand"))

        self.timer = Timer('auto', connect=self.update, start=True)
        if self.visualizers:
            self.visualizers[0].canvas.events.close.connect(self.on_close)

    def on_close(self, event):
        self.timer.stop()
        for pair in self.glove_pairs:
            pair.stop()
        if self.glove_pairs:
            self.glove_pairs[0].reader.stop()
        print("Visualization closed, system stopping.")
        app.quit()

    def update(self, event):
        for vis in self.visualizers:
            vis.update()


@dataclass
class NoteData:
    note: int # 0-127
    volume: int # 0-127
    stereo: float # between -0.5 and +0.5
    attack: int # 0-127
    instrument: str
    reverb_mode: int # 0, 1, or 2


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
    instrument: str = ""


@dataclass
class GlovePair:
    """Manages a pair of gloves and their processing thread."""
    device_ids: tuple[int, int]
    relay_id: int
    relay_queue: queue.Queue
    reader: ThreadedMultiDeviceReader
    current_octave: int = 4
    left_hand: LeftHand = field(init=False)
    right_hand: RightHand = field(init=False)
    running: bool = field(init=False, default=False)
    processing_thread: threading.Thread = field(init=False)

    def __post_init__(self):
        self.left_hand = LeftHand(device_id=self.device_ids[0], active_area_x_subsections=3, active_area_y_subsections=12)
        self.right_hand = RightHand(device_id=self.device_ids[1], active_area_x_subsections=5, active_area_y_subsections=8)

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
                glove.get_section_from_position()

                if self.left_hand.glove_state == 2 and self.right_hand.glove_state == 2:
                    self.section_to_note()
                    self.play_note()
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
        # TODO: the music playing function will go here and will accept the NoteData packet

    def section_to_note(self):
        """Get notes from the sectional data"""

        for hand in [self.left_hand, self.right_hand]:
            if hand == self.left_hand:
                if hand.button_pressed:
                    self.current_octave -= 1 if self.current_octave > 2 else 2

                hand.note = hand.y_section + self.current_octave * 12
                hand.reverb_mode = hand.x_section

            if hand == self.right_hand:
                if hand.button_pressed:
                    self.current_octave += 1 if self.current_octave < 5 else 5

                hand.stereo = hand.x_section - 2 * 0.25 # converts section number 0 to 5 to stereo -0.5 to 0.5
                hand.attack = hand.y_section * 18 # converts section number 0 to 7 to attack int 0 to 127 (approximately)


# endregion

# region ======================= Main Application =======================

def main():
    """Initializes and runs the main application."""
    reader = ThreadedMultiDeviceReader()
    for i, port in enumerate(COM_PORTS, start=1):
        if not reader.add_device(relay_id=i, port=port):
            print(f"Could not connect to device on {port}. Please check connection.")
            # return

    glove_pairs = []
    if 1 in reader.processing_queues:
        glove_pairs.append(GlovePair(device_ids=(1, 2), relay_id=1, reader=reader,
                                     relay_queue=reader.processing_queues[1]))

    reader.start()
    for pair in glove_pairs:
        pair.start()

    print("System running. Close the visualization window to stop.")
    visualizer = Visualizer(glove_pairs)
    app.run()

    print("System stopped.")


if __name__ == "__main__":
    main()

# endregion
