import queue
from typing import Callable
import serial, struct, time, threading
import numpy as np
from dataclasses import dataclass, field

# Definition of constants
COM_PORT_1: str = "insert comport here"
COM_PORT_2: str = "insert comport here"
TIMEOUT: float = 0 # Non-Blocking
PACKET_SIZE: int = 44  # Number of bytes per packet:
# Header (2) + Device Number (1) + X,Y,Z Accel (4*3 = 12) + Button State (1) + UWB (4x2=8) + Quaternion (4*4 = 16)
HEADER_BYTE: bytes = b'\xAA\xAA' # the header is two 0xAA
HEADER_SIZE: int = 2 # two the header is two bytes long
PACKET_FORMAT:str = '2B B l 3f 2f B 4f' # tells the unpacker the order of packet is 1 byte, a long (for time),  5 floats, 1 byte, then 4 floats
NUMBER_OF_DEVICES: int = 2


@dataclass
class PacketData:
    device_number:int
    accel_x: float
    accel_y: float
    accel_z: float
    UWB_distance_1: float
    UWB_distance_2: float
    button_state: bool
    quaternion: list[float]

    @classmethod
    def from_bytes(cls, binary_data: bytes) -> PacketData:
        """Parse the binary data"""
        if len(binary_data) != PACKET_SIZE:
            raise ValueError(f"Expected {PACKET_SIZE}, got {len(binary_data)}") # raise an error if wrong amount received
        unpacked = struct.unpack(PACKET_FORMAT, binary_data)
        return cls(
            device_number=unpacked[len(HEADER_BYTE)],
            accel_x=unpacked[1],
            accel_y=unpacked[2],
            accel_z=unpacked[3],
            UWB_distance_1=unpacked[4],
            UWB_distance_2=unpacked[5],
            button_state=bool(unpacked[6]),
            quaternion=[unpacked[7],unpacked[8],unpacked[9],unpacked[10]])

# NOT USED ANYMORE, HERE FOR REFERENCE
class PacketReader:
    """Initialize packet reader with all the serial port info. Call it like PacketReader(port)"""
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = TIMEOUT):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.buffer = bytearray()
        self.packet_counter: int = 0
        self.error_counter: int = 0

    def open(self)->None:
        """Open the serial port and initialize the packet reader."""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout, bytesize=8,
                                     parity=serial.PARITY_NONE)
            print(f"Serial port {self.port} opened")
        except serial.SerialException as e:
            print(f"Error opening port: {e}")
            raise

    def close(self)->None:
        """Close the serial port."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"Serial port {self.port} closed")

    def __enter__(self)-> PacketReader: # Called when "with" is used
        self.open()  # run opening logic
        return self  # update the serial port with the opened serial

    def __exit__(self)-> None: # Called when exiting "with"
        self.close() # run closing logic

    def read_packet(self) -> bytearray | None:
        """Read the packet from the serial port."""

        if self.ser and self.ser.is_open:
            # Add new data to buffer
            new_data = self.ser.read(self.ser.in_waiting or 1)
            self.buffer.extend(new_data)

            # Look for header byte anywhere in buffer
            while len(self.buffer) >= PACKET_SIZE:
                # Find the header
                try:
                    header_index = self.buffer.index(HEADER_BYTE)
                except ValueError:
                    # No header found, clear buffer and wait for more data
                    self.buffer.clear()
                    return None

                # Remove everything before header
                if header_index > 0:
                    print(f"Discarded {header_index} bytes before header")
                    self.buffer = self.buffer[header_index:]

                # Check if we have a complete packet
                if len(self.buffer) >= PACKET_SIZE:
                    packet = self.buffer[:PACKET_SIZE]
                    self.packet_counter += 1
                    self.buffer = self.buffer[PACKET_SIZE:]
                    return packet
                else:
                    # Have header but not enough bytes yet
                    return None

        return None  # Not enough data yet

class ThreadedMultiDeviceReader:
    """Manages multiple serial devices with thread-safe queuing"""
    def __init__(self, max_queue_size: int = 1000):
        # Main queue for all devices
        self.packet_queue:queue.Queue = queue.Queue(maxsize=max_queue_size)

        # Per-device queues (optional, for device-specific processing)
        self.device_queues: dict[int, queue.Queue] = {}

        # Device management
        self.devices: dict[int, dict] = {}  # device_id -> device info
        self.threads: dict[int, threading.Thread] = {}

        # Control
        self.running:bool = False
        self.stats:dict = {}  # Track packet counts, errors, etc.
        self.lock = threading.Lock()

        # Callbacks
        self.packet_callback_function: Callable | None = None
        self.error_callback_function: Callable | None = None


    def add_device(self, device_id: int, port: str, baudrate: int = 115200,
                   separate_queue: bool = False) -> bool:
        """Add a new device to monitor"""
        with self.lock:
            if device_id in self.devices:
                print(f"Device {device_id} already exists")
                return False

            try:
                # Create serial connection
                ser = serial.Serial(port, baudrate, timeout=1)

                # Store device info
                self.devices[device_id] = {
                    'port': port,
                    'serial': ser,
                    'buffer': bytearray(),
                    'packet_count': 0,
                    'error_count': 0,
                    'last_packet_time': None
                }

                # Create separate queue if requested
                if separate_queue:
                    self.device_queues[device_id] = queue.Queue(maxsize=100)

                # Initialize stats
                self.stats[device_id] = {
                    'packets_received': 0,
                    'bytes_received': 0,
                    'errors': 0,
                    'start_time': time.time()
                }

                # Start thread for this device
                thread = threading.Thread(
                    target=self._device_reader_thread,
                    args=(device_id,),
                    name=f"Reader-Device-{device_id}"
                )
                thread.daemon = True
                self.threads[device_id] = thread

                if self.running:
                    thread.start()

                print(f"Added device {device_id} on {port}")
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

    def remove_device(self, device_id: int):
        """Remove and stop monitoring a device"""
        with self.lock:
            if device_id not in self.devices:
                return

            # Close serial port
            self.devices[device_id]['serial'].close()

            # Remove from tracking
            del self.devices[device_id]
            if device_id in self.device_queues:
                del self.device_queues[device_id]

    def _device_reader_thread(self, device_id: int):
        """Thread function for reading from one device"""
        device = self.devices[device_id]
        ser = device['serial']
        buffer = device['buffer']

        print(f"Started thread for device {device_id}")

        while self.running and device_id in self.devices:
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
                                    device_id=device_id,
                                    packet=packet_data,
                                    sequence_num=device['packet_count']
                                )

                                # Put in device-specific queue if exists
                                if device_id in self.device_queues:
                                    try:
                                        self.device_queues[device_id].put(wrapped_packet, timeout=0.01)
                                    except queue.Full:
                                        print(f"Device {device_id} packet queue full! Dropping packet.")  # Drop packet for device queue

                                # Call callback if set
                                if self.packet_callback_function:
                                    self.packet_callback_function(wrapped_packet)

                                # Remove packet from buffer
                                buffer[:] = buffer[PACKET_SIZE:]

                            except Exception as e:
                                # Parse error
                                with self.lock:
                                    device['error_count'] += 1
                                    self.stats[device_id]['errors'] += 1

                                if self.error_callback_function:
                                    self.error_callback_function(device_id, e)

                                # Remove bad packet
                                buffer[:] = buffer[1:]  # Skip one byte and try again
                        else:
                            # Need more data
                            break

                else:
                    # No data available, small sleep
                    time.sleep(0.001)

            except Exception as e:
                print(f"Error in device {device_id} thread: {e}")
                time.sleep(0.1)  # Back off on error

        print(f"Stopped thread for device {device_id}")

    def start(self):
        """Start all device threads"""
        self.running = True
        for device_id, thread in self.threads.items():
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
            for device in self.devices.values():
                try:
                    device['serial'].close()
                except:
                    pass

        print("All threads stopped")

    def get_packet_from_device(self, device_id, timeout: float = 0.01) -> DevicePacket | None:
        """Get next packet from a particular device"""
        try:
            return self.device_queues[device_id].get(timeout=timeout)
        except queue.Empty:
            return None

    def get_device_packet(self, device_id: int, timeout: float = 0.01) -> DevicePacket | None:
        """Get packet from specific device queue"""
        if device_id not in self.device_queues:
            return None

        try:
            return self.device_queues[device_id].get(timeout=timeout)
        except queue.Empty:
            return None

    def get_stats(self) -> dict:
        """Get statistics for all devices"""
        with self.lock:
            stats_copy = {}
            for device_id, stats in self.stats.items():
                runtime = time.time() - stats['start_time']
                stats_copy[device_id] = {
                    **stats,
                    'runtime': runtime,
                    'packets_per_sec': stats['packets_received'] / runtime if runtime > 0 else 0,
                    'bytes_per_sec': stats['bytes_received'] / runtime if runtime > 0 else 0
                }
            return stats_copy

    def is_device_active(self, device_id: int, timeout: float = 1.0) -> bool:
        """Check if device received data recently"""
        with self.lock:
            if device_id not in self.devices:
                return False

            last_time = self.devices[device_id]['last_packet_time']
            if last_time is None:
                return False

            return (time.time() - last_time) < timeout


@dataclass
class DevicePacket:
    """Wrapper for packet with metadata"""
    device_id: int
    packet: PacketData
    timestamp: float = field(default_factory=time.time)
    sequence_num: int = 0

@dataclass
class Glove:
    """A container for the current states of an individual glove, DEVICE ID REQUIRED"""
    device_id: int
    button_state: bool
    hand: int = device_id % 2
    position = np.array([0, 0, 0])
    rotation_quaternion = np.array([0,0,0,0])
    rotation_euler = np.array([0,0,0])


def main():
    # Set up the glove objects
    gloves = []
    for i in range(NUMBER_OF_DEVICES):
        gloves.append(Glove(i))

    # Create reader
    with ThreadedMultiDeviceReader() as reader:

        # Add devices
        reader.add_device(device_id=1, port=COM_PORT_1, separate_queue=True)
        reader.add_device(device_id=2, port=COM_PORT_2, separate_queue=True)

        def on_packet(packet: DevicePacket):
            """Do stuff upon receiving a packet"""


        reader.packet_callback_function = on_packet

        # Start reading
        reader.start()

        try:
            # Main processing loop
            while True:


                # Check device health
                for device_id in [1, 2]:
                    if not reader.is_device_active(device_id, timeout=2.0):
                        print(f"Warning: Device {device_id} is not responding")

                # Print stats occasionally (for debugging)
                if int(time.time()) % 10 == 0:
                    stats = reader.get_stats()
                    for device_id, stat in stats.items():
                        print(f"Device {device_id}: {stat['packets_per_sec']:.1f} packets/sec")

                time.sleep(0.01)  # Main loop rate

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            reader.stop()


if __name__ == "__main__":
    main()