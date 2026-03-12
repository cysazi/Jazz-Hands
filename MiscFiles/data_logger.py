import csv
import time

# Import necessary classes and constants from JazzHands.py
from JazzHands import ThreadedMultiDeviceReader, DevicePacket, COM_PORTS

class DataLogger:
    """
    Listens to a JazzHands device and logs accelerometer and quaternion data to a CSV file.
    """
    def __init__(self, output_file: str):
        self.output_file = output_file
        self.file = open(self.output_file, 'w', newline='')
        self.writer = csv.writer(self.file)
        # Write the header row for the CSV file
        self.writer.writerow(['time', 'accel_x', 'accel_y', 'accel_z', 'quat_w', 'quat_i', 'quat_j', 'quat_k'])
        self.start_time = time.time()
        print(f"Logging data to {self.output_file}...")

    def packet_handler(self, packet: DevicePacket):
        """
        Callback function to be called by the ThreadedMultiDeviceReader.
        Writes packet data to the CSV file.
        """
        # Ensure the packet contains both acceleration and quaternion data before writing
        print("Got a packet!")
        if packet.data.accel_x is not None and packet.data.quat_w is not None and packet.data.timestamp is not None:
            self.writer.writerow([
                packet.data.timestamp,
                packet.data.accel_x,
                packet.data.accel_y,
                packet.data.accel_z,
                packet.data.quat_w,
                packet.data.quat_i,
                packet.data.quat_j,
                packet.data.quat_k
            ])

    def close(self):
        """Closes the CSV file."""
        self.file.close()
        print(f"Data saved to {self.output_file}.")

def main():
    """
    Main function to set up the data logger and device reader.
    """
    output_filename = 'accel_data.csv'
    logger = DataLogger(output_filename)

    # Initialize the device reader imported from JazzHands.py
    reader = ThreadedMultiDeviceReader()
    # Set the logger's packet handler as the callback
    reader.packet_callback_function = logger.packet_handler

    # Add the first available COM port from the list in JazzHands.py
    if COM_PORTS:
        # We'll log data from the first device (relay_id=1)
        reader.add_device(relay_id=1, port=COM_PORTS[0])
    else:
        print("No COM ports are defined in JazzHands.py. Please check your configuration.")
        logger.close()
        return

    # Start the reader thread
    reader.start()
    print("Reader started. Press Ctrl+C to stop logging.")

    try:
        # Keep the main thread alive while the reader works in the background
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Handle graceful shutdown
        print("\nStopping logger...")
        reader.stop()
        logger.close()
        print("Logger stopped.")

if __name__ == "__main__":
    main()
