import time
from JazzHands import ThreadedMultiDeviceReader, COM_PORTS, DevicePacket


def print_packet(packet: DevicePacket) -> None:
    d = packet.data
    print(
        f"relay={packet.relay_id} dev={d.device_number} ts={d.timestamp} "
        f"acc=({d.accel_x}, {d.accel_y}, {d.accel_z}) "
        f"quat=({d.quat_w}, {d.quat_i}, {d.quat_j}, {d.quat_k}) "
        f"flags=0b{d.packet_flags:08b}"
    )


def main() -> None:
    reader = ThreadedMultiDeviceReader()
    reader.packet_callback_function = print_packet

    for relay_id, port in enumerate(COM_PORTS, start=1):
        reader.add_device(relay_id=relay_id, port=port, baudrate=115200)

    reader.start()
    print(f"Listening on {COM_PORTS} (Ctrl+C to stop)")

    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        print("Stopped.")


if __name__ == "__main__":
    main()
