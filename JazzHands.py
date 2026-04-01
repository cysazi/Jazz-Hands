
import math
import os
import queue
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import serial  # type: ignore[import-untyped]


def _default_com_ports() -> list[str]:
    if os.name == "nt":
        return ["COM4"]
    return ["/dev/cu.usbserial-023B6AC7", "/dev/cu.usbserial-023B6B29"]


def _load_com_ports() -> list[str]:
    env_ports = os.getenv("JAZZHANDS_COM_PORTS", "").strip()
    if env_ports:
        ports = [p.strip() for p in env_ports.split(",") if p.strip()]
        if ports:
            return ports
    return _default_com_ports()


COM_PORTS: list[str] = _load_com_ports()
NUMBER_OF_DEVICES: int = 2
DISTANCE_BETWEEN_UWB_ANCHORS: float = 10.0
TIMEOUT: float = 0.0

PACKET_FORMAT_LEGACY: str = "<HBIBB3f6f3f4fB"  # 74 bytes
PACKET_FORMAT: str = "<HBIBB3f6f4f4fB"  # 78 bytes
PACKET_SIZE_LEGACY: int = struct.calcsize(PACKET_FORMAT_LEGACY)
PACKET_SIZE: int = struct.calcsize(PACKET_FORMAT)
SUPPORTED_PACKET_SIZES: tuple[int, int] = (PACKET_SIZE_LEGACY, PACKET_SIZE)
MIN_PACKET_SIZE: int = min(SUPPORTED_PACKET_SIZES)
HEADER_BYTE: bytes = b"\xAA\xAA"

PACKET_HAS_ACCEL: int = 0b00000001
PACKET_HAS_QUAT: int = 0b00000010
PACKET_HAS_UWB_1: int = 0b00000100
PACKET_HAS_UWB_2: int = 0b00001000
PACKET_HAS_UWB_3: int = 0b00010000
PACKET_HAS_UWB_4: int = 0b00100000
PACKET_FLAG_ANCHOR_SURVEY: int = 0b01000000
PACKET_HAS_ERROR: int = 0b10000000

ENABLE_UWB_4ANCHOR_FUSION: bool = True
ENABLE_UWB_3ANCHOR_FUSION: bool = ENABLE_UWB_4ANCHOR_FUSION
UWB_RANGE_MIN_M: float = 0.05
UWB_RANGE_MAX_M: float = 20.0
UWB_TRIPLET_MAX_AGE_US: int = 2000000
UWB_SURVEY_EPS: float = 1e-4
UWB_SURVEY_BASELINE_WINDOW: int = 220
UWB_SURVEY_MIN_VALID_TRIPLES: int = 30
UWB_SURVEY_MIN_RUNTIME_US: int = 2500000
UWB_SURVEY_PERCENTILE: float = 5.0
UWB_SURVEY_MIN_MARGIN_M: float = 0.08
UWB_SURVEY_MAX_RESIDUAL_M: float = 0.25

# Startup anchor survey (anchor-to-anchor) from receiver-coordinated packets.
HOST_CMD_MAGIC: int = 0x4843
HOST_CMD_START_SURVEY: int = 1
HOST_CMD_STOP_SURVEY: int = 2
HOST_CMD_STRUCT_FORMAT: str = "<HBBHH"  # magic, cmd, session_id, step_ms, reserved
HOST_CMD_SIZE: int = struct.calcsize(HOST_CMD_STRUCT_FORMAT)

ANCHOR_SURVEY_MIN_PAIR_SAMPLES: int = 18
ANCHOR_SURVEY_MAX_PAIR_SAMPLES: int = 240
ANCHOR_SURVEY_TIMEOUT_S: float = 25.0
ANCHOR_SURVEY_DEFAULT_STEP_MS: int = 3500
ANCHOR_SURVEY_PAIR_KEYS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
)

UWB_DISTANCE_MEDIAN_WINDOW: int = 9
UWB_DISTANCE_EMA_ALPHA: float = 0.20
UWB_DISTANCE_MAX_STEP_M: float = 0.16
UWB_RANGE_INNOVATION_GATE_M: float = 0.35
UWB_POSE_EMA_ALPHA: float = 0.18
UWB_POSE_MAX_STEP_M: float = 0.15
UWB_POSE_DEADBAND_M: float = 0.006
UWB_ONLY_POS_ALPHA_XY: float = 0.22
UWB_ONLY_POS_ALPHA_Z: float = 0.18
UWB_ONLY_MAX_CORRECTION_M: float = 0.18
UWB_Z_RECOVERY_GAIN: float = 0.35
UWB_MIRROR_LOCK_Z_EPS_M: float = 0.03
UWB_MIRROR_SWITCH_MARGIN_M: float = 0.05
UWB_MIRROR_SWITCH_CONFIRM_FRAMES: int = 3

SERIAL_PARSE_DEBUG: bool = False


@dataclass
class PacketData:
    device_number: int
    timestamp: int
    packet_flags: int
    packet_size: int
    button_state: bool
    accel_x: float | None
    accel_y: float | None
    accel_z: float | None
    pos_x: float | None
    pos_y: float | None
    pos_z: float | None
    vel_x: float | None
    vel_y: float | None
    vel_z: float | None
    UWB_distance_1: float | None
    UWB_distance_2: float | None
    UWB_distance_3: float | None
    UWB_distance_4: float | None
    quat_w: float | None
    quat_i: float | None
    quat_j: float | None
    quat_k: float | None
    error_handler: int | None

    @classmethod
    def from_bytes(cls, binary_data: bytes) -> "PacketData":
        if len(binary_data) not in SUPPORTED_PACKET_SIZES:
            raise ValueError(
                f"Unsupported packet length {len(binary_data)} "
                f"(expected {PACKET_SIZE_LEGACY} or {PACKET_SIZE})"
            )

        packet_size = len(binary_data)
        if packet_size == PACKET_SIZE:
            unpacked = struct.unpack(PACKET_FORMAT, binary_data)
            uwb4 = float(unpacked[17]) if (int(unpacked[3]) & PACKET_HAS_UWB_4) else None
            quat_offset = 18
            error_index = 22
        else:
            unpacked = struct.unpack(PACKET_FORMAT_LEGACY, binary_data)
            uwb4 = None
            quat_offset = 17
            error_index = 21

        header = unpacked[0]
        if header != 0xAAAA:
            raise ValueError(f"Invalid packet header: 0x{header:04X}")

        device_number = int(unpacked[1])
        if device_number <= 0:
            raise ValueError(f"Invalid device id: {device_number}")

        flags = int(unpacked[3])
        return cls(
            device_number=device_number,
            timestamp=int(unpacked[2]),
            packet_flags=flags,
            packet_size=packet_size,
            button_state=bool(unpacked[4]),
            accel_x=float(unpacked[5]),
            accel_y=float(unpacked[6]),
            accel_z=float(unpacked[7]),
            pos_x=float(unpacked[8]),
            pos_y=float(unpacked[9]),
            pos_z=float(unpacked[10]),
            vel_x=float(unpacked[11]),
            vel_y=float(unpacked[12]),
            vel_z=float(unpacked[13]),
            UWB_distance_1=float(unpacked[14]) if (flags & PACKET_HAS_UWB_1) else None,
            UWB_distance_2=float(unpacked[15]) if (flags & PACKET_HAS_UWB_2) else None,
            UWB_distance_3=float(unpacked[16]) if (flags & PACKET_HAS_UWB_3) else None,
            UWB_distance_4=uwb4,
            quat_w=float(unpacked[quat_offset]),
            quat_i=float(unpacked[quat_offset + 1]),
            quat_j=float(unpacked[quat_offset + 2]),
            quat_k=float(unpacked[quat_offset + 3]),
            error_handler=int(unpacked[error_index]) if (flags & PACKET_HAS_ERROR) else None,
        )


@dataclass
class DevicePacket:
    relay_id: int
    data: PacketData
    timestamp: float = field(default_factory=time.time)
    sequence_num: int = 0


def detect_packet_size(buffer: bytearray, locked_size: int | None) -> int | None:
    if len(buffer) < MIN_PACKET_SIZE:
        return None

    if locked_size in SUPPORTED_PACKET_SIZES and len(buffer) >= int(locked_size):
        return int(locked_size)

    # Infer packet size from the location of the next header when possible.
    for candidate in (PACKET_SIZE_LEGACY, PACKET_SIZE):
        if len(buffer) < candidate:
            continue
        if len(buffer) >= candidate + 2:
            if buffer[candidate : candidate + 2] == HEADER_BYTE:
                return candidate
        elif len(buffer) == candidate:
            # Exactly one packet currently buffered.
            return candidate

    if len(buffer) >= PACKET_SIZE_LEGACY:
        return PACKET_SIZE_LEGACY
    return None


def build_host_command_start_survey(session_id: int, step_ms: int = ANCHOR_SURVEY_DEFAULT_STEP_MS) -> bytes:
    sid = int(session_id) & 0xFF
    step_ms_clamped = int(np.clip(step_ms, 1200, 12000))
    return struct.pack(
        HOST_CMD_STRUCT_FORMAT,
        HOST_CMD_MAGIC,
        HOST_CMD_START_SURVEY,
        sid,
        step_ms_clamped,
        0,
    )


def build_host_command_stop_survey(session_id: int) -> bytes:
    sid = int(session_id) & 0xFF
    return struct.pack(
        HOST_CMD_STRUCT_FORMAT,
        HOST_CMD_MAGIC,
        HOST_CMD_STOP_SURVEY,
        sid,
        0,
        0,
    )


class ThreadedMultiDeviceReader:
    def __init__(self):
        self.processing_queues: dict[int, queue.Queue] = {}
        self.relays: dict[int, dict] = {}
        self.threads: dict[int, threading.Thread] = {}
        self.running: bool = False
        self.lock = threading.Lock()

        self.packet_callback_function: Callable | None = None
        self.error_callback_function: Callable | None = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def add_device(self, relay_id: int, port: str, baudrate: int = 115200) -> bool:
        with self.lock:
            if relay_id in self.relays:
                print(f"Device {relay_id} already exists")
                return False

            try:
                ser = serial.Serial(port, baudrate, timeout=TIMEOUT)
                try:
                    ser.set_buffer_size(rx_size=65536, tx_size=4096)
                except Exception:
                    pass
            except serial.SerialException as exc:
                print(f"Failed to open {port}: {exc}")
                return False

            self.processing_queues[relay_id] = queue.Queue()
            self.relays[relay_id] = {
                "port": port,
                "serial": ser,
                "buffer": bytearray(),
                "packet_count": 0,
                "error_count": 0,
                "last_packet_time": None,
                "packet_size": None,
                "start_time": time.time(),
                "bytes_received": 0,
            }

        if self.running:
            self._start_thread(relay_id)

        print(f"Added device {relay_id} on {port}")
        return True

    def remove_device(self, relay_id: int):
        with self.lock:
            if relay_id not in self.relays:
                return
            try:
                self.relays[relay_id]["serial"].close()
            except serial.SerialException:
                pass
            del self.relays[relay_id]
            self.processing_queues.pop(relay_id, None)

    def send_host_command(self, relay_id: int, payload: bytes) -> bool:
        with self.lock:
            d = self.relays.get(relay_id)
            if d is None:
                return False
            ser = d["serial"]
            try:
                ser.write(payload)
                ser.flush()
                return True
            except Exception:
                return False

    def _start_thread(self, relay_id: int):
        existing = self.threads.get(relay_id)
        if existing is not None and existing.is_alive():
            return

        thread = threading.Thread(
            target=self._device_reader_thread,
            args=(relay_id,),
            name=f"Reader-Device-{relay_id}",
            daemon=True,
        )
        self.threads[relay_id] = thread
        thread.start()

    def _device_reader_thread(self, relay_id: int):
        print(f"Started thread for device {relay_id}")

        while self.running:
            with self.lock:
                device = self.relays.get(relay_id)
            if device is None:
                break

            ser = device["serial"]
            buffer: bytearray = device["buffer"]

            try:
                in_waiting = int(ser.in_waiting)
                if in_waiting <= 0:
                    time.sleep(0.001)
                    continue

                buffer.extend(ser.read(in_waiting))

                while len(buffer) >= MIN_PACKET_SIZE:
                    header_index = buffer.find(HEADER_BYTE)
                    if header_index < 0:
                        buffer.clear()
                        if SERIAL_PARSE_DEBUG:
                            print("Discarded buffer (no packet header found)")
                        break

                    if header_index > 0:
                        del buffer[:header_index]

                    packet_size = detect_packet_size(buffer, device.get("packet_size"))
                    if packet_size is None:
                        break

                    packet_bytes = bytes(buffer[:packet_size])
                    try:
                        packet_data = PacketData.from_bytes(packet_bytes)
                    except Exception as exc:
                        with self.lock:
                            d = self.relays.get(relay_id)
                            if d is not None:
                                d["error_count"] += 1
                                d["packet_size"] = None
                        if self.error_callback_function is not None:
                            self.error_callback_function(relay_id, exc)
                        del buffer[:1]
                        continue

                    with self.lock:
                        d = self.relays.get(relay_id)
                        if d is None:
                            break
                        d["packet_count"] += 1
                        d["last_packet_time"] = time.time()
                        d["bytes_received"] += packet_size
                        if d.get("packet_size") is None:
                            d["packet_size"] = packet_size
                        sequence_num = int(d["packet_count"])

                    wrapped_packet = DevicePacket(
                        relay_id=relay_id,
                        data=packet_data,
                        sequence_num=sequence_num,
                    )
                    if self.packet_callback_function is not None:
                        self.packet_callback_function(wrapped_packet)
                    self.processing_queues[relay_id].put(wrapped_packet)

                    del buffer[:packet_size]

            except Exception as exc:
                if SERIAL_PARSE_DEBUG:
                    print(f"Error in device {relay_id} thread: {exc}")
                time.sleep(0.01)

        print(f"Stopped thread for device {relay_id}")

    def start(self):
        self.running = True
        with self.lock:
            relay_ids = list(self.relays.keys())
        for relay_id in relay_ids:
            self._start_thread(relay_id)
        print(f"Started {len(relay_ids)} device threads")

    def stop(self):
        self.running = False

        for thread in list(self.threads.values()):
            thread.join(timeout=2.0)

        with self.lock:
            for device in self.relays.values():
                try:
                    device["serial"].close()
                except serial.SerialException:
                    pass

        print("All threads stopped")

    def get_stats(self) -> dict[int, dict]:
        with self.lock:
            out: dict[int, dict] = {}
            for relay_id, d in self.relays.items():
                runtime = max(1e-6, time.time() - float(d["start_time"]))
                packets = int(d["packet_count"])
                bytes_received = int(d["bytes_received"])
                out[relay_id] = {
                    "packets_received": packets,
                    "bytes_received": bytes_received,
                    "errors": int(d["error_count"]),
                    "runtime": runtime,
                    "packets_per_sec": packets / runtime,
                    "bytes_per_sec": bytes_received / runtime,
                }
            return out

    def get_device_snapshot(self, relay_id: int) -> dict | None:
        with self.lock:
            d = self.relays.get(relay_id)
            if d is None:
                return None
            return {
                "port": d["port"],
                "packet_count": int(d["packet_count"]),
                "error_count": int(d["error_count"]),
                "last_packet_time": d["last_packet_time"],
                "packet_size": d.get("packet_size"),
                "queue_size": self.processing_queues[relay_id].qsize() if relay_id in self.processing_queues else 0,
            }

    def is_device_active(self, relay_id: int, timeout: float = 1.0) -> bool:
        with self.lock:
            d = self.relays.get(relay_id)
            if d is None:
                return False
            last_time = d.get("last_packet_time")
            if last_time is None:
                return False
            return (time.time() - float(last_time)) < timeout


def get_dt_seconds(current_us: int, last_us: int) -> float:
    dt_us = (current_us - last_us) & 0xFFFFFFFF
    return dt_us / 1_000_000.0


def triangulate_position(d1: float, d2: float, D: float) -> tuple[float, float]:
    if D <= 1e-6:
        return 0.0, 0.0
    if d1 + d2 < D:
        total = max(1e-6, d1 + d2)
        return (d1 / total) * D, 0.0

    x = (d1 * d1 + D * D - d2 * d2) / (2.0 * D)
    y2 = d1 * d1 - x * x
    y = math.sqrt(y2) if y2 > 0.0 else 0.0
    return x, y


class AnchorSurveyManager:
    def __init__(self):
        self.active: bool = False
        self.locked: bool = False
        self.session_id: int = 0
        self.started_at_s: float = 0.0
        self.locked_at_s: float = 0.0

        self.pair_samples: dict[tuple[int, int], deque] = {
            pair: deque(maxlen=ANCHOR_SURVEY_MAX_PAIR_SAMPLES)
            for pair in ANCHOR_SURVEY_PAIR_KEYS
        }

        self.l12_m: float = 0.0
        self.l13_m: float = 0.0
        self.l14_m: float = 0.0
        self.l23_m: float = 0.0
        self.l24_m: float = 0.0
        self.l34_m: float = 0.0

        self.a3_x_m: float = 0.0
        self.a3_y_m: float = 0.0
        self.a4_x_m: float = 0.0
        self.a4_y_m: float = 0.0
        self.a4_z_m: float = 0.0

    def reset(self, session_id: int) -> None:
        self.active = True
        self.locked = False
        self.session_id = int(session_id) & 0xFF
        self.started_at_s = time.time()
        self.locked_at_s = 0.0
        self.l12_m = 0.0
        self.l13_m = 0.0
        self.l14_m = 0.0
        self.l23_m = 0.0
        self.l24_m = 0.0
        self.l34_m = 0.0
        self.a3_x_m = 0.0
        self.a3_y_m = 0.0
        self.a4_x_m = 0.0
        self.a4_y_m = 0.0
        self.a4_z_m = 0.0
        for d in self.pair_samples.values():
            d.clear()

    def stop(self) -> None:
        self.active = False

    @staticmethod
    def _extract_anchor_id(device_number: int) -> int | None:
        if 101 <= device_number <= 104:
            return device_number - 100
        return None

    @staticmethod
    def _robust_median(values: deque) -> float | None:
        if len(values) == 0:
            return None
        arr = np.array(values, dtype=np.float64)
        med = float(np.median(arr))
        if arr.size < 5:
            return med
        mad = float(np.median(np.abs(arr - med)))
        if mad < 1e-6:
            return med
        sigma = 1.4826 * mad
        keep = np.abs(arr - med) <= 3.0 * sigma
        if np.sum(keep) < 3:
            return med
        return float(np.median(arr[keep]))

    @staticmethod
    def _point_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _compute_anchor_points_from_baselines(
        self,
        l12: float,
        l13: float,
        l14: float,
        l23: float,
        l24: float,
        l34: float,
    ) -> tuple[bool, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        if min(l12, l13, l14, l23, l24, l34) <= UWB_SURVEY_EPS:
            return False, (
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            )

        if (l12 + l13 <= l23) or (l12 + l23 <= l13) or (l13 + l23 <= l12):
            return False, (
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            )
        if (l12 + l14 <= l24) or (l12 + l24 <= l14) or (l14 + l24 <= l12):
            return False, (
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            )
        if (l13 + l14 <= l34) or (l13 + l34 <= l14) or (l14 + l34 <= l13):
            return False, (
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            )
        if (l23 + l24 <= l34) or (l23 + l34 <= l24) or (l24 + l34 <= l23):
            return False, (
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            )

        a1 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        a2 = np.array([l12, 0.0, 0.0], dtype=np.float64)

        x3 = (l13 * l13 + l12 * l12 - l23 * l23) / (2.0 * l12)
        y3_sq = l13 * l13 - x3 * x3
        if y3_sq <= UWB_SURVEY_EPS:
            return False, (a1, a2, np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
        y3 = math.sqrt(y3_sq)
        a3 = np.array([x3, y3, 0.0], dtype=np.float64)

        x4 = (l14 * l14 + l12 * l12 - l24 * l24) / (2.0 * l12)
        y4 = (l14 * l14 - l34 * l34 + x3 * x3 + y3 * y3 - 2.0 * x3 * x4) / (2.0 * y3)
        z4_sq = l14 * l14 - x4 * x4 - y4 * y4
        if z4_sq <= UWB_SURVEY_EPS:
            return False, (a1, a2, a3, np.zeros(3, dtype=np.float64))
        z4 = math.sqrt(z4_sq)
        a4 = np.array([x4, y4, z4], dtype=np.float64)

        predicted = (
            self._point_distance(a1, a2),
            self._point_distance(a1, a3),
            self._point_distance(a1, a4),
            self._point_distance(a2, a3),
            self._point_distance(a2, a4),
            self._point_distance(a3, a4),
        )
        measured = (l12, l13, l14, l23, l24, l34)
        max_err = max(abs(p - m) for p, m in zip(predicted, measured))
        if max_err > UWB_SURVEY_MAX_RESIDUAL_M:
            return False, (a1, a2, a3, a4)

        return True, (a1, a2, a3, a4)

    def ingest_packet(self, packet: PacketData) -> None:
        if not self.active:
            return

        if not (
            packet.packet_flags & PACKET_FLAG_ANCHOR_SURVEY
            or 101 <= packet.device_number <= 104
        ):
            return

        anchor_id = self._extract_anchor_id(packet.device_number)
        if anchor_id is None:
            return

        for other_id, distance in (
            (1, packet.UWB_distance_1),
            (2, packet.UWB_distance_2),
            (3, packet.UWB_distance_3),
            (4, packet.UWB_distance_4),
        ):
            if distance is None:
                continue
            d = float(distance)
            if not np.isfinite(d) or d < UWB_RANGE_MIN_M or d > UWB_RANGE_MAX_M:
                continue
            if other_id == anchor_id:
                continue
            key = tuple(sorted((anchor_id, other_id)))
            if key in self.pair_samples:
                self.pair_samples[key].append(d)

        self._try_lock_geometry()

    def _try_lock_geometry(self) -> None:
        if self.locked:
            return

        for key in ANCHOR_SURVEY_PAIR_KEYS:
            if len(self.pair_samples[key]) < ANCHOR_SURVEY_MIN_PAIR_SAMPLES:
                return

        medians: dict[tuple[int, int], float] = {}
        for key in ANCHOR_SURVEY_PAIR_KEYS:
            value = self._robust_median(self.pair_samples[key])
            if value is None:
                return
            medians[key] = value

        l12 = medians[(1, 2)]
        l13 = medians[(1, 3)]
        l14 = medians[(1, 4)]
        l23 = medians[(2, 3)]
        l24 = medians[(2, 4)]
        l34 = medians[(3, 4)]

        ok, points = self._compute_anchor_points_from_baselines(l12, l13, l14, l23, l24, l34)
        if not ok:
            return

        a1, a2, a3, a4 = points
        self.l12_m = float(l12)
        self.l13_m = float(l13)
        self.l14_m = float(l14)
        self.l23_m = float(l23)
        self.l24_m = float(l24)
        self.l34_m = float(l34)
        self.a3_x_m = float(a3[0])
        self.a3_y_m = float(a3[1])
        self.a4_x_m = float(a4[0])
        self.a4_y_m = float(a4[1])
        self.a4_z_m = float(a4[2])
        self.locked = True
        self.locked_at_s = time.time()

    @property
    def timed_out(self) -> bool:
        return self.active and (not self.locked) and ((time.time() - self.started_at_s) > ANCHOR_SURVEY_TIMEOUT_S)

    def anchor_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([self.l12_m, 0.0, 0.0], dtype=np.float64),
            np.array([self.a3_x_m, self.a3_y_m, 0.0], dtype=np.float64),
            np.array([self.a4_x_m, self.a4_y_m, self.a4_z_m], dtype=np.float64),
        )

    def status_snapshot(self) -> dict:
        anchor_points = self.anchor_points()
        return {
            "active": self.active,
            "locked": self.locked,
            "session_id": self.session_id,
            "counts": {k: len(v) for k, v in self.pair_samples.items()},
            "l12_m": self.l12_m,
            "l13_m": self.l13_m,
            "l14_m": self.l14_m,
            "l23_m": self.l23_m,
            "l24_m": self.l24_m,
            "l34_m": self.l34_m,
            "a3_x_m": self.a3_x_m,
            "a3_y_m": self.a3_y_m,
            "a4_x_m": self.a4_x_m,
            "a4_y_m": self.a4_y_m,
            "a4_z_m": self.a4_z_m,
            "anchor_points": tuple(tuple(float(coord) for coord in point) for point in anchor_points),
            "timed_out": self.timed_out,
        }


@dataclass
class Glove:
    device_id: int

    current_packet_timestamp: int = 0
    last_packet_timestamp: int = 0
    button_state: bool = False
    error: int = 0
    last_packet_flags: int = 0
    last_packet_size: int = 0

    local_acceleration: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64))
    global_acceleration: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64))
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64))
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64))
    kalman_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64))
    kalman_velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64))
    rotation_quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))

    acceleration_history: deque = field(default_factory=lambda: deque(maxlen=10))
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=10))
    position_history: deque = field(default_factory=lambda: deque(maxlen=10))
    rotation_history: deque = field(default_factory=lambda: deque(maxlen=10))

    is_rotation_calibrated: bool = False
    is_UWB_calibrated: bool = False
    reference_quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    reference_UWB_coordinates: tuple[float, float] = (0.0, 0.0)
    reference_UWB_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64))
    reference_IMU_position_for_UWB: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64))

    UWB_distance_1: float | None = None
    UWB_distance_2: float | None = None
    UWB_distance_3: float | None = None
    UWB_distance_4: float | None = None
    UWB_distance_1_filtered: float | None = None
    UWB_distance_2_filtered: float | None = None
    UWB_distance_3_filtered: float | None = None
    UWB_distance_4_filtered: float | None = None
    UWB_1_timestamp: int | None = None
    UWB_2_timestamp: int | None = None
    UWB_3_timestamp: int | None = None
    UWB_4_timestamp: int | None = None
    uwb_d1_window: deque = field(default_factory=lambda: deque(maxlen=UWB_DISTANCE_MEDIAN_WINDOW))
    uwb_d2_window: deque = field(default_factory=lambda: deque(maxlen=UWB_DISTANCE_MEDIAN_WINDOW))
    uwb_d3_window: deque = field(default_factory=lambda: deque(maxlen=UWB_DISTANCE_MEDIAN_WINDOW))
    uwb_d4_window: deque = field(default_factory=lambda: deque(maxlen=UWB_DISTANCE_MEDIAN_WINDOW))

    uwb_survey_started: bool = False
    uwb_survey_done: bool = False
    uwb_survey_start_us: int = 0
    uwb_survey_valid_triples: int = 0
    uwb_survey_s12_samples: deque = field(default_factory=lambda: deque(maxlen=UWB_SURVEY_BASELINE_WINDOW))
    uwb_survey_s13_samples: deque = field(default_factory=lambda: deque(maxlen=UWB_SURVEY_BASELINE_WINDOW))
    uwb_survey_s14_samples: deque = field(default_factory=lambda: deque(maxlen=UWB_SURVEY_BASELINE_WINDOW))
    uwb_survey_s23_samples: deque = field(default_factory=lambda: deque(maxlen=UWB_SURVEY_BASELINE_WINDOW))
    uwb_survey_s24_samples: deque = field(default_factory=lambda: deque(maxlen=UWB_SURVEY_BASELINE_WINDOW))
    uwb_survey_s34_samples: deque = field(default_factory=lambda: deque(maxlen=UWB_SURVEY_BASELINE_WINDOW))
    uwb_survey_min12_m: float = 1e9
    uwb_survey_min13_m: float = 1e9
    uwb_survey_min14_m: float = 1e9
    uwb_survey_min23_m: float = 1e9
    uwb_survey_min24_m: float = 1e9
    uwb_survey_min34_m: float = 1e9

    uwb_baseline12_m: float = 0.0
    uwb_baseline13_m: float = 0.0
    uwb_baseline14_m: float = 0.0
    uwb_baseline23_m: float = 0.0
    uwb_baseline24_m: float = 0.0
    uwb_baseline34_m: float = 0.0
    uwb_anchor3_x_m: float = 0.0
    uwb_anchor3_y_m: float = 0.0
    uwb_anchor4_x_m: float = 0.0
    uwb_anchor4_y_m: float = 0.0
    uwb_anchor4_z_m: float = 0.0
    uwb_anchor_geometry_valid: bool = False

    uwb_pose_valid: bool = False
    uwb_absolute_pose: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64))
    uwb_relative_pose: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64))
    uwb_last_processed_triplet_ts: int = 0
    uwb_last_fused_triplet_ts: int = 0
    uwb_branch_sign: int = 0
    uwb_branch_flip_votes: int = 0

    def calibrate_zero_frame(self):
        self.reference_quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.is_rotation_calibrated = True

    def _smooth_single_range(self, raw_value: float, window: deque, prev_filtered: float | None) -> float:
        sanitized = raw_value
        if prev_filtered is not None and np.isfinite(prev_filtered):
            innovation = raw_value - prev_filtered
            if abs(innovation) > UWB_RANGE_INNOVATION_GATE_M:
                sanitized = prev_filtered + np.sign(innovation) * UWB_RANGE_INNOVATION_GATE_M

        window.append(float(sanitized))
        median_value = float(np.median(np.array(window, dtype=np.float64)))

        if prev_filtered is None:
            return median_value

        ema_value = prev_filtered + UWB_DISTANCE_EMA_ALPHA * (median_value - prev_filtered)
        step = ema_value - prev_filtered
        if abs(step) > UWB_DISTANCE_MAX_STEP_M:
            ema_value = prev_filtered + np.sign(step) * UWB_DISTANCE_MAX_STEP_M
        return float(ema_value)

    def _update_filtered_uwb_ranges(self):
        if self.UWB_distance_1 is not None:
            self.UWB_distance_1_filtered = self._smooth_single_range(
                float(self.UWB_distance_1),
                self.uwb_d1_window,
                self.UWB_distance_1_filtered,
            )
        if self.UWB_distance_2 is not None:
            self.UWB_distance_2_filtered = self._smooth_single_range(
                float(self.UWB_distance_2),
                self.uwb_d2_window,
                self.UWB_distance_2_filtered,
            )
        if self.UWB_distance_3 is not None:
            self.UWB_distance_3_filtered = self._smooth_single_range(
                float(self.UWB_distance_3),
                self.uwb_d3_window,
                self.UWB_distance_3_filtered,
            )
        if self.UWB_distance_4 is not None:
            self.UWB_distance_4_filtered = self._smooth_single_range(
                float(self.UWB_distance_4),
                self.uwb_d4_window,
                self.UWB_distance_4_filtered,
            )

    def _range_value(self, index: int) -> float | None:
        raw = getattr(self, f"UWB_distance_{index}")
        filtered = getattr(self, f"UWB_distance_{index}_filtered")
        if filtered is not None:
            return float(filtered)
        if raw is not None:
            return float(raw)
        return None

    def _timestamp_value(self, index: int) -> int | None:
        return getattr(self, f"UWB_{index}_timestamp")

    def _has_fresh_uwb_quad(self) -> bool:
        if any(self._range_value(i) is None for i in (1, 2, 3, 4)):
            return False
        if any(self._timestamp_value(i) is None for i in (1, 2, 3, 4)):
            return False
        ranges = [float(self._range_value(i)) for i in (1, 2, 3, 4)]
        if any(d < UWB_RANGE_MIN_M or d > UWB_RANGE_MAX_M for d in ranges):
            return False
        ages = [
            (self.current_packet_timestamp - int(self._timestamp_value(i))) & 0xFFFFFFFF
            for i in (1, 2, 3, 4)
        ]
        return all(age <= UWB_TRIPLET_MAX_AGE_US for age in ages)

    def _has_fresh_uwb_triplet(self) -> bool:
        if any(self._range_value(i) is None for i in (1, 2, 3)):
            return False
        if any(self._timestamp_value(i) is None for i in (1, 2, 3)):
            return False
        ranges = [float(self._range_value(i)) for i in (1, 2, 3)]
        if any(d < UWB_RANGE_MIN_M or d > UWB_RANGE_MAX_M for d in ranges):
            return False
        ages = [
            (self.current_packet_timestamp - int(self._timestamp_value(i))) & 0xFFFFFFFF
            for i in (1, 2, 3)
        ]
        return all(age <= UWB_TRIPLET_MAX_AGE_US for age in ages)

    def _get_newest_uwb_timestamp(self) -> int | None:
        timestamps = [ts for ts in (self.UWB_1_timestamp, self.UWB_2_timestamp, self.UWB_3_timestamp, self.UWB_4_timestamp) if ts is not None]
        if not timestamps:
            return None
        return max(timestamps)

    def _get_anchor_points_legacy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a1 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        a2 = np.array([self.uwb_baseline12_m, 0.0, 0.0], dtype=np.float64)
        a3 = np.array([self.uwb_anchor3_x_m, self.uwb_anchor3_y_m, 0.0], dtype=np.float64)
        return a1, a2, a3

    def _get_anchor_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([self.uwb_baseline12_m, 0.0, 0.0], dtype=np.float64),
            np.array([self.uwb_anchor3_x_m, self.uwb_anchor3_y_m, 0.0], dtype=np.float64),
            np.array([self.uwb_anchor4_x_m, self.uwb_anchor4_y_m, self.uwb_anchor4_z_m], dtype=np.float64),
        )

    def _compute_anchor_geometry_4(
        self,
        b12: float,
        b13: float,
        b14: float,
        b23: float,
        b24: float,
        b34: float,
    ) -> tuple[bool, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        if min(b12, b13, b14, b23, b24, b34) <= UWB_SURVEY_EPS:
            return False, self._get_anchor_points()

        if (b12 + b13 <= b23) or (b12 + b23 <= b13) or (b13 + b23 <= b12):
            return False, self._get_anchor_points()
        if (b12 + b14 <= b24) or (b12 + b24 <= b14) or (b14 + b24 <= b12):
            return False, self._get_anchor_points()
        if (b13 + b14 <= b34) or (b13 + b34 <= b14) or (b14 + b34 <= b13):
            return False, self._get_anchor_points()
        if (b23 + b24 <= b34) or (b23 + b34 <= b24) or (b24 + b34 <= b23):
            return False, self._get_anchor_points()

        a1 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        a2 = np.array([b12, 0.0, 0.0], dtype=np.float64)

        x3 = (b13 * b13 + b12 * b12 - b23 * b23) / (2.0 * b12)
        y3_sq = b13 * b13 - x3 * x3
        if y3_sq <= UWB_SURVEY_EPS:
            return False, self._get_anchor_points()
        y3 = math.sqrt(y3_sq)
        a3 = np.array([x3, y3, 0.0], dtype=np.float64)

        x4 = (b14 * b14 + b12 * b12 - b24 * b24) / (2.0 * b12)
        y4 = (b14 * b14 - b34 * b34 + x3 * x3 + y3 * y3 - 2.0 * x3 * x4) / (2.0 * y3)
        z4_sq = b14 * b14 - x4 * x4 - y4 * y4
        if z4_sq <= UWB_SURVEY_EPS:
            return False, self._get_anchor_points()
        z4 = math.sqrt(z4_sq)
        a4 = np.array([x4, y4, z4], dtype=np.float64)

        predicted = (
            self._point_distance(a1, a2),
            self._point_distance(a1, a3),
            self._point_distance(a1, a4),
            self._point_distance(a2, a3),
            self._point_distance(a2, a4),
            self._point_distance(a3, a4),
        )
        measured = (b12, b13, b14, b23, b24, b34)
        max_err = max(abs(p - m) for p, m in zip(predicted, measured))
        if max_err > UWB_SURVEY_MAX_RESIDUAL_M:
            return False, (a1, a2, a3, a4)

        return True, (a1, a2, a3, a4)

    def _solve_uwb_position_4(self, dists: list[float]) -> tuple[bool, np.ndarray]:
        if len(dists) != 4:
            return False, np.array([0.0, 0.0, 0.0], dtype=np.float64)
        if not self.uwb_anchor_geometry_valid:
            return False, np.array([0.0, 0.0, 0.0], dtype=np.float64)

        anchors = self._get_anchor_points()
        if any(np.linalg.norm(a) < 1e-9 for a in anchors[1:]):
            return False, np.array([0.0, 0.0, 0.0], dtype=np.float64)

        a1 = anchors[0]
        rows = []
        rhs = []
        a1_norm_sq = float(np.dot(a1, a1))
        d1 = float(dists[0])
        for anchor, di in zip(anchors[1:], dists[1:]):
            if di <= UWB_RANGE_MIN_M or di > UWB_RANGE_MAX_M:
                return False, np.array([0.0, 0.0, 0.0], dtype=np.float64)
            rows.append(2.0 * (anchor - a1))
            rhs.append(float(np.dot(anchor, anchor) - a1_norm_sq + d1 * d1 - di * di))

        A = np.array(rows, dtype=np.float64)
        b = np.array(rhs, dtype=np.float64)
        try:
            pos, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return False, np.array([0.0, 0.0, 0.0], dtype=np.float64)

        if rank < 3 or not np.all(np.isfinite(pos)):
            return False, np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # A couple of Gauss-Newton refinements keep the solve stable when ranges are noisy.
        for _ in range(3):
            residual_list = []
            jacobian_rows = []
            for anchor, di in zip(anchors, dists):
                delta = pos - anchor
                norm = float(np.linalg.norm(delta))
                if norm <= 1e-9:
                    continue
                residual_list.append(norm - float(di))
                jacobian_rows.append(delta / norm)
            if len(residual_list) < 4:
                break
            J = np.array(jacobian_rows, dtype=np.float64)
            r = np.array(residual_list, dtype=np.float64)
            try:
                step, *_ = np.linalg.lstsq(J, -r, rcond=None)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(step)):
                break
            step_norm = float(np.linalg.norm(step))
            if step_norm > 0.25:
                step = step * (0.25 / step_norm)
            pos = pos + step
            if step_norm < 1e-4:
                break

        final_residuals = np.array(
            [float(np.linalg.norm(pos - anchor) - di) for anchor, di in zip(anchors, dists)],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(final_residuals)):
            return False, np.array([0.0, 0.0, 0.0], dtype=np.float64)
        if float(np.max(np.abs(final_residuals))) > 0.45:
            return False, np.array([0.0, 0.0, 0.0], dtype=np.float64)

        return True, pos.astype(np.float64)

    def _compute_anchor_geometry_legacy(self, b12: float, b13: float, b23: float) -> tuple[bool, float, float]:
        if b12 <= UWB_SURVEY_EPS or b13 <= UWB_SURVEY_EPS or b23 <= UWB_SURVEY_EPS:
            return False, 0.0, 0.0

        if (b12 + b13 <= b23) or (b12 + b23 <= b13) or (b13 + b23 <= b12):
            return False, 0.0, 0.0

        x3 = (b13 * b13 + b12 * b12 - b23 * b23) / (2.0 * b12)
        y2 = b13 * b13 - x3 * x3
        if y2 <= UWB_SURVEY_EPS:
            return False, 0.0, 0.0
        return True, x3, math.sqrt(y2)

    def _get_anchor_points_legacy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a1 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        a2 = np.array([self.uwb_baseline12_m, 0.0, 0.0], dtype=np.float64)
        a3 = np.array([self.uwb_anchor3_x_m, self.uwb_anchor3_y_m, 0.0], dtype=np.float64)
        return a1, a2, a3

    def _select_mirrored_solution(self, pose_plus: np.ndarray, pose_minus: np.ndarray, z_mag: float) -> np.ndarray:
        has_prev = bool(self.uwb_pose_valid) and np.all(np.isfinite(self.uwb_absolute_pose))
        if not has_prev:
            has_prev = np.linalg.norm(self.uwb_absolute_pose) > 1e-9

        if not has_prev:
            if self.uwb_branch_sign == 0:
                self.uwb_branch_sign = 1
            self.uwb_branch_flip_votes = 0
            return pose_plus if self.uwb_branch_sign >= 0 else pose_minus

        target = self.uwb_absolute_pose
        d_plus = float(np.linalg.norm(pose_plus - target))
        d_minus = float(np.linalg.norm(pose_minus - target))
        preferred_sign = 1 if d_plus <= d_minus else -1

        if self.uwb_branch_sign == 0:
            self.uwb_branch_sign = preferred_sign
            self.uwb_branch_flip_votes = 0
        elif preferred_sign != self.uwb_branch_sign:
            current_d = d_plus if self.uwb_branch_sign > 0 else d_minus
            alt_d = d_minus if self.uwb_branch_sign > 0 else d_plus
            improvement = current_d - alt_d
            if z_mag >= UWB_MIRROR_LOCK_Z_EPS_M and improvement > UWB_MIRROR_SWITCH_MARGIN_M:
                self.uwb_branch_flip_votes += 1
                if self.uwb_branch_flip_votes >= UWB_MIRROR_SWITCH_CONFIRM_FRAMES:
                    self.uwb_branch_sign = preferred_sign
                    self.uwb_branch_flip_votes = 0
            else:
                self.uwb_branch_flip_votes = 0
        else:
            self.uwb_branch_flip_votes = 0

        return pose_plus if self.uwb_branch_sign >= 0 else pose_minus

    def _trilaterate_uwb_pose_legacy(self, d1: float, d2: float, d3: float) -> tuple[bool, np.ndarray]:
        if not self.uwb_anchor_geometry_valid:
            return False, np.array([0.0, 0.0, 0.0], dtype=np.float64)
        if self.uwb_baseline12_m <= UWB_SURVEY_EPS or abs(self.uwb_anchor3_y_m) <= UWB_SURVEY_EPS:
            return False, np.array([0.0, 0.0, 0.0], dtype=np.float64)

        a1, a2, a3 = self._get_anchor_points_legacy()

        ex = a2 - a1
        d = float(np.linalg.norm(ex))
        if d <= UWB_SURVEY_EPS:
            return False, np.array([0.0, 0.0, 0.0], dtype=np.float64)
        ex /= d

        a3a1 = a3 - a1
        i = float(np.dot(ex, a3a1))
        ey_raw = a3a1 - i * ex
        ey_norm = float(np.linalg.norm(ey_raw))
        if ey_norm <= UWB_SURVEY_EPS:
            return False, np.array([0.0, 0.0, 0.0], dtype=np.float64)
        ey = ey_raw / ey_norm

        ez = np.cross(ex, ey)
        ez_norm = float(np.linalg.norm(ez))
        if ez_norm <= UWB_SURVEY_EPS:
            return False, np.array([0.0, 0.0, 0.0], dtype=np.float64)
        ez /= ez_norm

        j = float(np.dot(ey, a3a1))
        if abs(j) <= UWB_SURVEY_EPS:
            return False, np.array([0.0, 0.0, 0.0], dtype=np.float64)

        x = (d1 * d1 - d2 * d2 + d * d) / (2.0 * d)
        y = (d1 * d1 - d3 * d3 + i * i + j * j - 2.0 * i * x) / (2.0 * j)
        if not np.isfinite(x) or not np.isfinite(y):
            return False, np.array([0.0, 0.0, 0.0], dtype=np.float64)

        z2_1 = d1 * d1 - x * x - y * y
        z2_2 = d2 * d2 - (x - d) * (x - d) - y * y
        z2_3 = d3 * d3 - (x - i) * (x - i) - (y - j) * (y - j)
        z2_raw = np.array([z2_1, z2_2, z2_3], dtype=np.float64)

        z2_nonneg = z2_raw[z2_raw > 0.0]
        if z2_nonneg.size > 0:
            z2_mag = float(np.mean(z2_nonneg))
        else:
            z2_mag = float(np.mean(np.abs(z2_raw))) * UWB_Z_RECOVERY_GAIN

        if not np.isfinite(z2_mag):
            return False, np.array([0.0, 0.0, 0.0], dtype=np.float64)

        z_mag = math.sqrt(max(0.0, z2_mag))
        base = a1 + x * ex + y * ey
        pose_plus = base + z_mag * ez
        pose_minus = base - z_mag * ez
        selected = self._select_mirrored_solution(pose_plus, pose_minus, z_mag)
        return True, selected.astype(np.float64)

    def _update_uwb_survey_and_pose(self):
        newest_uwb_ts = self._get_newest_uwb_timestamp()
        if newest_uwb_ts is None:
            self.uwb_pose_valid = False
            return

        if newest_uwb_ts == self.uwb_last_processed_triplet_ts:
            return
        self.uwb_last_processed_triplet_ts = newest_uwb_ts

        self._update_filtered_uwb_ranges()
        d1 = self._range_value(1)
        d2 = self._range_value(2)
        d3 = self._range_value(3)
        d4 = self._range_value(4)

        has_quad = self._has_fresh_uwb_quad()
        has_triplet = self._has_fresh_uwb_triplet()

        if self.uwb_anchor_geometry_valid:
            if has_quad and d1 is not None and d2 is not None and d3 is not None and d4 is not None:
                ok, pose = self._solve_uwb_position_4([float(d1), float(d2), float(d3), float(d4)])
                self.uwb_pose_valid = ok
                if ok:
                    prev_pose = self.uwb_absolute_pose.copy()
                    if np.linalg.norm(prev_pose) > 1e-9:
                        delta = pose - prev_pose
                        step_norm = float(np.linalg.norm(delta))
                        if step_norm > UWB_POSE_MAX_STEP_M and step_norm > 1e-9:
                            pose = prev_pose + (delta / step_norm) * UWB_POSE_MAX_STEP_M
                        pose = prev_pose + UWB_POSE_EMA_ALPHA * (pose - prev_pose)
                        residual = pose - prev_pose
                        residual[np.abs(residual) < UWB_POSE_DEADBAND_M] = 0.0
                        pose = prev_pose + residual
                    self.uwb_absolute_pose = pose
                return

            if has_triplet and d1 is not None and d2 is not None and d3 is not None:
                ok, pose = self._trilaterate_uwb_pose_legacy(float(d1), float(d2), float(d3))
                self.uwb_pose_valid = ok
                if ok:
                    prev_pose = self.uwb_absolute_pose.copy()
                    if np.linalg.norm(prev_pose) > 1e-9:
                        delta = pose - prev_pose
                        step_norm = float(np.linalg.norm(delta))
                        if step_norm > UWB_POSE_MAX_STEP_M and step_norm > 1e-9:
                            pose = prev_pose + (delta / step_norm) * UWB_POSE_MAX_STEP_M
                        pose = prev_pose + UWB_POSE_EMA_ALPHA * (pose - prev_pose)
                        residual = pose - prev_pose
                        residual[np.abs(residual) < UWB_POSE_DEADBAND_M] = 0.0
                        pose = prev_pose + residual
                    self.uwb_absolute_pose = pose
                return

        if not has_triplet or d1 is None or d2 is None or d3 is None:
            self.uwb_pose_valid = False
            return

        if not self.uwb_survey_started:
            self.uwb_survey_started = True
            self.uwb_survey_start_us = self.current_packet_timestamp

        self.uwb_survey_valid_triples += 1
        s12 = float(d1 + d2)
        s13 = float(d1 + d3)
        s23 = float(d2 + d3)
        self.uwb_survey_s12_samples.append(s12)
        self.uwb_survey_s13_samples.append(s13)
        self.uwb_survey_s23_samples.append(s23)
        self.uwb_survey_min12_m = min(self.uwb_survey_min12_m, s12)
        self.uwb_survey_min13_m = min(self.uwb_survey_min13_m, s13)
        self.uwb_survey_min23_m = min(self.uwb_survey_min23_m, s23)

        if not self.uwb_survey_done:
            elapsed_us = (self.current_packet_timestamp - self.uwb_survey_start_us) & 0xFFFFFFFF
            if (
                self.uwb_survey_valid_triples >= UWB_SURVEY_MIN_VALID_TRIPLES
                and elapsed_us >= UWB_SURVEY_MIN_RUNTIME_US
            ):
                percentile = float(np.clip(UWB_SURVEY_PERCENTILE, 1.0, 49.0))
                b12 = float(np.percentile(np.array(self.uwb_survey_s12_samples, dtype=np.float64), percentile))
                b13 = float(np.percentile(np.array(self.uwb_survey_s13_samples, dtype=np.float64), percentile))
                b23 = float(np.percentile(np.array(self.uwb_survey_s23_samples, dtype=np.float64), percentile))

                b12 = min(b12, self.uwb_survey_min12_m + UWB_SURVEY_MIN_MARGIN_M)
                b13 = min(b13, self.uwb_survey_min13_m + UWB_SURVEY_MIN_MARGIN_M)
                b23 = min(b23, self.uwb_survey_min23_m + UWB_SURVEY_MIN_MARGIN_M)

                ok, x3, y3 = self._compute_anchor_geometry_legacy(b12, b13, b23)
                if ok:
                    self.uwb_baseline12_m = b12
                    self.uwb_baseline13_m = b13
                    self.uwb_baseline23_m = b23
                    self.uwb_anchor3_x_m = x3
                    self.uwb_anchor3_y_m = y3
                    self.uwb_anchor_geometry_valid = True
                    self.uwb_survey_done = True

        if self.uwb_survey_done:
            ok, pose = self._trilaterate_uwb_pose_legacy(float(d1), float(d2), float(d3))
            self.uwb_pose_valid = ok
            if ok:
                prev_pose = self.uwb_absolute_pose.copy()
                if np.linalg.norm(prev_pose) > 1e-9:
                    delta = pose - prev_pose
                    step_norm = float(np.linalg.norm(delta))
                    if step_norm > UWB_POSE_MAX_STEP_M and step_norm > 1e-9:
                        pose = prev_pose + (delta / step_norm) * UWB_POSE_MAX_STEP_M
                    pose = prev_pose + UWB_POSE_EMA_ALPHA * (pose - prev_pose)
                    residual = pose - prev_pose
                    residual[np.abs(residual) < UWB_POSE_DEADBAND_M] = 0.0
                    pose = prev_pose + residual
                self.uwb_absolute_pose = pose
        else:
            self.uwb_pose_valid = False

    def _apply_uwb_fusion_to_direct_state(self):
        newest_uwb_ts = self._get_newest_uwb_timestamp()
        if newest_uwb_ts is None:
            return

        if not self.uwb_survey_done or not self.uwb_anchor_geometry_valid or not self.uwb_pose_valid:
            return

        if not self.is_UWB_calibrated:
            self.reference_UWB_position = self.uwb_absolute_pose.copy()
            self.reference_UWB_coordinates = (
                float(self.reference_UWB_position[0]),
                float(self.reference_UWB_position[1]),
            )
            self.reference_IMU_position_for_UWB = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            self.is_UWB_calibrated = True

        self.uwb_last_fused_triplet_ts = newest_uwb_ts
        self.uwb_relative_pose = self.uwb_absolute_pose - self.reference_UWB_position
        # UWB-only translation mode: pose is determined solely by trilateration.
        self.position = self.reference_IMU_position_for_UWB + self.uwb_relative_pose

    def integrate_function(self):
        if self.last_packet_timestamp == 0:
            return
        dt = get_dt_seconds(self.current_packet_timestamp, self.last_packet_timestamp)
        if not (1e-4 < dt < 1.0):
            return
        self.global_acceleration = self.local_acceleration.copy()
        self.velocity += self.global_acceleration * dt
        self.velocity *= 0.995
        if float(np.linalg.norm(self.velocity)) < 0.01:
            self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.position += self.velocity * dt

    def update_values(self, p: DevicePacket, survey_manager: AnchorSurveyManager | None = None):
        self.last_packet_timestamp = self.current_packet_timestamp if self.current_packet_timestamp else 0
        self.current_packet_timestamp = int(p.data.timestamp)
        self.button_state = bool(p.data.button_state)
        self.last_packet_flags = int(p.data.packet_flags)
        self.last_packet_size = int(p.data.packet_size)

        if p.data.error_handler is not None:
            self.error = int(p.data.error_handler)

        if (
            p.data.quat_w is not None
            and p.data.quat_i is not None
            and p.data.quat_j is not None
            and p.data.quat_k is not None
        ):
            q = np.array([p.data.quat_w, p.data.quat_i, p.data.quat_j, p.data.quat_k], dtype=np.float64)
            if np.all(np.isfinite(q)):
                q_norm = float(np.linalg.norm(q))
                if q_norm > 1e-6:
                    self.rotation_quaternion = q / q_norm

        if p.data.accel_x is not None and p.data.accel_y is not None and p.data.accel_z is not None:
            self.local_acceleration = np.array([p.data.accel_x, p.data.accel_y, p.data.accel_z], dtype=np.float64)
        else:
            self.local_acceleration = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        has_direct_state = (
            p.data.pos_x is not None
            and p.data.pos_y is not None
            and p.data.pos_z is not None
            and p.data.vel_x is not None
            and p.data.vel_y is not None
            and p.data.vel_z is not None
        )

        if has_direct_state:
            state = np.array(
                [p.data.pos_x, p.data.pos_y, p.data.pos_z, p.data.vel_x, p.data.vel_y, p.data.vel_z],
                dtype=np.float64,
            )
            if np.all(np.isfinite(state)):
                self.kalman_position = state[:3].copy()
                self.kalman_velocity = state[3:].copy()
                self.position = self.kalman_position.copy()
                self.velocity = self.kalman_velocity.copy()
                self.global_acceleration = self.local_acceleration.copy()
            else:
                self.integrate_function()
        else:
            self.integrate_function()

        if p.data.UWB_distance_1 is not None:
            self.UWB_distance_1 = float(p.data.UWB_distance_1)
            self.UWB_1_timestamp = self.current_packet_timestamp
        if p.data.UWB_distance_2 is not None:
            self.UWB_distance_2 = float(p.data.UWB_distance_2)
            self.UWB_2_timestamp = self.current_packet_timestamp
        if p.data.UWB_distance_3 is not None:
            self.UWB_distance_3 = float(p.data.UWB_distance_3)
            self.UWB_3_timestamp = self.current_packet_timestamp
        if p.data.UWB_distance_4 is not None:
            self.UWB_distance_4 = float(p.data.UWB_distance_4)
            self.UWB_4_timestamp = self.current_packet_timestamp

        if ENABLE_UWB_4ANCHOR_FUSION:
            if survey_manager is not None:
                if survey_manager.locked:
                    # External startup auto-survey (anchor-to-anchor packets) owns geometry.
                    self.uwb_baseline12_m = float(survey_manager.l12_m)
                    self.uwb_baseline13_m = float(survey_manager.l13_m)
                    self.uwb_baseline14_m = float(survey_manager.l14_m)
                    self.uwb_baseline23_m = float(survey_manager.l23_m)
                    self.uwb_baseline24_m = float(survey_manager.l24_m)
                    self.uwb_baseline34_m = float(survey_manager.l34_m)
                    self.uwb_anchor3_x_m = float(survey_manager.a3_x_m)
                    self.uwb_anchor3_y_m = float(survey_manager.a3_y_m)
                    self.uwb_anchor4_x_m = float(survey_manager.a4_x_m)
                    self.uwb_anchor4_y_m = float(survey_manager.a4_y_m)
                    self.uwb_anchor4_z_m = float(survey_manager.a4_z_m)
                    self.uwb_anchor_geometry_valid = True
                    self.uwb_survey_done = True
                    self.uwb_survey_started = True
                    self._update_uwb_survey_and_pose()
                    self._apply_uwb_fusion_to_direct_state()
                else:
                    if survey_manager.timed_out:
                        # Fallback path: startup auto-survey failed to lock in time.
                        self._update_uwb_survey_and_pose()
                        self._apply_uwb_fusion_to_direct_state()
                    else:
                        # Wait for startup survey lock; do not run legacy glove-derived survey yet.
                        self.uwb_pose_valid = False
            else:
                # Fallback path when no external survey manager is provided.
                self._update_uwb_survey_and_pose()
                self._apply_uwb_fusion_to_direct_state()

        self.acceleration_history.append(self.local_acceleration.copy())
        self.velocity_history.append(self.velocity.copy())
        self.position_history.append(self.position.copy())
        self.rotation_history.append(self.rotation_quaternion.copy())

    def display_error(self):
        if self.error:
            print(f"ERROR {self.error}")

    def reset_glove(self):
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.kalman_position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.kalman_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.local_acceleration = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.global_acceleration = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.rotation_quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        self.reference_IMU_position_for_UWB = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.reference_UWB_position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.reference_UWB_coordinates = (0.0, 0.0)

        self.uwb_absolute_pose = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.uwb_relative_pose = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.uwb_last_processed_triplet_ts = 0
        self.uwb_last_fused_triplet_ts = 0
        self.uwb_branch_sign = 0
        self.uwb_branch_flip_votes = 0
        self.uwb_pose_valid = False

        self.uwb_anchor_geometry_valid = False
        self.uwb_survey_started = False
        self.uwb_survey_done = False
        self.uwb_survey_start_us = 0
        self.uwb_survey_valid_triples = 0
        self.uwb_survey_s12_samples.clear()
        self.uwb_survey_s13_samples.clear()
        self.uwb_survey_s14_samples.clear()
        self.uwb_survey_s23_samples.clear()
        self.uwb_survey_s24_samples.clear()
        self.uwb_survey_s34_samples.clear()
        self.uwb_survey_min12_m = 1e9
        self.uwb_survey_min13_m = 1e9
        self.uwb_survey_min14_m = 1e9
        self.uwb_survey_min23_m = 1e9
        self.uwb_survey_min24_m = 1e9
        self.uwb_survey_min34_m = 1e9
        self.uwb_baseline12_m = 0.0
        self.uwb_baseline13_m = 0.0
        self.uwb_baseline14_m = 0.0
        self.uwb_baseline23_m = 0.0
        self.uwb_baseline24_m = 0.0
        self.uwb_baseline34_m = 0.0
        self.uwb_anchor3_x_m = 0.0
        self.uwb_anchor3_y_m = 0.0
        self.uwb_anchor4_x_m = 0.0
        self.uwb_anchor4_y_m = 0.0
        self.uwb_anchor4_z_m = 0.0

        self.UWB_distance_1 = None
        self.UWB_distance_2 = None
        self.UWB_distance_3 = None
        self.UWB_distance_4 = None
        self.UWB_distance_1_filtered = None
        self.UWB_distance_2_filtered = None
        self.UWB_distance_3_filtered = None
        self.UWB_distance_4_filtered = None
        self.UWB_1_timestamp = None
        self.UWB_2_timestamp = None
        self.UWB_3_timestamp = None
        self.UWB_4_timestamp = None
        self.uwb_d1_window.clear()
        self.uwb_d2_window.clear()
        self.uwb_d3_window.clear()
        self.uwb_d4_window.clear()

        self.acceleration_history.clear()
        self.velocity_history.clear()
        self.position_history.clear()
        self.rotation_history.clear()

        self.is_rotation_calibrated = False
        self.is_UWB_calibrated = False


@dataclass
class LeftHand(Glove):
    device_id: int
    current_note: int = field(init=False, default=0)
    current_octave: int = field(init=False, default=4)


@dataclass
class RightHand(Glove):
    device_id: int


@dataclass
class GlovePair:
    device_ids: tuple[int, int]
    relay_id: int
    instrument_type: str
    relay_queue: queue.Queue
    integration_thread: threading.Thread = field(init=False)
    left_hand: LeftHand = field(init=False)
    right_hand: RightHand = field(init=False)
    survey_manager: AnchorSurveyManager = field(init=False)
    running: bool = field(init=False, default=False)

    def __post_init__(self):
        self.left_hand = LeftHand(device_id=self.device_ids[0])
        self.right_hand = RightHand(device_id=self.device_ids[1])
        self.survey_manager = AnchorSurveyManager()

    def start(self):
        self.running = True
        self.integration_thread = threading.Thread(target=self._math_processor, daemon=True)
        self.integration_thread.start()
        return self

    def stop(self):
        self.running = False
        self.integration_thread.join(timeout=2.0)

    def _math_processor(self):
        while self.running:
            try:
                packet = self.relay_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self.process_incoming_packet(packet)
            except Exception as exc:
                print(f"Error in math processor for relay {self.relay_id}: {exc}")
            finally:
                self.relay_queue.task_done()

    def process_incoming_packet(self, packet: DevicePacket):
        if packet.relay_id != self.relay_id:
            return

        device_id = int(packet.data.device_number)
        if packet.data.packet_flags & PACKET_FLAG_ANCHOR_SURVEY:
            self.survey_manager.ingest_packet(packet.data)
            return
        if 101 <= device_id <= 104:
            self.survey_manager.ingest_packet(packet.data)
            return

        if device_id == self.left_hand.device_id:
            self.left_hand.update_values(packet, survey_manager=self.survey_manager)
            return
        if device_id == self.right_hand.device_id:
            self.right_hand.update_values(packet, survey_manager=self.survey_manager)
            return

    def main_logic(self):
        pass


def start_startup_anchor_survey(
    reader: ThreadedMultiDeviceReader,
    relay_id: int,
    step_ms: int = ANCHOR_SURVEY_DEFAULT_STEP_MS,
    session_id: int | None = None,
) -> int | None:
    sid = int(session_id if session_id is not None else (time.time_ns() & 0xFF)) & 0xFF
    payload = build_host_command_start_survey(sid, step_ms=step_ms)
    ok = reader.send_host_command(relay_id, payload)
    if not ok:
        return None
    return sid


def stop_startup_anchor_survey(
    reader: ThreadedMultiDeviceReader,
    relay_id: int,
    session_id: int,
) -> bool:
    payload = build_host_command_stop_survey(session_id)
    return reader.send_host_command(relay_id, payload)
