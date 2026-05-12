from __future__ import annotations

import time
from typing import Callable

try:
    import serial
except ImportError:
    serial = None


class HapticsController:
    def __init__(
        self,
        port: str | None = None,
        baudrate: int = 115200,
        enabled: bool = True,
        send_func: Callable[[str, int, int], None] | None = None,
    ):
        self.port_name = port
        self.baudrate = int(baudrate)
        self.send_func = send_func
        self.enabled = bool(enabled and (port or send_func is not None))
        self.serial_port = None
        self.last_pulse_time_by_hand: dict[str, float] = {}

        if not self.enabled:
            return
        if self.send_func is not None:
            return
        if serial is None:
            print("[haptics] pyserial is not installed; haptics disabled")
            self.enabled = False
            return
        try:
            self.serial_port = serial.Serial(self.port_name, self.baudrate, timeout=0)
            print(f"[haptics] connected: {self.port_name} @ {self.baudrate}")
        except Exception as error:
            print(f"[haptics] could not open {self.port_name}: {error}")
            self.enabled = False

    def pulse(self, hand_label: str = "LEFT", intensity: int = 150, duration_ms: int = 55) -> None:
        if not self.enabled:
            return
        label = str(hand_label).upper()
        now = time.time()
        if now - self.last_pulse_time_by_hand.get(label, 0.0) < 0.03:
            return
        self.last_pulse_time_by_hand[label] = now
        if self.send_func is not None:
            try:
                self.send_func(label, int(intensity), int(duration_ms))
            except Exception as error:
                print(f"[haptics] send_func failed: {error}")
                self.enabled = False
            return
        if self.serial_port is None:
            return
        command = f"P,{label},{int(intensity)},{int(duration_ms)}\n".encode("ascii")
        try:
            self.serial_port.write(command)
        except Exception as error:
            print(f"[haptics] write failed: {error}")
            self.enabled = False

    def close(self) -> None:
        if self.serial_port is not None:
            try:
                self.serial_port.close()
            except Exception:
                pass
            self.serial_port = None
