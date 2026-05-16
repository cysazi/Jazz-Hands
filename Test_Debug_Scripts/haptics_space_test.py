from __future__ import annotations

import argparse
import struct
import sys
import time
import tkinter as tk

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


HAPTICS_COMMAND_HEADER = 0xCC33
ESPNOW_HAPTICS_COMMAND = struct.Struct("<HBBH")
DEFAULT_ESPNOW_BAUD = 115200
DEFAULT_DIRECT_BAUD = 115200


def auto_detect_port() -> str | None:
    if list_ports is None:
        return None
    ports = list(list_ports.comports())
    if not ports:
        return None

    preferred_keywords = (
        "usb",
        "uart",
        "cp210",
        "ch340",
        "wch",
        "silicon labs",
        "esp32",
    )
    for port_info in ports:
        text = " ".join(
            str(value).lower()
            for value in (
                port_info.device,
                port_info.description,
                port_info.manufacturer,
                port_info.hwid,
            )
            if value
        )
        if any(keyword in text for keyword in preferred_keywords):
            return str(port_info.device)
    return str(ports[0].device)


class HapticsSpaceTester:
    def __init__(
        self,
        port: str,
        baud: int,
        mode: str,
        hand: str,
        intensity: int,
        pulse_ms: int,
        repeat_ms: int,
    ):
        self.port = port
        self.baud = int(baud)
        self.mode = mode
        self.hand = hand.upper()
        self.intensity = max(0, min(int(intensity), 255))
        self.pulse_ms = max(1, min(int(pulse_ms), 1000))
        self.repeat_ms = max(10, int(repeat_ms))
        self.space_down = False
        self.after_id: str | None = None
        self.serial_port = serial.Serial(self.port, self.baud, timeout=0)

        self.root = tk.Tk()
        self.root.title("Jazz Hands Haptics Space Test")
        self.root.geometry("520x220")
        self.status_var = tk.StringVar(value="Click this window, then hold SPACE.")
        self.state_var = tk.StringVar(value="idle")
        self._build_ui()
        self.root.bind("<KeyPress-space>", self.on_space_down)
        self.root.bind("<KeyRelease-space>", self.on_space_up)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self) -> None:
        tk.Label(
            self.root,
            text="Jazz Hands Haptics Space Test",
            font=("Segoe UI", 16, "bold"),
        ).pack(pady=(18, 8))
        tk.Label(self.root, textvariable=self.status_var, font=("Segoe UI", 11)).pack()
        tk.Label(self.root, textvariable=self.state_var, font=("Segoe UI", 11)).pack(pady=(8, 0))
        tk.Label(
            self.root,
            text=f"mode={self.mode} port={self.port} baud={self.baud} hand={self.hand} "
            f"intensity={self.intensity} pulse={self.pulse_ms}ms",
            font=("Segoe UI", 9),
            fg="#555555",
        ).pack(pady=(18, 0))

    def run(self) -> None:
        print(
            f"[haptics test] connected {self.port} @ {self.baud}; "
            f"mode={self.mode}; hold SPACE to vibrate {self.hand}"
        )
        self.root.focus_force()
        self.root.mainloop()

    def on_space_down(self, _event) -> None:
        if self.space_down:
            return
        self.space_down = True
        self.status_var.set("SPACE held: vibrating")
        self._send_loop()

    def on_space_up(self, _event) -> None:
        if not self.space_down:
            return
        self.space_down = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.stop_motor()
        self.status_var.set("released: stopped")
        self.state_var.set("idle")

    def _send_loop(self) -> None:
        if not self.space_down:
            return
        self.send_pulse(self.intensity, self.pulse_ms)
        self.state_var.set(f"last pulse {time.strftime('%H:%M:%S')}")
        self.after_id = self.root.after(self.repeat_ms, self._send_loop)

    def send_pulse(self, intensity: int, duration_ms: int) -> None:
        if self.mode == "espnow":
            device_id = 1 if self.hand == "LEFT" else 2
            payload = ESPNOW_HAPTICS_COMMAND.pack(
                HAPTICS_COMMAND_HEADER,
                device_id,
                max(0, min(int(intensity), 255)),
                max(1, min(int(duration_ms), 1000)),
            )
            self.serial_port.write(payload)
            return

        line = f"P,{self.hand},{int(intensity)},{int(duration_ms)}\n"
        self.serial_port.write(line.encode("ascii"))

    def stop_motor(self) -> None:
        try:
            self.send_pulse(0, 1)
        except Exception as error:
            print(f"[haptics test] stop failed: {error}")

    def close(self) -> None:
        self.space_down = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.stop_motor()
        try:
            self.serial_port.close()
        finally:
            self.root.destroy()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hold SPACE to continuously vibrate a Jazz Hands haptics motor."
    )
    parser.add_argument(
        "--mode",
        choices=("espnow", "direct"),
        default="espnow",
        help="espnow talks to receiver_espnow_imu. direct talks to Haptics1.cpp.",
    )
    parser.add_argument("--port", help="Serial port, for example COM5. Default tries to auto-detect.")
    parser.add_argument("--baud", type=int, help="Serial baud. Defaults to mode-specific baud.")
    parser.add_argument("--hand", choices=("LEFT", "RIGHT"), default="LEFT")
    parser.add_argument("--intensity", type=int, default=180)
    parser.add_argument("--pulse-ms", type=int, default=120)
    parser.add_argument("--repeat-ms", type=int, default=45)
    parser.add_argument("--list-ports", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if serial is None:
        print("pyserial is not installed. Install it with: python -m pip install pyserial")
        return 1

    if args.list_ports:
        if list_ports is None:
            print("serial.tools.list_ports is unavailable")
            return 1
        for port_info in list_ports.comports():
            print(f"{port_info.device}: {port_info.description}")
        return 0

    port = args.port or auto_detect_port()
    if port is None:
        print("No serial port found. Re-run with --port COMx, or use --list-ports.")
        return 1

    baud = args.baud
    if baud is None:
        baud = DEFAULT_ESPNOW_BAUD if args.mode == "espnow" else DEFAULT_DIRECT_BAUD

    try:
        tester = HapticsSpaceTester(
            port=port,
            baud=baud,
            mode=args.mode,
            hand=args.hand,
            intensity=args.intensity,
            pulse_ms=args.pulse_ms,
            repeat_ms=args.repeat_ms,
        )
    except Exception as error:
        print(f"Could not open haptics serial port: {error}")
        return 1

    try:
        tester.run()
    except KeyboardInterrupt:
        tester.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
