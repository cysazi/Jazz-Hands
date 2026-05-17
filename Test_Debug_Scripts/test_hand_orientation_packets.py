"""Show fixed left/right hand meshes driven only by ESP-NOW IMU quaternions."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from jazzhands import app as jazz_app
from jazzhands.visualizer import fl_studio_debug_visualizer as fl_debug


HAND_POSITIONS = {
    "LEFT": np.array([-0.45, 0.0, 0.0], dtype=np.float64),
    "RIGHT": np.array([0.45, 0.0, 0.0], dtype=np.float64),
}

ERROR_LABELS = (
    (0b00000001, "espnow_send"),
    (0b00000010, "quat_stale"),
    (0b00000100, "accel_stale"),
    (0b00001000, "imu_missing"),
)


class HandOrientationDebugApp:
    def __init__(self, args: argparse.Namespace):
        fl_debug.configure_vispy_backend()
        self.args = args
        self.stop_event = threading.Event()
        self.reader = jazz_app.SerialImuReader(
            args.port,
            args.baud,
            args.stale_seconds,
            self.stop_event,
        )
        self.reader.start()
        self.use_correction = not args.raw
        self.use_visual_offsets = not args.no_visual_offsets
        self.selected_label = "RIGHT"
        self.signs = {
            label: values.astype(np.float64).copy()
            for label, values in jazz_app.IMU_QUATERNION_COMPONENT_SIGNS.items()
        }

        self.canvas = fl_debug.scene.SceneCanvas(
            keys="interactive",
            show=True,
            bgcolor="black",
            size=(1280, 760),
            title="Jazz Hands IMU Orientation Debug",
        )
        self.view = self.canvas.central_widget.add_view(border_color=(0.2, 0.2, 0.2, 1.0))
        self.view.camera = fl_debug.scene.cameras.TurntableCamera(
            fov=45,
            distance=2.1,
            elevation=18.0,
            azimuth=-35.0,
            center=(0, 0, 0),
        )
        fl_debug.scene.visuals.GridLines(
            scale=(0.25, 0.25),
            color=(0.25, 0.25, 0.25, 1.0),
            parent=self.view.scene,
        )
        fl_debug.scene.visuals.XYZAxis(width=2, parent=self.view.scene)
        self._add_axis_labels()
        self.hand_meshes = {}
        self._setup_hand_meshes()
        self.status_text = fl_debug.scene.visuals.Text(
            "",
            color="white",
            font_size=8,
            pos=(10, 10),
            anchor_x="left",
            anchor_y="bottom",
            parent=self.canvas.scene,
        )
        self.canvas.events.key_press.connect(self.on_key_press)
        self.canvas.events.close.connect(self.close)
        self.timer = fl_debug.Timer(
            interval=max(1.0 / max(float(args.update_hz), 1.0), 0.001),
            connect=self.update,
            start=True,
        )

    def _add_axis_labels(self) -> None:
        fl_debug.scene.visuals.Text("X", color="red", font_size=18, pos=(1.1, 0, 0), parent=self.view.scene)
        fl_debug.scene.visuals.Text("Y", color="green", font_size=18, pos=(0, 1.1, 0), parent=self.view.scene)
        fl_debug.scene.visuals.Text("Z", color="blue", font_size=18, pos=(0, 0, 1.1), parent=self.view.scene)
        fl_debug.scene.visuals.Text("LEFT", color=(0.25, 0.75, 0.95, 1), font_size=18, pos=(-0.45, -0.18, 0), parent=self.view.scene)
        fl_debug.scene.visuals.Text("RIGHT", color=(0.95, 0.45, 0.25, 1), font_size=18, pos=(0.45, -0.18, 0), parent=self.view.scene)

    def _setup_hand_meshes(self) -> None:
        colors = {
            "LEFT": (0.25, 0.75, 0.95, 0.9),
            "RIGHT": (0.95, 0.45, 0.25, 0.9),
        }
        for label in ("LEFT", "RIGHT"):
            vertices, faces, _normals, _texcoords = fl_debug.read_mesh(fl_debug.HAND_OBJ_PATHS[label])
            vertices = fl_debug.center_mesh_vertices(vertices)
            mesh = fl_debug.scene.visuals.Mesh(
                vertices=vertices,
                faces=faces,
                color=colors[label],
                shading=None,
                parent=self.view.scene,
            )
            mesh.transform = fl_debug.MatrixTransform()
            self.hand_meshes[label] = mesh

    def on_key_press(self, event) -> None:
        key = str(event.key).upper()
        if key in {"ESCAPE", "Q"}:
            self.canvas.close()
        elif key == "C":
            self.use_correction = not self.use_correction
        elif key == "V":
            self.use_visual_offsets = not self.use_visual_offsets
        elif key == "1":
            self.selected_label = "LEFT"
        elif key == "2":
            self.selected_label = "RIGHT"
        elif key in {"W", "X", "Y", "Z"}:
            index = {"W": 0, "X": 1, "Y": 2, "Z": 3}[key]
            self.signs[self.selected_label][index] *= -1.0
        elif key == "R":
            self.signs = {
                label: values.astype(np.float64).copy()
                for label, values in jazz_app.IMU_QUATERNION_COMPONENT_SIGNS.items()
            }

    def corrected_quaternion(self, label: str, raw_quaternion: np.ndarray) -> np.ndarray:
        q = fl_debug.normalize_quat(raw_quaternion)
        if self.use_correction:
            q = fl_debug.normalize_quat(q * self.signs[label])
            offset = jazz_app.IMU_MOUNTING_QUATERNION_OFFSETS[label]
            q = fl_debug.normalize_quat(fl_debug.quaternion_multiply(q, offset))
        return q

    def update(self, _event) -> None:
        lines = [
            "keys: C raw/corrected | V visual offsets | 1/2 select | W/X/Y/Z flip selected quat sign | R reset | Q quit",
            f"mode={'corrected' if self.use_correction else 'raw'} visual_offsets={'on' if self.use_visual_offsets else 'off'} selected={self.selected_label}",
        ]
        lines.extend(self.reader.status_lines())

        for label in ("LEFT", "RIGHT"):
            snapshot = self.reader.snapshot_for_hand(label)
            if snapshot is None:
                self._set_hand_transform(label, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
                lines.append(f"{label}: button=NO PACKET seq=- no fresh quaternion")
                continue

            button_text = "PRESSED" if snapshot.button_pressed else "released"
            packet_age_ms = (time.time() - snapshot.receive_time) * 1000.0
            error_text = self._format_errors(snapshot.error_handler)
            if snapshot.quaternion is None:
                self._set_hand_transform(label, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
                lines.append(
                    f"{label}: button={button_text} seq={snapshot.sequence} age={packet_age_ms:.0f}ms "
                    f"no fresh quaternion err=0x{snapshot.error_handler:02X} {error_text}"
                )
                continue

            raw_q = snapshot.quaternion
            display_q = self.corrected_quaternion(label, raw_q)
            self._set_hand_transform(label, display_q)
            raw_euler = fl_debug.quat_to_euler_deg(raw_q)
            display_euler = fl_debug.quat_to_euler_deg(display_q)
            lines.append(
                f"{label}: button={button_text} seq={snapshot.sequence} age={packet_age_ms:.0f}ms "
                f"raw_q={self._format_quat(raw_q)} raw_euler=({raw_euler[0]:+.1f},{raw_euler[1]:+.1f},{raw_euler[2]:+.1f})"
            )
            lines.append(
                f"{label}: signs={self._format_quat(self.signs[label])} "
                f"display_euler=({display_euler[0]:+.1f},{display_euler[1]:+.1f},{display_euler[2]:+.1f})"
            )

        self.status_text.text = "\n".join(lines)

    def _set_hand_transform(self, label: str, quaternion: np.ndarray) -> None:
        rotation = fl_debug.quaternion_to_rotation_matrix(quaternion)
        if self.use_visual_offsets:
            rotation = fl_debug.FRAME_MAP @ rotation @ fl_debug.FRAME_MAP.T
            rotation = rotation @ fl_debug.MODEL_OFFSET
            rotation = rotation @ fl_debug.HAND_MODEL_OFFSETS[label]
            rotation = fl_debug.HAND_VISUAL_OFFSETS[label] @ rotation

        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rotation.astype(np.float32) * fl_debug.MODEL_SCALE
        transform[3, :3] = HAND_POSITIONS[label].astype(np.float32)
        self.hand_meshes[label].transform.matrix = transform

    @staticmethod
    def _format_quat(quaternion: np.ndarray) -> str:
        return "(" + ",".join(f"{float(value):+.3f}" for value in quaternion) + ")"

    @staticmethod
    def _format_errors(error_handler: int) -> str:
        if error_handler == 0:
            return "ok"
        labels = [label for bit, label in ERROR_LABELS if error_handler & bit]
        unknown_bits = error_handler & ~sum(bit for bit, _label in ERROR_LABELS)
        if unknown_bits:
            labels.append(f"unknown=0x{unknown_bits:02X}")
        return "(" + ", ".join(labels) + ")"

    def close(self, _event=None) -> None:
        self.stop_event.set()
        if hasattr(self, "timer"):
            self.timer.stop()
        self.reader.join(timeout=0.5)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Debug ESP-NOW hand IMU packets and fixed-position hand orientation.",
    )
    parser.add_argument("--port", default=None, help="Receiver ESP32 serial port. Default auto-detects.")
    parser.add_argument("--baud", type=int, default=jazz_app.IMU_SERIAL_BAUD)
    parser.add_argument("--stale-seconds", type=float, default=jazz_app.IMU_PACKET_STALE_SECONDS)
    parser.add_argument("--update-hz", type=float, default=60.0)
    parser.add_argument("--raw", action="store_true", help="Start with raw packet quaternions instead of corrected quaternions.")
    parser.add_argument("--no-visual-offsets", action="store_true", help="Do not apply mesh/frame/model visual offsets.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    app = HandOrientationDebugApp(args)
    try:
        fl_debug.app.run()
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
