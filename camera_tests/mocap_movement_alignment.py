"""
Movement-based world-axis alignment for Jazz Hands mocap calibration.

This script uses the same camera detection, triangulation, combined OpenCV
preview, and VisPy trail style as mocap_tracker_combined_vispy.py.

Workflow:
- Start from an existing mocap_calibration.json.
- Track one visible marker.
- Capture the marker at the desired world origin.
- Move it straight upward and capture the positive Z direction.
- Move it forward and capture the positive X direction.
- Save a new calibration JSON whose camera extrinsics use the aligned world
  frame.

The saved frame is right-handed:
- +X is the captured forward direction projected onto the horizontal plane.
- +Z is the captured upward direction.
- +Y is left when looking in the +X direction.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import mocap_tracker as mocap
import mocap_tracker_combined_vispy as combined


DEFAULT_OUTPUT_PATH = Path(__file__).resolve().with_name("mocap_calibration_aligned.json")
DEFAULT_SAMPLE_FRAMES = 30
DEFAULT_MIN_AXIS_DISTANCE_M = 0.08
DEFAULT_UPDATE_HZ = 120.0
DEFAULT_TRAIL_SECONDS = 4.0
DEFAULT_PANEL_WIDTH = 640
DEFAULT_PANEL_HEIGHT = 400
DEFAULT_WINDOW_NAME = "mocap movement alignment preview"
LIVE_POINT_COLOR = (0.15, 1.00, 0.25, 1.0)
ORIGIN_COLOR = (1.00, 1.00, 1.00, 1.0)
UP_COLOR = (0.25, 0.45, 1.00, 1.0)
FORWARD_COLOR = (1.00, 0.30, 0.25, 1.0)
CAPTURE_ORDER = ("origin", "up", "forward")

app = combined.app
scene = combined.scene


@dataclass(slots=True)
class AlignmentCapture:
    label: str
    position: np.ndarray
    std_m: float
    reprojection_error_px: float
    sample_count: int
    captured_at_unix: float


@dataclass(slots=True)
class AlignmentTransform:
    origin_old_world: np.ndarray
    x_axis_old_world: np.ndarray
    y_axis_old_world: np.ndarray
    z_axis_old_world: np.ndarray
    new_from_old: np.ndarray
    old_from_new: np.ndarray


def build_arg_parser() -> argparse.ArgumentParser:
    parser = combined.build_arg_parser()
    parser.description = (
        "Align an existing mocap calibration to gravity-up and forward axes "
        "using origin/up/forward marker captures."
    )
    parser.set_defaults(
        trail_seconds=DEFAULT_TRAIL_SECONDS,
        update_hz=DEFAULT_UPDATE_HZ,
        panel_width=DEFAULT_PANEL_WIDTH,
        panel_height=DEFAULT_PANEL_HEIGHT,
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=(
            "Output calibration JSON. Defaults to mocap_calibration_aligned.json "
            "beside this script."
        ),
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=DEFAULT_SAMPLE_FRAMES,
        help="Number of tracked frames to average for each origin/up/forward capture.",
    )
    parser.add_argument(
        "--min-axis-distance",
        type=float,
        default=DEFAULT_MIN_AXIS_DISTANCE_M,
        help="Minimum origin-to-up and horizontal origin-to-forward distance in meters.",
    )
    return parser


def load_raw_calibration(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict):
        return data
    return {
        "created_at_unix": time.time(),
        "source": "converted list calibration",
        "cameras": data,
    }


def rotation_matrix_to_euler_xyz_deg(rotation: np.ndarray) -> list[float]:
    sy = math.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rotation[2, 1], rotation[2, 2])
        y = math.atan2(-rotation[2, 0], sy)
        z = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        x = math.atan2(-rotation[1, 2], rotation[1, 1])
        y = math.atan2(-rotation[2, 0], sy)
        z = 0.0

    return [math.degrees(value) for value in (x, y, z)]


def normalize(vector: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError(f"{label} is too small to normalize.")
    return vector / norm


def compute_alignment_transform(
    captures: dict[str, AlignmentCapture],
    min_axis_distance_m: float,
) -> AlignmentTransform:
    missing = [label for label in CAPTURE_ORDER if label not in captures]
    if missing:
        raise ValueError(f"Missing captures: {', '.join(missing)}")

    origin = captures["origin"].position
    up_vector = captures["up"].position - origin
    up_distance = float(np.linalg.norm(up_vector))
    if up_distance < min_axis_distance_m:
        raise ValueError(
            f"Up capture is only {up_distance:.3f}m from origin; "
            f"move at least {min_axis_distance_m:.3f}m upward."
        )
    z_axis = normalize(up_vector, "up axis")

    forward_vector = captures["forward"].position - origin
    forward_horizontal = forward_vector - float(np.dot(forward_vector, z_axis)) * z_axis
    forward_distance = float(np.linalg.norm(forward_horizontal))
    if forward_distance < min_axis_distance_m:
        raise ValueError(
            f"Forward capture has only {forward_distance:.3f}m of horizontal travel; "
            f"move at least {min_axis_distance_m:.3f}m forward."
        )
    x_axis = normalize(forward_horizontal, "forward axis")
    y_axis = normalize(np.cross(z_axis, x_axis), "left axis")

    old_from_new = np.column_stack((x_axis, y_axis, z_axis)).astype(np.float64)
    determinant = float(np.linalg.det(old_from_new))
    if determinant < 0.0:
        y_axis = -y_axis
        old_from_new = np.column_stack((x_axis, y_axis, z_axis)).astype(np.float64)

    new_from_old = old_from_new.T
    return AlignmentTransform(
        origin_old_world=origin.astype(np.float64),
        x_axis_old_world=x_axis.astype(np.float64),
        y_axis_old_world=y_axis.astype(np.float64),
        z_axis_old_world=z_axis.astype(np.float64),
        new_from_old=new_from_old.astype(np.float64),
        old_from_new=old_from_new.astype(np.float64),
    )


def transform_calibration_item(
    item: dict,
    calibration: mocap.CameraCalibration,
    transform: AlignmentTransform,
) -> dict:
    old_rotation = calibration.rotation.astype(np.float64)
    old_translation = calibration.translation.reshape(3).astype(np.float64)

    new_rotation = old_rotation @ transform.old_from_new
    new_translation = old_rotation @ transform.origin_old_world + old_translation
    old_camera_position = -old_rotation.T @ old_translation
    new_camera_position = transform.new_from_old @ (
        old_camera_position - transform.origin_old_world
    )
    rvec, _jacobian = mocap.cv2.Rodrigues(new_rotation)

    output = dict(item)
    output["rotation"] = new_rotation.tolist()
    output["translation"] = new_translation.reshape(3).tolist()
    output["rvec"] = rvec.reshape(3).tolist()
    output["tvec"] = new_translation.reshape(3).tolist()
    output["position"] = new_camera_position.reshape(3).tolist()
    output["euler_xyz_deg"] = rotation_matrix_to_euler_xyz_deg(new_rotation)
    return output


def capture_to_json(capture: AlignmentCapture) -> dict:
    return {
        "position_old_world_m": capture.position.reshape(3).tolist(),
        "std_m": capture.std_m,
        "reprojection_error_px": capture.reprojection_error_px,
        "sample_count": capture.sample_count,
        "captured_at_unix": capture.captured_at_unix,
    }


def write_aligned_calibration(
    input_path: Path,
    output_path: Path,
    raw_calibration: dict,
    calibrations: dict[int, mocap.CameraCalibration],
    captures: dict[str, AlignmentCapture],
    min_axis_distance_m: float,
) -> None:
    transform = compute_alignment_transform(captures, min_axis_distance_m)
    output = dict(raw_calibration)
    output["created_at_unix"] = time.time()
    output["source"] = "Camera_Tests/mocap_movement_alignment.py"
    output["aligned_from"] = str(input_path)
    output["world_axes"] = {
        "units": "meters",
        "origin": "origin capture",
        "x_positive": "forward capture projected onto the horizontal plane",
        "y_positive": "left when looking along +X",
        "z_positive": "up capture",
        "right_handed": True,
    }
    output["movement_alignment"] = {
        "captures": {label: capture_to_json(captures[label]) for label in CAPTURE_ORDER},
        "origin_old_world_m": transform.origin_old_world.reshape(3).tolist(),
        "axes_old_world": {
            "x": transform.x_axis_old_world.reshape(3).tolist(),
            "y": transform.y_axis_old_world.reshape(3).tolist(),
            "z": transform.z_axis_old_world.reshape(3).tolist(),
        },
        "new_from_old": transform.new_from_old.tolist(),
        "old_from_new": transform.old_from_new.tolist(),
    }

    transformed_cameras = []
    for item in raw_calibration.get("cameras", []):
        camera_id = int(item.get("id", item.get("camera_id")))
        calibration = calibrations.get(camera_id)
        if calibration is None:
            transformed_cameras.append(dict(item))
            continue
        transformed_cameras.append(transform_calibration_item(item, calibration, transform))
    output["cameras"] = transformed_cameras

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
        file.write("\n")
    temp_path.replace(output_path)


def format_point(position: np.ndarray) -> str:
    x, y, z = position
    return f"x={x:+.3f} y={y:+.3f} z={z:+.3f}m"


class MovementAlignmentApp:
    def __init__(
        self,
        args: argparse.Namespace,
        sources: list[mocap.CameraSource],
        calibrations: dict[int, mocap.CameraCalibration],
        raw_calibration: dict,
        input_path: Path,
        output_path: Path,
    ):
        if scene is None or app is None:
            raise RuntimeError("VisPy is not available.")

        self.args = args
        self.sources = sources
        self.calibrations = calibrations
        self.raw_calibration = raw_calibration
        self.input_path = input_path
        self.output_path = output_path
        self.preview_camera_ids = [source.camera_id for source in sources[:2]]
        self.settings_by_camera = {
            source.camera_id: mocap.build_detection_settings(args, source.camera_id)
            for source in sources
        }
        self.detectors = {
            camera_id: mocap.ReflectiveMarkerDetector(settings)
            for camera_id, settings in self.settings_by_camera.items()
        }
        self.triangulator = mocap.MultiCameraTriangulator(
            calibrations=calibrations,
            max_pair_error_px=args.max_reprojection_error,
            cluster_distance_m=args.cluster_distance,
            room_bounds=args.room_bounds,
        )
        self.tracker = mocap.MarkerTracker(
            max_match_distance_m=args.track_distance,
            max_missing_frames=args.max_missing_frames,
            min_confirmed_hits=args.track_confirmation_hits,
            max_tentative_missing_frames=args.tentative_max_missing_frames,
            duplicate_track_distance_m=args.duplicate_track_distance,
            velocity_damping=args.velocity_damping,
            stationary_distance_m=args.stationary_distance,
            max_prediction_dt=args.max_prediction_dt,
        )
        self.trail_points: list[tuple[float, np.ndarray]] = []
        self.captures: dict[str, AlignmentCapture] = {}
        self.pending_label: str | None = None
        self.pending_samples: list[tuple[np.ndarray, float]] = []
        self.current_track: mocap.MarkerTrack | None = None
        self.last_print_time = 0.0
        self.last_save_error: str | None = None
        self.closed = False

        self._setup_combined_preview_window()
        self._setup_vispy_window()
        self.timer = app.Timer(
            interval=max(1.0 / max(float(args.update_hz), 1.0), 0.001),
            connect=self.update,
            start=True,
        )

    def _setup_combined_preview_window(self) -> None:
        if self.args.no_preview:
            return
        mocap.cv2.namedWindow(DEFAULT_WINDOW_NAME, mocap.cv2.WINDOW_NORMAL)
        mocap.cv2.resizeWindow(
            DEFAULT_WINDOW_NAME,
            int(self.args.panel_width) * 2,
            int(self.args.panel_height) * 2,
        )

    def _setup_vispy_window(self) -> None:
        self.canvas = scene.SceneCanvas(
            keys="interactive",
            show=True,
            bgcolor="black",
            size=(1000, 750),
            title="Jazz Hands movement alignment",
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(
            fov=45,
            distance=4.0,
            elevation=25.0,
            azimuth=-35.0,
            center=(0, 0, 0),
        )
        scene.visuals.GridLines(
            scale=(0.25, 0.25),
            color=(0.25, 0.25, 0.25, 1.0),
            parent=self.view.scene,
        )
        scene.visuals.XYZAxis(width=2, parent=self.view.scene)
        self._add_axis_labels()
        self._add_camera_markers()

        self.trail_line = scene.visuals.Line(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(LIVE_POINT_COLOR[0], LIVE_POINT_COLOR[1], LIVE_POINT_COLOR[2], 0.0),
            width=float(self.args.line_width),
            parent=self.view.scene,
            method="gl",
        )
        self.live_marker = scene.visuals.Markers(parent=self.view.scene)
        self.live_marker.set_data(pos=np.zeros((0, 3), dtype=np.float32))
        self.capture_markers = scene.visuals.Markers(parent=self.view.scene)
        self.capture_markers.set_data(pos=np.zeros((0, 3), dtype=np.float32))
        self.axis_lines = scene.visuals.Line(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(1.0, 1.0, 1.0, 0.0),
            width=3.0,
            parent=self.view.scene,
            method="gl",
            connect="segments",
        )
        self.status_text = scene.visuals.Text(
            self._status_message(),
            color="white",
            font_size=9,
            pos=(10, 10),
            anchor_x="left",
            anchor_y="bottom",
            parent=self.canvas.scene,
        )
        self.canvas.events.close.connect(self.close)
        self.canvas.events.key_press.connect(self._handle_vispy_key)

    def _add_axis_labels(self) -> None:
        label_distance = 1.25 * self._position_scale()
        scene.visuals.Text("X", color="red", font_size=20, pos=(label_distance, 0, 0), parent=self.view.scene)
        scene.visuals.Text("Y", color="green", font_size=20, pos=(0, label_distance, 0), parent=self.view.scene)
        scene.visuals.Text("Z", color="blue", font_size=20, pos=(0, 0, label_distance), parent=self.view.scene)

    def _add_camera_markers(self) -> None:
        positions = []
        labels = []
        for camera_id, calibration in sorted(self.calibrations.items()):
            camera_position = -calibration.rotation.T @ calibration.translation.reshape(3)
            visual_position = self._visual_position(camera_position)
            positions.append(visual_position)
            labels.append((camera_id, visual_position))

        if not positions:
            return

        camera_points = scene.visuals.Markers(parent=self.view.scene)
        camera_points.set_data(
            pos=np.asarray(positions, dtype=np.float32),
            face_color=(0.2, 0.55, 1.0, 1.0),
            edge_color=(0.85, 0.95, 1.0, 1.0),
            size=8.0,
        )
        for camera_id, position in labels:
            scene.visuals.Text(
                f"cam {camera_id}",
                color=(0.75, 0.9, 1.0, 1.0),
                font_size=8,
                pos=tuple(position),
                parent=self.view.scene,
            )

    def _visual_position(self, position: np.ndarray) -> np.ndarray:
        return position.astype(np.float32) * self._position_scale()

    def _position_scale(self) -> float:
        return float(
            getattr(
                self.args,
                "position_scale",
                getattr(self.args, "scaling_factor", combined.SCALING_FACTOR),
            )
        )

    def update(self, _event) -> None:
        if self.closed:
            return

        timestamp = time.time()
        frames: dict[int, np.ndarray] = {}
        observations_by_camera: dict[int, list[mocap.MarkerObservation]] = {}

        for source in self.sources:
            ok, frame = source.read()
            if not ok or frame is None:
                observations_by_camera[source.camera_id] = []
                continue

            frames[source.camera_id] = frame
            observations_by_camera[source.camera_id] = self.detectors[source.camera_id].detect(
                frame,
                source.camera_id,
                timestamp,
            )

        mocap.lock_observations_to_existing_tracks(
            frames,
            observations_by_camera,
            self.tracker.tracks,
            self.settings_by_camera,
            self.args.track_memory_pixels,
            timestamp,
        )
        measurements = self.triangulator.triangulate(observations_by_camera)
        tracks = self.tracker.update(measurements, timestamp)
        live_tracks = [
            track for track in tracks if track.confirmed and track.missing_frames == 0
        ]
        self.current_track = max(live_tracks, key=lambda track: track.confidence, default=None)

        if self.current_track is not None:
            self.trail_points.append((timestamp, self.current_track.position.copy()))
            self._collect_pending_sample(self.current_track)

        self._draw_vispy(timestamp, observations_by_camera)
        self._print_status(timestamp, tracks, observations_by_camera)

        if not self.args.no_preview:
            combined_preview = combined.build_combined_preview(
                self.preview_camera_ids,
                frames,
                observations_by_camera,
                tracks,
                self.settings_by_camera,
                self.args.track_memory_pixels,
                int(self.args.panel_width),
                int(self.args.panel_height),
            )
            self._draw_preview_overlay(combined_preview)
            mocap.cv2.imshow(DEFAULT_WINDOW_NAME, combined_preview)
            key = mocap.cv2.waitKey(1) & 0xFF
            self._handle_key(key)

    def _collect_pending_sample(self, track: mocap.MarkerTrack) -> None:
        if self.pending_label is None:
            return
        self.pending_samples.append((track.position.copy(), track.reprojection_error_px))
        if len(self.pending_samples) >= max(int(self.args.sample_frames), 1):
            self._finish_capture()

    def _finish_capture(self) -> None:
        if self.pending_label is None or not self.pending_samples:
            return

        positions = np.asarray([sample[0] for sample in self.pending_samples], dtype=np.float64)
        errors = np.asarray([sample[1] for sample in self.pending_samples], dtype=np.float64)
        position = np.mean(positions, axis=0)
        deviations = np.linalg.norm(positions - position.reshape(1, 3), axis=1)
        capture = AlignmentCapture(
            label=self.pending_label,
            position=position,
            std_m=float(np.std(deviations)),
            reprojection_error_px=float(np.mean(errors)),
            sample_count=len(self.pending_samples),
            captured_at_unix=time.time(),
        )
        self.captures[self.pending_label] = capture
        print(
            f"[alignment] captured {self.pending_label}: "
            f"{format_point(capture.position)} std={capture.std_m:.4f}m "
            f"err={capture.reprojection_error_px:.1f}px"
        )
        self.pending_label = None
        self.pending_samples = []
        self.last_save_error = None

    def _draw_vispy(
        self,
        timestamp: float,
        observations_by_camera: dict[int, list[mocap.MarkerObservation]],
    ) -> None:
        self.trail_points = [
            (point_time, point)
            for point_time, point in self.trail_points
            if timestamp - point_time <= self.args.trail_seconds
        ]
        if self.trail_points:
            positions = np.asarray(
                [self._visual_position(position) for _point_time, position in self.trail_points],
                dtype=np.float32,
            )
            ages = np.asarray(
                [timestamp - point_time for point_time, _position in self.trail_points],
                dtype=np.float32,
            )
            alphas = np.clip(1.0 - ages / max(float(self.args.trail_seconds), 0.001), 0.05, 1.0)
            colors = np.tile(np.asarray(LIVE_POINT_COLOR, dtype=np.float32), (len(positions), 1))
            colors[:, 3] = alphas

            if len(positions) >= 2:
                self.trail_line.set_data(
                    pos=positions,
                    color=colors,
                    width=float(self.args.line_width),
                )
            self.live_marker.set_data(
                pos=positions[-1:],
                face_color=LIVE_POINT_COLOR,
                edge_color=(1.0, 1.0, 1.0, 1.0),
                size=float(self.args.point_size),
            )
        else:
            self.trail_line.set_data(
                pos=np.zeros((1, 3), dtype=np.float32),
                color=(LIVE_POINT_COLOR[0], LIVE_POINT_COLOR[1], LIVE_POINT_COLOR[2], 0.0),
                width=float(self.args.line_width),
            )
            self.live_marker.set_data(pos=np.zeros((0, 3), dtype=np.float32))

        self._draw_capture_markers()
        self.status_text.text = self._status_message(observations_by_camera)

    def _draw_capture_markers(self) -> None:
        if not self.captures:
            self.capture_markers.set_data(pos=np.zeros((0, 3), dtype=np.float32))
            self.axis_lines.set_data(
                pos=np.zeros((1, 3), dtype=np.float32),
                color=(1.0, 1.0, 1.0, 0.0),
                width=3.0,
                connect="segments",
            )
            return

        colors_by_label = {
            "origin": ORIGIN_COLOR,
            "up": UP_COLOR,
            "forward": FORWARD_COLOR,
        }
        positions = []
        colors = []
        for label in CAPTURE_ORDER:
            capture = self.captures.get(label)
            if capture is None:
                continue
            positions.append(self._visual_position(capture.position))
            colors.append(colors_by_label[label])

        self.capture_markers.set_data(
            pos=np.asarray(positions, dtype=np.float32),
            face_color=np.asarray(colors, dtype=np.float32),
            edge_color=(1.0, 1.0, 1.0, 1.0),
            size=float(self.args.point_size) * 1.25,
        )

        if "origin" not in self.captures:
            return
        origin = self._visual_position(self.captures["origin"].position)
        segments = []
        segment_colors = []
        for label, color in (("forward", FORWARD_COLOR), ("up", UP_COLOR)):
            capture = self.captures.get(label)
            if capture is None:
                continue
            segments.extend([origin, self._visual_position(capture.position)])
            segment_colors.extend([color, color])

        if segments:
            self.axis_lines.set_data(
                pos=np.asarray(segments, dtype=np.float32),
                color=np.asarray(segment_colors, dtype=np.float32),
                width=3.0,
                connect="segments",
            )

    def _draw_preview_overlay(self, image: np.ndarray) -> None:
        lines = [
            self._short_status_message(),
            "space/c: capture next | 1 origin | 2 up | 3 forward | s save | r reset | q quit",
        ]
        y = 34
        for line in lines:
            mocap.cv2.putText(
                image,
                line,
                (12, y),
                mocap.cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 0, 0),
                3,
                mocap.cv2.LINE_AA,
            )
            mocap.cv2.putText(
                image,
                line,
                (12, y),
                mocap.cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                1,
                mocap.cv2.LINE_AA,
            )
            y += 28

    def _handle_key(self, key: int) -> None:
        if key in (255, -1):
            return
        if key in (ord("q"), 27):
            self.canvas.close()
        elif key in (ord("c"), ord(" ")):
            self._start_capture(self._next_capture_label())
        elif key == ord("1"):
            self._start_capture("origin")
        elif key == ord("2"):
            self._start_capture("up")
        elif key == ord("3"):
            self._start_capture("forward")
        elif key == ord("r"):
            self._reset_captures()
        elif key == ord("s"):
            self._save_alignment()

    def _handle_vispy_key(self, event) -> None:
        text = getattr(event, "text", "") or ""
        if text:
            self._handle_key(ord(text.lower()[0]))
            return

        key = getattr(event, "key", None)
        key_name = getattr(key, "name", str(key)).lower()
        if key_name in ("escape", "esc"):
            self._handle_key(27)
        elif key_name == "space":
            self._handle_key(ord(" "))

    def _next_capture_label(self) -> str:
        for label in CAPTURE_ORDER:
            if label not in self.captures:
                return label
        return CAPTURE_ORDER[-1]

    def _start_capture(self, label: str) -> None:
        if label not in CAPTURE_ORDER:
            return
        self.pending_label = label
        self.pending_samples = []
        self.last_save_error = None
        print(
            f"[alignment] collecting {self.args.sample_frames} frames for {label}. "
            "Keep the marker still."
        )

    def _reset_captures(self) -> None:
        self.captures.clear()
        self.pending_label = None
        self.pending_samples = []
        self.last_save_error = None
        print("[alignment] reset captures")

    def _save_alignment(self) -> None:
        try:
            write_aligned_calibration(
                self.input_path,
                self.output_path,
                self.raw_calibration,
                self.calibrations,
                self.captures,
                self.args.min_axis_distance,
            )
        except ValueError as error:
            self.last_save_error = str(error)
            print(f"[alignment] cannot save: {error}")
            return

        self.last_save_error = None
        print(f"[alignment] saved aligned calibration to {self.output_path}")

    def _status_message(
        self,
        observations_by_camera: dict[int, list[mocap.MarkerObservation]] | None = None,
    ) -> str:
        parts = [self._short_status_message()]
        if observations_by_camera is not None:
            blob_counts = ", ".join(
                f"cam {camera_id}: {len(observations)}"
                for camera_id, observations in sorted(observations_by_camera.items())
            )
            parts.append(blob_counts)
        if self.current_track is not None:
            parts.append(format_point(self.current_track.position))
        else:
            parts.append("waiting for tracked marker")
        if self.last_save_error:
            parts.append(f"save blocked: {self.last_save_error}")
        return " | ".join(parts)

    def _short_status_message(self) -> str:
        done = [
            f"{label} ok" if label in self.captures else f"{label} --"
            for label in CAPTURE_ORDER
        ]
        if self.pending_label is not None:
            return (
                f"capturing {self.pending_label} "
                f"{len(self.pending_samples)}/{self.args.sample_frames} | "
                + ", ".join(done)
            )
        if all(label in self.captures for label in CAPTURE_ORDER):
            return "ready to save aligned calibration | " + ", ".join(done)
        return "capture origin, then up, then forward | " + ", ".join(done)

    def _print_status(
        self,
        timestamp: float,
        tracks: list[mocap.MarkerTrack],
        observations_by_camera: dict[int, list[mocap.MarkerObservation]],
    ) -> None:
        if timestamp - self.last_print_time < self.args.print_interval:
            return
        self.last_print_time = timestamp
        mocap.print_status(
            tracks,
            observations_by_camera,
            calibrated_camera_count=len(set(self.calibrations) & {source.camera_id for source in self.sources}),
            triangulator=self.triangulator,
        )
        print(f"[alignment] {self._short_status_message()}")

    def close(self, _event=None) -> None:
        if self.closed:
            return
        self.closed = True
        if hasattr(self, "timer"):
            self.timer.stop()
        for source in self.sources:
            source.close()
        if mocap.cv2 is not None:
            mocap.cv2.destroyAllWindows()


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if mocap.cv2 is None:
        print("OpenCV is required for camera mocap. Install it with: python -m pip install opencv-python")
        return 1

    input_path = Path(args.calibration).expanduser()
    output_path = Path(args.output).expanduser()
    if not input_path.exists():
        print(f"[alignment] calibration file not found: {input_path}")
        return 1

    try:
        backend = combined.configure_vispy_backend()
    except RuntimeError as error:
        print(f"[alignment] {error}")
        return 1
    print(f"[alignment] VisPy backend: {backend}")

    raw_calibration = load_raw_calibration(input_path)
    calibrations = mocap.load_calibration_file(
        str(input_path),
        args.width,
        args.height,
        args.focal_length_px,
    )
    sources = mocap.open_available_cameras(
        args.cameras,
        args.width,
        args.height,
        args.fps,
        args.exposure,
        args.auto_exposure,
        args.gain,
    )
    if not sources:
        print("[alignment] no cameras opened")
        return 1

    connected_ids = {source.camera_id for source in sources}
    calibrated_connected_ids = connected_ids & set(calibrations)
    if len(calibrated_connected_ids) < 2:
        print("[alignment] need at least two connected calibrated cameras for 3D alignment")
        for source in sources:
            source.close()
        return 1

    print("[alignment] controls:")
    print("[alignment]   space/c = capture next point: origin, up, forward")
    print("[alignment]   1/2/3 = recapture origin/up/forward")
    print("[alignment]   s = save aligned calibration")
    print("[alignment]   r = reset captures")
    print("[alignment]   q/Esc = quit")
    print(f"[alignment] input : {input_path}")
    print(f"[alignment] output: {output_path}")

    visualizer: MovementAlignmentApp | None = None
    try:
        visualizer = MovementAlignmentApp(
            args,
            sources,
            calibrations,
            raw_calibration,
            input_path,
            output_path,
        )
        app.run()
    except KeyboardInterrupt:
        print("\n[alignment] stopped")
    finally:
        if visualizer is not None:
            visualizer.close()
        else:
            for source in sources:
                source.close()
            if mocap.cv2 is not None:
                mocap.cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
