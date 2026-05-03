"""
Mocap tracker variant with one combined OpenCV preview window and one VisPy
3D trail window.

For two cameras, the combined preview window is a 2x2 grid:
- top-left: camera 1 tracked preview
- top-right: camera 1 binary threshold
- bottom-left: camera 2 tracked preview
- bottom-right: camera 2 binary threshold
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np

try:
    from vispy import app, scene  # type: ignore[import-untyped]
except ImportError:  # Keep --help and py_compile usable without VisPy installed.
    app = None
    scene = None

import mocap_tracker as mocap


# Set this to False to use mocap_calibration.json instead.
USE_ALIGNED_MOVEMENT_CALIBRATION = True
STANDARD_CALIBRATION_PATH = Path(__file__).resolve().with_name("mocap_calibration.json")
ALIGNED_MOVEMENT_CALIBRATION_PATH = Path(__file__).resolve().with_name(
    "mocap_calibration_aligned.json"
)
TRACKED_POINT_COUNT = 2
SCALING_FACTOR = 2.0

DEFAULT_TRAIL_SECONDS = 4.0
DEFAULT_UPDATE_HZ = 120.0
DEFAULT_PANEL_WIDTH = 640
DEFAULT_PANEL_HEIGHT = 400
DEFAULT_COMBINED_WINDOW_NAME = "mocap combined preview"
DEFAULT_LINE_WIDTH = 4.0
DEFAULT_POINT_SIZE = 12.0
TRAIL_COLORS = [
    (0.15, 1.00, 0.25, 1.0),
    (1.00, 0.30, 0.20, 1.0),
    (0.20, 0.55, 1.00, 1.0),
    (1.00, 0.72, 0.15, 1.0),
]


def default_calibration_path() -> Path:
    if USE_ALIGNED_MOVEMENT_CALIBRATION:
        return ALIGNED_MOVEMENT_CALIBRATION_PATH
    return STANDARD_CALIBRATION_PATH


def configure_vispy_backend() -> str:
    if app is None:
        raise RuntimeError("VisPy is not installed. Install it with: python -m pip install vispy PyQt6")

    for backend in ("pyqt6", "pyside6", "tkinter"):
        try:
            app.use_app(backend)
            return backend
        except Exception:
            continue
    raise RuntimeError(
        "VisPy could not load a GUI backend. Install one with: python -m pip install PyQt6"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = mocap.build_arg_parser()
    parser.set_defaults(calibration=str(default_calibration_path()))
    parser.description = (
        "Track mocap markers with a single 2x2 camera preview window and "
        "a VisPy 3D trail window."
    )
    parser.add_argument("--trail-seconds", type=float, default=DEFAULT_TRAIL_SECONDS)
    parser.add_argument("--update-hz", type=float, default=DEFAULT_UPDATE_HZ)
    parser.add_argument("--panel-width", type=int, default=DEFAULT_PANEL_WIDTH)
    parser.add_argument("--panel-height", type=int, default=DEFAULT_PANEL_HEIGHT)
    parser.add_argument("--tracked-point-count", type=int, default=TRACKED_POINT_COUNT)
    parser.add_argument("--scaling-factor", type=float, default=SCALING_FACTOR)
    parser.add_argument(
        "--position-scale",
        dest="scaling_factor",
        type=float,
        default=argparse.SUPPRESS,
        help="Alias for --scaling-factor.",
    )
    parser.add_argument("--line-width", type=float, default=DEFAULT_LINE_WIDTH)
    parser.add_argument("--point-size", type=float, default=DEFAULT_POINT_SIZE)
    return parser


def load_calibrations(args: argparse.Namespace) -> dict[int, mocap.CameraCalibration]:
    calibration_path = Path(args.calibration).expanduser() if args.calibration else None
    if calibration_path is not None and calibration_path.exists():
        print(f"[mocap combined] loading calibration from {calibration_path}")
        return mocap.load_calibration_file(
            str(calibration_path),
            args.width,
            args.height,
            args.focal_length_px,
        )

    if calibration_path is not None:
        print(f"[mocap combined] calibration file not found: {calibration_path}")
    print(
        "[mocap combined] using placeholder camera calibration. "
        "Run calibrate_mocap_cameras.py first for real 3D positions."
    )
    return mocap.build_default_room_calibrations(
        args.cameras,
        args.width,
        args.height,
        args.focal_length_px,
    )


def resize_panel(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return mocap.cv2.resize(frame, (width, height), interpolation=mocap.cv2.INTER_AREA)


def blank_panel(title: str, width: int, height: int) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    draw_panel_title(panel, title)
    mocap.cv2.putText(
        panel,
        "no frame",
        (24, height // 2),
        mocap.cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
        mocap.cv2.LINE_AA,
    )
    return panel


def draw_panel_title(panel: np.ndarray, title: str) -> None:
    mocap.cv2.rectangle(panel, (0, 0), (panel.shape[1], 32), (0, 0, 0), -1)
    mocap.cv2.putText(
        panel,
        title,
        (12, 23),
        mocap.cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        mocap.cv2.LINE_AA,
    )


def binary_preview(
    frame: np.ndarray,
    observations: list[mocap.MarkerObservation],
    settings: mocap.DetectionSettings,
    camera_id: int,
) -> np.ndarray:
    mask = mocap.threshold_mask(frame, settings)
    preview = mocap.cv2.cvtColor(mask, mocap.cv2.COLOR_GRAY2BGR)
    for index, observation in enumerate(observations, start=1):
        center = tuple(int(round(value)) for value in observation.pixel)
        radius = max(3, int(round(observation.radius_px)))
        mocap.cv2.circle(preview, center, radius, (0, 255, 0), 2)
        mocap.cv2.putText(
            preview,
            str(index),
            (center[0] + 8, center[1] - 8),
            mocap.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            mocap.cv2.LINE_AA,
        )
    draw_panel_title(preview, f"camera {camera_id} binary")
    return preview


def build_combined_preview(
    preview_camera_ids: list[int],
    frames: dict[int, np.ndarray],
    observations_by_camera: dict[int, list[mocap.MarkerObservation]],
    tracks: list[mocap.MarkerTrack],
    settings_by_camera: dict[int, mocap.DetectionSettings],
    track_memory_pixels: float,
    panel_width: int,
    panel_height: int,
) -> np.ndarray:
    panels: list[np.ndarray] = []
    for camera_id in preview_camera_ids[:2]:
        frame = frames.get(camera_id)
        if frame is None:
            raw_panel = blank_panel(f"camera {camera_id} tracked", panel_width, panel_height)
            binary_panel = blank_panel(f"camera {camera_id} binary", panel_width, panel_height)
        else:
            observations = observations_by_camera.get(camera_id, [])
            raw_panel = mocap.draw_preview(
                frame,
                observations,
                tracks,
                camera_id,
                track_memory_pixels,
            )
            draw_panel_title(raw_panel, f"camera {camera_id} tracked")
            binary_panel = binary_preview(
                frame,
                observations,
                settings_by_camera[camera_id],
                camera_id,
            )

        panels.append(resize_panel(raw_panel, panel_width, panel_height))
        panels.append(resize_panel(binary_panel, panel_width, panel_height))

    while len(panels) < 4:
        panels.append(blank_panel("unused", panel_width, panel_height))

    top_row = mocap.cv2.hconcat([panels[0], panels[1]])
    bottom_row = mocap.cv2.hconcat([panels[2], panels[3]])
    return mocap.cv2.vconcat([top_row, bottom_row])


class CombinedMocapTrailApp:
    def __init__(
        self,
        args: argparse.Namespace,
        sources: list[mocap.CameraSource],
        calibrations: dict[int, mocap.CameraCalibration],
    ):
        if scene is None or app is None:
            raise RuntimeError("VisPy is not available.")

        self.args = args
        self.sources = sources
        self.calibrations = calibrations
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
        self.trail_points_by_track: dict[int, deque[tuple[float, np.ndarray]]] = {}
        self.track_lines: dict[int, object] = {}
        self.track_markers: dict[int, object] = {}
        self.last_print_time = 0.0
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
        mocap.cv2.namedWindow(DEFAULT_COMBINED_WINDOW_NAME, mocap.cv2.WINDOW_NORMAL)
        mocap.cv2.resizeWindow(
            DEFAULT_COMBINED_WINDOW_NAME,
            int(self.args.panel_width) * 2,
            int(self.args.panel_height) * 2,
        )

    def _setup_vispy_window(self) -> None:
        self.canvas = scene.SceneCanvas(
            keys="interactive",
            show=True,
            bgcolor="black",
            size=(1000, 750),
            title="Jazz Hands mocap 3D point trail",
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

        self.status_text = scene.visuals.Text(
            "Waiting for triangulated points...",
            color="white",
            font_size=9,
            pos=(10, 10),
            anchor_x="left",
            anchor_y="bottom",
            parent=self.canvas.scene,
        )
        self.canvas.events.close.connect(self.close)

    def _add_axis_labels(self) -> None:
        label_distance = 1.25 * float(self.args.scaling_factor)
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
        return position.astype(np.float32) * float(self.args.scaling_factor)

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
        selected_tracks = self._select_display_tracks(live_tracks)
        selected_track_ids = {track.track_id for track in selected_tracks}
        for track in selected_tracks:
            trail = self.trail_points_by_track.setdefault(track.track_id, deque())
            trail.append((timestamp, track.position.copy()))

        self._draw_trails(timestamp, selected_tracks, observations_by_camera)
        self._hide_unused_track_visuals(selected_track_ids)
        self._print_status(timestamp, tracks, observations_by_camera)

        if not self.args.no_preview:
            combined = build_combined_preview(
                self.preview_camera_ids,
                frames,
                observations_by_camera,
                tracks,
                self.settings_by_camera,
                self.args.track_memory_pixels,
                int(self.args.panel_width),
                int(self.args.panel_height),
            )
            mocap.cv2.imshow(DEFAULT_COMBINED_WINDOW_NAME, combined)
            key = mocap.cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                self.canvas.close()

    def _select_display_tracks(
        self,
        live_tracks: list[mocap.MarkerTrack],
    ) -> list[mocap.MarkerTrack]:
        max_tracks = max(int(self.args.tracked_point_count), 1)
        return sorted(
            live_tracks,
            key=lambda track: (-track.confidence, -track.total_hits, track.track_id),
        )[:max_tracks]

    def _track_color(self, track_id: int) -> tuple[float, float, float, float]:
        return TRAIL_COLORS[(track_id - 1) % len(TRAIL_COLORS)]

    def _ensure_track_visuals(self, track_id: int):
        if track_id not in self.track_lines:
            color = self._track_color(track_id)
            self.track_lines[track_id] = scene.visuals.Line(
                pos=np.zeros((1, 3), dtype=np.float32),
                color=(color[0], color[1], color[2], 0.0),
                width=float(self.args.line_width),
                parent=self.view.scene,
                method="gl",
            )
            marker = scene.visuals.Markers(parent=self.view.scene)
            marker.set_data(pos=np.zeros((0, 3), dtype=np.float32))
            self.track_markers[track_id] = marker
        return self.track_lines[track_id], self.track_markers[track_id]

    def _draw_trails(
        self,
        timestamp: float,
        selected_tracks: list[mocap.MarkerTrack],
        observations_by_camera: dict[int, list[mocap.MarkerObservation]],
    ) -> None:
        for track in selected_tracks:
            trail = self.trail_points_by_track.setdefault(track.track_id, deque())
            while trail and timestamp - trail[0][0] > self.args.trail_seconds:
                trail.popleft()

            line, marker = self._ensure_track_visuals(track.track_id)
            color = self._track_color(track.track_id)
            if not trail:
                line.set_data(
                    pos=np.zeros((1, 3), dtype=np.float32),
                    color=(color[0], color[1], color[2], 0.0),
                    width=float(self.args.line_width),
                )
                marker.set_data(pos=np.zeros((0, 3), dtype=np.float32))
                continue

            positions = np.asarray(
                [self._visual_position(position) for _point_time, position in trail],
                dtype=np.float32,
            )
            ages = np.asarray(
                [timestamp - point_time for point_time, _position in trail],
                dtype=np.float32,
            )
            alphas = np.clip(1.0 - ages / max(float(self.args.trail_seconds), 0.001), 0.05, 1.0)
            colors = np.tile(np.asarray(color, dtype=np.float32), (len(positions), 1))
            colors[:, 3] = alphas

            if len(positions) >= 2:
                line.set_data(
                    pos=positions,
                    color=colors,
                    width=float(self.args.line_width),
                )
            else:
                line.set_data(
                    pos=np.zeros((1, 3), dtype=np.float32),
                    color=(color[0], color[1], color[2], 0.0),
                    width=float(self.args.line_width),
                )
            marker.set_data(
                pos=positions[-1:],
                face_color=color,
                edge_color=(1.0, 1.0, 1.0, 1.0),
                size=float(self.args.point_size),
            )

        blob_counts = ", ".join(
            f"cam {camera_id}: {len(observations)}"
            for camera_id, observations in sorted(observations_by_camera.items())
        )
        if not selected_tracks:
            self.status_text.text = f"Waiting for 3D points | scale={self.args.scaling_factor:.2f} | {blob_counts}"
            return

        track_parts = []
        for track in selected_tracks:
            x, y, z = track.position
            track_parts.append(
                f"id {track.track_id}: x={x:+.3f} y={y:+.3f} z={z:+.3f} "
                f"err={track.reprojection_error_px:.1f}px"
            )
        self.status_text.text = (
            " | ".join(track_parts)
            + f" | scale={self.args.scaling_factor:.2f} | {blob_counts}"
        )

    def _hide_unused_track_visuals(self, selected_track_ids: set[int]) -> None:
        for track_id, line in self.track_lines.items():
            if track_id in selected_track_ids:
                continue
            color = self._track_color(track_id)
            line.set_data(
                pos=np.zeros((1, 3), dtype=np.float32),
                color=(color[0], color[1], color[2], 0.0),
                width=float(self.args.line_width),
            )
            marker = self.track_markers.get(track_id)
            if marker is not None:
                marker.set_data(pos=np.zeros((0, 3), dtype=np.float32))

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

    try:
        backend = configure_vispy_backend()
    except RuntimeError as error:
        print(f"[mocap combined] {error}")
        return 1
    print(f"[mocap combined] VisPy backend: {backend}")

    calibrations = load_calibrations(args)
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
        print("[mocap combined] no cameras opened")
        return 1

    connected_ids = {source.camera_id for source in sources}
    calibrated_connected_ids = connected_ids & set(calibrations)
    if len(calibrated_connected_ids) < 2:
        print(
            "[mocap combined] fewer than two connected cameras have calibration. "
            "The 3D trail needs at least two calibrated camera views."
        )

    visualizer: CombinedMocapTrailApp | None = None
    try:
        visualizer = CombinedMocapTrailApp(args, sources, calibrations)
        app.run()
    except KeyboardInterrupt:
        print("\n[mocap combined] stopped")
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
