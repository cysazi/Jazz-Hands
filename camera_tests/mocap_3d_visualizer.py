"""
VisPy 3D trail visualizer for the standalone camera mocap tracker.

Run this after calibrate_mocap_cameras.py has produced mocap_calibration.json.
It reuses mocap_tracker.py for camera capture, blob filtering, triangulation,
and track IDs, then draws each live 3D track as a short trail that disappears
after a configurable amount of time.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from vispy import app, scene  # type: ignore[import-untyped]

import mocap_tracker as mocap


DEFAULT_TRAIL_SECONDS = 2.5
DEFAULT_UPDATE_HZ = 60.0
DEFAULT_LINE_WIDTH = 4.0
DEFAULT_POINT_SIZE = 12.0
DEFAULT_CAMERA_MARKER_SIZE = 8.0
DEFAULT_POSITION_SCALE = 1.0

TRACK_COLORS = [
    (0.15, 1.00, 0.25, 1.0),
    (0.20, 0.55, 1.00, 1.0),
    (1.00, 0.72, 0.15, 1.0),
    (1.00, 0.22, 0.45, 1.0),
    (0.70, 0.35, 1.00, 1.0),
    (0.10, 0.95, 0.95, 1.0),
]


@dataclass
class TrackTrail:
    points: deque[tuple[float, np.ndarray]]
    line: scene.visuals.Line
    marker: scene.visuals.Markers
    color: tuple[float, float, float, float]


def configure_vispy_backend() -> str:
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
    parser = argparse.ArgumentParser(
        description="Draw mocap_tracker.py triangulated marker positions as fading 3D VisPy trails.",
    )
    parser.add_argument(
        "--cameras",
        type=mocap.parse_camera_ids,
        default=mocap.DEFAULT_CAMERA_IDS,
        help=(
            "Comma-separated OpenCV camera indexes to try. Default: 1,2,3. "
            "Use 1,2,3,4 when the full rig is plugged in."
        ),
    )
    parser.add_argument("--width", type=int, default=mocap.DEFAULT_FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=mocap.DEFAULT_FRAME_HEIGHT)
    parser.add_argument("--fps", type=int, default=mocap.DEFAULT_FPS)
    parser.add_argument(
        "--exposure",
        type=float,
        default=mocap.DEFAULT_EXPOSURE,
        help="Manual camera exposure value. The live slider uses negative values like -8.",
    )
    parser.add_argument(
        "--auto-exposure",
        type=float,
        default=None,
        help="Override auto-exposure for every camera. Default uses the tracker per-camera values.",
    )
    parser.add_argument("--gain", type=float, default=mocap.DEFAULT_GAIN)
    parser.add_argument(
        "--calibration",
        default=str(mocap.DEFAULT_CALIBRATION_PATH),
        help="JSON calibration file. Default: camera_tests/mocap_calibration.json",
    )
    parser.add_argument("--focal-length-px", type=float, default=mocap.DEFAULT_FOCAL_LENGTH_PX)
    parser.add_argument(
        "--room-bounds",
        type=mocap.parse_room_bounds,
        default=mocap.DEFAULT_ROOM_BOUNDS,
        help="xmin,xmax,ymin,ymax,zmin,zmax in meters. Default: -5,5,-5,5,-5,5",
    )
    parser.add_argument("--threshold", type=int, default=mocap.DEFAULT_THRESHOLD)
    parser.add_argument("--min-area", type=float, default=5.0)
    parser.add_argument("--max-area", type=float, default=4500.0)
    parser.add_argument("--min-radius", type=float, default=1.5)
    parser.add_argument("--max-radius", type=float, default=80.0)
    parser.add_argument("--min-circularity", type=float, default=0.30)
    parser.add_argument("--min-fill-ratio", type=float, default=0.20)
    parser.add_argument("--max-aspect-ratio", type=float, default=3.0)
    parser.add_argument("--min-brightness", type=float, default=0.0)
    parser.add_argument("--max-markers-per-camera", type=int, default=12)
    parser.add_argument("--max-reprojection-error", type=float, default=45.0)
    parser.add_argument("--cluster-distance", type=float, default=0.45)
    parser.add_argument("--track-distance", type=float, default=0.75)
    parser.add_argument(
        "--max-missing-frames",
        type=int,
        default=mocap.DEFAULT_MAX_MISSING_FRAMES,
        help="How long the tracker keeps a 3D ID alive without a fresh triangulation.",
    )
    parser.add_argument(
        "--track-memory-pixels",
        type=float,
        default=mocap.DEFAULT_TRACK_MEMORY_PIXEL_DISTANCE,
        help="2D pixel search radius used to keep a briefly weak blob attached to its track.",
    )
    parser.add_argument("--trail-seconds", type=float, default=DEFAULT_TRAIL_SECONDS)
    parser.add_argument("--update-hz", type=float, default=DEFAULT_UPDATE_HZ)
    parser.add_argument("--line-width", type=float, default=DEFAULT_LINE_WIDTH)
    parser.add_argument("--point-size", type=float, default=DEFAULT_POINT_SIZE)
    parser.add_argument(
        "--position-scale",
        type=float,
        default=DEFAULT_POSITION_SCALE,
        help="Visual scale multiplier applied after triangulation. Tracking units stay in meters.",
    )
    parser.add_argument(
        "--camera-preview",
        action="store_true",
        help="Also show the mocap raw and binary OpenCV preview windows.",
    )
    parser.add_argument(
        "--no-controls",
        action="store_true",
        help="Hide the OpenCV threshold/exposure/gain slider window.",
    )
    return parser


class MocapTrailVisualizer:
    def __init__(
        self,
        args: argparse.Namespace,
        sources: list[mocap.CameraSource],
        calibrations: dict[int, mocap.CameraCalibration],
    ):
        self.args = args
        self.sources = sources
        self.calibrations = calibrations
        self.settings = mocap.DetectionSettings(
            threshold=args.threshold,
            min_area=args.min_area,
            max_area=args.max_area,
            min_radius_px=args.min_radius,
            max_radius_px=args.max_radius,
            min_circularity=args.min_circularity,
            min_fill_ratio=args.min_fill_ratio,
            max_aspect_ratio=args.max_aspect_ratio,
            min_brightness=args.min_brightness,
            max_markers_per_camera=args.max_markers_per_camera,
        )
        self.detector = mocap.ReflectiveMarkerDetector(self.settings)
        self.triangulator = mocap.MultiCameraTriangulator(
            calibrations=calibrations,
            max_pair_error_px=args.max_reprojection_error,
            cluster_distance_m=args.cluster_distance,
            room_bounds=args.room_bounds,
        )
        self.tracker = mocap.MarkerTracker(
            max_match_distance_m=args.track_distance,
            max_missing_frames=args.max_missing_frames,
        )
        self.trails: dict[int, TrackTrail] = {}
        self.last_applied_camera_settings: dict[int, tuple[float, float, float]] = {}
        self.last_status_time = 0.0
        self.is_closed = False

        self.canvas = scene.SceneCanvas(
            keys="interactive",
            show=True,
            bgcolor="black",
            size=(1000, 750),
            title="Jazz Hands mocap 3D trail visualizer",
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(
            fov=45,
            distance=4.0,
            elevation=25.0,
            azimuth=-35.0,
            center=(0, 0, 0),
        )

        scene.visuals.GridLines(scale=(0.25, 0.25), color=(0.25, 0.25, 0.25, 1.0), parent=self.view.scene)
        scene.visuals.XYZAxis(width=2, parent=self.view.scene)
        self._add_axis_labels()
        self._add_camera_markers()

        self.status_text = scene.visuals.Text(
            "Waiting for triangulated marker...",
            color="white",
            font_size=9,
            pos=(10, 10),
            anchor_x="left",
            anchor_y="bottom",
            parent=self.canvas.scene,
        )

        self.canvas.events.close.connect(self.close)
        self.timer = app.Timer(
            interval=max(1.0 / max(args.update_hz, 1.0), 0.001),
            connect=self.update,
            start=True,
        )

    def _add_axis_labels(self) -> None:
        label_distance = 1.25 * float(self.args.position_scale)
        scene.visuals.Text(
            "X",
            color="red",
            font_size=20,
            pos=(label_distance, 0, 0),
            parent=self.view.scene,
        )
        scene.visuals.Text(
            "Y",
            color="green",
            font_size=20,
            pos=(0, label_distance, 0),
            parent=self.view.scene,
        )
        scene.visuals.Text(
            "Z",
            color="blue",
            font_size=20,
            pos=(0, 0, label_distance),
            parent=self.view.scene,
        )

    def _add_camera_markers(self) -> None:
        if not self.calibrations:
            return

        positions = []
        labels = []
        for camera_id, calibration in sorted(self.calibrations.items()):
            camera_position = -calibration.rotation.T @ calibration.translation.reshape(3)
            positions.append(self._visual_pos(camera_position))
            labels.append((camera_id, positions[-1]))

        if not positions:
            return

        camera_points = scene.visuals.Markers(parent=self.view.scene)
        camera_points.set_data(
            pos=np.asarray(positions, dtype=np.float32),
            face_color=(0.2, 0.55, 1.0, 1.0),
            edge_color=(0.85, 0.95, 1.0, 1.0),
            size=DEFAULT_CAMERA_MARKER_SIZE,
        )
        for camera_id, position in labels:
            scene.visuals.Text(
                f"cam {camera_id}",
                color=(0.75, 0.9, 1.0, 1.0),
                font_size=8,
                pos=tuple(position),
                parent=self.view.scene,
            )

    def _visual_pos(self, position: np.ndarray) -> np.ndarray:
        return position.astype(np.float32) * float(self.args.position_scale)

    def _ensure_trail(self, track_id: int) -> TrackTrail:
        trail = self.trails.get(track_id)
        if trail is not None:
            return trail

        color = TRACK_COLORS[(track_id - 1) % len(TRACK_COLORS)]
        line = scene.visuals.Line(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(color[0], color[1], color[2], 0.0),
            width=float(self.args.line_width),
            parent=self.view.scene,
            method="gl",
        )
        marker = scene.visuals.Markers(parent=self.view.scene)
        marker.set_data(
            pos=np.zeros((0, 3), dtype=np.float32),
            face_color=(color[0], color[1], color[2], 1.0),
            edge_color=(1.0, 1.0, 1.0, 1.0),
            size=float(self.args.point_size),
        )
        trail = TrackTrail(points=deque(), line=line, marker=marker, color=color)
        self.trails[track_id] = trail
        return trail

    def update(self, _event) -> None:
        if self.is_closed:
            return

        timestamp = time.time()
        frames: dict[int, np.ndarray] = {}
        observations_by_camera: dict[int, list[mocap.MarkerObservation]] = {}

        if not self.args.no_controls:
            mocap.apply_mocap_controls(
                self.sources,
                self.settings,
                self.last_applied_camera_settings,
            )

        for source in self.sources:
            ok, frame = source.read()
            if not ok or frame is None:
                observations_by_camera[source.camera_id] = []
                continue
            frames[source.camera_id] = frame
            observations_by_camera[source.camera_id] = self.detector.detect(
                frame,
                source.camera_id,
                timestamp,
            )

        mocap.lock_observations_to_existing_tracks(
            frames,
            observations_by_camera,
            self.tracker.tracks,
            self.settings,
            self.args.track_memory_pixels,
            timestamp,
        )
        measurements = self.triangulator.triangulate(observations_by_camera)
        tracks = self.tracker.update(measurements, timestamp)

        for track in tracks:
            if track.missing_frames == 0:
                self._ensure_trail(track.track_id).points.append(
                    (timestamp, track.position.copy())
                )

        self._draw_trails(timestamp)
        self._update_status(timestamp, tracks, observations_by_camera)

        if self.args.camera_preview:
            self._draw_camera_preview(frames, observations_by_camera, tracks)

        if not self.args.no_controls or self.args.camera_preview:
            key = mocap.cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                self.canvas.close()

    def _draw_trails(self, timestamp: float) -> None:
        for track_id, trail in list(self.trails.items()):
            while trail.points and timestamp - trail.points[0][0] > self.args.trail_seconds:
                trail.points.popleft()

            if not trail.points:
                trail.line.parent = None
                trail.marker.parent = None
                del self.trails[track_id]
                continue

            positions = np.asarray(
                [self._visual_pos(position) for _point_time, position in trail.points],
                dtype=np.float32,
            )
            ages = np.asarray(
                [timestamp - point_time for point_time, _position in trail.points],
                dtype=np.float32,
            )
            alphas = np.clip(1.0 - ages / max(float(self.args.trail_seconds), 0.001), 0.05, 1.0)
            colors = np.tile(np.asarray(trail.color, dtype=np.float32), (len(positions), 1))
            colors[:, 3] = alphas

            if len(positions) >= 2:
                trail.line.set_data(
                    pos=positions,
                    color=colors,
                    width=float(self.args.line_width),
                )
            else:
                invisible_color = (trail.color[0], trail.color[1], trail.color[2], 0.0)
                trail.line.set_data(
                    pos=np.zeros((1, 3), dtype=np.float32),
                    color=invisible_color,
                    width=float(self.args.line_width),
                )

            trail.marker.set_data(
                pos=positions[-1:].astype(np.float32),
                face_color=trail.color,
                edge_color=(1.0, 1.0, 1.0, 1.0),
                size=float(self.args.point_size),
            )

    def _update_status(
        self,
        timestamp: float,
        tracks: list[mocap.MarkerTrack],
        observations_by_camera: dict[int, list[mocap.MarkerObservation]],
    ) -> None:
        if timestamp - self.last_status_time < 0.10:
            return
        self.last_status_time = timestamp

        fresh_tracks = [track for track in tracks if track.missing_frames == 0]
        blob_counts = ", ".join(
            f"cam {camera_id}: {len(observations)}"
            for camera_id, observations in sorted(observations_by_camera.items())
        )
        if fresh_tracks:
            best_track = max(fresh_tracks, key=lambda track: track.confidence)
            x, y, z = best_track.position
            self.status_text.text = (
                f"track {best_track.track_id} | "
                f"x={x:+.3f}m y={y:+.3f}m z={z:+.3f}m | "
                f"err={best_track.reprojection_error_px:.1f}px | {blob_counts}"
            )
        else:
            self.status_text.text = f"Waiting for 3D triangulation | {blob_counts}"

    def _draw_camera_preview(
        self,
        frames: dict[int, np.ndarray],
        observations_by_camera: dict[int, list[mocap.MarkerObservation]],
        tracks: list[mocap.MarkerTrack],
    ) -> None:
        for camera_id, frame in frames.items():
            preview = mocap.draw_preview(
                frame,
                observations_by_camera.get(camera_id, []),
                tracks,
                camera_id,
                self.args.track_memory_pixels,
            )
            mocap.cv2.imshow(f"mocap camera {camera_id}", preview)
            mocap.cv2.imshow(f"mocap binary camera {camera_id}", mocap.threshold_mask(frame, self.settings))

    def close(self, _event=None) -> None:
        if self.is_closed:
            return
        self.is_closed = True
        if hasattr(self, "timer"):
            self.timer.stop()
        for source in self.sources:
            source.close()
        if mocap.cv2 is not None:
            mocap.cv2.destroyAllWindows()


def load_calibrations(args: argparse.Namespace) -> dict[int, mocap.CameraCalibration]:
    calibration_path = Path(args.calibration).expanduser() if args.calibration else None
    if calibration_path is not None and calibration_path.exists():
        print(f"[mocap-3d] loading calibration from {calibration_path}")
        return mocap.load_calibration_file(
            str(calibration_path),
            args.width,
            args.height,
            args.focal_length_px,
        )

    if calibration_path is not None:
        print(f"[mocap-3d] calibration file not found: {calibration_path}")
    print(
        "[mocap-3d] using placeholder camera calibration. "
        "Run calibrate_mocap_cameras.py first for real 3D positions."
    )
    return mocap.build_default_room_calibrations(
        args.cameras,
        args.width,
        args.height,
        args.focal_length_px,
    )


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if mocap.cv2 is None:
        print("OpenCV is required for camera mocap. Install it with: python -m pip install opencv-python")
        return 1

    backend = configure_vispy_backend()
    print(f"[mocap-3d] VisPy backend: {backend}")

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
        print("[mocap-3d] no cameras opened")
        return 1

    connected_ids = {source.camera_id for source in sources}
    calibrated_connected_ids = connected_ids & set(calibrations)
    if len(calibrated_connected_ids) < 2:
        print(
            "[mocap-3d] fewer than two connected cameras have calibration. "
            "The trail needs at least two calibrated views of the same blob."
        )

    if not args.no_controls:
        mocap.setup_mocap_controls(args)

    visualizer: MocapTrailVisualizer | None = None
    try:
        visualizer = MocapTrailVisualizer(args, sources, calibrations)
        app.run()
    except KeyboardInterrupt:
        print("\n[mocap-3d] stopped")
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
