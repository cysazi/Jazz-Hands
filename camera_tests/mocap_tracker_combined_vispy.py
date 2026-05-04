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
SCALING_FACTOR = 4.0
SCALING_FACTOR_X = SCALING_FACTOR
SCALING_FACTOR_Y = SCALING_FACTOR
SCALING_FACTOR_Z = SCALING_FACTOR

DEFAULT_TRAIL_SECONDS = 4.0
DEFAULT_UPDATE_HZ = 120.0
DEFAULT_PANEL_WIDTH = 640
DEFAULT_PANEL_HEIGHT = 400
DEFAULT_COMBINED_WINDOW_NAME = "mocap combined preview"
DEFAULT_LINE_WIDTH = 4.0
DEFAULT_POINT_SIZE = 12.0
DEFAULT_CLUSTER_DISTANCE_M = 0.08
DEFAULT_TRACK_DISTANCE_M = 0.35
DEFAULT_DUPLICATE_TRACK_DISTANCE_M = 0.03
USE_EXCLUSIVE_TWO_CAMERA_PAIRING = True
DEFAULT_MIN_MEASUREMENT_SEPARATION_M = 0.03
DEFAULT_PAIRING_TRACK_BIAS_DISTANCE_M = 0.28
DEFAULT_VISUAL_SMOOTHING = 0.35
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
    parser.set_defaults(
        calibration=str(default_calibration_path()),
        cluster_distance=DEFAULT_CLUSTER_DISTANCE_M,
        track_distance=DEFAULT_TRACK_DISTANCE_M,
        duplicate_track_distance=DEFAULT_DUPLICATE_TRACK_DISTANCE_M,
    )
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
    parser.add_argument("--x-scaling-factor", type=float, default=None)
    parser.add_argument("--y-scaling-factor", type=float, default=None)
    parser.add_argument("--z-scaling-factor", type=float, default=None)
    parser.add_argument(
        "--position-scale",
        dest="scaling_factor",
        type=float,
        default=argparse.SUPPRESS,
        help="Alias for --scaling-factor.",
    )
    parser.add_argument(
        "--disable-exclusive-pairing",
        action="store_false",
        dest="exclusive_pairing",
        default=USE_EXCLUSIVE_TWO_CAMERA_PAIRING,
        help="Use the generic clustered triangulator only.",
    )
    parser.add_argument(
        "--min-measurement-separation",
        type=float,
        default=DEFAULT_MIN_MEASUREMENT_SEPARATION_M,
        help="Minimum 3D distance between separately displayed marker measurements.",
    )
    parser.add_argument(
        "--pairing-track-bias-distance",
        type=float,
        default=DEFAULT_PAIRING_TRACK_BIAS_DISTANCE_M,
        help=(
            "For two-camera tracking, prefer blob pairings within this 3D distance "
            "of an existing track before falling back to reprojection error."
        ),
    )
    parser.add_argument(
        "--visual-smoothing",
        type=float,
        default=DEFAULT_VISUAL_SMOOTHING,
        help="Display-only smoothing for VisPy points and trails. 0 is raw, higher is steadier.",
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


def triangulate_exclusive_two_camera_pairs(
    observations_by_camera: dict[int, list[mocap.MarkerObservation]],
    calibrations: dict[int, mocap.CameraCalibration],
    camera_ids: list[int],
    room_bounds,
    max_reprojection_error_px: float,
    max_measurements: int,
    min_measurement_separation_m: float,
    reference_positions: list[np.ndarray] | None = None,
    track_bias_distance_m: float = DEFAULT_PAIRING_TRACK_BIAS_DISTANCE_M,
) -> list[mocap.MarkerMeasurement]:
    calibrated_camera_ids = [
        camera_id
        for camera_id in camera_ids
        if camera_id in calibrations and observations_by_camera.get(camera_id)
    ][:2]
    if len(calibrated_camera_ids) != 2:
        return []

    camera_a, camera_b = calibrated_camera_ids
    calib_a = calibrations[camera_a]
    calib_b = calibrations[camera_b]
    candidates: list[mocap.TriangulationCandidate] = []
    for obs_a in observations_by_camera.get(camera_a, []):
        for obs_b in observations_by_camera.get(camera_b, []):
            point = mocap.triangulate_two_views(obs_a, obs_b, calib_a, calib_b)
            if point is None or not mocap.point_is_inside_bounds(point, room_bounds):
                continue

            error = mocap.mean_reprojection_error(point, [obs_a, obs_b], calibrations)
            if error > max_reprojection_error_px:
                continue

            candidates.append(
                mocap.TriangulationCandidate(
                    position=point,
                    observations=(obs_a, obs_b),
                    reprojection_error_px=error,
                )
            )

    candidates.sort(
        key=lambda candidate: (
            candidate.reprojection_error_px,
            -sum(observation.score for observation in candidate.observations),
        )
    )

    selected: list[mocap.MarkerMeasurement] = []
    used_observation_ids: set[int] = set()
    min_separation = max(float(min_measurement_separation_m), 0.0)

    def candidate_score(candidate: mocap.TriangulationCandidate) -> float:
        return float(sum(observation.score for observation in candidate.observations))

    def candidate_is_available(candidate: mocap.TriangulationCandidate) -> bool:
        if any(id(observation) in used_observation_ids for observation in candidate.observations):
            return False
        if any(
            float(np.linalg.norm(candidate.position - measurement.position)) < min_separation
            for measurement in selected
        ):
            return False
        return True

    def add_candidate(candidate: mocap.TriangulationCandidate) -> None:
        observations = list(candidate.observations)
        selected.append(
            mocap.MarkerMeasurement(
                position=candidate.position,
                observations=observations,
                reprojection_error_px=candidate.reprojection_error_px,
            )
        )
        used_observation_ids.update(id(observation) for observation in observations)

    max_count = max(int(max_measurements), 1)
    bias_distance = max(float(track_bias_distance_m), 0.0)
    if reference_positions and bias_distance > 0.0:
        biased_options = []
        for reference_index, reference_position in enumerate(reference_positions[:max_count]):
            for candidate_index, candidate in enumerate(candidates):
                distance = float(np.linalg.norm(candidate.position - reference_position))
                if distance > bias_distance:
                    continue
                biased_options.append(
                    (
                        distance,
                        candidate.reprojection_error_px,
                        -candidate_score(candidate),
                        reference_index,
                        candidate_index,
                        candidate,
                    )
                )

        assigned_reference_indices: set[int] = set()
        for (
            _distance,
            _error,
            _score,
            reference_index,
            _candidate_index,
            candidate,
        ) in sorted(biased_options):
            if reference_index in assigned_reference_indices:
                continue
            if not candidate_is_available(candidate):
                continue

            add_candidate(candidate)
            assigned_reference_indices.add(reference_index)
            if len(selected) >= max_count:
                return selected

    for candidate in candidates:
        if not candidate_is_available(candidate):
            continue

        add_candidate(candidate)
        if len(selected) >= max_count:
            break

    return selected


def apply_scaling_defaults(args: argparse.Namespace) -> None:
    base_scale = float(args.scaling_factor)
    axis_defaults = {
        "x_scaling_factor": SCALING_FACTOR_X,
        "y_scaling_factor": SCALING_FACTOR_Y,
        "z_scaling_factor": SCALING_FACTOR_Z,
    }
    for name, axis_default in axis_defaults.items():
        if getattr(args, name) is None:
            if float(axis_default) == float(SCALING_FACTOR):
                setattr(args, name, base_scale)
            else:
                setattr(args, name, float(axis_default))


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
        self.display_positions_by_track: dict[int, np.ndarray] = {}
        self.track_lines: dict[int, object] = {}
        self.track_markers: dict[int, object] = {}
        self.track_labels: dict[int, object] = {}
        self.display_track_ids: list[int] = []
        self.last_measurement_count = 0
        self.last_live_track_count = 0
        self.last_used_exclusive_pairing = False
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
        scale = self._scale_vector()
        scene.visuals.Text("X", color="red", font_size=20, pos=(1.25 * scale[0], 0, 0), parent=self.view.scene)
        scene.visuals.Text("Y", color="green", font_size=20, pos=(0, 1.25 * scale[1], 0), parent=self.view.scene)
        scene.visuals.Text("Z", color="blue", font_size=20, pos=(0, 0, 1.25 * scale[2]), parent=self.view.scene)

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
        return position.astype(np.float32) * self._scale_vector()

    def _scale_vector(self) -> np.ndarray:
        return np.array(
            [
                float(self.args.x_scaling_factor),
                float(self.args.y_scaling_factor),
                float(self.args.z_scaling_factor),
            ],
            dtype=np.float32,
        )

    def _scale_text(self) -> str:
        scale = self._scale_vector()
        return f"scale x/y/z=({scale[0]:.2f}, {scale[1]:.2f}, {scale[2]:.2f})"

    def _smoothed_display_position(self, track: mocap.MarkerTrack) -> np.ndarray:
        raw_position = track.position.astype(np.float64)
        smoothing = float(np.clip(self.args.visual_smoothing, 0.0, 0.98))
        previous_position = self.display_positions_by_track.get(track.track_id)
        if previous_position is None or smoothing <= 0.0:
            display_position = raw_position.copy()
        else:
            display_position = smoothing * previous_position + (1.0 - smoothing) * raw_position
        self.display_positions_by_track[track.track_id] = display_position
        return display_position

    def _display_position(self, track: mocap.MarkerTrack) -> np.ndarray:
        return self.display_positions_by_track.get(track.track_id, track.position)

    def _prune_display_positions(self, valid_track_ids: set[int]) -> None:
        for track_id in list(self.display_positions_by_track):
            if track_id not in valid_track_ids:
                self.display_positions_by_track.pop(track_id, None)

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
        measurements = self._triangulate_measurements(observations_by_camera, timestamp)
        self.last_measurement_count = len(measurements)
        tracks = self.tracker.update(measurements, timestamp)

        live_tracks = [
            track for track in tracks if track.confirmed and track.missing_frames == 0
        ]
        self.last_live_track_count = len(live_tracks)
        selected_tracks = self._select_display_tracks(live_tracks)
        selected_track_ids = {track.track_id for track in selected_tracks}
        self._prune_display_positions({track.track_id for track in tracks if track.confirmed})
        for track in selected_tracks:
            display_position = self._smoothed_display_position(track)
            trail = self.trail_points_by_track.setdefault(track.track_id, deque())
            trail.append((timestamp, display_position.copy()))

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

    def _triangulate_measurements(
        self,
        observations_by_camera: dict[int, list[mocap.MarkerObservation]],
        timestamp: float,
    ) -> list[mocap.MarkerMeasurement]:
        self.last_used_exclusive_pairing = False
        visible_calibrated_camera_ids = [
            source.camera_id
            for source in self.sources
            if source.camera_id in self.calibrations
            and observations_by_camera.get(source.camera_id)
        ]
        if self.args.exclusive_pairing and len(visible_calibrated_camera_ids) == 2:
            expected_count = self._expected_measurement_count(observations_by_camera)
            exclusive_measurements = triangulate_exclusive_two_camera_pairs(
                observations_by_camera,
                self.calibrations,
                [source.camera_id for source in self.sources],
                self.args.room_bounds,
                self.args.max_reprojection_error,
                expected_count,
                self.args.min_measurement_separation,
                reference_positions=self._pairing_reference_positions(timestamp),
                track_bias_distance_m=self.args.pairing_track_bias_distance,
            )
            if exclusive_measurements:
                self.last_used_exclusive_pairing = True
                return exclusive_measurements

        measurements = self.triangulator.triangulate(observations_by_camera)
        if not self.args.exclusive_pairing:
            return measurements

        expected_count = self._expected_measurement_count(observations_by_camera)
        if len(measurements) >= expected_count:
            return measurements

        exclusive_measurements = triangulate_exclusive_two_camera_pairs(
            observations_by_camera,
            self.calibrations,
            [source.camera_id for source in self.sources],
            self.args.room_bounds,
            self.args.max_reprojection_error,
            expected_count,
            self.args.min_measurement_separation,
            reference_positions=self._pairing_reference_positions(timestamp),
            track_bias_distance_m=self.args.pairing_track_bias_distance,
        )
        if len(exclusive_measurements) > len(measurements):
            self.last_used_exclusive_pairing = True
            return exclusive_measurements
        return measurements

    def _pairing_reference_positions(self, timestamp: float) -> list[np.ndarray]:
        max_tracks = max(int(self.args.tracked_point_count), 1)
        live_reference_tracks = [
            track
            for track in self.tracker.tracks
            if track.confirmed and track.missing_frames <= self.args.max_missing_frames
        ]
        tracks_by_id = {track.track_id: track for track in live_reference_tracks}
        ordered_tracks = [
            tracks_by_id[track_id]
            for track_id in self.display_track_ids
            if track_id in tracks_by_id
        ]
        ordered_track_ids = {track.track_id for track in ordered_tracks}
        ordered_tracks.extend(
            track
            for track in sorted(live_reference_tracks, key=lambda item: item.track_id)
            if track.track_id not in ordered_track_ids
        )
        return [
            self.tracker._predicted_position(track, timestamp)
            for track in ordered_tracks[:max_tracks]
        ]

    def _expected_measurement_count(
        self,
        observations_by_camera: dict[int, list[mocap.MarkerObservation]],
    ) -> int:
        calibrated_counts = [
            len(observations_by_camera.get(source.camera_id, []))
            for source in self.sources
            if source.camera_id in self.calibrations
        ]
        if len(calibrated_counts) < 2:
            return 1
        return max(1, min(int(self.args.tracked_point_count), *calibrated_counts[:2]))

    def _select_display_tracks(
        self,
        live_tracks: list[mocap.MarkerTrack],
    ) -> list[mocap.MarkerTrack]:
        max_tracks = max(int(self.args.tracked_point_count), 1)
        live_by_id = {track.track_id: track for track in live_tracks}
        selected_ids = [
            track_id
            for track_id in self.display_track_ids
            if track_id in live_by_id
        ][:max_tracks]

        if len(selected_ids) < max_tracks:
            for track in sorted(live_tracks, key=lambda item: item.track_id):
                if track.track_id in selected_ids:
                    continue
                selected_ids.append(track.track_id)
                if len(selected_ids) >= max_tracks:
                    break

        self.display_track_ids = selected_ids
        return [live_by_id[track_id] for track_id in selected_ids]

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
        if track_id not in self.track_markers:
            marker = scene.visuals.Markers(parent=self.view.scene)
            marker.set_data(pos=np.zeros((0, 3), dtype=np.float32))
            self.track_markers[track_id] = marker
        if track_id not in self.track_labels:
            self.track_labels[track_id] = scene.visuals.Text(
                "",
                color=self._track_color(track_id),
                font_size=9,
                pos=(0, 0, 0),
                parent=self.view.scene,
            )
        return self.track_lines[track_id]

    def _draw_trails(
        self,
        timestamp: float,
        selected_tracks: list[mocap.MarkerTrack],
        observations_by_camera: dict[int, list[mocap.MarkerObservation]],
    ) -> None:
        self._draw_live_markers(selected_tracks)

        for track in selected_tracks:
            trail = self.trail_points_by_track.setdefault(track.track_id, deque())
            while trail and timestamp - trail[0][0] > self.args.trail_seconds:
                trail.popleft()

            line = self._ensure_track_visuals(track.track_id)
            color = self._track_color(track.track_id)
            if not trail:
                line.set_data(
                    pos=np.zeros((1, 3), dtype=np.float32),
                    color=(color[0], color[1], color[2], 0.0),
                    width=float(self.args.line_width),
                )
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

        blob_counts = ", ".join(
            f"cam {camera_id}: {len(observations)}"
            for camera_id, observations in sorted(observations_by_camera.items())
        )
        if not selected_tracks:
            self.status_text.text = (
                f"Waiting for 3D points | visible=0/{self.args.tracked_point_count} "
                f"| measurements={self.last_measurement_count} "
                f"| live={self.last_live_track_count} | {self._scale_text()} | {blob_counts}"
            )
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
            + f" | visible={len(selected_tracks)}/{self.args.tracked_point_count}"
            + f" | measurements={self.last_measurement_count}"
            + f" | live={self.last_live_track_count} | {self._scale_text()} | {blob_counts}"
        )

    def _draw_live_markers(self, selected_tracks: list[mocap.MarkerTrack]) -> None:
        if not selected_tracks:
            for marker in self.track_markers.values():
                marker.set_data(pos=np.zeros((0, 3), dtype=np.float32))
            for label in self.track_labels.values():
                label.text = ""
            return

        label_offset = np.array([0.025, 0.025, 0.025], dtype=np.float32) * self._scale_vector()
        for track in selected_tracks:
            self._ensure_track_visuals(track.track_id)
            position = self._visual_position(self._display_position(track))
            color = self._track_color(track.track_id)
            marker = self.track_markers[track.track_id]
            marker.set_data(
                pos=np.asarray([position], dtype=np.float32),
                face_color=color,
                edge_color=(1.0, 1.0, 1.0, 1.0),
                size=float(self.args.point_size) * 1.6,
            )
            label = self.track_labels[track.track_id]
            label.text = f"id {track.track_id}"
            label.pos = tuple(position + label_offset)

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
            label = self.track_labels.get(track_id)
            if label is not None:
                label.text = ""

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
        print(
            "[mocap combined] "
            f"measurements={self.last_measurement_count}, "
            f"live_tracks={self.last_live_track_count}, "
            f"selected={len(self.display_track_ids)}, "
            f"exclusive_pairing={'yes' if self.last_used_exclusive_pairing else 'no'}, "
            f"cluster_distance={self.args.cluster_distance:.3f}m, "
            f"track_distance={self.args.track_distance:.3f}m, "
            f"{self._scale_text()}"
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
    apply_scaling_defaults(args)

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
