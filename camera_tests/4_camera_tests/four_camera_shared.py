from __future__ import annotations

import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CAMERA_TESTS_DIR = Path(__file__).resolve().parents[1]
if str(CAMERA_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(CAMERA_TESTS_DIR))

import mocap_tracker as mocap
import multithreaded_camera_testing as camera_settings


if mocap.cv2 is not None:
    mocap.cv2.setNumThreads(1)


CAMERA_IDS = list(camera_settings.FOUR_CAMERA_IDS)
FRONT_CAMERA_IDS = CAMERA_IDS[:2]
TOP_CAMERA_IDS = CAMERA_IDS[2:4]

STANDARD_CALIBRATION_PATH = CAMERA_TESTS_DIR / "mocap_calibration.json"
ALIGNED_CALIBRATION_PATH = CAMERA_TESTS_DIR / "mocap_calibration_aligned.json"
USE_ALIGNED_MOVEMENT_CALIBRATION = True

DEFAULT_PANEL_WIDTH = 420
DEFAULT_PANEL_HEIGHT = 260
DEFAULT_UPDATE_HZ = 120.0
DEFAULT_TRAIL_SECONDS = 4.0
DEFAULT_TRACKED_POINT_COUNT = 2
DEFAULT_SCALING_FACTOR = 4.0
DEFAULT_SCALING_FACTOR_X = DEFAULT_SCALING_FACTOR
DEFAULT_SCALING_FACTOR_Y = DEFAULT_SCALING_FACTOR
DEFAULT_SCALING_FACTOR_Z = DEFAULT_SCALING_FACTOR

DEFAULT_FUSION_Y_TOLERANCE_M = 0.35
DEFAULT_MAX_FUSED_REPROJECTION_ERROR_PX = 95.0
DEFAULT_MAX_LAYOUT_MEASUREMENTS = 4
DEFAULT_MIN_MEASUREMENT_SEPARATION_M = 0.03
DEFAULT_PAIRING_TRACK_BIAS_DISTANCE_M = 0.28
DEFAULT_VISUAL_SMOOTHING = 0.35


def default_calibration_path() -> Path:
    if USE_ALIGNED_MOVEMENT_CALIBRATION:
        return ALIGNED_CALIBRATION_PATH
    return STANDARD_CALIBRATION_PATH


def parse_camera_pair(text: str) -> tuple[int, int]:
    camera_ids = mocap.parse_camera_ids(text)
    if len(camera_ids) != 2:
        raise ValueError("camera pair must contain exactly two camera indexes")
    return int(camera_ids[0]), int(camera_ids[1])


def configure_vispy_backend(app_module) -> str:
    if app_module is None:
        raise RuntimeError("VisPy is not installed. Install it with: python -m pip install vispy PyQt6")

    for backend in ("pyqt6", "pyside6", "tkinter"):
        try:
            app_module.use_app(backend)
            return backend
        except Exception:
            continue
    raise RuntimeError(
        "VisPy could not load a GUI backend. Install one with: python -m pip install PyQt6"
    )


class DeliveredFpsCounter:
    def __init__(self) -> None:
        self.window_start = time.perf_counter()
        self.frames_this_window = 0
        self.fps = 0.0

    def mark_frame(self) -> float:
        self.frames_this_window += 1
        now = time.perf_counter()
        elapsed = now - self.window_start
        if elapsed >= 1.0:
            self.fps = self.frames_this_window / elapsed
            self.frames_this_window = 0
            self.window_start = now
        return self.fps


def average_ms(values: deque[float]) -> float:
    if not values:
        return 0.0
    return 1000.0 * sum(values) / len(values)


@dataclass(slots=True)
class CameraSnapshot:
    camera_id: int
    frame: np.ndarray | None
    mask: np.ndarray | None
    observations: list[mocap.MarkerObservation]
    fps: float
    frame_number: int
    timestamp: float
    opened: bool
    failed: bool
    open_attempt_done: bool
    read_ms: float
    detect_ms: float
    mask_ms: float
    error: str | None


class ThreadedMocapCamera:
    def __init__(
        self,
        camera_id: int,
        args,
        stop_event: threading.Event,
        settings: mocap.DetectionSettings,
        build_masks: bool = True,
        label: str = "4cam",
    ) -> None:
        self.camera_id = int(camera_id)
        self.args = args
        self.stop_event = stop_event
        self.settings = settings
        self.detector = mocap.ReflectiveMarkerDetector(settings)
        self.build_masks = build_masks
        self.label = label
        self.lock = threading.Lock()
        self.fps_counter = DeliveredFpsCounter()
        self.read_times: deque[float] = deque(maxlen=60)
        self.detect_times: deque[float] = deque(maxlen=60)
        self.mask_times: deque[float] = deque(maxlen=60)
        self.source: mocap.CameraSource | None = None
        self.frame: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.observations: list[mocap.MarkerObservation] = []
        self.fps = 0.0
        self.frame_number = 0
        self.timestamp = 0.0
        self.opened = False
        self.failed = False
        self.open_attempt_done = False
        self.error: str | None = None
        self.thread = threading.Thread(
            target=self.run,
            name=f"{label} camera {camera_id}",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def join(self) -> None:
        self.thread.join()

    def snapshot(self) -> CameraSnapshot:
        with self.lock:
            return CameraSnapshot(
                camera_id=self.camera_id,
                frame=self.frame,
                mask=self.mask,
                observations=list(self.observations),
                fps=float(self.fps),
                frame_number=int(self.frame_number),
                timestamp=float(self.timestamp),
                opened=bool(self.opened),
                failed=bool(self.failed),
                open_attempt_done=bool(self.open_attempt_done),
                read_ms=average_ms(self.read_times),
                detect_ms=average_ms(self.detect_times),
                mask_ms=average_ms(self.mask_times),
                error=self.error,
            )

    def close(self) -> None:
        if self.source is not None:
            self.source.close()

    def run(self) -> None:
        print(f"[{self.label}] opening camera {self.camera_id} on thread {threading.get_native_id()}...")
        source = mocap.CameraSource(
            self.camera_id,
            int(self.args.width),
            int(self.args.height),
            int(self.args.fps),
            mocap.camera_exposure(self.camera_id, self.args.exposure),
            mocap.camera_auto_exposure(self.camera_id, self.args.auto_exposure),
            mocap.camera_gain(self.camera_id, self.args.gain),
            configure_capture=True,
        )
        self.source = source
        if not source.open():
            with self.lock:
                self.failed = True
                self.open_attempt_done = True
                self.error = f"could not open camera {self.camera_id}"
            print(f"[{self.label}] camera {self.camera_id} not available")
            return

        with self.lock:
            self.opened = True
            self.open_attempt_done = True
            self.error = None
        mocap.print_camera_settings(source)

        try:
            while not self.stop_event.is_set():
                read_start = time.perf_counter()
                ok, frame = source.read()
                read_seconds = time.perf_counter() - read_start
                if not ok or frame is None:
                    with self.lock:
                        self.error = f"could not read frame from camera {self.camera_id}"
                    time.sleep(0.005)
                    continue

                timestamp = time.time()
                detect_start = time.perf_counter()
                observations = self.detector.detect(frame, self.camera_id, timestamp)
                detect_seconds = time.perf_counter() - detect_start

                mask = None
                mask_seconds = 0.0
                if self.build_masks:
                    mask_start = time.perf_counter()
                    mask = mocap.threshold_mask(frame, self.settings)
                    mask_seconds = time.perf_counter() - mask_start

                fps = self.fps_counter.mark_frame()
                with self.lock:
                    self.frame = frame
                    self.mask = mask
                    self.observations = observations
                    self.fps = fps
                    self.frame_number += 1
                    self.timestamp = timestamp
                    self.read_times.append(read_seconds)
                    self.detect_times.append(detect_seconds)
                    self.mask_times.append(mask_seconds)
                    self.error = None
        finally:
            source.close()


def start_threaded_cameras(
    args,
    camera_ids: list[int],
    settings_by_camera: dict[int, mocap.DetectionSettings] | None = None,
    build_masks: bool = True,
    label: str = "4cam",
) -> tuple[threading.Event, list[ThreadedMocapCamera]]:
    stop_event = threading.Event()
    if settings_by_camera is None:
        settings_by_camera = {
            camera_id: mocap.build_detection_settings(args, camera_id)
            for camera_id in camera_ids
        }
    workers = [
        ThreadedMocapCamera(
            camera_id,
            args,
            stop_event,
            settings_by_camera[camera_id],
            build_masks=build_masks,
            label=label,
        )
        for camera_id in camera_ids
    ]
    for worker in workers:
        worker.start()
    return stop_event, workers


def stop_threaded_cameras(stop_event: threading.Event, workers: list[ThreadedMocapCamera]) -> None:
    stop_event.set()
    for worker in workers:
        worker.join()


def wait_for_open_attempts(workers: list[ThreadedMocapCamera], timeout_seconds: float = 8.0) -> None:
    start = time.time()
    while time.time() - start < timeout_seconds:
        snapshots = [worker.snapshot() for worker in workers]
        if any(snapshot.opened for snapshot in snapshots):
            return
        if all(snapshot.open_attempt_done for snapshot in snapshots):
            return
        time.sleep(0.03)


def collect_snapshots(workers: list[ThreadedMocapCamera]) -> dict[int, CameraSnapshot]:
    return {worker.camera_id: worker.snapshot() for worker in workers}


def frames_from_snapshots(snapshots: dict[int, CameraSnapshot]) -> dict[int, np.ndarray]:
    return {
        camera_id: snapshot.frame
        for camera_id, snapshot in snapshots.items()
        if snapshot.frame is not None
    }


def observations_from_snapshots(
    snapshots: dict[int, CameraSnapshot],
) -> dict[int, list[mocap.MarkerObservation]]:
    return {
        camera_id: list(snapshot.observations)
        for camera_id, snapshot in snapshots.items()
    }


def any_camera_open(snapshots: dict[int, CameraSnapshot]) -> bool:
    return any(snapshot.opened for snapshot in snapshots.values())


def all_open_attempts_done(snapshots: dict[int, CameraSnapshot]) -> bool:
    return bool(snapshots) and all(snapshot.open_attempt_done for snapshot in snapshots.values())


def draw_panel_title(panel: np.ndarray, title: str) -> None:
    cv2 = mocap.cv2
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(
        panel,
        title,
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
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
        0.8,
        (0, 0, 255),
        2,
        mocap.cv2.LINE_AA,
    )
    return panel


def draw_observation_dots(
    panel: np.ndarray,
    observations: list[mocap.MarkerObservation],
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    for index, observation in enumerate(observations, start=1):
        center = tuple(int(round(value)) for value in observation.pixel)
        radius = max(3, int(round(observation.radius_px)))
        mocap.cv2.circle(panel, center, radius, color, 2)
        mocap.cv2.putText(
            panel,
            str(index),
            (center[0] + 8, center[1] - 8),
            mocap.cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            mocap.cv2.LINE_AA,
        )


def mask_to_bgr(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    if len(mask.shape) == 2:
        return mocap.cv2.cvtColor(mask, mocap.cv2.COLOR_GRAY2BGR)
    return mask


def camera_role(camera_id: int, front_camera_ids: tuple[int, int], top_camera_ids: tuple[int, int]) -> str:
    if camera_id in front_camera_ids:
        return "front Y/Z"
    if camera_id in top_camera_ids:
        return "top X/Y"
    return "unassigned"


def build_four_camera_preview(
    camera_ids: list[int],
    snapshots: dict[int, CameraSnapshot],
    tracks: list[mocap.MarkerTrack] | None,
    settings_by_camera: dict[int, mocap.DetectionSettings],
    track_memory_pixels: float,
    panel_width: int,
    panel_height: int,
    front_camera_ids: tuple[int, int] = tuple(FRONT_CAMERA_IDS),
    top_camera_ids: tuple[int, int] = tuple(TOP_CAMERA_IDS),
) -> np.ndarray:
    sections: list[np.ndarray] = []
    for camera_id in camera_ids[:4]:
        snapshot = snapshots.get(camera_id)
        role = camera_role(camera_id, front_camera_ids, top_camera_ids)
        if snapshot is None or snapshot.frame is None:
            raw_panel = blank_panel(f"cam {camera_id} raw | {role}", panel_width, panel_height)
            binary_panel = blank_panel(f"cam {camera_id} threshold", panel_width, panel_height)
        else:
            observations = list(snapshot.observations)
            if tracks is None:
                raw_panel = snapshot.frame.copy()
                draw_observation_dots(raw_panel, observations)
            else:
                raw_panel = mocap.draw_preview(
                    snapshot.frame,
                    observations,
                    tracks,
                    camera_id,
                    track_memory_pixels,
                )
            draw_panel_title(raw_panel, f"cam {camera_id} raw | {role} | {snapshot.fps:.1f} fps")

            mask_bgr = mask_to_bgr(snapshot.mask)
            if mask_bgr is None:
                mask = mocap.threshold_mask(snapshot.frame, settings_by_camera[camera_id])
                mask_bgr = mask_to_bgr(mask)
            binary_panel = mask_bgr.copy()
            draw_observation_dots(binary_panel, observations)
            draw_panel_title(
                binary_panel,
                f"cam {camera_id} threshold | blobs {len(observations)} | {snapshot.detect_ms:.1f} ms",
            )

        raw_panel = resize_panel(raw_panel, panel_width, panel_height)
        binary_panel = resize_panel(binary_panel, panel_width, panel_height)
        sections.append(mocap.cv2.hconcat([raw_panel, binary_panel]))

    while len(sections) < 4:
        left = blank_panel("unused raw", panel_width, panel_height)
        right = blank_panel("unused threshold", panel_width, panel_height)
        sections.append(mocap.cv2.hconcat([left, right]))

    top_row = mocap.cv2.hconcat([sections[0], sections[1]])
    bottom_row = mocap.cv2.hconcat([sections[2], sections[3]])
    return mocap.cv2.vconcat([top_row, bottom_row])


@dataclass(slots=True)
class StereoPairCandidate:
    pair_name: str
    camera_ids: tuple[int, int]
    position: np.ndarray
    observations: tuple[mocap.MarkerObservation, mocap.MarkerObservation]
    reprojection_error_px: float


@dataclass(slots=True)
class FusionDiagnostics:
    front_candidates: int
    top_candidates: int
    y_matched_pairs: int
    fused_measurements: int


def build_stereo_pair_candidates(
    pair_name: str,
    pair_camera_ids: tuple[int, int],
    observations_by_camera: dict[int, list[mocap.MarkerObservation]],
    calibrations: dict[int, mocap.CameraCalibration],
    room_bounds,
    max_pair_error_px: float,
) -> list[StereoPairCandidate]:
    camera_a, camera_b = pair_camera_ids
    if camera_a not in calibrations or camera_b not in calibrations:
        return []

    observations_a = observations_by_camera.get(camera_a, [])
    observations_b = observations_by_camera.get(camera_b, [])
    if not observations_a or not observations_b:
        return []

    calib_a = calibrations[camera_a]
    calib_b = calibrations[camera_b]
    candidates: list[StereoPairCandidate] = []
    for obs_a in observations_a:
        for obs_b in observations_b:
            point = mocap.triangulate_two_views(obs_a, obs_b, calib_a, calib_b)
            if point is None or not mocap.point_is_inside_bounds(point, room_bounds):
                continue

            error = mocap.mean_reprojection_error(point, [obs_a, obs_b], calibrations)
            if error > max_pair_error_px:
                continue

            candidates.append(
                StereoPairCandidate(
                    pair_name=pair_name,
                    camera_ids=pair_camera_ids,
                    position=point,
                    observations=(obs_a, obs_b),
                    reprojection_error_px=error,
                )
            )

    candidates.sort(key=lambda candidate: candidate.reprojection_error_px)
    return candidates


def weighted_shared_y(front: StereoPairCandidate, top: StereoPairCandidate) -> float:
    front_weight = 1.0 / max(front.reprojection_error_px, 0.5)
    top_weight = 1.0 / max(top.reprojection_error_px, 0.5)
    return float(
        (front.position[1] * front_weight + top.position[1] * top_weight)
        / (front_weight + top_weight)
    )


def fused_observations(
    front: StereoPairCandidate,
    top: StereoPairCandidate,
) -> list[mocap.MarkerObservation]:
    by_camera: dict[int, mocap.MarkerObservation] = {}
    for observation in (*front.observations, *top.observations):
        by_camera.setdefault(observation.camera_id, observation)
    return list(by_camera.values())


def apply_scaling_defaults(args) -> None:
    base_scale = float(args.scaling_factor)
    axis_defaults = {
        "x_scaling_factor": DEFAULT_SCALING_FACTOR_X,
        "y_scaling_factor": DEFAULT_SCALING_FACTOR_Y,
        "z_scaling_factor": DEFAULT_SCALING_FACTOR_Z,
    }
    for name, axis_default in axis_defaults.items():
        if getattr(args, name, None) is None:
            if float(axis_default) == float(DEFAULT_SCALING_FACTOR):
                setattr(args, name, base_scale)
            else:
                setattr(args, name, float(axis_default))


def scaled_position(position: np.ndarray, args) -> np.ndarray:
    scale = np.array(
        [
            float(args.x_scaling_factor),
            float(args.y_scaling_factor),
            float(args.z_scaling_factor),
        ],
        dtype=np.float64,
    )
    return np.asarray(position, dtype=np.float64) * scale


def scale_text(args) -> str:
    return (
        "scale x/y/z="
        f"({float(args.x_scaling_factor):.2f}, "
        f"{float(args.y_scaling_factor):.2f}, "
        f"{float(args.z_scaling_factor):.2f})"
    )


def fuse_front_top_candidates(
    front_candidates: list[StereoPairCandidate],
    top_candidates: list[StereoPairCandidate],
    calibrations: dict[int, mocap.CameraCalibration],
    room_bounds,
    y_tolerance_m: float,
    max_fused_reprojection_error_px: float,
    max_measurements: int,
    min_measurement_separation_m: float = DEFAULT_MIN_MEASUREMENT_SEPARATION_M,
    reference_positions: list[np.ndarray] | None = None,
    track_bias_distance_m: float = DEFAULT_PAIRING_TRACK_BIAS_DISTANCE_M,
) -> tuple[list[mocap.MarkerMeasurement], int]:
    possible_matches: list[
        tuple[
            float,
            float,
            int,
            int,
            np.ndarray,
            list[mocap.MarkerObservation],
            float,
        ]
    ] = []
    tolerance = max(float(y_tolerance_m), 1e-6)
    references = [np.asarray(position, dtype=np.float64) for position in (reference_positions or [])]
    bias_distance = max(float(track_bias_distance_m), 0.0)
    for front_index, front in enumerate(front_candidates):
        for top_index, top in enumerate(top_candidates):
            y_delta = abs(float(front.position[1] - top.position[1]))
            if y_delta > y_tolerance_m:
                continue
            fused_position = np.array(
                [
                    top.position[0],
                    weighted_shared_y(front, top),
                    front.position[2],
                ],
                dtype=np.float64,
            )
            if not mocap.point_is_inside_bounds(fused_position, room_bounds):
                continue

            observations = fused_observations(front, top)
            fused_error = mocap.mean_reprojection_error(
                fused_position,
                observations,
                calibrations,
            )
            if fused_error > max_fused_reprojection_error_px:
                continue

            score = (
                y_delta / tolerance
                + 0.015 * front.reprojection_error_px
                + 0.015 * top.reprojection_error_px
                + 0.010 * fused_error
            )
            nearest_reference_distance = float("inf")
            if references:
                nearest_reference_distance = min(
                    float(np.linalg.norm(fused_position - reference))
                    for reference in references
                )
                if bias_distance > 0.0 and nearest_reference_distance <= bias_distance:
                    score = -1.0 + (nearest_reference_distance / bias_distance) + 0.10 * score
                elif bias_distance > 0.0:
                    score += 1.0

            possible_matches.append(
                (
                    score,
                    nearest_reference_distance,
                    front_index,
                    top_index,
                    fused_position,
                    observations,
                    fused_error,
                )
            )

    possible_matches.sort(key=lambda item: item[0])
    used_front: set[int] = set()
    used_top: set[int] = set()
    used_observation_ids: set[int] = set()
    measurements: list[mocap.MarkerMeasurement] = []
    min_separation = max(float(min_measurement_separation_m), 0.0)

    for (
        _score,
        _nearest_reference_distance,
        front_index,
        top_index,
        fused_position,
        observations,
        fused_error,
    ) in possible_matches:
        if front_index in used_front or top_index in used_top:
            continue
        if any(id(observation) in used_observation_ids for observation in observations):
            continue
        if any(
            float(np.linalg.norm(fused_position - measurement.position)) < min_separation
            for measurement in measurements
        ):
            continue

        measurements.append(
            mocap.MarkerMeasurement(
                position=fused_position,
                observations=observations,
                reprojection_error_px=fused_error,
            )
        )
        used_front.add(front_index)
        used_top.add(top_index)
        used_observation_ids.update(id(observation) for observation in observations)
        if len(measurements) >= max_measurements:
            break

    return measurements, len(possible_matches)


def fuse_layout_measurements(
    observations_by_camera: dict[int, list[mocap.MarkerObservation]],
    calibrations: dict[int, mocap.CameraCalibration],
    front_camera_ids: tuple[int, int],
    top_camera_ids: tuple[int, int],
    room_bounds,
    max_pair_error_px: float,
    y_tolerance_m: float = DEFAULT_FUSION_Y_TOLERANCE_M,
    max_fused_reprojection_error_px: float = DEFAULT_MAX_FUSED_REPROJECTION_ERROR_PX,
    max_measurements: int = DEFAULT_MAX_LAYOUT_MEASUREMENTS,
    min_measurement_separation_m: float = DEFAULT_MIN_MEASUREMENT_SEPARATION_M,
    reference_positions: list[np.ndarray] | None = None,
    track_bias_distance_m: float = DEFAULT_PAIRING_TRACK_BIAS_DISTANCE_M,
) -> tuple[list[mocap.MarkerMeasurement], FusionDiagnostics]:
    front_candidates = build_stereo_pair_candidates(
        "front",
        front_camera_ids,
        observations_by_camera,
        calibrations,
        room_bounds,
        max_pair_error_px,
    )
    top_candidates = build_stereo_pair_candidates(
        "top",
        top_camera_ids,
        observations_by_camera,
        calibrations,
        room_bounds,
        max_pair_error_px,
    )
    measurements, y_matched_pairs = fuse_front_top_candidates(
        front_candidates,
        top_candidates,
        calibrations,
        room_bounds,
        y_tolerance_m,
        max_fused_reprojection_error_px,
        max_measurements,
        min_measurement_separation_m,
        reference_positions,
        track_bias_distance_m,
    )
    diagnostics = FusionDiagnostics(
        front_candidates=len(front_candidates),
        top_candidates=len(top_candidates),
        y_matched_pairs=y_matched_pairs,
        fused_measurements=len(measurements),
    )
    return measurements, diagnostics


def load_calibrations(args, label: str) -> dict[int, mocap.CameraCalibration]:
    calibration_path = Path(args.calibration).expanduser() if args.calibration else None
    if calibration_path is not None and calibration_path.exists():
        print(f"[{label}] loading calibration from {calibration_path}")
        return mocap.load_calibration_file(
            str(calibration_path),
            args.width,
            args.height,
            args.focal_length_px,
        )

    if calibration_path is not None:
        print(f"[{label}] calibration file not found: {calibration_path}")
    print(f"[{label}] using placeholder calibration; run four_camera_calibration.py first for real positions")
    return mocap.build_default_room_calibrations(
        args.cameras,
        args.width,
        args.height,
        args.focal_length_px,
    )


def print_threaded_camera_stats(
    label: str,
    snapshots: dict[int, CameraSnapshot],
    min_interval_seconds: float,
    last_print_time: float,
) -> float:
    now = time.time()
    if now - last_print_time < min_interval_seconds:
        return last_print_time

    parts = []
    for camera_id, snapshot in sorted(snapshots.items()):
        if snapshot.opened:
            parts.append(
                f"camera {camera_id}: {snapshot.fps:.1f} fps, "
                f"read {snapshot.read_ms:.2f} ms, detect {snapshot.detect_ms:.2f} ms, "
                f"mask {snapshot.mask_ms:.2f} ms, blobs {len(snapshot.observations)}"
            )
        elif snapshot.error:
            parts.append(f"camera {camera_id}: {snapshot.error}")
    if parts:
        print(f"[{label} stats] " + " | ".join(parts))
    return now
