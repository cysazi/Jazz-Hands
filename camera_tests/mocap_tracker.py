"""
Standalone IR mocap prototype for the Jazz Hands camera pivot.

This file is intentionally not wired into the main JazzHands logic yet. It:
- Opens any available camera indexes from the requested list.
- Detects bright, circular IR-reflective blobs in each camera image.
- Triangulates marker positions when at least two calibrated cameras see a blob.
- Assigns simple persistent track IDs by nearest-neighbor matching.

World coordinate convention used here:
- X: player right
- Y: player forward
- Z: up
- Units: meters

Calibration matters. The default camera poses are only placeholders arranged
around the player so the pipeline can run immediately. Replace them with real
calibration data before trusting the reported 3D positions.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import cv2  # type: ignore[import-untyped]
except ImportError:  # Keep --help and py_compile usable without OpenCV installed.
    cv2 = None

try:
    import multithreaded_camera_testing as camera_settings
except ImportError:  # Keep this script usable if the test helper is copied alone.
    camera_settings = None

# Change this to [1, 2, 3, 4] when running the full camera rig.
CAMERA_IDS = list(camera_settings.CAMERA_IDS) if camera_settings is not None else [1, 2]
DEFAULT_CAMERA_IDS = CAMERA_IDS
FOUR_CAMERA_IDS = (
    list(camera_settings.FOUR_CAMERA_IDS)
    if camera_settings is not None
    else [1, 2, 3, 4]
)
DEFAULT_FRAME_WIDTH = camera_settings.FRAME_WIDTH if camera_settings is not None else 1280
DEFAULT_FRAME_HEIGHT = camera_settings.FRAME_HEIGHT if camera_settings is not None else 800
DEFAULT_FPS = camera_settings.FPS if camera_settings is not None else 120
DEFAULT_FOCAL_LENGTH_PX = 850.0
DEFAULT_ROOM_BOUNDS = ((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0))
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().with_name("mocap_calibration.json")
DEFAULT_TRACK_MEMORY_PIXEL_DISTANCE = 90.0
DEFAULT_MAX_MISSING_FRAMES = 120
DEFAULT_TRACK_CONFIRMATION_HITS = 3
DEFAULT_TENTATIVE_MAX_MISSING_FRAMES = 8
DEFAULT_DUPLICATE_TRACK_DISTANCE_M = 0.10
DEFAULT_VELOCITY_DAMPING = 0.25
DEFAULT_MISSING_VELOCITY_DECAY = 0.80
DEFAULT_STATIONARY_DISTANCE_M = 0.03
DEFAULT_MAX_PREDICTION_DT = 0.10
DEFAULT_THRESHOLD = camera_settings.THRESHOLD if camera_settings is not None else 230
DEFAULT_MIN_BLOB_AREA = (
    camera_settings.MIN_BLOB_AREA if camera_settings is not None else 0.0
)
DEFAULT_MAX_BLOB_AREA = (
    camera_settings.MAX_BLOB_AREA if camera_settings is not None else 4500.0
)
DEFAULT_MIN_BLOB_RADIUS = (
    camera_settings.MIN_BLOB_RADIUS if camera_settings is not None else 0.0
)
DEFAULT_MAX_BLOB_RADIUS = (
    camera_settings.MAX_BLOB_RADIUS if camera_settings is not None else 80.0
)
DEFAULT_MIN_BLOB_CIRCULARITY = (
    camera_settings.MIN_BLOB_CIRCULARITY if camera_settings is not None else 0.30
)
DEFAULT_MIN_BLOB_FILL_RATIO = (
    camera_settings.MIN_BLOB_FILL_RATIO if camera_settings is not None else 0.20
)
DEFAULT_MAX_BLOB_ASPECT_RATIO = (
    camera_settings.MAX_BLOB_ASPECT_RATIO if camera_settings is not None else 3.0
)
DEFAULT_MORPHOLOGY_KERNEL = (
    camera_settings.MORPHOLOGY_KERNEL if camera_settings is not None else 1
)
DEFAULT_EXPOSURE = (
    camera_settings.EXPOSURE_BY_CAMERA.get(1, -8)
    if camera_settings is not None
    else -8
)
DEFAULT_GAIN = (
    camera_settings.GAIN_BY_CAMERA.get(1, 0.0)
    if camera_settings is not None
    else 0.0
)
DEFAULT_AUTOFOCUS = camera_settings.AUTOFOCUS if camera_settings is not None else 0
BLUR_KERNEL_BY_CAMERA = (
    dict(camera_settings.BLUR_KERNEL_BY_CAMERA)
    if camera_settings is not None
    else {1: 5, 2: 5, 3: 15, 4: 15}
)
DEFAULT_AUTO_EXPOSURE_BY_CAMERA = (
    dict(camera_settings.AUTO_EXPOSURE_BY_CAMERA)
    if camera_settings is not None
    else {1: 0, 2: 0, 3: 0.25, 4: 0}
)
DEFAULT_EXPOSURE_BY_CAMERA = (
    dict(camera_settings.EXPOSURE_BY_CAMERA)
    if camera_settings is not None
    else {1: -8, 2: -8, 3: -8, 4: -8}
)
DEFAULT_GAIN_BY_CAMERA = (
    dict(camera_settings.GAIN_BY_CAMERA)
    if camera_settings is not None
    else {1: 0, 2: 0, 3: 0, 4: 0}
)


@dataclass(slots=True)
class DetectionSettings:
    threshold: int | None = None
    threshold_percentile: float = 99.75
    min_threshold: int = 150
    min_area: float = DEFAULT_MIN_BLOB_AREA
    max_area: float = DEFAULT_MAX_BLOB_AREA
    min_radius_px: float = DEFAULT_MIN_BLOB_RADIUS
    max_radius_px: float = DEFAULT_MAX_BLOB_RADIUS
    min_circularity: float = DEFAULT_MIN_BLOB_CIRCULARITY
    min_fill_ratio: float = DEFAULT_MIN_BLOB_FILL_RATIO
    max_aspect_ratio: float = DEFAULT_MAX_BLOB_ASPECT_RATIO
    min_brightness: float = 0.0
    blur_kernel: int = 3
    morphology_kernel: int = DEFAULT_MORPHOLOGY_KERNEL
    max_markers_per_camera: int = 12


@dataclass(slots=True)
class MarkerObservation:
    camera_id: int
    pixel: np.ndarray
    radius_px: float
    area_px: float
    circularity: float
    brightness: float
    score: float
    timestamp: float


@dataclass(slots=True)
class CameraCalibration:
    camera_id: int
    name: str
    intrinsic: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float64))

    @property
    def projection_matrix(self) -> np.ndarray:
        extrinsic = np.hstack((self.rotation, self.translation.reshape(3, 1)))
        return self.intrinsic @ extrinsic


@dataclass(slots=True)
class TriangulationCandidate:
    position: np.ndarray
    observations: tuple[MarkerObservation, MarkerObservation]
    reprojection_error_px: float


@dataclass(slots=True)
class MarkerMeasurement:
    position: np.ndarray
    observations: list[MarkerObservation]
    reprojection_error_px: float


@dataclass(slots=True)
class MarkerTrack:
    track_id: int
    position: np.ndarray
    velocity: np.ndarray
    observations: list[MarkerObservation]
    reprojection_error_px: float
    last_update: float
    age_frames: int = 1
    missing_frames: int = 0
    confidence: float = 1.0
    confirmed: bool = True
    total_hits: int = 1
    hit_streak: int = 1


class ReflectiveMarkerDetector:
    def __init__(self, settings: DetectionSettings):
        self.settings = settings

    def detect(self, frame: np.ndarray, camera_id: int, timestamp: float) -> list[MarkerObservation]:
        gray = self._to_gray(frame)
        mask = threshold_mask(frame, self.settings)

        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        observations: list[MarkerObservation] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.settings.min_area or area > self.settings.max_area:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-6:
                continue

            circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
            if circularity < self.settings.min_circularity:
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius <= 1e-6:
                continue
            if radius < self.settings.min_radius_px or radius > self.settings.max_radius_px:
                continue

            _bx, _by, width, height = cv2.boundingRect(contour)
            aspect_ratio = max(width, height) / max(min(width, height), 1)
            if aspect_ratio > self.settings.max_aspect_ratio:
                continue

            fill_ratio = float(area / (math.pi * radius * radius))
            if fill_ratio < self.settings.min_fill_ratio:
                continue

            moments = cv2.moments(contour)
            if abs(moments["m00"]) <= 1e-6:
                center = np.array([x, y], dtype=np.float64)
            else:
                center = np.array(
                    [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
                    dtype=np.float64,
                )

            contour_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            brightness = float(cv2.mean(gray, mask=contour_mask)[0])
            if brightness < self.settings.min_brightness:
                continue

            score = brightness * circularity * min(fill_ratio, 1.0) * math.sqrt(area)
            observations.append(
                MarkerObservation(
                    camera_id=camera_id,
                    pixel=center,
                    radius_px=float(radius),
                    area_px=area,
                    circularity=circularity,
                    brightness=brightness,
                    score=float(score),
                    timestamp=timestamp,
                )
            )

        observations.sort(key=lambda obs: obs.score, reverse=True)
        return observations[: self.settings.max_markers_per_camera]

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _preprocess(self, gray: np.ndarray) -> np.ndarray:
        kernel_size = self.settings.blur_kernel
        if kernel_size <= 1:
            return gray
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    def _choose_threshold(self, gray: np.ndarray) -> int:
        if self.settings.threshold is not None:
            return int(np.clip(self.settings.threshold, 0, 255))

        percentile_value = float(np.percentile(gray, self.settings.threshold_percentile))
        threshold = max(self.settings.min_threshold, int(percentile_value))
        return int(np.clip(threshold, 0, 255))


class MultiCameraTriangulator:
    def __init__(
        self,
        calibrations: dict[int, CameraCalibration],
        max_pair_error_px: float,
        cluster_distance_m: float,
        room_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    ):
        self.calibrations = calibrations
        self.max_pair_error_px = max_pair_error_px
        self.cluster_distance_m = cluster_distance_m
        self.room_bounds = room_bounds

    def triangulate(self, observations_by_camera: dict[int, list[MarkerObservation]]) -> list[MarkerMeasurement]:
        candidates = self._build_pair_candidates(observations_by_camera)
        clusters = self._cluster_candidates(candidates)
        measurements = [self._cluster_to_measurement(cluster) for cluster in clusters]
        measurements = [measurement for measurement in measurements if measurement is not None]
        measurements.sort(key=lambda measurement: measurement.reprojection_error_px)
        return measurements

    def _build_pair_candidates(
        self,
        observations_by_camera: dict[int, list[MarkerObservation]],
    ) -> list[TriangulationCandidate]:
        candidates: list[TriangulationCandidate] = []
        calibrated_camera_ids = [
            camera_id
            for camera_id, observations in observations_by_camera.items()
            if observations and camera_id in self.calibrations
        ]

        for camera_a, camera_b in combinations(calibrated_camera_ids, 2):
            calib_a = self.calibrations[camera_a]
            calib_b = self.calibrations[camera_b]
            for obs_a in observations_by_camera[camera_a]:
                for obs_b in observations_by_camera[camera_b]:
                    point = triangulate_two_views(obs_a, obs_b, calib_a, calib_b)
                    if point is None or not point_is_inside_bounds(point, self.room_bounds):
                        continue

                    error = mean_reprojection_error(point, [obs_a, obs_b], self.calibrations)
                    if error <= self.max_pair_error_px:
                        candidates.append(
                            TriangulationCandidate(
                                position=point,
                                observations=(obs_a, obs_b),
                                reprojection_error_px=error,
                            )
                        )

        candidates.sort(key=lambda candidate: candidate.reprojection_error_px)
        return candidates

    def _cluster_candidates(self, candidates: list[TriangulationCandidate]) -> list[list[TriangulationCandidate]]:
        clusters: list[list[TriangulationCandidate]] = []

        for candidate in candidates:
            best_cluster: list[TriangulationCandidate] | None = None
            best_distance = float("inf")

            for cluster in clusters:
                center = weighted_average_positions(cluster)
                distance = float(np.linalg.norm(candidate.position - center))
                if distance < best_distance:
                    best_cluster = cluster
                    best_distance = distance

            if best_cluster is not None and best_distance <= self.cluster_distance_m:
                best_cluster.append(candidate)
            else:
                clusters.append([candidate])

        return clusters

    def _cluster_to_measurement(self, cluster: list[TriangulationCandidate]) -> MarkerMeasurement | None:
        observations_by_camera: dict[int, MarkerObservation] = {}
        for candidate in sorted(cluster, key=lambda item: item.reprojection_error_px):
            for observation in candidate.observations:
                observations_by_camera.setdefault(observation.camera_id, observation)

        observations = list(observations_by_camera.values())
        if len(observations) < 2:
            return None

        point = triangulate_many_views(observations, self.calibrations)
        if point is None or not point_is_inside_bounds(point, self.room_bounds):
            return None

        error = mean_reprojection_error(point, observations, self.calibrations)
        if error > self.max_pair_error_px:
            return None

        return MarkerMeasurement(
            position=point,
            observations=observations,
            reprojection_error_px=error,
        )


class MarkerTracker:
    def __init__(
        self,
        max_match_distance_m: float = 0.35,
        smoothing: float = 0.65,
        max_missing_frames: int = DEFAULT_MAX_MISSING_FRAMES,
        min_confirmed_hits: int = DEFAULT_TRACK_CONFIRMATION_HITS,
        max_tentative_missing_frames: int = DEFAULT_TENTATIVE_MAX_MISSING_FRAMES,
        duplicate_track_distance_m: float = DEFAULT_DUPLICATE_TRACK_DISTANCE_M,
        velocity_damping: float = DEFAULT_VELOCITY_DAMPING,
        missing_velocity_decay: float = DEFAULT_MISSING_VELOCITY_DECAY,
        stationary_distance_m: float = DEFAULT_STATIONARY_DISTANCE_M,
        max_prediction_dt: float = DEFAULT_MAX_PREDICTION_DT,
    ):
        self.max_match_distance_m = max_match_distance_m
        self.smoothing = float(np.clip(smoothing, 0.0, 1.0))
        self.max_missing_frames = max_missing_frames
        self.min_confirmed_hits = max(1, min_confirmed_hits)
        self.max_tentative_missing_frames = max(0, max_tentative_missing_frames)
        self.duplicate_track_distance_m = max(0.0, duplicate_track_distance_m)
        self.velocity_damping = float(np.clip(velocity_damping, 0.0, 1.0))
        self.missing_velocity_decay = float(np.clip(missing_velocity_decay, 0.0, 1.0))
        self.stationary_distance_m = max(0.0, stationary_distance_m)
        self.max_prediction_dt = max(0.0, max_prediction_dt)
        self.tracks: list[MarkerTrack] = []
        self._next_track_id = 1

    def update(self, measurements: list[MarkerMeasurement], timestamp: float) -> list[MarkerTrack]:
        possible_matches: list[tuple[float, float, int, int]] = []

        for track_index, track in enumerate(self.tracks):
            predicted = self._predicted_position(track, timestamp)
            for measurement_index, measurement in enumerate(measurements):
                distance = float(np.linalg.norm(measurement.position - predicted))
                if distance <= self.max_match_distance_m:
                    match_score = distance * (0.90 if track.confirmed else 1.0)
                    possible_matches.append((match_score, distance, track_index, measurement_index))

        possible_matches.sort(key=lambda item: (item[0], item[1]))
        used_tracks: set[int] = set()
        used_measurements: set[int] = set()

        for _score, _distance, track_index, measurement_index in possible_matches:
            if track_index in used_tracks or measurement_index in used_measurements:
                continue
            self._update_track(self.tracks[track_index], measurements[measurement_index], timestamp)
            used_tracks.add(track_index)
            used_measurements.add(measurement_index)

        for track_index, track in enumerate(self.tracks):
            if track_index in used_tracks:
                continue
            track.missing_frames += 1
            track.hit_streak = 0
            track.confidence *= 0.85 if track.confirmed else 0.50
            track.velocity *= self.missing_velocity_decay

        for measurement_index, measurement in enumerate(measurements):
            if measurement_index in used_measurements:
                continue
            if self._is_duplicate_measurement(measurement, timestamp):
                continue
            self._add_track(measurement, timestamp)

        self.tracks = [track for track in self.tracks if self._should_keep_track(track)]
        self.tracks.sort(key=self._track_sort_key)
        return self.tracks

    def _update_track(self, track: MarkerTrack, measurement: MarkerMeasurement, timestamp: float) -> None:
        dt = max(timestamp - track.last_update, 1e-3)
        old_position = track.position.copy()
        smoothed_position = (
            (1.0 - self.smoothing) * track.position + self.smoothing * measurement.position
        )
        measured_step = float(np.linalg.norm(measurement.position - old_position))
        observed_velocity = (smoothed_position - old_position) / dt
        if measured_step <= self.stationary_distance_m:
            observed_velocity = np.zeros(3, dtype=np.float64)

        track.position = smoothed_position
        track.velocity = (
            (1.0 - self.velocity_damping) * track.velocity
            + self.velocity_damping * observed_velocity
        )
        track.observations = measurement.observations
        track.reprojection_error_px = measurement.reprojection_error_px
        track.last_update = timestamp
        track.age_frames += 1
        track.missing_frames = 0
        track.total_hits += 1
        track.hit_streak += 1
        track.confidence = min(1.0, track.confidence + (0.08 if track.confirmed else 0.18))
        self._maybe_confirm_track(track)

    def _add_track(self, measurement: MarkerMeasurement, timestamp: float) -> None:
        confirmed = self.min_confirmed_hits <= 1
        track = MarkerTrack(
            track_id=self._allocate_track_id() if confirmed else 0,
            position=measurement.position,
            velocity=np.zeros(3, dtype=np.float64),
            observations=measurement.observations,
            reprojection_error_px=measurement.reprojection_error_px,
            last_update=timestamp,
            confidence=1.0 if confirmed else 0.25,
            confirmed=confirmed,
        )
        self.tracks.append(track)

    def _allocate_track_id(self) -> int:
        track_id = self._next_track_id
        self._next_track_id += 1
        return track_id

    def _maybe_confirm_track(self, track: MarkerTrack) -> None:
        if track.confirmed:
            return
        if track.hit_streak < self.min_confirmed_hits:
            return
        track.track_id = self._allocate_track_id()
        track.confirmed = True
        track.confidence = max(track.confidence, 0.70)

    def _predicted_position(self, track: MarkerTrack, timestamp: float) -> np.ndarray:
        dt = max(timestamp - track.last_update, 0.0)
        prediction_dt = min(dt, self.max_prediction_dt)
        return track.position + track.velocity * prediction_dt

    def _is_duplicate_measurement(
        self,
        measurement: MarkerMeasurement,
        timestamp: float,
    ) -> bool:
        if self.duplicate_track_distance_m <= 0.0:
            return False

        for track in self.tracks:
            predicted = self._predicted_position(track, timestamp)
            distance = float(np.linalg.norm(measurement.position - predicted))
            if distance <= self.duplicate_track_distance_m:
                return True
        return False

    def _should_keep_track(self, track: MarkerTrack) -> bool:
        if track.confirmed:
            return track.missing_frames <= self.max_missing_frames
        return track.missing_frames <= self.max_tentative_missing_frames

    def _track_sort_key(self, track: MarkerTrack) -> tuple[bool, int, int]:
        if track.confirmed:
            return (False, track.track_id, -track.total_hits)
        return (True, 0, -track.total_hits)


class CameraSource:
    def __init__(
        self,
        camera_id: int,
        width: int,
        height: int,
        fps: int,
        exposure: float | None,
        auto_exposure: float | None,
        gain: float | None,
        configure_capture: bool = True,
    ):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.exposure = exposure
        self.auto_exposure = auto_exposure
        self.gain = gain
        self.configure_capture = configure_capture
        self.capture = None
        self._last_read_error_print_time = 0.0

    def open(self) -> bool:
        capture = cv2.VideoCapture(self.camera_id)
        if not capture.isOpened():
            capture.release()
            return False

        self.capture = capture
        if self.configure_capture:
            self._configure_capture()

        for _ in range(5):
            ok, frame = self.read()
            if ok and frame is not None:
                return True
            time.sleep(0.03)

        self.close()
        return False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.capture is None:
            return False, None
        try:
            return self.capture.read()
        except cv2.error as error:
            now = time.time()
            if now - self._last_read_error_print_time >= 1.0:
                self._last_read_error_print_time = now
                first_line = str(error).splitlines()[0]
                print(f"[camera {self.camera_id}] OpenCV read error skipped: {first_line}")
            return False, None

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _configure_capture(self) -> None:
        if self.capture is None:
            return

        if camera_settings is not None:
            camera_settings.apply_camera_capture_settings(
                self.capture,
                self.camera_id,
                width=self.width,
                height=self.height,
                fps=self.fps,
                exposure=self.exposure,
                auto_exposure=self.auto_exposure,
                gain=self.gain,
            )
            return

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        self.capture.set(cv2.CAP_PROP_AUTOFOCUS, DEFAULT_AUTOFOCUS)

        if self.auto_exposure is not None:
            self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(self.auto_exposure))
        if self.exposure is not None:
            self.capture.set(cv2.CAP_PROP_EXPOSURE, float(self.exposure))
        if self.gain is not None:
            self.capture.set(cv2.CAP_PROP_GAIN, float(self.gain))


def triangulate_two_views(
    obs_a: MarkerObservation,
    obs_b: MarkerObservation,
    calib_a: CameraCalibration,
    calib_b: CameraCalibration,
) -> np.ndarray | None:
    point_a = undistorted_pixel(obs_a, calib_a).reshape(2, 1)
    point_b = undistorted_pixel(obs_b, calib_b).reshape(2, 1)
    homogeneous = cv2.triangulatePoints(
        calib_a.projection_matrix,
        calib_b.projection_matrix,
        point_a,
        point_b,
    )

    weight = float(homogeneous[3, 0])
    if abs(weight) <= 1e-9:
        return None

    return (homogeneous[:3, 0] / weight).astype(np.float64)


def triangulate_many_views(
    observations: list[MarkerObservation],
    calibrations: dict[int, CameraCalibration],
) -> np.ndarray | None:
    rows: list[np.ndarray] = []

    for observation in observations:
        calibration = calibrations.get(observation.camera_id)
        if calibration is None:
            continue

        pixel = undistorted_pixel(observation, calibration)
        projection = calibration.projection_matrix
        rows.append(pixel[0] * projection[2, :] - projection[0, :])
        rows.append(pixel[1] * projection[2, :] - projection[1, :])

    if len(rows) < 4:
        return None

    matrix = np.asarray(rows, dtype=np.float64)
    _u, _s, vh = np.linalg.svd(matrix)
    homogeneous = vh[-1, :]
    if abs(float(homogeneous[3])) <= 1e-9:
        return None

    return (homogeneous[:3] / homogeneous[3]).astype(np.float64)


def undistorted_pixel(observation: MarkerObservation, calibration: CameraCalibration) -> np.ndarray:
    if calibration.dist_coeffs.size == 0 or np.allclose(calibration.dist_coeffs, 0.0):
        return observation.pixel.astype(np.float64)

    points = observation.pixel.reshape(1, 1, 2).astype(np.float64)
    corrected = cv2.undistortPoints(
        points,
        calibration.intrinsic,
        calibration.dist_coeffs,
        P=calibration.intrinsic,
    )
    return corrected.reshape(2).astype(np.float64)


def project_point(point: np.ndarray, calibration: CameraCalibration) -> np.ndarray | None:
    camera_point = calibration.rotation @ point.reshape(3) + calibration.translation.reshape(3)
    if camera_point[2] <= 1e-6:
        return None

    projected = calibration.intrinsic @ camera_point
    return np.array([projected[0] / projected[2], projected[1] / projected[2]], dtype=np.float64)


def mean_reprojection_error(
    point: np.ndarray,
    observations: Iterable[MarkerObservation],
    calibrations: dict[int, CameraCalibration],
) -> float:
    errors: list[float] = []
    for observation in observations:
        calibration = calibrations.get(observation.camera_id)
        if calibration is None:
            continue
        projected = project_point(point, calibration)
        if projected is None:
            errors.append(float("inf"))
        else:
            errors.append(float(np.linalg.norm(projected - observation.pixel)))

    if not errors:
        return float("inf")
    return float(np.mean(errors))


def weighted_average_positions(candidates: list[TriangulationCandidate]) -> np.ndarray:
    weights = np.array(
        [1.0 / max(candidate.reprojection_error_px, 0.001) for candidate in candidates],
        dtype=np.float64,
    )
    positions = np.array([candidate.position for candidate in candidates], dtype=np.float64)
    return np.average(positions, axis=0, weights=weights)


def point_is_inside_bounds(
    point: np.ndarray,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> bool:
    return all(bounds[axis][0] <= float(point[axis]) <= bounds[axis][1] for axis in range(3))


def build_default_room_calibrations(
    camera_ids: list[int],
    frame_width: int,
    frame_height: int,
    focal_length_px: float,
) -> dict[int, CameraCalibration]:
    intrinsic = default_intrinsic(frame_width, frame_height, focal_length_px)
    layout = [
        ("front", np.array([0.0, -1.8, 1.25]), np.array([0.0, 0.0, 0.95])),
        ("right", np.array([1.8, 0.0, 1.25]), np.array([0.0, 0.0, 0.95])),
        ("back", np.array([0.0, 1.8, 1.25]), np.array([0.0, 0.0, 0.95])),
        ("left", np.array([-1.8, 0.0, 1.25]), np.array([0.0, 0.0, 0.95])),
    ]

    calibrations: dict[int, CameraCalibration] = {}
    for index, camera_id in enumerate(camera_ids):
        name, position, look_at = layout[index % len(layout)]
        rotation, translation = look_at_extrinsics(position, look_at)
        calibrations[camera_id] = CameraCalibration(
            camera_id=camera_id,
            name=f"{name}_{camera_id}",
            intrinsic=intrinsic.copy(),
            rotation=rotation,
            translation=translation,
        )

    return calibrations


def default_intrinsic(frame_width: int, frame_height: int, focal_length_px: float) -> np.ndarray:
    return np.array(
        [
            [focal_length_px, 0.0, frame_width / 2.0],
            [0.0, focal_length_px, frame_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def look_at_extrinsics(position: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position = position.astype(np.float64)
    target = target.astype(np.float64)
    forward = normalize(target - position)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    if abs(float(np.dot(forward, world_up))) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    right = normalize(np.cross(forward, world_up))
    down = normalize(np.cross(forward, right))

    camera_to_world = np.column_stack((right, down, forward))
    rotation = camera_to_world.T
    translation = -rotation @ position
    return rotation.astype(np.float64), translation.astype(np.float64)


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vector / norm


def load_calibration_file(
    path: str,
    fallback_frame_width: int,
    fallback_frame_height: int,
    fallback_focal_length_px: float,
) -> dict[int, CameraCalibration]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        camera_items = data.get("cameras", [])
        fallback_frame_width = int(data.get("frame_width", fallback_frame_width))
        fallback_frame_height = int(data.get("frame_height", fallback_frame_height))
        fallback_focal_length_px = float(data.get("focal_length_px", fallback_focal_length_px))
    else:
        camera_items = data

    default_k = default_intrinsic(
        fallback_frame_width,
        fallback_frame_height,
        fallback_focal_length_px,
    )
    calibrations: dict[int, CameraCalibration] = {}

    for item in camera_items:
        camera_id = int(item.get("id", item.get("camera_id")))
        name = str(item.get("name", f"camera_{camera_id}"))
        intrinsic_data = item.get("intrinsic", item.get("K"))
        intrinsic = np.asarray(intrinsic_data, dtype=np.float64) if intrinsic_data else default_k.copy()
        dist_coeffs = np.asarray(item.get("dist_coeffs", []), dtype=np.float64)

        if "rotation" in item:
            rotation = np.asarray(item["rotation"], dtype=np.float64)
        elif "rvec" in item:
            rotation, _jacobian = cv2.Rodrigues(np.asarray(item["rvec"], dtype=np.float64))
        elif "position" in item:
            position = np.asarray(item["position"], dtype=np.float64)
            look_at = np.asarray(item.get("look_at", [0.0, 0.0, 1.0]), dtype=np.float64)
            rotation, translation = look_at_extrinsics(position, look_at)
            calibrations[camera_id] = CameraCalibration(
                camera_id=camera_id,
                name=name,
                intrinsic=intrinsic,
                rotation=rotation,
                translation=translation,
                dist_coeffs=dist_coeffs,
            )
            continue
        else:
            raise ValueError(
                f"Camera {camera_id} needs rotation/rvec or position/look_at calibration."
            )

        if "translation" in item:
            translation = np.asarray(item["translation"], dtype=np.float64)
        elif "tvec" in item:
            translation = np.asarray(item["tvec"], dtype=np.float64)
        else:
            raise ValueError(f"Camera {camera_id} needs translation or tvec calibration.")

        calibrations[camera_id] = CameraCalibration(
            camera_id=camera_id,
            name=name,
            intrinsic=intrinsic,
            rotation=rotation,
            translation=translation,
            dist_coeffs=dist_coeffs,
        )

    return calibrations


def open_available_cameras(
    camera_ids: list[int],
    width: int,
    height: int,
    fps: int,
    exposure: float | None,
    auto_exposure: float | None,
    gain: float | None,
) -> list[CameraSource]:
    sources: list[CameraSource] = []
    for camera_id in camera_ids:
        source = CameraSource(
            camera_id,
            width,
            height,
            fps,
            camera_exposure(camera_id, exposure),
            camera_auto_exposure(camera_id, auto_exposure),
            camera_gain(camera_id, gain),
            configure_capture=True,
        )
        if source.open():
            sources.append(source)
            print_camera_settings(source)
        else:
            print(f"[mocap] camera {camera_id} not available; continuing")
    return sources


def camera_setting(
    camera_id: int,
    values_by_camera: dict[int, float],
    override: float | None,
    fallback: float,
) -> float:
    if override is not None:
        return float(override)
    return float(values_by_camera.get(camera_id, fallback))


def camera_auto_exposure(camera_id: int, override: float | None) -> float:
    return camera_setting(camera_id, DEFAULT_AUTO_EXPOSURE_BY_CAMERA, override, 0.0)


def camera_exposure(camera_id: int, override: float | None) -> float:
    return camera_setting(camera_id, DEFAULT_EXPOSURE_BY_CAMERA, override, DEFAULT_EXPOSURE)


def camera_gain(camera_id: int, override: float | None) -> float:
    return camera_setting(camera_id, DEFAULT_GAIN_BY_CAMERA, override, DEFAULT_GAIN)


def apply_camera_settings(
    source: CameraSource,
    auto_exposure: float | None,
    exposure: float | None,
    gain: float | None,
) -> None:
    if source.capture is None:
        return
    if camera_settings is not None:
        camera_settings.apply_camera_capture_settings(
            source.capture,
            source.camera_id,
            width=source.width,
            height=source.height,
            fps=source.fps,
            exposure=exposure,
            auto_exposure=auto_exposure,
            gain=gain,
        )
        return

    if auto_exposure is not None:
        source.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure)
    source.capture.set(cv2.CAP_PROP_AUTOFOCUS, DEFAULT_AUTOFOCUS)
    if exposure is not None:
        source.capture.set(cv2.CAP_PROP_EXPOSURE, exposure)
    if gain is not None:
        source.capture.set(cv2.CAP_PROP_GAIN, gain)


def print_camera_settings(source: CameraSource) -> None:
    if source.capture is None:
        return

    print(
        f"[mocap] opened camera {source.camera_id} | "
        f"auto_exposure={source.capture.get(cv2.CAP_PROP_AUTO_EXPOSURE):.2f} "
        f"autofocus={source.capture.get(cv2.CAP_PROP_AUTOFOCUS):.2f} "
        f"exposure={source.capture.get(cv2.CAP_PROP_EXPOSURE):.2f} "
        f"gain={source.capture.get(cv2.CAP_PROP_GAIN):.2f}"
    )


def build_detection_settings(args: argparse.Namespace, camera_id: int) -> DetectionSettings:
    return DetectionSettings(
        threshold=args.threshold,
        min_area=args.min_area,
        max_area=args.max_area,
        min_radius_px=args.min_radius,
        max_radius_px=args.max_radius,
        min_circularity=args.min_circularity,
        min_fill_ratio=args.min_fill_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
        min_brightness=args.min_brightness,
        blur_kernel=BLUR_KERNEL_BY_CAMERA.get(camera_id, 1),
        morphology_kernel=DEFAULT_MORPHOLOGY_KERNEL,
        max_markers_per_camera=args.max_markers_per_camera,
    )


def lock_observations_to_existing_tracks(
    frames: dict[int, np.ndarray],
    observations_by_camera: dict[int, list[MarkerObservation]],
    tracks: list[MarkerTrack],
    settings: DetectionSettings | dict[int, DetectionSettings],
    box_radius_px: float,
    timestamp: float,
) -> None:
    used_observation_ids_by_camera: dict[int, set[int]] = {
        camera_id: set() for camera_id in frames
    }

    for track in tracks:
        if not track.observations:
            continue

        for previous_observation in list(track.observations):
            camera_id = previous_observation.camera_id
            frame = frames.get(camera_id)
            if frame is None:
                continue

            used_ids = used_observation_ids_by_camera.setdefault(camera_id, set())
            camera_observations = observations_by_camera.setdefault(camera_id, [])
            observation = nearest_observation(
                previous_observation.pixel,
                camera_observations,
                used_ids,
                box_radius_px,
            )

            if observation is None:
                observation = detect_blob_inside_box(
                    frame,
                    previous_observation,
                    settings_for_camera(settings, camera_id),
                    box_radius_px,
                    timestamp,
                )
                if observation is not None:
                    camera_observations.insert(0, observation)

            if observation is None:
                continue

            used_ids.add(id(observation))
            remember_track_observation(track, camera_id, observation)


def settings_for_camera(
    settings: DetectionSettings | dict[int, DetectionSettings],
    camera_id: int,
) -> DetectionSettings:
    if isinstance(settings, dict):
        camera_settings = settings.get(camera_id)
        if camera_settings is not None:
            return camera_settings
        return next(iter(settings.values()))
    return settings


def detect_blob_inside_box(
    frame: np.ndarray,
    previous_observation: MarkerObservation,
    settings: DetectionSettings,
    box_radius_px: float,
    timestamp: float,
) -> MarkerObservation | None:
    gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = threshold_mask(frame, settings)

    x_center, y_center = previous_observation.pixel
    height, width = mask.shape[:2]
    x_min = max(0, int(round(x_center - box_radius_px)))
    x_max = min(width, int(round(x_center + box_radius_px)))
    y_min = max(0, int(round(y_center - box_radius_px)))
    y_max = min(height, int(round(y_center + box_radius_px)))
    if x_max <= x_min or y_max <= y_min:
        return None

    roi_mask = mask[y_min:y_max, x_min:x_max]
    contours, _hierarchy = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_observation: MarkerObservation | None = None
    best_score = float("inf")

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < settings.min_area or area > settings.max_area * 4.0:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius <= 0.0 or radius > settings.max_radius_px * 2.0:
            continue

        moments = cv2.moments(contour)
        if abs(moments["m00"]) <= 1e-6:
            pixel = np.array([x + x_min, y + y_min], dtype=np.float64)
        else:
            pixel = np.array(
                [
                    moments["m10"] / moments["m00"] + x_min,
                    moments["m01"] / moments["m00"] + y_min,
                ],
                dtype=np.float64,
            )

        distance = float(np.linalg.norm(pixel - previous_observation.pixel))
        if distance > box_radius_px:
            continue

        contour_mask = np.zeros(roi_mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        roi_gray = gray[y_min:y_max, x_min:x_max]
        brightness = float(cv2.mean(roi_gray, mask=contour_mask)[0])
        score = distance - 0.01 * brightness

        if score < best_score:
            best_score = score
            best_observation = MarkerObservation(
                camera_id=previous_observation.camera_id,
                pixel=pixel,
                radius_px=float(radius),
                area_px=area,
                circularity=previous_observation.circularity,
                brightness=brightness,
                score=float(brightness * max(area, 1.0)),
                timestamp=timestamp,
            )

    return best_observation


def preprocess_gray(gray: np.ndarray, blur_kernel: int) -> np.ndarray:
    if camera_settings is not None:
        return camera_settings.threshold_source_gray(gray, 0, blur_kernel=blur_kernel)

    if blur_kernel <= 1:
        return gray
    blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    return cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)


def choose_threshold(gray: np.ndarray, settings: DetectionSettings) -> int:
    if settings.threshold is not None:
        return int(np.clip(settings.threshold, 0, 255))
    percentile_value = float(np.percentile(gray, settings.threshold_percentile))
    threshold = max(settings.min_threshold, int(percentile_value))
    return int(np.clip(threshold, 0, 255))


def threshold_mask(frame: np.ndarray, settings: DetectionSettings) -> np.ndarray:
    if camera_settings is not None:
        return camera_settings.build_threshold_mask(
            frame,
            0,
            threshold=settings.threshold,
            blur_kernel=settings.blur_kernel,
        )

    gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed = preprocess_gray(gray, settings.blur_kernel)
    threshold = choose_threshold(processed, settings)
    _ok, mask = cv2.threshold(processed, threshold, 255, cv2.THRESH_BINARY)

    kernel_size = settings.morphology_kernel
    if kernel_size > 1:
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def setup_mocap_controls(args: argparse.Namespace) -> None:
    _ = args
    return


def apply_mocap_controls(
    sources: list[CameraSource],
    settings: DetectionSettings,
    last_applied: dict[int, tuple[float, float, float]],
) -> None:
    _ = sources, settings, last_applied
    return


def draw_preview(
    frame: np.ndarray,
    observations: list[MarkerObservation],
    tracks: list[MarkerTrack],
    camera_id: int,
    track_memory_distance_px: float,
) -> np.ndarray:
    preview = frame.copy()
    used_observation_ids: set[int] = set()

    for track in tracks:
        if not track.confirmed:
            continue
        observation = observation_for_track(
            track,
            observations,
            camera_id,
            used_observation_ids,
            track_memory_distance_px,
        )
        if observation is None:
            continue

        used_observation_ids.add(id(observation))
        center = tuple(int(round(value)) for value in observation.pixel)
        radius = max(3, int(round(observation.radius_px)))
        cv2.circle(preview, center, radius, (0, 255, 0), 2)
        cv2.putText(
            preview,
            f"id {track.track_id}",
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        preview,
        f"camera {camera_id} | tracked blobs only",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return preview


def observation_for_track(
    track: MarkerTrack,
    observations: list[MarkerObservation],
    camera_id: int,
    used_observation_ids: set[int],
    track_memory_distance_px: float,
) -> MarkerObservation | None:
    current_track_observation = next(
        (
            observation
            for observation in track.observations
            if observation.camera_id == camera_id and id(observation) not in used_observation_ids
        ),
        None,
    )
    if track.missing_frames == 0 or current_track_observation is None:
        return current_track_observation

    nearest = nearest_observation(
        current_track_observation.pixel,
        observations,
        used_observation_ids,
        track_memory_distance_px,
    )
    if nearest is None:
        return None

    remember_track_observation(track, camera_id, nearest)
    return nearest


def nearest_observation(
    pixel: np.ndarray,
    observations: list[MarkerObservation],
    used_observation_ids: set[int],
    max_distance_px: float,
) -> MarkerObservation | None:
    best_observation: MarkerObservation | None = None
    best_distance = float("inf")

    for observation in observations:
        if id(observation) in used_observation_ids:
            continue
        distance = float(np.linalg.norm(observation.pixel - pixel))
        if distance < best_distance:
            best_observation = observation
            best_distance = distance

    if best_distance <= max_distance_px:
        return best_observation
    return None


def remember_track_observation(
    track: MarkerTrack,
    camera_id: int,
    observation: MarkerObservation,
) -> None:
    for index, existing in enumerate(track.observations):
        if existing.camera_id == camera_id:
            track.observations[index] = observation
            return
    track.observations.append(observation)


def print_status(
    tracks: list[MarkerTrack],
    observations_by_camera: dict[int, list[MarkerObservation]],
    calibrated_camera_count: int,
    triangulator: MultiCameraTriangulator,
) -> None:
    blob_counts = ", ".join(
        f"cam {camera_id}: {len(observations)}"
        for camera_id, observations in sorted(observations_by_camera.items())
    )
    live_tracks = [
        track for track in tracks if track.confirmed and track.missing_frames == 0
    ]
    tentative_tracks = [
        track for track in tracks if not track.confirmed and track.missing_frames == 0
    ]
    print(
        f"[mocap] calibrated cams: {calibrated_camera_count} | blobs: {blob_counts} | "
        f"live ids: {len(live_tracks)} tentative: {len(tentative_tracks)}"
    )

    if not live_tracks:
        print("[mocap] no 3D tracks yet")
        diagnostics = triangulation_diagnostics(
            observations_by_camera,
            triangulator.calibrations,
            triangulator.room_bounds,
            triangulator.max_pair_error_px,
        )
        for line in diagnostics:
            print(line)
        return

    for track in live_tracks:
        cameras = sorted({observation.camera_id for observation in track.observations})
        x, y, z = track.position
        print(
            "[mocap] "
            f"id {track.track_id}: "
            f"x={x:+.3f}m y={y:+.3f}m z={z:+.3f}m "
            f"cams={cameras} err={track.reprojection_error_px:.1f}px "
            f"conf={track.confidence:.2f}"
        )


def triangulation_diagnostics(
    observations_by_camera: dict[int, list[MarkerObservation]],
    calibrations: dict[int, CameraCalibration],
    room_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    max_pair_error_px: float,
) -> list[str]:
    calibrated_camera_ids = [
        camera_id
        for camera_id, observations in sorted(observations_by_camera.items())
        if observations and camera_id in calibrations
    ]
    if len(calibrated_camera_ids) < 2:
        return ["[mocap] triangulation: need blobs in at least two calibrated cameras"]

    lines: list[str] = []
    for camera_a, camera_b in combinations(calibrated_camera_ids, 2):
        obs_a = observations_by_camera[camera_a][0]
        obs_b = observations_by_camera[camera_b][0]
        point = triangulate_two_views(
            obs_a,
            obs_b,
            calibrations[camera_a],
            calibrations[camera_b],
        )
        if point is None:
            lines.append(f"[mocap] pair {camera_a}-{camera_b}: triangulation failed")
            continue

        in_bounds = point_is_inside_bounds(point, room_bounds)
        error = mean_reprojection_error(point, [obs_a, obs_b], calibrations)
        x, y, z = point
        reason = "accepted"
        if not in_bounds:
            reason = "rejected by room bounds"
        elif error > max_pair_error_px:
            reason = f"rejected by reprojection error > {max_pair_error_px:.1f}px"

        lines.append(
            "[mocap] "
            f"pair {camera_a}-{camera_b}: {reason}; "
            f"point=({x:+.3f},{y:+.3f},{z:+.3f})m err={error:.1f}px"
        )
    return lines[:3]


def parse_camera_ids(text: str) -> list[int]:
    values = [value.strip() for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one camera id.")
    return [int(value) for value in values]


def parse_room_bounds(text: str) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if len(values) != 6:
        raise argparse.ArgumentTypeError(
            "Room bounds must be six comma-separated values: xmin,xmax,ymin,ymax,zmin,zmax"
        )
    return ((values[0], values[1]), (values[2], values[3]), (values[4], values[5]))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track IR-reflective mocap marker balls from a variable number of cameras.",
    )
    parser.add_argument(
        "--cameras",
        type=parse_camera_ids,
        default=DEFAULT_CAMERA_IDS,
        help=(
            "Comma-separated OpenCV camera indexes to try. "
            "Default uses CAMERA_IDS near the top of this file."
        ),
    )
    parser.add_argument("--width", type=int, default=DEFAULT_FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_FRAME_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--exposure",
        type=float,
        default=None,
        help="Override manual exposure for every camera. Default uses the camera-test values.",
    )
    parser.add_argument(
        "--auto-exposure",
        type=float,
        default=None,
        help=(
            "Override auto-exposure value for every camera. Default uses per-camera "
            "camera-test values: camera 1 -> 0.0, camera 2 -> 0.0, camera 3 -> 0.25."
        ),
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=None,
        help="Override gain for every camera. Default uses the camera-test values.",
    )
    parser.add_argument(
        "--calibration",
        default=str(DEFAULT_CALIBRATION_PATH),
        help=(
            "JSON calibration file. By default, uses mocap_calibration.json "
            "beside this script when it exists; otherwise placeholder poses are used."
        ),
    )
    parser.add_argument("--focal-length-px", type=float, default=DEFAULT_FOCAL_LENGTH_PX)
    parser.add_argument(
        "--room-bounds",
        type=parse_room_bounds,
        default=DEFAULT_ROOM_BOUNDS,
        help="xmin,xmax,ymin,ymax,zmin,zmax in meters. Default: -5,5,-5,5,-5,5",
    )
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--min-area", type=float, default=DEFAULT_MIN_BLOB_AREA)
    parser.add_argument("--max-area", type=float, default=DEFAULT_MAX_BLOB_AREA)
    parser.add_argument("--min-radius", type=float, default=DEFAULT_MIN_BLOB_RADIUS)
    parser.add_argument("--max-radius", type=float, default=DEFAULT_MAX_BLOB_RADIUS)
    parser.add_argument("--min-circularity", type=float, default=DEFAULT_MIN_BLOB_CIRCULARITY)
    parser.add_argument("--min-fill-ratio", type=float, default=DEFAULT_MIN_BLOB_FILL_RATIO)
    parser.add_argument("--max-aspect-ratio", type=float, default=DEFAULT_MAX_BLOB_ASPECT_RATIO)
    parser.add_argument("--min-brightness", type=float, default=0.0)
    parser.add_argument("--max-markers-per-camera", type=int, default=12)
    parser.add_argument("--max-reprojection-error", type=float, default=45.0)
    parser.add_argument("--cluster-distance", type=float, default=0.45)
    parser.add_argument("--track-distance", type=float, default=0.75)
    parser.add_argument(
        "--track-confirmation-hits",
        type=int,
        default=DEFAULT_TRACK_CONFIRMATION_HITS,
        help="How many consecutive 3D hits a new marker needs before receiving a visible ID.",
    )
    parser.add_argument(
        "--tentative-max-missing-frames",
        type=int,
        default=DEFAULT_TENTATIVE_MAX_MISSING_FRAMES,
        help="How long to keep an unconfirmed candidate if it stops matching.",
    )
    parser.add_argument(
        "--duplicate-track-distance",
        type=float,
        default=DEFAULT_DUPLICATE_TRACK_DISTANCE_M,
        help="Suppress new candidate tracks this close to an existing predicted track.",
    )
    parser.add_argument(
        "--velocity-damping",
        type=float,
        default=DEFAULT_VELOCITY_DAMPING,
        help="How strongly each new measurement updates track velocity. Lower is steadier.",
    )
    parser.add_argument(
        "--stationary-distance",
        type=float,
        default=DEFAULT_STATIONARY_DISTANCE_M,
        help="Ignore per-frame 3D movement smaller than this when estimating velocity.",
    )
    parser.add_argument(
        "--max-prediction-dt",
        type=float,
        default=DEFAULT_MAX_PREDICTION_DT,
        help="Cap velocity prediction time so missing tracks do not coast too far.",
    )
    parser.add_argument(
        "--max-missing-frames",
        type=int,
        default=DEFAULT_MAX_MISSING_FRAMES,
        help="How long to keep a locked track alive without a fresh 3D triangulation.",
    )
    parser.add_argument(
        "--track-memory-pixels",
        type=float,
        default=DEFAULT_TRACK_MEMORY_PIXEL_DISTANCE,
        help=(
            "If a 3D track briefly disappears, draw a nearby 2D blob in the same "
            "camera as that same green tracked blob."
        ),
    )
    parser.add_argument("--print-interval", type=float, default=0.25)
    parser.add_argument("--no-preview", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if cv2 is None:
        print("OpenCV is required for camera mocap. Install it with: python -m pip install opencv-python")
        return 1

    calibration_path = Path(args.calibration).expanduser() if args.calibration else None
    if calibration_path is not None and calibration_path.exists():
        print(f"[mocap] loading calibration from {calibration_path}")
        calibrations = load_calibration_file(
            str(calibration_path),
            args.width,
            args.height,
            args.focal_length_px,
        )
    else:
        if calibration_path is not None:
            print(f"[mocap] calibration file not found: {calibration_path}")
        print(
            "[mocap] using placeholder camera calibration. "
            "Run calibrate_mocap_cameras.py first for real 3D positions."
        )
        calibrations = build_default_room_calibrations(
            args.cameras,
            args.width,
            args.height,
            args.focal_length_px,
        )

    sources = open_available_cameras(
        args.cameras,
        args.width,
        args.height,
        args.fps,
        args.exposure,
        args.auto_exposure,
        args.gain,
    )
    if not sources:
        print("[mocap] no cameras opened")
        return 1

    connected_ids = {source.camera_id for source in sources}
    calibrated_connected_ids = connected_ids & set(calibrations)
    if len(calibrated_connected_ids) < 2:
        print(
            "[mocap] fewer than two connected cameras have calibration. "
            "2D blobs will show, but 3D tracks need at least two calibrated views."
        )

    settings_by_camera = {
        source.camera_id: build_detection_settings(args, source.camera_id)
        for source in sources
    }
    detectors = {
        camera_id: ReflectiveMarkerDetector(settings)
        for camera_id, settings in settings_by_camera.items()
    }
    triangulator = MultiCameraTriangulator(
        calibrations=calibrations,
        max_pair_error_px=args.max_reprojection_error,
        cluster_distance_m=args.cluster_distance,
        room_bounds=args.room_bounds,
    )
    tracker = MarkerTracker(
        max_match_distance_m=args.track_distance,
        max_missing_frames=args.max_missing_frames,
        min_confirmed_hits=args.track_confirmation_hits,
        max_tentative_missing_frames=args.tentative_max_missing_frames,
        duplicate_track_distance_m=args.duplicate_track_distance,
        velocity_damping=args.velocity_damping,
        stationary_distance_m=args.stationary_distance,
        max_prediction_dt=args.max_prediction_dt,
    )

    show_preview = not args.no_preview
    last_print_time = 0.0

    try:
        while True:
            timestamp = time.time()
            frames: dict[int, np.ndarray] = {}
            observations_by_camera: dict[int, list[MarkerObservation]] = {}

            for source in sources:
                ok, frame = source.read()
                if not ok or frame is None:
                    observations_by_camera[source.camera_id] = []
                    continue

                frames[source.camera_id] = frame
                observations_by_camera[source.camera_id] = detectors[source.camera_id].detect(
                    frame,
                    source.camera_id,
                    timestamp,
                )

            lock_observations_to_existing_tracks(
                frames,
                observations_by_camera,
                tracker.tracks,
                settings_by_camera,
                args.track_memory_pixels,
                timestamp,
            )
            measurements = triangulator.triangulate(observations_by_camera)
            tracks = tracker.update(measurements, timestamp)

            if timestamp - last_print_time >= args.print_interval:
                last_print_time = timestamp
                print_status(
                    tracks,
                    observations_by_camera,
                    calibrated_camera_count=len(calibrated_connected_ids),
                    triangulator=triangulator,
                )

            if show_preview:
                for camera_id, frame in frames.items():
                    preview = draw_preview(
                        frame,
                        observations_by_camera.get(camera_id, []),
                        tracks,
                        camera_id,
                        args.track_memory_pixels,
                    )
                    cv2.imshow(f"mocap camera {camera_id}", preview)
                    cv2.imshow(
                        f"mocap binary camera {camera_id}",
                        threshold_mask(frame, settings_by_camera[camera_id]),
                    )

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    except KeyboardInterrupt:
        print("\n[mocap] stopped")
    finally:
        for source in sources:
            source.close()
        if cv2 is not None:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
