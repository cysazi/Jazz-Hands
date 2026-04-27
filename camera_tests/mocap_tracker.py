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
import platform
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


DEFAULT_CAMERA_IDS = [1, 2, 3, 4]
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 800
DEFAULT_FPS = 60
DEFAULT_FOCAL_LENGTH_PX = 850.0
DEFAULT_ROOM_BOUNDS = ((-3.0, 3.0), (-3.0, 3.0), (0.0, 3.0))
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().with_name("mocap_calibration.json")


@dataclass(slots=True)
class DetectionSettings:
    threshold: int | None = None
    threshold_percentile: float = 99.75
    min_threshold: int = 170
    min_area: float = 8.0
    max_area: float = 3500.0
    min_circularity: float = 0.45
    min_fill_ratio: float = 0.35
    blur_kernel: int = 3
    morphology_kernel: int = 3
    max_markers_per_camera: int = 8


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


class ReflectiveMarkerDetector:
    def __init__(self, settings: DetectionSettings):
        self.settings = settings

    def detect(self, frame: np.ndarray, camera_id: int, timestamp: float) -> list[MarkerObservation]:
        gray = self._to_gray(frame)
        processed = self._preprocess(gray)
        threshold = self._choose_threshold(processed)
        _ok, mask = cv2.threshold(processed, threshold, 255, cv2.THRESH_BINARY)

        kernel_size = self.settings.morphology_kernel
        if kernel_size > 1:
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

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
        max_missing_frames: int = 15,
    ):
        self.max_match_distance_m = max_match_distance_m
        self.smoothing = float(np.clip(smoothing, 0.0, 1.0))
        self.max_missing_frames = max_missing_frames
        self.tracks: list[MarkerTrack] = []
        self._next_track_id = 1

    def update(self, measurements: list[MarkerMeasurement], timestamp: float) -> list[MarkerTrack]:
        possible_matches: list[tuple[float, int, int]] = []

        for track_index, track in enumerate(self.tracks):
            dt = max(timestamp - track.last_update, 1e-3)
            predicted = track.position + track.velocity * dt
            for measurement_index, measurement in enumerate(measurements):
                distance = float(np.linalg.norm(measurement.position - predicted))
                if distance <= self.max_match_distance_m:
                    possible_matches.append((distance, track_index, measurement_index))

        possible_matches.sort(key=lambda item: item[0])
        used_tracks: set[int] = set()
        used_measurements: set[int] = set()

        for _distance, track_index, measurement_index in possible_matches:
            if track_index in used_tracks or measurement_index in used_measurements:
                continue
            self._update_track(self.tracks[track_index], measurements[measurement_index], timestamp)
            used_tracks.add(track_index)
            used_measurements.add(measurement_index)

        for track_index, track in enumerate(self.tracks):
            if track_index in used_tracks:
                continue
            track.missing_frames += 1
            track.confidence *= 0.85
            track.observations = []

        for measurement_index, measurement in enumerate(measurements):
            if measurement_index in used_measurements:
                continue
            self._add_track(measurement, timestamp)

        self.tracks = [
            track for track in self.tracks if track.missing_frames <= self.max_missing_frames
        ]
        self.tracks.sort(key=lambda track: track.track_id)
        return self.tracks

    def _update_track(self, track: MarkerTrack, measurement: MarkerMeasurement, timestamp: float) -> None:
        dt = max(timestamp - track.last_update, 1e-3)
        old_position = track.position.copy()
        smoothed_position = (
            (1.0 - self.smoothing) * track.position + self.smoothing * measurement.position
        )
        track.position = smoothed_position
        track.velocity = (smoothed_position - old_position) / dt
        track.observations = measurement.observations
        track.reprojection_error_px = measurement.reprojection_error_px
        track.last_update = timestamp
        track.age_frames += 1
        track.missing_frames = 0
        track.confidence = min(1.0, track.confidence + 0.08)

    def _add_track(self, measurement: MarkerMeasurement, timestamp: float) -> None:
        track = MarkerTrack(
            track_id=self._next_track_id,
            position=measurement.position,
            velocity=np.zeros(3, dtype=np.float64),
            observations=measurement.observations,
            reprojection_error_px=measurement.reprojection_error_px,
            last_update=timestamp,
        )
        self._next_track_id += 1
        self.tracks.append(track)


class CameraSource:
    def __init__(
        self,
        camera_id: int,
        width: int,
        height: int,
        fps: int,
        exposure: float | None,
    ):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.exposure = exposure
        self.capture = None

    def open(self) -> bool:
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else 0
        capture = cv2.VideoCapture(self.camera_id, backend)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(self.camera_id)

        if not capture.isOpened():
            capture.release()
            return False

        self.capture = capture
        self._configure_capture()

        for _ in range(5):
            ok, frame = self.capture.read()
            if ok and frame is not None:
                return True
            time.sleep(0.03)

        self.close()
        return False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.capture is None:
            return False, None
        return self.capture.read()

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _configure_capture(self) -> None:
        if self.capture is None:
            return

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.capture.set(cv2.CAP_PROP_FOURCC, fourcc)

        if self.exposure is not None:
            # On many UVC cameras/OpenCV backends, 0.25 means manual exposure.
            self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.capture.set(cv2.CAP_PROP_EXPOSURE, float(self.exposure))


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
) -> list[CameraSource]:
    sources: list[CameraSource] = []
    for camera_id in camera_ids:
        source = CameraSource(camera_id, width, height, fps, exposure)
        if source.open():
            sources.append(source)
            print(f"[mocap] camera {camera_id} opened")
        else:
            print(f"[mocap] camera {camera_id} not available; continuing")
    return sources


def draw_preview(
    frame: np.ndarray,
    observations: list[MarkerObservation],
    tracks: list[MarkerTrack],
    camera_id: int,
) -> np.ndarray:
    preview = frame.copy()
    used_observation_ids: set[int] = set()

    for track in tracks:
        if track.missing_frames != 0:
            continue
        for observation in track.observations:
            if observation.camera_id != camera_id:
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

    for observation in observations:
        if id(observation) in used_observation_ids:
            continue
        center = tuple(int(round(value)) for value in observation.pixel)
        radius = max(3, int(round(observation.radius_px)))
        cv2.circle(preview, center, radius, (0, 200, 255), 1)

    cv2.putText(
        preview,
        f"camera {camera_id} | blobs {len(observations)}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return preview


def print_status(
    tracks: list[MarkerTrack],
    observations_by_camera: dict[int, list[MarkerObservation]],
    calibrated_camera_count: int,
) -> None:
    blob_counts = ", ".join(
        f"cam {camera_id}: {len(observations)}"
        for camera_id, observations in sorted(observations_by_camera.items())
    )
    live_tracks = [track for track in tracks if track.missing_frames == 0]
    print(f"[mocap] calibrated cams: {calibrated_camera_count} | blobs: {blob_counts}")

    if not live_tracks:
        print("[mocap] no 3D tracks yet")
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
        help="Comma-separated OpenCV camera indexes to try. Default: 1,2,3,4",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_FRAME_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--exposure",
        type=float,
        default=None,
        help="Optional manual camera exposure value. Backend-specific.",
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
        help="xmin,xmax,ymin,ymax,zmin,zmax in meters. Default: -3,3,-3,3,0,3",
    )
    parser.add_argument("--threshold", type=int, default=None)
    parser.add_argument("--min-area", type=float, default=8.0)
    parser.add_argument("--max-area", type=float, default=3500.0)
    parser.add_argument("--min-circularity", type=float, default=0.45)
    parser.add_argument("--max-markers-per-camera", type=int, default=8)
    parser.add_argument("--max-reprojection-error", type=float, default=14.0)
    parser.add_argument("--cluster-distance", type=float, default=0.20)
    parser.add_argument("--track-distance", type=float, default=0.35)
    parser.add_argument("--print-interval", type=float, default=0.25)
    parser.add_argument("--no-preview", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if cv2 is None:
        print("OpenCV is required for camera mocap. Install it with: python -m pip install opencv-python")
        return 1

    settings = DetectionSettings(
        threshold=args.threshold,
        min_area=args.min_area,
        max_area=args.max_area,
        min_circularity=args.min_circularity,
        max_markers_per_camera=args.max_markers_per_camera,
    )

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

    detector = ReflectiveMarkerDetector(settings)
    triangulator = MultiCameraTriangulator(
        calibrations=calibrations,
        max_pair_error_px=args.max_reprojection_error,
        cluster_distance_m=args.cluster_distance,
        room_bounds=args.room_bounds,
    )
    tracker = MarkerTracker(max_match_distance_m=args.track_distance)

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
                observations_by_camera[source.camera_id] = detector.detect(
                    frame,
                    source.camera_id,
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
                )

            if show_preview:
                for camera_id, frame in frames.items():
                    preview = draw_preview(
                        frame,
                        observations_by_camera.get(camera_id, []),
                        tracks,
                        camera_id,
                    )
                    cv2.imshow(f"mocap camera {camera_id}", preview)

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
