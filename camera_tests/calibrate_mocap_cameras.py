"""
Calibrate Jazz Hands mocap camera poses from the tetrahedral IR target.

Use this with the tetrahedral chemistry-kit target:
- center carbon is the world origin
- short bond + small atom
- short bond + large atom
- long bond + small atom
- long bond + large atom

The script opens any available cameras from --cameras, detects the four bright
reflective endpoint atoms, solves each camera pose relative to the target, and
writes a calibration JSON that mocap_tracker.py can read.

Controls:
- c: capture/accept the current valid pose for every visible camera
- s: save captured poses to the output JSON
- q or Esc: quit

Important: keep the target still while capturing. If each camera is calibrated
from a different target pose, the cameras will not share the same world origin.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path

import numpy as np

import mocap_tracker as mocap

cv2 = mocap.cv2

CM_TO_M = 0.01
# Change this to [1, 2, 3, 4] when calibrating the full camera rig.
CAMERA_IDS = [1, 2]
DEFAULT_CAMERA_IDS = CAMERA_IDS
FOUR_CAMERA_IDS = [1, 2, 3, 4]
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 800
DEFAULT_FPS = 120
DEFAULT_THRESHOLD = 230
DEFAULT_OUTPUT_PATH = str(Path(__file__).resolve().with_name("mocap_calibration.json"))
PREVIEW_SCALE = 0.75
PREVIEW_WINDOW_WIDTH = 1600
PREVIEW_WINDOW_HEIGHT = 600
POSE_SOLVE_INTERVAL_SECONDS = 0.20
TOP_MARKER_LABEL: str | None = None
TOP_MARKER_MARGIN_PX = 8.0
DEFAULT_ASSIGNMENT_STABILITY_PIXELS = 35.0
DEFAULT_STABLE_POSE_FRAMES = 4
DEFAULT_ASSIGNMENT_MEMORY_WEIGHT = 0.04
DEFAULT_ASSIGNMENT_SWITCH_MARGIN = 2.0
DEFAULT_AUTOFOCUS = 0
BLUR_KERNEL_BY_CAMERA = {
    1: 15,
    2: 15,
    3: 15,
    4: 15,
}
DEFAULT_AUTO_EXPOSURE_BY_CAMERA = {
    1: 0,
    2: 0,
    3: 0.25,
    4: 0,
}
DEFAULT_EXPOSURE_BY_CAMERA = {
    1: -8,
    2: -8,
    3: -8,
    4: -8,
}
DEFAULT_GAIN_BY_CAMERA = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
}

CARBON_RADIUS_CM = 2.2 / 2.0
SMALL_ATOM_RADIUS_CM = 1.6 / 2.0
LARGE_ATOM_RADIUS_CM = 2.2 / 2.0
TARGET_SURFACE_GAP_CM = {
    # Measured surface of center carbon to surface of outer atom.
    "short_small": 8.9,
    "short_large": 7.5,
    "long_small": 13.9,
    "long_large": 11.3,
}

TARGET_DIRECTIONS = {
    "short_small": np.array([1.0, 1.0, 1.0], dtype=np.float64),
    "short_large": np.array([1.0, -1.0, -1.0], dtype=np.float64),
    "long_small": np.array([-1.0, 1.0, -1.0], dtype=np.float64),
    "long_large": np.array([-1.0, -1.0, 1.0], dtype=np.float64),
}

TARGET_MARKER_RADII_M = {
    "short_small": SMALL_ATOM_RADIUS_CM * CM_TO_M,
    "short_large": LARGE_ATOM_RADIUS_CM * CM_TO_M,
    "long_small": SMALL_ATOM_RADIUS_CM * CM_TO_M,
    "long_large": LARGE_ATOM_RADIUS_CM * CM_TO_M,
}

TARGET_POINT_LABELS = ["short_small", "short_large", "long_small", "long_large"]


@dataclass(slots=True)
class PoseEstimate:
    camera_id: int
    rvec: np.ndarray
    tvec: np.ndarray
    rotation: np.ndarray
    position: np.ndarray
    assignment: dict[str, mocap.MarkerObservation]
    marker_count: int
    reprojection_error_px: float
    size_error: float
    score: float


@dataclass(slots=True)
class PoseStabilityState:
    estimate: PoseEstimate | None = None
    stable_frames: int = 0


def build_default_target_points() -> dict[str, np.ndarray]:
    center_to_center_cm = {
        "short_small": (
            CARBON_RADIUS_CM
            + TARGET_SURFACE_GAP_CM["short_small"]
            + SMALL_ATOM_RADIUS_CM
        ),
        "short_large": (
            CARBON_RADIUS_CM
            + TARGET_SURFACE_GAP_CM["short_large"]
            + LARGE_ATOM_RADIUS_CM
        ),
        "long_small": (
            CARBON_RADIUS_CM
            + TARGET_SURFACE_GAP_CM["long_small"]
            + SMALL_ATOM_RADIUS_CM
        ),
        "long_large": (
            CARBON_RADIUS_CM
            + TARGET_SURFACE_GAP_CM["long_large"]
            + LARGE_ATOM_RADIUS_CM
        ),
    }

    points: dict[str, np.ndarray] = {}
    for label, direction in TARGET_DIRECTIONS.items():
        unit_direction = direction / float(np.linalg.norm(direction))
        points[label] = unit_direction * center_to_center_cm[label] * CM_TO_M
    return points


def load_target_points(path: str | None) -> dict[str, np.ndarray]:
    if path is None:
        return build_default_target_points()

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    raw_points = data.get("target_points_m", data.get("points_m", data))
    points = {
        str(label): np.asarray(raw_points[label], dtype=np.float64)
        for label in TARGET_POINT_LABELS
    }
    return points


def solve_camera_pose(
    camera_id: int,
    observations: list[mocap.MarkerObservation],
    target_points: dict[str, np.ndarray],
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray,
    max_reprojection_error_px: float,
    size_weight: float,
    min_pose_markers: int,
    max_pose_candidates: int,
    top_marker_label: str | None,
    top_marker_margin_px: float,
    previous_estimate: PoseEstimate | None,
    assignment_memory_weight: float,
    assignment_switch_margin: float,
    assignment_stability_pixels: float,
) -> PoseEstimate | None:
    min_pose_markers = int(np.clip(min_pose_markers, 3, 4))
    if len(observations) < min_pose_markers:
        return None

    object_points_by_label = {
        label: target_points[label].astype(np.float64)
        for label in TARGET_POINT_LABELS
    }
    best: PoseEstimate | None = None
    best_previous_like: PoseEstimate | None = None

    max_pose_candidates = max(min_pose_markers, max_pose_candidates)
    candidate_observations = observations[: min(len(observations), max_pose_candidates)]
    marker_counts = [4, 3] if min_pose_markers <= 3 else [4]
    for marker_count in marker_counts:
        if len(candidate_observations) < marker_count:
            continue
        for target_labels in combinations(TARGET_POINT_LABELS, marker_count):
            for observation_group in combinations(candidate_observations, marker_count):
                for ordered_observations in permutations(observation_group, marker_count):
                    assignment = {
                        label: observation
                        for label, observation in zip(target_labels, ordered_observations)
                    }
                    if not assignment_matches_top_marker(
                        assignment,
                        top_marker_label,
                        top_marker_margin_px,
                    ):
                        continue
                    object_points = np.asarray(
                        [object_points_by_label[label] for label in target_labels],
                        dtype=np.float64,
                    )
                    image_points = np.asarray(
                        [assignment[label].pixel for label in target_labels],
                        dtype=np.float64,
                    )

                    estimate = solve_pnp_candidate(
                        camera_id,
                        list(target_labels),
                        object_points,
                        image_points,
                        assignment,
                        intrinsic,
                        dist_coeffs,
                        size_weight,
                    )
                    if estimate is None:
                        continue
                    if estimate.reprojection_error_px > max_reprojection_error_px:
                        continue

                    estimate.score += 2.0 * (4 - marker_count)
                    if previous_estimate is not None:
                        continuity_error = assignment_distance_px(
                            estimate.assignment,
                            previous_estimate.assignment,
                        )
                        if math.isfinite(continuity_error):
                            estimate.score += assignment_memory_weight * continuity_error
                            if continuity_error <= assignment_stability_pixels:
                                if (
                                    best_previous_like is None
                                    or estimate.score < best_previous_like.score
                                ):
                                    best_previous_like = estimate

                    if best is None or estimate.score < best.score:
                        best = estimate

    if best is None:
        return None

    if (
        best_previous_like is not None
        and best is not best_previous_like
        and best.score > best_previous_like.score - assignment_switch_margin
    ):
        return best_previous_like

    return best


def assignment_distance_px(
    assignment: dict[str, mocap.MarkerObservation],
    reference: dict[str, mocap.MarkerObservation],
) -> float:
    distances = [
        float(np.linalg.norm(assignment[label].pixel - reference[label].pixel))
        for label in TARGET_POINT_LABELS
        if label in assignment and label in reference
    ]
    if not distances:
        return float("inf")
    return float(np.mean(distances))


def update_pose_stability(
    state: PoseStabilityState,
    estimate: PoseEstimate | None,
    assignment_stability_pixels: float,
    stable_pose_frames: int,
) -> PoseEstimate | None:
    required_frames = max(1, stable_pose_frames)
    if estimate is None:
        state.estimate = None
        state.stable_frames = 0
        return None

    if state.estimate is not None:
        distance = assignment_distance_px(estimate.assignment, state.estimate.assignment)
        if distance <= assignment_stability_pixels:
            state.stable_frames += 1
        else:
            state.stable_frames = 1
    else:
        state.stable_frames = 1

    state.estimate = estimate
    if state.stable_frames >= required_frames:
        return estimate
    return None


def assignment_matches_top_marker(
    assignment: dict[str, mocap.MarkerObservation],
    top_marker_label: str | None,
    top_marker_margin_px: float,
) -> bool:
    if top_marker_label is None:
        return True
    top_observation = assignment.get(top_marker_label)
    if top_observation is None:
        return False

    min_y = min(float(observation.pixel[1]) for observation in assignment.values())
    return float(top_observation.pixel[1]) <= min_y + max(0.0, top_marker_margin_px)


def solve_pnp_candidate(
    camera_id: int,
    labels: list[str],
    object_points: np.ndarray,
    image_points: np.ndarray,
    assignment: dict[str, mocap.MarkerObservation],
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray,
    size_weight: float,
) -> PoseEstimate | None:
    pnp_flag = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_EPNP)
    try:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            intrinsic,
            dist_coeffs,
            flags=pnp_flag,
        )
    except cv2.error:
        return None
    if not ok:
        return None

    if len(object_points) >= 4:
        try:
            ok_refined, rvec_refined, tvec_refined = cv2.solvePnP(
                object_points,
                image_points,
                intrinsic,
                dist_coeffs,
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok_refined:
                rvec, tvec = rvec_refined, tvec_refined
        except cv2.error:
            pass

    rotation, _jacobian = cv2.Rodrigues(rvec)
    camera_points = (rotation @ object_points.T).T + tvec.reshape(1, 3)
    if np.any(camera_points[:, 2] <= 0.0):
        return None

    projected, _jacobian = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        intrinsic,
        dist_coeffs,
    )
    projected_points = projected.reshape(-1, 2)
    reprojection_error = float(
        np.mean(np.linalg.norm(projected_points - image_points, axis=1))
    )

    size_error = estimate_marker_size_error(labels, assignment, camera_points)
    position = (-rotation.T @ tvec.reshape(3)).astype(np.float64)
    score = reprojection_error + size_weight * size_error

    return PoseEstimate(
        camera_id=camera_id,
        rvec=rvec.reshape(3).astype(np.float64),
        tvec=tvec.reshape(3).astype(np.float64),
        rotation=rotation.astype(np.float64),
        position=position,
        assignment=assignment,
        marker_count=len(labels),
        reprojection_error_px=reprojection_error,
        size_error=size_error,
        score=float(score),
    )


def estimate_marker_size_error(
    labels: list[str],
    assignment: dict[str, mocap.MarkerObservation],
    camera_points: np.ndarray,
) -> float:
    observed = np.asarray(
        [assignment[label].radius_px for label in labels],
        dtype=np.float64,
    )
    expected = np.asarray(
        [
            TARGET_MARKER_RADII_M[label] / max(float(camera_points[index, 2]), 1e-6)
            for index, label in enumerate(labels)
        ],
        dtype=np.float64,
    )

    denominator = float(np.dot(expected, expected))
    if denominator <= 1e-12:
        return 0.0

    scale = float(np.dot(observed, expected) / denominator)
    predicted = expected * scale
    relative_error = (observed - predicted) / np.maximum(observed, 1.0)
    return float(np.sqrt(np.mean(relative_error * relative_error)))


def draw_estimate_overlay(
    frame: np.ndarray,
    observations: list[mocap.MarkerObservation],
    estimate: PoseEstimate | None,
    target_points: dict[str, np.ndarray],
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray,
    detector_settings: mocap.DetectionSettings,
    captured: bool,
) -> np.ndarray:
    preview = frame.copy()

    for observation in observations:
        center = tuple(int(round(value)) for value in observation.pixel)
        radius = max(3, int(round(observation.radius_px)))
        cv2.circle(preview, center, radius, (0, 200, 255), 1)

    if estimate is not None:
        estimate_labels = [
            label for label in TARGET_POINT_LABELS if label in estimate.assignment
        ]
        object_points = np.asarray(
            [target_points[label] for label in estimate_labels],
            dtype=np.float64,
        )
        projected, _jacobian = cv2.projectPoints(
            object_points,
            estimate.rvec.reshape(3, 1),
            estimate.tvec.reshape(3, 1),
            intrinsic,
            dist_coeffs,
        )
        projected_points = projected.reshape(-1, 2)

        for index, label in enumerate(estimate_labels):
            observation = estimate.assignment[label]
            observed_center = tuple(int(round(value)) for value in observation.pixel)
            projected_center = tuple(int(round(value)) for value in projected_points[index])
            cv2.circle(preview, observed_center, 8, (0, 255, 0), 2)
            cv2.circle(preview, projected_center, 3, (255, 0, 255), -1)
            cv2.line(preview, observed_center, projected_center, (255, 0, 255), 1)
            cv2.putText(
                preview,
                label,
                (observed_center[0] + 8, observed_center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.drawFrameAxes(
            preview,
            intrinsic,
            dist_coeffs,
            estimate.rvec.reshape(3, 1),
            estimate.tvec.reshape(3, 1),
            0.05,
        )

    status_color = (0, 255, 0) if estimate is not None else (0, 0, 255)
    status = (
        f"valid pose ({estimate.marker_count} markers)"
        if estimate is not None
        else "need 3-4 clean blobs"
    )
    if captured:
        status += " | captured"
    cv2.putText(
        preview,
        status,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2,
        cv2.LINE_AA,
    )
    if estimate is not None:
        cv2.putText(
            preview,
            f"err {estimate.reprojection_error_px:.2f}px size {estimate.size_error:.3f}",
            (12, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    threshold_preview = build_threshold_preview(frame, observations, detector_settings)
    return cv2.hconcat([resize_preview(preview), resize_preview(threshold_preview)])


def resize_preview(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(
        frame,
        None,
        fx=PREVIEW_SCALE,
        fy=PREVIEW_SCALE,
        interpolation=cv2.INTER_AREA,
    )


def build_threshold_preview(
    frame: np.ndarray,
    observations: list[mocap.MarkerObservation],
    settings: mocap.DetectionSettings,
) -> np.ndarray:
    mask = threshold_mask(frame, settings)
    preview = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for index, observation in enumerate(observations, start=1):
        center = tuple(int(round(value)) for value in observation.pixel)
        radius = max(3, int(round(observation.radius_px)))
        cv2.circle(preview, center, radius, (0, 255, 0), 2)
        cv2.putText(
            preview,
            f"{index}",
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        preview,
        f"threshold {settings.threshold}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return preview


def threshold_mask(frame: np.ndarray, settings: mocap.DetectionSettings) -> np.ndarray:
    gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel_size = settings.blur_kernel
    if kernel_size > 1:
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    threshold = settings.threshold
    if threshold is None:
        threshold = max(settings.min_threshold, int(np.percentile(gray, settings.threshold_percentile)))
    _ok, mask = cv2.threshold(gray, int(np.clip(threshold, 0, 255)), 255, cv2.THRESH_BINARY)

    kernel_size = settings.morphology_kernel
    if kernel_size > 1:
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def calibration_window_name(camera_id: int) -> str:
    return f"calibration camera {camera_id}"


def create_preview_windows(sources: list[mocap.CameraSource]) -> None:
    for source in sources:
        window_name = calibration_window_name(source.camera_id)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, PREVIEW_WINDOW_WIDTH, PREVIEW_WINDOW_HEIGHT)


def write_calibration_json(
    output_path: str,
    captured_estimates: dict[int, PoseEstimate],
    target_points: dict[str, np.ndarray],
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray,
    frame_width: int,
    frame_height: int,
    focal_length_px: float,
) -> None:
    output = {
        "created_at_unix": time.time(),
        "source": "Camera_Tests/calibrate_mocap_cameras.py",
        "frame_width": frame_width,
        "frame_height": frame_height,
        "focal_length_px": focal_length_px,
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        "target": {
            "name": "tetrahedral_ochem_reflective_target",
            "origin": "center carbon",
            "surface_gap_cm": TARGET_SURFACE_GAP_CM,
            "points_m": {
                label: target_points[label].reshape(3).tolist()
                for label in TARGET_POINT_LABELS
            },
            "marker_radii_m": TARGET_MARKER_RADII_M,
        },
        "cameras": [],
    }

    for camera_id in sorted(captured_estimates):
        estimate = captured_estimates[camera_id]
        used_labels = [
            label for label in TARGET_POINT_LABELS if label in estimate.assignment
        ]
        missing_labels = [
            label for label in TARGET_POINT_LABELS if label not in estimate.assignment
        ]
        output["cameras"].append(
            {
                "id": camera_id,
                "name": f"camera_{camera_id}",
                "intrinsic": intrinsic.tolist(),
                "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
                "rotation": estimate.rotation.tolist(),
                "translation": estimate.tvec.reshape(3).tolist(),
                "rvec": estimate.rvec.reshape(3).tolist(),
                "tvec": estimate.tvec.reshape(3).tolist(),
                "position": estimate.position.reshape(3).tolist(),
                "euler_xyz_deg": rotation_matrix_to_euler_xyz_deg(estimate.rotation),
                "marker_count": estimate.marker_count,
                "used_target_labels": used_labels,
                "missing_target_labels": missing_labels,
                "reprojection_error_px": estimate.reprojection_error_px,
                "size_error": estimate.size_error,
                "assigned_pixels": {
                    label: estimate.assignment[label].pixel.reshape(2).tolist()
                    for label in used_labels
                },
            }
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
        file.write("\n")


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


def parse_camera_ids(text: str) -> list[int]:
    values = [value.strip() for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one camera id.")
    return [int(value) for value in values]


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
    return camera_setting(camera_id, DEFAULT_EXPOSURE_BY_CAMERA, override, -8.0)


def camera_gain(camera_id: int, override: float | None) -> float:
    return camera_setting(camera_id, DEFAULT_GAIN_BY_CAMERA, override, 0.0)


def open_calibration_cameras(
    camera_ids: list[int],
    width: int,
    height: int,
    fps: int,
    auto_exposure: float | None,
    exposure: float | None,
    gain: float | None,
) -> list[mocap.CameraSource]:
    sources: list[mocap.CameraSource] = []
    for camera_id in camera_ids:
        source = mocap.CameraSource(
            camera_id,
            width,
            height,
            fps,
            None,
            None,
            None,
            configure_capture=False,
        )
        if source.open():
            # Camera parameters are managed by external camera software.
            # Do not call apply_camera_settings() here.
            sources.append(source)
            print_camera_settings(source)
        else:
            print(f"[calibration] camera {camera_id} not available; continuing")
    return sources


def apply_camera_settings(
    source: mocap.CameraSource,
    auto_exposure: float | None,
    exposure: float | None,
    gain: float | None,
) -> None:
    _ = source, auto_exposure, exposure, gain
    # Camera parameters are managed by external camera software.
    # source.capture.set(cv2.CAP_PROP_FRAME_WIDTH, source.width)
    # source.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, source.height)
    # source.capture.set(cv2.CAP_PROP_FPS, source.fps)
    # source.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, source.auto_exposure)
    # source.capture.set(cv2.CAP_PROP_AUTOFOCUS, DEFAULT_AUTOFOCUS)
    # source.capture.set(cv2.CAP_PROP_EXPOSURE, source.exposure)
    # source.capture.set(cv2.CAP_PROP_GAIN, source.gain)
    return


def print_camera_settings(source: mocap.CameraSource) -> None:
    if source.capture is None:
        return

    print(
        f"[calibration] opened camera {source.camera_id} | "
        f"auto_exposure={source.capture.get(cv2.CAP_PROP_AUTO_EXPOSURE):.2f} "
        f"autofocus={source.capture.get(cv2.CAP_PROP_AUTOFOCUS):.2f} "
        f"exposure={source.capture.get(cv2.CAP_PROP_EXPOSURE):.2f} "
        f"gain={source.capture.get(cv2.CAP_PROP_GAIN):.2f}"
    )


def build_detection_settings(args: argparse.Namespace, camera_id: int) -> mocap.DetectionSettings:
    return mocap.DetectionSettings(
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
        max_markers_per_camera=max(args.max_pose_candidates, 4),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create mocap camera calibration JSON from the tetrahedral IR target.",
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
    parser.add_argument("--focal-length-px", type=float, default=mocap.DEFAULT_FOCAL_LENGTH_PX)
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
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--min-area", type=float, default=8.0)
    parser.add_argument("--max-area", type=float, default=3500.0)
    parser.add_argument("--min-radius", type=float, default=2.0)
    parser.add_argument("--max-radius", type=float, default=60.0)
    parser.add_argument("--min-circularity", type=float, default=0.45)
    parser.add_argument("--min-fill-ratio", type=float, default=0.35)
    parser.add_argument("--max-aspect-ratio", type=float, default=2.0)
    parser.add_argument("--min-brightness", type=float, default=0.0)
    parser.add_argument("--max-reprojection-error", type=float, default=8.0)
    parser.add_argument(
        "--min-pose-markers",
        type=int,
        default=4,
        help="Use 4 for the full tetrahedron. Use 3 only if one marker is hidden.",
    )
    parser.add_argument(
        "--max-pose-candidates",
        type=int,
        default=5,
        help="Only try this many brightest blobs for pose matching. Lower is faster.",
    )
    parser.add_argument(
        "--pose-solve-interval",
        type=float,
        default=POSE_SOLVE_INTERVAL_SECONDS,
        help="Seconds between expensive pose solves per camera. Preview still updates every frame.",
    )
    parser.add_argument(
        "--size-weight",
        type=float,
        default=12.0,
        help="How strongly marker size helps disambiguate labels. Use 0 if tape sizes are unreliable.",
    )
    parser.add_argument(
        "--top-marker-label",
        choices=TARGET_POINT_LABELS,
        default=TOP_MARKER_LABEL,
        help=(
            "Optional hard constraint for known target orientation. If set, this "
            "target label must be the topmost detected marker in the image."
        ),
    )
    parser.add_argument(
        "--top-marker-margin-px",
        type=float,
        default=TOP_MARKER_MARGIN_PX,
        help="Pixel tolerance for the top-marker constraint.",
    )
    parser.add_argument(
        "--assignment-stability-pixels",
        type=float,
        default=DEFAULT_ASSIGNMENT_STABILITY_PIXELS,
        help="How far a label can move in pixels and still count as the same stable assignment.",
    )
    parser.add_argument(
        "--stable-pose-frames",
        type=int,
        default=DEFAULT_STABLE_POSE_FRAMES,
        help="Number of consecutive solve cycles required before a pose can be captured.",
    )
    parser.add_argument(
        "--assignment-memory-weight",
        type=float,
        default=DEFAULT_ASSIGNMENT_MEMORY_WEIGHT,
        help="Penalty weight for label assignments that jump away from the previous stable candidate.",
    )
    parser.add_argument(
        "--assignment-switch-margin",
        type=float,
        default=DEFAULT_ASSIGNMENT_SWITCH_MARGIN,
        help="New label assignment must beat the previous-like assignment by this score margin.",
    )
    parser.add_argument(
        "--target-config",
        default=None,
        help="Optional JSON with target_points_m/points_m for measured target coordinates.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument(
        "--auto-save",
        action="store_true",
        help="Save and exit once every connected camera has a valid pose.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if cv2 is None:
        print("OpenCV is required. Install it with: python -m pip install opencv-python")
        return 1

    target_points = load_target_points(args.target_config)
    intrinsic = mocap.default_intrinsic(args.width, args.height, args.focal_length_px)
    dist_coeffs = np.zeros(5, dtype=np.float64)

    print_target_geometry(target_points)
    if args.top_marker_label is not None:
        print(
            "[calibration] top-marker constraint: "
            f"{args.top_marker_label} must have the smallest image y value "
            f"(within {args.top_marker_margin_px:.1f}px)"
        )
    sources = open_calibration_cameras(
        args.cameras,
        args.width,
        args.height,
        args.fps,
        args.auto_exposure,
        args.exposure,
        args.gain,
    )

    if not sources:
        print("[calibration] no cameras opened")
        return 1

    settings_by_camera = {
        source.camera_id: build_detection_settings(args, source.camera_id)
        for source in sources
    }
    detectors = {
        camera_id: mocap.ReflectiveMarkerDetector(settings)
        for camera_id, settings in settings_by_camera.items()
    }

    print("[calibration] hold the tetrahedral target still at the intended origin")
    print("[calibration] press c to capture valid camera poses, s to save, q/Esc to quit")

    show_preview = not args.no_preview
    if show_preview:
        create_preview_windows(sources)

    captured_estimates: dict[int, PoseEstimate] = {}
    latest_estimates: dict[int, PoseEstimate] = {}
    pose_stability_by_camera: dict[int, PoseStabilityState] = {
        source.camera_id: PoseStabilityState() for source in sources
    }
    next_pose_solve_time_by_camera: dict[int, float] = {
        source.camera_id: 0.0 for source in sources
    }
    last_print_time = 0.0

    try:
        while True:
            timestamp = time.time()
            latest_observations: dict[int, list[mocap.MarkerObservation]] = {}
            latest_frames: dict[int, np.ndarray] = {}

            for source in sources:
                ok, frame = source.read()
                if not ok or frame is None:
                    latest_observations[source.camera_id] = []
                    continue

                latest_frames[source.camera_id] = frame
                observations = detectors[source.camera_id].detect(
                    frame,
                    source.camera_id,
                    timestamp,
                )
                latest_observations[source.camera_id] = observations
                if timestamp >= next_pose_solve_time_by_camera[source.camera_id]:
                    next_pose_solve_time_by_camera[source.camera_id] = (
                        timestamp + max(args.pose_solve_interval, 0.05)
                    )
                    stability_state = pose_stability_by_camera[source.camera_id]
                    estimate = solve_camera_pose(
                        source.camera_id,
                        observations,
                        target_points,
                        intrinsic,
                        dist_coeffs,
                        args.max_reprojection_error,
                        args.size_weight,
                        args.min_pose_markers,
                        args.max_pose_candidates,
                        args.top_marker_label,
                        args.top_marker_margin_px,
                        stability_state.estimate,
                        args.assignment_memory_weight,
                        args.assignment_switch_margin,
                        args.assignment_stability_pixels,
                    )
                    stable_estimate = update_pose_stability(
                        stability_state,
                        estimate,
                        args.assignment_stability_pixels,
                        args.stable_pose_frames,
                    )
                    if stable_estimate is not None:
                        latest_estimates[source.camera_id] = stable_estimate
                    else:
                        latest_estimates.pop(source.camera_id, None)

            if timestamp - last_print_time > 0.5:
                last_print_time = timestamp
                print_pose_status(
                    sources,
                    latest_observations,
                    latest_estimates,
                    captured_estimates,
                    pose_stability_by_camera,
                    args.stable_pose_frames,
                )

            if args.auto_save and len(latest_estimates) == len(sources):
                captured_estimates.update(latest_estimates)
                write_calibration_json(
                    args.output,
                    captured_estimates,
                    target_points,
                    intrinsic,
                    dist_coeffs,
                    args.width,
                    args.height,
                    args.focal_length_px,
                )
                print(f"[calibration] saved {len(captured_estimates)} cameras to {args.output}")
                break

            if show_preview:
                for camera_id, frame in latest_frames.items():
                    preview = draw_estimate_overlay(
                        frame,
                        latest_observations.get(camera_id, []),
                        latest_estimates.get(camera_id),
                        target_points,
                        intrinsic,
                        dist_coeffs,
                        settings_by_camera[camera_id],
                        camera_id in captured_estimates,
                    )
                    cv2.imshow(calibration_window_name(camera_id), preview)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    if latest_estimates:
                        captured_estimates.update(latest_estimates)
                        print(f"[calibration] captured cameras: {sorted(captured_estimates)}")
                    else:
                        print("[calibration] no valid poses to capture yet")
                if key == ord("s"):
                    if not captured_estimates:
                        print("[calibration] nothing captured yet; press c first")
                    else:
                        write_calibration_json(
                            args.output,
                            captured_estimates,
                            target_points,
                            intrinsic,
                            dist_coeffs,
                            args.width,
                            args.height,
                            args.focal_length_px,
                        )
                        print(f"[calibration] saved {len(captured_estimates)} cameras to {args.output}")
            else:
                if latest_estimates:
                    captured_estimates.update(latest_estimates)
                    write_calibration_json(
                        args.output,
                        captured_estimates,
                        target_points,
                        intrinsic,
                        dist_coeffs,
                        args.width,
                        args.height,
                        args.focal_length_px,
                    )
                    print(f"[calibration] saved {len(captured_estimates)} cameras to {args.output}")
                    break

    except KeyboardInterrupt:
        print("\n[calibration] stopped")
    finally:
        for source in sources:
            source.close()
        if cv2 is not None:
            cv2.destroyAllWindows()

    return 0


def print_target_geometry(target_points: dict[str, np.ndarray]) -> None:
    print("[calibration] target points, meters from center carbon:")
    for label in TARGET_POINT_LABELS:
        point = target_points[label]
        center_to_center_cm = float(np.linalg.norm(point)) / CM_TO_M
        surface_gap_cm = TARGET_SURFACE_GAP_CM.get(label)
        if surface_gap_cm is None:
            print(f"  {label:12s}: [{point[0]:+.5f}, {point[1]:+.5f}, {point[2]:+.5f}]")
        else:
            print(
                f"  {label:12s}: "
                f"surface_gap={surface_gap_cm:5.1f}cm "
                f"center_to_center={center_to_center_cm:5.1f}cm "
                f"point=[{point[0]:+.5f}, {point[1]:+.5f}, {point[2]:+.5f}]"
            )


def print_pose_status(
    sources: list[mocap.CameraSource],
    observations_by_camera: dict[int, list[mocap.MarkerObservation]],
    estimates: dict[int, PoseEstimate],
    captured_estimates: dict[int, PoseEstimate],
    pose_stability_by_camera: dict[int, PoseStabilityState],
    stable_pose_frames: int,
) -> None:
    parts: list[str] = []
    for source in sources:
        camera_id = source.camera_id
        blob_count = len(observations_by_camera.get(camera_id, []))
        estimate = estimates.get(camera_id)
        stability = pose_stability_by_camera.get(camera_id)
        stable_count = stability.stable_frames if stability is not None else 0
        captured = " captured" if camera_id in captured_estimates else ""
        if estimate is None:
            if stable_count > 0:
                parts.append(
                    f"cam {camera_id}: {blob_count} blobs, stabilizing "
                    f"{stable_count}/{max(1, stable_pose_frames)}{captured}"
                )
            else:
                parts.append(f"cam {camera_id}: {blob_count} blobs, no pose{captured}")
        else:
            x, y, z = estimate.position
            parts.append(
                f"cam {camera_id}: {blob_count} blobs, {estimate.marker_count} marker pose, "
                f"stable={stable_count}/{max(1, stable_pose_frames)}, "
                f"err={estimate.reprojection_error_px:.2f}px, "
                f"pos=({x:+.2f},{y:+.2f},{z:+.2f})m{captured}"
            )
    print("[calibration] " + " | ".join(parts))


if __name__ == "__main__":
    raise SystemExit(main())
