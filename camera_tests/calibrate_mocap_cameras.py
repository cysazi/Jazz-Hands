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
# Default to the three physical mocap cameras currently used for testing.
DEFAULT_CAMERA_IDS = [1, 2, 3]
FOUR_CAMERA_IDS = [1, 2, 3, 4]
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 800
DEFAULT_FPS = 120
DEFAULT_OUTPUT_PATH = str(Path(__file__).resolve().with_name("mocap_calibration.json"))
PREVIEW_SCALE = 0.5
CONTROLS_WINDOW = "Calibration Controls"
POSE_SOLVE_INTERVAL_SECONDS = 0.20
DEFAULT_AUTO_EXPOSURE_BY_CAMERA = {
    1: 0.25,
    2: 0.0,
    3: 0.0,
}

CARBON_RADIUS_CM = 2.2 / 2.0
SMALL_ATOM_RADIUS_CM = 1.6 / 2.0
LARGE_ATOM_RADIUS_CM = 2.2 / 2.0
SHORT_BOND_CM = 1.8
LONG_BOND_CM = 3.2

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


def build_default_target_points() -> dict[str, np.ndarray]:
    lengths_cm = {
        "short_small": CARBON_RADIUS_CM + SHORT_BOND_CM + SMALL_ATOM_RADIUS_CM,
        "short_large": CARBON_RADIUS_CM + SHORT_BOND_CM + LARGE_ATOM_RADIUS_CM,
        "long_small": CARBON_RADIUS_CM + LONG_BOND_CM + SMALL_ATOM_RADIUS_CM,
        "long_large": CARBON_RADIUS_CM + LONG_BOND_CM + LARGE_ATOM_RADIUS_CM,
    }

    points: dict[str, np.ndarray] = {}
    for label, direction in TARGET_DIRECTIONS.items():
        unit_direction = direction / float(np.linalg.norm(direction))
        points[label] = unit_direction * lengths_cm[label] * CM_TO_M
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
) -> PoseEstimate | None:
    min_pose_markers = int(np.clip(min_pose_markers, 3, 4))
    if len(observations) < min_pose_markers:
        return None

    object_points_by_label = {
        label: target_points[label].astype(np.float64)
        for label in TARGET_POINT_LABELS
    }
    best: PoseEstimate | None = None

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
                    if best is None or estimate.score < best.score:
                        best = estimate

    return best


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


def apply_extra_camera_settings(
    sources: list[mocap.CameraSource],
    auto_exposure: float | None,
    exposure: float | None,
    gain: float | None,
) -> None:
    for source in sources:
        if source.capture is None:
            continue
        manual_auto_exposure = (
            float(auto_exposure)
            if auto_exposure is not None
            else DEFAULT_AUTO_EXPOSURE_BY_CAMERA.get(source.camera_id, 0.0)
        )
        source.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, manual_auto_exposure)
        source.capture.set(cv2.CAP_PROP_AUTOFOCUS, mocap.DEFAULT_AUTOFOCUS)
        if exposure is not None:
            source.capture.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
        if gain is not None:
            source.capture.set(cv2.CAP_PROP_GAIN, float(gain))


def setup_detection_controls(args: argparse.Namespace) -> None:
    cv2.namedWindow(CONTROLS_WINDOW)
    cv2.createTrackbar("threshold", CONTROLS_WINDOW, int(args.threshold), 255, lambda _value: None)
    cv2.createTrackbar("min area", CONTROLS_WINDOW, int(args.min_area), 1000, lambda _value: None)
    cv2.createTrackbar("max area", CONTROLS_WINDOW, int(args.max_area), 20000, lambda _value: None)
    cv2.createTrackbar("min radius", CONTROLS_WINDOW, int(args.min_radius), 80, lambda _value: None)
    cv2.createTrackbar("max radius", CONTROLS_WINDOW, int(args.max_radius), 180, lambda _value: None)
    cv2.createTrackbar(
        "min circularity %",
        CONTROLS_WINDOW,
        int(args.min_circularity * 100),
        100,
        lambda _value: None,
    )
    cv2.createTrackbar(
        "min fill %",
        CONTROLS_WINDOW,
        int(args.min_fill_ratio * 100),
        100,
        lambda _value: None,
    )
    cv2.createTrackbar(
        "max aspect x10",
        CONTROLS_WINDOW,
        int(args.max_aspect_ratio * 10),
        60,
        lambda _value: None,
    )


def update_settings_from_controls(settings: mocap.DetectionSettings) -> None:
    settings.threshold = cv2.getTrackbarPos("threshold", CONTROLS_WINDOW)
    settings.min_area = max(1.0, float(cv2.getTrackbarPos("min area", CONTROLS_WINDOW)))
    settings.max_area = max(
        settings.min_area + 1.0,
        float(cv2.getTrackbarPos("max area", CONTROLS_WINDOW)),
    )
    settings.min_radius_px = max(0.0, float(cv2.getTrackbarPos("min radius", CONTROLS_WINDOW)))
    settings.max_radius_px = max(
        settings.min_radius_px + 1.0,
        float(cv2.getTrackbarPos("max radius", CONTROLS_WINDOW)),
    )
    settings.min_circularity = cv2.getTrackbarPos("min circularity %", CONTROLS_WINDOW) / 100.0
    settings.min_fill_ratio = cv2.getTrackbarPos("min fill %", CONTROLS_WINDOW) / 100.0
    settings.max_aspect_ratio = max(
        1.0,
        cv2.getTrackbarPos("max aspect x10", CONTROLS_WINDOW) / 10.0,
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
            "Comma-separated OpenCV camera indexes to try. Default: 1,2,3. "
            "Use 1,2,3,4 for the full rig."
        ),
    )
    parser.add_argument("--width", type=int, default=DEFAULT_FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_FRAME_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--focal-length-px", type=float, default=mocap.DEFAULT_FOCAL_LENGTH_PX)
    parser.add_argument("--exposure", type=float, default=-13.0)
    parser.add_argument(
        "--auto-exposure",
        type=float,
        default=None,
        help=(
            "Override auto-exposure value for every camera. Default uses per-camera "
            "manual values: camera 1 -> 0.25, camera 2 -> 0.0, camera 3 -> 0.0."
        ),
    )
    parser.add_argument("--gain", type=float, default=0.0)
    parser.add_argument("--threshold", type=int, default=200)
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
        default=3,
        help="Use 3 for fallback calibration when one tetrahedron marker is hidden; 4 is stricter.",
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

    settings = mocap.DetectionSettings(
        threshold=args.threshold,
        min_area=args.min_area,
        max_area=args.max_area,
        min_radius_px=args.min_radius,
        max_radius_px=args.max_radius,
        min_circularity=args.min_circularity,
        min_fill_ratio=args.min_fill_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
        min_brightness=args.min_brightness,
        max_markers_per_camera=max(args.max_pose_candidates, 4),
    )
    detector = mocap.ReflectiveMarkerDetector(settings)

    print_target_geometry(target_points)
    sources = mocap.open_available_cameras(
        args.cameras,
        args.width,
        args.height,
        args.fps,
        args.exposure,
        args.auto_exposure,
        args.gain,
    )
    apply_extra_camera_settings(sources, args.auto_exposure, args.exposure, args.gain)

    if not sources:
        print("[calibration] no cameras opened")
        return 1

    print("[calibration] hold the tetrahedral target still at the intended origin")
    print("[calibration] press c to capture valid camera poses, s to save, q/Esc to quit")

    show_preview = not args.no_preview
    if show_preview:
        setup_detection_controls(args)

    captured_estimates: dict[int, PoseEstimate] = {}
    latest_estimates: dict[int, PoseEstimate] = {}
    next_pose_solve_time_by_camera: dict[int, float] = {
        source.camera_id: 0.0 for source in sources
    }
    last_print_time = 0.0

    try:
        while True:
            timestamp = time.time()
            latest_observations: dict[int, list[mocap.MarkerObservation]] = {}
            latest_frames: dict[int, np.ndarray] = {}
            if show_preview:
                update_settings_from_controls(settings)

            for source in sources:
                ok, frame = source.read()
                if not ok or frame is None:
                    latest_observations[source.camera_id] = []
                    continue

                latest_frames[source.camera_id] = frame
                observations = detector.detect(frame, source.camera_id, timestamp)
                latest_observations[source.camera_id] = observations
                if timestamp >= next_pose_solve_time_by_camera[source.camera_id]:
                    next_pose_solve_time_by_camera[source.camera_id] = (
                        timestamp + max(args.pose_solve_interval, 0.05)
                    )
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
                    )
                    if estimate is not None:
                        latest_estimates[source.camera_id] = estimate
                    else:
                        latest_estimates.pop(source.camera_id, None)

            if timestamp - last_print_time > 0.5:
                last_print_time = timestamp
                print_pose_status(sources, latest_observations, latest_estimates, captured_estimates)

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
                        settings,
                        camera_id in captured_estimates,
                    )
                    cv2.imshow(f"calibration camera {camera_id}", preview)

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
        print(f"  {label:12s}: [{point[0]:+.5f}, {point[1]:+.5f}, {point[2]:+.5f}]")


def print_pose_status(
    sources: list[mocap.CameraSource],
    observations_by_camera: dict[int, list[mocap.MarkerObservation]],
    estimates: dict[int, PoseEstimate],
    captured_estimates: dict[int, PoseEstimate],
) -> None:
    parts: list[str] = []
    for source in sources:
        camera_id = source.camera_id
        blob_count = len(observations_by_camera.get(camera_id, []))
        estimate = estimates.get(camera_id)
        captured = " captured" if camera_id in captured_estimates else ""
        if estimate is None:
            parts.append(f"cam {camera_id}: {blob_count} blobs, no pose{captured}")
        else:
            x, y, z = estimate.position
            parts.append(
                f"cam {camera_id}: {blob_count} blobs, {estimate.marker_count} marker pose, "
                f"err={estimate.reprojection_error_px:.2f}px, "
                f"pos=({x:+.2f},{y:+.2f},{z:+.2f})m{captured}"
            )
    print("[calibration] " + " | ".join(parts))


if __name__ == "__main__":
    raise SystemExit(main())
