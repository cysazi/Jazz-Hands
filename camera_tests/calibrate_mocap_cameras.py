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
DEFAULT_CAMERA_IDS = mocap.DEFAULT_CAMERA_IDS
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 800
DEFAULT_FPS = 120
DEFAULT_OUTPUT_PATH = str(Path(__file__).resolve().with_name("mocap_calibration.json"))

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
) -> PoseEstimate | None:
    if len(observations) < 4:
        return None

    object_points_by_label = {
        label: target_points[label].astype(np.float64)
        for label in TARGET_POINT_LABELS
    }
    best: PoseEstimate | None = None

    candidate_observations = observations[: min(len(observations), 8)]
    for observation_group in combinations(candidate_observations, 4):
        for ordered_observations in permutations(observation_group, 4):
            assignment = {
                label: observation
                for label, observation in zip(TARGET_POINT_LABELS, ordered_observations)
            }
            object_points = np.asarray(
                [object_points_by_label[label] for label in TARGET_POINT_LABELS],
                dtype=np.float64,
            )
            image_points = np.asarray(
                [assignment[label].pixel for label in TARGET_POINT_LABELS],
                dtype=np.float64,
            )

            estimate = solve_pnp_candidate(
                camera_id,
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
            if best is None or estimate.score < best.score:
                best = estimate

    return best


def solve_pnp_candidate(
    camera_id: int,
    object_points: np.ndarray,
    image_points: np.ndarray,
    assignment: dict[str, mocap.MarkerObservation],
    intrinsic: np.ndarray,
    dist_coeffs: np.ndarray,
    size_weight: float,
) -> PoseEstimate | None:
    pnp_flag = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_EPNP)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        intrinsic,
        dist_coeffs,
        flags=pnp_flag,
    )
    if not ok:
        return None

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

    size_error = estimate_marker_size_error(assignment, camera_points)
    position = (-rotation.T @ tvec.reshape(3)).astype(np.float64)
    score = reprojection_error + size_weight * size_error

    return PoseEstimate(
        camera_id=camera_id,
        rvec=rvec.reshape(3).astype(np.float64),
        tvec=tvec.reshape(3).astype(np.float64),
        rotation=rotation.astype(np.float64),
        position=position,
        assignment=assignment,
        reprojection_error_px=reprojection_error,
        size_error=size_error,
        score=float(score),
    )


def estimate_marker_size_error(
    assignment: dict[str, mocap.MarkerObservation],
    camera_points: np.ndarray,
) -> float:
    observed = np.asarray(
        [assignment[label].radius_px for label in TARGET_POINT_LABELS],
        dtype=np.float64,
    )
    expected = np.asarray(
        [
            TARGET_MARKER_RADII_M[label] / max(float(camera_points[index, 2]), 1e-6)
            for index, label in enumerate(TARGET_POINT_LABELS)
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
    captured: bool,
) -> np.ndarray:
    preview = frame.copy()

    for observation in observations:
        center = tuple(int(round(value)) for value in observation.pixel)
        radius = max(3, int(round(observation.radius_px)))
        cv2.circle(preview, center, radius, (0, 200, 255), 1)

    if estimate is not None:
        object_points = np.asarray(
            [target_points[label] for label in TARGET_POINT_LABELS],
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

        for index, label in enumerate(TARGET_POINT_LABELS):
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
    status = "valid pose" if estimate is not None else "need 4 clean blobs"
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
    return preview


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
                "reprojection_error_px": estimate.reprojection_error_px,
                "size_error": estimate.size_error,
                "assigned_pixels": {
                    label: estimate.assignment[label].pixel.reshape(2).tolist()
                    for label in TARGET_POINT_LABELS
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
        if auto_exposure is not None:
            source.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(auto_exposure))
        if exposure is not None:
            source.capture.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
        if gain is not None:
            source.capture.set(cv2.CAP_PROP_GAIN, float(gain))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create mocap camera calibration JSON from the tetrahedral IR target.",
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
    parser.add_argument("--focal-length-px", type=float, default=mocap.DEFAULT_FOCAL_LENGTH_PX)
    parser.add_argument("--exposure", type=float, default=-10.0)
    parser.add_argument("--auto-exposure", type=float, default=0.0)
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
        max_markers_per_camera=8,
    )
    detector = mocap.ReflectiveMarkerDetector(settings)

    print_target_geometry(target_points)
    sources = mocap.open_available_cameras(
        args.cameras,
        args.width,
        args.height,
        args.fps,
        args.exposure,
    )
    apply_extra_camera_settings(sources, args.auto_exposure, args.exposure, args.gain)

    if not sources:
        print("[calibration] no cameras opened")
        return 1

    print("[calibration] hold the tetrahedral target still at the intended origin")
    print("[calibration] press c to capture valid camera poses, s to save, q/Esc to quit")

    show_preview = not args.no_preview
    captured_estimates: dict[int, PoseEstimate] = {}
    latest_estimates: dict[int, PoseEstimate] = {}
    last_print_time = 0.0

    try:
        while True:
            timestamp = time.time()
            latest_estimates = {}
            latest_observations: dict[int, list[mocap.MarkerObservation]] = {}
            latest_frames: dict[int, np.ndarray] = {}

            for source in sources:
                ok, frame = source.read()
                if not ok or frame is None:
                    latest_observations[source.camera_id] = []
                    continue

                latest_frames[source.camera_id] = frame
                observations = detector.detect(frame, source.camera_id, timestamp)
                latest_observations[source.camera_id] = observations
                estimate = solve_camera_pose(
                    source.camera_id,
                    observations,
                    target_points,
                    intrinsic,
                    dist_coeffs,
                    args.max_reprojection_error,
                    args.size_weight,
                )
                if estimate is not None:
                    latest_estimates[source.camera_id] = estimate

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
                f"cam {camera_id}: {blob_count} blobs, "
                f"err={estimate.reprojection_error_px:.2f}px, "
                f"pos=({x:+.2f},{y:+.2f},{z:+.2f})m{captured}"
            )
    print("[calibration] " + " | ".join(parts))


if __name__ == "__main__":
    raise SystemExit(main())
