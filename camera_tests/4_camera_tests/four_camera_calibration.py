from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

CAMERA_TESTS_DIR = Path(__file__).resolve().parents[1]
if str(CAMERA_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(CAMERA_TESTS_DIR))

import calibrate_mocap_cameras as calibration
import four_camera_shared as shared

mocap = shared.mocap
cv2 = mocap.cv2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = calibration.build_arg_parser()
    parser.description = "Four-camera threaded calibration from the tetrahedral IR target."
    parser.set_defaults(
        cameras=list(shared.CAMERA_IDS),
        output=str(shared.STANDARD_CALIBRATION_PATH),
    )
    return parser


def create_preview_windows(camera_ids: list[int]) -> None:
    for camera_id in camera_ids:
        cv2.namedWindow(calibration.calibration_window_name(camera_id), cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            calibration.calibration_window_name(camera_id),
            calibration.PREVIEW_WINDOW_WIDTH,
            calibration.PREVIEW_WINDOW_HEIGHT,
        )


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if cv2 is None:
        print("OpenCV is required. Install it with: python -m pip install opencv-python")
        return 1

    target_points = calibration.load_target_points(args.target_config)
    intrinsic = mocap.default_intrinsic(args.width, args.height, args.focal_length_px)
    dist_coeffs = np.zeros(5, dtype=np.float64)

    calibration.print_target_geometry(target_points)
    if args.top_marker_label is not None:
        print(
            "[4cam calibration] top-marker constraint: "
            f"{args.top_marker_label} must have the smallest image y value "
            f"(within {args.top_marker_margin_px:.1f}px)"
        )

    settings_by_camera = {
        camera_id: calibration.build_detection_settings(args, camera_id)
        for camera_id in args.cameras
    }
    stop_event, workers = shared.start_threaded_cameras(
        args,
        list(args.cameras),
        settings_by_camera=settings_by_camera,
        build_masks=False,
        label="4cam calibration",
    )
    shared.wait_for_open_attempts(workers)
    snapshots = shared.collect_snapshots(workers)
    if not shared.any_camera_open(snapshots) and shared.all_open_attempts_done(snapshots):
        print("[4cam calibration] no cameras opened")
        shared.stop_threaded_cameras(stop_event, workers)
        return 1

    print("[4cam calibration] hold the tetrahedral target still at the intended origin")
    print("[4cam calibration] press c to capture valid camera poses, s to save, q/Esc to quit")

    show_preview = not args.no_preview
    if show_preview:
        create_preview_windows(list(args.cameras))

    captured_estimates: dict[int, calibration.PoseEstimate] = {}
    latest_estimates: dict[int, calibration.PoseEstimate] = {}
    pose_stability_by_camera: dict[int, calibration.PoseStabilityState] = {
        camera_id: calibration.PoseStabilityState() for camera_id in args.cameras
    }
    next_pose_solve_time_by_camera: dict[int, float] = {
        camera_id: 0.0 for camera_id in args.cameras
    }
    last_print_time = 0.0
    last_camera_stats_time = 0.0

    try:
        while True:
            timestamp = time.time()
            snapshots = shared.collect_snapshots(workers)
            latest_observations = shared.observations_from_snapshots(snapshots)
            latest_frames = shared.frames_from_snapshots(snapshots)
            last_camera_stats_time = shared.print_threaded_camera_stats(
                "4cam calibration",
                snapshots,
                1.0,
                last_camera_stats_time,
            )

            for camera_id, observations in latest_observations.items():
                if not snapshots[camera_id].opened:
                    continue
                if timestamp < next_pose_solve_time_by_camera.get(camera_id, 0.0):
                    continue
                next_pose_solve_time_by_camera[camera_id] = (
                    timestamp + max(args.pose_solve_interval, 0.05)
                )
                stability_state = pose_stability_by_camera[camera_id]
                estimate = calibration.solve_camera_pose(
                    camera_id,
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
                stable_estimate = calibration.update_pose_stability(
                    stability_state,
                    estimate,
                    args.assignment_stability_pixels,
                    args.stable_pose_frames,
                )
                if stable_estimate is not None:
                    latest_estimates[camera_id] = stable_estimate
                else:
                    latest_estimates.pop(camera_id, None)

            if timestamp - last_print_time > 0.5:
                last_print_time = timestamp
                calibration.print_pose_status(
                    workers,
                    latest_observations,
                    latest_estimates,
                    captured_estimates,
                    pose_stability_by_camera,
                    args.stable_pose_frames,
                )

            opened_camera_count = sum(1 for snapshot in snapshots.values() if snapshot.opened)
            if args.auto_save and opened_camera_count > 0 and len(latest_estimates) >= opened_camera_count:
                captured_estimates.update(latest_estimates)
                calibration.write_calibration_json(
                    args.output,
                    captured_estimates,
                    target_points,
                    intrinsic,
                    dist_coeffs,
                    args.width,
                    args.height,
                    args.focal_length_px,
                )
                print(f"[4cam calibration] saved {len(captured_estimates)} cameras to {args.output}")
                break

            if show_preview:
                for camera_id, frame in latest_frames.items():
                    preview = calibration.draw_estimate_overlay(
                        frame,
                        latest_observations.get(camera_id, []),
                        latest_estimates.get(camera_id),
                        target_points,
                        intrinsic,
                        dist_coeffs,
                        settings_by_camera[camera_id],
                        camera_id in captured_estimates,
                    )
                    cv2.imshow(calibration.calibration_window_name(camera_id), preview)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    if latest_estimates:
                        captured_estimates.update(latest_estimates)
                        print(f"[4cam calibration] captured cameras: {sorted(captured_estimates)}")
                    else:
                        print("[4cam calibration] no valid poses to capture yet")
                if key == ord("s"):
                    if not captured_estimates:
                        print("[4cam calibration] nothing captured yet; press c first")
                    else:
                        calibration.write_calibration_json(
                            args.output,
                            captured_estimates,
                            target_points,
                            intrinsic,
                            dist_coeffs,
                            args.width,
                            args.height,
                            args.focal_length_px,
                        )
                        print(f"[4cam calibration] saved {len(captured_estimates)} cameras to {args.output}")
            else:
                if latest_estimates:
                    captured_estimates.update(latest_estimates)
                    calibration.write_calibration_json(
                        args.output,
                        captured_estimates,
                        target_points,
                        intrinsic,
                        dist_coeffs,
                        args.width,
                        args.height,
                        args.focal_length_px,
                    )
                    print(f"[4cam calibration] saved {len(captured_estimates)} cameras to {args.output}")
                    break

    except KeyboardInterrupt:
        print("\n[4cam calibration] stopped")
    finally:
        shared.stop_threaded_cameras(stop_event, workers)
        if cv2 is not None:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
