import time
from pathlib import Path

import cv2
import numpy as np

import multithreaded_camera_testing as camera_settings
import mocap_tracker as mocap


# Exactly two OpenCV camera indexes. Change these when Windows assigns different indexes.
CAMERA_IDS = list(camera_settings.CAMERA_IDS[:2])

CALIBRATION_PATH = Path(__file__).resolve().with_name("mocap_calibration.json")
FRAME_WIDTH = camera_settings.FRAME_WIDTH
FRAME_HEIGHT = camera_settings.FRAME_HEIGHT
FOCAL_LENGTH_PX = 850.0

THRESHOLD = camera_settings.THRESHOLD
BLUR_KERNEL_BY_CAMERA = camera_settings.BLUR_KERNEL_BY_CAMERA
MIN_AREA = camera_settings.MIN_BLOB_AREA
MAX_AREA = camera_settings.MAX_BLOB_AREA
MIN_CIRCULARITY = camera_settings.MIN_BLOB_CIRCULARITY
MAX_ASPECT_RATIO = camera_settings.MAX_BLOB_ASPECT_RATIO

SHOW_WINDOWS = True
SHOW_BINARY_WINDOWS = False
PRINT_INTERVAL_SECONDS = 0.05


def open_camera(camera_id):
    cap = camera_settings.configure_camera(camera_id)
    if not cap.isOpened():
        cap.release()
        return None
    print(f"[simple mocap] opened camera {camera_id}")
    return cap


def preprocess(gray, camera_id):
    return camera_settings.threshold_source_gray(gray, camera_id)


def threshold_frame(frame, camera_id):
    gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray, camera_settings.build_threshold_mask(frame, camera_id)


def detect_single_blob(frame, camera_id, timestamp):
    gray, mask = threshold_frame(frame, camera_id)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_observation = None
    best_score = float("-inf")

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < MIN_AREA or area > MAX_AREA:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1e-6:
            continue

        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        if circularity < MIN_CIRCULARITY:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius <= 0.0:
            continue

        _bx, _by, width, height = cv2.boundingRect(contour)
        aspect_ratio = max(width, height) / max(min(width, height), 1)
        if aspect_ratio > MAX_ASPECT_RATIO:
            continue

        moments = cv2.moments(contour)
        if abs(moments["m00"]) <= 1e-6:
            pixel = np.array([x, y], dtype=np.float64)
        else:
            pixel = np.array(
                [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
                dtype=np.float64,
            )

        contour_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        brightness = float(cv2.mean(gray, mask=contour_mask)[0])
        score = area * circularity * brightness

        if score > best_score:
            best_score = score
            best_observation = mocap.MarkerObservation(
                camera_id=camera_id,
                pixel=pixel,
                radius_px=float(radius),
                area_px=area,
                circularity=circularity,
                brightness=brightness,
                score=float(score),
                timestamp=timestamp,
            )

    return best_observation, mask


def draw_preview(frame, observation, camera_id):
    preview = frame.copy()
    if observation is not None:
        center = tuple(int(round(value)) for value in observation.pixel)
        radius = max(3, int(round(observation.radius_px)))
        cv2.circle(preview, center, radius, (0, 255, 0), 2)
        cv2.putText(
            preview,
            f"blob ({center[0]}, {center[1]})",
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            preview,
            "no blob",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        preview,
        f"camera {camera_id}",
        (20, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return preview


def print_point(point, error_px, observations):
    x, y, z = point
    pixels = " | ".join(
        f"cam {obs.camera_id}: ({obs.pixel[0]:.1f}, {obs.pixel[1]:.1f})"
        for obs in observations
    )
    print(
        f"[simple mocap] x={x:+.3f}m y={y:+.3f}m z={z:+.3f}m "
        f"err={error_px:.1f}px | {pixels}"
    )


def main():
    if len(CAMERA_IDS) != 2:
        print("[simple mocap] CAMERA_IDS must contain exactly two camera indexes.")
        return 1

    if not CALIBRATION_PATH.exists():
        print(f"[simple mocap] calibration file not found: {CALIBRATION_PATH}")
        return 1

    calibrations = mocap.load_calibration_file(
        str(CALIBRATION_PATH),
        FRAME_WIDTH,
        FRAME_HEIGHT,
        FOCAL_LENGTH_PX,
    )
    missing_calibrations = [
        camera_id for camera_id in CAMERA_IDS if camera_id not in calibrations
    ]
    if missing_calibrations:
        print(f"[simple mocap] missing calibration for cameras: {missing_calibrations}")
        return 1

    cameras = {}
    for camera_id in CAMERA_IDS:
        cap = open_camera(camera_id)
        if cap is not None:
            cameras[camera_id] = cap

    if len(cameras) != 2:
        print("[simple mocap] both cameras must open for this simple two-camera test.")
        for cap in cameras.values():
            cap.release()
        return 1

    last_print_time = 0.0
    try:
        while True:
            timestamp = time.time()
            frames = {}
            observations = {}
            masks = {}

            for camera_id, cap in cameras.items():
                ok, frame = cap.read()
                if not ok or frame is None:
                    observations[camera_id] = None
                    continue

                observation, mask = detect_single_blob(frame, camera_id, timestamp)
                frames[camera_id] = frame
                observations[camera_id] = observation
                masks[camera_id] = mask

            obs_a = observations.get(CAMERA_IDS[0])
            obs_b = observations.get(CAMERA_IDS[1])
            if obs_a is not None and obs_b is not None:
                point = mocap.triangulate_two_views(
                    obs_a,
                    obs_b,
                    calibrations[CAMERA_IDS[0]],
                    calibrations[CAMERA_IDS[1]],
                )
                if point is not None and timestamp - last_print_time >= PRINT_INTERVAL_SECONDS:
                    last_print_time = timestamp
                    error_px = mocap.mean_reprojection_error(
                        point,
                        [obs_a, obs_b],
                        calibrations,
                    )
                    print_point(point, error_px, [obs_a, obs_b])

            if SHOW_WINDOWS:
                for camera_id, frame in frames.items():
                    cv2.imshow(
                        f"simple mocap camera {camera_id}",
                        draw_preview(frame, observations.get(camera_id), camera_id),
                    )
                    if SHOW_BINARY_WINDOWS:
                        cv2.imshow(f"simple mocap binary {camera_id}", masks[camera_id])

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    except KeyboardInterrupt:
        print("\n[simple mocap] stopped")
    finally:
        for cap in cameras.values():
            cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
