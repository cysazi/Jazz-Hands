from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

CAMERA_TESTS_DIR = Path(__file__).resolve().parents[1]
if str(CAMERA_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(CAMERA_TESTS_DIR))

import multithreaded_camera_testing as camera_test


CAMERA_IDS = list(camera_test.FOUR_CAMERA_IDS)
COMBINED_WINDOW_NAME = "4 Camera Multithreaded Preview"
CONTROL_WINDOW_NAME = "4 Camera Controls"

# 4-camera test-local knobs. The base frame size/FPS/threshold/blob settings
# still come from camera_tests/multithreaded_camera_testing.py so every script
# sees the same raw/threshold behavior. The UVC JSON below is for AMCap-style
# controls like brightness, contrast, gamma, backlight compensation, etc.
PANEL_WIDTH = 420
PANEL_HEIGHT = 260
PREVIEW_HZ = 30.0
CONTROL_PREVIEW_HZ = 5.0
APPLY_UVC_SETTINGS_ON_START = True
UVC_SETTINGS_JSON_PATH = Path(__file__).resolve().with_name("four_camera_uvc_settings_values.json")
EXPOSURE_BY_CAMERA = {
    1: -11,
    2: -11,
    3: -11,
    4: -11,
}
GAIN_BY_CAMERA = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
}


def configure_four_camera_defaults() -> None:
    camera_test.CAMERA_IDS = list(CAMERA_IDS)
    camera_test.DISPLAY_CAMERA_IDS = list(CAMERA_IDS)
    camera_test.COMBINED_WINDOW_NAME = COMBINED_WINDOW_NAME
    camera_test.CONTROL_WINDOW_NAME = CONTROL_WINDOW_NAME
    camera_test.PANEL_WIDTH = PANEL_WIDTH
    camera_test.PANEL_HEIGHT = PANEL_HEIGHT
    camera_test.EXPOSURE_BY_CAMERA.update(EXPOSURE_BY_CAMERA)
    camera_test.GAIN_BY_CAMERA.update(GAIN_BY_CAMERA)


def apply_four_camera_uvc_settings_on_start() -> None:
    if not APPLY_UVC_SETTINGS_ON_START:
        return
    try:
        import camera_uvc_settings
    except Exception as error:
        print(f"[4cam settings] could not import camera_uvc_settings: {error}")
        return

    if not UVC_SETTINGS_JSON_PATH.exists():
        print(f"[4cam settings] settings JSON not found: {UVC_SETTINGS_JSON_PATH}")
        return

    print(f"[4cam settings] applying saved UVC settings from {UVC_SETTINGS_JSON_PATH}")
    try:
        applied = camera_uvc_settings.apply_configured_camera_settings(
            camera_ids=list(CAMERA_IDS),
            settings_path=UVC_SETTINGS_JSON_PATH,
        )
    except Exception as error:
        print(f"[4cam settings] could not apply UVC settings: {error}")
        return
    print(f"[4cam settings] wrote {applied} UVC controls")


def create_windows(controls: camera_test.ProcessingControls) -> None:
    cv2 = camera_test.cv2
    cv2.namedWindow(COMBINED_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(COMBINED_WINDOW_NAME, PANEL_WIDTH * 4, PANEL_HEIGHT * 2)

    cv2.namedWindow(CONTROL_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROL_WINDOW_NAME, 720, 520)
    cv2.createTrackbar(
        "threshold",
        CONTROL_WINDOW_NAME,
        int(camera_test.THRESHOLD),
        255,
        controls.set_threshold,
    )

    for camera_id in CAMERA_IDS:
        cv2.createTrackbar(
            f"blur cam {camera_id}",
            CONTROL_WINDOW_NAME,
            camera_test.normalized_blur_kernel(
                camera_test.BLUR_KERNEL_BY_CAMERA.get(camera_id, 1)
            ),
            camera_test.MAX_BLUR_TRACKBAR_VALUE,
            lambda value, selected_camera_id=camera_id: controls.set_blur_kernel(
                selected_camera_id,
                value,
            ),
        )
        cv2.createTrackbar(
            f"exp cam {camera_id}",
            CONTROL_WINDOW_NAME,
            camera_test.exposure_to_slider(
                camera_test.EXPOSURE_BY_CAMERA.get(camera_id, -8)
            ),
            camera_test.EXPOSURE_SLIDER_MAX - camera_test.EXPOSURE_SLIDER_MIN,
            lambda value, selected_camera_id=camera_id: controls.set_exposure(
                selected_camera_id,
                value,
            ),
        )
        cv2.createTrackbar(
            f"gain cam {camera_id}",
            CONTROL_WINDOW_NAME,
            camera_test.gain_to_slider(camera_test.GAIN_BY_CAMERA.get(camera_id, 0)),
            camera_test.GAIN_SLIDER_MAX,
            lambda value, selected_camera_id=camera_id: controls.set_gain(
                selected_camera_id,
                value,
            ),
        )


def draw_panel_title(panel: np.ndarray, title: str) -> None:
    cv2 = camera_test.cv2
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(
        panel,
        title,
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def resize_panel(frame: np.ndarray) -> np.ndarray:
    return camera_test.cv2.resize(
        frame,
        (PANEL_WIDTH, PANEL_HEIGHT),
        interpolation=camera_test.cv2.INTER_AREA,
    )


def blank_panel(title: str) -> np.ndarray:
    panel = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    draw_panel_title(panel, title)
    camera_test.cv2.putText(
        panel,
        "no frame",
        (24, PANEL_HEIGHT // 2),
        camera_test.cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        camera_test.cv2.LINE_AA,
    )
    return panel


def threshold_to_bgr(threshold_frame: np.ndarray | None) -> np.ndarray | None:
    if threshold_frame is None:
        return None
    if len(threshold_frame.shape) == 2:
        return camera_test.cv2.cvtColor(threshold_frame, camera_test.cv2.COLOR_GRAY2BGR)
    return threshold_frame


def camera_role(camera_id: int) -> str:
    if camera_id in CAMERA_IDS[:2]:
        return "front Y/Z"
    if camera_id in CAMERA_IDS[2:4]:
        return "top X/Y"
    return "unassigned"


def build_controls_preview(controls: camera_test.ProcessingControls) -> np.ndarray:
    threshold, blur_by_camera, exposure_by_camera, gain_by_camera = controls.all_values()
    height = max(240, 86 + 34 * len(CAMERA_IDS))
    image = np.zeros((height, 720, 3), dtype=np.uint8)
    lines = [
        "Controls: q closes both windows",
        f"threshold: {threshold}",
        f"exposure slider maps {camera_test.EXPOSURE_SLIDER_MIN}..{camera_test.EXPOSURE_SLIDER_MAX}",
    ]
    for camera_id in CAMERA_IDS:
        lines.append(
            f"cam {camera_id} {camera_role(camera_id)}: "
            f"blur {blur_by_camera.get(camera_id, 1)} | "
            f"exposure {exposure_by_camera.get(camera_id, -8):+.0f} | "
            f"gain {gain_by_camera.get(camera_id, 0):.0f}"
        )

    y = 28
    for line in lines:
        camera_test.cv2.putText(
            image,
            line,
            (16, y),
            camera_test.cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            camera_test.cv2.LINE_AA,
        )
        y += 34
    return image


def build_camera_section(camera_id: int, snapshot, controls: camera_test.ProcessingControls) -> np.ndarray:
    if snapshot is None:
        return camera_test.cv2.hconcat(
            [
                blank_panel(f"cam {camera_id} raw | {camera_role(camera_id)}"),
                blank_panel(f"cam {camera_id} threshold"),
            ]
        )

    (
        raw_frame,
        threshold_frame,
        fps,
        opened,
        _failed,
        _open_attempt_done,
        _frame_number,
        _interval_ms,
        _read_ms,
        _process_ms,
        _preview_ms,
        error,
    ) = snapshot
    threshold, blur_kernel = controls.processing_settings_for_camera(camera_id)
    exposure, gain = controls.camera_settings_for_camera(camera_id)

    if raw_frame is None:
        raw_panel = blank_panel(f"cam {camera_id} raw | {camera_role(camera_id)}")
        if error is not None:
            camera_test.cv2.putText(
                raw_panel,
                error[:56],
                (20, PANEL_HEIGHT // 2 + 34),
                camera_test.cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 255),
                1,
                camera_test.cv2.LINE_AA,
            )
    else:
        raw_panel = raw_frame.copy()
        draw_panel_title(
            raw_panel,
            f"cam {camera_id} raw | {camera_role(camera_id)} | {fps:.1f} fps | exp {exposure:+.0f} gain {gain:.0f}",
        )

    threshold_bgr = threshold_to_bgr(threshold_frame)
    if threshold_bgr is None:
        threshold_panel = blank_panel(f"cam {camera_id} threshold")
    else:
        threshold_panel = threshold_bgr.copy()
        draw_panel_title(
            threshold_panel,
            f"cam {camera_id} threshold | {fps:.1f} fps | t {threshold} blur {blur_kernel}",
        )

    if not opened and raw_frame is None:
        draw_panel_title(raw_panel, f"cam {camera_id} opening")

    return camera_test.cv2.hconcat([resize_panel(raw_panel), resize_panel(threshold_panel)])


def build_combined_preview(snapshots: dict[int, tuple], controls: camera_test.ProcessingControls) -> np.ndarray:
    sections = [
        build_camera_section(camera_id, snapshots.get(camera_id), controls)
        for camera_id in CAMERA_IDS[:4]
    ]
    while len(sections) < 4:
        sections.append(
            camera_test.cv2.hconcat([blank_panel("unused raw"), blank_panel("unused threshold")])
        )
    top_row = camera_test.cv2.hconcat([sections[0], sections[1]])
    bottom_row = camera_test.cv2.hconcat([sections[2], sections[3]])
    return camera_test.cv2.vconcat([top_row, bottom_row])


def main() -> int:
    configure_four_camera_defaults()
    apply_four_camera_uvc_settings_on_start()
    stop_event = threading.Event()
    controls = camera_test.ProcessingControls()
    workers = [
        camera_test.CameraWorker(camera_id, stop_event, controls)
        for camera_id in CAMERA_IDS
    ]

    if camera_test.SHOW_WINDOWS:
        create_windows(controls)

    for worker in workers:
        worker.start()

    last_errors: dict[int, str] = {}
    last_displayed_frames: dict[int, int] = {}
    last_stats_time = time.time()
    last_preview_time = 0.0
    last_control_preview_time = 0.0
    controls_preview = None
    try:
        while True:
            any_open = False
            all_open_attempts_done = True
            displayed_new_frame = False
            snapshots = {}
            for worker in workers:
                snapshot = worker.snapshot()
                snapshots[worker.camera_id] = snapshot
                (
                    _raw_frame,
                    _threshold_frame,
                    _fps,
                    opened,
                    _failed,
                    open_attempt_done,
                    frame_number,
                    _interval_ms,
                    _read_ms,
                    _process_ms,
                    _preview_ms,
                    error,
                ) = snapshot
                any_open = any_open or opened
                all_open_attempts_done = all_open_attempts_done and open_attempt_done

                if error is not None and last_errors.get(worker.camera_id) != error:
                    print(error)
                    last_errors[worker.camera_id] = error
                elif error is None:
                    last_errors.pop(worker.camera_id, None)

                if last_displayed_frames.get(worker.camera_id) != frame_number:
                    last_displayed_frames[worker.camera_id] = frame_number
                    displayed_new_frame = True

            if not any_open and all_open_attempts_done:
                print("Error: Could not open any cameras")
                break

            now = time.time()
            if camera_test.SHOW_WINDOWS:
                preview_interval = 1.0 / max(float(PREVIEW_HZ), 1.0)
                if displayed_new_frame and now - last_preview_time >= preview_interval:
                    last_preview_time = now
                    camera_test.cv2.imshow(
                        COMBINED_WINDOW_NAME,
                        build_combined_preview(snapshots, controls),
                    )

                control_interval = 1.0 / max(float(CONTROL_PREVIEW_HZ), 1.0)
                if controls_preview is None or now - last_control_preview_time >= control_interval:
                    last_control_preview_time = now
                    controls_preview = build_controls_preview(controls)
                    camera_test.cv2.imshow(CONTROL_WINDOW_NAME, controls_preview)

            if now - last_stats_time >= camera_test.PRINT_STATS_EVERY_SECONDS:
                stat_parts = []
                for worker in workers:
                    (
                        _raw_frame,
                        _threshold_frame,
                        fps,
                        opened,
                        _failed,
                        _open_attempt_done,
                        _frame_number,
                        interval_ms,
                        read_ms,
                        process_ms,
                        preview_ms,
                        _error,
                    ) = snapshots.get(worker.camera_id, worker.snapshot())
                    if opened:
                        threshold, blur_kernel = controls.processing_settings_for_camera(worker.camera_id)
                        exposure, gain = controls.camera_settings_for_camera(worker.camera_id)
                        stat_parts.append(
                            f"camera {worker.camera_id}: "
                            f"{fps:.1f} fps, interval {interval_ms:.2f} ms, "
                            f"read {read_ms:.2f} ms, process {process_ms:.2f} ms, "
                            f"preview {preview_ms:.2f} ms, threshold {threshold}, "
                            f"blur {blur_kernel}, exposure {exposure:+.0f}, gain {gain:.0f}"
                        )
                if stat_parts:
                    print("[4cam stats] " + " | ".join(stat_parts))
                last_stats_time = now

            if camera_test.SHOW_WINDOWS:
                key = camera_test.cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if not displayed_new_frame:
                time.sleep(camera_test.UI_IDLE_SLEEP_SECONDS)
    finally:
        stop_event.set()
        for worker in workers:
            worker.join()
        camera_test.cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
