from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import numpy as np


cv2.setNumThreads(1)

# Change these if Windows assigns the cameras different indexes.
CAMERA_IDS = [1, 2]
FOUR_CAMERA_IDS = [1, 2, 3, 4]

FRAME_WIDTH = 1280
FRAME_HEIGHT = 800
FPS = 120
THRESHOLD = 230
MIN_BLOB_AREA = 0.0
MAX_BLOB_AREA = 4500.0
MIN_BLOB_RADIUS = 0.0
MAX_BLOB_RADIUS = 80.0
MIN_BLOB_CIRCULARITY = 0.30
MIN_BLOB_FILL_RATIO = 0.20
MAX_BLOB_ASPECT_RATIO = 3.0
MORPHOLOGY_KERNEL = 1
AUTOFOCUS = 0
BLUR_KERNEL_BY_CAMERA = {
    1: 15,
    2: 15,
    3: 15,
    4: 15,
}
AUTO_EXPOSURE_BY_CAMERA = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
}
EXPOSURE_BY_CAMERA = {
    1: -10,
    2: -10,
    3: -8,
    4: -8,
}
GAIN_BY_CAMERA = {
    1: 25,
    2: 25,
    3: 0,
    4: 0,
}

PRINT_STATS_EVERY_SECONDS = 2.0
UI_IDLE_SLEEP_SECONDS = 0.001
PROCESS_FRAMES = True
SHOW_WINDOWS = True
APPLY_CAMERA_SETTINGS_ON_OPEN = True

COMBINED_WINDOW_NAME = "Multithreaded Camera Preview"
CONTROL_WINDOW_NAME = "Multithreaded Camera Controls"
DISPLAY_CAMERA_IDS = CAMERA_IDS[:2]
PANEL_WIDTH = 640
PANEL_HEIGHT = 400
MAX_BLUR_TRACKBAR_VALUE = 31
EXPOSURE_SLIDER_MIN = -13
EXPOSURE_SLIDER_MAX = 0
GAIN_SLIDER_MAX = 255


def average_ms(values):
    if not values:
        return 0.0
    return 1000.0 * sum(values) / len(values)


def normalized_blur_kernel(value):
    value = int(value)
    if value <= 1:
        return 1
    value = value if value % 2 == 1 else value + 1
    return max(1, min(value, MAX_BLUR_TRACKBAR_VALUE))


def blur_kernel_for_camera(camera_id, blur_kernel=None):
    if blur_kernel is None:
        blur_kernel = BLUR_KERNEL_BY_CAMERA.get(camera_id, 1)
    return normalized_blur_kernel(blur_kernel)


def threshold_value(threshold=None):
    if threshold is None:
        threshold = THRESHOLD
    return int(np.clip(threshold, 0, 255))


def frame_to_gray(frame):
    if len(frame.shape) == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def threshold_source_gray(frame, camera_id, blur_kernel=None):
    gray = frame_to_gray(frame)
    kernel = blur_kernel_for_camera(camera_id, blur_kernel)
    if kernel > 1:
        gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    return gray


def build_threshold_mask(frame, camera_id, threshold=None, blur_kernel=None):
    gray = threshold_source_gray(frame, camera_id, blur_kernel)
    _ok, mask = cv2.threshold(gray, threshold_value(threshold), 255, cv2.THRESH_BINARY)

    kernel_size = int(MORPHOLOGY_KERNEL)
    if kernel_size > 1:
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def exposure_to_slider(exposure):
    value = int(round(float(exposure) - EXPOSURE_SLIDER_MIN))
    return int(np.clip(value, 0, EXPOSURE_SLIDER_MAX - EXPOSURE_SLIDER_MIN))


def slider_to_exposure(value):
    return float(EXPOSURE_SLIDER_MIN + int(value))


def gain_to_slider(gain):
    return int(np.clip(round(float(gain)), 0, GAIN_SLIDER_MAX))


def camera_setting(camera_id, values_by_camera, override, fallback):
    if override is not None:
        return float(override)
    return float(values_by_camera.get(camera_id, fallback))


def camera_auto_exposure(camera_id, override=None):
    return camera_setting(camera_id, AUTO_EXPOSURE_BY_CAMERA, override, 0.0)


def camera_exposure(camera_id, override=None):
    return camera_setting(camera_id, EXPOSURE_BY_CAMERA, override, -8.0)


def camera_gain(camera_id, override=None):
    return camera_setting(camera_id, GAIN_BY_CAMERA, override, 0.0)


def apply_camera_capture_settings(
    cap,
    camera_id,
    width=FRAME_WIDTH,
    height=FRAME_HEIGHT,
    fps=FPS,
    exposure=None,
    auto_exposure=None,
    gain=None,
):
    if not APPLY_CAMERA_SETTINGS_ON_OPEN:
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, camera_auto_exposure(camera_id, auto_exposure))
    cap.set(cv2.CAP_PROP_AUTOFOCUS, AUTOFOCUS)
    cap.set(cv2.CAP_PROP_EXPOSURE, camera_exposure(camera_id, exposure))
    cap.set(cv2.CAP_PROP_GAIN, camera_gain(camera_id, gain))


class ProcessingControls:
    def __init__(self):
        self.lock = threading.Lock()
        self.threshold = int(THRESHOLD)
        self.blur_kernel_by_camera = {
            camera_id: normalized_blur_kernel(blur_kernel)
            for camera_id, blur_kernel in BLUR_KERNEL_BY_CAMERA.items()
        }
        self.exposure_by_camera = {
            camera_id: float(EXPOSURE_BY_CAMERA.get(camera_id, -8))
            for camera_id in CAMERA_IDS
        }
        self.gain_by_camera = {
            camera_id: float(GAIN_BY_CAMERA.get(camera_id, 0))
            for camera_id in CAMERA_IDS
        }

    def processing_settings_for_camera(self, camera_id):
        with self.lock:
            return (
                int(self.threshold),
                int(self.blur_kernel_by_camera.get(camera_id, 1)),
            )

    def camera_settings_for_camera(self, camera_id):
        with self.lock:
            return (
                float(self.exposure_by_camera.get(camera_id, -8)),
                float(self.gain_by_camera.get(camera_id, 0)),
            )

    def all_values(self):
        with self.lock:
            return (
                int(self.threshold),
                dict(self.blur_kernel_by_camera),
                dict(self.exposure_by_camera),
                dict(self.gain_by_camera),
            )

    def set_threshold(self, value):
        with self.lock:
            self.threshold = int(np.clip(value, 0, 255))

    def set_blur_kernel(self, camera_id, value):
        with self.lock:
            self.blur_kernel_by_camera[camera_id] = normalized_blur_kernel(value)

    def set_exposure(self, camera_id, value):
        with self.lock:
            self.exposure_by_camera[camera_id] = slider_to_exposure(value)

    def set_gain(self, camera_id, value):
        with self.lock:
            self.gain_by_camera[camera_id] = float(np.clip(value, 0, GAIN_SLIDER_MAX))


class DeliveredFpsCounter:
    def __init__(self):
        self.window_start = time.perf_counter()
        self.frames_this_window = 0
        self.fps = 0.0

    def mark_frame(self):
        self.frames_this_window += 1
        now = time.perf_counter()
        elapsed = now - self.window_start
        if elapsed >= 1.0:
            self.fps = self.frames_this_window / elapsed
            self.frames_this_window = 0
            self.window_start = now
        return self.fps


def configure_camera(camera_id):
    cap = cv2.VideoCapture(camera_id)
    apply_camera_capture_settings(cap, camera_id)
    return cap


def apply_manual_exposure(cap, camera_id):
    apply_camera_capture_settings(cap, camera_id)


def draw_fps(frame, fps):
    cv2.putText(
        frame,
        f"Delivered FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    return frame


def process_frame(frame, camera_id, controls):
    threshold, blur_kernel = controls.processing_settings_for_camera(camera_id)
    return build_threshold_mask(
        frame,
        camera_id,
        threshold=threshold,
        blur_kernel=blur_kernel,
    )


def apply_live_camera_settings(cap, camera_id, controls, last_applied):
    exposure, gain = controls.camera_settings_for_camera(camera_id)
    auto_exposure = float(AUTO_EXPOSURE_BY_CAMERA.get(camera_id, 0.0))
    settings = (auto_exposure, exposure, gain)
    if settings == last_applied:
        return last_applied

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, AUTOFOCUS)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    cap.set(cv2.CAP_PROP_GAIN, gain)
    return settings


def create_windows(controls):
    cv2.namedWindow(COMBINED_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(COMBINED_WINDOW_NAME, PANEL_WIDTH * 2, PANEL_HEIGHT * 2)

    cv2.namedWindow(CONTROL_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROL_WINDOW_NAME, 620, 280)
    cv2.createTrackbar(
        "threshold",
        CONTROL_WINDOW_NAME,
        int(THRESHOLD),
        255,
        controls.set_threshold,
    )

    for camera_id in CAMERA_IDS:
        initial_blur = normalized_blur_kernel(BLUR_KERNEL_BY_CAMERA.get(camera_id, 1))
        cv2.createTrackbar(
            f"blur cam {camera_id}",
            CONTROL_WINDOW_NAME,
            initial_blur,
            MAX_BLUR_TRACKBAR_VALUE,
            lambda value, selected_camera_id=camera_id: controls.set_blur_kernel(
                selected_camera_id,
                value,
            ),
        )
        cv2.createTrackbar(
            f"exp cam {camera_id}",
            CONTROL_WINDOW_NAME,
            exposure_to_slider(EXPOSURE_BY_CAMERA.get(camera_id, -8)),
            EXPOSURE_SLIDER_MAX - EXPOSURE_SLIDER_MIN,
            lambda value, selected_camera_id=camera_id: controls.set_exposure(
                selected_camera_id,
                value,
            ),
        )
        cv2.createTrackbar(
            f"gain cam {camera_id}",
            CONTROL_WINDOW_NAME,
            gain_to_slider(GAIN_BY_CAMERA.get(camera_id, 0)),
            GAIN_SLIDER_MAX,
            lambda value, selected_camera_id=camera_id: controls.set_gain(
                selected_camera_id,
                value,
            ),
        )


def build_controls_preview(controls):
    threshold, blur_by_camera, exposure_by_camera, gain_by_camera = controls.all_values()
    height = max(220, 70 + 34 * len(CAMERA_IDS))
    image = np.zeros((height, 620, 3), dtype=np.uint8)
    lines = [
        "Controls: q closes both windows",
        f"threshold: {threshold}",
        f"exposure slider maps {EXPOSURE_SLIDER_MIN}..{EXPOSURE_SLIDER_MAX}",
    ]
    for camera_id in CAMERA_IDS:
        lines.append(
            f"cam {camera_id}: "
            f"blur {blur_by_camera.get(camera_id, 1)} | "
            f"exposure {exposure_by_camera.get(camera_id, -8):+.0f} | "
            f"gain {gain_by_camera.get(camera_id, 0):.0f}"
        )

    y = 28
    for line in lines:
        cv2.putText(
            image,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 34
    return image


def draw_panel_title(panel, title):
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(
        panel,
        title,
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def resize_panel(frame):
    return cv2.resize(frame, (PANEL_WIDTH, PANEL_HEIGHT), interpolation=cv2.INTER_AREA)


def blank_panel(title):
    panel = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    draw_panel_title(panel, title)
    cv2.putText(
        panel,
        "no frame",
        (24, PANEL_HEIGHT // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def threshold_to_bgr(threshold_frame):
    if threshold_frame is None:
        return None
    if len(threshold_frame.shape) == 2:
        return cv2.cvtColor(threshold_frame, cv2.COLOR_GRAY2BGR)
    return threshold_frame


def build_combined_preview(camera_ids, snapshots, controls):
    panels = []
    for camera_id in camera_ids[:2]:
        snapshot = snapshots.get(camera_id)
        if snapshot is None:
            panels.append(blank_panel(f"camera {camera_id} raw"))
            panels.append(blank_panel(f"camera {camera_id} threshold"))
            continue

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
            raw_panel = blank_panel(f"camera {camera_id} raw")
            if error is not None:
                cv2.putText(
                    raw_panel,
                    error[:60],
                    (24, PANEL_HEIGHT // 2 + 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
        else:
            raw_panel = raw_frame.copy()
            title = f"cam {camera_id} raw | {fps:.1f} fps | exp {exposure:+.0f} gain {gain:.0f}"
            draw_panel_title(raw_panel, title)

        threshold_bgr = threshold_to_bgr(threshold_frame)
        if threshold_bgr is None:
            threshold_panel = blank_panel(f"camera {camera_id} threshold")
        else:
            threshold_panel = threshold_bgr.copy()
            title = (
                f"cam {camera_id} threshold | {fps:.1f} fps | "
                f"t {threshold} blur {blur_kernel}"
            )
            draw_panel_title(threshold_panel, title)

        if not opened and raw_frame is None:
            draw_panel_title(raw_panel, f"camera {camera_id} opening")

        panels.append(resize_panel(raw_panel))
        panels.append(resize_panel(threshold_panel))

    while len(panels) < 4:
        panels.append(blank_panel("unused"))

    top_row = cv2.hconcat([panels[0], panels[1]])
    bottom_row = cv2.hconcat([panels[2], panels[3]])
    return cv2.vconcat([top_row, bottom_row])


class CameraWorker:
    def __init__(self, camera_id, stop_event, controls):
        self.camera_id = camera_id
        self.stop_event = stop_event
        self.controls = controls
        self.lock = threading.Lock()
        self.fps_counter = DeliveredFpsCounter()
        self.frame_interval_times = deque(maxlen=60)
        self.read_times = deque(maxlen=60)
        self.process_times = deque(maxlen=60)
        self.preview_times = deque(maxlen=60)
        self.raw_frame = None
        self.threshold_frame = None
        self.fps = 0.0
        self.frame_number = 0
        self.last_frame_time = None
        self.opened = False
        self.failed = False
        self.open_attempt_done = False
        self.error = None
        self.thread = threading.Thread(
            target=self.run,
            name=f"Camera {camera_id} worker",
            daemon=True,
        )

    def start(self):
        self.thread.start()

    def join(self):
        self.thread.join()

    def snapshot(self):
        with self.lock:
            return (
                self.raw_frame,
                self.threshold_frame,
                self.fps,
                self.opened,
                self.failed,
                self.open_attempt_done,
                self.frame_number,
                average_ms(self.frame_interval_times),
                average_ms(self.read_times),
                average_ms(self.process_times),
                average_ms(self.preview_times),
                self.error,
            )

    def run(self):
        print(f"Opening camera {self.camera_id} on thread {threading.get_native_id()}...")
        cap = configure_camera(self.camera_id)
        if not cap.isOpened():
            cap.release()
            with self.lock:
                self.failed = True
                self.open_attempt_done = True
                self.error = f"Error: Could not open camera {self.camera_id}"
            return

        apply_manual_exposure(cap, self.camera_id)
        with self.lock:
            self.opened = True
            self.open_attempt_done = True

        print(
            f"Opened camera {self.camera_id} | "
            f"auto_exposure={cap.get(cv2.CAP_PROP_AUTO_EXPOSURE):.2f} "
            f"autofocus={cap.get(cv2.CAP_PROP_AUTOFOCUS):.2f} "
            f"exposure={cap.get(cv2.CAP_PROP_EXPOSURE):.2f} "
            f"gain={cap.get(cv2.CAP_PROP_GAIN):.2f}"
        )

        last_applied_settings = None
        while not self.stop_event.is_set():
            last_applied_settings = apply_live_camera_settings(
                cap,
                self.camera_id,
                self.controls,
                last_applied_settings,
            )

            read_start = time.perf_counter()
            ret, frame = cap.read()
            read_seconds = time.perf_counter() - read_start
            if not ret:
                with self.lock:
                    self.error = (
                        f"Error: Could not read frame from camera {self.camera_id}"
                    )
                time.sleep(0.01)
                continue

            frame_time = time.perf_counter()
            if self.last_frame_time is None:
                frame_interval_seconds = 0.0
            else:
                frame_interval_seconds = frame_time - self.last_frame_time
            self.last_frame_time = frame_time

            fps = self.fps_counter.mark_frame()

            if PROCESS_FRAMES:
                process_start = time.perf_counter()
                thresh = process_frame(frame, self.camera_id, self.controls)
                process_seconds = time.perf_counter() - process_start
            else:
                thresh = None
                process_seconds = 0.0

            preview_start = time.perf_counter()
            raw_preview = draw_fps(frame, fps) if SHOW_WINDOWS else None
            preview_seconds = time.perf_counter() - preview_start

            with self.lock:
                if frame_interval_seconds > 0:
                    self.frame_interval_times.append(frame_interval_seconds)
                self.read_times.append(read_seconds)
                self.process_times.append(process_seconds)
                self.preview_times.append(preview_seconds)
                self.raw_frame = raw_preview
                self.threshold_frame = thresh
                self.fps = fps
                self.frame_number += 1
                self.error = None

        cap.release()


def main():
    stop_event = threading.Event()
    controls = ProcessingControls()
    workers = [
        CameraWorker(camera_id, stop_event, controls)
        for camera_id in CAMERA_IDS
    ]

    if SHOW_WINDOWS:
        create_windows(controls)

    for worker in workers:
        worker.start()

    last_errors = {}
    last_displayed_frames = {}
    last_stats_time = time.time()
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

            if SHOW_WINDOWS and displayed_new_frame:
                combined_preview = build_combined_preview(
                    DISPLAY_CAMERA_IDS,
                    snapshots,
                    controls,
                )
                cv2.imshow(COMBINED_WINDOW_NAME, combined_preview)
                cv2.imshow(CONTROL_WINDOW_NAME, build_controls_preview(controls))

            now = time.time()
            if now - last_stats_time >= PRINT_STATS_EVERY_SECONDS:
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
                        threshold, blur_kernel = controls.processing_settings_for_camera(
                            worker.camera_id
                        )
                        exposure, gain = controls.camera_settings_for_camera(worker.camera_id)
                        stat_parts.append(
                            f"camera {worker.camera_id}: "
                            f"{fps:.1f} delivered fps, "
                            f"interval {interval_ms:.2f} ms, "
                            f"read {read_ms:.2f} ms, "
                            f"process {process_ms:.2f} ms, "
                            f"preview {preview_ms:.2f} ms, "
                            f"threshold {threshold}, blur {blur_kernel}, "
                            f"exposure {exposure:+.0f}, gain {gain:.0f}"
                        )
                if stat_parts:
                    print("[stats] " + " | ".join(stat_parts))
                last_stats_time = now

            if SHOW_WINDOWS:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if not displayed_new_frame:
                time.sleep(UI_IDLE_SLEEP_SECONDS)
    finally:
        stop_event.set()
        for worker in workers:
            worker.join()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
