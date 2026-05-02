import cv2
import time
import threading
from collections import deque


cv2.setNumThreads(1)

# Change these if Windows assigns the cameras different indexes.
CAMERA_IDS = [1, 2]

FRAME_WIDTH = 1280
FRAME_HEIGHT = 800
FPS = 120
THRESHOLD = 230
AUTOFOCUS = 0
BLUR_KERNEL_BY_CAMERA = {
    1: 5,
    2: 5,
    3: 15,
}
AUTO_EXPOSURE_BY_CAMERA = {
    1: 0,
    2: 0,
    3: 0.25,
}
EXPOSURE_BY_CAMERA = {
    1: -8,
    2: -8,
    3: -8,
}
GAIN_BY_CAMERA = {
    1: 0,
    2: 0,
    3: 0,
}
PRINT_STATS_EVERY_SECONDS = 2.0
UI_IDLE_SLEEP_SECONDS = 0.001
PROCESS_FRAMES = True
SHOW_WINDOWS = True
SHOW_THRESHOLD_WINDOWS = True


def average_ms(values):
    if not values:
        return 0.0
    return 1000.0 * sum(values) / len(values)


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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    apply_manual_exposure(cap, camera_id)
    return cap


def apply_manual_exposure(cap, camera_id):
    cap.set(
        cv2.CAP_PROP_AUTO_EXPOSURE,
        AUTO_EXPOSURE_BY_CAMERA.get(camera_id, 0.0),
    )
    cap.set(cv2.CAP_PROP_AUTOFOCUS, AUTOFOCUS)
    cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE_BY_CAMERA.get(camera_id, -10))
    cap.set(cv2.CAP_PROP_GAIN, GAIN_BY_CAMERA.get(camera_id, 0))


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


def process_frame(frame, camera_id):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_kernel = BLUR_KERNEL_BY_CAMERA.get(camera_id, 1)
    if blur_kernel > 1:
        blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    _, thresh = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
    return thresh


class CameraWorker:
    def __init__(self, camera_id, stop_event):
        self.camera_id = camera_id
        self.stop_event = stop_event
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
            if self.raw_frame is None or self.threshold_frame is None:
                return (
                    None,
                    None,
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

        while not self.stop_event.is_set():
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
                thresh = process_frame(frame, self.camera_id)
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
    workers = [CameraWorker(camera_id, stop_event) for camera_id in CAMERA_IDS]

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
            for worker in workers:
                (
                    raw_frame,
                    threshold_frame,
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
                ) = worker.snapshot()
                any_open = any_open or opened
                all_open_attempts_done = all_open_attempts_done and open_attempt_done

                if error is not None and last_errors.get(worker.camera_id) != error:
                    print(error)
                    last_errors[worker.camera_id] = error
                elif error is None:
                    last_errors.pop(worker.camera_id, None)

                if not SHOW_WINDOWS or raw_frame is None:
                    continue

                if last_displayed_frames.get(worker.camera_id) == frame_number:
                    continue

                cv2.imshow(f"Camera {worker.camera_id} Raw", raw_frame)
                if SHOW_THRESHOLD_WINDOWS and threshold_frame is not None:
                    cv2.imshow(
                        f"Camera {worker.camera_id} Threshold",
                        threshold_frame,
                    )
                last_displayed_frames[worker.camera_id] = frame_number
                displayed_new_frame = True

            if not any_open and all_open_attempts_done:
                print("Error: Could not open any cameras")
                break

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
                    ) = worker.snapshot()
                    if opened:
                        stat_parts.append(
                            f"camera {worker.camera_id}: "
                            f"{fps:.1f} delivered fps, "
                            f"interval {interval_ms:.2f} ms, "
                            f"read {read_ms:.2f} ms, "
                            f"process {process_ms:.2f} ms, "
                            f"preview {preview_ms:.2f} ms"
                        )
                if stat_parts:
                    print("[stats] " + " | ".join(stat_parts))
                last_stats_time = now

            if cv2.waitKey(1) & 0xFF == ord("q"):
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
