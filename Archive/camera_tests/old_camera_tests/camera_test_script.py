import cv2
import time

import multithreaded_camera_testing as camera_settings

# Change these if Windows assigns the cameras different indexes.
CAMERA_INDEXES = list(camera_settings.CAMERA_IDS)

# CAMERA_ROLES = {
#     "vertical_1": "Vertical_plane_camera_1",
#     "vertical_2": "Vertical_plane_camera_2",
# }

FRAME_WIDTH = camera_settings.FRAME_WIDTH
FRAME_HEIGHT = camera_settings.FRAME_HEIGHT
FPS = camera_settings.FPS
THRESHOLD = camera_settings.THRESHOLD
AUTOFOCUS = camera_settings.AUTOFOCUS
BLUR_KERNEL_BY_CAMERA = camera_settings.BLUR_KERNEL_BY_CAMERA
AUTO_EXPOSURE_BY_CAMERA = camera_settings.AUTO_EXPOSURE_BY_CAMERA
EXPOSURE_BY_CAMERA = camera_settings.EXPOSURE_BY_CAMERA
GAIN_BY_CAMERA = camera_settings.GAIN_BY_CAMERA


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


def configure_camera(camera_index):
    return camera_settings.configure_camera(camera_index)


def apply_manual_exposure(cap, camera_index):
    camera_settings.apply_manual_exposure(cap, camera_index)


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


def process_frame(frame, camera_index):
    return camera_settings.build_threshold_mask(frame, camera_index)


captures = {}
fps_counters = {}

for camera_index in CAMERA_INDEXES:
    cap = configure_camera(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        cap.release()
        continue

    captures[camera_index] = cap
    fps_counters[camera_index] = DeliveredFpsCounter()
    apply_manual_exposure(cap, camera_index)
    print(
        f"Opened camera {camera_index} | "
        f"auto_exposure={cap.get(cv2.CAP_PROP_AUTO_EXPOSURE):.2f} "
        f"autofocus={cap.get(cv2.CAP_PROP_AUTOFOCUS):.2f} "
        f"exposure={cap.get(cv2.CAP_PROP_EXPOSURE):.2f} "
        f"gain={cap.get(cv2.CAP_PROP_GAIN):.2f}"
    )

if not captures:
    print("Error: Could not open any cameras")
    exit()

while True:
    for camera_index, cap in captures.items():
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame from camera {camera_index}")
            continue

        fps = fps_counters[camera_index].mark_frame()
        thresh = process_frame(frame, camera_index)
        frame = draw_fps(frame, fps)
        cv2.imshow(f"Camera {camera_index} Raw", frame)
        cv2.imshow(f"Camera {camera_index} Threshold", thresh)

    # Press 'q' to exit the loop.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

for cap in captures.values():
    cap.release()

cv2.destroyAllWindows()
