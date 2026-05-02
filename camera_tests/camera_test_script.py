import cv2
import time

# Change these if Windows assigns the cameras different indexes.
CAMERA_INDEXES = [1]

# CAMERA_ROLES = {
#     "vertical_1": "Vertical_plane_camera_1",
#     "vertical_2": "Vertical_plane_camera_2",
# }

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
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    apply_manual_exposure(cap, camera_index)
    return cap


def apply_manual_exposure(cap, camera_index):
    cap.set(
        cv2.CAP_PROP_AUTO_EXPOSURE,
        AUTO_EXPOSURE_BY_CAMERA.get(camera_index, 0.0),
    )
    cap.set(cv2.CAP_PROP_AUTOFOCUS, AUTOFOCUS)
    cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE_BY_CAMERA.get(camera_index, -10))
    cap.set(cv2.CAP_PROP_GAIN, GAIN_BY_CAMERA.get(camera_index, 0))


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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_kernel = BLUR_KERNEL_BY_CAMERA.get(camera_index, 1)
    if blur_kernel > 1:
        blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    _, thresh = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
    return thresh


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
