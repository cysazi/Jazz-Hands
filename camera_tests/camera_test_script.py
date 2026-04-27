import cv2
import time
from collections import deque


# Change these if Windows assigns the cameras different indexes.
CAMERA_INDEXES = [1, 2]
FRAME_WIDTH = 1280
FRAME_HEIGHT = 800
FPS = 120
THRESHOLD = 200
PREVIEW_SCALE = 0.5


def configure_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, -10)
    cap.set(cv2.CAP_PROP_GAIN, 0)
    return cap


def draw_label(frame, text, color=(255, 255, 255)):
    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
    )
    return frame


def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold bright regions.
    _, thresh = cv2.threshold(blur, THRESHOLD, 255, cv2.THRESH_BINARY)

    return thresh


def resize_preview(frame):
    return cv2.resize(
        frame,
        None,
        fx=PREVIEW_SCALE,
        fy=PREVIEW_SCALE,
        interpolation=cv2.INTER_AREA,
    )


def build_camera_preview(camera_index, frame, thresh, fps):
    raw_preview = resize_preview(frame.copy())
    thresh_preview = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    thresh_preview = resize_preview(thresh_preview)

    draw_label(raw_preview, f"Camera {camera_index} Raw | FPS: {fps:.1f}")
    draw_label(thresh_preview, f"Camera {camera_index} Threshold", color=(0, 255, 0))

    return cv2.hconcat([raw_preview, thresh_preview])


captures = {}
frame_times = {}

for camera_index in CAMERA_INDEXES:
    cap = configure_camera(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        cap.release()
        continue

    captures[camera_index] = cap
    frame_times[camera_index] = deque(maxlen=60)
    print(f"Opened camera {camera_index}")

if not captures:
    print("Error: Could not open any cameras")
    exit()

while True:
    current_time = time.time()
    preview_rows = []

    for camera_index, cap in captures.items():
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame from camera {camera_index}")
            continue

        frame_times[camera_index].append(current_time)
        if len(frame_times[camera_index]) > 1:
            fps = (
                (len(frame_times[camera_index]) - 1)
                / (frame_times[camera_index][-1] - frame_times[camera_index][0])
            )
        else:
            fps = 0

        thresh = process_frame(frame)
        preview_rows.append(build_camera_preview(camera_index, frame, thresh, fps))

    if preview_rows:
        cv2.imshow("Camera Test: Raw + Threshold", cv2.vconcat(preview_rows))

    # Press 'q' to exit the loop.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

for cap in captures.values():
    cap.release()

cv2.destroyAllWindows()
