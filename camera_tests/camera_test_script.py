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
MIN_DOT_AREA = 8
MAX_DOT_AREA = 2500
MIN_CIRCULARITY_PERCENT = 55
MIN_FILL_PERCENT = 40
MAX_RADIUS_PX = 45
CONTROLS_WINDOW = "Controls"


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
    threshold = cv2.getTrackbarPos("threshold", CONTROLS_WINDOW)
    _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

    return thresh


def detect_dot_candidates(thresh):
    min_area = max(1, cv2.getTrackbarPos("min area", CONTROLS_WINDOW))
    max_area = max(min_area + 1, cv2.getTrackbarPos("max area", CONTROLS_WINDOW))
    min_circularity = cv2.getTrackbarPos("min circularity %", CONTROLS_WINDOW) / 100.0
    min_fill = cv2.getTrackbarPos("min fill %", CONTROLS_WINDOW) / 100.0
    max_radius = max(1, cv2.getTrackbarPos("max radius px", CONTROLS_WINDOW))

    contours, _hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots = []
    rejected_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        circularity = 4.0 * 3.14159 * area / (perimeter * perimeter)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        fill = area / max(3.14159 * radius * radius, 1.0)

        if (
            min_area <= area <= max_area
            and circularity >= min_circularity
            and fill >= min_fill
            and radius <= max_radius
        ):
            dots.append((int(x), int(y), int(radius), area, circularity))
        else:
            rejected_contours.append(contour)

    dots.sort(key=lambda item: item[3], reverse=True)
    return dots, rejected_contours


def resize_preview(frame):
    return cv2.resize(
        frame,
        None,
        fx=PREVIEW_SCALE,
        fy=PREVIEW_SCALE,
        interpolation=cv2.INTER_AREA,
    )


def build_camera_preview(camera_index, frame, thresh, dots, rejected_contours, fps):
    raw_preview = resize_preview(frame.copy())
    thresh_preview = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    thresh_preview = resize_preview(thresh_preview)
    filtered_preview = frame.copy()

    cv2.drawContours(filtered_preview, rejected_contours, -1, (0, 0, 120), 1)
    for index, (x, y, radius, _area, circularity) in enumerate(dots, start=1):
        cv2.circle(filtered_preview, (x, y), max(radius, 4), (0, 255, 0), 2)
        cv2.putText(
            filtered_preview,
            f"{index}: {circularity:.2f}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    filtered_preview = resize_preview(filtered_preview)

    draw_label(raw_preview, f"Camera {camera_index} Raw | FPS: {fps:.1f}")
    draw_label(thresh_preview, f"Camera {camera_index} Threshold", color=(0, 255, 0))
    draw_label(filtered_preview, f"Camera {camera_index} Filtered Dots: {len(dots)}", color=(0, 255, 0))

    return cv2.hconcat([raw_preview, thresh_preview, filtered_preview])


def setup_controls():
    cv2.namedWindow(CONTROLS_WINDOW)
    cv2.createTrackbar("threshold", CONTROLS_WINDOW, THRESHOLD, 255, lambda _value: None)
    cv2.createTrackbar("min area", CONTROLS_WINDOW, MIN_DOT_AREA, 1000, lambda _value: None)
    cv2.createTrackbar("max area", CONTROLS_WINDOW, MAX_DOT_AREA, 10000, lambda _value: None)
    cv2.createTrackbar(
        "min circularity %",
        CONTROLS_WINDOW,
        MIN_CIRCULARITY_PERCENT,
        100,
        lambda _value: None,
    )
    cv2.createTrackbar("min fill %", CONTROLS_WINDOW, MIN_FILL_PERCENT, 100, lambda _value: None)
    cv2.createTrackbar("max radius px", CONTROLS_WINDOW, MAX_RADIUS_PX, 150, lambda _value: None)


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

setup_controls()

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
        dots, rejected_contours = detect_dot_candidates(thresh)
        preview_rows.append(build_camera_preview(camera_index, frame, thresh, dots, rejected_contours, fps))

    if preview_rows:
        cv2.imshow("Camera Test: Raw + Threshold", cv2.vconcat(preview_rows))

    # Press 'q' to exit the loop.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

for cap in captures.values():
    cap.release()

cv2.destroyAllWindows()
