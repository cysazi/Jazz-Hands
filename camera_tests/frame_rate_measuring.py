import cv2
import time
from collections import deque

# Use 0, 1, or 2 to find the correct device node
camera_index = 1
cap = cv2.VideoCapture(camera_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cap.set(cv2.CAP_PROP_FPS, 120)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_EXPOSURE, -10)
cap.set(cv2.CAP_PROP_GAIN, 0)


# print("Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print("FPS requested/reported:", cap.get(cv2.CAP_PROP_FPS))

# On some macOS versions, explicitly set the backend
# cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print(f"Error: Could not open camera {camera_index}")
    exit()

# Optional: Set resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_time=time.time()
frame_times = deque(maxlen=60)

def draw_fps(frame, fps):
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )
    return frame

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 100)

    # Threshold bright regions
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

    return gray, blur, edges, thresh

while True:
    ret, frame = cap.read()
    current_time = time.time()
    frame_times.append(current_time)

    if len(frame_times) > 1:
        fps = (len(frame_times)-1)/(frame_times[-1]-frame_times[0])
    else:
        fps = 0

    if not ret:
        print("Error: Could not read frame")
        break

    gray, blur, edges, thresh = process_frame(frame)

    frame = draw_fps(frame, fps)
    cv2.imshow('Arducam UVC Feed', frame)
    # cv2.imshow("Gray", gray)
    # cv2.imshow("Blur", blur)
    # cv2.imshow("Edges", edges)
    cv2.imshow("Threshold", thresh)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
