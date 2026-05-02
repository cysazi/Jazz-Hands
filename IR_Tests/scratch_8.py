import cv2
import time
from collections import deque

# Use 0, 1, or 2 to find the correct device node
camera_index = 0
cap = cv2.VideoCapture(camera_index)
print(cap.get(cv2.CAP_PROP_FPS))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cap.set(cv2.CAP_PROP_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_ISO_SPEED, 100)

print("Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("FPS requested/reported:", cap.get(cv2.CAP_PROP_FPS))


print(cap.get(cv2.CAP_PROP_FPS))
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

    frame = draw_fps(frame, fps)
    cv2.imshow('Arducam UVC Feed', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()