import cv2


# Use 0, 1, or 2 to find the correct device node
camera_index = 1
cap = cv2.VideoCapture(camera_index)

# On some macOS versions, explicitly set the backend
# cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print(f"Error: Could not open camera {camera_index}")
    exit()

# Optional: Set resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    cv2.imshow('Arducam UVC Feed', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()