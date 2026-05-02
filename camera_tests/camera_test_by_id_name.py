import cv2
from pygrabber.dshow_graph import FilterGraph


CAMERA_ROLES = {
    "vertical_1": "Vertical_plane_camera_1",
    "vertical_2": "Vertical_plane_camera_2",
}


def list_windows_cameras():
    graph = FilterGraph()
    return graph.get_input_devices()


def find_camera_index_by_name(target_name):
    devices = list_windows_cameras()

    for index, device_name in enumerate(devices):
        if target_name.lower() in device_name.lower():
            return index

    raise RuntimeError(f"Could not find camera named: {target_name}")


def open_camera(role_name, device_name):
    index = find_camera_index_by_name(device_name)

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {role_name} at index {index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    cap.set(cv2.CAP_PROP_FPS, 120)

    # You can change these later
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    cap.set(cv2.CAP_PROP_GAIN, 0)

    print(f"{role_name} opened as OpenCV index {index}: {device_name}")

    return cap


cameras = {}

for role_name, device_name in CAMERA_ROLES.items():
    cameras[role_name] = open_camera(role_name, device_name)


while True:
    ret1, frame1 = cameras["vertical_1"].read()
    ret2, frame2 = cameras["vertical_2"].read()

    if ret1:
        cv2.imshow("Vertical Camera 1", frame1)

    if ret2:
        cv2.imshow("Vertical Camera 2", frame2)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


for cap in cameras.values():
    cap.release()

cv2.destroyAllWindows()