from vispy import app, scene
import numpy as np

# 1) Window
canvas = scene.SceneCanvas(
    keys="interactive",
    show=True,
    bgcolor="black",
    size=(900, 700))
view = canvas.central_widget.add_view()

# 2) 3D camera (mouse: left-drag orbit, right-drag zoom, middle-drag pan)
view.camera = scene.cameras.TurntableCamera(
    fov=45,          # field of view (bigger = more wide-angle)
    distance=6.0,    # how far the camera starts from the center
    center=(0, 0, 0) # what point the camera orbits around
)

# 3) A reference grid on the "floor" (XY plane at z=0)
grid = scene.visuals.GridLines(
    scale=(1, 1),
    color=(0.3, 0.3, 0.3, 1.0))
view.add(grid)

# 4) XYZ axes (red=X, green=Y, blue=Z)
axes = scene.visuals.XYZAxis(width=2)
view.add(axes)

# 5) A simple cube (to prove 3D is working)
cube = scene.visuals.Box(
    width=1,
    height=1,
    depth=1,
    color=(0.3, 0.6, 1.0, 1),
    # edge_color="white",
)
view.add(cube)

cube_transform = scene.transforms.MatrixTransform()
cube.transform = cube_transform

x_angle_deg = 0

def apply_rotation():
    cube_transform.reset()
    cube_transform.rotate(x_angle_deg, (1,0,0)) # rotate around the X axis
    canvas.update()

def on_key_press(event):
    global x_angle_deg

    key = event.key.name

    if key == "Left":
        x_angle_deg -= 15
        apply_rotation()

    elif key == "Right":
        x_angle_deg += 15
        apply_rotation()

canvas.events.key_press.connect(on_key_press)


app.run()
