from vispy import scene
def setup_canvas():
    """Call as:

    canvas, view, grid, axes, x_label, y_label, z_label = setup_canvas()

    Sets up the Canvas window in 1 function for importing and repeatability

    """
    # Set the canvas
    canvas = scene.SceneCanvas(
        keys="interactive",
        show=True,
        bgcolor="black",
        size=(900, 700)
    )
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(
        fov=45,
        distance=6.0,
        center=(0, 0, 0)
    )

    # Grid + axes
    grid = scene.visuals.GridLines(
        scale=(1, 1),
        color=(0.3, 0.3, 0.3, 1.0)
    )
    view.add(grid)

    axes = scene.visuals.XYZAxis(width=2)
    view.add(axes)

    #
    x_label = scene.visuals.Text(
        "X",
        color="red",
        font_size=25,
        pos=(1.2, 0, 0),
        parent=view.scene
    )
    y_label = scene.visuals.Text(
        "Y",
        color="green",
        font_size=25,
        pos=(0, 1.2, 0),
        parent=view.scene
    )
    z_label = scene.visuals.Text(
        "Z",
        color="blue",
        font_size=25,
        pos=(0, 0, 1.2),
        parent=view.scene
    )
    return canvas, view, grid, axes, x_label, y_label, z_label

def apply_euler(transform, roll, pitch, yaw):
    """Apply intrinsic ZYX Euler rotation (in degrees) to a MatrixTransform."""
    transform.rotate(yaw, (0, 0, 1))
    transform.rotate(pitch, (0, 1, 0))
    transform.rotate(roll, (1, 0, 0))