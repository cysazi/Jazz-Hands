# Import VisPy's application system (event loop, timers, etc.)
# and the scene system (camera + visuals).
from vispy import app, scene

# Import NumPy for working with numerical arrays.
import numpy as np


# Create the main window (the actual OS window you see).
canvas = scene.SceneCanvas(
    keys='interactive',   # Allow keyboard/mouse interaction events
    show=True,            # Show the window immediately
    bgcolor="black",      # Set background color to black
    size=(800,800),       # Window size in pixels (width x height)
)

# Add a "view" inside the canvas.
# The view is like the camera space where visuals are placed.
view = canvas.central_widget.add_view()

# If you don't set a camera, VisPy uses a default one.
# Here you are manually setting the visible coordinate range.
view.camera.set_range(x=(0, 800), y=(0, 800))
# This means:
# Left edge = x=0
# Right edge = x=800
# Bottom = y=0
# Top = y=800


# Boolean flag: are we currently drawing?
is_drawing = False

# This list will store the points of the stroke currently being drawn.
current_stroke = []

# This list stores all completed stroke visuals (permanent lines).
stroke_visuals = []


# This is the temporary preview line (the one that updates while dragging).
current_line = scene.visuals.Line(method="gl", width=2)

# Add the preview line to the view so it appears.
view.add(current_line)


def redraw_current_line():
    """
    Update the preview line using the points collected in current_stroke.
    """

    # A line needs at least 2 points to exist.
    if len(current_stroke) < 2:
        return

    # Convert Python list of (x,y) points into a NumPy array.
    # Shape will be (N, 2)
    pts = np.array(current_stroke, dtype=np.float32)

    # Send the points to the GPU and update the preview line.
    # color=(1,0,0,1) means red (RGBA).
    current_line.set_data(pts, color=(1, 0, 0, 1))

    # Tell VisPy: "Something changed — redraw the screen."
    canvas.update()


def on_mouse_press(event):
    # Print debugging info in the console.
    print("Mouse Pressed", event.pos,
          "button:", event.button,
          "modifiers", event.modifiers)

    # Tell Python we want to modify these global variables.
    global is_drawing, current_stroke

    # Only respond to left mouse button (button 1).
    if event.button != 1:
        return

    # Start drawing mode.
    is_drawing = True

    # Start a new stroke with the first point.
    current_stroke = [event.pos]

    # Clear the preview line (empty it).
    current_line.set_data(np.zeros((0,2), dtype=np.float32))

    # Force screen redraw.
    canvas.update()


# Connect the mouse press event to the function.
canvas.events.mouse_press.connect(on_mouse_press)


def on_mouse_move(event):

    # Tell Python we will modify current_stroke.
    global current_stroke

    # If we are not currently drawing, do nothing.
    if not is_drawing:
        return

    # Add the current mouse position to the stroke.
    current_stroke.append(event.pos)

    # Update the preview line with the new point.
    redraw_current_line()


# Connect mouse movement to function.
canvas.events.mouse_move.connect(on_mouse_move)


def on_mouse_release(event):

    # Print debugging info.
    print("Mouse Released", event.pos,
          "button:", event.button,
          "modifiers", event.modifiers)

    global is_drawing, current_stroke

    # Only react to left mouse button release.
    if event.button != 1:
        return

    # Stop drawing mode.
    is_drawing = False

    # If we collected enough points to form a line:
    if len(current_stroke) >= 2:

        # Convert stroke points into NumPy array.
        pts = np.array(current_stroke, dtype=np.float32)

        # Create a NEW permanent line visual.
        finished_line = scene.visuals.Line(method="gl", width=2)

        # Send the stroke data to that line.
        finished_line.set_data(pts, color=(1,0,0,1))

        # Add the permanent line to the view.
        view.add(finished_line)

        # Store it in our list so it doesn't get lost.
        stroke_visuals.append(finished_line)

    # Reset stroke list for next drawing.
    current_stroke = []

    # Clear the preview line (so it's invisible).
    current_line.set_data(np.zeros((0,2), dtype=np.float32))

    # Redraw screen.
    canvas.update()


# Connect mouse release to function.
canvas.events.mouse_release.connect(on_mouse_release)


# Start the VisPy event loop.
# This keeps the window alive and listening for events.
app.run()
