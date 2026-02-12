# Import deque (a “double-ended queue”) from Python’s collections library.
# We’ll use it like a list, but it’s efficient for keeping only the most recent items.
from collections import deque

# Import VisPy’s app system (event loop + timers) and scene system (camera + visuals).
from vispy import app, scene

# Import NumPy, which we’ll use for arrays and math like sin/cos.
import numpy as np

# Create the main window (“canvas”) where everything will be drawn.
canvas = scene.SceneCanvas(
    keys="interactive",  # Allow mouse/keyboard interactions (panning, zooming, events, etc.)
    show=True,           # Show the window immediately when the program runs
    bgcolor="black",     # Set the window background color to black
    size=(800, 600)      # Set the window size to 800 pixels wide by 600 pixels tall
)
# Think of the canvas as the actual app window + the system that receives mouse/keyboard events.

# Add a “view” inside the canvas. The view is like the camera window into your scene.
view = canvas.central_widget.add_view()

# Use a PanZoom camera so we can see the scene in 2D and allow panning/zooming.
view.camera = "panzoom"
# The view controls how you look at the scene (camera position, zoom level, etc.)

# Create a “Markers” visual (basically a dot or many dots).
dot = scene.visuals.Markers()

# Create a NumPy array holding the dot position.
# It’s shaped like (1, 2): one point, with x and y.
pos = np.array([[0.0, 0.0]], dtype=np.float32)

# Send the initial position to the dot visual.
# face_color="white" makes the dot white, size=12 sets its diameter-ish in pixels.
dot.set_data(pos, face_color="white", size=12)

# Add the dot visual into the view so it actually appears on screen.
view.add(dot)
# Visuals are “things you see”: dots, lines, shapes, etc.

# Create a Line visual that will be used as a “trail” behind the dot.
trail = scene.visuals.Line(method="gl", width=2)

# Add the trail line to the view so it can be drawn.
view.add(trail)

# Set the maximum number of points we want to keep in the trail.
max_trail = 600

# Create a history buffer that stores up to 600 recent positions.
# When it fills up, the oldest points automatically drop off.
history = deque(maxlen=max_trail)  # Think of this as “keep last 600 points”

# Set the visible camera range.
# Because the window is 800x600 (which is 4:3 aspect ratio), x is wider than y.
# So we show x from -4/3 to +4/3, and y from -1 to +1, centered at the origin (0,0).
view.camera.set_range(x=(-(4/3), (4/3)), y=(-1, 1))

# Create a time variable that controls the animation.
t = 0.0  # Starts at time = 0

# This function will be called repeatedly by the timer (like a game update loop).
def update(event):
    global t  # We want to modify the outer variable t inside this function

    # Increase time a little bit each frame.
    # Bigger number = faster motion. Smaller number = slower motion.
    t += 0.01

    # Compute the dot’s x-position using cosine.
    # cos(t) smoothly oscillates between -1 and +1.
    pos[0, 0] = np.cos(t)

    # Compute the dot’s y-position using sine.
    # sin(t) smoothly oscillates between -1 and +1.
    pos[0, 1] = np.sin(t)

    # Update the dot’s position on the screen by sending the new pos array to the GPU.
    dot.set_data(pos)

    # Add the current point (x,y) to the history list.
    # pos[0] is the first (and only) point: shape (2,) like [x, y].
    # .copy() ensures we store a snapshot of the values at this moment,
    # not a reference that could change later.
    history.append(pos[0].copy())

    # A line needs at least two points to draw.
    if len(history) >= 2:

        # Convert the history (a deque of [x,y] points) into a NumPy array.
        # This becomes shape (N, 2) where N is how many points are stored.
        trail_points = np.array(history, dtype=np.float32)

        # Update the trail line with all points in the history.
        # color=(1,1,1,1) means white (R,G,B) with full opacity (A).
        trail.set_data(trail_points, color=(1, 1, 1, 1))

# Create a timer that repeatedly calls update().
timer = app.Timer(
    interval=1/120,  # How often to call update(), in seconds (about 120 times per second)
    connect=update,  # The function to call each tick
    start=True       # Start the timer immediately
)

# Start the VisPy event loop.
# This keeps the window open and allows timers/events to run.
app.run()
