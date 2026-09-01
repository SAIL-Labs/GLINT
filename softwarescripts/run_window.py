import numpy as np
import matplotlib.pyplot as plt
import time
from pyMilk.interfacing.isio_shmlib import SHM
from matplotlib.animation import FuncAnimation

# Toggle variables: set these to True to display the corresponding lines
SHOW_RATIO = False
SHOW_BRIGHT1 = True
SHOW_BRIGHT2 = True
SHOW_IMWFS2 = False

# Settings
loopnum = 6
window_duration = 10.0  # Window duration in seconds
interval_ms = 40       # Update interval in milliseconds
cred2 = False

box_halfwidth = 3

if cred2:
    top1 = 8
    bottom1 = 68
    top2 = 8
    bottom2 = 68
    bright1, bright2 = 115, 177

    left1 = bright1-box_halfwidth
    right1 = bright1+box_halfwidth

    left2 = bright2-box_halfwidth
    right2 = bright2+box_halfwidth

else:
    

# Initialise shared memory interface
if SHOW_IMWFS2:
    detector = SHM(f'aol{loopnum}_imWFS2')
else:
    detector = SHM('apapane')


# Create plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_ylim(-1e-1, 1e-1)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Value')

# Lists to store data points
times = []          # Timestamps (in seconds)
ratio_data = []     # (tri1_bright1 - tri1_bright2) / (tri1_bright1 + tri1_bright2)
bright1_data = []   # tri1_bright1 values
bright2_data = []   # tri1_bright2 values

# Create line objects for each series
line_ratio, = ax.plot([], [], 'r-', label='Ratio')
line_bright1, = ax.plot([], [], 'b-', label='tri1_bright1')
line_bright2, = ax.plot([], [], 'g-', label='tri1_bright2')

# Set visibility based on toggles
line_ratio.set_visible(SHOW_RATIO)
line_bright1.set_visible(SHOW_BRIGHT1)
line_bright2.set_visible(SHOW_BRIGHT2)

# Add legend only for visible lines
lines_for_legend = []
if SHOW_RATIO:
    lines_for_legend.append(line_ratio)
if SHOW_BRIGHT1:
    lines_for_legend.append(line_bright1)
if SHOW_BRIGHT2:
    lines_for_legend.append(line_bright2)
ax.legend(handles=lines_for_legend)

# Record the starting time
start_time = time.time()

def init():
    # Set a fixed x-axis range initially
    ax.set_xlim(0, window_duration)
    if SHOW_RATIO:
        line_ratio.set_data([], [])
    if SHOW_BRIGHT1:
        line_bright1.set_data([], [])
    if SHOW_BRIGHT2:
        line_bright2.set_data([], [])
    return (line_ratio, line_bright1, line_bright2)

def update(_):
    # Compute elapsed time in seconds
    t = time.time() - start_time

    # Retrieve the current frame and compute region sums
    frame = detector.get_data()
    tri1_bright1_val = np.sum(frame[top1:bottom1, left1:right1])
    tri1_bright2_val = np.sum(frame[top2:bottom2, left2:right2])
    summed = tri1_bright1_val + tri1_bright2_val
    ratio = (tri1_bright1_val - tri1_bright2_val) / summed if summed != 0 else 0

    # Append new data points
    times.append(t)
    ratio_data.append(ratio)
    bright1_data.append(tri1_bright1_val)
    bright2_data.append(tri1_bright2_val)

    # Remove data points older than the current window_duration
    while times and times[0] < t - window_duration:
        times.pop(0)
        ratio_data.pop(0)
        bright1_data.pop(0)
        bright2_data.pop(0)

    # Update x-axis to show a fixed window_duration seconds
    if t < window_duration:
        ax.set_xlim(0, window_duration)
    else:
        ax.set_xlim(t - window_duration, t)

    # Update the data for each line based on toggles
    if SHOW_RATIO:
        line_ratio.set_data(times, ratio_data)
    if SHOW_BRIGHT1:
        line_bright1.set_data(times, bright1_data)
    if SHOW_BRIGHT2:
        line_bright2.set_data(times, bright2_data)

    return (line_ratio, line_bright1, line_bright2)

ani = FuncAnimation(fig, update, init_func=init, interval=interval_ms, blit=False)
plt.show()
