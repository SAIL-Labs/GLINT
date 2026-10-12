import numpy as np
import subprocess
import findhotspot
from astropy.io import fits
from pyMilk.interfacing.shm import SHM
import time
import os
import matplotlib.pyplot as plt
import glint_paths

# ============================
# --- Response Matrix IO ----
# ============================
DEFAULT_RM_FILENAME = str(glint_paths.data_dir("benchalignment", "hotspotalignment") / "response_matrix.npy")

def save_RM(R, filename=DEFAULT_RM_FILENAME):
    np.save(filename, R)
    print(f"Response matrix saved to {filename}.")

def load_RM(filename=DEFAULT_RM_FILENAME):
    if os.path.exists(filename):
        print(f"Loaded response matrix from {filename}.")
        return np.load(filename)
    else:
        print(f"Response matrix file {filename} not found.")
        return None


# ============================
# --- Simple CLI Prompts -----
# ============================
def _prompt_yes_no(prompt: str, default: bool = False) -> bool:
    d = "y" if default else "n"
    while True:
        s = input(f"{prompt} [y/n] (default {d}): ").strip().lower()
        if s == "":
            return default
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False
        print("Please enter y or n.")

def _prompt_int(prompt: str, default: int, min_value: int = 1, max_value: int | None = None) -> int:
    while True:
        s = input(f"{prompt} (default {default}): ").strip()
        if s == "":
            return default
        try:
            v = int(s)
        except ValueError:
            print("Please enter an integer.")
            continue
        if v < min_value:
            print(f"Please enter a value >= {min_value}.")
            continue
        if max_value is not None and v > max_value:
            print(f"Please enter a value <= {max_value}.")
            continue
        return v


# ============================
# --- Main Loop -------------
# ============================
def mainloop(update_RM=False, RM_filename=DEFAULT_RM_FILENAME):
    """
    Aligns the PSF to a goal position using a feedback loop based on a response matrix.
    """
    # --- User prompts at start ---
    plot_updates = _prompt_yes_no("Plot PSF position each step?", default=False)
    nframes_avg = _prompt_int("How many frames to average per PSF measurement?", default=1, min_value=1, max_value=5000)
    max_steps = _prompt_int("Max alignment steps to allow?", default=10, min_value=1, max_value=10000)

    # Print current mirror positions before any alignment
    thetas_now = get_thetas()
    print(f"\nCurrent mirror positions (before alignment): u = {thetas_now[0]:.6f}, v = {thetas_now[1]:.6f}")

    # Load goal coordinates
    # goal_bright = fits.getdata(f'{glint_paths.DATA_ROOT}/benchalignment/psfpupilframes/20251021/psf_zerovolts.fits')
    # goal_dark = fits.getdata(f'{glint_paths.DATA_ROOT}/benchalignment/psfpupilframes/20251021/psf_dark.fits')
    goal_bright = fits.getdata(str(glint_paths.DATA_ROOT / 'benchalignment' / 'psfpupilframes' / '20260602' / 'ir_irisclosed_psf_zerodm.fits'))
    goal_dark = fits.getdata(str(glint_paths.DATA_ROOT / 'benchalignment' / 'psfpupilframes' / '20260602' / 'dark_psf.fits'))
    goal_pos = get_hotspot(goal_bright, goal_dark, nframes=nframes_avg, do_plot=plot_updates)
    print(f"\nGoal pos: {goal_pos}\n")

    # Step 1: Load or calculate RM
    if update_RM or not os.path.exists(RM_filename):
        print("Calculating new response matrix...")
        R = calculate_RM([0.01, 0.01], nframes=nframes_avg, do_plot=False)  # don't spam plots during RM
        save_RM(R, RM_filename)
    else:
        R = load_RM(RM_filename)

    # Step 2: Check invertibility
    print(f"Matrix R: {R}\n")
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        print("Warning: Response matrix is singular. Using pseudo-inverse.")
        R_inv = np.linalg.pinv(R)

    # Step 3: Setup loop parameters
    base_gain = 0.3
    min_gain = 0.01
    gain = base_gain
    tolerance = 2       # pixel tolerance
    sleep_time = 0.6

    last_offset = None
    initial_offset = None
    count = 0

    while count < max_steps:
        current_pos = get_hotspot(nframes=nframes_avg, do_plot=plot_updates)
        error = np.array(goal_pos) - np.array(current_pos)
        offset = np.linalg.norm(error)

        print(f"\nStep {count}: Current offset = {offset:.2f} pixels")

        if initial_offset is None:
            initial_offset = offset

        if offset < tolerance:
            print("Alignment achieved.")
            break

        # --- Adaptive gain control ---
        if last_offset is not None and last_offset > 0:
            ratio = offset / last_offset  # <1 = improved; >1 = worse
            if ratio > 1.10:
                gain *= 0.5
                print(f"Worse ({ratio:.2f}x), reducing gain -> {gain:.5f}")
            elif ratio < 0.98 and ratio > 0.5:
                pass
            elif ratio >= 0.98:
                gain *= 1.25
                print(f"Stalled ({ratio:.2f}x), increasing gain -> {gain:.5f}")
            gain = max(gain, min_gain)

        print(f"Using gain: {gain:.5f}")

        # Calculate mirror move
        thetas = get_thetas()
        print(f"Current mirror positions: u = {thetas[0]:.6f}, v = {thetas[1]:.6f}")

        delta_thetas = (R_inv @ error) * gain
        new_thetas = thetas + delta_thetas

        print(f"Moving mirrors to: u = {new_thetas[0]:.6f}, v = {new_thetas[1]:.6f}")
        movemirror(new_thetas)

        time.sleep(sleep_time)  # settle before measuring again

        last_offset = offset
        count += 1

    if count >= max_steps:
        print("Warning: Max steps reached without achieving alignment.")


def calculate_RM(perturbations, nframes: int = 50, do_plot: bool = False):
    """
    Build Jacobian R = [[dx/du, dx/dv],
                        [dy/du, dy/dv]]
    using small independent pokes on u and v around the current operating point.
    """
    perturbations = np.asarray(perturbations, dtype=float)
    assert perturbations.shape == (2,), "perturbations must be [du, dv]"

    x0, y0 = get_hotspot(nframes=nframes, do_plot=do_plot)
    thetas0 = get_thetas()

    R = np.zeros((2, 2), dtype=float)  # rows=(x,y), cols=(u,v)

    for i in range(2):
        thetas_p = thetas0.copy()
        thetas_p[i] += perturbations[i]

        movemirror(thetas_p)
        time.sleep(1.0)  # allow to settle

        xi, yi = get_hotspot(nframes=nframes, do_plot=do_plot)
        dpos = np.array([xi - x0, yi - y0], dtype=float) / perturbations[i]
        R[:, i] = dpos

    # return to nominal
    movemirror(thetas0)
    time.sleep(0.5)

    return R


def movemirror(newthetas):
    """
    Send commands to move the mirror actuators to new positions.
    """
    commands = [
        f"glint_steering2 u goto {newthetas[0]}",
        f"glint_steering2 v goto {newthetas[1]}"
    ]
    for command in commands:
        subprocess.run(command, shell=True)


def get_thetas():
    """
    Queries the current mirror positions for axes u and v.
    Returns ndarray([u, v]) as floats.
    """
    commands = [
        "glint_steering2 u status",
        "glint_steering2 v status"
    ]

    thetas = []
    for command in commands:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        try:
            thetas.append(float(result.stdout.split()[-1]))
        except ValueError:
            print("Warning: Could not parse theta from:", result.stdout)
            thetas.append(0.0)

    return np.array(thetas, dtype=float)


def get_hotspot(brightframe=None, dark=None, nframes: int = 50, do_plot: bool = True):
    """
    Gets the current hotspot (PSF) position from the detector frame.

    Parameters
    ----------
    brightframe : np.ndarray | None
        If provided, use this as the "bright" frame (already averaged).
    dark : np.ndarray | None
        Dark frame to subtract. If None, tries to load 'dark.fits'.
    nframes : int
        Number of frames to average if brightframe is None.
    do_plot : bool
        If True, plot the cropped image and centroid.

    Returns
    -------
    np.ndarray
        [x, y] hotspot coordinates (in cropped-image pixels).
    """
    start_x, start_y, end_x, end_y = 300, 225, 800, 725

    if brightframe is None:
        frames = [get_frame() for _ in range(nframes)]
        brightframe = np.mean(np.asarray(frames, dtype=float), axis=0)

    if dark is None:
        try:
            dark = fits.getdata(str(glint_paths.data_dir("benchalignment", "hotspotalignment") / "dark.fits")).astype(float)
        except Exception:
            print('save a dark')
            return

    img = brightframe.astype(float) - dark
    vmax = np.max(img)
    if vmax > 0:
        img = img / vmax

    img = img[start_y:end_y, start_x:end_x]

    coords = findhotspot.find_origin_com(img, threshold=0.8)

    if do_plot:
        plt.figure()
        plt.imshow(img, origin='lower')
        plt.plot(coords[0], coords[1], 'rx')
        plt.title('PSF centre: ({:.1f}, {:.1f})'.format(coords[0], coords[1]))
        plt.show()

    return np.array(coords, dtype=float)


def get_frame():
    """
    Returns the current frame from the shared memory.
    """
    return SHM('glintpg2').get_data()


if __name__ == '__main__':
    mainloop(update_RM=False)
