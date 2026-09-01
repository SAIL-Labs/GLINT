"""
Raster-scans the mount over two axes (default: x and y), recording APAPANE
spectral photometry at each grid point, then saves the results as a FITS
cube and hands off to read_xyscan.py for fitting.

Scan inputs (which axes, step size, range, spectral peaks, ...) live under
the "config" key of scanparameters.json and are never modified by this
script -- edit them by hand before a run. Bookkeeping the script itself
needs to track between runs (today's date, the scan iteration number, the
last fit result) lives under "state" and is read/written automatically.
Keeping these separate means editing your scan parameters can never be
silently overwritten by the script's own bookkeeping.
"""
from hardware_control.mountcontrol.chipMountControl import Mount
from pyMilk.interfacing.shm import SHM
import numpy as np
from astropy.io import fits
import json
import os
import subprocess
from datetime import datetime, timezone
import tqdm
import glint_paths

# Mount axis name -> axis number, as expected by chipMountControl.Mount.
AXES = {'pitch': 1, 'roll': 2, 'yaw': 3, 'x': 4, 'z': 5, 'y': 6}

PARAM_FILE = str(glint_paths.CODE_ROOT / 'benchalignment' / 'chipmountalignment' / 'xyscans' / 'scanparameters.json')
DARK_FILEPATH = str(glint_paths.CALIBRATION_ROOT / 'dark.fits')
ANALYSIS_SCRIPT = str(glint_paths.CODE_ROOT / 'benchalignment' / 'chipmountalignment' / 'xyscans' / 'read_xyscan.py')

# Keys required in scanparameters.json under "config". Checked up front so a
# typo or missing field fails fast with a clear message instead of a bare
# KeyError partway through a scan.
REQUIRED_CONFIG_KEYS = [
    'select_axes', 'step_size', 'scan_range', 'start_pos',
    'peaks', 'boundingvals', 'box_halfwidth',
]


# ---------------------------------------------------------------------------
# Parameter file I/O
# ---------------------------------------------------------------------------

def load_params() -> dict:
    """Load scanparameters.json (both "config" and "state" sections)."""
    with open(PARAM_FILE, 'r') as f:
        return json.load(f)


def save_params(params: dict) -> None:
    """Write scanparameters.json back out."""
    with open(PARAM_FILE, 'w') as f:
        json.dump(params, f, indent=2)


def validate_config(config: dict) -> None:
    """Raise a clear error if scanparameters.json is missing required config fields."""
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise KeyError(
            f"scanparameters.json is missing required 'config' field(s): {missing}"
        )


def checksavepath(base_dir: str, params: dict) -> str:
    """
    Find (or create) the directory to save this scan's data into.

    Save directories are named f'{base_dir}/scan{iteration}'. If the
    directory for the current iteration already exists and has files in it,
    the iteration number is bumped (and persisted to scanparameters.json)
    until an empty or nonexistent directory is found.

    Parameters
    ----------
    base_dir : str
        Directory containing the per-date scanN folders.
    params : dict
        Full contents of scanparameters.json, as returned by load_params().
        params['state']['iteration'] is updated and saved as needed.

    Returns
    -------
    savepath : str
        Directory to save this scan's FITS files into.
    """
    state = params['state']

    while True:
        savepath = os.path.join(base_dir, f"scan{state['iteration']}")

        if os.path.isdir(savepath):
            if os.listdir(savepath):
                print(f"{savepath} exists and is not empty. Incrementing iteration.")
                state['iteration'] += 1
                save_params(params)
                continue

            print(f"{savepath} exists but is empty. Using it.")
            save_params(params)
            return savepath

        os.makedirs(savepath)
        print(f"{savepath} created.")
        save_params(params)
        return savepath


# ---------------------------------------------------------------------------
# Hardware / data helpers
# ---------------------------------------------------------------------------

def getdark(dark_filepath: str) -> np.ndarray:
    """
    Load a dark frame and average it down to a single 2D frame.

    Parameters
    ----------
    dark_filepath : str
        Filepath to the dark frame FITS file.

    Returns
    -------
    dark : np.ndarray
        Mean dark frame, as float (to avoid overflow when subtracted from
        later frames).
    """
    with fits.open(dark_filepath) as hdul:
        dark = hdul[0].data
        dark = np.array(np.mean(dark, axis=0), dtype=float)

    return dark


def open_devices():
    """
    Connect to the APAPANE camera and the mount controller.

    Returns
    -------
    apapane : SHM
        Shared-memory handle for the APAPANE camera.
    mount : chipMountControl.Mount
        Mount controller.
    """
    apapane = SHM('apapane')
    mount = Mount('/dev/serial/by-id/usb-SURUGA_SEIKI_SURUGA_SEIKI_DS102-if00-port0', 38400)

    return apapane, mount


def getbox(peak: int, boundingvals: list, box_halfwidth: int) -> list:
    """
    Build the pixel bounding box for one spectral trace.

    Parameters
    ----------
    peak : int
        Pixel coordinate of the trace centre, along the axis perpendicular
        to dispersion.
    boundingvals : list
        [start, end] pixel coordinates along the dispersion axis, i.e. how
        much of the spectrum (in the dispersion direction) to keep.
    box_halfwidth : int
        Half-width, in pixels, of the box around `peak`.

    Returns
    -------
    box : list
        [top, bottom, left, right] pixel bounds of the box.
    """
    val1, val2 = boundingvals

    top, bottom = peak - box_halfwidth, peak + box_halfwidth
    left, right = val1, val2


    return [top, bottom, left, right]


def getdata(apapane: SHM, box: list, dark: np.ndarray, nframes: int = 1) -> np.ndarray:
    """
    Grab frame(s) from APAPANE, dark-subtract, and crop to one spectral box.

    Parameters
    ----------
    apapane : SHM
        Shared-memory handle for the APAPANE camera.
    box : list
        [top, bottom, left, right] pixel bounds to crop to (see getbox()).
    dark : np.ndarray
        Dark frame to subtract, same shape as the raw APAPANE frame.
    nframes : int, optional
        Number of frames to average over. Default is 1.

    Returns
    -------
    data : np.ndarray
        Dark-subtracted, cropped data.
    """
    top, bottom, left, right = box

    frames = apapane.multi_recv_data(nframes)
    frames = np.array(frames, dtype=float)

    avg = np.mean(frames, axis=0)
    data = avg - dark
    data = data[top:bottom, left:right]

    return data


def savedata(path: str, photometry_scans, mount_positions: np.ndarray,
             select_axes: list, iteration: int) -> None:
    """
    Save each spectral box's photometry cube (plus the mount positions
    scanned) to its own multi-extension FITS file.

    Parameters
    ----------
    path : str
        Directory to save the FITS files into.
    photometry_scans : np.ndarray
        Shape (n_boxes, n_steps, box_height, box_width) -- one photometry
        cube per spectral box.
    mount_positions : np.ndarray
        Shape (2, n_steps_per_axis), the two axes' scanned positions.
    select_axes : list
        The two axis names scanned, e.g. ['x', 'y'].
    iteration : int
        Scan iteration number, used in the output filename.
    """
    axlabel1, axlabel2 = select_axes

    for i, photometry in enumerate(photometry_scans):
        primary_hdu = fits.PrimaryHDU()
        phot_hdu = fits.ImageHDU(data=photometry, name='PHOTOMETRY')
        mount_hdu = fits.ImageHDU(data=mount_positions, name='MOUNT_POS')

        hdul = fits.HDUList([primary_hdu, phot_hdu, mount_hdu])

        filename = f'{path}/{axlabel1}{axlabel2}scan_spectra{i + 1}_{iteration}.fits'
        hdul.writeto(filename, overwrite=True)
        print(f'Saved {filename}')


# ---------------------------------------------------------------------------
# Scan logic
# ---------------------------------------------------------------------------

def move_axis_and_wait(mount: Mount, axis: int, target_pos: int) -> None:
    """Command an axis to a position and block until it arrives."""
    mount.set_pos(axis, target_pos)
    while mount.in_motion(axis) or mount.get_pos(axis) != target_pos:
        pass


def rasterscan(apapane: SHM, mount: Mount, dark: np.ndarray, select_axes: list,
                mountpositions: np.ndarray, boxes: list, box_halfwidth: int) -> np.ndarray:
    """
    Raster-scan the mount over two axes, recording photometry at every grid
    point. Axis 1 is the slow (outer) axis, axis 2 is the fast (inner) axis
    -- for each axis-1 position, axis 2 sweeps its full range before axis 1
    steps again, and axis 2 is returned to its start position each time.

    Parameters
    ----------
    apapane : SHM
        Shared-memory handle for the APAPANE camera.
    mount : chipMountControl.Mount
        Mount controller.
    dark : np.ndarray
        Dark frame to subtract from every exposure.
    select_axes : list
        The two axis names to scan, e.g. ['x', 'y'].
    mountpositions : np.ndarray
        Shape (2, n_steps_per_axis): positions to visit for each axis.
    boxes : list
        One [top, bottom, left, right] pixel box per spectral trace (see
        getbox()).
    box_halfwidth : int
        Half-width, in pixels, of each spectral box.

    Returns
    -------
    photometry_scans : np.ndarray
        Shape (n_boxes, n_steps, box_height, box_width).
    """
    top, bottom, left, right = boxes[0]
    boxheight, boxwidth = bottom - top, right - left

    axlabel1, axlabel2 = select_axes
    ax1, ax2 = AXES[axlabel1], AXES[axlabel2]
    startpos1, startpos2 = mountpositions[0][0], mountpositions[1][0]
    nsteps = [len(mountpositions[0]), len(mountpositions[1])]
    totalsteps = nsteps[0] * nsteps[1]

    photometry_scans = np.zeros((len(boxes), totalsteps, boxheight, boxwidth))

    # Move both axes to their scan start positions before beginning.
    print(f'Axis {axlabel1} position before moving: {mount.get_pos(ax1)}')
    move_axis_and_wait(mount, ax1, startpos1)
    print(f'Axis {axlabel1} scan start position: {mount.get_pos(ax1)}')

    print(f'Axis {axlabel2} position before moving: {mount.get_pos(ax2)}')
    move_axis_and_wait(mount, ax2, startpos2)
    print(f'Axis {axlabel2} scan start position: {mount.get_pos(ax2)}')

    pbar = tqdm.tqdm(total=totalsteps, desc=f"{axlabel1}-{axlabel2} scan")
    step = 0
    for i in range(nsteps[0]):
        for j in range(nsteps[1]):
            # Record photometry for every spectral box at this grid point.
            for box_idx, box in enumerate(boxes):
                photometry_scans[box_idx][step] = getdata(apapane, box, dark, nframes=1)

            move_axis_and_wait(mount, ax2, mountpositions[1, j])
            step += 1
            pbar.update(1)
            pbar.set_postfix({
                axlabel1: mount.get_pos(ax1),
                axlabel2: mount.get_pos(ax2),
            })

        # Step axis 1, then return axis 2 to its start before the next row.
        mount.set_pos(ax1, mountpositions[0, i])
        mount.set_pos(ax2, startpos2)
        while (mount.in_motion(ax1) or mount.get_pos(ax1) != mountpositions[0, i]
               or mount.in_motion(ax2) or mount.get_pos(ax2) != startpos2):
            pass

    pbar.close()
    return photometry_scans


if __name__ == "__main__":

    params = load_params()
    validate_config(params['config'])
    config, state = params['config'], params['state']

    # Roll over to a fresh iteration count whenever the UTC date changes.
    now = datetime.now(timezone.utc)
    year, date = now.year, now.strftime("%m-%d")
    if state.get('date') != date or state.get('year') != year:
        state['iteration'] = 1
    state['date'], state['year'] = date, year
    save_params(params)

    base_dir = str(glint_paths.data_dir('benchalignment', 'chipmountalignment', 'xyscans', 'scanoutput', year, date))

    # May bump state['iteration'] if today's folder already has data in it.
    savepath = checksavepath(base_dir, params)

    # Reload in case checksavepath() updated the iteration on disk.
    params = load_params()
    config, state = params['config'], params['state']
    iteration = state['iteration']

    select_axes = config['select_axes']
    step_size = np.array(config['step_size'])
    scan_range = np.array(config['scan_range'])
    start_pos = np.array(config['start_pos'])
    peaks = config['peaks']
    boundingvals = config['boundingvals']
    box_halfwidth = config['box_halfwidth']

    dark = getdark(DARK_FILEPATH)
    apapane, mount = open_devices()

    # One [top, bottom, left, right] box per spectral peak.
    boxes = [getbox(peak, boundingvals, box_halfwidth) for peak in peaks]

    # Number of steps per axis (+1 to include the final position), and the
    # actual positions to visit.
    nsteps = np.ceil(scan_range / step_size).astype(int) + 1
    mountpositions = np.array(
        [np.linspace(start_pos[i], start_pos[i] + scan_range[i], nsteps[i]) for i in range(2)],
        dtype=int,
    )

    photometry_scans = rasterscan(apapane, mount, dark, select_axes, mountpositions, boxes, box_halfwidth)

    savedata(savepath, photometry_scans, mountpositions, select_axes, iteration)

    print("\nRunning x/y analysis and updating parameter file...")
    try:
        subprocess.run(
            ["python", ANALYSIS_SCRIPT, "--update-param-file"],
            check=True,
        )
        print("X/y analysis complete and parameters updated.")
    except subprocess.CalledProcessError as e:
        print("Error running x/y analysis script:")
        print(e)
