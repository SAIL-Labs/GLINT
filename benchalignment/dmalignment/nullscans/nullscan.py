"""
Scans DM segment piston (per baseline, or per single segment) to measure the narrowband null depth
at each aperture pair, recording APAPANE spectral photometry through the
scan and saving movie/scan/OPD data as FITS files.

Scan inputs (piston range, spectral peaks, tip/tilt hold positions, ...)
live under the "config" key of scanparameters.json and are never modified
by this script -- edit them by hand before a run. Bookkeeping the script
itself needs to track between runs (today's date, the scan iteration
number) lives under "state" and is read/written automatically. Keeping
these separate means editing your scan parameters can never be silently
overwritten by the script's own bookkeeping.

All bench-alignment scans (xy, pitch/yaw, null) share one dark frame at
DARK_FILEPATH, rather than each scan type keeping its own dark.
"""
import sys

from hardware_control.dmcontrol import shmDMcontrol
from pyMilk.interfacing.shm import SHM
import numpy as np
from astropy.io import fits
import time
import matplotlib.pyplot as plt
import os
import json
import tqdm
from datetime import datetime, timezone
import glint_paths

PARAM_FILE = str(glint_paths.CODE_ROOT / 'benchalignment' / 'dmalignment' / 'nullscans' / 'scanparameters.json')
DARK_FILEPATH = str(glint_paths.DATA_ROOT / 'benchalignment' / 'spectraldark.fits')

# Segments not being actively scanned are tilted away to this tip/tilt 
TIP, TILT = 5.0, 0.0

# The three GLINT apertures, and the baselines (null channels) they form, in
# the fixed spectral order used by nullpeaks/boundingvals.
APERTURES = ["11", "20", "31"]
BASELINES = [["11", "31"], ["11", "20"], ["20", "31"]]

# Keys required in scanparameters.json under "config". Checked up front so a
# typo or missing field fails fast 
REQUIRED_CONFIG_KEYS = [
    'nframes', 'numavg', 'step_size', 'scan_range', 'start_pos',
    'nullpeaks', 'tips', 'tilts', 'boundingvals', 'box_halfwidth',
]


# ---------------------------------------------------------------------------
# Parameter file I/O
# ---------------------------------------------------------------------------

def load_params() -> dict:
    """Load scanparameters.json"""
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
    Connect to the APAPANE camera and the DM.

    Returns
    -------
    apapane : SHM
        Shared-memory handle for the APAPANE camera.
    dm : shmDMcontrol.DM
        DM object.
    """
    apapane = SHM('apapane')
    dm = shmDMcontrol.DM()
    return apapane, dm


def getdata(apapane: SHM, dark: np.ndarray, nframes: int = 1) -> np.ndarray:
    """
    Grab frame(s) from APAPANE and dark-subtract.
    Future versions of this code will instead formally spectrally extract the data with wavelength bins.

    Parameters
    ----------
    apapane : SHM
        Shared-memory handle for the APAPANE camera.
    dark : np.ndarray
        Dark frame to subtract, same shape as the raw APAPANE frame.
    nframes : int, optional
        Number of frames to average over. Default is 1.

    Returns
    -------
    data : np.ndarray
        Dark-subtracted full frame.
    """
    frames = apapane.multi_recv_data(nframes)
    frames = np.array(frames, dtype=float)

    avg = np.mean(frames, axis=0)
    return avg - dark


def crop_to_box(data: np.ndarray, box: list) -> np.ndarray:
    """
    Crop a full frame to one spectral box.

    Parameters
    ----------
    data : np.ndarray
        Full (dark-subtracted) frame.
    box : list
        [top, bottom, left, right] pixel bounds to crop to (see getbox()).

    Returns
    -------
    cropped : np.ndarray
        Cropped data.
    """
    top, bottom, left, right = box
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid box dimensions: top={top}, bottom={bottom}, left={left}, right={right}")

    return data[top:bottom, left:right]


def getbox(baseline: list, nullpeaks: list, boundingvals: list, box_halfwidth: int) -> list:
    """
    Build the pixel bounding box for one baseline's null channel.

    Parameters
    ----------
    baseline : list
        The two segments in this baseline, e.g. ["11", "31"].
    nullpeaks : list
        [null1, null2, null3] pixel centres of each baseline's null
        channel, in the fixed order (11,31), (11,20), (20,31).
    boundingvals : list
        [[start, end], ...] pixel bounds along the dispersion axis, one
        pair per baseline, in the same order as nullpeaks.
    box_halfwidth : int
        Half-width, in pixels, of the box around the null peak.

    Returns
    -------
    box : list
        [top, bottom, left, right] pixel bounds of the box.
    """
    null1, null2, null3 = nullpeaks

    # Map this baseline to its null channel peak/bounds. Order is fixed by
    # the instrument's spectral layout, not by baseline argument order.
    if set(baseline) == {"11", "31"}:
        null, (val1, val2) = null1, boundingvals[0]
    elif set(baseline) == {"11", "20"}:
        null, (val1, val2) = null2, boundingvals[1]
    elif set(baseline) == {"20", "31"}:
        null, (val1, val2) = null3, boundingvals[2]
    else:
        raise ValueError(f"Unknown baseline {baseline} for nullpeaks/boundingvals mapping.")


    top, bottom = null - box_halfwidth, null + box_halfwidth
    left, right = val1, val2


    return [int(top), int(bottom), int(left), int(right)]


def tilt_all_except(dm, keep: list) -> None:
    """
    Tilt all segments away except those in the `keep` list.

    Parameters
    ----------
    dm : shmDMcontrol.DM
        DM object.
    keep : list of str or int
        Segments to leave untouched.
    """
    keep = set(str(k) for k in keep)
    for seg in range(37):
        if str(seg) not in keep:
            dm.set_segment(seg, 0, TIP, TILT)


def save_data(baseline: list, movie: np.ndarray, opd: np.ndarray, timestamps, savepath: str, avg: bool,
              fluxes: dict = None) -> None:
    """
    Save one baseline's (or one segment's) photometry movie, plus
    OPD/timestamp metadata, to a multi-extension FITS file.

    Parameters
    ----------
    baseline : list
        The segment(s) scanned, e.g. ["11", "31"] for a baseline scan or
        ["11"] for a single-segment scan.
    movie : np.ndarray
        Shape (n_frames, box_height, box_width) photometry cube.
    opd : np.ndarray
        OPD (um) at each frame.
    timestamps : np.ndarray or None
        Wall-clock timestamp at each frame, or None/mismatched-length to
        fill with NaN.
    savepath : str
        Directory to save the FITS file into.
    avg : bool
        True if `movie` is an average over multiple repeat scans (changes
        the output filename).
    fluxes : dict, optional
        {column_name: per-frame flux array} extra columns to add to the
        METADATA table (e.g. summed flux in each null channel box).
    """
    name = ':'.join(str(b) for b in baseline)

    primary_hdu = fits.PrimaryHDU()
    movie_hdu = fits.ImageHDU(movie, name='MOVIE')

    # Ensure timestamps length matches OPD length
    if timestamps is None or len(timestamps) != len(opd):
        timestamps = np.full(len(opd), np.nan, dtype=float)

    # Table HDU for OPD and Timestamps
    col_opd = fits.Column(name='OPD', array=opd, format='E') # 32-bit float
    col_time = fits.Column(name='TIMESTAMP', array=timestamps, format='D')  # 64-bit float
    cols = [col_opd, col_time]
    for colname, flux in (fluxes or {}).items():
        cols.append(fits.Column(name=colname, array=np.asarray(flux, dtype=float), format='D'))
    metadata_hdu = fits.BinTableHDU.from_columns(cols, name='METADATA')

    hdul = fits.HDUList([primary_hdu, movie_hdu, metadata_hdu])

    prefix = 'avgmovie' if avg else 'movie'
    filename = f'{savepath}/{prefix}_{name}.fits'
    hdul.writeto(filename, overwrite=True)
    print(f'Saved {filename}')


def plot_scan(baseline: list, scan: np.ndarray, opd: np.ndarray, iteration: int,
              savepath: str, save: bool, avg: bool) -> None:
    """Plot summed flux vs OPD for one baseline, optionally saving to PNG."""
    seg1, seg2 = baseline

    plt.plot(opd, scan)
    plt.xlabel('OPD (um)')
    plt.ylabel('Summed flux')
    plt.title(f'Piston scan for baseline {seg1}:{seg2}')
    plt.grid(True)

    if save:
        prefix = 'avg_segments' if avg else 'segments'
        plt.savefig(f'{savepath}/{prefix}_{seg1}:{seg2}_scan{iteration}.png')

    plt.clf()


def plot_normscan(baseline: list, scan: np.ndarray, opd: np.ndarray, iteration: int,
                   savepath: str, save: bool, avg: bool) -> None:
    """Plot flux (normalised to its own peak) vs OPD for one baseline, optionally saving to PNG."""
    seg1, seg2 = baseline
    normscan = scan / np.max(scan)

    plt.plot(opd, normscan)
    plt.xlabel('OPD (um)')
    plt.ylabel('Normalised summed flux')
    plt.title(f'Narrowband piston scan for aperture {seg1}')
    plt.grid(True)

    if save:
        prefix = 'normalised_avg_segments' if avg else 'normalised_segments'
        plt.savefig(f'{savepath}/{prefix}_{seg1}:{seg2}_scan{iteration}.png')

    plt.clf()


# ---------------------------------------------------------------------------
# Scan logic
# ---------------------------------------------------------------------------

def nullscan(apapane: SHM, dm, baseline: list, tip: list, tilt: list, pistpositions: np.ndarray,
             box: list, dark: np.ndarray, savepath: str, nframes: int = 1,
             scan_one_segment: bool = True, tilt_unused: bool = True,
             save_timestamps: bool = False, save_full_frame: bool = False):
    """
    Piston-scan one baseline's segment(s) recording
    the summed flux in its null-channel spectral box at every position.

    If `scan_one_segment` is False, this runs two back-to-back scans: first
    segment 1 pistons from minimum (`pistpositions[0]`) to maximum (`pistpositions[-1]`) while
    segment 2 holds at maximum (`pistpositions[-1]`), then segment 2 pistons back down minimum
    while segment 1 holds -- stitched into one OPD-vs-flux curve.

    Parameters
    ----------
    apapane : SHM
        Shared-memory handle for the APAPANE camera.
    dm : shmDMcontrol.DM
        DM object.
    baseline : list
        The two segments to scan, e.g. ["11", "31"].
    tip, tilt : list
        [tip_seg1, tip_seg2] / [tilt_seg1, tilt_seg2] hold values for each
        segment while its piston is scanned.
    pistpositions : array-like
        Piston positions to scan through, min to max.
    box : list
        [top, bottom, left, right] spectral box to sum flux over (see
        getbox()).
    dark : np.ndarray
        Dark frame to subtract from every exposure.
    savepath : str
        Directory to save the per-scan movie FITS file into.
    nframes : int, optional
        Frames to average per exposure. Default 1.
    scan_one_segment : bool, optional
        If True, only segment 1 is scanned (segment 2 stays parked).
        Default True.
    tilt_unused : bool, optional
        If True, park every segment not involved in this baseline before
        scanning. Default True.
    save_timestamps : bool, optional
        If True, record a wall-clock timestamp per frame. Default False.
    save_full_frame : bool, optional
        If True, `movie` stores the full APAPANE frame at each position
        instead of the cropped spectral box. The scanned flux is still
        summed over the box either way. Default False.

    Returns
    -------
    movie : np.ndarray
        Shape (n_frames, box_height, box_width) photometry cube, or
        (n_frames, frame_height, frame_width) if `save_full_frame` is True.
    scan : np.ndarray
        Summed flux at each frame.
    opd : np.ndarray
        OPD (um) at each frame, accounting for the double pass off the DM.
    timestamps : np.ndarray
        Wall-clock timestamp per frame (NaN-filled if not requested).
    """
    seg1, seg2 = int(baseline[0]), int(baseline[1])
    seg1 = 11
    tip_seg1, tip_seg2 = float(tip[0]), float(tip[1])
    tilt_seg1, tilt_seg2 = float(tilt[0]), float(tilt[1])
    pistpositions = np.asarray(pistpositions, dtype=float)
    n = len(pistpositions)

    if tilt_unused:
        # keep = [seg1] if scan_one_segment else [11, 20, 31]
        keep = [11, 20, 31]
        tilt_all_except(dm, keep=keep)

    print(f'tip_seg1 = {tip_seg1}, tilt_seg1 = {tilt_seg1}')
    print(f'tip_seg2 = {tip_seg2}, tilt_seg2 = {tilt_seg2}')

    nullscan_seg1 = np.zeros(n, dtype=float)
    nullscan_seg2 = None if scan_one_segment else np.zeros(n - 1, dtype=float)

    # Scan 1: seg1 sweeps min -> max while seg2 holds at max.
    # Scan 2: seg2 sweeps max -> min (skipping the shared endpoint) while
    # seg1 holds at max.
    pos_seg1_scan1 = pistpositions
    pos_seg2_scan1 = pistpositions[-1]
    pos_seg2_scan2 = np.flip(pistpositions)[1:]

    dm.set_segment(seg1, pos_seg1_scan1[0], tip_seg1, tilt_seg1)
    time.sleep(0.01)
    # if not scan_one_segment:
    #     dm.set_segment(seg2, pos_seg2_scan1, tip_seg2, tilt_seg2)
    #     time.sleep(0.01)

    print(f'Scanning baseline {seg1}-{seg2}')
    height, width = (dark.shape if save_full_frame
                      else (box[1] - box[0], box[3] - box[2]))
    movie_len = n if scan_one_segment else (2 * n - 1)
    movie = np.zeros((movie_len, height, width), dtype=float)
    timestamps = [] if save_timestamps else None

    pbar = tqdm.tqdm(desc="Null scan", total=movie_len)

    for posnum in range(movie_len):
        full_frame = getdata(apapane, dark, nframes)
        boxed_frame = crop_to_box(full_frame, box)
        movie[posnum] = full_frame if save_full_frame else boxed_frame

        if posnum < n:
            # Scan 1: move seg1.
            dm.set_segment(seg1, pos_seg1_scan1[posnum], tip_seg1, tilt_seg1)
            time.sleep(0.001)
            nullscan_seg1[posnum] = np.sum(boxed_frame)
        elif not scan_one_segment:
            # Scan 2: move seg2.
            idx = posnum - n
            dm.set_segment(seg2, pos_seg2_scan2[idx], tip_seg2, tilt_seg2)
            time.sleep(0.001)
            nullscan_seg2[idx] = np.sum(boxed_frame)

        if save_timestamps:
            timestamps.append(time.time())

        pbar.update()

    pbar.close()

    scan = nullscan_seg1 if scan_one_segment else np.concatenate([nullscan_seg1, nullscan_seg2])

    # OPD stitching: scan 1's OPD is centred so its last point reads 0. Scan 2's 
    # OPD continues from there as seg2 moves away from that same shared position.
    opd1 = pistpositions - pistpositions[-1]
    opd2 = np.abs(np.flip(pistpositions)[1:] - np.flip(pistpositions)[0])
    opd = opd1 if scan_one_segment else np.concatenate([opd1, opd2])
    opd = 2.0 * opd  # Account for the double pass (reflection off the DM).

    ts = np.asarray(timestamps) if save_timestamps else np.full(movie_len, np.nan, dtype=float)
    save_data(baseline, movie, opd, ts, savepath, avg=False)

    return movie, scan, opd, ts


def plot_segmentscan(segment: str, scans: np.ndarray, labels: list, opd: np.ndarray, iteration: int,
                     savepath: str, save: bool, avg: bool, normalise: bool = False) -> None:
    """
    Plot flux vs OPD in every null channel box for a single-segment piston
    scan, optionally normalising each curve to its own peak and saving to PNG.
    """
    for k, label in enumerate(labels):
        curve = scans[:, k]
        if normalise:
            curve = curve / np.max(curve)
        plt.plot(opd, curve, label=f'null {label}')

    plt.xlabel('OPD (um)')
    plt.ylabel('Normalised summed flux' if normalise else 'Summed flux')
    plt.title(f'Piston scan of segment {segment}')
    plt.legend()
    plt.grid(True)

    if save:
        prefix = 'segment_avg' if avg else 'segment'
        if normalise:
            prefix = 'normalised_' + prefix
        plt.savefig(f'{savepath}/{prefix}_{segment}_scan{iteration}.png')

    plt.clf()


def set_reference(dm, tips: dict, tilts: dict, segments: list = APERTURES) -> None:
    """
    Put each GLINT aperture at zero piston and its optimal tip/tilt.

    Parameters
    ----------
    dm : shmDMcontrol.DM
        DM object.
    tips, tilts : dict
        {segment: value} optimal tip/tilt for each aperture (config 'tips'/'tilts').
    segments : list of str, optional
        Apertures to reset. Default all of APERTURES.
    """
    for seg in segments:
        dm.set_segment(int(seg), 0.0, float(tips[seg]), float(tilts[seg]))
    time.sleep(0.01)


def segmentscan(apapane: SHM, dm, segment: str, tips: dict, tilts: dict, pistpositions: np.ndarray,
                boxes: list, dark: np.ndarray, savepath: str, nframes: int = 1,
                tilt_unused: bool = True, save_timestamps: bool = False):
    """
    Piston-scan a single segment while the other GLINT apertures sit at
    zero piston and optimal tip/tilt, recording the summed flux in every
    null channel box at each position.

    All non-GLINT segments are tilted away (if `tilt_unused`). The scanned
    segment holds its own optimal tip/tilt throughout and is returned to
    zero piston at the end.

    Parameters
    ----------
    apapane : SHM
        Shared-memory handle for the APAPANE camera.
    dm : shmDMcontrol.DM
        DM object.
    segment : str
        Segment to scan, e.g. "11".
    tips, tilts : dict
        {segment: value} optimal tip/tilt for each aperture.
    pistpositions : array-like
        Piston positions to scan through, min to max.
    boxes : list
        [top, bottom, left, right] boxes to sum flux over, one per null
        channel (see getbox()).
    dark : np.ndarray
        Dark frame to subtract from every exposure.
    savepath : str
        Directory to save the per-scan movie FITS file into.
    nframes : int, optional
        Frames to average per exposure. Default 1.
    tilt_unused : bool, optional
        If True, tilt away every segment that is not a GLINT aperture.
        Default True.
    save_timestamps : bool, optional
        If True, record a wall-clock timestamp per frame. Default False.

    Returns
    -------
    movie : np.ndarray
        Shape (n_frames, frame_height, frame_width) full-frame cube (always
        full frame, since flux is recorded in several boxes).
    scans : np.ndarray
        Shape (n_frames, n_boxes) summed flux in each box at each frame.
    opd : np.ndarray
        OPD (um) of the scanned segment relative to the zero-piston
        apertures, accounting for the double pass off the DM.
    timestamps : np.ndarray
        Wall-clock timestamp per frame (NaN-filled if not requested).
    """
    seg = int(segment)
    tip_seg, tilt_seg = float(tips[segment]), float(tilts[segment])
    pistpositions = np.asarray(pistpositions, dtype=float)
    n = len(pistpositions)

    if tilt_unused:
        tilt_all_except(dm, keep=APERTURES)

    # Every aperture (including the one about to be scanned) starts at zero
    # piston and its optimal tip/tilt.
    set_reference(dm, tips, tilts)

    print(f'Scanning segment {segment} (tip = {tip_seg}, tilt = {tilt_seg}); '
          f'others at zero piston: {[s for s in APERTURES if s != segment]}')

    movie = np.zeros((n, *dark.shape), dtype=float)
    scans = np.zeros((n, len(boxes)), dtype=float)
    timestamps = [] if save_timestamps else None

    for posnum in tqdm.tqdm(range(n), desc=f"Segment {segment} scan"):
        dm.set_segment(seg, pistpositions[posnum], tip_seg, tilt_seg)
        time.sleep(0.001)

        full_frame = getdata(apapane, dark, nframes)
        movie[posnum] = full_frame
        for k, box in enumerate(boxes):
            scans[posnum, k] = np.sum(crop_to_box(full_frame, box))

        if save_timestamps:
            timestamps.append(time.time())

    # Return the scanned segment to zero piston.
    set_reference(dm, tips, tilts, segments=[segment])

    opd = 2.0 * pistpositions  # Double pass; reference apertures sit at 0 piston.

    ts = np.asarray(timestamps) if save_timestamps else np.full(n, np.nan, dtype=float)
    return movie, scans, opd, ts


def null_columns(scans: np.ndarray) -> dict:
    """Map an (n_frames, 3) segment-scan flux array to named METADATA columns."""
    return {f"NULL_{a}:{b}": scans[:, k] for k, (a, b) in enumerate(BASELINES)}


if __name__ == "__main__":

    # 'baseline' scans the baselines selected by b1/b2/b3 in scanparameters.json.
    # 'segment' piston-scans each segment in segments_to_scan in turn, with the
    # other GLINT apertures at zero piston / optimal tip-tilt.
    scan_mode = 'segment'
    segments_to_scan = ["11", "20", "31"]

    scan_one_segment = True
    tilt_unused = True
    save_timestamps = False
    save_full_frame = True

    params = load_params()
    validate_config(params['config'])
    config, state = params['config'], params['state']

    # Roll over to a fresh iteration count whenever the UTC date changes.
    now = datetime.now(timezone.utc)
    year = now.year
    date = now.strftime("%m-%d")

    # Reset iteration number if the date changed
    if state.get('date') != date or state.get('year') != year:
        state['iteration'] = 1
        
    state['date'] = date
    state['year'] = year
    save_params(params)

    # Build base directory up to the date folder
    base_dir = str(glint_paths.data_dir('benchalignment', 'dmalignment', 'nullscans', 'scanoutputs', year, date))

    # May bump state['iteration'] if today's folder already has data in it.
    savepath = checksavepath(base_dir, params)

    # Reload in case checksavepath() updated the iteration on disk.
    params = load_params()
    config, state = params['config'], params['state']
    iteration = state['iteration']

    nframes = config['nframes']
    numavg = config['numavg']
    step_size = np.array(config['step_size'])
    scan_range = np.array(config['scan_range'])
    start_pos = np.array(config['start_pos'])
    nullpeaks = config['nullpeaks']
    tips = config['tips']
    tilts = config['tilts']
    boundingvals = config['boundingvals']
    box_halfwidth = config['box_halfwidth']
    b1 = config['b1']
    b2 = config['b2']
    b3 = config['b3']

    apapane, dm = open_devices()
    dark = getdark(DARK_FILEPATH)

    nsteps = np.ceil(scan_range / step_size).astype(int) + 1
    pistpositions = np.linspace(start_pos, start_pos + scan_range, nsteps)

    if scan_mode == 'segment':
        # Record flux in all three null channels; the one not involving the
        # scanned segment acts as a stability reference.
        boxes = [getbox(b, nullpeaks, boundingvals, box_halfwidth) for b in BASELINES]
        labels = [':'.join(b) for b in BASELINES]

        for segment in segments_to_scan:
            if segment not in APERTURES:
                raise ValueError(f"Segment {segment} is not one of {APERTURES}.")

            scan_list, movie_list = [], []
            for _ in range(numavg):
                movie, scans, opd, ts = segmentscan(
                    apapane, dm, segment, tips, tilts, pistpositions, boxes, dark, savepath,
                    nframes=nframes, tilt_unused=tilt_unused, save_timestamps=save_timestamps,
                )
                save_data([segment], movie, opd, ts, savepath, avg=False, fluxes=null_columns(scans))
                plot_segmentscan(segment, scans, labels, opd, iteration, savepath, save=True, avg=False)
                scan_list.append(scans)
                movie_list.append(movie)

            avgscans = np.mean(np.stack(scan_list, axis=0), axis=0)
            avgmovie = np.mean(np.stack(movie_list, axis=0), axis=0)
            del movie_list

            plot_segmentscan(segment, avgscans, labels, opd, iteration, savepath, save=True, avg=True)
            plot_segmentscan(segment, avgscans, labels, opd, iteration, savepath, save=True, avg=True,
                             normalise=True)
            save_data([segment], avgmovie, opd, ts, savepath, avg=True, fluxes=null_columns(avgscans))

        sys.exit(0)

    baseline_map = {
        'b1': ["11", "31"],
        'b2': ["11", "20"],
        'b3': ["20", "31"],
    }

    to_scan = []
    if b1: to_scan.append(baseline_map['b1'])
    if b2: to_scan.append(baseline_map['b2'])
    if b3: to_scan.append(baseline_map['b3'])

    if not to_scan:
        print("No baselines selected (all b1/b2/b3 False). Exiting.")
        sys.exit(0)

    for baseline in to_scan:
        seg1, seg2 = baseline
        tip = [tips[seg1], tips[seg2]]
        tilt = [tilts[seg1], tilts[seg2]]
        box = getbox(baseline, nullpeaks, boundingvals, box_halfwidth)

        avgscan_list, avgmovie_list = [], []
        last_opd, last_ts = None, None

        for _ in range(numavg):
            movie, scan, opd, ts = nullscan(
                apapane, dm, baseline, tip, tilt, pistpositions, box, dark, savepath,
                nframes=nframes, scan_one_segment=scan_one_segment,
                tilt_unused=tilt_unused, save_timestamps=save_timestamps,
                save_full_frame=save_full_frame,
            )
            plot_scan(baseline, scan, opd, iteration, savepath, save=True, avg=False)
            avgscan_list.append(scan)
            avgmovie_list.append(movie)
            last_opd, last_ts = opd, ts

        avgscan = np.mean(np.stack(avgscan_list, axis=0), axis=0)
        avgmovie = np.mean(np.stack(avgmovie_list, axis=0), axis=0)

        plot_scan(baseline, avgscan, last_opd, iteration, savepath, save=True, avg=True)
        plot_normscan(baseline, avgscan, last_opd, iteration, savepath, save=True, avg=True)
        save_data(baseline, avgmovie, last_opd, last_ts, savepath, avg=True)
