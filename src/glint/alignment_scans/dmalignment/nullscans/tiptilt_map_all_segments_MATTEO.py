"""
tiptilt_map_all_segments.py

Standalone diagnostic scan for parasitic coupling from every DM segment.

Test performed
--------------
1. Park all 37 DM segments at a common tilted-away state.
2. For each segment 0..36:
      - scan that segment in tip/tilt over a 2D grid
      - keep all other segments parked
      - record the flux in photometric channels 1, 2, and 3
3. Save one FITS binary table with one row per:
      segment, photometric channel, tip, tilt, average repeat

The output is similar in spirit to pistonscan_map_single_active.py, but the
independent variables are TIP and TILT instead of OPD. The table extension is
still named SCAN_DATA.

Default save location:
    OUTPUT_ROOT / YEAR / DATE / scanITERATION / tiptilt_map_table_ITERATION.fits

No scanparameters.json is read or written.
"""

import os
import time
import shutil
from datetime import datetime

import numpy as np
from astropy.io import fits
import tqdm

from hardware_control.dmcontrol import shmDMcontrol
from pyMilk.interfacing.shm import SHM
import glint_paths


# =============================================================================
# USER PARAMETERS - EDIT THIS SECTION BEFORE RUNNING
# =============================================================================

# Output bookkeeping. The FITS file will be written to:
#   OUTPUT_ROOT / YEAR / DATE / scanITERATION / tiptilt_map_table_ITERATION.fits
OUTPUT_ROOT = str(glint_paths.data_dir('alignment_scans', 'nullscans'))
YEAR = 2026
DATE = "05-21"          # e.g. "05-15"
ITERATION = 1

# Camera / dark settings
APAPANE_SHM_NAME = "apapane"
DARK_FILEPATH = str(glint_paths.DATA_ROOT / '_dataarchive' / 'darkMatteo.fits')
NFRAMES = 30             # camera frames averaged at each tip/tilt point
NUMAVG = 1              # repeat the full 37-segment scan this many times

# Global parking state for all non-scanned segments.
# This is the "all segments tilted away by some value" state.
# Format: (tip, tilt), in the same DM command units used by the GUI/scripts.
PARKED_TIPTILT = (0.0, 5.0)
PARKED_PISTON = 0.0

# Tip/tilt grid for the scanned segment.
# Inclusive grid from -5 to +5 in steps of 0.2.
TIP_START = -4.0
TIP_STOP = 4.0
TIP_STEP = 0.2
TILT_START = -4.0
TILT_STOP = 4.0
TILT_STEP = 0.2
SCANNED_SEGMENT_PISTON = 0.0

# Photometric extraction boxes. These are kept consistent with the working
# piston_map_single_active.py version.
# Original convention for ISCRED1=True:
#   box = [peak - halfwidth, peak + halfwidth, left, right]
# and getdata() slices image[top:bottom, left:right].
PHOTOMETRIC_PEAKS = {
    1: 243,
    2: 224,
    3: 205,
}
PHOTOMETRIC_BOUNDINGVALS = {
    1: (229, 267),
    2: (229, 267),
    3: (229, 267),
}
BOX_HALFWIDTH = 2
ISCRED1 = True
PHOTOMETRIC_CHANNELS = (1, 2, 3)

# Safety / timing
DM_SETTLE_SECONDS = 0.001
INITIAL_SETTLE_SECONDS = 0.01
ABORT_IF_OUTPUT_EXISTS = True


# =============================================================================
# Helpers
# =============================================================================

def make_grid_positions(start, stop, step):
    """Return inclusive grid positions from start to stop with requested step."""
    if step <= 0:
        raise ValueError("Grid step must be positive.")
    nsteps = int(np.floor((stop - start) / step + 0.5)) + 1
    positions = start + step * np.arange(nsteps, dtype=float)

    if not np.isclose(positions[-1], stop, atol=abs(step) * 1e-6):
        if positions[-1] < stop:
            positions = np.append(positions, stop)
        else:
            positions[-1] = stop
    return positions


def getbox(phot_chan):
    """Build [top, bottom, left, right] boxes using the original convention."""
    peak = PHOTOMETRIC_PEAKS[int(phot_chan)]
    val1, val2 = PHOTOMETRIC_BOUNDINGVALS[int(phot_chan)]
    if ISCRED1:
        top, bottom = peak - BOX_HALFWIDTH, peak + BOX_HALFWIDTH
        left, right = val1, val2
    else:
        left, right = peak - BOX_HALFWIDTH, peak + BOX_HALFWIDTH
        top, bottom = val1, val2
    return [int(top), int(bottom), int(left), int(right)]


def getdata(apapane, box, dark, nframes=1) -> np.ndarray:
    """Read APAPANE frames, average them, subtract dark, and crop a box."""
    top, bottom, left, right = box
    if right <= left or bottom <= top:
        raise ValueError(
            f"Invalid box dimensions: top={top}, bottom={bottom}, "
            f"left={left}, right={right}"
        )

    bright = apapane.multi_recv_data(nframes)
    bright = np.array(bright, dtype=float)
    avg = np.mean(bright, axis=0)
    data = avg - dark
    return data[int(top):int(bottom), int(left):int(right)]


def getdark(dark_filepath: str) -> np.ndarray:
    """Load and average the dark FITS cube/frame."""
    with fits.open(dark_filepath) as hdul:
        dark = hdul[0].data
        dark = np.array(np.mean(dark, axis=0), dtype=float)
    return dark


def open_devices():
    apapane = SHM(APAPANE_SHM_NAME)
    dm = shmDMcontrol.DM()
    return apapane, dm


def output_directory():
    return os.path.join(OUTPUT_ROOT, str(YEAR), str(DATE), f"scan{ITERATION}")


def prepare_savepath(savepath):
    if os.path.isdir(savepath) and os.listdir(savepath):
        msg = f"Output directory already exists and is not empty: {savepath}"
        if ABORT_IF_OUTPUT_EXISTS:
            raise FileExistsError(msg + "\nChange ITERATION or set ABORT_IF_OUTPUT_EXISTS=False.")
        backup = savepath + "_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.move(savepath, backup)
        print(f"Moved existing output directory to {backup}")
    os.makedirs(savepath, exist_ok=True)
    print(f"Using output directory: {savepath}")


def set_segment(dm, segment, piston, tip, tilt):
    dm.set_segment(int(segment), float(piston), float(tip), float(tilt))


def park_segment(dm, segment):
    tip, tilt = PARKED_TIPTILT
    set_segment(dm, segment, PARKED_PISTON, tip, tilt)


def park_all_segments(dm):
    for seg in range(37):
        park_segment(dm, seg)
    time.sleep(INITIAL_SETTLE_SECONDS)


def measure_photometric_fluxes(apapane, dark, nframes):
    """Return {phot_chan: summed_flux} for all requested photometric channels."""
    fluxes = {}
    for phot_chan in PHOTOMETRIC_CHANNELS:
        box = getbox(phot_chan)
        frame = getdata(apapane, box, dark, nframes)
        fluxes[int(phot_chan)] = float(np.sum(frame))
    return fluxes


def scan_tiptilt_for_segment(apapane, dm, segment, tips, tilts, dark, nframes):
    """
    Scan one segment over the tip/tilt grid.

    All other segments are parked at PARKED_TIPTILT. The scanned segment is set
    to SCANNED_SEGMENT_PISTON and the current grid tip/tilt.
    """
    rows = []

    park_all_segments(dm)
    total = len(tips) * len(tilts)
    pbar = tqdm.tqdm(desc=f"segment {segment} tip/tilt map", total=total, leave=False)

    for tip in tips:
        for tilt in tilts:
            set_segment(dm, segment, SCANNED_SEGMENT_PISTON, tip, tilt)
            time.sleep(DM_SETTLE_SECONDS)

            fluxes = measure_photometric_fluxes(apapane, dark, nframes)
            for phot_chan, flux in fluxes.items():
                rows.append((
                    int(segment),
                    int(phot_chan),
                    float(tip),
                    float(tilt),
                    float(flux),
                ))
            pbar.update()

    pbar.close()
    park_segment(dm, segment)
    return rows


def save_tiptilt_map_table(savefile, all_rows, header_info=None, comments=None):
    """Save a SCAN_DATA binary table for the tip/tilt heatmap scan."""
    all_rows = np.array(
        all_rows,
        dtype=[
            ('SEGMENT', 'i4'),
            ('PHOT_CHAN', 'i4'),
            ('TIP', 'f8'),
            ('TILT', 'f8'),
            ('FLUX', 'f8'),
            ('AVG_INDEX', 'i4'),
        ]
    )

    cols = fits.ColDefs([
        fits.Column(name='SEGMENT', format='J', array=all_rows['SEGMENT']),
        fits.Column(name='PHOT_CHAN', format='J', array=all_rows['PHOT_CHAN']),
        fits.Column(name='TIP', format='D', array=all_rows['TIP']),
        fits.Column(name='TILT', format='D', array=all_rows['TILT']),
        fits.Column(name='FLUX', format='D', array=all_rows['FLUX']),
        fits.Column(name='AVG_INDEX', format='J', array=all_rows['AVG_INDEX']),
    ])

    primary_hdu = fits.PrimaryHDU()
    if header_info is not None:
        for key, value in header_info.items():
            if key.upper() not in ['SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND']:
                primary_hdu.header[key[:8].upper()] = value

    if comments is not None:
        for c in comments:
            primary_hdu.header.add_comment(c)

    table_hdu = fits.BinTableHDU.from_columns(cols, name='SCAN_DATA')
    fits.HDUList([primary_hdu, table_hdu]).writeto(savefile, overwrite=True)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    savepath = output_directory()
    prepare_savepath(savepath)

    tips = make_grid_positions(TIP_START, TIP_STOP, TIP_STEP)
    tilts = make_grid_positions(TILT_START, TILT_STOP, TILT_STEP)

    print(f"Tip grid:  {tips[0]:.6g} to {tips[-1]:.6g}, N={len(tips)}, step~{TIP_STEP}")
    print(f"Tilt grid: {tilts[0]:.6g} to {tilts[-1]:.6g}, N={len(tilts)}, step~{TILT_STEP}")
    print(f"Parked state for non-scanned segments: piston={PARKED_PISTON}, tip/tilt={PARKED_TIPTILT}")
    print("Photometric boxes:", {ch: getbox(ch) for ch in PHOTOMETRIC_CHANNELS})
    print(f"Total grid points per segment: {len(tips) * len(tilts)}")
    print(f"Total measurements per average: {37 * len(tips) * len(tilts) * len(PHOTOMETRIC_CHANNELS)} rows")

    dark = getdark(DARK_FILEPATH)
    apapane, dm = open_devices()

    all_rows = []

    for avg_index in range(NUMAVG):
        print(f"\nStarting average repeat {avg_index + 1}/{NUMAVG}")
        park_all_segments(dm)

        for segment in range(37):
            segment_rows = scan_tiptilt_for_segment(
                apapane=apapane,
                dm=dm,
                segment=segment,
                tips=tips,
                tilts=tilts,
                dark=dark,
                nframes=NFRAMES,
            )

            for seg, phot_chan, tip, tilt, flux in segment_rows:
                all_rows.append((
                    int(seg),
                    int(phot_chan),
                    float(tip),
                    float(tilt),
                    float(flux),
                    int(avg_index),
                ))

        park_all_segments(dm)

    header_info = {
        'DATE': str(DATE),
        'YEAR': int(YEAR),
        'ITER': int(ITERATION),
        'NFRAME': int(NFRAMES),
        'NUMAVG': int(NUMAVG),
        'TIP0': float(TIP_START),
        'TIP1': float(TIP_STOP),
        'TIPSTEP': float(TIP_STEP),
        'TILT0': float(TILT_START),
        'TILT1': float(TILT_STOP),
        'TILSTEP': float(TILT_STEP),
        'BOXHW': int(BOX_HALFWIDTH),
    }

    comments = [
        "All-segment tip/tilt parasitic coupling map.",
        "All non-scanned segments are parked at PARKED_TIPTILT.",
        "Each segment is scanned over TIP/TILT and photometric channels 1, 2, 3 are recorded.",
        f"Parked state: piston={PARKED_PISTON}, tip/tilt={PARKED_TIPTILT}.",
        f"Boxes: { {ch: getbox(ch) for ch in PHOTOMETRIC_CHANNELS} }.",
    ]

    savefile = os.path.join(savepath, f"tiptilt_map_table_{ITERATION}.fits")
    save_tiptilt_map_table(savefile, all_rows, header_info=header_info, comments=comments)
    print(f"\nSaved {savefile}")
