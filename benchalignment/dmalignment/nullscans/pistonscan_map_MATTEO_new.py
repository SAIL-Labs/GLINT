"""
pistonscan_map_MATTEO_modes.py

Piston-scan map script with selectable injection/photometry modes.

Modes
-----
1. OPTIMISE_ALL_INJECTED_SEGMENTS_SIMULTANEOUSLY = False
   One injected segment is optimised at a time.
   The other two injected segments are parked like all the other non-active
   segments.

   RECORD_ALL_PHOTOMETRY_SIMULTANEOUSLY = False
      active 11 -> record Phot 1 only
      active 20 -> record Phot 2 only
      active 31 -> record Phot 3 only
      Final table has 3 photometry maps total, like the previous script.

   RECORD_ALL_PHOTOMETRY_SIMULTANEOUSLY = True
      active 11 -> record Phot 1, 2, 3 simultaneously
      active 20 -> record Phot 1, 2, 3 simultaneously
      active 31 -> record Phot 1, 2, 3 simultaneously
      Final table has 9 active-segment/photometry maps total.

2. OPTIMISE_ALL_INJECTED_SEGMENTS_SIMULTANEOUSLY = True
   Segments 11, 20, and 31 are all held at their optimised tip/tilt values
   simultaneously. All other segments are parked according to PARK_OTHER_SEGMENTS.
   Phot 1, 2, and 3 are always recorded simultaneously.
   Final table has 3 photometry maps total.

The FITS table keeps the original style:
    SEGMENT, PHOT_CHAN, OPD, FLUX, AVG_INDEX
and adds:
    ACTIVE_SEGMENT
where ACTIVE_SEGMENT is 11/20/31 for the one-active-at-a-time modes, and -1 for
all-injected-segments-simultaneously mode.
"""

import sys
sys.path.append('/home/scexao/glint/control-code/')

import os
import time
import shutil
from datetime import datetime

import numpy as np
from astropy.io import fits
import tqdm

import shmDMcontrol
from pyMilk.interfacing.shm import SHM


# =============================================================================
# USER PARAMETERS - EDIT THIS SECTION BEFORE RUNNING
# =============================================================================

# Output bookkeeping. The FITS file will be written to:
#   OUTPUT_ROOT / YEAR / DATE / scanITERATION / nullscan_table_ITERATION.fits
OUTPUT_ROOT = "/home/scexao/glint/alignment_scans/nullscans"
YEAR = 2026
#DATE = "06-01_TILTED"           #False tilted
#DATE = "06-01_TILTED_ALLINJ"   #True tilted
#DATE = "06-01_FLAT"            #False flat
#DATE = "06-01_FLAT_ALLINJ"     #True flat
DATE = "06-09"
ITERATION = 1

# -----------------------------------------------------------------------------
# Scan mode switches
# -----------------------------------------------------------------------------

# Record all three photometric channels from the same APAPANE acquisition.
# This option is used only when OPTIMISE_ALL_INJECTED_SEGMENTS_SIMULTANEOUSLY=False.
# If all injected segments are optimised simultaneously, all three photometries
# are always recorded, regardless of this value.
RECORD_ALL_PHOTOMETRY_SIMULTANEOUSLY = True

# False -> optimise only one injected segment at a time.
# True  -> optimise all injected segments, 11/20/31, at the same time.
OPTIMISE_ALL_INJECTED_SEGMENTS_SIMULTANEOUSLY = False

# Parking state for every segment that is not currently optimised/injected.
# In one-active mode, this includes the other two nominally injected segments.
# In all-injected mode, this means every segment except 11, 20, and 31.
# Choose ONE of:
#   PARK_OTHER_SEGMENTS = "flat"    -> parked segments use (tip, tilt) = (0, 0)
#   PARK_OTHER_SEGMENTS = "tilted"  -> parked segments use PARKED_SEGMENT_TIPTILT[segment]
#
# IMPORTANT: when PARK_OTHER_SEGMENTS="tilted", every DM segment 0..36 must
# have its own explicit (tip, tilt) entry in PARKED_SEGMENT_TIPTILT. This avoids
# accidentally applying the same parked vector to all segments on a sensitive setup.
PARK_OTHER_SEGMENTS = "tilted"      # "flat" or "tilted"
PARKED_SEGMENT_TIPTILT = {
    # EDIT THESE VALUES MANUALLY. The values below preserve the previous
    # behaviour until you replace each entry with its own optimised park vector.
    # Format: segment: (tip, tilt)
    0: (0, 0),
    1: (0, 0),
    2: (0, 4),
    3: (0, 4),
    4: (0, 4),
    5: (-2, -3),
    6: (-2, -3),
    7: (-3, -3),
    8: (-3, -3),
    9: (0, 0),
    10: (-3, -1),
    11: (-3, -2),
    12: (-4, -2),
    13: (-4, -3),
    14: (-4, 2),
    15: (0, 0),
    16: (-3, -4),
    17: (-3, -4),
    18: (-3, -4),
    19: (-5, 3),
    20: (-3, 4),
    21: (-3, 4),
    22: (0, 0),
    23: (2, -4),
    24: (2, -4),
    25: (2, -4),
    26: (5, -3),
    27: (-3, 4),
    28: (3, -3),
    29: (3, -3),
    30: (4, -3),
    31: (3, -4),
    32: (4, 2),
    33: (2, 0),
    34: (3, -1),
    35: (3, -4),
    36: (3, -2),
}

# Camera / dark settings
APAPANE_SHM_NAME = "apapane"
DARK_FILEPATH = "/home/scexao/glint/darkMatteo.fits"
NFRAMES = 25
NUMAVG = 1

# Piston scan settings in DM piston-command units.
# OPD is written as 2 * (piston_position - final_piston_position), matching the
# original pistonscan_map.py convention.
PISTON_START = -0.6
PISTON_STOP = 0.3
PISTON_STEP = 0.05

# Active injected segment -> corresponding photometric channel mapping.
ACTIVE_SEGMENT_TO_PHOT_CHAN = {
    11: 1,
    20: 2,
    31: 3,
}

# Optimised tip/tilt values for the injected segments.
# EDIT THESE VALUES MANUALLY after your latest pitch/yaw or tip/tilt optimisation.
# Format: segment: (tip, tilt)
INJECTED_SEGMENT_TIPTILT = {
    11: (0.49, -0.09),
    20: (0.1, -0.27),
    31: (-0.02, 0.25),
}

# Spectral extraction boxes.
# Original convention for ISCRED1=True:
#   box = [peak - halfwidth, peak + halfwidth, left, right]
# and getdata/get_fluxes slices image[top:bottom, left:right].
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

# Safety / timing
DM_SETTLE_SECONDS = 0.001
INITIAL_SETTLE_SECONDS = 0.01
ABORT_IF_OUTPUT_EXISTS = True


# =============================================================================
# Helpers
# =============================================================================

def injected_segments():
    return tuple(ACTIVE_SEGMENT_TO_PHOT_CHAN.keys())


def all_photometric_channels():
    return tuple(sorted(PHOTOMETRIC_PEAKS.keys()))


def check_tiptilt_pair(name, value):
    """Return a validated (tip, tilt) pair as floats."""
    try:
        tip, tilt = value
    except Exception as exc:
        raise ValueError(f"{name} must be a two-value iterable: (tip, tilt). Got {value!r}") from exc
    return float(tip), float(tilt)


def get_park_tiptilt(segment):
    """Return the parked tip/tilt for one specific segment."""
    segment = int(segment)
    mode = PARK_OTHER_SEGMENTS.lower()

    if mode == "flat":
        return 0.0, 0.0

    if mode == "tilted":
        if segment not in PARKED_SEGMENT_TIPTILT:
            raise KeyError(
                f"No parked tip/tilt provided for segment {segment}. "
                "When PARK_OTHER_SEGMENTS='tilted', PARKED_SEGMENT_TIPTILT "
                "must contain every segment 0..36."
            )
        return check_tiptilt_pair(
            f"PARKED_SEGMENT_TIPTILT[{segment}]",
            PARKED_SEGMENT_TIPTILT[segment],
        )

    raise ValueError(
        f"Unknown PARK_OTHER_SEGMENTS={PARK_OTHER_SEGMENTS!r}. "
        "Use 'flat' or 'tilted'."
    )


def validate_configuration():
    """Fail before touching the DM if the scan configuration is inconsistent."""
    mode = PARK_OTHER_SEGMENTS.lower()
    if mode not in {"flat", "tilted"}:
        raise ValueError(
            f"Unknown PARK_OTHER_SEGMENTS={PARK_OTHER_SEGMENTS!r}. "
            "Use 'flat' or 'tilted'."
        )

    expected_segments = set(range(37))
    if mode == "tilted":
        provided_segments = set(int(seg) for seg in PARKED_SEGMENT_TIPTILT.keys())
        missing = sorted(expected_segments - provided_segments)
        extra = sorted(provided_segments - expected_segments)
        if missing:
            raise KeyError(f"PARKED_SEGMENT_TIPTILT is missing segment(s): {missing}")
        if extra:
            raise KeyError(f"PARKED_SEGMENT_TIPTILT has invalid segment(s): {extra}")
        for seg in range(37):
            check_tiptilt_pair(f"PARKED_SEGMENT_TIPTILT[{seg}]", PARKED_SEGMENT_TIPTILT[seg])

    for seg in injected_segments():
        if seg not in INJECTED_SEGMENT_TIPTILT:
            raise KeyError(f"No optimised injected tip/tilt provided for segment {seg}")
        check_tiptilt_pair(f"INJECTED_SEGMENT_TIPTILT[{seg}]", INJECTED_SEGMENT_TIPTILT[seg])

    for active_segment, phot_chan in ACTIVE_SEGMENT_TO_PHOT_CHAN.items():
        if int(phot_chan) not in PHOTOMETRIC_PEAKS:
            raise KeyError(
                f"ACTIVE_SEGMENT_TO_PHOT_CHAN[{active_segment}] maps to Phot {phot_chan}, "
                "but that photometric channel is not present in PHOTOMETRIC_PEAKS."
            )


def parked_tiptilt_summary():
    if PARK_OTHER_SEGMENTS.lower() == "flat":
        return "flat, all parked tip/tilt=(0.0, 0.0)"
    return f"tilted, per-segment parked tip/tilt dictionary with {len(PARKED_SEGMENT_TIPTILT)} entries"


def make_piston_positions(start, stop, step):
    """Return inclusive piston positions from start to stop with the requested step."""
    if step <= 0:
        raise ValueError("PISTON_STEP must be positive.")
    nsteps = int(np.floor((stop - start) / step + 0.5)) + 1
    positions = start + step * np.arange(nsteps, dtype=float)

    # Include stop robustly if it lies on the grid within floating-point tolerance.
    if not np.isclose(positions[-1], stop, atol=abs(step) * 1e-6):
        if positions[-1] < stop:
            positions = np.append(positions, stop)
        else:
            positions[-1] = stop
    return positions


def getbox(phot_chan):
    """Build [top, bottom, left, right] boxes using the original scan convention."""
    peak = PHOTOMETRIC_PEAKS[int(phot_chan)]
    val1, val2 = PHOTOMETRIC_BOUNDINGVALS[int(phot_chan)]
    if ISCRED1:
        top, bottom = peak - BOX_HALFWIDTH, peak + BOX_HALFWIDTH
        left, right = val1, val2
    else:
        left, right = peak - BOX_HALFWIDTH, peak + BOX_HALFWIDTH
        top, bottom = val1, val2
    return [int(top), int(bottom), int(left), int(right)]


def crop_sum(image, box):
    """Sum one photometric extraction box from an already dark-subtracted image."""
    top, bottom, left, right = box
    if right <= left or bottom <= top:
        raise ValueError(
            f"Invalid box dimensions: top={top}, bottom={bottom}, "
            f"left={left}, right={right}"
        )
    return float(np.sum(image[int(top):int(bottom), int(left):int(right)]))


def get_fluxes(apapane, phot_chans, dark, nframes=1):
    """
    Read APAPANE once, average nframes, subtract dark, and return fluxes for all
    requested photometric channels from the same acquisition.
    """
    bright = apapane.multi_recv_data(nframes)
    bright = np.array(bright, dtype=float)
    avg = np.mean(bright, axis=0)
    data = avg - dark
    return {int(ch): crop_sum(data, getbox(int(ch))) for ch in phot_chans}


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


def park_segment(dm, segment, piston=0.0):
    tip, tilt = get_park_tiptilt(segment)
    set_segment(dm, segment, piston, tip, tilt)


def set_injected_segment(dm, segment, piston=0.0):
    if segment not in INJECTED_SEGMENT_TIPTILT:
        raise KeyError(f"No optimised tip/tilt provided for segment {segment}")
    tip, tilt = check_tiptilt_pair(
        f"INJECTED_SEGMENT_TIPTILT[{segment}]",
        INJECTED_SEGMENT_TIPTILT[segment],
    )
    set_segment(dm, segment, piston, tip, tilt)


def park_all_segments(dm):
    for seg in range(37):
        park_segment(dm, seg, piston=0.0)


def initialise_dm_one_active(dm, active_segment):
    """
    Park all segments, then put exactly one injected segment into its optimised
    tip/tilt state at piston=0. The other two injected segments remain parked.
    """
    park_all_segments(dm)
    set_injected_segment(dm, active_segment, piston=0.0)
    time.sleep(INITIAL_SETTLE_SECONDS)


def initialise_dm_all_injected(dm):
    """
    Park all segments, then put all injected segments into their optimised
    tip/tilt state at piston=0.
    """
    park_all_segments(dm)
    for seg in injected_segments():
        set_injected_segment(dm, seg, piston=0.0)
    time.sleep(INITIAL_SETTLE_SECONDS)


def apply_state_one_active(dm, scanned_segment, piston, active_segment):
    """
    One-active mode:
    - active_segment is optimised;
    - every other segment, including the other two nominally injected segments,
      is parked;
    - scanned_segment receives the piston command.
    """
    if scanned_segment == active_segment:
        set_injected_segment(dm, active_segment, piston=piston)
    else:
        set_injected_segment(dm, active_segment, piston=0.0)
        park_segment(dm, scanned_segment, piston=piston)


def apply_state_all_injected(dm, scanned_segment, piston):
    """
    All-injected mode:
    - injected segments 11, 20, and 31 are all optimised;
    - all other segments are parked;
    - scanned_segment receives the piston command, while preserving either its
      optimised tip/tilt if it is injected, or its parked tip/tilt if it is not.
    """
    inj = set(injected_segments())
    for seg in inj:
        set_injected_segment(dm, seg, piston=piston if scanned_segment == seg else 0.0)

    if scanned_segment not in inj:
        park_segment(dm, scanned_segment, piston=piston)


def piston_scan_one_segment_one_active(apapane, dm, scanned_segment, active_segment,
                                       phot_chans, pistpositions, dark, nframes):
    """
    Piston one segment while keeping only active_segment optimised. Return a
    dict {phot_chan: flux_array} and the OPD vector.
    """
    phot_chans = tuple(int(ch) for ch in phot_chans)
    fluxes = {ch: np.zeros(len(pistpositions), dtype=float) for ch in phot_chans}

    apply_state_one_active(dm, scanned_segment, pistpositions[0], active_segment)
    time.sleep(INITIAL_SETTLE_SECONDS)

    pbar = tqdm.tqdm(
        desc=f"active {active_segment}, phot {phot_chans}, scan seg {scanned_segment}",
        total=len(pistpositions),
        leave=False,
    )

    for i, piston in enumerate(pistpositions):
        apply_state_one_active(dm, scanned_segment, piston, active_segment)
        time.sleep(DM_SETTLE_SECONDS)
        sample = get_fluxes(apapane, phot_chans, dark, nframes)
        for ch in phot_chans:
            fluxes[ch][i] = sample[ch]
        pbar.update()

    pbar.close()

    # Restore post-scan state.
    if scanned_segment == active_segment:
        set_injected_segment(dm, active_segment, piston=0.0)
    else:
        park_segment(dm, scanned_segment, piston=0.0)
        set_injected_segment(dm, active_segment, piston=0.0)

    opd = 2.0 * (pistpositions - pistpositions[-1])
    return fluxes, opd


def piston_scan_one_segment_all_injected(apapane, dm, scanned_segment,
                                         phot_chans, pistpositions, dark, nframes):
    """
    Piston one segment while keeping all injected segments optimised. Return a
    dict {phot_chan: flux_array} and the OPD vector.
    """
    phot_chans = tuple(int(ch) for ch in phot_chans)
    fluxes = {ch: np.zeros(len(pistpositions), dtype=float) for ch in phot_chans}

    apply_state_all_injected(dm, scanned_segment, pistpositions[0])
    time.sleep(INITIAL_SETTLE_SECONDS)

    pbar = tqdm.tqdm(
        desc=f"all injected active, phot {phot_chans}, scan seg {scanned_segment}",
        total=len(pistpositions),
        leave=False,
    )

    for i, piston in enumerate(pistpositions):
        apply_state_all_injected(dm, scanned_segment, piston)
        time.sleep(DM_SETTLE_SECONDS)
        sample = get_fluxes(apapane, phot_chans, dark, nframes)
        for ch in phot_chans:
            fluxes[ch][i] = sample[ch]
        pbar.update()

    pbar.close()

    # Restore post-scan state.
    if scanned_segment in injected_segments():
        set_injected_segment(dm, scanned_segment, piston=0.0)
    else:
        park_segment(dm, scanned_segment, piston=0.0)
    for seg in injected_segments():
        set_injected_segment(dm, seg, piston=0.0)

    opd = 2.0 * (pistpositions - pistpositions[-1])
    return fluxes, opd


def append_flux_rows(all_rows, scanned_segment, active_segment, fluxes, opd, avg_index):
    for phot_chan, flux in fluxes.items():
        for i in range(len(opd)):
            all_rows.append((
                int(scanned_segment),
                int(active_segment),
                int(phot_chan),
                float(opd[i]),
                float(flux[i]),
                int(avg_index),
            ))


def save_nullscan_table(savefile, all_rows, header_info=None, comments=None):
    """Save the SCAN_DATA table with the original columns plus ACTIVE_SEGMENT."""
    all_rows = np.array(
        all_rows,
        dtype=[
            ('SEGMENT', 'i4'),
            ('ACTIVE_SEGMENT', 'i4'),
            ('PHOT_CHAN', 'i4'),
            ('OPD', 'f8'),
            ('FLUX', 'f8'),
            ('AVG_INDEX', 'i4'),
        ]
    )

    cols = fits.ColDefs([
        fits.Column(name='SEGMENT', format='J', array=all_rows['SEGMENT']),
        fits.Column(name='ACTIVE_SEGMENT', format='J', array=all_rows['ACTIVE_SEGMENT']),
        fits.Column(name='PHOT_CHAN', format='J', array=all_rows['PHOT_CHAN']),
        fits.Column(name='OPD', format='D', array=all_rows['OPD']),
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
    validate_configuration()

    savepath = output_directory()
    prepare_savepath(savepath)

    pistpositions = make_piston_positions(PISTON_START, PISTON_STOP, PISTON_STEP)
    print(
        f"Piston scan: {pistpositions[0]:.6g} to {pistpositions[-1]:.6g} "
        f"in {len(pistpositions)} steps of ~{PISTON_STEP}"
    )
    print(f"Optimise all injected segments simultaneously: {OPTIMISE_ALL_INJECTED_SEGMENTS_SIMULTANEOUSLY}")
    print(f"Record all 3 photometry simultaneously: {RECORD_ALL_PHOTOMETRY_SIMULTANEOUSLY}")
    print(f"Parking mode for non-optimised segments: {parked_tiptilt_summary()}")
    if PARK_OTHER_SEGMENTS.lower() == "tilted":
        print(f"Parked segment tip/tilts: {PARKED_SEGMENT_TIPTILT}")
    print(f"Injected segment tip/tilts: {INJECTED_SEGMENT_TIPTILT}")
    print("Photometric boxes:", {ch: getbox(ch) for ch in all_photometric_channels()})

    dark = getdark(DARK_FILEPATH)
    apapane, dm = open_devices()

    all_rows = []

    for avg_index in range(NUMAVG):
        print(f"\nStarting average repeat {avg_index + 1}/{NUMAVG}")

        if OPTIMISE_ALL_INJECTED_SEGMENTS_SIMULTANEOUSLY:
            # In this mode all three photometries are necessarily simultaneous.
            phot_chans = all_photometric_channels()
            active_segment_label = -1
            print(
                "\nAll injected segments optimised simultaneously; "
                "recording Phot 1, 2, and 3 simultaneously."
            )
            initialise_dm_all_injected(dm)

            for scanned_segment in range(37):
                fluxes, opd = piston_scan_one_segment_all_injected(
                    apapane=apapane,
                    dm=dm,
                    scanned_segment=scanned_segment,
                    phot_chans=phot_chans,
                    pistpositions=pistpositions,
                    dark=dark,
                    nframes=NFRAMES,
                )
                append_flux_rows(
                    all_rows=all_rows,
                    scanned_segment=scanned_segment,
                    active_segment=active_segment_label,
                    fluxes=fluxes,
                    opd=opd,
                    avg_index=avg_index,
                )

            park_all_segments(dm)
            time.sleep(INITIAL_SETTLE_SECONDS)

        else:
            # In this mode only one injected segment is optimised at a time.
            for active_segment, mapped_phot_chan in ACTIVE_SEGMENT_TO_PHOT_CHAN.items():
                if RECORD_ALL_PHOTOMETRY_SIMULTANEOUSLY:
                    phot_chans = all_photometric_channels()
                    phot_msg = "recording Phot 1, 2, and 3 simultaneously"
                else:
                    phot_chans = (mapped_phot_chan,)
                    phot_msg = f"recording corresponding Phot {mapped_phot_chan} only"

                print(f"\nActive optimised segment {active_segment}; {phot_msg}.")
                initialise_dm_one_active(dm, active_segment)

                for scanned_segment in range(37):
                    fluxes, opd = piston_scan_one_segment_one_active(
                        apapane=apapane,
                        dm=dm,
                        scanned_segment=scanned_segment,
                        active_segment=active_segment,
                        phot_chans=phot_chans,
                        pistpositions=pistpositions,
                        dark=dark,
                        nframes=NFRAMES,
                    )
                    append_flux_rows(
                        all_rows=all_rows,
                        scanned_segment=scanned_segment,
                        active_segment=active_segment,
                        fluxes=fluxes,
                        opd=opd,
                        avg_index=avg_index,
                    )

                # After finishing one active injected-segment map, park everything
                # before switching to the next optimised injected segment.
                park_all_segments(dm)
                time.sleep(INITIAL_SETTLE_SECONDS)

    header_info = {
        'DATE': str(DATE),
        'YEAR': int(YEAR),
        'ITER': int(ITERATION),
        'NFRAME': int(NFRAMES),
        'NUMAVG': int(NUMAVG),
        'PSTART': float(PISTON_START),
        'PSTOP': float(PISTON_STOP),
        'PSTEP': float(PISTON_STEP),
        'BOXHW': int(BOX_HALFWIDTH),
        'ALLINJ': bool(OPTIMISE_ALL_INJECTED_SEGMENTS_SIMULTANEOUSLY),
        'ALLPHOT': bool(RECORD_ALL_PHOTOMETRY_SIMULTANEOUSLY),
    }

    if OPTIMISE_ALL_INJECTED_SEGMENTS_SIMULTANEOUSLY:
        mode_comment = "All injected segments optimised simultaneously; Phot 1/2/3 recorded simultaneously."
    elif RECORD_ALL_PHOTOMETRY_SIMULTANEOUSLY:
        mode_comment = "One injected segment optimised at a time; Phot 1/2/3 recorded for each active segment."
    else:
        mode_comment = "One injected segment optimised at a time; only corresponding photometric channel recorded."

    comments = [
        mode_comment,
        "FITS table uses SEGMENT, ACTIVE_SEGMENT, PHOT_CHAN, OPD, FLUX, AVG_INDEX.",
        "ACTIVE_SEGMENT is -1 when all injected segments are optimised simultaneously.",
        f"Non-optimised segments parked with {parked_tiptilt_summary()}.",
        f"Parked segment tip/tilts: {PARKED_SEGMENT_TIPTILT}.",
        f"Injected tip/tilts: {INJECTED_SEGMENT_TIPTILT}.",
        f"Boxes: { {ch: getbox(ch) for ch in all_photometric_channels()} }.",
    ]

    savefile = os.path.join(savepath, f"nullscan_table_{ITERATION}.fits")
    save_nullscan_table(savefile, all_rows, header_info=header_info, comments=comments)
    print(f"\nSaved {savefile}")
