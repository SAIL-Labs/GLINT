"""
baseline_differential_null_scan.py

Differential-piston null scan for selected injected GLINT/HEX111 baselines.

Purpose
-------
After pitch/yaw, XY, and segment tip/tilt optimisation, this script scans the
remaining degree of freedom: differential piston between the two injected
segments of each baseline. It measures the integrated flux in the corresponding
null-channel box, finds the differential piston/OPD that minimises the null, and
solves for a consistent set of piston commands for segments 11, 20, and 31 that
lies inside the allowed piston range of each segment.

This script is intentionally self-contained: it does NOT read scanparameters.json.
Edit the USER PARAMETERS block before running.

Safety notes
------------
- APPLY_FINAL_PISTONS is False by default. The script will measure and compute
  the recommended pistons, but it will not apply them at the end unless you set
  APPLY_FINAL_PISTONS = True.
- CONFIRM_HARDWARE = True requires typing YES before moving hardware.
- The script aborts if it cannot place the solved pistons inside the requested
  per-segment limits.

Table output
------------
The scan FITS file contains extension SCAN_DATA with columns:
    BASELINE, SEG_A, SEG_B, NULL_CHAN,
    DIFF_PISTON, OPD, PISTON_A, PISTON_B, FLUX, AVG_INDEX

The solution FITS file contains extension PISTON_SOLUTION with columns:
    SEGMENT, PISTON

A JSON sidecar is also saved with the same solution and diagnostic information.
"""

import sys
sys.path.append('/home/scexao/glint/control-code/')

import os
import json
import time
import shutil
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import tqdm

import shmDMcontrol
from pyMilk.interfacing.shm import SHM


# =============================================================================
# USER PARAMETERS - EDIT THIS SECTION BEFORE RUNNING
# =============================================================================

# Output bookkeeping. Files will be written to:
#   OUTPUT_ROOT / YEAR / DATE / scanITERATION /
OUTPUT_ROOT = "/home/scexao/glint/alignment_scans/nullscans"
YEAR = 2026
DATE = "06-02"          # e.g. "05-28"
ITERATION = 7

# Camera / dark settings.
APAPANE_SHM_NAME = "apapane"
DARK_FILEPATH = "/home/scexao/glint/darkMatteo.fits"
NFRAMES = 25
NUMAVG = 1

# Differential piston scan settings, in DM piston-command units.
# For a reflective DM, OPD = 2 * differential piston.
DIFF_PISTON_STEP = 0.05

# Piston limits for the injected segments, in the same units used by dm.set_segment.
# The solver chooses the arbitrary global piston offset so all final pistons fall
# within these limits. You can set different limits per segment if needed.
PISTON_LIMITS = {
    11: (-1.3, 1.3),
    20: (-1.3, 1.3),
    31: (-1.3, 1.3),
}

# Reference pistons used for injected segments not involved in the current baseline.
# These should normally be inside PISTON_LIMITS. They only affect the scan state,
# not the final relative-piston solution.
REFERENCE_PISTONS = {
    11: 0.0,
    20: 0.0,
    31: 0.0,
}

# Optimised tip/tilt values for the injected segments.
# Fill these from your latest injection optimisation.
# Format: segment: (tip, tilt)
INJECTED_SEGMENT_TIPTILT = {
    11: (0.39, -0.1),
    20: (0.14, -0.49),
    31: (0.02, 0.17),
}

# Commands for all non-injected segments while scanning the baseline nulls.
# Format: segment: (piston, tip, tilt)
OTHER_SEGMENT_COMMANDS = {
    0:  (0, -3, 4),
    1:  (0, -3, 4),
    2:  (0, -3, 4),
    3:  (0, -3, 4),
    4:  (0, -3, 4),
    5:  (0, -3, 4),
    6:  (0, -3, 4),
    7:  (0, -3, 4),
    8:  (0, 3, -4),
    9:  (0, 3, -4),
    10: (0, 3, -4),
    12: (0, 3, -4),
    13: (0, -3, 4),
    14: (0, -3, 4),
    15: (0, 3, -4),
    16: (0, 0, -3),
    17: (0, 2, -4),
    18: (0, 3, -4),
    19: (0, 4, -5),
    21: (0, -3, 4),
    22: (0, 3, -4),
    23: (0, 3, -4),
    24: (0, 3, -4),
    25: (0, 3, -4),
    26: (0, 5, -2),
    27: (0, 4, 3),
    28: (0, 3, -4),
    29: (0, 3, -4),
    30: (0, 3, -4),
    32: (0, 3, -4),
    33: (0, 3, -4),
    34: (0, 3, -4),
    35: (0, 3, -4),
    36: (0, 3, -4),
}

# Baseline definitions.
# IMPORTANT: BOX convention is [top, bottom, left, right], used as:
#     image[top:bottom, left:right]
# Replace the BOX values with the null-channel boxes you measured.
BASELINES = [
    {
        "name": "11-20",
        "use": True,
        "seg_a": 11,
        "seg_b": 20,
        "null_chan": 1,
        "box": [110, 114, 229, 267],
    },
    {
        "name": "11-31",
        "use": True,
        "seg_a": 11,
        "seg_b": 31,
        "null_chan": 3,
        "box": [166, 170, 229, 267],
    },
    {
        "name": "20-31",
        "use": False,  # disabled: chip path mismatch cannot be compensated
        "seg_a": 20,
        "seg_b": 31,
        "null_chan": 2,
        "box": [35, 39, 229, 267],
    },
]

# Timing.
DM_SETTLE_SECONDS = 0.01
INITIAL_SETTLE_SECONDS = 0.05

# Safety.
CONFIRM_HARDWARE = True
ABORT_IF_OUTPUT_EXISTS = True
APPLY_FINAL_PISTONS = False  # keep False unless you are ready to command the final solution

# Fitting/selection of best null.
# "sample_min" chooses the measured minimum. "parabolic" fits the minimum point
# and its two neighbours for a sub-step estimate, but falls back safely to sample_min.
MINIMUM_METHOD = "parabolic"  # "sample_min" or "parabolic"

# Baseline selection / plotting.
# Set use=True/False in BASELINES below to choose which baselines are scanned.
SAVE_OPD_PLOT = True
SHOW_OPD_PLOT = True


# =============================================================================
# Low-level helpers
# =============================================================================

def validate_parameters() -> None:
    injected = set(INJECTED_SEGMENT_TIPTILT.keys())
    required = {11, 20, 31}
    if injected != required:
        raise ValueError(f"INJECTED_SEGMENT_TIPTILT must contain exactly {required}; got {injected}")

    if set(PISTON_LIMITS.keys()) != required:
        raise ValueError(f"PISTON_LIMITS must contain exactly {required}; got {set(PISTON_LIMITS.keys())}")

    if set(REFERENCE_PISTONS.keys()) != required:
        raise ValueError(f"REFERENCE_PISTONS must contain exactly {required}; got {set(REFERENCE_PISTONS.keys())}")

    if DIFF_PISTON_STEP <= 0:
        raise ValueError("DIFF_PISTON_STEP must be positive.")

    names = set()
    for baseline in BASELINES:
        for key in ["name", "seg_a", "seg_b", "null_chan", "box"]:
            if key not in baseline:
                raise ValueError(f"Baseline missing key {key!r}: {baseline}")
        if baseline["name"] in names:
            raise ValueError(f"Duplicate baseline name: {baseline['name']}")
        names.add(baseline["name"])

        a = int(baseline["seg_a"])
        b = int(baseline["seg_b"])
        if a == b:
            raise ValueError(f"Baseline has identical segments: {baseline}")
        if a not in required or b not in required:
            raise ValueError(f"Baseline segments must be among {required}: {baseline}")
        validate_box(baseline["box"])

    used = [b for b in BASELINES if bool(b.get("use", True))]
    if len(used) == 0:
        raise ValueError("No baselines enabled. Set baseline['use'] = True for at least one baseline.")

    if MINIMUM_METHOD not in ["sample_min", "parabolic"]:
        raise ValueError("MINIMUM_METHOD must be 'sample_min' or 'parabolic'.")

    for seg, (lo, hi) in PISTON_LIMITS.items():
        if not lo < hi:
            raise ValueError(f"Bad piston limits for segment {seg}: {(lo, hi)}")
        ref = REFERENCE_PISTONS[seg]
        if not (lo <= ref <= hi):
            raise ValueError(f"REFERENCE_PISTONS[{seg}]={ref} outside limits {(lo, hi)}")


def validate_box(box: List[int]) -> None:
    if len(box) != 4:
        raise ValueError(f"Box must have four entries [top,bottom,left,right], got {box}")
    top, bottom, left, right = [int(v) for v in box]
    if bottom <= top or right <= left:
        raise ValueError(f"Invalid box [top,bottom,left,right]={box}")


def output_directory() -> str:
    return os.path.join(OUTPUT_ROOT, str(YEAR), str(DATE), f"scan{ITERATION}")


def prepare_savepath(savepath: str) -> None:
    if os.path.isdir(savepath) and os.listdir(savepath):
        msg = f"Output directory already exists and is not empty: {savepath}"
        if ABORT_IF_OUTPUT_EXISTS:
            raise FileExistsError(msg + "\nChange ITERATION or set ABORT_IF_OUTPUT_EXISTS=False.")
        backup = savepath + "_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.move(savepath, backup)
        print(f"Moved existing output directory to {backup}")
    os.makedirs(savepath, exist_ok=True)
    print(f"Using output directory: {savepath}")


def getdark(dark_filepath: str) -> np.ndarray:
    with fits.open(dark_filepath) as hdul:
        dark = np.asarray(hdul[0].data, dtype=float)
        if dark.ndim == 3:
            dark = np.mean(dark, axis=0)
        elif dark.ndim != 2:
            raise ValueError(f"Dark frame must be 2-D or 3-D, got shape {dark.shape}")
        return np.asarray(dark, dtype=float)


def read_box_flux(apapane: SHM, box: List[int], dark: np.ndarray, nframes: int) -> float:
    top, bottom, left, right = [int(v) for v in box]
    frames = apapane.multi_recv_data(int(nframes))
    frames = np.asarray(frames, dtype=float)
    if frames.ndim == 2:
        avg = frames
    else:
        avg = np.mean(frames, axis=0)
    corrected = avg - dark
    crop = corrected[top:bottom, left:right]
    return float(np.sum(crop))


def open_devices() -> Tuple[SHM, Any]:
    apapane = SHM(APAPANE_SHM_NAME)
    dm = shmDMcontrol.DM()
    return apapane, dm


def set_segment(dm: Any, segment: int, piston: float, tip: float, tilt: float) -> None:
    dm.set_segment(int(segment), float(piston), float(tip), float(tilt))


def park_non_injected_segment(dm: Any, segment: int) -> None:
    piston, tip, tilt = OTHER_SEGMENT_COMMANDS[int(segment)]
    set_segment(dm, segment, piston, tip, tilt)


def set_injected_segment(dm: Any, segment: int, piston: float) -> None:
    tip, tilt = INJECTED_SEGMENT_TIPTILT[int(segment)]
    set_segment(dm, int(segment), float(piston), float(tip), float(tilt))


def initialise_dm_for_nulling(dm: Any) -> None:
    """Park non-injected segments and put 11/20/31 at optimised tip/tilt."""
    injected = set(INJECTED_SEGMENT_TIPTILT.keys())
    for seg in range(37):
        if seg in injected:
            set_injected_segment(dm, seg, REFERENCE_PISTONS[seg])
        else:
            park_non_injected_segment(dm, seg)
    time.sleep(INITIAL_SETTLE_SECONDS)


# =============================================================================
# Differential scan mathematics
# =============================================================================

def feasible_diff_range(seg_a: int, seg_b: int) -> Tuple[float, float]:
    """Return feasible mechanical differential piston d = p_a - p_b."""
    amin, amax = PISTON_LIMITS[int(seg_a)]
    bmin, bmax = PISTON_LIMITS[int(seg_b)]
    return float(amin - bmax), float(amax - bmin)


def make_diff_positions(seg_a: int, seg_b: int, step: float) -> np.ndarray:
    dmin, dmax = feasible_diff_range(seg_a, seg_b)
    nsteps = int(np.floor((dmax - dmin) / step + 0.5)) + 1
    dvals = dmin + step * np.arange(nsteps, dtype=float)
    if dvals[-1] < dmax - abs(step) * 1e-6:
        dvals = np.append(dvals, dmax)
    else:
        dvals[-1] = dmax
    return dvals


def choose_pair_pistons_for_diff(seg_a: int, seg_b: int, diff_piston: float) -> Tuple[float, float]:
    """
    Choose p_a and p_b such that p_a - p_b = diff_piston and both are inside
    PISTON_LIMITS. The common piston is chosen as close as possible to the mean
    of the two segment reference pistons.
    """
    a_min, a_max = PISTON_LIMITS[int(seg_a)]
    b_min, b_max = PISTON_LIMITS[int(seg_b)]
    d = float(diff_piston)

    # p_a = c + d/2, p_b = c - d/2.
    c_low = max(a_min - d / 2.0, b_min + d / 2.0)
    c_high = min(a_max - d / 2.0, b_max + d / 2.0)
    if c_low > c_high + 1e-12:
        raise ValueError(
            f"Differential piston {d} infeasible for baseline {seg_a}-{seg_b}. "
            f"Common interval [{c_low}, {c_high}] is empty."
        )

    desired_c = 0.5 * (REFERENCE_PISTONS[int(seg_a)] + REFERENCE_PISTONS[int(seg_b)])
    c = float(np.clip(desired_c, c_low, c_high))
    p_a = c + d / 2.0
    p_b = c - d / 2.0
    return float(p_a), float(p_b)


def scan_baseline(apapane: SHM, dm: Any, dark: np.ndarray,
                  baseline: Dict[str, Any], avg_index: int) -> Tuple[List[Tuple], Dict[str, Any]]:
    """Run one differential piston scan for one baseline."""
    name = str(baseline["name"])
    seg_a = int(baseline["seg_a"])
    seg_b = int(baseline["seg_b"])
    null_chan = int(baseline["null_chan"])
    box = [int(v) for v in baseline["box"]]

    dvals = make_diff_positions(seg_a, seg_b, DIFF_PISTON_STEP)

    # Hard endpoint check: with limits [-1.3,+1.3], this reaches
    # pA=-1.3,pB=+1.3 at the first sample and pA=+1.3,pB=-1.3 at the last.
    p_a0, p_b0 = choose_pair_pistons_for_diff(seg_a, seg_b, float(dvals[0]))
    p_a1, p_b1 = choose_pair_pistons_for_diff(seg_a, seg_b, float(dvals[-1]))
    print(
        f"Endpoint check {name}: first p{seg_a}={p_a0:.6g}, p{seg_b}={p_b0:.6g}; "
        f"last p{seg_a}={p_a1:.6g}, p{seg_b}={p_b1:.6g}"
    )

    rows = []
    fluxes = np.zeros(len(dvals), dtype=float)

    print(
        f"\nBaseline {name}: seg_a={seg_a}, seg_b={seg_b}, "
        f"null_chan={null_chan}, box={box}"
    )
    print(
        f"Scanning mechanical differential piston p{seg_a}-p{seg_b} "
        f"from {dvals[0]:.6g} to {dvals[-1]:.6g} in {len(dvals)} points."
    )

    initialise_dm_for_nulling(dm)

    pbar = tqdm.tqdm(total=len(dvals), desc=f"baseline {name}", leave=False)
    for i, d in enumerate(dvals):
        p_a, p_b = choose_pair_pistons_for_diff(seg_a, seg_b, d)

        # Keep all injected segments in their optimised tip/tilt state. Only the
        # two baseline pistons change; the third injected segment stays at reference.
        for seg in INJECTED_SEGMENT_TIPTILT:
            if seg == seg_a:
                set_injected_segment(dm, seg, p_a)
            elif seg == seg_b:
                set_injected_segment(dm, seg, p_b)
            else:
                set_injected_segment(dm, seg, REFERENCE_PISTONS[seg])

        time.sleep(DM_SETTLE_SECONDS)
        flux = read_box_flux(apapane, box, dark, NFRAMES)
        fluxes[i] = flux

        opd = 2.0 * d
        rows.append((
            name,
            seg_a,
            seg_b,
            null_chan,
            float(d),
            float(opd),
            float(p_a),
            float(p_b),
            float(flux),
            int(avg_index),
        ))
        pbar.update()
    pbar.close()

    best = estimate_minimum(dvals, fluxes, method=MINIMUM_METHOD)
    best.update({
        "baseline": name,
        "seg_a": seg_a,
        "seg_b": seg_b,
        "null_chan": null_chan,
        "box": box,
        "best_opd": 2.0 * best["best_diff_piston"],
        "avg_index": int(avg_index),
    })
    print(
        f"Baseline {name}: best diff piston p{seg_a}-p{seg_b} = "
        f"{best['best_diff_piston']:.6g}; OPD = {best['best_opd']:.6g}; "
        f"flux = {best['best_flux']:.6g}; method={best['method_used']}"
    )
    return rows, best


def estimate_minimum(x: np.ndarray, y: np.ndarray, method: str = "parabolic") -> Dict[str, Any]:
    """Find minimum measured flux, optionally refine with a local parabola."""
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("x and y must have same nonzero length.")

    idx = int(np.nanargmin(y))
    best_x = float(x[idx])
    best_y = float(y[idx])
    method_used = "sample_min"

    if method == "parabolic" and 0 < idx < len(x) - 1:
        xs = x[idx - 1: idx + 2].astype(float)
        ys = y[idx - 1: idx + 2].astype(float)
        if np.all(np.isfinite(xs)) and np.all(np.isfinite(ys)):
            try:
                a, b, c = np.polyfit(xs, ys, 2)
                if a > 0:
                    xv = -b / (2.0 * a)
                    if xs[0] <= xv <= xs[-1]:
                        yv = a * xv * xv + b * xv + c
                        best_x = float(xv)
                        best_y = float(yv)
                        method_used = "parabolic"
            except Exception:
                pass

    return {
        "best_diff_piston": best_x,
        "best_flux": best_y,
        "sample_min_diff_piston": float(x[idx]),
        "sample_min_flux": float(y[idx]),
        "sample_min_index": idx,
        "method_used": method_used,
    }


# =============================================================================
# Relative piston solver
# =============================================================================

def solve_relative_pistons(best_by_baseline: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Solve p_i - p_j = d_ij for segments 11/20/31 in least squares.
    The returned relative solution has mean zero; a global offset is chosen later.
    """
    segments = [11, 20, 31]
    seg_to_col = {seg: i for i, seg in enumerate(segments)}

    A = []
    b = []
    eq_labels = []
    for item in best_by_baseline:
        row = np.zeros(3, dtype=float)
        row[seg_to_col[int(item["seg_a"])]] = 1.0
        row[seg_to_col[int(item["seg_b"])]] = -1.0
        A.append(row)
        b.append(float(item["best_diff_piston"]))
        eq_labels.append(str(item["baseline"]))

    # Add gauge condition: mean piston = 0. This only fixes representation;
    # it does not affect differential OPDs.
    A.append(np.ones(3, dtype=float))
    b.append(0.0)
    eq_labels.append("gauge_mean_zero")

    A = np.vstack(A)
    b = np.asarray(b, dtype=float)
    p_rel, residuals, rank, svals = np.linalg.lstsq(A, b, rcond=None)

    rel = {seg: float(p_rel[seg_to_col[seg]]) for seg in segments}
    predicted = []
    residual_list = []
    for item in best_by_baseline:
        a = int(item["seg_a"])
        bb = int(item["seg_b"])
        pred = rel[a] - rel[bb]
        res = pred - float(item["best_diff_piston"])
        predicted.append(float(pred))
        residual_list.append({
            "baseline": str(item["baseline"]),
            "measured_diff_piston": float(item["best_diff_piston"]),
            "predicted_diff_piston": float(pred),
            "residual": float(res),
        })

    final, offset_info = choose_global_offset_inside_limits(rel)
    return {
        "relative_pistons_mean_zero": rel,
        "final_pistons": final,
        "global_offset": offset_info["global_offset"],
        "global_offset_interval": offset_info["global_offset_interval"],
        "residuals_by_baseline": residual_list,
        "lstsq_rank": int(rank),
        "lstsq_singular_values": [float(v) for v in svals],
    }


def choose_global_offset_inside_limits(relative_pistons: Dict[int, float]) -> Tuple[Dict[int, float], Dict[str, Any]]:
    """Choose C so p_final[seg] = p_relative[seg] + C lies within limits."""
    c_low = -np.inf
    c_high = np.inf
    for seg, p in relative_pistons.items():
        lo, hi = PISTON_LIMITS[int(seg)]
        c_low = max(c_low, lo - p)
        c_high = min(c_high, hi - p)

    if c_low > c_high + 1e-12:
        raise RuntimeError(
            "No global piston offset can place all solved pistons inside limits.\n"
            f"relative_pistons={relative_pistons}\n"
            f"PISTON_LIMITS={PISTON_LIMITS}\n"
            f"Required offset interval would be [{c_low}, {c_high}]."
        )

    # Prefer zero global offset if possible; otherwise choose the middle of the
    # feasible interval to maximise margin to the rails.
    if c_low <= 0.0 <= c_high:
        c = 0.0
    else:
        c = 0.5 * (c_low + c_high)

    final = {int(seg): float(p + c) for seg, p in relative_pistons.items()}

    # Final hard check.
    for seg, p in final.items():
        lo, hi = PISTON_LIMITS[int(seg)]
        if not (lo - 1e-9 <= p <= hi + 1e-9):
            raise RuntimeError(f"Internal error: final piston for segment {seg}={p} outside {(lo, hi)}")

    return final, {
        "global_offset": float(c),
        "global_offset_interval": [float(c_low), float(c_high)],
    }


# =============================================================================
# Saving
# =============================================================================

def save_scan_table(savefile: str, rows: List[Tuple], metadata: Dict[str, Any]) -> None:
    dtype = [
        ("BASELINE", "U16"),
        ("SEG_A", "i4"),
        ("SEG_B", "i4"),
        ("NULL_CHAN", "i4"),
        ("DIFF_PISTON", "f8"),
        ("OPD", "f8"),
        ("PISTON_A", "f8"),
        ("PISTON_B", "f8"),
        ("FLUX", "f8"),
        ("AVG_INDEX", "i4"),
    ]
    arr = np.array(rows, dtype=dtype)

    cols = fits.ColDefs([
        fits.Column(name="BASELINE", format="16A", array=arr["BASELINE"]),
        fits.Column(name="SEG_A", format="J", array=arr["SEG_A"]),
        fits.Column(name="SEG_B", format="J", array=arr["SEG_B"]),
        fits.Column(name="NULL_CHAN", format="J", array=arr["NULL_CHAN"]),
        fits.Column(name="DIFF_PISTON", format="D", array=arr["DIFF_PISTON"]),
        fits.Column(name="OPD", format="D", array=arr["OPD"]),
        fits.Column(name="PISTON_A", format="D", array=arr["PISTON_A"]),
        fits.Column(name="PISTON_B", format="D", array=arr["PISTON_B"]),
        fits.Column(name="FLUX", format="D", array=arr["FLUX"]),
        fits.Column(name="AVG_INDEX", format="J", array=arr["AVG_INDEX"]),
    ])

    primary = fits.PrimaryHDU()
    add_metadata_to_header(primary.header, metadata)
    table_hdu = fits.BinTableHDU.from_columns(cols, name="SCAN_DATA")
    fits.HDUList([primary, table_hdu]).writeto(savefile, overwrite=True)


def save_solution_fits(savefile: str, solution: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    final = solution["final_pistons"]
    segments = np.array(sorted(final.keys()), dtype=np.int32)
    pistons = np.array([final[int(seg)] for seg in segments], dtype=np.float64)

    cols = fits.ColDefs([
        fits.Column(name="SEGMENT", format="J", array=segments),
        fits.Column(name="PISTON", format="D", array=pistons),
    ])
    primary = fits.PrimaryHDU()
    add_metadata_to_header(primary.header, metadata)
    primary.header["GOFFSET"] = float(solution["global_offset"])
    table_hdu = fits.BinTableHDU.from_columns(cols, name="PISTON_SOLUTION")
    fits.HDUList([primary, table_hdu]).writeto(savefile, overwrite=True)


def add_metadata_to_header(header: fits.Header, metadata: Dict[str, Any]) -> None:
    safe_items = {
        "YEAR": YEAR,
        "DATESTR": str(DATE),
        "ITER": ITERATION,
        "NFRAMES": NFRAMES,
        "NUMAVG": NUMAVG,
        "DSTEP": DIFF_PISTON_STEP,
        "APPLY": int(bool(APPLY_FINAL_PISTONS)),
    }
    for key, value in safe_items.items():
        try:
            header[key[:8]] = value
        except Exception:
            pass
    header.add_comment("Differential baseline null scan.")
    header.add_comment("OPD = 2 * DIFF_PISTON for reflective DM.")
    for item in metadata.get("comments", []):
        header.add_comment(str(item))


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(json_safe(payload), f, indent=2, sort_keys=True)



# =============================================================================
# Plotting
# =============================================================================

def plot_intensity_vs_opd(rows: List[Tuple], savepath: str) -> None:
    """Plot integrated null-channel flux versus OPD for each enabled baseline."""
    if len(rows) == 0:
        print("No rows to plot.")
        return

    dtype = [
        ("BASELINE", "U16"),
        ("SEG_A", "i4"),
        ("SEG_B", "i4"),
        ("NULL_CHAN", "i4"),
        ("DIFF_PISTON", "f8"),
        ("OPD", "f8"),
        ("PISTON_A", "f8"),
        ("PISTON_B", "f8"),
        ("FLUX", "f8"),
        ("AVG_INDEX", "i4"),
    ]
    arr = np.array(rows, dtype=dtype)

    fig, ax = plt.subplots(figsize=(9, 5))
    used_names = [str(b["name"]) for b in BASELINES if bool(b.get("use", True))]

    for name in used_names:
        mask = arr["BASELINE"] == name
        if not np.any(mask):
            continue

        opd_vals = np.unique(arr["OPD"][mask])
        opd_vals.sort()
        mean_flux = np.array([
            np.mean(arr["FLUX"][mask & np.isclose(arr["OPD"], opd)])
            for opd in opd_vals
        ])

        ax.plot(opd_vals, mean_flux, marker="o", ms=3, lw=1.5, label=name)
        idx = int(np.nanargmin(mean_flux))
        ax.axvline(opd_vals[idx], ls="--", alpha=0.5)
        ax.scatter([opd_vals[idx]], [mean_flux[idx]], s=70, zorder=5)

    ax.set_xlabel("Differential OPD = 2 × (pA - pB)")
    ax.set_ylabel("Integrated null-channel flux [ADU]")
    ax.set_title("Differential baseline null scans")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plot_file = os.path.join(savepath, f"baseline_nullscan_opd_plot_{ITERATION}.png")
    if SAVE_OPD_PLOT:
        fig.savefig(plot_file, dpi=150)
        print(f"Saved OPD plot: {plot_file}")
    if SHOW_OPD_PLOT:
        plt.show()
    else:
        plt.close(fig)

# =============================================================================
# Optional final application
# =============================================================================

def apply_final_pistons(dm: Any, final_pistons: Dict[int, float]) -> None:
    """Apply solved pistons to injected segments; leave non-injected segments parked."""
    initialise_dm_for_nulling(dm)
    for seg, piston in final_pistons.items():
        set_injected_segment(dm, int(seg), float(piston))
    time.sleep(INITIAL_SETTLE_SECONDS)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    validate_parameters()

    print("\n=== Differential baseline null scan ===")
    print(f"Output: {output_directory()}")
    print(f"Injected tip/tilts: {INJECTED_SEGMENT_TIPTILT}")
    print(f"Piston limits: {PISTON_LIMITS}")
    print(f"Reference pistons: {REFERENCE_PISTONS}")
    print(f"Other segment commands: {OTHER_SEGMENT_COMMANDS}")
    print(f"Baselines: {BASELINES}")
    print(f"APPLY_FINAL_PISTONS = {APPLY_FINAL_PISTONS}")

    if CONFIRM_HARDWARE:
        answer = input("Type YES to start moving the DM and reading APAPANE: ").strip()
        if answer != "YES":
            raise SystemExit("Aborted by user before hardware motion.")

    savepath = output_directory()
    prepare_savepath(savepath)

    dark = getdark(DARK_FILEPATH)
    apapane, dm = open_devices()

    all_rows: List[Tuple] = []
    best_all_avgs: List[Dict[str, Any]] = []

    try:
        for avg_index in range(NUMAVG):
            print(f"\n=== Average repeat {avg_index + 1}/{NUMAVG} ===")
            for baseline in BASELINES:
                if not bool(baseline.get("use", True)):
                    print(f"Skipping disabled baseline {baseline['name']}")
                    continue
                rows, best = scan_baseline(apapane, dm, dark, baseline, avg_index)
                all_rows.extend(rows)
                best_all_avgs.append(best)

        # If NUMAVG > 1, combine by averaging the best differential piston per baseline.
        best_combined = []
        for baseline in BASELINES:
            if not bool(baseline.get("use", True)):
                continue
            name = str(baseline["name"])
            items = [b for b in best_all_avgs if b["baseline"] == name]
            if len(items) == 0:
                raise RuntimeError(f"No best result for baseline {name}")
            d_mean = float(np.mean([item["best_diff_piston"] for item in items]))
            flux_mean = float(np.mean([item["best_flux"] for item in items]))
            combined = dict(items[0])
            combined["best_diff_piston"] = d_mean
            combined["best_opd"] = 2.0 * d_mean
            combined["best_flux"] = flux_mean
            combined["combined_from_numavg"] = len(items)
            best_combined.append(combined)

        solution = solve_relative_pistons(best_combined)

        print("\n=== Best differential pistons ===")
        for item in best_combined:
            print(
                f"{item['baseline']}: p{item['seg_a']}-p{item['seg_b']} = "
                f"{item['best_diff_piston']:.6g}; OPD={item['best_opd']:.6g}"
            )

        print("\n=== Least-squares residuals ===")
        for item in solution["residuals_by_baseline"]:
            print(
                f"{item['baseline']}: measured={item['measured_diff_piston']:.6g}, "
                f"predicted={item['predicted_diff_piston']:.6g}, "
                f"residual={item['residual']:.6g}"
            )

        print("\n=== Recommended final injected-segment pistons ===")
        print(f"Global offset chosen: {solution['global_offset']:.6g}")
        print(f"Allowed global offset interval: {solution['global_offset_interval']}")
        for seg in [11, 20, 31]:
            lo, hi = PISTON_LIMITS[seg]
            print(f"segment {seg}: piston={solution['final_pistons'][seg]:.6g}  limits=({lo}, {hi})")

        metadata = {
            "comments": [
                f"Piston limits: {PISTON_LIMITS}",
                f"Injected tip/tilts: {INJECTED_SEGMENT_TIPTILT}",
                f"Other segment commands: {OTHER_SEGMENT_COMMANDS}",
                f"Reference pistons: {REFERENCE_PISTONS}",
            ]
        }
        scan_fits = os.path.join(savepath, f"baseline_nullscan_table_{ITERATION}.fits")
        solution_fits = os.path.join(savepath, f"baseline_piston_solution_{ITERATION}.fits")
        solution_json = os.path.join(savepath, f"baseline_piston_solution_{ITERATION}.json")

        save_scan_table(scan_fits, all_rows, metadata)
        plot_intensity_vs_opd(all_rows, savepath)
        save_solution_fits(solution_fits, solution, metadata)
        save_json(solution_json, {
            "best_per_average": best_all_avgs,
            "best_combined": best_combined,
            "solution": solution,
            "parameters": {
                "PISTON_LIMITS": PISTON_LIMITS,
                "REFERENCE_PISTONS": REFERENCE_PISTONS,
                "INJECTED_SEGMENT_TIPTILT": INJECTED_SEGMENT_TIPTILT,
                "OTHER_SEGMENT_COMMANDS": OTHER_SEGMENT_COMMANDS,
                "BASELINES": BASELINES,
                "DIFF_PISTON_STEP": DIFF_PISTON_STEP,
                "NFRAMES": NFRAMES,
                "NUMAVG": NUMAVG,
            },
        })

        print(f"\nSaved scan table: {scan_fits}")
        print(f"Saved solution FITS: {solution_fits}")
        print(f"Saved solution JSON: {solution_json}")

        if APPLY_FINAL_PISTONS:
            print("\nApplying final piston solution to injected segments...")
            apply_final_pistons(dm, solution["final_pistons"])
            print("Final piston solution applied.")
        else:
            print("\nAPPLY_FINAL_PISTONS=False, so the final solution was NOT applied.")

    finally:
        # Leave the DM in a benign state consistent with the injection setup:
        # non-injected segments parked, injected segments at reference pistons.
        try:
            if not APPLY_FINAL_PISTONS:
                initialise_dm_for_nulling(dm)
        except Exception as exc:
            print(f"WARNING: failed to restore reference nulling state: {exc}")
