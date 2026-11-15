#!/usr/bin/env python3
"""Single-fiber GLINT X/Y raster, optionally repeated over mount Z positions.

Set SPECTRAL_BOX below using calibration bounds [y0, y1, x0, x1] (upper
bounds excluded). Enter the X/Y grid and optional Z grid in the terminal.
At each position, average NFRAMES, subtract the saved APAPANE dark, then
average the rows within each detector column of the selected box. The FITS
SPECTRA array has NumPy order [x, y, z, spectral_column], with one Z entry
at the launch-time mount position when automatic Z scanning is disabled.

The GLINT DM is kept awake by periodically republishing the exact commanded
37 x 3 PTT state through the refactor's GLINT ``dmptt`` interface.  No SSH,
SCExAO-DM command, camera write, or camera configuration command is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------

# REQUIRED: set the single fiber's detector box for the current alignment.
# Same convention as the calibration files: [y0, y1, x0, x1], zero-based,
# with exclusive upper bounds. Example syntax: (240, 246, 187, 259).
# No photometric-channel box is selected automatically.
SPECTRAL_BOX: tuple[int, int, int, int] | None = None

# X start/end, Y start/end and XY step are requested in the terminal on every
# run, in absolute integer mount pulses. Endpoints are included. Answer YES
# to the automatic Z scan prompt to also enter Z start/end/step; otherwise
# Z stays at its launch-time position. --yes only skips final confirmation.

# Number of distinct APAPANE frames acquired at every mount position.
NFRAMES = 5

# Optional explicit saved-dark file (.fits, .npy, or .npz). Leave as None to
# use <output_root>/darks/apapane_dark.*, falling back to the newest dated
# APAPANE-dark scan saved by the refactor.
DARK_PATH: str | None = None

# move_mount() already waits for controller motion to finish and verifies the
# commanded position. This short additional delay is only for residual bench
# vibration. Increase it if a repeatability test shows flux transients.
SETTLE_S = 0

# The output arrays are small, but opening and checksumming a FITS file often is
# comparatively expensive. An exception/interrupt always triggers an immediate
# checkpoint independently of this cadence.
CHECKPOINT_EVERY = 200

# GLINT-DM keep-alive.  The script republishes the unchanged PTT state through
# the GLINT dmptt interface at this interval.  It never invokes ssh/sc6 or a
# SCExAO-DM command.
DM_KEEPALIVE_INTERVAL_MIN = 15.0
DM_KEEPALIVE_SETTLE_S = 0.10

# Stop and restore if any raw APAPANE pixel reaches this fraction of the ADC
# ceiling.  This check is performed before dark subtraction.
ADC_MAX_ADU = 3000000.0
SATURATION_FRACTION = 0.80

# Segment PTT order is (surface piston in um, tip in mrad, tilt in mrad).
# Edit any individual row as required.  Segment IDs match the current GUI:
# injected segments are 11, 20, and 31.  All values default to zero.
SEGMENT_PTT: dict[int, tuple[float, float, float]] = {
    0: (0.0, 0.0, 0.0),
    1: (0.0, 0.0, 0.0),
    2: (0.0, 0.0, 0.0),
    3: (0.0, 0.0, 0.0),
    4: (0.0, 0.0, 0.0),
    5: (0.0, 0.0, 0.0),
    6: (0.0, 0.0, 0.0),
    7: (0.0, 0.0, 0.0),
    8: (0.0, 0.0, 0.0),
    9: (0.0, 0.0, 0.0),
    10: (0.0, 0.0, 0.0),
    11: (0.4, 0.0, 0.0),
    12: (0.0, 0.0, 0.0),
    13: (0.0, 0.0, 0.0),
    14: (0.0, 0.0, 0.0),
    15: (0.0, 0.0, 0.0),
    16: (0.0, 0.0, 0.0),
    17: (0.0, 0.0, 0.0),
    18: (0.0, 0.0, 0.0),
    19: (0.0, 0.0, 0.0),
    20: (0.0, 0.0, 0.0),
    21: (0.0, 0.0, 0.0),
    22: (0.0, 0.0, 0.0),
    23: (0.0, 0.0, 0.0),
    24: (0.0, 0.0, 0.0),
    25: (0.0, 0.0, 0.0),
    26: (0.0, 0.0, 0.0),
    27: (0.0, 0.0, 0.0),
    28: (0.0, 0.0, 0.0),
    29: (0.0, 0.0, 0.0),
    30: (0.0, 0.0, 0.0),
    31: (0.0, 0.0, 0.0),
    32: (0.0, 0.0, 0.0),
    33: (0.0, 0.0, 0.0),
    34: (0.0, 0.0, 0.0),
    35: (0.0, 0.0, 0.0),
    36: (0.0, 0.0, 0.0)
}

OUTPUT_FOLDER = "Fiber_scan_focus"


# ---------------------------------------------------------------------------
# GLINT refactor imports
# ---------------------------------------------------------------------------

SCRIPT_DIRECTORY = Path(__file__).resolve().parent


def _find_refactor_root() -> Path:
    configured = os.environ.get("GLINT_ALIGNMENT_REFACTOR_ROOT")
    candidates = [
        Path(configured).expanduser() if configured else None,
        # The intended installation is beside mount_xy_hdr_scan.py.  These
        # candidates mirror that script's location-based discovery order.
        SCRIPT_DIRECTORY,
        SCRIPT_DIRECTORY.parents[1] / "glint_alignment_refactor"
        if len(SCRIPT_DIRECTORY.parents) > 1
        else None,
        SCRIPT_DIRECTORY.parent / "glint_alignment_refactor",
        Path.cwd(),
        Path.cwd() / "glint_alignment_refactor",
    ]
    tried: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        root = candidate.resolve()
        if str(root) in tried:
            continue
        tried.append(str(root))
        if (root / "glint_alignment" / "__init__.py").is_file():
            return root
    raise ModuleNotFoundError(
        "Could not locate glint_alignment_refactor. Tried: "
        + ", ".join(tried)
        + ". Set GLINT_ALIGNMENT_REFACTOR_ROOT to its absolute directory."
    )


GLINT_ALIGNMENT_REFACTOR_ROOT = _find_refactor_root()
if str(GLINT_ALIGNMENT_REFACTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(GLINT_ALIGNMENT_REFACTOR_ROOT))

from glint_alignment.config import load_config
from glint_alignment.instrument import InstrumentHardware
from glint_alignment.models import SegmentPTT, dm_state_to_dict


DEFAULT_INSTRUMENT_CONFIG = GLINT_ALIGNMENT_REFACTOR_ROOT / "config" / "instrument.json"


class SaturationDetected(RuntimeError):
    """Raised when a raw APAPANE frame reaches the configured guard level."""


class GLINTDMKeepAlive:
    """Republish an unchanged GLINT-DM PTT state at a controlled interval."""

    def __init__(
        self,
        hardware: InstrumentHardware,
        state: dict[int, SegmentPTT],
        interval_s: float,
    ) -> None:
        self.hardware = hardware
        self.state = state
        self.interval_s = float(interval_s)
        self.last_refresh_monotonic = time.monotonic()
        self.events: list[dict[str, Any]] = []

    def maybe_refresh(self) -> bool:
        elapsed = time.monotonic() - self.last_refresh_monotonic
        if elapsed < self.interval_s:
            return False
        # InstrumentHardware.set_dm publishes to the configured GLINT dmptt
        # shared memory and verifies the readback.  No other DM is addressed.
        self.hardware.set_dm(self.state)
        now = datetime.now(timezone.utc).isoformat()
        self.events.append({"utc": now, "elapsed_s": float(elapsed)})
        self.last_refresh_monotonic = time.monotonic()
        print(f"  GLINT DM keep-alive: unchanged PTT republished at {now}")
        return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instrument-config",
        type=Path,
        default=DEFAULT_INSTRUMENT_CONFIG,
        help="Active GLINT alignment instrument.json.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate settings and print the complete plan without touching hardware.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the final interactive confirmation.",
    )
    return parser.parse_args(argv)


def inclusive_axis(start: float, stop: float, step: float, name: str) -> np.ndarray:
    start, stop, step = float(start), float(stop), float(step)
    if not all(np.isfinite(value) for value in (start, stop, step)):
        raise ValueError(f"{name} start/stop/step must be finite")
    if step <= 0 or stop < start:
        raise ValueError(f"{name} requires step > 0 and end >= start")
    intervals = (stop - start) / step
    rounded = int(round(intervals))
    if not np.isclose(intervals, rounded, rtol=0, atol=1e-9):
        raise ValueError(f"{name} range must contain an integer number of steps")
    values = start + np.arange(rounded + 1, dtype=float) * step
    if not np.allclose(values, np.rint(values), rtol=0, atol=1e-9):
        raise ValueError(f"{name} scan coordinates must be integer mount pulses")
    if not np.isclose(step, round(step), rtol=0, atol=1e-9):
        raise ValueError(f"{name} step must be an integer number of mount pulses")
    return values


def validate_user_settings() -> tuple[int, int, int, int]:
    if SPECTRAL_BOX is None:
        raise ValueError("Set SPECTRAL_BOX in this file to the fiber's [y0, y1, x0, x1] bounds")
    bounds = np.asarray(SPECTRAL_BOX, dtype=float)
    if (bounds.shape != (4,) or not np.all(np.isfinite(bounds))
            or not np.all(bounds == np.rint(bounds))):
        raise ValueError("SPECTRAL_BOX must contain four finite integer bounds [y0, y1, x0, x1]")
    y0, y1, x0, x1 = map(int, bounds)
    if min(y0, x0) < 0 or y1 <= y0 or x1 <= x0:
        raise ValueError(f"Invalid SPECTRAL_BOX bounds: {SPECTRAL_BOX}")
    if int(NFRAMES) < 1:
        raise ValueError("NFRAMES must be at least 1")
    if float(SETTLE_S) < 0 or float(DM_KEEPALIVE_SETTLE_S) < 0:
        raise ValueError("settling times must be non-negative")
    if int(CHECKPOINT_EVERY) < 1:
        raise ValueError("CHECKPOINT_EVERY must be at least 1")
    if not np.isfinite(float(DM_KEEPALIVE_INTERVAL_MIN)) or float(DM_KEEPALIVE_INTERVAL_MIN) <= 0:
        raise ValueError("DM_KEEPALIVE_INTERVAL_MIN must be positive")
    if float(ADC_MAX_ADU) <= 0 or not 0 < float(SATURATION_FRACTION) < 1:
        raise ValueError("ADC_MAX_ADU must be positive and SATURATION_FRACTION must be in (0, 1)")
    if set(SEGMENT_PTT) != set(range(37)):
        missing = sorted(set(range(37)).difference(SEGMENT_PTT))
        extra = sorted(set(SEGMENT_PTT).difference(range(37)))
        raise ValueError(f"SEGMENT_PTT must contain exactly segments 0..36; missing={missing}, extra={extra}")
    for segment, values in SEGMENT_PTT.items():
        if len(values) != 3 or not np.all(np.isfinite(np.asarray(values, dtype=float))):
            raise ValueError(f"SEGMENT_PTT[{segment}] must contain three finite values")
    return y0, y1, x0, x1


def prompt_scan_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float]:
    """Require an explicit scan grid every run, including --check-only runs."""
    print("Enter absolute mount positions in integer pulses (inclusive endpoints).")
    x_start = float(input("X start: ").strip())
    x_end = float(input("X end: ").strip())
    y_start = float(input("Y start: ").strip())
    y_end = float(input("Y end: ").strip())
    xy_step = float(input("XY step: ").strip())
    x = inclusive_axis(x_start, x_end, xy_step, "X")
    y = inclusive_axis(y_start, y_end, xy_step, "Y")
    while True:
        answer = input("Enable automatic Z scan? YES/NO: ").strip().upper()
        if answer in {"YES", "NO"}:
            break
        print("Enter YES or NO.")
    z = None
    if answer == "YES":
        z_start = float(input("Z start: ").strip())
        z_end = float(input("Z end: ").strip())
        z_step = float(input("Z step: ").strip())
        z = inclusive_axis(z_start, z_end, z_step, "Z")
    return x, y, z, xy_step


def build_dm_state() -> dict[int, SegmentPTT]:
    return {
        segment: SegmentPTT(
            piston_um=float(values[0]),
            tip_mrad=float(values[1]),
            tilt_mrad=float(values[2]),
        )
        for segment, values in SEGMENT_PTT.items()
    }


def serpentine_points(x: np.ndarray, y: np.ndarray):
    for iy in range(len(y)):
        columns = range(len(x)) if iy % 2 == 0 else range(len(x) - 1, -1, -1)
        for ix in columns:
            yield iy, ix


def scan_points(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Repeat XY rasters at each Z, reversing the route on alternate planes."""
    xy_points = tuple(serpentine_points(x, y))
    for iz in range(len(z)):
        plane = xy_points if iz % 2 == 0 else reversed(xy_points)
        for iy, ix in plane:
            yield ix, iy, iz


def _frame_cube(values: np.ndarray) -> np.ndarray:
    frames = np.asarray(values, dtype=float)
    if frames.ndim == 2:
        return frames[None, ...]
    if frames.ndim == 3:
        return frames
    raise ValueError(f"Expected APAPANE frame or frame cube, received shape {frames.shape}")


def _dark_image(values: np.ndarray) -> np.ndarray:
    dark = np.asarray(values, dtype=float)
    if dark.ndim == 2:
        return dark
    if dark.ndim == 3:
        return np.nanmean(dark, axis=0)
    raise ValueError(f"Expected a 2-D dark or 3-D dark cube, received shape {dark.shape}")


def dark_subtracted_spectrum(
    raw_frames: np.ndarray,
    dark: np.ndarray,
    box: Sequence[int],
) -> np.ndarray:
    """Average frames, subtract dark, then average rows within each column."""
    frames = _frame_cube(raw_frames)
    if frames.shape[1:] != dark.shape:
        raise ValueError(f"Frame shape {frames.shape[1:]} differs from dark {dark.shape}")
    y0, y1, x0, x1 = map(int, box)
    if min(y0, x0) < 0 or y1 <= y0 or x1 <= x0 or y1 > dark.shape[0] or x1 > dark.shape[1]:
        raise ValueError(f"Invalid spectral box {box} for dark shape {dark.shape}")
    averaged = np.mean(frames, axis=0, dtype=float)
    corrected = averaged - dark
    spectrum = np.mean(corrected[y0:y1, x0:x1], axis=0, dtype=float)
    if not np.all(np.isfinite(spectrum)):
        raise ValueError("Non-finite dark-subtracted spectrum in selected box")
    return spectrum


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _next_output_base(root: Path) -> Path:
    day = root / datetime.now().astimezone().strftime("%Y%m%d") / OUTPUT_FOLDER
    day.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^fiber_focus_(\d+)\.(?:json|fits|png)$")
    indices = []
    for path in day.glob("fiber_focus_*.*"):
        match = pattern.match(path.name)
        if match:
            indices.append(int(match.group(1)))
    return day / f"fiber_focus_{max(indices, default=0) + 1}"


def _alignment_root(config: dict[str, Any]) -> Path:
    root = Path(config["output_root"]).expanduser()
    if not root.is_absolute():
        root = Path(config["_root"]) / root
    return root.resolve()


def _dark_path(root: Path) -> Path:
    if DARK_PATH is not None:
        explicit = Path(DARK_PATH).expanduser()
        if not explicit.is_absolute():
            explicit = SCRIPT_DIRECTORY / explicit
        explicit = explicit.resolve()
        if not explicit.is_file():
            raise FileNotFoundError(f"Configured DARK_PATH does not exist: {explicit}")
        return explicit

    shared_candidates = [
        root / "darks" / "apapane_dark.fits",
        root / "darks" / "apapane_dark.npy",
        root / "darks" / "apapane_dark.npz",
    ]
    for candidate in shared_candidates:
        if candidate.is_file():
            return candidate

    # Compatibility with refactor versions that save the APAPANE dark as an
    # ordinary dated scan instead of maintaining a shared dark file.
    saved_scan_candidates: list[Path] = []
    for suffix in ("fits", "npy", "npz"):
        saved_scan_candidates.extend(root.glob(f"*/*apapane_dark*/scan_data.{suffix}"))
    if saved_scan_candidates:
        return max(saved_scan_candidates, key=lambda path: path.stat().st_mtime)

    searched = ", ".join(str(path) for path in shared_candidates)
    raise FileNotFoundError(
        "No saved APAPANE dark was found. Set DARK_PATH explicitly or acquire/save "
        f"an APAPANE dark first. Checked {searched} and dated apapane_dark scans below {root}."
    )


def _load_dark(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _dark_image(np.load(path, allow_pickle=False))
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            for key in ("map__dark", "dark"):
                if key in archive:
                    return _dark_image(archive[key])
            raise KeyError(f"No 'map__dark' or 'dark' array in {path}")
    if suffix == ".fits":
        try:
            from astropy.io import fits
        except ImportError as error:
            raise RuntimeError("astropy is required to read a FITS dark") from error
        with fits.open(path, memmap=False) as hdus:
            # Shared files normally use the primary HDU. Refactor scan files
            # identify this array using LONGNAME=map__dark.
            for hdu in hdus:
                long_name = str(hdu.header.get("LONGNAME", "")).lower()
                if long_name in {"map__dark", "dark"} and hdu.data is not None:
                    return _dark_image(hdu.data)
            for hdu in hdus:
                if str(hdu.name).upper() == "DARK" and hdu.data is not None:
                    return _dark_image(hdu.data)
            for hdu in hdus:
                if hdu.data is not None:
                    return _dark_image(hdu.data)
        raise ValueError(f"No image data found in dark FITS file {path}")
    raise ValueError(f"Unsupported dark format {path.suffix!r}; use .fits, .npy, or .npz")


def write_fits_atomic(
    path: Path,
    *,
    status: str,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    box: Sequence[int],
    z_scan: bool,
    acquired: np.ndarray,
    actual_x: np.ndarray,
    actual_y: np.ndarray,
    actual_z: np.ndarray,
    raw_max: np.ndarray,
    spectra: np.ndarray,
    dm_ptt: np.ndarray,
    dark_path: Path,
) -> None:
    try:
        from astropy.io import fits
    except ImportError as error:
        raise RuntimeError("astropy is required to save this scan") from error

    primary = fits.PrimaryHDU()
    primary.header["SCANTYPE"] = "FIBER_XYZ"
    primary.header["STATUS"] = status
    primary.header["DARKSUB"] = True
    primary.header["NFRAMES"] = int(NFRAMES)
    primary.header["KEEP_MIN"] = float(DM_KEEPALIVE_INTERVAL_MIN)
    primary.header["DARKFILE"] = dark_path.name
    primary.header["ZSCAN"] = z_scan
    primary.header["BOXCONV"] = "[y0, y1, x0, x1], upper bounds excluded"
    for key, value in zip(("BOX_Y0", "BOX_Y1", "BOX_X0", "BOX_X1"), box):
        primary.header[key] = int(value)
    spectral_hdu = fits.ImageHDU(np.asarray(spectra, dtype=np.float64), name="SPECTRA")
    spectral_hdu.header["NPORDER"] = "x,y,z,spectral_column"
    spectral_hdu.header["BUNIT"] = "ADU/pixel/frame"
    spectral_hdu.header["REDUCE"] = "mean_frames -> subtract_dark -> mean_rows"
    # FITS dimension order is the reverse of the array order exposed by NumPy.
    for index, label in enumerate(("DETECTOR_X", "MOUNT_Z", "MOUNT_Y", "MOUNT_X"), start=1):
        spectral_hdu.header[f"CTYPE{index}"] = label
    spectral_hdu.header.add_comment("Coordinate vectors: SPECTRAL_X, Z_CMD, Y_CMD, X_CMD.")
    hdus = [
        primary,
        spectral_hdu,
        fits.ImageHDU(np.asarray(x, dtype=np.float64), name="X_CMD"),
        fits.ImageHDU(np.asarray(y, dtype=np.float64), name="Y_CMD"),
        fits.ImageHDU(np.asarray(z, dtype=np.float64), name="Z_CMD"),
        fits.ImageHDU(np.arange(box[2], box[3], dtype=np.int32), name="SPECTRAL_X"),
        fits.ImageHDU(np.asarray(actual_x, dtype=np.float64), name="X_ACTUAL"),
        fits.ImageHDU(np.asarray(actual_y, dtype=np.float64), name="Y_ACTUAL"),
        fits.ImageHDU(np.asarray(actual_z, dtype=np.float64), name="Z_ACTUAL"),
        fits.ImageHDU(np.asarray(acquired, dtype=np.uint8), name="ACQUIRED"),
        fits.ImageHDU(np.asarray(raw_max, dtype=np.float32), name="RAW_MAX"),
        fits.ImageHDU(np.asarray(dm_ptt, dtype=np.float64), name="DM_PTT"),
    ]
    hdus[-1].header["PTTCOLS"] = "piston_um,tip_mrad,tilt_mrad"
    for hdu in hdus:
        if hdu.name in {"X_ACTUAL", "Y_ACTUAL", "Z_ACTUAL", "ACQUIRED", "RAW_MAX"}:
            hdu.header["NPORDER"] = "x,y,z"
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    fits.HDUList(hdus).writeto(temporary, overwrite=True, checksum=True)
    os.replace(temporary, path)


def save_plot(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    spectra: np.ndarray,
    box: Sequence[int],
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2 if len(z) > 1 else 1, figsize=(10.4 if len(z) > 1 else 5.8, 4.6), squeeze=False)
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    dy = float(np.median(np.diff(y))) if len(y) > 1 else 1.0
    extent = (x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2)
    # Multiply the saved column means by box height to display integrated flux.
    integrated = np.sum(spectra, axis=-1) * (box[1] - box[0])
    valid = np.isfinite(integrated)
    best_xy = np.max(np.where(valid, integrated, -np.inf), axis=2)
    best_xy[~np.any(valid, axis=2)] = np.nan
    axis = axes[0, 0]
    image = axis.imshow(best_xy.T, origin="lower", extent=extent, aspect="equal",
                        interpolation="nearest", cmap="magma")
    axis.set_title("Maximum integrated flux over Z" if len(z) > 1 else f"Integrated flux at Z={z[0]:g}")
    axis.set_xlabel("Mount X [pulses]")
    axis.set_ylabel("Mount Y [pulses]")
    figure.colorbar(image, ax=axis, label="dark-subtracted ADU / frame")
    if len(z) > 1:
        best_z = np.max(np.where(valid, integrated, -np.inf), axis=(0, 1))
        best_z[~np.any(valid, axis=(0, 1))] = np.nan
        axes[0, 1].plot(z, best_z, marker="o")
        axes[0, 1].set_xlabel("Mount Z [pulses]")
        axes[0, 1].set_ylabel("maximum XY integrated flux [ADU / frame]")
        axes[0, 1].set_title("Fiber focus curve")
    figure.suptitle(f"Single-fiber scan: {int(NFRAMES)} frames per position")
    figure.tight_layout()
    figure.savefig(path, dpi=170, facecolor="white")
    plt.close(figure)


def show_fixed_z_diagnostic(
    x: np.ndarray,
    y: np.ndarray,
    z: float,
    spectra: np.ndarray,
    acquired: np.ndarray,
) -> None:
    """Display the measured XY points with log-coloured spectral sums; do not save."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    integrated = np.sum(spectra[:, :, 0, :], axis=-1)
    measured = acquired[:, :, 0] & np.isfinite(integrated)
    positive = measured & (integrated > 0)
    nonpositive = measured & (integrated <= 0)

    figure, axis = plt.subplots(figsize=(7.2, 6.0))
    try:
        if np.any(positive):
            values = integrated[positive]
            vmin, vmax = float(np.min(values)), float(np.max(values))
            if vmin == vmax:
                vmin, vmax = vmin / 2, vmax * 2
            points = axis.scatter(
                grid_x[positive], grid_y[positive], c=values,
                norm=LogNorm(vmin=vmin, vmax=vmax), cmap="magma",
                s=85, edgecolors="0.3", linewidths=0.4, zorder=3,
            )
            figure.colorbar(points, ax=axis, label="Integrated spectral flux (sum of vector; log scale)")
        else:
            axis.text(0.5, 1.02, "No positive flux for a logarithmic colour scale",
                      transform=axis.transAxes, ha="center", fontsize=9)
        if np.any(nonpositive):
            # Dark-subtracted values may be zero/negative; keep these visible
            # without clipping them to an artificial positive flux.
            axis.scatter(grid_x[nonpositive], grid_y[nonpositive], color="0.55",
                         marker="x", s=65, label="Flux <= 0 (outside log scale)", zorder=3)
            axis.legend(loc="best")
        dx = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
        dy = float(np.median(np.diff(y))) if len(y) > 1 else 1.0
        axis.set_xlim(x[0] - dx / 2, x[-1] + dx / 2)
        axis.set_ylim(y[0] - dy / 2, y[-1] + dy / 2)
        axis.set_xticks(x, minor=True)
        axis.set_yticks(y, minor=True)
        axis.grid(which="both", alpha=0.25)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Mount X [pulses]")
        axis.set_ylabel("Mount Y [pulses]")
        axis.set_title(f"Fiber alignment diagnostic: Z={z:g}, {int(NFRAMES)} frames per point")
        figure.tight_layout()
        print("Displaying fixed-Z alignment diagnostic (not saved). Close the plot window to finish.")
        plt.show(block=True)
    finally:
        plt.close(figure)


def _confirm() -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("Interactive confirmation required; rerun with --yes only after reviewing the plan")
    answer = input("Apply the listed GLINT-DM PTT state and start the physical fiber scan? Type YES: ").strip()
    if answer != "YES":
        raise RuntimeError("Confirmation not received")


def _require_fits_support() -> None:
    try:
        from astropy.io import fits as _fits
    except ImportError as error:
        raise RuntimeError(
            "astropy is required for this scan's checkpoint FITS files; install the "
            "glint_alignment_refactor requirements before touching hardware"
        ) from error
    del _fits


def run(args: argparse.Namespace) -> Path | None:
    box = validate_user_settings()
    x, y, entered_z, xy_step = prompt_scan_axes()
    z_scan = entered_z is not None
    # With no Z scan, fill the singleton axis from controller readback before
    # acquisition. A failed startup is recorded as null in JSON / NaN in FITS.
    z = entered_z if entered_z is not None else np.full(1, np.nan)
    dm_state = build_dm_state()
    npoints = len(x) * len(y) * len(z)
    keepalive_interval_s = float(DM_KEEPALIVE_INTERVAL_MIN) * 60.0
    print("Single-fiber GLINT X/Y/Z spectral scan")
    print(f"  grid             : X={x[0]:g}..{x[-1]:g}, Y={y[0]:g}..{y[-1]:g}, step={xy_step:g}")
    print(f"  Z positions      : {z.tolist() if z_scan else 'current mount Z, read at startup; no Z motion'}")
    print(f"  points           : {npoints} ({len(x)} x {len(y)} x {len(z)})")
    print(f"  spectral box     : {box} = [y0, y1, x0, x1], upper bounds excluded")
    print(f"  NumPy cube shape : {(len(x), len(y), len(z), box[3] - box[2])} = [x, y, z, spectral_column]")
    print(f"  frames / point   : {int(NFRAMES)}")
    print(f"  extra settling   : {float(SETTLE_S):g} s after verified motion")
    print(f"  checkpoint       : every {int(CHECKPOINT_EVERY)} points")
    print("  mount commands   : fixed roll once; XY serpentine; Z changes between planes if enabled")
    print("  saved values     : one 1D spectrum per position; mean frames -> subtract dark -> mean rows")
    print(f"  GLINT keep-alive : unchanged PTT every {DM_KEEPALIVE_INTERVAL_MIN:g} min")
    print("  other DM access  : none (no SSH and no SCExAO-DM command)")
    print("  final state      : launch-time mount and GLINT-DM PTT are restored")

    instrument_path = args.instrument_config.expanduser().resolve()
    instrument = load_config(instrument_path, simulation=False)
    if args.check_only:
        return None
    _require_fits_support()
    if not args.yes:
        _confirm()

    output_root = _alignment_root(instrument)
    dark_path = _dark_path(output_root)
    dark = _load_dark(dark_path)
    if not np.all(np.isfinite(dark)):
        raise ValueError(f"Non-finite saved dark in {dark_path}")
    if box[1] > dark.shape[0] or box[3] > dark.shape[1]:
        raise ValueError(f"Spectral box {box} exceeds dark shape {dark.shape}")

    output_base = _next_output_base(output_root)
    json_path = output_base.with_suffix(".json")
    fits_path = output_base.with_suffix(".fits")
    png_path = output_base.with_suffix(".png")
    shape = (len(x), len(y), len(z))
    acquired = np.zeros(shape, dtype=bool)
    actual_x = np.full(shape, np.nan)
    actual_y = np.full(shape, np.nan)
    actual_z = np.full(shape, np.nan)
    raw_max = np.full(shape, np.nan)
    spectra = np.full((*shape, box[3] - box[2]), np.nan)
    dm_ptt_array = np.asarray(
        [
            [dm_state[segment].piston_um, dm_state[segment].tip_mrad, dm_state[segment].tilt_mrad]
            for segment in range(37)
        ],
        dtype=float,
    )
    started = datetime.now(timezone.utc)
    run_started_monotonic = time.monotonic()
    timing_totals = {
        "mount_s": 0.0,
        "settle_s": 0.0,
        "camera_s": 0.0,
        "photometry_s": 0.0,
        "keepalive_s": 0.0,
        "checkpoint_s": 0.0,
        "restore_s": 0.0,
    }
    mount_axis_commands = {"roll": 0, "x": 0, "y": 0, "z": 0}
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "fiber_focus",
        "status": "starting",
        "started_utc": started.isoformat(),
        "instrument_config": str(instrument_path),
        "axes": {"x": x.tolist(), "y": y.tolist(), "z": z.tolist() if z_scan else [None]},
        "axis_units": {"x": "mount pulses", "y": "mount pulses", "z": "mount pulses"},
        "automatic_z_scan": z_scan,
        "cube_axis_order": ["x", "y", "z", "spectral_column"],
        "cube_shape": list(spectra.shape),
        "position_array_axis_order": ["x", "y", "z"],
        "spectral_column_detector_x": list(range(box[2], box[3])),
        "spectral_box_y0_y1_x0_x1": list(box),
        "box_upper_bounds_excluded": True,
        "spectral_units": "ADU/pixel/frame",
        "missing_value_fits": "NaN; ACQUIRED marks completed positions",
        "acquisition": {
            "nframes": int(NFRAMES),
            "settle_s": float(SETTLE_S),
            "serpentine": True,
            "checkpoint_every": int(CHECKPOINT_EVERY),
            "saved_detector_pixels": False,
            "flux_definition": (
                "mean raw frames along frame axis, subtract saved dark image, "
                "then mean box rows within each detector X column"
            ),
            "xy_step_mount_pulses": xy_step,
            "mount_strategy": "fixed roll once; XY serpentine, reversed on alternate Z planes; command changing axes only",
            "per_point_full_mount_readback": False,
        },
        "commanded_dm_ptt": dm_state_to_dict(dm_state),
        "glint_dm_keepalive": {
            "method": "republish unchanged state through InstrumentHardware.set_dm / GLINT dmptt",
            "interval_min": float(DM_KEEPALIVE_INTERVAL_MIN),
            "events": [],
            "ssh_used": False,
            "scexao_dm_accessed": False,
        },
        "dark": {"path": str(dark_path), "sha256": _sha256(dark_path),
                 "subtracted_after_frame_average": True, "subtracted_before_column_average": True},
        "saturation_guard": {
            "adc_max_adu": float(ADC_MAX_ADU),
            "fraction": float(SATURATION_FRACTION),
            "limit_adu": float(ADC_MAX_ADU) * float(SATURATION_FRACTION),
        },
        "planned_points": npoints,
        "acquired_points": 0,
        "timing": {},
        "outputs": {"json": json_path.name, "fits": fits_path.name, "figure": png_path.name},
    }
    _atomic_json(json_path, payload)

    hardware: InstrumentHardware | None = None
    pre_mount: dict[str, float] | None = None
    pre_dm: dict[int, SegmentPTT] | None = None
    keepalive: GLINTDMKeepAlive | None = None
    failure: BaseException | None = None
    command_log_start = 0
    timeout = float(instrument["safety"]["motion_timeout_s"])
    guard_limit = float(ADC_MAX_ADU) * float(SATURATION_FRACTION)
    fixed_roll = float(instrument["mount_control"]["fixed_roll"])

    def timing_summary() -> dict[str, Any]:
        acquired_count = int(np.count_nonzero(acquired))
        elapsed = float(time.monotonic() - run_started_monotonic)
        return {
            "elapsed_wall_s": elapsed,
            "totals_s": {key: float(value) for key, value in timing_totals.items()},
            "mean_per_acquired_point_s": (
                None if acquired_count == 0 else elapsed / float(acquired_count)
            ),
            "mount_axis_command_counts": dict(mount_axis_commands),
        }

    def timed_mount_move(targets: dict[str, float]) -> None:
        if not targets:
            return
        move_started = time.monotonic()
        try:
            assert hardware is not None
            hardware.move_mount(targets, timeout)
        finally:
            timing_totals["mount_s"] += time.monotonic() - move_started
        for axis in targets:
            if axis in mount_axis_commands:
                mount_axis_commands[axis] += 1

    def checkpoint(status: str) -> None:
        checkpoint_started = time.monotonic()
        try:
            write_fits_atomic(
                fits_path,
                status=status,
                x=x,
                y=y,
                z=z,
                box=box,
                z_scan=z_scan,
                acquired=acquired,
                actual_x=actual_x,
                actual_y=actual_y,
                actual_z=actual_z,
                raw_max=raw_max,
                spectra=spectra,
                dm_ptt=dm_ptt_array,
                dark_path=dark_path,
            )
            payload["status"] = status
            payload["acquired_points"] = int(np.count_nonzero(acquired))
            if keepalive is not None:
                payload["glint_dm_keepalive"]["events"] = list(keepalive.events)
            payload["timing"] = timing_summary()
            _atomic_json(json_path, payload)
        finally:
            timing_totals["checkpoint_s"] += time.monotonic() - checkpoint_started

    try:
        hardware = InstrumentHardware(instrument)
        limits = hardware.mount_limits()
        limit_axes = {"x": x, "y": y}
        if z_scan:
            limit_axes["z"] = z
        for name, values in limit_axes.items():
            lower, upper = map(float, limits[name])
            if float(values[0]) < lower or float(values[-1]) > upper:
                raise ValueError(
                    f"{name.upper()} grid {values[0]:g}..{values[-1]:g} exceeds configured limits "
                    f"[{lower:g}, {upper:g}]"
                )
        command_log_start = hardware.command_log_marker()
        # Query and restore only the two devices this script changes. Neither
        # steering mirror is queried or commanded by this standalone scan.
        pre_mount = hardware.get_mount()
        pre_dm = hardware.get_dm()
        if not z_scan:
            z[0] = float(pre_mount["z"])
            payload["axes"]["z"] = z.tolist()
        print(f"  acquisition Z    : {z.tolist()} mount pulses")
        payload["pre_scan_state"] = {
            "mount": {key: float(value) for key, value in pre_mount.items()},
            "dm": dm_state_to_dict(pre_dm),
        }
        hardware.set_dm(dm_state)
        keepalive = GLINTDMKeepAlive(hardware, dm_state, keepalive_interval_s)

        # Roll is a scan invariant. Command it at most once, rather than adding
        # a redundant serial command and readback to every raster position.
        if int(round(float(pre_mount["roll"]))) != int(round(fixed_roll)):
            timed_mount_move({"roll": fixed_roll})
        payload["status"] = "running"
        _atomic_json(json_path, payload)

        report_every = max(1, npoints // 25)
        acquisition_started_monotonic = time.monotonic()
        previous_x: int | None = None
        previous_y: int | None = None
        previous_z: int | None = None
        for count, (ix, iy, iz) in enumerate(scan_points(x, y, z), start=1):
            keepalive_started = time.monotonic()
            refreshed = keepalive.maybe_refresh()
            if refreshed:
                time.sleep(float(DM_KEEPALIVE_SETTLE_S))
            timing_totals["keepalive_s"] += time.monotonic() - keepalive_started

            target_x = int(round(float(x[ix])))
            target_y = int(round(float(y[iy])))
            target_z = int(round(float(z[iz])))
            move_targets: dict[str, float] = {}
            if previous_x is None or target_x != previous_x:
                move_targets["x"] = float(target_x)
            if previous_y is None or target_y != previous_y:
                move_targets["y"] = float(target_y)
            if z_scan and (previous_z is None or target_z != previous_z):
                move_targets["z"] = float(target_z)
            # Alternate-plane reversal keeps XY fixed at each Z transition.
            # Every point after the first changes exactly one raster axis.
            if count > 1 and len(move_targets) != 1:
                raise RuntimeError(
                    f"Non-adjacent serpentine move at point {count}: targets={move_targets}"
                )
            timed_mount_move(move_targets)
            previous_x, previous_y, previous_z = target_x, target_y, target_z

            settle_started = time.monotonic()
            time.sleep(float(SETTLE_S))
            timing_totals["settle_s"] += time.monotonic() - settle_started

            camera_started = time.monotonic()
            frames = _frame_cube(hardware.apapane_frames(int(NFRAMES)))
            timing_totals["camera_s"] += time.monotonic() - camera_started
            if frames.shape[0] != int(NFRAMES):
                raise RuntimeError(
                    f"Requested {int(NFRAMES)} APAPANE frames but received {frames.shape[0]}"
                )
            if frames.shape[1:] != dark.shape:
                raise RuntimeError(
                    f"APAPANE frame shape {frames.shape[1:]} does not match dark shape {dark.shape}"
                )
            if not np.all(np.isfinite(frames)):
                raise ValueError(f"Non-finite APAPANE frame at X={target_x}, Y={target_y}, Z={target_z}")
            point_raw_max = float(np.max(frames))
            if point_raw_max >= guard_limit:
                raise SaturationDetected(
                    f"Raw APAPANE maximum {point_raw_max:.1f} ADU reached the "
                    f"{guard_limit:.1f} ADU guard at X={x[ix]:g}, Y={y[iy]:g}, Z={z[iz]:g}"
                )
            photometry_started = time.monotonic()
            spectra[ix, iy, iz, :] = dark_subtracted_spectrum(frames, dark, box)
            timing_totals["photometry_s"] += time.monotonic() - photometry_started

            # move_mount() has already waited for exact controller readback.
            # Recording those verified integer targets avoids six additional
            # serial position queries at every point.
            actual_x[ix, iy, iz] = float(target_x)
            actual_y[ix, iy, iz] = float(target_y)
            actual_z[ix, iy, iz] = float(target_z)
            raw_max[ix, iy, iz] = point_raw_max
            acquired[ix, iy, iz] = True
            if count == 1 or count == npoints or count % report_every == 0:
                value = float(np.mean(spectra[ix, iy, iz, :]))
                elapsed = time.monotonic() - acquisition_started_monotonic
                eta_s = elapsed / float(count) * float(npoints - count)
                print(
                    f"  [{count:5d}/{npoints}] X={target_x:g}, Y={target_y:g}, Z={target_z:g}; "
                    f"mean spectral flux={value:.6g}; elapsed={elapsed / 60:.1f} min, ETA={eta_s / 60:.1f} min"
                )
            if count % int(CHECKPOINT_EVERY) == 0:
                checkpoint("running")
        checkpoint("acquisition_complete")
    except BaseException as error:
        failure = error
        payload["status"] = "aborted" if isinstance(error, KeyboardInterrupt) else "failed"
        payload["error"] = f"{type(error).__name__}: {error}"
        try:
            checkpoint(payload["status"])
        except Exception as checkpoint_error:
            payload["checkpoint_error"] = f"{type(checkpoint_error).__name__}: {checkpoint_error}"
            _atomic_json(json_path, payload)
    finally:
        if hardware is not None and pre_mount is not None and pre_dm is not None:
            restore_errors: list[str] = []
            restore_started = time.monotonic()
            try:
                restore_targets = {axis: float(pre_mount[axis]) for axis in ("roll", "x", "y")}
                if z_scan:
                    restore_targets["z"] = float(pre_mount["z"])
                hardware.move_mount(
                    restore_targets,
                    timeout,
                )
            except BaseException as error:
                restore_errors.append(f"mount: {type(error).__name__}: {error}")
            try:
                hardware.set_dm(pre_dm)
            except BaseException as error:
                restore_errors.append(f"GLINT DM PTT: {type(error).__name__}: {error}")
            timing_totals["restore_s"] += time.monotonic() - restore_started
            payload["restored_pre_scan_state"] = not restore_errors
            if restore_errors:
                payload["restore_errors"] = restore_errors
                if failure is None:
                    failure = RuntimeError("; ".join(restore_errors))
            else:
                payload["state_after_restore"] = {
                    "mount": {key: float(value) for key, value in hardware.get_mount().items()},
                    "dm": dm_state_to_dict(hardware.get_dm()),
                }
            try:
                command_log = hardware.export_command_log(output_base.parent, command_log_start)
                if command_log is not None:
                    payload["command_log"] = command_log
            except Exception as error:
                payload["command_log_error"] = f"{type(error).__name__}: {error}"
        if hardware is not None:
            hardware.close()

    try:
        if np.any(acquired):
            save_plot(png_path, x, y, z, spectra, box)
    except Exception as error:
        payload["plot_error"] = f"{type(error).__name__}: {error}"
    payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
    payload["duration_s"] = (
        datetime.fromisoformat(payload["completed_utc"]) - started
    ).total_seconds()
    if keepalive is not None:
        payload["glint_dm_keepalive"]["events"] = list(keepalive.events)
    payload["timing"] = timing_summary()
    if failure is None:
        payload["status"] = "completed"
    _atomic_json(json_path, payload)
    if failure is not None:
        raise failure
    print(f"\nSaved single-fiber spectral scan: {fits_path}")
    if not z_scan:
        # Display only after saving data, restoring the mount/DM and closing
        # hardware, so the interactive window cannot delay restoration.
        try:
            show_fixed_z_diagnostic(x, y, float(z[0]), spectra, acquired)
        except Exception as error:
            print(f"WARNING: alignment diagnostic could not be displayed: {error}", file=sys.stderr)
    return output_base.parent


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
