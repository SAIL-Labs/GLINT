#!/usr/bin/env python3
"""Single-fiber GLINT XY rasters at successive pistons, with mount Z fixed.

Based on fiber_focus_scans.py. Edit the settings and SEGMENT_PTT below, then
enter an XY grid, segment ID, absolute surface-piston start/end in um, and
number of piston samples. Both endpoints are included. Each piston gets a
complete XY raster; the chosen segment always has zero tip and tilt, and
all other segments retain their complete SEGMENT_PTT settings.

Mount Z stays at its launch-time readback and is never commanded. Spectra
have NumPy order [x, y, piston, spectral_column]. Only the GLINT dmptt and
mount XY/fixed roll are commanded; camera access is passive frame reading.
Launch-time mount XY/roll and full DM PTT are restored on completion/error.
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


# USER SETTINGS ---------------------------------------------------------------
# Detector bounds [y0, y1, x0, x1], zero-indexed, upper bounds excluded.
SPECTRAL_BOX: tuple[int, int, int, int] | None = (72, 77, 158, 267)
NFRAMES = 5
DARK_PATH: str | None = None
SETTLE_S = 0
PISTON_SETTLE_S = 0.10  # additional delay after each verified piston command
CHECKPOINT_EVERY = 200
DM_KEEPALIVE_INTERVAL_MIN = 15.0
DM_KEEPALIVE_SETTLE_S = 0.10
ADC_MAX_ADU = 3000000.0
SATURATION_FRACTION = 0.80

# (surface piston [um], tip [mrad], tilt [mrad]); IDs are the GUI's 0..36.
# All 37 segments default to (0, -5.0, 0). Edit individual rows as needed.
# The scanned segment's piston is REPLACED by the absolute scan value.
# Its tip/tilt are always forced to zero; all other segments retain their full PTT.
SEGMENT_PTT: dict[int, tuple[float, float, float]] = {
    0: (0.0, -5.0, 0.0),
    1: (0.0, -5.0, 0.0),
    2: (0.0, -5.0, 0.0),
    3: (0.0, -5.0, 0.0),
    4: (0.0, -5.0, 0.0),
    5: (0.0, -5.0, 0.0),
    6: (0.0, -5.0, 0.0),
    7: (0.0, -5.0, 0.0),
    8: (0.0, -5.0, 0.0),
    9: (0.0, -5.0, 0.0),
    10: (0.0, -5.0, 0.0),
    11: (0.0, -5.0, 0.0),
    12: (0.0, -5.0, 0.0),
    13: (0.0, -5.0, 0.0),
    14: (0.0, -5.0, 0.0),
    15: (0.0, -5.0, 0.0),
    16: (0.0, -5.0, 0.0),
    17: (0.0, -5.0, 0.0),
    18: (0.0, -5.0, 0.0),
    19: (0.0, -5.0, 0.0),
    20: (0.0, -5.0, 0.0),
    21: (0.0, -5.0, 0.0),
    22: (0.0, -5.0, 0.0),
    23: (0.0, -5.0, 0.0),
    24: (0.0, -5.0, 0.0),
    25: (0.0, -5.0, 0.0),
    26: (0.0, -5.0, 0.0),
    27: (0.0, -5.0, 0.0),
    28: (0.0, -5.0, 0.0),
    29: (0.0, -5.0, 0.0),
    30: (0.0, -5.0, 0.0),
    31: (0.0, -5.0, 0.0),
    32: (0.0, -5.0, 0.0),
    33: (0.0, -5.0, 0.0),
    34: (0.0, -5.0, 0.0),
    35: (0.0, -5.0, 0.0),
    36: (0.0, -5.0, 0.0),
}

OUTPUT_FOLDER = "Fiber_scan_piston"
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
    if not np.isfinite(float(PISTON_SETTLE_S)) or float(PISTON_SETTLE_S) < 0:
        raise ValueError("PISTON_SETTLE_S must be finite and non-negative")
    return y0, y1, x0, x1



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



def scan_points(x: np.ndarray, y: np.ndarray, pistons: np.ndarray):
    """Repeat XY rasters at each piston, reversing the route on alternate planes."""
    xy_points = tuple(serpentine_points(x, y))
    for ip in range(len(pistons)):
        plane = xy_points if ip % 2 == 0 else reversed(xy_points)
        for iy, ix in plane:
            yield ix, iy, ip



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
    pattern = re.compile(r"^fiber_piston_(\d+)\.(?:json|fits|png)$")
    indices = []
    for path in day.glob("fiber_piston_*.*"):
        match = pattern.match(path.name)
        if match:
            indices.append(int(match.group(1)))
    return day / f"fiber_piston_{max(indices, default=0) + 1}"



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


def piston_axis(start: float, end: float, count: int) -> np.ndarray:
    """Inclusive absolute surface-piston samples; descending sweeps are allowed."""
    if not isinstance(count, (int, np.integer)) or isinstance(count, bool) or count < 1:
        raise ValueError("Number of pistons must be a positive integer")
    start, end = float(start), float(end)
    if not np.all(np.isfinite([start, end])):
        raise ValueError("Piston start and end must be finite surface displacements in um")
    if count == 1 and start != end:
        raise ValueError("For one piston sample, piston start and end must be equal")
    return np.linspace(start, end, int(count))


def prompt_scan_axes() -> tuple[np.ndarray, np.ndarray, float, int, np.ndarray]:
    """Prompt every run, including --check-only; Z is never scanned."""
    print("Enter absolute mount positions in integer pulses (inclusive endpoints).")
    x_start = float(input("X start: ").strip())
    x_end = float(input("X end: ").strip())
    y_start = float(input("Y start: ").strip())
    y_end = float(input("Y end: ").strip())
    xy_step = float(input("XY step: ").strip())
    x = inclusive_axis(x_start, x_end, xy_step, "X")
    y = inclusive_axis(y_start, y_end, xy_step, "Y")
    print("Mount Z stays at its launch-time position; no Z motion is commanded.")
    segment = int(input("Segment to piston [0..36]: ").strip())
    if segment not in range(37):
        raise ValueError("Segment to piston must be in 0..36")
    print("Pistons are absolute SURFACE positions in um, replacing this segment's base piston.")
    start = float(input("Piston start [surface um]: ").strip())
    end = float(input("Piston end [surface um]: ").strip())
    count = int(input("Number of pistons (samples, including both endpoints): ").strip())
    return x, y, xy_step, segment, piston_axis(start, end, count)


def state_for_piston(segment: int, piston_um: float) -> dict[int, SegmentPTT]:
    """Use the scanned piston with zero tip/tilt; retain every other base PTT."""
    if segment not in range(37) or not np.isfinite(piston_um):
        raise ValueError("A finite piston and segment in 0..36 are required")
    state = build_dm_state()
    state[segment] = SegmentPTT(piston_um=float(piston_um), tip_mrad=0.0, tilt_mrad=0.0)
    return state


def write_fits_atomic(
    path: Path, *, payload: dict[str, Any], x: np.ndarray, y: np.ndarray,
    pistons: np.ndarray, fixed_z: float, box: Sequence[int], acquired: np.ndarray,
    actual_x: np.ndarray, actual_y: np.ndarray, actual_z: np.ndarray,
    raw_max: np.ndarray, spectra: np.ndarray, dm_ptt: np.ndarray,
) -> None:
    """Save the piston cube and complete settings, even after an interrupted scan."""
    from astropy.io import fits

    primary = fits.PrimaryHDU()
    header = primary.header
    header["SCANTYPE"] = "FIBER_XY_PISTON"
    header["STATUS"] = payload["status"]
    header["DARKSUB"] = True
    header["NFRAMES"] = int(NFRAMES)
    header["ZSCAN"] = False
    if np.isfinite(fixed_z):
        header["FIXEDZ"] = (float(fixed_z), "Launch-time mount Z [pulses]; never commanded")
    header["PISTSEG"] = payload["piston_scan"]["segment"]
    header["PISTMODE"] = "ABSOLUTE"
    header["PISTUNIT"] = "surface um"
    header["PISTSTRT"] = float(pistons[0])
    header["PISTEND"] = float(pistons[-1])
    header["NPISTON"] = len(pistons)
    header["PISTWAIT"] = float(PISTON_SETTLE_S)
    header["KEEP_MIN"] = float(DM_KEEPALIVE_INTERVAL_MIN)
    header["DARKFILE"] = Path(payload["dark"]["path"]).name
    header["BOXCONV"] = "[y0, y1, x0, x1], upper bounds excluded"
    for key, value in zip(("BOX_Y0", "BOX_Y1", "BOX_X0", "BOX_X1"), box):
        header[key] = int(value)
    spectral = fits.ImageHDU(np.asarray(spectra, dtype=np.float64), name="SPECTRA")
    spectral.header["NPORDER"] = "x,y,piston,spectral_column"
    spectral.header["BUNIT"] = "ADU/pixel/frame"
    spectral.header["REDUCE"] = "mean_frames -> subtract_dark -> mean_rows"
    for index, label in enumerate(("DETECTOR_X", "SURFACE_PISTON", "MOUNT_Y", "MOUNT_X"), 1):
        spectral.header[f"CTYPE{index}"] = label
    hdus = [primary, spectral]
    for name, values, unit in (
        ("X_CMD", x, "pulses"), ("Y_CMD", y, "pulses"),
        ("Z_CMD", np.array([fixed_z]), "pulses"), ("PISTON_CMD", pistons, "um"),
        ("SPECTRAL_X", np.arange(box[2], box[3], dtype=np.int32), "pixel"),
    ):
        hdu = fits.ImageHDU(np.asarray(values), name=name)
        hdu.header["BUNIT"] = unit
        hdus.append(hdu)
    for name, values in (
        ("X_ACTUAL", actual_x), ("Y_ACTUAL", actual_y), ("Z_ACTUAL", actual_z),
        ("ACQUIRED", acquired.astype(np.uint8)), ("RAW_MAX", raw_max),
    ):
        hdu = fits.ImageHDU(np.asarray(values), name=name)
        hdu.header["NPORDER"] = "x,y,piston"
        hdus.append(hdu)
    dm_hdu = fits.ImageHDU(np.asarray(dm_ptt, dtype=np.float64), name="DM_PTT")
    dm_hdu.header["NPORDER"] = "piston,segment,ptt"
    dm_hdu.header["PTTCOLS"] = "piston_um,tip_mrad,tilt_mrad"
    dm_hdu.header.add_comment("Planned full commanded states; ACQUIRED identifies measured samples.")
    hdus.append(dm_hdu)
    encoded = json.dumps(payload, allow_nan=False, ensure_ascii=True)
    hdus.append(fits.BinTableHDU.from_columns([
        fits.Column(name="JSON", format=f"{len(encoded)}A", array=[encoded]),
    ], name="SCAN_SETTINGS"))
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    fits.HDUList(hdus).writeto(temporary, overwrite=True, checksum=True)
    os.replace(temporary, path)


def save_plot(
    path: Path, x: np.ndarray, y: np.ndarray, pistons: np.ndarray,
    fixed_z: float, segment: int, spectra: np.ndarray, box: Sequence[int],
) -> None:
    """Save an integrated XY map for each piston with a common colour scale."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    integrated = np.sum(spectra, axis=-1) * (box[1] - box[0])
    finite = integrated[np.isfinite(integrated)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0., 1.)
    if lo == hi:
        hi = lo + 1.
    norm = Normalize(lo, hi)
    ncols = min(4, len(pistons))
    nrows = int(np.ceil(len(pistons) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(3.8*ncols+.6, 3.4*nrows+.4),
                               squeeze=False, layout="constrained")
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 1.
    dy = float(np.median(np.diff(y))) if len(y) > 1 else 1.
    extent = (x[0]-dx/2, x[-1]+dx/2, y[0]-dy/2, y[-1]+dy/2)
    try:
        for ip, axis in enumerate(axes.flat):
            if ip >= len(pistons):
                axis.set_visible(False)
                continue
            image = axis.imshow(np.ma.masked_invalid(integrated[:, :, ip].T),
                                origin="lower", extent=extent, aspect="equal",
                                interpolation="nearest", cmap="magma", norm=norm)
            axis.set(title=f"Piston = {pistons[ip]:.6g} um", xlabel="Mount X [pulses]",
                     ylabel="Mount Y [pulses]")
        figure.colorbar(image, ax=list(axes.flat)[:len(pistons)], shrink=.85,
                        label="Integrated dark-subtracted ADU / frame")
        figure.suptitle(f"Fiber piston scan: segment {segment}, fixed Z={fixed_z:g} pulses")
        figure.savefig(path, dpi=150, facecolor="white")
    finally:
        plt.close(figure)


def run(args: argparse.Namespace) -> Path | None:
    box = validate_user_settings()
    x, y, xy_step, segment, pistons = prompt_scan_axes()
    base_state = build_dm_state()
    states = [state_for_piston(segment, value) for value in pistons]
    dm_ptt = np.array([
        [[state[s].piston_um, state[s].tip_mrad, state[s].tilt_mrad] for s in range(37)]
        for state in states
    ], dtype=float)
    shape = (len(x), len(y), len(pistons))
    npoints = int(np.prod(shape))
    print("Single-fiber GLINT XY / segment-piston spectral scan")
    print(f"  XY grid          : X={x.tolist()}, Y={y.tolist()}, step={xy_step:g} pulses")
    print("  fixed mount Z    : launch-time controller readback; NO Z commands")
    print(f"  selected segment : {segment}; tip=0.0, tilt=0.0 mrad throughout the scan")
    print(f"  absolute pistons : {pistons.tolist()} surface um (inclusive endpoints)")
    print(f"  points           : {npoints} = {len(x)} X * {len(y)} Y * {len(pistons)} pistons")
    print(f"  spectral box     : {box} = [y0,y1,x0,x1], upper bounds excluded")
    print(f"  NumPy cube shape : {(*shape, box[3]-box[2])} = [x,y,piston,spectral_column]")
    print(f"  frames / point   : {NFRAMES}; mean frames -> subtract dark -> mean rows")
    print(f"  settling         : XY={SETTLE_S:g} s; piston={PISTON_SETTLE_S:g} s")
    print(f"  checkpoint       : every {CHECKPOINT_EVERY} points, plus errors and final state")
    print(f"  GLINT keep-alive : current piston state every {DM_KEEPALIVE_INTERVAL_MIN:g} min")
    print("  final state      : restore launch-time XY/roll and full GLINT DM PTT")
    print("  base SEGMENT_PTT : surface piston [um], tip [mrad], tilt [mrad]")
    for s, values in sorted(SEGMENT_PTT.items()):
        note = " (overridden during scan: swept piston, tip=0.0, tilt=0.0)" if s == segment else ""
        print(f"    {s:2d}: {values}{note}")

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
    json_path, fits_path, png_path = (output_base.with_suffix(s) for s in (".json", ".fits", ".png"))
    acquired = np.zeros(shape, dtype=bool)
    actual_x, actual_y, actual_z, raw_max = (np.full(shape, np.nan) for _ in range(4))
    spectra = np.full((*shape, box[3]-box[2]), np.nan)
    fixed_z = float("nan")
    started = datetime.now(timezone.utc)
    run_clock = time.monotonic()
    timing_totals = {key: 0. for key in (
        "mount_s", "piston_s", "settle_s", "camera_s", "photometry_s",
        "keepalive_s", "checkpoint_s", "restore_s",
    )}
    mount_commands = {"roll": 0, "x": 0, "y": 0, "z": 0}
    payload: dict[str, Any] = {
        "schema_version": 1, "kind": "fiber_piston", "status": "starting",
        "started_utc": started.isoformat(), "instrument_config": str(instrument_path),
        "instrument_config_snapshot": instrument,
        "axes": {"x": x.tolist(), "y": y.tolist(), "z": [None], "piston": pistons.tolist()},
        "axis_units": {"x": "mount pulses", "y": "mount pulses", "z": "mount pulses",
                       "piston": "surface um"},
        "automatic_z_scan": False,
        "cube_axis_order": ["x", "y", "piston", "spectral_column"],
        "cube_shape": list(spectra.shape), "position_array_axis_order": ["x", "y", "piston"],
        "spectral_column_detector_x": list(range(box[2], box[3])),
        "spectral_box_y0_y1_x0_x1": list(box), "box_upper_bounds_excluded": True,
        "spectral_units": "ADU/pixel/frame",
        "missing_value_fits": "NaN; ACQUIRED marks completed positions",
        "piston_scan": {"segment": segment, "mode": "absolute surface piston; replaces base piston",
                        "start_um": float(pistons[0]), "end_um": float(pistons[-1]),
                        "count": len(pistons), "settle_s": float(PISTON_SETTLE_S),
                        "retain_selected_tip_tilt": False,
                        "selected_tip_mrad": 0.0, "selected_tilt_mrad": 0.0,
                        "other_segments": "unchanged from SEGMENT_PTT throughout acquisition"},
        "acquisition": {
            "nframes": int(NFRAMES), "settle_s": float(SETTLE_S),
            "serpentine": True, "reverse_xy_on_alternate_pistons": True,
            "checkpoint_every": int(CHECKPOINT_EVERY), "saved_detector_pixels": False,
            "flux_definition": "mean raw frames, subtract saved dark, then mean box rows per detector column",
            "xy_step_mount_pulses": xy_step,
            "mount_strategy": "fixed roll once; XY raster per piston; Z never commanded",
            "per_point_full_mount_readback": False,
            "position_provenance": "XY verified by move_mount; Z is launch-time readback",
        },
        "base_segment_ptt": dm_state_to_dict(base_state),
        "dm_ptt_axis_order": ["piston", "segment", "ptt"],
        "dm_piston_commands": [],
        "glint_dm_keepalive": {
            "method": "republish current piston state through InstrumentHardware.set_dm / GLINT dmptt",
            "interval_min": float(DM_KEEPALIVE_INTERVAL_MIN),
            "settle_s": float(DM_KEEPALIVE_SETTLE_S), "events": [],
            "ssh_used": False, "scexao_dm_accessed": False,
        },
        "dark": {"path": str(dark_path), "sha256": _sha256(dark_path),
                 "subtracted_after_frame_average": True, "subtracted_before_column_average": True},
        "saturation_guard": {"adc_max_adu": float(ADC_MAX_ADU),
                             "fraction": float(SATURATION_FRACTION),
                             "limit_adu": float(ADC_MAX_ADU)*float(SATURATION_FRACTION)},
        "planned_points": npoints, "acquired_points": 0,
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
    fixed_roll = float(instrument["mount_control"]["fixed_roll"])
    guard_limit = float(ADC_MAX_ADU)*float(SATURATION_FRACTION)

    def checkpoint(status: str) -> None:
        before = time.monotonic()
        payload["status"] = status
        payload["acquired_points"] = int(np.count_nonzero(acquired))
        if keepalive is not None:
            payload["glint_dm_keepalive"]["events"] = list(keepalive.events)
        elapsed = float(time.monotonic()-run_clock)
        payload["timing"] = {
            "elapsed_wall_s": elapsed, "totals_s": dict(timing_totals),
            "mean_per_acquired_point_s": elapsed/payload["acquired_points"] if payload["acquired_points"] else None,
            "mount_axis_command_counts": dict(mount_commands),
        }
        try:
            write_fits_atomic(fits_path, payload=payload, x=x, y=y, pistons=pistons,
                              fixed_z=fixed_z, box=box, acquired=acquired,
                              actual_x=actual_x, actual_y=actual_y, actual_z=actual_z,
                              raw_max=raw_max, spectra=spectra, dm_ptt=dm_ptt)
            _atomic_json(json_path, payload)
        finally:
            timing_totals["checkpoint_s"] += time.monotonic()-before

    def move(targets: dict[str, float]) -> None:
        if not targets:
            return
        before = time.monotonic()
        try:
            assert hardware is not None
            hardware.move_mount(targets, timeout)
            for axis in targets:
                mount_commands[axis] += 1
        finally:
            timing_totals["mount_s"] += time.monotonic()-before

    try:
        hardware = InstrumentHardware(instrument)
        for name, values in (("x", x), ("y", y)):
            lower, upper = map(float, hardware.mount_limits()[name])
            if values[0] < lower or values[-1] > upper:
                raise ValueError(f"{name.upper()} grid exceeds configured limits [{lower:g}, {upper:g}]")
        command_log_start = hardware.command_log_marker()
        pre_mount = hardware.get_mount()
        pre_dm = hardware.get_dm()
        fixed_z = float(pre_mount["z"])
        if not np.isfinite(fixed_z):
            raise ValueError("Launch-time mount Z readback must be finite")
        payload["axes"]["z"] = [fixed_z]
        payload["pre_scan_state"] = {"mount": {k: float(v) for k, v in pre_mount.items()},
                                     "dm": dm_state_to_dict(pre_dm)}
        print(f"  acquisition Z    : fixed at {fixed_z:g} mount pulses")
        if int(round(pre_mount["roll"])) != int(round(fixed_roll)):
            move({"roll": fixed_roll})
        previous_x = previous_y = previous_piston = None
        report_every = max(1, npoints//25)
        for count, (ix, iy, ip) in enumerate(scan_points(x, y, pistons), 1):
            if ip != previous_piston:
                before = time.monotonic()
                hardware.set_dm(states[ip])
                timing_totals["piston_s"] += time.monotonic()-before
                if keepalive is None:
                    keepalive = GLINTDMKeepAlive(hardware, states[ip], float(DM_KEEPALIVE_INTERVAL_MIN)*60)
                else:
                    keepalive.state = states[ip]
                    keepalive.last_refresh_monotonic = time.monotonic()
                payload["dm_piston_commands"].append({
                    "piston_index": ip, "piston_um": float(pistons[ip]),
                    "utc": datetime.now(timezone.utc).isoformat(),
                })
                before = time.monotonic()
                time.sleep(float(PISTON_SETTLE_S))
                timing_totals["settle_s"] += time.monotonic()-before
                print(f"  Piston {ip+1}/{len(pistons)}: segment {segment} = {pistons[ip]:g} surface um")
            before = time.monotonic()
            if keepalive.maybe_refresh():
                keepalive.events[-1].update(piston_index=ip, piston_um=float(pistons[ip]))
                time.sleep(float(DM_KEEPALIVE_SETTLE_S))
            timing_totals["keepalive_s"] += time.monotonic()-before
            target_x, target_y = int(round(x[ix])), int(round(y[iy]))
            targets = {}
            if previous_x is None or target_x != previous_x:
                targets["x"] = float(target_x)
            if previous_y is None or target_y != previous_y:
                targets["y"] = float(target_y)
            # Between pistons the reversed raster starts at the same XY endpoint.
            if count > 1 and len(targets) > 1:
                raise RuntimeError(f"Non-adjacent raster move at point {count}: {targets}")
            move(targets)
            previous_x, previous_y, previous_piston = target_x, target_y, ip
            before = time.monotonic()
            time.sleep(float(SETTLE_S))
            timing_totals["settle_s"] += time.monotonic()-before
            before = time.monotonic()
            frames = _frame_cube(hardware.apapane_frames(int(NFRAMES)))
            timing_totals["camera_s"] += time.monotonic()-before
            if frames.shape != (int(NFRAMES), *dark.shape):
                raise RuntimeError(f"APAPANE frame shape {frames.shape} != {(int(NFRAMES), *dark.shape)}")
            if not np.all(np.isfinite(frames)):
                raise ValueError(f"Non-finite APAPANE frame at X={target_x}, Y={target_y}, piston={pistons[ip]:g}")
            maximum = float(np.max(frames))
            if maximum >= guard_limit:
                raise SaturationDetected(f"Raw APAPANE maximum {maximum:g} ADU reached guard {guard_limit:g} "
                                         f"at X={target_x}, Y={target_y}, piston={pistons[ip]:g} um")
            before = time.monotonic()
            spectra[ix, iy, ip] = dark_subtracted_spectrum(frames, dark, box)
            timing_totals["photometry_s"] += time.monotonic()-before
            actual_x[ix, iy, ip], actual_y[ix, iy, ip], actual_z[ix, iy, ip] = target_x, target_y, fixed_z
            raw_max[ix, iy, ip] = maximum
            acquired[ix, iy, ip] = True
            if count == 1 or count == npoints or count % report_every == 0:
                elapsed = time.monotonic()-run_clock
                eta = elapsed/count*(npoints-count)
                print(f"  [{count}/{npoints}] X={target_x}, Y={target_y}, Z={fixed_z:g}; "
                      f"segment {segment} piston={pistons[ip]:g} um; "
                      f"elapsed={elapsed/60:.1f} min, ETA={eta/60:.1f} min")
            if count % int(CHECKPOINT_EVERY) == 0:
                checkpoint("running")
        checkpoint("acquisition_complete")
    except BaseException as error:
        failure = error
        payload["error"] = f"{type(error).__name__}: {error}"
        try:
            checkpoint("aborted" if isinstance(error, KeyboardInterrupt) else "failed")
        except Exception as save_error:
            payload["checkpoint_error"] = f"{type(save_error).__name__}: {save_error}"
    finally:
        if hardware is not None:
            restore_errors = []
            before = time.monotonic()
            if pre_mount is not None and pre_dm is not None:
                try:
                    hardware.move_mount({axis: float(pre_mount[axis]) for axis in ("roll", "x", "y")}, timeout)
                except BaseException as error:
                    restore_errors.append(f"mount: {type(error).__name__}: {error}")
                try:
                    hardware.set_dm(pre_dm)
                except BaseException as error:
                    restore_errors.append(f"GLINT DM PTT: {type(error).__name__}: {error}")
                payload["restored_pre_scan_state"] = not restore_errors
                if not restore_errors:
                    try:
                        payload["state_after_restore"] = {
                            "mount": {k: float(v) for k, v in hardware.get_mount().items()},
                            "dm": dm_state_to_dict(hardware.get_dm()),
                        }
                    except Exception as error:
                        payload["restore_readback_error"] = f"{type(error).__name__}: {error}"
                try:
                    log = hardware.export_command_log(output_base.parent, command_log_start)
                    if log is not None:
                        payload["command_log"] = log
                except Exception as error:
                    payload["command_log_error"] = f"{type(error).__name__}: {error}"
            timing_totals["restore_s"] += time.monotonic()-before
            try:
                hardware.close()
            except Exception as error:
                restore_errors.append(f"close: {type(error).__name__}: {error}")
            if restore_errors:
                payload["restore_errors"] = restore_errors
                if failure is None:
                    failure = RuntimeError("; ".join(restore_errors))
                    payload["error"] = str(failure)

    if np.any(acquired):
        try:
            save_plot(png_path, x, y, pistons, fixed_z, segment, spectra, box)
        except Exception as error:
            payload["plot_error"] = f"{type(error).__name__}: {error}"
    payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
    payload["duration_s"] = (datetime.fromisoformat(payload["completed_utc"])-started).total_seconds()
    status = "completed" if failure is None else ("aborted" if isinstance(failure, KeyboardInterrupt) else "failed")
    try:
        checkpoint(status)
    except Exception as error:
        payload["checkpoint_error"] = f"{type(error).__name__}: {error}"
        if failure is None:
            failure = error
            payload["status"] = "failed"
        _atomic_json(json_path, payload)
    if failure is not None:
        raise failure
    print(f"\nSaved single-fiber piston scan: {fits_path}")
    return output_base.parent


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
