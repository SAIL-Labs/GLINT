#!/usr/bin/env python3
"""Extended, unmasked GLINT mount X/Y raster with configurable DM PTT.

This is a normal raster: every X/Y grid point is acquired in a serpentine
order.  It passively reads APAPANE, subtracts the saved APAPANE dark from each
individual frame, then saves only the integrated Phot 1/2/3 flux statistics.

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

# Absolute GLINT mount positions.  The defaults are a 300 x 300 extension of
# the latest normal 06_xy scan (3000..3100, 3420..3520), centred near the same
# measured injection location.  All endpoints are included.
X_START = 2700.0
X_STOP = 3300.0
Y_START = 3200.0
Y_STOP = 3700.0
XY_STEP = 5.0

# Number of distinct APAPANE frames acquired at every mount position.
NFRAMES = 5

# Optional explicit saved-dark file (.fits, .npy, or .npz). Leave as None to
# use <output_root>/darks/apapane_dark.*, falling back to the newest dated
# APAPANE-dark scan saved by the refactor.
DARK_PATH: str | None = None

# move_mount() already waits for controller motion to finish and verifies the
# commanded position. This short additional delay is only for residual bench
# vibration. Increase it if a repeatability test shows flux transients.
SETTLE_S = 0.02

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

OUTPUT_FOLDER = "Alignment_output_individual"


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

from glint_alignment.boxes import PHOT_BOXES
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
    if step <= 0 or stop <= start:
        raise ValueError(f"{name} requires step > 0 and stop > start")
    intervals = (stop - start) / step
    rounded = int(round(intervals))
    if not np.isclose(intervals, rounded, atol=1e-9):
        raise ValueError(f"{name} range must contain an integer number of steps")
    if rounded < 1:
        raise ValueError(f"{name} must contain at least two points")
    return start + np.arange(rounded + 1, dtype=float) * step


def validate_user_settings() -> tuple[np.ndarray, np.ndarray]:
    x = inclusive_axis(X_START, X_STOP, XY_STEP, "X")
    y = inclusive_axis(Y_START, Y_STOP, XY_STEP, "Y")
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
    # The Suruga adapter commands integer pulses. Reject sub-pulse grids rather
    # than silently rounding multiple requested points onto the same position.
    if not np.allclose(x, np.rint(x), atol=1e-9) or not np.allclose(y, np.rint(y), atol=1e-9):
        raise ValueError("X/Y scan coordinates must be integer mount pulses")
    return x, y


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


def dark_subtracted_flux_statistics(
    raw_frames: np.ndarray,
    box: Sequence[int],
    dark_box_sum: float,
) -> tuple[float, float, float]:
    y0, y1, x0, x1 = map(int, box)
    # This is exactly equivalent to subtracting the complete dark image from
    # every frame and then integrating, while touching only the pixels retained
    # in the output photometric channels.
    per_frame = (
        np.asarray(raw_frames[:, y0:y1, x0:x1], dtype=float).sum(axis=(1, 2))
        - float(dark_box_sum)
    )
    mean = float(np.nanmean(per_frame))
    std = float(np.nanstd(per_frame, ddof=1)) if len(per_frame) > 1 else 0.0
    sem = std / float(np.sqrt(len(per_frame))) if len(per_frame) > 1 else 0.0
    return mean, std, sem


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
    pattern = re.compile(r"^normal_xy_extended_(\d+)\.(?:json|fits|png)$")
    indices = []
    for path in day.glob("normal_xy_extended_*.*"):
        match = pattern.match(path.name)
        if match:
            indices.append(int(match.group(1)))
    return day / f"normal_xy_extended_{max(indices, default=0) + 1}"


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


def _fits_name(channel: str, suffix: str = "") -> str:
    compact = re.sub(r"[^A-Za-z0-9]", "", channel).upper()
    return f"{compact}{suffix}"[:68]


def write_fits_atomic(
    path: Path,
    *,
    status: str,
    x: np.ndarray,
    y: np.ndarray,
    acquired: np.ndarray,
    actual_x: np.ndarray,
    actual_y: np.ndarray,
    raw_max: np.ndarray,
    flux_mean: dict[str, np.ndarray],
    flux_std: dict[str, np.ndarray],
    flux_sem: dict[str, np.ndarray],
    dm_ptt: np.ndarray,
    dark_path: Path,
) -> None:
    try:
        from astropy.io import fits
    except ImportError as error:
        raise RuntimeError("astropy is required to save this scan") from error

    primary = fits.PrimaryHDU()
    primary.header["SCANTYPE"] = "NORM_XY"
    primary.header["STATUS"] = status
    primary.header["DARKSUB"] = True
    primary.header["NFRAMES"] = int(NFRAMES)
    primary.header["KEEP_MIN"] = float(DM_KEEPALIVE_INTERVAL_MIN)
    primary.header["DARKFILE"] = dark_path.name
    hdus = [
        primary,
        fits.ImageHDU(np.asarray(x, dtype=np.float64), name="X_CMD"),
        fits.ImageHDU(np.asarray(y, dtype=np.float64), name="Y_CMD"),
        fits.ImageHDU(np.asarray(actual_x, dtype=np.float64), name="X_ACTUAL"),
        fits.ImageHDU(np.asarray(actual_y, dtype=np.float64), name="Y_ACTUAL"),
        fits.ImageHDU(np.asarray(acquired, dtype=np.uint8), name="ACQUIRED"),
        fits.ImageHDU(np.asarray(raw_max, dtype=np.float32), name="RAW_MAX"),
        fits.ImageHDU(np.asarray(dm_ptt, dtype=np.float64), name="DM_PTT"),
    ]
    hdus[-1].header["PTTCOLS"] = "piston_um,tip_mrad,tilt_mrad"
    for channel in PHOT_BOXES:
        hdus.extend(
            [
                fits.ImageHDU(np.asarray(flux_mean[channel], dtype=np.float64), name=_fits_name(channel)),
                fits.ImageHDU(np.asarray(flux_std[channel], dtype=np.float64), name=_fits_name(channel, "STD")),
                fits.ImageHDU(np.asarray(flux_sem[channel], dtype=np.float64), name=_fits_name(channel, "SEM")),
            ]
        )
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    fits.HDUList(hdus).writeto(temporary, overwrite=True, checksum=True)
    os.replace(temporary, path)


def save_plot(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    flux_mean: dict[str, np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    channels = list(PHOT_BOXES)
    figure, axes = plt.subplots(1, len(channels), figsize=(5.2 * len(channels), 4.6), squeeze=False)
    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    extent = (x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2)
    for axis, channel in zip(axes[0], channels):
        values = np.asarray(flux_mean[channel], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            vmin, vmax = np.nanpercentile(finite, (1, 99))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite) + 1.0)
        else:
            vmin, vmax = 0.0, 1.0
        image = axis.imshow(
            values,
            origin="lower",
            extent=extent,
            aspect="equal",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
        )
        axis.set_title(f"{channel}: dark-subtracted mean integrated flux")
        axis.set_xlabel("Mount X")
        axis.set_ylabel("Mount Y")
        figure.colorbar(image, ax=axis, label="ADU / frame")
    figure.suptitle(f"Extended normal X/Y scan — {int(NFRAMES)} frames per point")
    figure.tight_layout()
    figure.savefig(path, dpi=170, facecolor="white")
    plt.close(figure)


def _confirm() -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("Interactive confirmation required; rerun with --yes only after reviewing the plan")
    answer = input("Apply the listed GLINT-DM PTT state and start the physical X/Y scan? Type YES: ").strip()
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
    x, y = validate_user_settings()
    dm_state = build_dm_state()
    points = tuple(serpentine_points(x, y))
    keepalive_interval_s = float(DM_KEEPALIVE_INTERVAL_MIN) * 60.0
    print("Extended normal GLINT X/Y scan")
    print(f"  grid             : X={x[0]:g}..{x[-1]:g}, Y={y[0]:g}..{y[-1]:g}, step={XY_STEP:g}")
    print(f"  points           : {len(points)} ({len(x)} x {len(y)})")
    print(f"  frames / point   : {int(NFRAMES)}")
    print(f"  extra settling   : {float(SETTLE_S):g} s after verified motion")
    print(f"  checkpoint       : every {int(CHECKPOINT_EVERY)} points")
    print("  mount commands   : fixed roll once; only the changing raster axis thereafter")
    print("  saved values     : dark-subtracted integrated Phot 1/2/3 mean, STD and SEM")
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
    dark_box_sums: dict[str, float] = {}
    for channel, box in PHOT_BOXES.items():
        y0, y1, x0, x1 = map(int, box)
        if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
            raise ValueError(f"Invalid {channel} box {box}")
        if y1 > dark.shape[0] or x1 > dark.shape[1]:
            raise ValueError(f"{channel} box {box} exceeds dark shape {dark.shape}")
        dark_box_sums[channel] = float(np.sum(dark[y0:y1, x0:x1], dtype=float))

    output_base = _next_output_base(output_root)
    json_path = output_base.with_suffix(".json")
    fits_path = output_base.with_suffix(".fits")
    png_path = output_base.with_suffix(".png")
    shape = (len(y), len(x))
    acquired = np.zeros(shape, dtype=bool)
    actual_x = np.full(shape, np.nan)
    actual_y = np.full(shape, np.nan)
    raw_max = np.full(shape, np.nan)
    flux_mean = {channel: np.full(shape, np.nan) for channel in PHOT_BOXES}
    flux_std = {channel: np.full(shape, np.nan) for channel in PHOT_BOXES}
    flux_sem = {channel: np.full(shape, np.nan) for channel in PHOT_BOXES}
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
    mount_axis_commands = {"roll": 0, "x": 0, "y": 0}
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "normal_xy_extended",
        "status": "starting",
        "started_utc": started.isoformat(),
        "instrument_config": str(instrument_path),
        "axes": {"x": x.tolist(), "y": y.tolist()},
        "acquisition": {
            "nframes": int(NFRAMES),
            "settle_s": float(SETTLE_S),
            "serpentine": True,
            "checkpoint_every": int(CHECKPOINT_EVERY),
            "saved_detector_pixels": False,
            "flux_definition": (
                "mean of per-frame raw box sums minus the matching saved-dark box sum; "
                "exactly equivalent to dark subtraction before integration"
            ),
            "mount_strategy": "fixed roll once; command only X/Y axes that change along serpentine route",
            "per_point_full_mount_readback": False,
        },
        "photometry_boxes_y0_y1_x0_x1": {
            channel: list(map(int, box)) for channel, box in PHOT_BOXES.items()
        },
        "commanded_dm_ptt": dm_state_to_dict(dm_state),
        "glint_dm_keepalive": {
            "method": "republish unchanged state through InstrumentHardware.set_dm / GLINT dmptt",
            "interval_min": float(DM_KEEPALIVE_INTERVAL_MIN),
            "events": [],
            "ssh_used": False,
            "scexao_dm_accessed": False,
        },
        "dark": {"path": str(dark_path), "sha256": _sha256(dark_path), "subtracted_per_frame": True},
        "saturation_guard": {
            "adc_max_adu": float(ADC_MAX_ADU),
            "fraction": float(SATURATION_FRACTION),
            "limit_adu": float(ADC_MAX_ADU) * float(SATURATION_FRACTION),
        },
        "planned_points": len(points),
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
                acquired=acquired,
                actual_x=actual_x,
                actual_y=actual_y,
                raw_max=raw_max,
                flux_mean=flux_mean,
                flux_std=flux_std,
                flux_sem=flux_sem,
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
        for name, values in (("x", x), ("y", y)):
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

        report_every = max(1, len(points) // 25)
        acquisition_started_monotonic = time.monotonic()
        previous_x: int | None = None
        previous_y: int | None = None
        for count, (iy, ix) in enumerate(points, start=1):
            keepalive_started = time.monotonic()
            refreshed = keepalive.maybe_refresh()
            if refreshed:
                time.sleep(float(DM_KEEPALIVE_SETTLE_S))
            timing_totals["keepalive_s"] += time.monotonic() - keepalive_started

            target_x = int(round(float(x[ix])))
            target_y = int(round(float(y[iy])))
            move_targets: dict[str, float] = {}
            if previous_x is None or target_x != previous_x:
                move_targets["x"] = float(target_x)
            if previous_y is None or target_y != previous_y:
                move_targets["y"] = float(target_y)
            # The first point moves X and Y. Serpentine ordering guarantees
            # that every subsequent point changes exactly one of those axes.
            if count > 1 and len(move_targets) != 1:
                raise RuntimeError(
                    f"Non-adjacent serpentine move at point {count}: targets={move_targets}"
                )
            timed_mount_move(move_targets)
            previous_x, previous_y = target_x, target_y

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
            point_raw_max = float(np.nanmax(frames))
            if point_raw_max >= guard_limit:
                raise SaturationDetected(
                    f"Raw APAPANE maximum {point_raw_max:.1f} ADU reached the "
                    f"{guard_limit:.1f} ADU guard at X={x[ix]:g}, Y={y[iy]:g}"
                )
            photometry_started = time.monotonic()
            for channel, box in PHOT_BOXES.items():
                mean, std, sem = dark_subtracted_flux_statistics(
                    frames,
                    box,
                    dark_box_sums[channel],
                )
                flux_mean[channel][iy, ix] = mean
                flux_std[channel][iy, ix] = std
                flux_sem[channel][iy, ix] = sem
            timing_totals["photometry_s"] += time.monotonic() - photometry_started

            # move_mount() has already waited for exact controller readback.
            # Recording those verified integer targets avoids six additional
            # serial position queries at every point.
            actual_x[iy, ix] = float(target_x)
            actual_y[iy, ix] = float(target_y)
            raw_max[iy, ix] = point_raw_max
            acquired[iy, ix] = True
            if count == 1 or count == len(points) or count % report_every == 0:
                values = ", ".join(
                    f"{channel}={flux_mean[channel][iy, ix]:.6g}" for channel in PHOT_BOXES
                )
                elapsed = time.monotonic() - acquisition_started_monotonic
                eta_s = elapsed / float(count) * float(len(points) - count)
                print(
                    f"  [{count:5d}/{len(points)}] X={target_x:g}, Y={target_y:g}; "
                    f"{values}; elapsed={elapsed / 60:.1f} min, ETA={eta_s / 60:.1f} min"
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
                hardware.move_mount(
                    {
                        "roll": float(pre_mount["roll"]),
                        "x": float(pre_mount["x"]),
                        "y": float(pre_mount["y"]),
                    },
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
            save_plot(png_path, x, y, flux_mean)
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
    print(f"\nSaved extended normal X/Y scan: {output_base.parent}")
    return output_base.parent


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
