#!/usr/bin/env python3
"""Scan all 37 GLINT DM pistons at successive absolute mount-Z focus positions.

Run from a terminal; edit USER SETTINGS / SEGMENT_PTT or pass axis options.
Only Z is moved, once per focus. X/Y/pitch/yaw/roll retain launch positions.
At each focus, scan segments 0..36 individually, restoring the configured base
PTT between segments. Absolute scanned piston replaces only that segment's
base piston; its configured tip/tilt and every other segment's PTT are retained.
Launch-time Z and full DM PTT are restored on completion, Ctrl+C or error.

SPECTRA[focus, segment, piston, spectral_column] contains the mean of NFRAMES
frames, dark subtracted once, with detector rows SUMMED. FLUX is its column
sum in ADU/frame. No box-height or NFRAMES factor is needed in analysis.
OPD = 2 * commanded surface piston (um); BASE_PTT separately records the
reference state. FITS and JSON checkpoints retain partial acquisitions.
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
from astropy.io import fits

# USER SETTINGS ---------------------------------------------------------------
PISTON_START = -0.6  # absolute DM surface stroke [um], not an offset to base PTT
PISTON_END = 0.3
PISTON_STEP = 0.05  # [um]; endpoints must be separated by an integer step count
NFRAMES = 25       # averaged dark-subtracted frames per piston step

# Absolute mount Z [integer controller pulses]. None prompts at launch.
# All other mount axes stay at their launch positions, including roll.
FOCUS_START = None
FOCUS_END = None
FOCUS_STEP = None  # positive magnitude; start > end scans downwards

SPECTRAL_BOX = (72, 78, 261, 271)  # [y0,y1,x0,x1], EXCLUSIVE upper bounds
#SPECTRAL_BOX = (72, 77, 158, 267)
DARK_PATH = None  # None: find the saved APAPANE dark under alignment output root
OUTPUT_ROOT = None  # None: instrument.json output_root
OUTPUT_FOLDER = "FIber_DMfocus_piston_scan"  # exact instrument data-folder name
DM_SETTLE_S = 0.10
FOCUS_SETTLE_S = 0.10
CHECKPOINT_EVERY = 100  # samples; also checkpoint after every segment and on error
ADC_MAX_ADU = 3000000.0
SATURATION_FRACTION = 0.80

# (surface piston [um], tip [mrad], tilt [mrad]), GUI IDs 0..36.
# Flat defaults preserve the starter script's flat setting. Edit all PTT here.
# The scanned segment RETAINS this tip/tilt while its absolute piston is swept.
SEGMENT_PTT = {
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
    11: (0.0, 0.0, 0.0),
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
    36: (0.0, 0.0, 0.0),
}
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
        return np.mean(dark, axis=0)
    raise ValueError(f"Expected a 2-D dark or 3-D dark cube, received shape {dark.shape}")



def dark_subtracted_spectrum(
    raw_frames: np.ndarray,
    dark: np.ndarray,
    box: Sequence[int],
) -> np.ndarray:
    """Average frames, subtract dark, then sum rows within each column."""
    frames = _frame_cube(raw_frames)
    if frames.shape[1:] != dark.shape:
        raise ValueError(f"Frame shape {frames.shape[1:]} differs from dark {dark.shape}")
    y0, y1, x0, x1 = map(int, box)
    if min(y0, x0) < 0 or y1 <= y0 or x1 <= x0 or y1 > dark.shape[0] or x1 > dark.shape[1]:
        raise ValueError(f"Invalid spectral box {box} for dark shape {dark.shape}")
    averaged = np.mean(frames, axis=0, dtype=float)
    corrected = averaged - dark
    spectrum = np.sum(corrected[y0:y1, x0:x1], axis=0, dtype=float)
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
    pattern = re.compile(r"^fiber_dm_piston_(\d+)\.(?:json|fits|png)$")
    indices = []
    for path in day.glob("fiber_dm_piston_*.*"):
        match = pattern.match(path.name)
        if match:
            indices.append(int(match.group(1)))
    return day / f"fiber_dm_piston_{max(indices, default=0) + 1}"



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





def scan_axis(start, end, step, name, *, pulses=False):
    """Inclusive uniform grid; positive step magnitude, either scan direction."""
    start, end, step = float(start), float(end), float(step)
    if not np.all(np.isfinite([start, end, step])) or step <= 0:
        raise ValueError(f"{name}: finite endpoints and positive step required")
    intervals = abs(end-start)/step
    count = int(round(intervals))
    if not np.isclose(intervals, count, rtol=0, atol=1e-8):
        raise ValueError(f"{name}: abs(end-start) must be an integer multiple of step")
    values = start + np.sign(end-start)*step*np.arange(count+1, dtype=float)
    if pulses and (not np.allclose(values, np.rint(values), rtol=0, atol=1e-8)
                   or not float(step).is_integer()):
        raise ValueError(f"{name}: coordinates and step must be integer mount pulses")
    values[-1] = end
    return values


def validate_settings():
    bounds = np.asarray(SPECTRAL_BOX, dtype=float)
    if (bounds.shape != (4,) or not np.all(np.isfinite(bounds))
            or not np.all(bounds == np.rint(bounds))):
        raise ValueError("SPECTRAL_BOX must contain four finite integer bounds")
    y0, y1, x0, x1 = map(int, bounds)
    if min(y0, x0) < 0 or y1 <= y0 or x1 <= x0:
        raise ValueError("Invalid spectral box")
    for name, value in (("NFRAMES", NFRAMES), ("CHECKPOINT_EVERY", CHECKPOINT_EVERY)):
        if not np.isfinite(value) or value < 1 or int(value) != value:
            raise ValueError(f"{name} must be a positive integer")
    if (not np.all(np.isfinite([DM_SETTLE_S, FOCUS_SETTLE_S, ADC_MAX_ADU, SATURATION_FRACTION]))
            or min(DM_SETTLE_S, FOCUS_SETTLE_S) < 0 or ADC_MAX_ADU <= 0
            or not 0 < SATURATION_FRACTION < 1):
        raise ValueError("Invalid settling time or saturation settings")
    if set(SEGMENT_PTT) != set(range(37)):
        raise ValueError("SEGMENT_PTT must contain exactly segments 0..36")
    for segment, ptt in SEGMENT_PTT.items():
        values = np.asarray(ptt, dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError(f"SEGMENT_PTT[{segment}] needs three finite values")
    return y0, y1, x0, x1


def state_for_piston(base, segment, piston):
    """A fresh full state prevents the previous segment's sweep carrying over."""
    return {s: SegmentPTT(float(piston) if s == segment else p.piston_um,
                          p.tip_mrad, p.tilt_mrad) for s, p in base.items()}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument-config", type=Path, default=DEFAULT_INSTRUMENT_CONFIG)
    for name, default in (("focus-start", FOCUS_START), ("focus-end", FOCUS_END),
                          ("focus-step", FOCUS_STEP), ("piston-start", PISTON_START),
                          ("piston-end", PISTON_END), ("piston-step", PISTON_STEP)):
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--check-only", action="store_true", help="Print plan; no hardware or files")
    parser.add_argument("--yes", action="store_true", help="Start without the terminal confirmation")
    return parser.parse_args(argv)


def get_axes(args):
    for key in ("focus_start", "focus_end", "focus_step"):
        if getattr(args, key) is None:
            if args.check_only:
                raise ValueError(f"Set {key.upper()} or pass --{key.replace('_', '-')}")
            setattr(args, key, float(input(f"{key.replace('_', ' ').capitalize()} [absolute Z pulses]: ")))
    return (scan_axis(args.focus_start, args.focus_end, args.focus_step, "Focus", pulses=True),
            scan_axis(args.piston_start, args.piston_end, args.piston_step, "Piston"))


def write_fits_atomic(path, payload, focus, pistons, base, box, arrays):
    """Self-describing cube; identical snapshot metadata in FITS and JSON."""
    header = fits.Header()
    header["SCHEMA"] = 1
    header["SCANTYPE"] = "FIBER_DM_PISTON"
    header["STATUS"] = payload["status"]
    header["NFRAMES"] = int(NFRAMES)
    header["NFOCUS"] = len(focus)
    header["NSEG"] = 37
    header["NPISTON"] = len(pistons)
    header["NACQ"] = int(arrays["ACQUIRED"].sum())
    header["DATE-OBS"] = payload["started_utc"]
    header["ROWSUM"] = (True, "Dark-subtracted row sums")
    header["DARKSUB"] = True
    header["Y0"], header["Y1"], header["X0"], header["X1"] = box
    header["OPDDEF"] = "2 * absolute DM surface stroke [um]"
    header["COMMENT"] = "NumPy SPECTRA axes: focus,segment,piston,spectral_column."
    header["COMMENT"] = "Box upper bounds excluded; FLUX = SPECTRA.sum(axis=-1)."
    hdus = [fits.PrimaryHDU(header=header)]
    units = {"SPECTRA": "ADU/frame", "FLUX": "ADU/frame", "RAW_MAX": "ADU",
             "TIME_S": "s", "Z_ACTUAL": "pulse"}
    for name, values in arrays.items():
        hdu = fits.ImageHDU(values.astype(np.uint8) if name == "ACQUIRED" else values, name=name)
        if name in units:
            hdu.header["BUNIT"] = units[name]
        if name == "SPECTRA":
            hdu.header["NPAXES"] = "focus,segment,piston,spectral_column"
        hdus.append(hdu)
    for name, values, unit in (("FOCUS_CMD", focus, "pulse"), ("PISTON_CMD", pistons, "um"),
                                ("OPD", 2*pistons, "um"), ("SEGMENT", np.arange(37), ""),
                                ("SPECTRAL_X", np.arange(box[2], box[3]), "pixel"),
                                ("BASE_PTT", base, "")):
        hdu = fits.ImageHDU(values, name=name)
        if unit:
            hdu.header["BUNIT"] = unit
        if name == "BASE_PTT":
            hdu.header["PTTORDER"] = "surface_um,tip_mrad,tilt_mrad"
        hdus.append(hdu)
    encoded = json.dumps(payload, allow_nan=False)
    hdus.append(fits.BinTableHDU.from_columns(
        [fits.Column(name="JSON", format=f"{len(encoded)}A", array=[encoded])], name="SCAN_SETTINGS"))
    temporary = path.with_name(f".{path.name}.tmp")
    with fits.HDUList(hdus) as hdul:
        hdul.writeto(temporary, overwrite=True, checksum=True)
    os.replace(temporary, path)


def run(args):
    box = validate_settings()
    focus, pistons = get_axes(args)
    config = load_config(args.instrument_config, simulation=False)
    # Check configured limits before any physical device is opened.
    configured_limits = config.get("mount_control", {}).get("limits", {}).get("z")
    if configured_limits is not None:
        lower, upper = map(float, configured_limits)
        if focus.min() < lower or focus.max() > upper:
            raise ValueError(f"Focus range exceeds configured Z limits [{lower:g}, {upper:g}]")
    base = {s: SegmentPTT(*map(float, ptt)) for s, ptt in SEGMENT_PTT.items()}
    base_array = np.array([SEGMENT_PTT[s] for s in range(37)], dtype=float)
    shape = (len(focus), 37, len(pistons))
    npoints = int(np.prod(shape))
    root = Path(OUTPUT_ROOT).expanduser().resolve() if OUTPUT_ROOT is not None else _alignment_root(config)
    print("Single-fiber DM piston scan at absolute mount-Z focus positions")
    print(f"  Focus Z [pulses]: {focus.tolist()}")
    print(f"  DM stroke [um] : {np.round(pistons, 10).tolist()}; OPD = 2 * stroke")
    print(f"  Samples        : {len(focus)} focus * 37 segments * {len(pistons)} pistons = {npoints}")
    print(f"  Spectral cube  : {(*shape, box[3]-box[2])} [focus,segment,piston,column]")
    print(f"  Frames/sample  : {NFRAMES}; mean frames -> subtract dark -> SUM rows {box}")
    print("  Mount          : Z only; X/Y/pitch/yaw/roll remain at launch positions")
    print("  Segment PTT    : scanned absolute piston; configured tip/tilt retained")
    for s, values in sorted(SEGMENT_PTT.items()):
        print(f"    {s:2d}: {values}")
    print("  Final state    : restore launch-time Z and full GLINT DM PTT")
    print(f"  Save directory : {root / datetime.now().astimezone().strftime('%Y%m%d') / OUTPUT_FOLDER}")
    print("  Output files   : fiber_dm_piston_N.fits + fiber_dm_piston_N.json")
    if args.check_only:
        return None
    dark_path = _dark_path(root)
    dark = _load_dark(dark_path)
    if (not np.all(np.isfinite(dark)) or box[1] > dark.shape[0] or box[3] > dark.shape[1]):
        raise ValueError(f"Non-finite dark or spectral box {box} exceeds dark shape {dark.shape}")
    if not args.yes and input("Start the listed physical DM/Z scan? Type YES: ").strip() != "YES":
        raise RuntimeError("Scan not started")
    output = _next_output_base(root)
    fits_path, json_path = output.with_suffix(".fits"), output.with_suffix(".json")
    arrays = {"SPECTRA": np.full((*shape, box[3]-box[2]), np.nan),
              "ACQUIRED": np.zeros(shape, dtype=bool),
              "Z_ACTUAL": np.full(len(focus), np.nan)}
    arrays.update({name: np.full(shape, np.nan) for name in ("FLUX", "RAW_MAX", "TIME_S")})
    payload = {
        "schema_version": 1, "kind": "fiber_dm_piston_focus", "status": "starting",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "instrument_config": str(args.instrument_config.resolve()), "instrument_config_snapshot": config,
        "axes": {"focus_z_pulses": focus.tolist(), "segment": list(range(37)),
                 "piston_um": pistons.tolist(), "opd_um": (2*pistons).tolist()},
        "requested_axes": {k: float(getattr(args, k)) for k in
                           ("focus_start", "focus_end", "focus_step", "piston_start", "piston_end", "piston_step")},
        "cube_axis_order": ["focus", "segment", "piston", "spectral_column"],
        "cube_shape": [*shape, box[3]-box[2]],
        "spectral_box_y0_y1_x0_x1": list(box), "box_upper_bounds_excluded": True,
        "nframes": int(NFRAMES), "planned_points": npoints, "acquired_points": 0,
        "spectral_units": "ADU/frame per detector column, summed over box rows",
        "flux_definition": "mean raw frames; subtract saved dark once; sum box rows and columns",
        "opd_definition": "2 * absolute commanded DM surface stroke in um",
        "base_segment_ptt": dm_state_to_dict(base),
        "piston_mode": "replace scanned segment piston; preserve all configured tip/tilt and other base pistons",
        "mount_strategy": "absolute Z per focus; no X/Y/pitch/yaw/roll commands",
        "timing_definition": "TIME_S = midpoint of frame-batch read relative to run start (monotonic)",
        "settle_s": {"dm": float(DM_SETTLE_S), "focus": float(FOCUS_SETTLE_S)},
        "checkpoint_every": int(CHECKPOINT_EVERY),
        "dark": {"path": str(dark_path), "sha256": _sha256(dark_path)},
        "saturation_limit_adu": float(ADC_MAX_ADU*SATURATION_FRACTION),
        "outputs": {"fits": fits_path.name, "json": json_path.name},
        "missing_data": "NaN, with ACQUIRED marking completed piston steps",
    }
    clock_start = time.monotonic()
    hardware = None
    pre_mount = pre_dm = None
    failure = None
    timeout = float(config["safety"]["motion_timeout_s"])

    def checkpoint(status):
        payload.update(status=status, acquired_points=int(arrays["ACQUIRED"].sum()),
                       elapsed_s=float(time.monotonic()-clock_start))
        write_fits_atomic(fits_path, payload, focus, pistons, base_array, box, arrays)
        _atomic_json(json_path, payload)

    # Reserve the output index even if opening the instrument fails.
    _atomic_json(json_path, payload)
    try:
        hardware = InstrumentHardware(config)
        lower, upper = hardware.mount_limits()["z"]
        if focus.min() < lower or focus.max() > upper:
            raise ValueError(f"Focus grid exceeds live Z limits [{lower:g}, {upper:g}]")
        pre_mount, pre_dm = hardware.get_mount(), hardware.get_dm()
        if not np.all(np.isfinite(list(pre_mount.values()))):
            raise ValueError("Launch mount readbacks must be finite")
        payload["pre_scan_state"] = {"mount": {k: float(v) for k, v in pre_mount.items()},
                                     "dm": dm_state_to_dict(pre_dm)}
        hardware.set_dm(base)
        count = 0
        for iz, z in enumerate(focus):
            result = hardware.move_mount({"z": float(z)}, timeout)
            arrays["Z_ACTUAL"][iz] = float(result["z"])
            time.sleep(float(FOCUS_SETTLE_S))
            print(f"Focus {iz+1}/{len(focus)}: Z={z:g} pulses", flush=True)
            for segment in range(37):
                for ip, piston in enumerate(pistons):
                    hardware.set_dm(state_for_piston(base, segment, piston))
                    time.sleep(float(DM_SETTLE_S))
                    t0 = time.monotonic()-clock_start
                    frames = _frame_cube(hardware.apapane_frames(int(NFRAMES)))
                    t1 = time.monotonic()-clock_start
                    if frames.shape != (int(NFRAMES), *dark.shape) or not np.all(np.isfinite(frames)):
                        raise ValueError(f"Bad APAPANE batch: {frames.shape}; expected {(int(NFRAMES), *dark.shape)}")
                    maximum = float(frames.max())
                    if maximum >= ADC_MAX_ADU*SATURATION_FRACTION:
                        raise RuntimeError(f"Saturated frame ({maximum:g} ADU): focus {z:g}, segment {segment}, piston {piston:g}")
                    spectrum = dark_subtracted_spectrum(frames, dark, box)
                    index = (iz, segment, ip)
                    arrays["SPECTRA"][index] = spectrum
                    arrays["FLUX"][index] = spectrum.sum()
                    arrays["RAW_MAX"][index] = maximum
                    arrays["TIME_S"][index] = (t0+t1)/2
                    arrays["ACQUIRED"][index] = True
                    count += 1
                    if count % int(CHECKPOINT_EVERY) == 0:
                        checkpoint("running")
                hardware.set_dm(base)
                checkpoint("running")
                print(f"  Segment {segment:2d}: {len(pistons)} spectra; {count}/{npoints} samples", flush=True)
    except BaseException as error:
        failure = error
        payload["error"] = f"{type(error).__name__}: {error}"
    finally:
        restore_errors = []
        if hardware is not None:
            # Separate attempts: a failed Z restoration must not skip DM restoration.
            if pre_mount is not None:
                try:
                    hardware.move_mount({"z": float(pre_mount["z"])}, timeout)
                except BaseException as error:
                    restore_errors.append(f"mount Z: {type(error).__name__}: {error}")
            if pre_dm is not None:
                try:
                    hardware.set_dm(pre_dm)
                except BaseException as error:
                    restore_errors.append(f"DM PTT: {type(error).__name__}: {error}")
            payload["restored_pre_scan_state"] = pre_mount is not None and pre_dm is not None and not restore_errors
            if payload["restored_pre_scan_state"]:
                try:
                    payload["state_after_restore"] = {"mount": {k: float(v) for k, v in hardware.get_mount().items()},
                                                       "dm": dm_state_to_dict(hardware.get_dm())}
                except Exception as error:
                    payload["restore_readback_error"] = str(error)
            try:
                hardware.close()
            except BaseException as error:
                restore_errors.append(f"close: {type(error).__name__}: {error}")
        if restore_errors:
            payload["restore_errors"] = restore_errors
            if failure is None:
                failure = RuntimeError("; ".join(restore_errors))
                payload["error"] = str(failure)
        payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
        status = "completed" if failure is None else ("aborted" if isinstance(failure, KeyboardInterrupt) else "failed")
        try:
            checkpoint(status)
        except Exception as error:
            payload.update(status="failed", checkpoint_error=f"{type(error).__name__}: {error}")
            _atomic_json(json_path, payload)
            if failure is None:
                failure = error
    print(f"Saved {payload['acquired_points']}/{npoints} spectra: {fits_path}", flush=True)
    if failure is not None:
        raise failure
    return fits_path


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
