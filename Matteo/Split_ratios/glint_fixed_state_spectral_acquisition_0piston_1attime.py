#!/usr/bin/env python3
"""Acquire one fixed-state GLINT/APAPANE spectrum per output, for N frames.

Edit the USER SETTINGS below. Run from the same environment as the earlier
normal_xy_spectral_piston_scan.py (glint_alignment_refactor, astropy, numpy).
The APAPANE stream is read only: exposure, gain and camera state are untouched.
No piston scan, discarded trial, or extra preview frames are acquired.

Each FITS image extension is [frame, detector-X spectral column], obtained by
subtracting the saved dark and SUMMING pixels along detector Y. Detector X is
pixel position, not wavelength. The CHANNELS table gives each extension's
detector box and baseline; DM_PTT records the commanded 37-segment state.

Only mount X/Y are moved. Roll, pitch, yaw and Z retain their current values.
At exit, the original X/Y and GLINT DM state are restored, even after abort.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from astropy.io import fits


# ============================ USER SETTINGS ================================
# Required: the desired mount coordinates (pulses), output file label, and N.
X_MOUNT: int | None = None
Y_MOUNT: int | None = None
X_MOUNT = 3038
Y_MOUNT = 3278
FITS_LABEL = "ml11_only_injected"  # e.g. "ml11_only_injected"; filename is FITS_LABEL.fits
NFRAMES = 100

# Surface piston [um], tip [mrad], tilt [mrad]. Edit EVERY row you need.
# These are GLINT DM segments 00..36, not the SCExAO DM. Unedited rows are
# flat; set tilt-away explicitly where needed. Confirm all 37 before --yes.
SEGMENT_PTT: dict[int, tuple[float, float, float]] = {
     0: (0.0, -5.0, 0.0),  1: (0.0, -5.0, 0.0),  2: (0.0, -5.0, 0.0),
     3: (0.0, -5.0, 0.0),  4: (0.0, -5.0, 0.0),  5: (0.0, -5.0, 0.0),
     6: (0.0, -5.0, 0.0),  7: (0.0, -5.0, 0.0),  8: (0.0, -5.0, 0.0),
     9: (0.0, -5.0, 0.0), 10: (0.0, -5.0, 0.0), 11: (0.0, -5.0, 0.0),
    12: (0.0, -5.0, 0.0), 13: (0.0, -5.0, 0.0), 14: (0.0, -5.0, 0.0),
    15: (0.0, -5.0, 0.0), 16: (0.0, -5.0, 0.0), 17: (0.0, -5.0, 0.0),
    18: (0.0, -5.0, 0.0), 19: (0.0, -5.0, 0.0), 20: (0.0, -5.0, 0.0),
    21: (0.0, -5.0, 0.0), 22: (0.0, -5.0, 0.0), 23: (0.0, -5.0, 0.0),
    24: (0.0, -5.0, 0.0), 25: (0.0, -5.0, 0.0), 26: (0.0, -5.0, 0.0),
    27: (0.0, -5.0, 0.0), 28: (0.0, -5.0, 0.0), 29: (0.0, -5.0, 0.0),
    30: (0.0, -5.0, 0.0), 31: (0.0, 0.07, 0.27), 32: (0.0, -5.0, 0.0),
    33: (0.0, -5.0, 0.0), 34: (0.0, -5.0, 0.0), 35: (0.0, -5.0, 0.0),
    36: (0.0, -5.0, 0.0),
}

# Optional full detector-box JSON (same nested schema as detector_boxes.json).
# If None, searches alongside this script, then refactor/config. A file is
# REQUIRED; the example calibration may only cover 1500-1700 nm, so use the
# appropriate boxes for the band being observed.
DETECTOR_BOXES_PATH: str | None = None
DARK_PATH: str | None = None  # Otherwise locate the saved APAPANE dark as below.
OUTPUT_FOLDER = "split_ratios_scans"
FRAMES_PER_BATCH = 50  # Checkpoint spectra and refresh the GLINT DM between batches.
MOUNT_SETTLE_S = 0.1
DM_SETTLE_S = 0.1
DM_KEEPALIVE_INTERVAL_MIN = 15.0
DM_KEEPALIVE_SETTLE_S = 0.1
ADC_MAX_ADU = 3000000.0  # Confirm for the current APAPANE readout mode.
SATURATION_FRACTION = 0.8
# ==========================================================================


SCRIPT_DIRECTORY = Path(__file__).resolve().parent


def _find_refactor_root() -> Path:
    configured = os.environ.get("GLINT_ALIGNMENT_REFACTOR_ROOT")
    candidates = [
        Path(configured).expanduser() if configured else None,
        SCRIPT_DIRECTORY,
        SCRIPT_DIRECTORY.parent / "glint_alignment_refactor",
        SCRIPT_DIRECTORY.parents[1] / "glint_alignment_refactor"
        if len(SCRIPT_DIRECTORY.parents) > 1 else None,
        Path.cwd(),
        Path.cwd() / "glint_alignment_refactor",
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "glint_alignment" / "__init__.py").is_file():
            return candidate.resolve()
    raise ModuleNotFoundError(
        "Cannot find glint_alignment_refactor. Set GLINT_ALIGNMENT_REFACTOR_ROOT "
        "to its absolute directory (as for normal_xy_spectral_piston_scan.py)."
    )


GLINT_ALIGNMENT_REFACTOR_ROOT = _find_refactor_root()
if str(GLINT_ALIGNMENT_REFACTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(GLINT_ALIGNMENT_REFACTOR_ROOT))

from glint_alignment.config import load_config
from glint_alignment.instrument import InstrumentHardware
from glint_alignment.models import SegmentPTT, dm_state_to_array

DEFAULT_INSTRUMENT_CONFIG = GLINT_ALIGNMENT_REFACTOR_ROOT / "config" / "instrument.json"
Channel = tuple[str, str, str, str]  # full name, group, FITS EXTNAME, baseline


class SaturationDetected(RuntimeError):
    """A raw APAPANE pixel exceeded the specified guard."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument-config", type=Path, default=DEFAULT_INSTRUMENT_CONFIG)
    parser.add_argument("--boxes-config", type=Path, help="Full detector_boxes.json override")
    parser.add_argument("--check-only", action="store_true", help="Validate and print, no hardware")
    parser.add_argument("--yes", action="store_true", help="Skip initial YES confirmation")
    return parser.parse_args(argv)


def validate_settings() -> dict[int, SegmentPTT]:
    for axis, value in (("X_MOUNT", X_MOUNT), ("Y_MOUNT", Y_MOUNT)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"Edit {axis}: an integer number of mount pulses is required")
    if not isinstance(FITS_LABEL, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", FITS_LABEL):
        raise ValueError("Edit FITS_LABEL: use letters, digits, underscores or hyphens")
    if isinstance(NFRAMES, bool) or not isinstance(NFRAMES, (int, np.integer)) or NFRAMES < 1:
        raise ValueError("NFRAMES must be an integer >= 1")
    if isinstance(FRAMES_PER_BATCH, bool) or not isinstance(FRAMES_PER_BATCH, int) or FRAMES_PER_BATCH < 1:
        raise ValueError("FRAMES_PER_BATCH must be an integer >= 1")
    if set(SEGMENT_PTT) != set(range(37)):
        raise ValueError("SEGMENT_PTT must specify exactly the GLINT DM segments 0..36")
    state: dict[int, SegmentPTT] = {}
    for segment in range(37):
        row = SEGMENT_PTT[segment]
        if not isinstance(row, (tuple, list)) or len(row) != 3:
            raise ValueError(f"SEGMENT_PTT[{segment}] needs (piston_um, tip_mrad, tilt_mrad)")
        try:
            values = tuple(float(v) for v in row)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Non-numeric PTT in segment {segment}") from exc
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite PTT in segment {segment}")
        state[segment] = SegmentPTT(*values)
    if (not np.isfinite(ADC_MAX_ADU) or ADC_MAX_ADU <= 0
            or not 0 < SATURATION_FRACTION < 1 or DM_KEEPALIVE_INTERVAL_MIN <= 0
            or min(MOUNT_SETTLE_S, DM_SETTLE_S, DM_KEEPALIVE_SETTLE_S) < 0):
        raise ValueError("Invalid guard, keepalive or settling setting")
    return state


def boxes_path(override: Path | None) -> Path:
    if override is not None or DETECTOR_BOXES_PATH is not None:
        chosen = override if override is not None else Path(DETECTOR_BOXES_PATH)
        path = chosen.expanduser()
        path = path if path.is_absolute() else SCRIPT_DIRECTORY / path
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    for candidate in (
        SCRIPT_DIRECTORY / "detector_boxes.json",
        SCRIPT_DIRECTORY / "config" / "detector_boxes.json",
        GLINT_ALIGNMENT_REFACTOR_ROOT / "config" / "detector_boxes.json",
    ):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Full detector_boxes.json not found; set DETECTOR_BOXES_PATH")


def load_boxes(path: Path) -> tuple[dict[str, tuple[int, int, int, int]], list[Channel]]:
    import json

    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    result: dict[str, tuple[int, int, int, int]] = {}
    channels: list[Channel] = []

    def add(name: str, bounds: Any) -> None:
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 4:
            raise ValueError(f"Invalid [y0,y1,x0,x1] for {name}")
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not np.isfinite(v) or int(v) != v for v in bounds):
            raise ValueError(f"Non-integer box bounds for {name}")
        y0, y1, x0, x1 = map(int, bounds)
        if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
            raise ValueError(f"Invalid box bounds for {name}")
        result[name] = (y0, y1, x0, x1)

    for n in (1, 2, 3):
        name = f"Phot {n}"
        add(name, document["photometry"][name])
        channels.append((name, "photometry", f"PHOT{n}", ""))
    null_by_number: dict[int, str] = {}
    for full_name in document["null"]:
        match = re.fullmatch(r"N([123])\s*/\s*(.+)", full_name)
        if match is None:
            raise ValueError(f"Unexpected null name {full_name!r}")
        number = int(match.group(1))
        if number in null_by_number:
            raise ValueError(f"Duplicate null N{number}")
        null_by_number[number] = full_name
    if set(null_by_number) != {1, 2, 3}:
        raise ValueError("Detector boxes need N1, N2 and N3")
    if set(document["fringe_tracking"]) != set(null_by_number.values()):
        raise ValueError("Fringe-tracking baseline names must match null baseline names")
    for n in (1, 2, 3):
        full_name = null_by_number[n]
        baseline = full_name.split("/", 1)[1].strip()
        add(full_name, document["null"][full_name])
        channels.append((full_name, "null", f"NULL{n}", baseline))
        for ft in ("FT1", "FT2"):
            name = f"{full_name} {ft}"
            add(name, document["fringe_tracking"][full_name][ft])
            channels.append((name, "fringe_tracking",
                             f"FT{n}{'A' if ft == 'FT1' else 'B'}", baseline))
    return result, channels


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def alignment_root(config: dict[str, Any]) -> Path:
    root = Path(config["output_root"]).expanduser()
    return (root if root.is_absolute() else Path(config["_root"]) / root).resolve()


def resolve_dark_path(root: Path) -> Path:
    if DARK_PATH is not None:
        path = Path(DARK_PATH).expanduser()
        path = path if path.is_absolute() else SCRIPT_DIRECTORY / path
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    for suffix in ("fits", "npy", "npz"):
        candidate = root / "darks" / f"apapane_dark.{suffix}"
        if candidate.is_file():
            return candidate
    dated = [p for suffix in ("fits", "npy", "npz")
             for p in root.glob(f"*/*apapane_dark*/scan_data.{suffix}")]
    if dated:
        return max(dated, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(f"No saved APAPANE dark under {root}; set DARK_PATH")


def load_dark(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        data = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            key = next((key for key in ("map__dark", "dark") if key in archive), None)
            if key is None:
                raise KeyError(f"No dark array in {path}")
            data = archive[key]
    elif path.suffix.lower() == ".fits":
        with fits.open(path, memmap=False) as hdus:
            hdu = next((hdu for hdu in hdus if hdu.data is not None and
                        str(hdu.header.get("LONGNAME", "")).lower() in ("map__dark", "dark")), None)
            if hdu is None:
                hdu = next((hdu for hdu in hdus if hdu.data is not None and hdu.name == "DARK"), None)
            if hdu is None:
                hdu = next((hdu for hdu in hdus if hdu.data is not None), None)
            if hdu is None:
                raise ValueError(f"No dark image in {path}")
            data = np.asarray(hdu.data, dtype=float)
    else:
        raise ValueError(f"Unsupported dark format: {path}")
    data = np.asarray(data, dtype=float)
    if data.ndim == 3:
        data = np.mean(data, axis=0)
    if data.ndim != 2 or not np.all(np.isfinite(data)):
        raise ValueError(f"Dark must be a finite 2D mean frame: {path}")
    return data


def frame_cube(values: np.ndarray, expected_n: int, shape: tuple[int, int]) -> np.ndarray:
    cube = np.asarray(values, dtype=float)
    if cube.ndim == 2:
        cube = cube[None, :, :]
    if cube.shape != (expected_n, *shape) or not np.all(np.isfinite(cube)):
        raise ValueError(f"Expected {expected_n} finite APAPANE frames of shape {shape}, got {cube.shape}")
    return cube


def spectra_for_frames(frames: np.ndarray, dark: np.ndarray,
                       boxes: dict[str, tuple[int, int, int, int]]) -> dict[str, np.ndarray]:
    """All outputs use the same APAPANE frames and per-column row sum."""
    spectra: dict[str, np.ndarray] = {}
    for name, (y0, y1, x0, x1) in boxes.items():
        values = frames[:, y0:y1, x0:x1].sum(axis=1, dtype=float)
        dark_columns = dark[y0:y1, x0:x1].sum(axis=0, dtype=float)
        spectra[name] = (values - dark_columns[None, :]).astype(np.float32)
    return spectra


def save_fits(path: Path, spectra: dict[str, np.ndarray], boxes: dict[str, tuple[int, int, int, int]],
              channels: list[Channel], state: dict[int, SegmentPTT], metadata: dict[str, Any]) -> None:
    header = fits.Header()
    header["RUNSTAT"] = metadata["status"]
    header["LABEL"] = FITS_LABEL
    header["NREQ"] = int(NFRAMES)
    header["NACQ"] = int(metadata["frames_acquired"])
    header["XMOUNT"] = int(X_MOUNT)
    header["YMOUNT"] = int(Y_MOUNT)
    if metadata.get("actual_mount"):
        header["XACT"] = float(metadata["actual_mount"]["x"])
        header["YACT"] = float(metadata["actual_mount"]["y"])
    header["DATE-OBS"] = metadata["started_utc"]
    header["DATE-END"] = datetime.now(timezone.utc).isoformat()
    header["MAXRAW"] = float(metadata["max_raw_adu"])
    header["ADCGUARD"] = float(ADC_MAX_ADU * SATURATION_FRACTION)
    header["DARKFILE"] = Path(metadata["dark_path"]).name
    header["DARKSHA"] = metadata["dark_sha256"]
    header["BOXFILE"] = Path(metadata["boxes_path"]).name
    header["BOXSHA"] = metadata["boxes_sha256"]
    header["SUMAXIS"] = "detector Y"
    header["SPECTRAX"] = "detector X pixel; no wavelength calibration"
    header.add_history(f"GLINT fixed DM PTT, no piston scan. Instrument: {metadata['instrument_path']}")
    header.add_history("Camera settings unchanged; mean saved dark subtracted per frame.")
    header.add_history(f"Initial mount state: {metadata['initial_mount']}")
    if metadata.get("error"):
        header.add_history(f"Acquisition stopped: {metadata['error'][:65]}")
    hdus: list[fits.hdu.base.ExtensionHDU] = [fits.PrimaryHDU(header=header)]
    for name, group, extname, baseline in channels:
        y0, y1, x0, x1 = boxes[name]
        hdu = fits.ImageHDU(data=spectra[name], name=extname)
        hdu.header["CHANNEL"] = name
        hdu.header["GROUP"] = group
        hdu.header["BASELINE"] = baseline
        for key, val in (("Y0", y0), ("Y1", y1), ("X0", x0), ("X1", x1)):
            hdu.header[key] = val
        hdu.header["BUNIT"] = "dark-subtracted ADU / column / frame"
        hdus.append(hdu)
    hdus.append(fits.BinTableHDU.from_columns([
        fits.Column(name="SEGMENT", format="I", array=np.arange(37, dtype=np.int16)),
        fits.Column(name="PISTON_UM", format="D", array=[state[i].piston_um for i in range(37)]),
        fits.Column(name="TIP_MRAD", format="D", array=[state[i].tip_mrad for i in range(37)]),
        fits.Column(name="TILT_MRAD", format="D", array=[state[i].tilt_mrad for i in range(37)]),
    ], name="DM_PTT"))
    temporary = path.with_name(f".{path.stem}.tmp.fits")
    try:
        fits.HDUList(hdus).writeto(temporary, overwrite=True, checksum=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> Path | None:
    state = validate_settings()
    selected_boxes_path = boxes_path(args.boxes_config)
    boxes, channels = load_boxes(selected_boxes_path)
    instrument_path = args.instrument_config.expanduser().resolve()
    instrument = load_config(instrument_path, simulation=False)
    output_root = alignment_root(instrument)
    path = output_root / datetime.now().astimezone().strftime("%Y%m%d") / OUTPUT_FOLDER / f"{FITS_LABEL}.fits"
    if path.exists():
        raise FileExistsError(f"Will not overwrite existing acquisition: {path}")
    print(f"Output: {path}")
    print(f"Move X/Y to ({X_MOUNT}, {Y_MOUNT}); preserve roll/pitch/yaw/Z; N={NFRAMES}")
    print(f"Detector boxes ({selected_boxes_path}): {boxes}")
    print(f"GLINT DM surface piston [um], tip/tilt [mrad]:")
    for i in range(37):
        print(f"  {i:02d}: {SEGMENT_PTT[i]}")
    if all(state[i].tip_mrad == state[i].tilt_mrad == 0 for i in range(37)):
        print("WARNING: all 37 GLINT DM segments are flat.")
    print("APAPANE stream read only; dark-subtracted spectra in 12 FITS extensions.")
    if args.check_only:
        return None
    dark_path = resolve_dark_path(output_root)
    dark = load_dark(dark_path)
    for name, (_, y1, _, x1) in boxes.items():
        if y1 > dark.shape[0] or x1 > dark.shape[1]:
            raise ValueError(f"{name} box exceeds dark frame {dark.shape}")
    if not args.yes:
        if not sys.stdin.isatty() or input("Verify mount, full DM map, boxes and flux; type YES: ").strip() != "YES":
            raise RuntimeError("Operator did not confirm acquisition")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the selected label before touching hardware. No implicit overwrite.
    with path.open("xb"):
        pass
    hardware: InstrumentHardware | None = None
    initial_mount: dict[str, float] | None = None
    initial_dm: dict[int, SegmentPTT] | None = None
    commanded = False
    started = datetime.now(timezone.utc).isoformat()
    metadata: dict[str, Any] = {
        "status": "PARTIAL", "frames_acquired": 0, "max_raw_adu": 0.0,
        "started_utc": started, "dark_path": str(dark_path), "dark_sha256": sha256(dark_path),
        "boxes_path": str(selected_boxes_path), "boxes_sha256": sha256(selected_boxes_path),
        "instrument_path": str(instrument_path), "initial_mount": {},
    }
    spectra = {name: np.empty((0, box[3] - box[2]), dtype=np.float32)
               for name, box in boxes.items()}
    error: BaseException | None = None
    restore_errors: list[str] = []
    try:
        hardware = InstrumentHardware(instrument)
        for axis, value in (("x", X_MOUNT), ("y", Y_MOUNT)):
            low, high = map(float, hardware.mount_limits()[axis])
            if not low <= value <= high:
                raise ValueError(f"{axis}={value} outside mount limits [{low}, {high}]")
        initial_mount = hardware.get_mount()
        metadata["initial_mount"] = dict(initial_mount)
        initial_dm = hardware.get_dm()
        # Travel at the launch-time DM state; apply the requested PTT at XY.
        hardware.move_mount({"x": float(X_MOUNT), "y": float(Y_MOUNT)},
                            float(instrument["safety"]["motion_timeout_s"]))
        time.sleep(MOUNT_SETTLE_S)
        metadata["actual_mount"] = dict(hardware.get_mount())
        if abs(metadata["actual_mount"]["x"] - X_MOUNT) > 1 or abs(metadata["actual_mount"]["y"] - Y_MOUNT) > 1:
            raise RuntimeError(f"Mount X/Y did not reach requested position: {metadata['actual_mount']}")
        commanded = True  # Restore even if the DM write fails partway through.
        hardware.set_dm(state)
        time.sleep(DM_SETTLE_S)
        if not np.allclose(dm_state_to_array(hardware.get_dm()), dm_state_to_array(state), atol=1e-6):
            raise RuntimeError("GLINT DM readback differs from requested PTT state")
        last_dm_command = time.monotonic()
        guard = ADC_MAX_ADU * SATURATION_FRACTION
        while metadata["frames_acquired"] < NFRAMES:
            if time.monotonic() - last_dm_command >= DM_KEEPALIVE_INTERVAL_MIN * 60:
                hardware.set_dm(state)  # GLINT dmptt keepalive; no SSH/SCExAO DM.
                last_dm_command = time.monotonic()
                time.sleep(DM_KEEPALIVE_SETTLE_S)
            count = min(FRAMES_PER_BATCH, NFRAMES - metadata["frames_acquired"])
            # The first frame belongs to N; there are no unrecorded probe frames.
            frames = frame_cube(hardware.apapane_frames(count), count, dark.shape)
            raw_max = float(np.max(frames))
            if raw_max >= guard:
                raise SaturationDetected(f"raw APAPANE maximum {raw_max:g} ADU >= {guard:g} ADU")
            batch = spectra_for_frames(frames, dark, boxes)
            for name in spectra:
                spectra[name] = np.concatenate((spectra[name], batch[name]), axis=0)
            metadata["frames_acquired"] += count
            metadata["max_raw_adu"] = max(metadata["max_raw_adu"], raw_max)
            save_fits(path, spectra, boxes, channels, state, metadata)
            print(f"  Saved {metadata['frames_acquired']}/{NFRAMES} frames; raw max {raw_max:g} ADU")
        if not np.allclose(dm_state_to_array(hardware.get_dm()), dm_state_to_array(state), atol=1e-6):
            raise RuntimeError("GLINT DM readback changed during acquisition")
        metadata["status"] = "COMPLETE"
    except BaseException as exc:
        error = exc
        metadata["status"] = "ABORTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if hardware is not None:
            if initial_dm is not None and commanded:
                try:
                    hardware.set_dm(initial_dm)
                except BaseException as exc:
                    restore_errors.append(f"GLINT DM: {exc}")
            if initial_mount is not None:
                try:
                    hardware.move_mount({"x": float(initial_mount["x"]),
                                         "y": float(initial_mount["y"])},
                                        float(instrument["safety"]["motion_timeout_s"]))
                except BaseException as exc:
                    restore_errors.append(f"mount X/Y: {exc}")
            try:
                hardware.close()
            except BaseException as exc:
                restore_errors.append(f"hardware close: {exc}")
        if restore_errors:
            metadata["status"] = "RESTFAIL"
            metadata["error"] = "; ".join(restore_errors)
        try:
            save_fits(path, spectra, boxes, channels, state, metadata)
        except BaseException as save_error:
            if path.exists() and path.stat().st_size == 0:
                path.unlink()
            raise RuntimeError(f"Could not save acquisition to {path}") from save_error
    if restore_errors:
        raise RuntimeError(f"Restore failed: {restore_errors}; acquisition error: {error}")
    if error is not None:
        raise error
    print(f"Completed: {path}")
    return path


def main() -> None:
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("Interrupted by operator", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
