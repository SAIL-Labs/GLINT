#!/usr/bin/env python3
"""Randomized eight-piston GLINT spectral scan, one FITS per piston/repeat.

Edit USER SETTINGS and check with --check-only. With --input all (the default),
acquire 11, 20, 31 in one camera-gain mode, then change the camera gain and
run again in the other mode. The script only READS the APAPANE camera stream.
X/Y mount move; roll, pitch, yaw and Z stay at their original coordinates.
The original mount X/Y is restored on exit. The script commands the DM to
SEGMENT_PTT (the tilted-away state) on exit, including after interruptions.
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


import csv
import json

# ============================ USER SETTINGS ================================
# Eight commanded SURFACE piston offsets in micrometres, added to the scanned
# segment's injection piston. In reflection optical path difference is 2*piston.
# Example is 8 steps across one phase cycle at 1600 nm.
PISTON_OFFSETS_UM = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
N_REPEATS = 5
FRAMES_PER_POINT = 1000
FRAMES_PER_BATCH = 50
CHECKPOINT_EVERY_FRAMES = 250  # Partial FITS checkpoint frequency per point.
RANDOM_SEED: int | None = None  # None: generated and saved to scan_order.csv

# Each illuminated microlens: enter X/Y mount pulses, scanned segment, and
# its injection-optimized (piston_um, tip_mrad, tilt_mrad). The injection PTT
# replaces SEGMENT_PTT for that segment ONLY during that microlens's scan.
# None forces manual entry; edit all three when using --input all.
MICROLENSES = {
    11: {"x": 3038, "y": 3278, "segment": 31,
         "injection_ptt": (0.0, 0.07, 0.27), "label": "ml11"},
    20: {"x": 2940, "y": 3612, "segment": 11,
         "injection_ptt": (0.0, -0.09, 0.12), "label": "ml20"},
    31: {"x": 3059, "y": 3667, "segment": 11,
         "injection_ptt": (0.0, -0.11, 0.38), "label": "ml31"},
}
# The user-facing camera setting is a label, not an ADU conversion. Supply a
# measured conversion for EACH mode: ADU per electron (not electrons per ADU).
# Use a matching dark for each gain/exposure. If exposure differs, rates are
# compared in electrons / second. Do not infer conversion from gain-setting ID.
CAMERA = {
    "high": {"gain_setting": "", "adu_per_electron": None,
             "exposure_s": None, "dark_path": None,
             "saturation_guard_adu": 2400000.0},
    "low":  {"gain_setting": "", "adu_per_electron": None,
             "exposure_s": None, "dark_path": None,
             "saturation_guard_adu": 2400000.0},
}
# TILTED-AWAY state for ALL 37 segments: surface piston [um], tip and tilt
# [mrad]. This is the full DM state between microlenses and at scan exit.
# In particular, put tilted-away PTTs at segments 11, 20 and 31 here.
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
FITS_LABEL = "piston_scan"  # Short user-chosen identifier for the run directory
DETECTOR_BOXES_PATH: str | None = None  # default: adjacent JSON
OUTPUT_FOLDER = "piston_scan_spectra"
MOUNT_SETTLE_S = 0.1
DM_SETTLE_S = 0.1
DM_KEEPALIVE_INTERVAL_MIN = 15.0
DM_KEEPALIVE_SETTLE_S = 0.1
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



def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", choices=("all", "11", "20", "31"), default="all")
    parser.add_argument("--gain-mode", choices=("high", "low"), required=True)
    parser.add_argument("--instrument-config", type=Path, default=DEFAULT_INSTRUMENT_CONFIG)
    parser.add_argument("--boxes-config", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip initial operator confirmation")
    return parser.parse_args(argv)


def validate_settings(mode: str, selected: list[int]) -> tuple[dict[int, SegmentPTT], dict[str, Any]]:
    if len(PISTON_OFFSETS_UM) != 8 or len(set(map(float, PISTON_OFFSETS_UM))) != 8 \
            or not np.all(np.isfinite(PISTON_OFFSETS_UM)):
        raise ValueError("PISTON_OFFSETS_UM must contain eight distinct finite values")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in
           (N_REPEATS, FRAMES_PER_POINT, FRAMES_PER_BATCH, CHECKPOINT_EVERY_FRAMES)):
        raise ValueError("Repeat, frame, batch and checkpoint counts must be positive integers")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", FITS_LABEL):
        raise ValueError("Set a valid FITS_LABEL (letters, digits, hyphen, underscore)")
    if set(SEGMENT_PTT) != set(range(37)) or set(MICROLENSES) != {11, 20, 31}:
        raise ValueError("Specify all 37 DM segments and microlenses 11, 20, 31")
    state = {}
    for i in range(37):
        row = SEGMENT_PTT[i]
        if len(row) != 3 or not np.all(np.isfinite(row)):
            raise ValueError(f"Invalid piston/tip/tilt for segment {i}")
        state[i] = SegmentPTT(*map(float, row))
    for source in selected:
        entry = MICROLENSES[source]
        for axis in ("x", "y"):
            if isinstance(entry[axis], bool) or not isinstance(entry[axis], (int, np.integer)):
                raise ValueError(f"Edit MICROLENSES[{source}]['{axis}'] (integer mount pulses)")
        segment = entry["segment"]
        if isinstance(segment, bool) or not isinstance(segment, (int, np.integer)) or segment not in state:
            raise ValueError(f"Edit MICROLENSES[{source}]['segment'] (0..36)")
        injection = entry["injection_ptt"]
        if not isinstance(injection, (tuple, list)) or len(injection) != 3:
            raise ValueError(f"Set MICROLENSES[{source}]['injection_ptt'] to (piston_um, tip_mrad, tilt_mrad)")
        try:
            injection_values = np.asarray(injection, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid injection PTT for ML{source}") from exc
        if not np.all(np.isfinite(injection_values)):
            raise ValueError(f"Injection PTT for ML{source} must be finite")
        if np.allclose(injection_values[1:],
                       [state[segment].tip_mrad, state[segment].tilt_mrad],
                       atol=1e-9, rtol=0):
            raise ValueError(f"ML{source}: injection tip/tilt equals its tilted-away tip/tilt in SEGMENT_PTT")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", str(entry["label"])):
            raise ValueError(f"Invalid microlens label for {source}")
    cam = CAMERA[mode]
    if cam["gain_setting"] is None or str(cam["gain_setting"]).strip() == "":
        raise ValueError(f"Set CAMERA[{mode!r}]['gain_setting'] to the actual camera setting")
    for key in ("adu_per_electron", "exposure_s", "saturation_guard_adu"):
        value = cam[key]
        if value is None or not np.isfinite(value) or float(value) <= 0:
            raise ValueError(f"Set a positive calibrated CAMERA[{mode!r}][{key!r}]")
    if not cam["dark_path"]:
        raise ValueError(f"Set CAMERA[{mode!r}]['dark_path'] to a same-gain/exposure dark")
    if DM_KEEPALIVE_INTERVAL_MIN <= 0 or min(MOUNT_SETTLE_S, DM_SETTLE_S, DM_KEEPALIVE_SETTLE_S) < 0:
        raise ValueError("Invalid settling/keepalive settings")
    return state, cam


def commanded_state(base: dict[int, SegmentPTT], segment: int,
                    injection_ptt: tuple[float, float, float], offset: float) -> dict[int, SegmentPTT]:
    state = dict(base)
    injection_piston, injection_tip, injection_tilt = map(float, injection_ptt)
    state[segment] = SegmentPTT(injection_piston + float(offset), injection_tip, injection_tilt)
    return state


def spectra_and_sat(frames: np.ndarray, dark: np.ndarray, boxes: dict[str, tuple[int,int,int,int]],
                    threshold: float) -> tuple[dict[str,np.ndarray],dict[str,np.ndarray]]:
    spectra = spectra_for_frames(frames, dark, boxes)
    masks = {}
    for name, (y0, y1, x0, x1) in boxes.items():
        masks[name] = np.any(frames[:, y0:y1, x0:x1] >= threshold, axis=1).astype(np.uint8)
    return spectra, masks


def save_point(path: Path, spectra: dict[str,np.ndarray], masks: dict[str,np.ndarray],
               boxes: dict[str,tuple[int,int,int,int]], channels: list[Channel],
               state: dict[int,SegmentPTT], meta: dict[str,Any]) -> None:
    header = fits.Header()
    for key, value in meta.items():
        if key != "history":
            header[key] = value
    for item in meta.get("history", []):
        header.add_history(str(item))
    hdus = [fits.PrimaryHDU(header=header)]
    for name, group, ext, baseline in channels:
        y0,y1,x0,x1 = boxes[name]
        hdu = fits.ImageHDU(spectra[name], name=ext)
        hdu.header["CHANNEL"], hdu.header["GROUP"], hdu.header["BASELINE"] = name, group, baseline
        for key, value in (("Y0",y0),("Y1",y1),("X0",x0),("X1",x1)):
            hdu.header[key] = value
        hdu.header["BUNIT"] = "dark-subtracted ADU / column / frame"
        hdus.append(hdu)
        sat_hdu = fits.ImageHDU(masks[name], name="SAT_" + ext)
        sat_hdu.header["BUNIT"] = "1 if any raw ROI pixel >= ADCGUARD"
        hdus.append(sat_hdu)
    hdus.append(fits.BinTableHDU.from_columns([
        fits.Column(name="SEGMENT", format="I", array=np.arange(37,dtype=np.int16)),
        fits.Column(name="PISTON_UM", format="D", array=[state[i].piston_um for i in range(37)]),
        fits.Column(name="TIP_MRAD", format="D", array=[state[i].tip_mrad for i in range(37)]),
        fits.Column(name="TILT_MRAD", format="D", array=[state[i].tilt_mrad for i in range(37)]),
    ], name="DM_PTT"))
    tmp = path.with_name("." + path.name + ".tmp")
    try:
        fits.HDUList(hdus).writeto(tmp, overwrite=True, checksum=True)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> Path | None:
    selected = [11,20,31] if args.input == "all" else [int(args.input)]
    base, cam = validate_settings(args.gain_mode, selected)
    selected_boxes_path = boxes_path(args.boxes_config)
    boxes, channels = load_boxes(selected_boxes_path)
    instrument_path = args.instrument_config.expanduser().resolve()
    instrument = load_config(instrument_path, simulation=False)
    root = alignment_root(instrument)
    dark_path = Path(cam["dark_path"]).expanduser()
    dark_path = (dark_path if dark_path.is_absolute() else SCRIPT_DIRECTORY / dark_path).resolve()
    if not dark_path.is_file():
        raise FileNotFoundError(dark_path)
    dark = load_dark(dark_path)
    dark_sha = sha256(dark_path)
    boxes_sha = sha256(selected_boxes_path)
    for name, (_,y1,_,x1) in boxes.items():
        if y1 > dark.shape[0] or x1 > dark.shape[1]:
            raise ValueError(f"{name} exceeds dark image {dark.shape}")
    seed = int(RANDOM_SEED) if RANDOM_SEED is not None else int.from_bytes(os.urandom(8), "big")
    rng = np.random.default_rng(seed)
    # Independent permutation for every repeat and microlens; exactly one visit
    # to each piston per repeat.
    order = [(source, rep, int(pi), float(PISTON_OFFSETS_UM[pi]))
             for source in selected for rep in range(N_REPEATS)
             for pi in rng.permutation(8)]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = root / datetime.now().astimezone().strftime("%Y%m%d") / OUTPUT_FOLDER / f"{FITS_LABEL}_{args.gain_mode}_{stamp}"
    print(f"Run: {path}; inputs {selected}; mode {args.gain_mode}; camera setting {cam['gain_setting']}")
    print(f"ADU/e- {cam['adu_per_electron']}; exposure {cam['exposure_s']} s; saturation guard {cam['saturation_guard_adu']} ADU")
    print(f"Dark {dark_path}; boxes {selected_boxes_path}; seed {seed}")
    print(f"Pistons {PISTON_OFFSETS_UM} um, repeats {N_REPEATS}, frames/point {FRAMES_PER_POINT}")
    for source in selected:
        print(f"ML{source}: {MICROLENSES[source]}")
    print("Tilted-away DM PTT (used between microlenses and at exit):")
    for i in range(37):
        print(f"  {i:02d}: {SEGMENT_PTT[i]}")
    if args.check_only:
        return None
    if not args.yes and (not sys.stdin.isatty() or input("Verify gain, matched dark, mount, DM and saturation guard; type YES: ").strip() != "YES"):
        raise RuntimeError("Operator did not confirm acquisition")
    path.mkdir(parents=True, exist_ok=False)
    with (path/"scan_order.csv").open("w", newline="", encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["sequence", "input", "repeat", "piston_index", "piston_offset_um"])
        for sequence, row in enumerate(order):
            writer.writerow([sequence,*row])
    (path/"run_metadata.json").write_text(json.dumps({
        "label":FITS_LABEL,"gain_mode":args.gain_mode,"gain_setting":cam["gain_setting"],
        "adu_per_electron":cam["adu_per_electron"],"exposure_s":cam["exposure_s"],
        "dark_file":str(dark_path),"dark_sha256":dark_sha,
        "boxes_file":str(selected_boxes_path),"boxes_sha256":boxes_sha,
        "piston_offsets_um":PISTON_OFFSETS_UM,"repeats":N_REPEATS,
        "frames_per_point":FRAMES_PER_POINT,"seed":seed,"input_order":selected,
        "microlenses":{str(k):MICROLENSES[k] for k in selected}
    }, indent=2,default=int)+"\n",encoding="utf-8")
    hardware=None
    initial_mount=None
    commanded=False
    restore_errors=[]
    try:
        hardware=InstrumentHardware(instrument)
        limits=hardware.mount_limits()
        for source in selected:
            for axis in ("x","y"):
                value=float(MICROLENSES[source][axis])
                lo,hi=map(float,limits[axis])
                if not lo <= value <= hi:
                    raise ValueError(f"ML{source} {axis}={value} outside [{lo}, {hi}]")
        initial_mount=hardware.get_mount()
        current_source=None
        last_dm_command=0.0
        for sequence,(source,rep,pi,piston) in enumerate(order):
            entry=MICROLENSES[source]
            if source != current_source:
                # First tilt EVERY segment away, including the previously
                # injected one. Keep this state while moving between inputs.
                commanded=True
                hardware.set_dm(base)
                time.sleep(DM_SETTLE_S)
                if not np.allclose(dm_state_to_array(hardware.get_dm()),
                                   dm_state_to_array(base), atol=1e-6):
                    raise RuntimeError("Tilted-away GLINT DM state readback mismatch")
                hardware.move_mount({"x":float(entry["x"]),"y":float(entry["y"])},
                                    float(instrument["safety"]["motion_timeout_s"]))
                time.sleep(MOUNT_SETTLE_S)
                actual=hardware.get_mount()
                if abs(actual["x"]-entry["x"])>1 or abs(actual["y"]-entry["y"])>1:
                    raise RuntimeError(f"Mount did not reach ML{source}: {actual}")
                current_source=source
            state=commanded_state(base,int(entry["segment"]),entry["injection_ptt"],piston)
            hardware.set_dm(state)
            time.sleep(DM_SETTLE_S)
            if not np.allclose(dm_state_to_array(hardware.get_dm()),dm_state_to_array(state),atol=1e-6):
                raise RuntimeError(f"GLINT DM readback mismatch at sequence {sequence}")
            last_dm_command=time.monotonic()
            point=path/f"{entry['label']}_{args.gain_mode}_r{rep:02d}_p{pi:02d}.fits"
            spectra={name:np.empty((0,b[3]-b[2]),dtype=np.float32) for name,b in boxes.items()}
            masks={name:np.empty((0,b[3]-b[2]),dtype=np.uint8) for name,b in boxes.items()}
            meta={"RUNSTAT":"PARTIAL","LABEL":FITS_LABEL,"RUNID":path.name,
                  "INPUT":int(source),"GAINMODE":args.gain_mode,"GAINSET":str(cam["gain_setting"]),
                  "ADUPERE":float(cam["adu_per_electron"]),"EXPOSURE":float(cam["exposure_s"]),
                  "NREQ":FRAMES_PER_POINT,"NACQ":0,"REPEAT":int(rep),"PINDEX":int(pi),
                  "PISTUM":float(piston),"SEGSCAN":int(entry["segment"]),"SEQ":int(sequence),
                  "INJPUM":float(entry["injection_ptt"][0]),
                  "INJTIP":float(entry["injection_ptt"][1]),
                  "INJTLT":float(entry["injection_ptt"][2]),
                  "AWAYPUM":float(base[int(entry["segment"])].piston_um),
                  "AWAYTIP":float(base[int(entry["segment"])].tip_mrad),
                  "AWAYTLT":float(base[int(entry["segment"])].tilt_mrad),
                  "SEED":str(seed),"XMOUNT":int(entry["x"]),"YMOUNT":int(entry["y"]),
                  "DATE-OBS":datetime.now(timezone.utc).isoformat(),"ADCGUARD":float(cam["saturation_guard_adu"]),
                  "MAXRAW":0.0,"DARKSHA":dark_sha,"BOXSHA":boxes_sha,
                  "history":[f"Dark: {dark_path}",f"Boxes: {selected_boxes_path}",
                             "Injection PTT overrides tilted-away SEGMENT_PTT for SEGSCAN only.",
                             "PISTUM added to injection surface piston; reflected OPD = 2*PISTUM.",
                             "Camera gain setting recorded manually; camera stream read only."]}
            try:
                while meta["NACQ"]<FRAMES_PER_POINT:
                    if time.monotonic()-last_dm_command >= DM_KEEPALIVE_INTERVAL_MIN*60:
                        hardware.set_dm(state)
                        last_dm_command=time.monotonic()
                        time.sleep(DM_KEEPALIVE_SETTLE_S)
                    count=min(FRAMES_PER_BATCH,FRAMES_PER_POINT-meta["NACQ"])
                    frames=frame_cube(hardware.apapane_frames(count),count,dark.shape)
                    batch,sat=spectra_and_sat(frames,dark,boxes,float(cam["saturation_guard_adu"]))
                    for name in spectra:
                        spectra[name]=np.concatenate((spectra[name],batch[name]),axis=0)
                        masks[name]=np.concatenate((masks[name],sat[name]),axis=0)
                    meta["NACQ"]+=count
                    meta["MAXRAW"]=max(meta["MAXRAW"],float(np.max(frames)))
                    if (meta["NACQ"] % CHECKPOINT_EVERY_FRAMES == 0
                            or meta["NACQ"] == FRAMES_PER_POINT):
                        save_point(point,spectra,masks,boxes,channels,state,meta)
                if not np.allclose(dm_state_to_array(hardware.get_dm()),dm_state_to_array(state),atol=1e-6):
                    raise RuntimeError("DM readback changed during point")
                meta["RUNSTAT"]="COMPLETE"
                meta["DATE-END"]=datetime.now(timezone.utc).isoformat()
                save_point(point,spectra,masks,boxes,channels,state,meta)
                print(f"{sequence+1}/{len(order)} ML{source} repeat {rep} piston {piston:g} um, max raw {meta['MAXRAW']:g}")
            except BaseException:
                # Last checkpoint stays PARTIAL; keep it for inspection.
                raise
    finally:
        if hardware is not None:
            if commanded:
                try: hardware.set_dm(base)
                except BaseException as exc: restore_errors.append(f"DM: {exc}")
            if initial_mount is not None:
                try:
                    hardware.move_mount({"x":float(initial_mount["x"]),"y":float(initial_mount["y"])},
                                        float(instrument["safety"]["motion_timeout_s"]))
                except BaseException as exc: restore_errors.append(f"mount: {exc}")
            try: hardware.close()
            except BaseException as exc: restore_errors.append(f"close: {exc}")
        if restore_errors:
            raise RuntimeError(f"Restoration failure: {restore_errors}")
    print(f"Completed: {path}")
    return path


def main() -> None:
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("Interrupted by operator",file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}",file=sys.stderr)
        raise SystemExit(1)


if __name__=="__main__":
    main()
