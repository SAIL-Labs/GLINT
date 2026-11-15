#!/usr/bin/env python3
"""Sample chosen GLINT mount X/Y points with repeated GLINT-DM piston sweeps.

One JSON per point contains dark-subtracted per-frame photometric column sums
as [spectral_column, photometry_channel, piston, frame, repeat]. At each XY
point, one full trial sweep is acquired and discarded before recorded repeats.
Recorded piston orders are shuffled and balanced across acquisition slots;
the cube always uses the original PISTON_FRACTIONS order. The camera is only
read from its already-running APAPANE shared-memory stream; it is never set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import select
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# USER SETTINGS: replace these seven example absolute mount coordinates.
# Dictionary insertion order is acquisition order.
# Entries are (X, Y, selected_tip_mrad, selected_tilt_mrad, (segments, ...)).
# The listed segments receive these tip/tilt values and the SAME surface-
# piston offset together; other segments retain their SEGMENT_PTT settings.
# Existing three-field (X, Y, segment) entries still work unchanged.
# All listed segments receive the SAME surface-piston offset in one DM command.
# Example: "point_01": (3060, 3670, 0.0, 0.0, (11, 20, 31)),
# ---------------------------------------------------------------------------
XY_POSITIONS: dict[str, tuple[int, int, float, float, tuple[int, ...]]] = {
    "point_01": (2820, 3490, 0.0, 0.0, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36)),
}


# (3060, 3670, 0.0, 0.0, 11)
# (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36)
# (5, 6, 10, 11, 12, 17, 18)
# (23, 24, 28, 29, 30, 33, 34)
# (16, 17, 18, 22, 23, 24, 25, 28, 29, 30, 31, 33, 34, 35)
# (0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 23, 24, 25)

# Surface-piston units: lambda in nm; the mirror reflects light twice.
WAVELENGTH_NM = 1500.0
PISTON_FRACTIONS = (0.0, 1 / 12, 1 / 6, 1 / 4, 1 / 3, 5 / 12)
NFRAMES = 200
N_REPEATS = 10  # Recorded full piston sweeps at EACH XY point, after one trial.
RANDOM_SEED: int | None = None  # None: new seed per run; actual seed is saved.
# The discarded trial uses NFRAMES at every piston, just like each repeat.
PREVIEW_NFRAMES = 10  # At the first XY position before any piston sweep.

# Actual defaults below command all 37 segments (piston,tip,tilt) as (0,0,0):
# all segments FLAT. This is NOT a tilt-away detector safeguard. Edit the
# individual PTT rows as needed before running at high flux.
SEGMENT_PTT: dict[int, tuple[float, float, float]] = {
    segment: (0.0, 0, 0.0) for segment in range(37)
}

# Prefer photometry_boxes.json, as requested. If absent, use the same
# detector_boxes.json imported by the attached scan and warn about its width.
PHOTOMETRY_BOXES_PATH: str | None = None
DARK_PATH: str | None = None
SETTLE_S = 0.02
PISTON_SETTLE_S = 0.10
DM_KEEPALIVE_INTERVAL_MIN = 15.0  # GLINT dmptt only; never SSH/SCExAO DM.
DM_KEEPALIVE_SETTLE_S = 0.10
ADC_MAX_ADU = 3000000.0  # Preserve uploaded scan's ADC setting; verify mode.
SATURATION_FRACTION = 0.80
OUTPUT_FOLDER = "Alignment_output_xy_spextral"


SCRIPT_DIRECTORY = Path(__file__).resolve().parent


def _find_refactor_root() -> Path:
    configured = os.environ.get("GLINT_ALIGNMENT_REFACTOR_ROOT")
    candidates = [
        Path(configured).expanduser() if configured else None,
        SCRIPT_DIRECTORY,
        SCRIPT_DIRECTORY.parents[1] / "glint_alignment_refactor"
        if len(SCRIPT_DIRECTORY.parents) > 1 else None,
        SCRIPT_DIRECTORY.parent / "glint_alignment_refactor",
        Path.cwd(),
        Path.cwd() / "glint_alignment_refactor",
    ]
    tried: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        root = candidate.resolve()
        tried.append(str(root))
        if (root / "glint_alignment" / "__init__.py").is_file():
            return root
    raise ModuleNotFoundError(
        f"Could not find glint_alignment_refactor; tried {tried}. "
        "Set GLINT_ALIGNMENT_REFACTOR_ROOT to its absolute directory."
    )


GLINT_ALIGNMENT_REFACTOR_ROOT = _find_refactor_root()
if str(GLINT_ALIGNMENT_REFACTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(GLINT_ALIGNMENT_REFACTOR_ROOT))

from glint_alignment.config import load_config
from glint_alignment.instrument import InstrumentHardware
from glint_alignment.models import SegmentPTT, dm_state_to_array, dm_state_to_dict

DEFAULT_INSTRUMENT_CONFIG = GLINT_ALIGNMENT_REFACTOR_ROOT / "config" / "instrument.json"


class SaturationDetected(RuntimeError):
    """Raw detector pixel reached the configured guard threshold."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument-config", type=Path, default=DEFAULT_INSTRUMENT_CONFIG)
    parser.add_argument("--boxes-config", type=Path, help="Override photometry_boxes.json path")
    parser.add_argument("--check-only", action="store_true", help="Validate/print plan; no hardware access")
    parser.add_argument("--yes", action="store_true", help="Skip initial YES prompt only")
    return parser.parse_args(argv)


def resolve_boxes_path(override: Path | None) -> Path:
    if override is not None or PHOTOMETRY_BOXES_PATH is not None:
        chosen = override if override is not None else Path(PHOTOMETRY_BOXES_PATH)
        assert chosen is not None
        path = chosen.expanduser()
        return (path if path.is_absolute() else SCRIPT_DIRECTORY / path).resolve()
    candidates = (
        SCRIPT_DIRECTORY / "config" / "photometry_boxes.json",
        SCRIPT_DIRECTORY / "photometry_boxes.json",
        GLINT_ALIGNMENT_REFACTOR_ROOT / "config" / "photometry_boxes.json",
        GLINT_ALIGNMENT_REFACTOR_ROOT / "config" / "detector_boxes.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("No photometry_boxes.json or refactor detector_boxes.json found")


def load_photometry_boxes(path: Path) -> dict[str, tuple[int, int, int, int]]:
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    entries = document.get("photometry", document)
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"No photometry channel boxes in {path}")
    boxes: dict[str, tuple[int, int, int, int]] = {}
    for channel, bounds in entries.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
            raise ValueError(f"{channel!r} must be [y0,y1,x0,x1]")
        integer_bounds = tuple(int(v) for v in bounds)
        if any(float(v) != float(i) for v, i in zip(bounds, integer_bounds)):
            raise ValueError(f"{channel!r} box bounds must be integer pixels")
        y0, y1, x0, x1 = integer_bounds
        if min(y0, x0) < 0 or y1 <= y0 or x1 <= x0:
            raise ValueError(f"Invalid {channel!r} box: {bounds}")
        boxes[str(channel)] = integer_bounds
    if set(boxes) != {"Phot 1", "Phot 2", "Phot 3"}:
        raise ValueError("Photometry boxes must contain exactly Phot 1, Phot 2 and Phot 3")
    return {channel: boxes[channel] for channel in ("Phot 1", "Phot 2", "Phot 3")}


def validated_points() -> list[tuple[str, int, int, float | None, float | None, tuple[int, ...]]]:
    if not isinstance(XY_POSITIONS, dict) or not XY_POSITIONS:
        raise ValueError("XY_POSITIONS must be a non-empty dictionary")
    positions: list[tuple[str, int, int, float | None, float | None, tuple[int, ...]]] = []
    for name, pair in XY_POSITIONS.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"Invalid position name {name!r}; use letters, digits, _ or -")
        if not isinstance(pair, (tuple, list)) or len(pair) not in (3, 5):
            raise ValueError(
                f"XY_POSITIONS[{name!r}] must be (x,y,segment_ids) or "
                "(x,y,tip_mrad,tilt_mrad,segment_ids)"
            )
        if not all(np.isfinite(float(v)) and float(v).is_integer() for v in pair[:2]):
            raise ValueError(f"Mount X/Y must be integers at {name}")
        x, y = int(pair[0]), int(pair[1])
        tip_mrad: float | None = None
        tilt_mrad: float | None = None
        if len(pair) == 5:
            try:
                tip_mrad, tilt_mrad = float(pair[2]), float(pair[3])
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"Selected tip/tilt must be finite numbers at {name}") from error
            if not np.isfinite(tip_mrad) or not np.isfinite(tilt_mrad):
                raise ValueError(f"Selected tip/tilt must be finite numbers at {name}")
        selection = pair[-1]
        selected = selection if isinstance(selection, (tuple, list)) else (selection,)
        if not selected:
            raise ValueError(f"No GLINT DM segments selected at {name}")
        segments: list[int] = []
        for value in selected:
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"Invalid GLINT DM segment {value!r} at {name}") from error
            if not np.isfinite(numeric) or not numeric.is_integer():
                raise ValueError(f"GLINT DM segments must be integer IDs at {name}")
            segment = int(numeric)
            if not 0 <= segment < 37:
                raise ValueError(f"Invalid GLINT DM segment {segment} at {name}")
            if segment in segments:
                raise ValueError(f"Repeated GLINT DM segment {segment} at {name}")
            segments.append(segment)
        # Different named experiments may intentionally share an XY coordinate
        # while piston-scanning different groups (or repeating the same group).
        positions.append((name, x, y, tip_mrad, tilt_mrad, tuple(segments)))
    for setting, value in (("NFRAMES", NFRAMES), ("N_REPEATS", N_REPEATS),
                           ("PREVIEW_NFRAMES", PREVIEW_NFRAMES)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{setting} must be an integer >= 1")
    if RANDOM_SEED is not None and (
        isinstance(RANDOM_SEED, (bool, np.bool_))
        or not isinstance(RANDOM_SEED, (int, np.integer)) or RANDOM_SEED < 0
    ):
        raise ValueError("RANDOM_SEED must be None or a nonnegative integer")
    if not np.isfinite(WAVELENGTH_NM) or WAVELENGTH_NM <= 0:
        raise ValueError("WAVELENGTH_NM must be positive and finite")
    if not PISTON_FRACTIONS or any(not np.isfinite(v) for v in PISTON_FRACTIONS):
        raise ValueError("PISTON_FRACTIONS must contain at least one finite value")
    if len(set(PISTON_FRACTIONS)) != len(PISTON_FRACTIONS):
        raise ValueError("PISTON_FRACTIONS must not contain duplicate values; use N_REPEATS")
    if set(SEGMENT_PTT) != set(range(37)):
        raise ValueError("SEGMENT_PTT must contain all 37 GLINT DM segment IDs")
    for segment, row in SEGMENT_PTT.items():
        if len(row) != 3 or not np.all(np.isfinite(row)):
            raise ValueError(f"SEGMENT_PTT[{segment}] must contain three finite PTT values")
    if min(SETTLE_S, PISTON_SETTLE_S, DM_KEEPALIVE_SETTLE_S) < 0:
        raise ValueError("Settling times must be nonnegative")
    if DM_KEEPALIVE_INTERVAL_MIN <= 0 or ADC_MAX_ADU <= 0 or not 0 < SATURATION_FRACTION < 1:
        raise ValueError("Invalid DM keepalive or ADC guard configuration")
    return positions


def base_dm_state() -> dict[int, SegmentPTT]:
    return {
        index: SegmentPTT(*map(float, SEGMENT_PTT[index]))
        for index in range(37)
    }


def state_for_point(
    base: dict[int, SegmentPTT], segments: Sequence[int],
    tip_mrad: float | None, tilt_mrad: float | None,
) -> dict[int, SegmentPTT]:
    """Apply per-point selected tip/tilt without changing other segments."""
    state = {index: SegmentPTT(v.piston_um, v.tip_mrad, v.tilt_mrad) for index, v in base.items()}
    if tip_mrad is not None and tilt_mrad is not None:
        for segment in segments:
            state[int(segment)].tip_mrad = tip_mrad
            state[int(segment)].tilt_mrad = tilt_mrad
    return state


def state_for_piston(
    base: dict[int, SegmentPTT], offset_um: float, segments: Sequence[int]
) -> dict[int, SegmentPTT]:
    state = {index: SegmentPTT(v.piston_um, v.tip_mrad, v.tilt_mrad) for index, v in base.items()}
    for segment in segments:
        state[int(segment)].piston_um += float(offset_um)
    return state


def surface_piston_offsets_um() -> list[float]:
    return [float(WAVELENGTH_NM) / 1000 * float(fraction) for fraction in PISTON_FRACTIONS]


def piston_acquisition_orders(
    npistons: int, nrepeats: int, rng: np.random.Generator
) -> list[list[int]]:
    """Randomized cyclic blocks: each piston visits each slot once per block.

    A block contains npistons repeats. Partial blocks still use distinct
    slots for each piston. Adjacent orders differ unless there is one piston.
    These orders describe acquisition only, never the stored piston axis.
    """
    orders: list[list[int]] = []
    while len(orders) < nrepeats:
        base_order = rng.permutation(npistons)
        shifts = rng.permutation(npistons).tolist()
        if orders and npistons > 1 and np.roll(base_order, shifts[0]).tolist() == orders[-1]:
            shifts[0], shifts[1] = shifts[1], shifts[0]
        for shift in shifts:
            orders.append(np.roll(base_order, shift).tolist())
            if len(orders) == nrepeats:
                break
    return orders


def frame_cube(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim == 2:
        result = result[None, :, :]
    if result.ndim != 3:
        raise ValueError(f"Expected a 2D APAPANE frame or 3D frame cube; got {result.shape}")
    return result


def dark_image(values: np.ndarray) -> np.ndarray:
    dark = np.asarray(values, dtype=float)
    if dark.ndim == 3:
        dark = np.nanmean(dark, axis=0)
    if dark.ndim != 2:
        raise ValueError(f"Expected a 2D/3D saved APAPANE dark, got {dark.shape}")
    return dark


def alignment_root(config: dict[str, Any]) -> Path:
    root = Path(config["output_root"]).expanduser()
    return (root if root.is_absolute() else Path(config["_root"]) / root).resolve()


def resolve_dark_path(root: Path) -> Path:
    if DARK_PATH is not None:
        path = Path(DARK_PATH).expanduser()
        path = (path if path.is_absolute() else SCRIPT_DIRECTORY / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"DARK_PATH not found: {path}")
        return path
    for suffix in ("fits", "npy", "npz"):
        candidate = root / "darks" / f"apapane_dark.{suffix}"
        if candidate.is_file():
            return candidate
    dated = [file for suffix in ("fits", "npy", "npz")
             for file in root.glob(f"*/*apapane_dark*/scan_data.{suffix}")]
    if dated:
        return max(dated, key=lambda file: file.stat().st_mtime)
    raise FileNotFoundError(f"No saved APAPANE dark below {root}; specify DARK_PATH")


def load_dark(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return dark_image(np.load(path, allow_pickle=False))
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            for key in ("map__dark", "dark"):
                if key in archive:
                    return dark_image(archive[key])
        raise KeyError(f"No dark array in {path}")
    if path.suffix.lower() == ".fits":
        from astropy.io import fits
        with fits.open(path, memmap=False) as hdus:
            for hdu in hdus:
                if hdu.data is not None and str(hdu.header.get("LONGNAME", "")).lower() in ("map__dark", "dark"):
                    return dark_image(hdu.data)
            for hdu in hdus:
                if hdu.data is not None and str(hdu.name).upper() == "DARK":
                    return dark_image(hdu.data)
            for hdu in hdus:
                if hdu.data is not None:
                    return dark_image(hdu.data)
        raise ValueError(f"No dark image in {path}")
    raise ValueError(f"Unsupported dark file extension {path.suffix}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def column_sums_per_frame(
    frames: np.ndarray, dark: np.ndarray, box: Sequence[int]
) -> np.ndarray:
    """Return [frame, spectral column]: sum over dark-subtracted rows."""
    y0, y1, x0, x1 = map(int, box)
    if y1 > dark.shape[0] or x1 > dark.shape[1]:
        raise ValueError(f"Box {box} exceeds dark image {dark.shape}")
    values = frame_cube(frames)
    if values.shape[1:] != dark.shape:
        raise ValueError(f"Frame shape {values.shape[1:]} differs from dark {dark.shape}")
    raw_columns = values[:, y0:y1, x0:x1].sum(axis=1, dtype=float)
    dark_columns = dark[y0:y1, x0:x1].sum(axis=0, dtype=float)
    result = raw_columns - dark_columns[None, :]
    if not np.all(np.isfinite(result)):
        raise ValueError(f"Non-finite dark-subtracted pixels in box {box}")
    return result


def preview_figure(
    path: Path,
    frames: np.ndarray,
    dark: np.ndarray,
    boxes: dict[str, tuple[int, int, int, int]],
    *,
    label: str,
    x: int,
    y: int,
    segments: Sequence[int],
) -> None:
    """Save detector subimages and their dark-subtracted column sums."""
    import matplotlib.pyplot as plt

    averaged = np.mean(frame_cube(frames), axis=0) - dark
    figure, axes = plt.subplots(2, 3, figsize=(17.5, 7.2), constrained_layout=True)
    for channel_index, (channel, box) in enumerate(boxes.items()):
        y0, y1, x0, x1 = box
        patch = averaged[y0:y1, x0:x1]
        image = axes[0, channel_index].imshow(
            patch,
            origin="lower",
            aspect="auto",
            extent=[x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5],
            interpolation="nearest",
            cmap="magma",
        )
        figure.colorbar(image, ax=axes[0, channel_index], label="mean dark-subtracted ADU/pixel")
        axes[0, channel_index].set_title(f"{channel}: box [{y0},{y1},{x0},{x1}]")
        axes[0, channel_index].set_xlabel("APAPANE detector X [pixel]")
        axes[0, channel_index].set_ylabel("APAPANE detector Y [pixel]")
        vector = column_sums_per_frame(frames, dark, box).mean(axis=0)
        axes[1, channel_index].plot(np.arange(x0, x1), vector, color="#2a759e")
        axes[1, channel_index].axhline(0, color="0.55", linewidth=0.8)
        axes[1, channel_index].set_title(f"{channel}: dark-subtracted column sums")
        axes[1, channel_index].set_xlabel("APAPANE detector X [pixel]")
        axes[1, channel_index].set_ylabel("ADU/column/frame, sum over box rows")
    figure.suptitle(
        f"First-position pilot {label}: mount X={x}, Y={y}, piston segments {list(segments)}; "
        f"mean of {len(frames)} dark-subtracted APAPANE frames"
    )
    figure.savefig(path, dpi=165, facecolor="white")
    plt.close(figure)


def preview_confirmation(
    path: Path, raw_max: float, keepalive_callback: Any
) -> None:
    print(f"\nFIRST-POSITION PREVIEW: {path}")
    print(f"Pilot raw detector maximum: {raw_max:g} ADU")
    print("Open the PNG and check Phot 1/2/3 and the source flux before proceeding.")
    if not sys.stdin.isatty():
        raise RuntimeError("Preview review requires an interactive terminal; scan cannot proceed")
    print("After reviewing the PNG and confirming safe flux, type PREVIEW_OK: ", end="", flush=True)
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], float(DM_KEEPALIVE_INTERVAL_MIN) * 60)
        if ready:
            response = sys.stdin.readline().strip()
            break
        keepalive_callback()  # Keep GLINT DM alive while the operator reviews.
        print("Still waiting for PREVIEW_OK: ", end="", flush=True)
    if response != "PREVIEW_OK":
        raise RuntimeError("Preview not approved; aborting before piston acquisition")


def json_cube(values: np.ndarray) -> list[Any]:
    """Represent unmeasured/padded values as JSON null; never emit NaN tokens."""
    return np.where(np.isfinite(values), values, None).tolist()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def next_output_directory(root: Path) -> Path:
    parent = root / datetime.now().astimezone().strftime("%Y%m%d") / OUTPUT_FOLDER
    parent.mkdir(parents=True, exist_ok=True)
    for index in range(1, 100000):
        candidate = parent / f"normal_xy_spectral_piston_{index:03d}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("No free numbered run directory")


def require_confirmation() -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("Physical scan requires terminal YES confirmation (or --yes)")
    if input("Confirm listed XY positions, simultaneous piston segments, boxes and flux. Type YES: ").strip() != "YES":
        raise RuntimeError("Confirmation not received")


def run(args: argparse.Namespace) -> Path | None:
    positions = validated_points()
    boxes_path = resolve_boxes_path(args.boxes_config)
    boxes = load_photometry_boxes(boxes_path)
    channels = list(boxes)
    widths = [box[3] - box[2] for box in boxes.values()]
    max_width = max(widths)
    offsets = surface_piston_offsets_um()
    random_seed = int(RANDOM_SEED) if RANDOM_SEED is not None else int.from_bytes(os.urandom(16), "big")
    rng = np.random.default_rng(random_seed)
    acquisition_plans = {
        label: {
            "trial_order": rng.permutation(len(offsets)).tolist(),
            "repeat_orders": piston_acquisition_orders(len(offsets), int(N_REPEATS), rng),
        }
        for label, *_ in positions
    }
    instrument_path = args.instrument_config.expanduser().resolve()
    instrument = load_config(instrument_path, simulation=False)
    base = base_dm_state()
    guard_limit = float(ADC_MAX_ADU * SATURATION_FRACTION)
    print("Selected-position GLINT X/Y + spectral piston scan")
    print(f"  XY positions: {positions}")
    print(f"  selected tip/tilt/segments per point: "
          f"{[(tip, tilt, list(segs)) for _, _, _, tip, tilt, segs in positions]}; "
          f"wavelength: {WAVELENGTH_NM:g} nm")
    if all(row[1:] == (0, 0.0) for row in SEGMENT_PTT.values()):
        print("  WARNING: all GLINT DM segments have zero tip/tilt (all-flat base map).")
    print(f"  surface-piston offsets in µm: {[round(v, 6) for v in offsets]}")
    print(f"  reflected phases in degrees: {[round(720 * v, 3) for v in PISTON_FRACTIONS]}")
    print(f"  frames / piston / repeat / position: {NFRAMES}; recorded repeats: {N_REPEATS}")
    print(f"  discarded trial: one full sweep ({len(offsets) * NFRAMES} frames) at EACH XY point")
    print(f"  saved frames per XY point: {len(offsets) * NFRAMES * N_REPEATS}")
    print(f"  randomized balanced piston order; seed: {random_seed}")
    for label, plan in acquisition_plans.items():
        print(f"    {label} recorded orders (zero-based piston indices): {plan['repeat_orders']}")
    print(f"  initial pilot: {PREVIEW_NFRAMES} additional dark-subtracted frames at first XY point; "
          "save PNG and require PREVIEW_OK")
    print(f"  boxes: {boxes_path}: {boxes} (column widths: {widths})")
    print(f"  data cube: [{max_width}, {len(channels)}, {len(offsets)}, {NFRAMES}, {N_REPEATS}]; "
          "axes = spectral column, channel, piston, frame, repeat; unequal widths padded with null")
    print(f"  GLINT DM refresh: every {DM_KEEPALIVE_INTERVAL_MIN:g} min via dmptt; no SSH")
    if boxes_path.name == "detector_boxes.json":
        print("  WARNING: photometry_boxes.json not found; bundled detector_boxes.json "
              "was chosen (not necessarily 7x110). Supply --boxes-config to override.")
    print(f"  raw APAPANE guard: {guard_limit:g} ADU; verify ADC_MAX_ADU for current mode")
    print("  APAPANE: passive frame reads only; per-frame saved-dark subtraction")
    if args.check_only:
        return None
    if not args.yes:
        require_confirmation()

    output_root = alignment_root(instrument)
    dark_path = resolve_dark_path(output_root)
    dark = load_dark(dark_path)
    if not np.all(np.isfinite(dark)):
        raise ValueError(f"Non-finite saved dark in {dark_path}")
    for channel, box in boxes.items():
        if box[1] > dark.shape[0] or box[3] > dark.shape[1]:
            raise ValueError(f"{channel} box {box} exceeds saved dark {dark.shape}")
    output_dir = next_output_directory(output_root)
    timeout = float(instrument["safety"]["motion_timeout_s"])
    fixed_roll = float(instrument["mount_control"]["fixed_roll"])
    hardware: InstrumentHardware | None = None
    before_mount: dict[str, float] | None = None
    before_dm: dict[int, SegmentPTT] | None = None
    active_state = base
    run_clock_start = time.monotonic()
    last_dm_refresh = run_clock_start
    dm_events: list[dict[str, Any]] = []
    run_error: BaseException | None = None
    restore_errors: list[str] = []
    current_json: Path | None = None
    current_payload: dict[str, Any] | None = None
    current_cube: np.ndarray | None = None

    def record_state(state: dict[int, SegmentPTT], reason: str) -> None:
        nonlocal active_state, last_dm_refresh
        assert hardware is not None
        hardware.set_dm(state)  # Refactor checks dmptt readback immediately.
        active_state = state
        last_dm_refresh = time.monotonic()
        dm_events.append({"utc": datetime.now(timezone.utc).isoformat(), "reason": reason})

    def refresh_if_due() -> None:
        if time.monotonic() - last_dm_refresh >= DM_KEEPALIVE_INTERVAL_MIN * 60:
            record_state(active_state, "timed_keepalive")
            time.sleep(DM_KEEPALIVE_SETTLE_S)

    def verify_state() -> None:
        assert hardware is not None
        readback = dm_state_to_array(hardware.get_dm())
        expected = dm_state_to_array(active_state)
        if not np.allclose(readback, expected, atol=1e-6):
            raise RuntimeError("GLINT DM PTT readback changed unexpectedly; aborting")

    def persist_current(status: str, error: str | None = None) -> None:
        if current_json is None or current_payload is None or current_cube is None:
            return
        current_payload["status"] = status
        current_payload["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
        current_payload["dm_command_events"] = list(dm_events)
        if error is not None:
            current_payload["error"] = error
        current_payload["spectral_cube"] = json_cube(current_cube)
        save_json(current_json, current_payload)
        # Retain just the NumPy cube in memory, not a second Python copy.
        del current_payload["spectral_cube"]

    def acquire_guarded_frames(nframes: int, label: str) -> tuple[np.ndarray, float]:
        """Read-only APAPANE access; probe raw full-frame saturation first."""
        assert hardware is not None
        probe = frame_cube(hardware.apapane_frames(1))
        if probe.shape != (1, *dark.shape):
            raise RuntimeError(f"APAPANE probe shape {probe.shape} differs from saved dark {dark.shape}")
        first_max = float(np.nanmax(probe))
        if not np.isfinite(first_max) or first_max >= guard_limit:
            record_state(base, "saturation_park_base_state")
            raise SaturationDetected(f"Raw APAPANE probe reached {first_max:g} ADU at {label}")
        if nframes == 1:
            return probe, first_max
        remaining = frame_cube(hardware.apapane_frames(nframes - 1))
        if remaining.shape != (nframes - 1, *dark.shape):
            raise RuntimeError(f"Requested {nframes-1} further frames; received {remaining.shape}")
        other_max = float(np.nanmax(remaining))
        if not np.isfinite(other_max) or other_max >= guard_limit:
            record_state(base, "saturation_park_base_state")
            raise SaturationDetected(f"Raw APAPANE batch reached {other_max:g} ADU at {label}")
        return np.concatenate((probe, remaining), axis=0), max(first_max, other_max)

    def acquire_piston(
        point_base: dict[int, SegmentPTT], segments: Sequence[int],
        piston_index: int, label: str, reason: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Use exactly the same command, settling and readout for trial/data."""
        refresh_if_due()
        offset = offsets[piston_index]
        command_utc = datetime.now(timezone.utc).isoformat()
        record_state(state_for_piston(point_base, offset, segments), reason)
        time.sleep(PISTON_SETTLE_S)
        verify_state()
        started = time.monotonic()
        started_utc = datetime.now(timezone.utc).isoformat()
        frames, raw_max = acquire_guarded_frames(int(NFRAMES), label)
        finished = time.monotonic()
        finished_utc = datetime.now(timezone.utc).isoformat()
        verify_state()
        return frames, {
            "piston_index": piston_index,
            "piston_offset_surface_um": offset,
            "dm_command_utc": command_utc,
            "acquisition_started_utc": started_utc,
            "acquisition_finished_utc": finished_utc,
            "acquisition_start_elapsed_s": started - run_clock_start,
            "acquisition_end_elapsed_s": finished - run_clock_start,
            "read_duration_s": finished - started,
            "frames": int(NFRAMES),
            "raw_max_adu": raw_max,
        }

    try:
        hardware = InstrumentHardware(instrument)
        limits = hardware.mount_limits()
        for label, x, y, tip_mrad, tilt_mrad, segments in positions:
            for axis, value in (("x", x), ("y", y)):
                low, high = map(float, limits[axis])
                if not low <= value <= high:
                    raise ValueError(f"{label} {axis}={value} is outside mount limits [{low},{high}]")
        before_mount = hardware.get_mount()
        before_dm = hardware.get_dm()
        record_state(base, "base_state_before_mount_travel")
        if round(before_mount["roll"]) != round(fixed_roll):
            hardware.move_mount({"roll": fixed_roll}, timeout)
        previous_xy: tuple[int, int] | None = None
        for point_index, (label, x, y, tip_mrad, tilt_mrad, segments) in enumerate(positions, start=1):
            record_state(base, "base_state_for_transit")
            refresh_if_due()
            targets = {axis: float(value) for axis, value in (("x", x), ("y", y))
                       if previous_xy is None or value != dict(zip(("x", "y"), previous_xy))[axis]}
            if targets:
                hardware.move_mount(targets, timeout)
            previous_xy = (x, y)
            time.sleep(SETTLE_S)
            point_base = state_for_point(base, segments, tip_mrad, tilt_mrad)
            record_state(point_base, "point_selected_tip_tilt_before_preview")
            time.sleep(PISTON_SETTLE_S)
            if point_index == 1:
                verify_state()
                pilot_frames, pilot_max = acquire_guarded_frames(
                    int(PREVIEW_NFRAMES), f"{label} first-position preview"
                )
                verify_state()
                pilot_path = output_dir / "first_position_photometry_preview.png"
                preview_figure(
                    pilot_path, pilot_frames, dark, boxes,
                    label=label, x=x, y=y, segments=segments,
                )
                preview_confirmation(
                    pilot_path, pilot_max,
                    lambda: record_state(point_base, "preview_pause_keepalive"),
                )
                record_state(point_base, "post_preview_point_state")
            current_json = output_dir / f"{label}.json"
            current_cube = np.full(
                (max_width, len(channels), len(offsets), int(NFRAMES), int(N_REPEATS)), np.nan
            )
            plan = acquisition_plans[label]
            current_payload = {
                "schema_version": 3,
                "kind": "normal_xy_spectral_piston",
                "status": "running",
                "position_name": label,
                "position_index": point_index,
                "requested_xy_mount_pulses": {"x": x, "y": y},
                "actual_xy_mount_pulses": {"x": x, "y": y},
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "instrument_config": str(instrument_path),
                "boxes_file": str(boxes_path),
                "boxes_sha256": sha256(boxes_path),
                "photometry_channels_axis": channels,
                "photometry_boxes_y0_y1_x0_x1": {k: list(v) for k, v in boxes.items()},
                "spectral_column_detector_x": {k: list(range(v[2], v[3])) for k, v in boxes.items()},
                "spectral_width_by_channel": dict(zip(channels, widths)),
                "cube_axis_order": ["spectral_column", "photometry_channel", "piston", "frame", "repeat"],
                "cube_shape": list(current_cube.shape),
                "piston_axis_description": "Original PISTON_FRACTIONS order, independent of acquisition order",
                "repeat_axis_description": "Recorded repeats in chronological order; trial excluded; zero-based indices",
                "missing_value": None,
                "spectral_definition": "sum over the dark-subtracted rows of each photometry box, per column per frame",
                "wavelength_nm": float(WAVELENGTH_NM),
                "selected_segments_tip_mrad": tip_mrad,
                "selected_segments_tilt_mrad": tilt_mrad,
                "piston_segment": segments[0] if len(segments) == 1 else None,
                "piston_segments": list(segments),
                "piston_fractions_of_wavelength": [float(v) for v in PISTON_FRACTIONS],
                "piston_offsets_surface_um": offsets,
                "piston_command_surface_um": (
                    [point_base[segments[0]].piston_um + v for v in offsets]
                    if len(segments) == 1 else None
                ),
                "piston_commands_surface_um_by_segment": {
                    str(segment): [point_base[segment].piston_um + v for v in offsets]
                    for segment in segments
                },
                "reflected_relative_phase_deg": [720 * float(v) for v in PISTON_FRACTIONS],
                "pistons_completed": 0,
                "pistons_completed_description": "Total completed saved piston blocks across all repeats",
                "pistons_completed_per_repeat": [0] * int(N_REPEATS),
                "repeats_requested": int(N_REPEATS),
                "repeats_completed": 0,
                "frames_per_piston": int(NFRAMES),
                "frames_per_piston_description": "Frames per piston in EACH recorded repeat",
                "per_piston_raw_max_adu": [[None] * int(N_REPEATS) for _ in offsets],
                "frames_recorded": [[0] * int(N_REPEATS) for _ in offsets],
                "quality_array_axis_order": ["piston", "repeat"],
                "random_seed": random_seed,
                "acquisition_order_method": "randomized cyclic blocks, balanced across slots within each full block of len(pistons) repeats",
                "acquisition_order_piston_indices": plan["repeat_orders"],
                "acquisition_order_axis_order": ["repeat", "acquisition_slot"],
                "acquisition_indices_are_zero_based": True,
                "acquisition_log": [],
                "trial_scan": {
                    "status": "pending",
                    "frames_per_piston": int(NFRAMES),
                    "order_piston_indices": plan["trial_order"],
                    "frames_retained": False,
                    "pistons_completed": 0,
                    "acquisition_log": [],
                },
                "dark": {"file": str(dark_path), "sha256": sha256(dark_path), "subtracted_per_frame": True},
                "adc_guard_limit_adu": guard_limit,
                "base_dm_ptt": dm_state_to_dict(base),
                "point_dm_ptt_before_piston": dm_state_to_dict(point_base),
            }
            if point_index == 1:
                current_payload["first_position_preview"] = {
                    "file": "first_position_photometry_preview.png",
                    "frames": int(PREVIEW_NFRAMES),
                    "raw_max_adu": pilot_max,
                    "operator_confirmed": True,
                }
            # One unrecorded full sweep at each XY point conditions the same
            # selected segment group and reads the same N frames per piston.
            # No base/zero reset occurs between this trial and recorded repeats.
            trial = current_payload["trial_scan"]
            trial["status"] = "running"
            for slot, piston_index in enumerate(plan["trial_order"]):
                frames, event = acquire_piston(
                    point_base, segments, piston_index,
                    f"{label} discarded trial piston {piston_index}",
                    f"{label}_trial_slot_{slot}_piston_{piston_index}",
                )
                # Apply the normal extraction checks too, but retain no trial spectra.
                for channel in channels:
                    column_sums_per_frame(frames, dark, boxes[channel])
                del frames
                event["acquisition_slot"] = slot
                trial["acquisition_log"].append(event)
                trial["pistons_completed"] += 1
                print(f"  [{point_index}/{len(positions)}] {label} TRIAL (discarded): "
                      f"slot {slot+1}/{len(offsets)}, piston index {piston_index}, "
                      f"offset={offsets[piston_index]:g} µm")
            trial["status"] = "completed"
            for repeat_index, order in enumerate(plan["repeat_orders"]):
                for slot, piston_index in enumerate(order):
                    frames, event = acquire_piston(
                        point_base, segments, piston_index,
                        f"{label} repeat {repeat_index+1} piston {piston_index}",
                        f"{label}_repeat_{repeat_index}_slot_{slot}_piston_{piston_index}",
                    )
                    for channel_index, channel in enumerate(channels):
                        columns = column_sums_per_frame(frames, dark, boxes[channel])
                        current_cube[:widths[channel_index], channel_index, piston_index, :, repeat_index] = columns.T
                    del frames
                    event.update({"repeat_index": repeat_index, "acquisition_slot": slot})
                    current_payload["acquisition_log"].append(event)
                    current_payload["per_piston_raw_max_adu"][piston_index][repeat_index] = event["raw_max_adu"]
                    current_payload["frames_recorded"][piston_index][repeat_index] = int(NFRAMES)
                    current_payload["pistons_completed"] += 1
                    current_payload["pistons_completed_per_repeat"][repeat_index] += 1
                    if slot == len(offsets) - 1:
                        current_payload["repeats_completed"] += 1
                    persist_current("running")  # Checkpoint after every saved piston block.
                    print(f"  [{point_index}/{len(positions)}] {label} X={x},Y={y},"
                          f"tip={tip_mrad},tilt={tilt_mrad},segments={list(segments)}; "
                          f"repeat {repeat_index+1}/{N_REPEATS}, slot {slot+1}/{len(offsets)}, "
                          f"piston index {piston_index} (offset={offsets[piston_index]:g} µm); "
                          f"raw max={event['raw_max_adu']:g} ADU")
            record_state(base, "base_state_after_point")
            persist_current("completed")
            current_json = None
            current_payload = None
            current_cube = None
    except BaseException as error:
        run_error = error
        try:
            if current_payload is not None and current_payload["trial_scan"]["status"] == "running":
                current_payload["trial_scan"]["status"] = "aborted" if isinstance(error, KeyboardInterrupt) else "failed"
            persist_current("aborted" if isinstance(error, KeyboardInterrupt) else "failed",
                            f"{type(error).__name__}: {error}")
        except Exception as save_error:
            print(f"WARNING: partial JSON could not be saved: {save_error}", file=sys.stderr)
    finally:
        if hardware is not None:
            # Preserve source-script policy: return mount first with the base
            # state (no piston offset), then restore launch-time GLINT DM PTT.
            if before_mount is not None and before_dm is not None:
                try:
                    record_state(base, "base_state_before_restore")
                    hardware.move_mount({axis: float(before_mount[axis])
                                         for axis in ("roll", "x", "y")}, timeout)
                except BaseException as error:
                    restore_errors.append(f"mount/base: {type(error).__name__}: {error}")
                try:
                    hardware.set_dm(before_dm)
                except BaseException as error:
                    restore_errors.append(f"dm: {type(error).__name__}: {error}")
            hardware.close()
    if restore_errors:
        raise RuntimeError(f"Restore failed: {restore_errors}; original scan error={run_error}")
    if run_error is not None:
        raise run_error
    print(f"Saved {len(positions)} spectral-cube JSON files in {output_dir}")
    return output_dir


def main() -> None:
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("Interrupted by operator", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
