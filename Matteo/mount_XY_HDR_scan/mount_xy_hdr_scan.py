#!/usr/bin/env python3
"""Masked high-dynamic-range mount X/Y scan for one GLINT DM segment.

The scan uses the newest valid ``06_xy_*.json`` plus its FITS/NPZ array file
under the active alignment ``output_root``.  Bright positions measured by that
reference scan are excluded before the operator raises the source flux.  When
only one injected segment remains flat, translated copies of the appropriate
reference masks also exclude positions where that segment would illuminate the
other injected microlenses.  Mount motion between acquired positions is
constrained to adjacent safe grid points, so a row transition cannot cut across
any excluded bright footprint.

This is a standalone instrument script.  It uses the same configuration,
hardware adapter, detector boxes, command logging, and shared APAPANE dark as
the refactored GLINT alignment GUI.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree


SCRIPT_DIRECTORY = Path(__file__).resolve().parent


def _find_refactor_root() -> Path:
    """Locate the refactored GUI package before importing glint_alignment.

    On the instrument this standalone script normally lives at
    ``glint/Matteo/mount_XY_HDR_scan`` and the package at
    ``glint/glint_alignment_refactor``.  Keeping the repository root as a
    candidate also allows this file to be run directly from the refactor.
    """
    configured = os.environ.get("GLINT_ALIGNMENT_REFACTOR_ROOT")
    candidates = [
        Path(configured).expanduser() if configured else None,
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
        "Could not locate the glint_alignment_refactor package. Tried: "
        + ", ".join(tried)
        + ". Set GLINT_ALIGNMENT_REFACTOR_ROOT to its absolute directory if it is elsewhere."
    )


GLINT_ALIGNMENT_REFACTOR_ROOT = _find_refactor_root()
if str(GLINT_ALIGNMENT_REFACTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(GLINT_ALIGNMENT_REFACTOR_ROOT))

from glint_alignment.boxes import INJECTED_SEGMENTS, PHOT_BOXES, SEGMENT_TO_PHOT, integrate_box
from glint_alignment.config import load_config, segment_state
from glint_alignment.instrument import InstrumentHardware
from glint_alignment.models import SegmentPTT, dm_state_to_dict
from glint_alignment.storage import alignment_root, load_saved_dark


DEFAULT_INSTRUMENT_CONFIG = GLINT_ALIGNMENT_REFACTOR_ROOT / "config" / "instrument.json"
DEFAULT_SCAN_CONFIG = next(
    (
        candidate
        for candidate in (
            SCRIPT_DIRECTORY / "config" / "mount_xy_hdr_scan.json",
            SCRIPT_DIRECTORY / "mount_xy_hdr_scan.json",
            GLINT_ALIGNMENT_REFACTOR_ROOT / "config" / "mount_xy_hdr_scan.json",
        )
        if candidate.is_file()
    ),
    SCRIPT_DIRECTORY / "config" / "mount_xy_hdr_scan.json",
)
XY_NAME = re.compile(r"^06_xy_(\d+)\.json$")


class SaturationDetected(RuntimeError):
    """Raised after saving the first measurement that trips the live guard."""


@dataclass(frozen=True)
class SourceXY:
    json_path: Path
    array_path: Path
    completed_utc: datetime
    summary: dict[str, Any]
    x: np.ndarray
    y: np.ndarray
    maps: dict[str, np.ndarray]


@dataclass(frozen=True)
class ScanPlan:
    x: np.ndarray
    y: np.ndarray
    excluded: np.ndarray
    route: tuple[tuple[int, int], ...]
    acquisition_order: tuple[tuple[int, int], ...]
    center_x: float
    center_y: float
    source_bright: np.ndarray
    mask_components: dict[str, np.ndarray]
    mask_metadata: dict[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instrument-config",
        type=Path,
        default=DEFAULT_INSTRUMENT_CONFIG,
        help="Active GLINT alignment instrument.json.",
    )
    parser.add_argument(
        "--scan-config",
        type=Path,
        default=DEFAULT_SCAN_CONFIG,
        help="HDR scan settings JSON.",
    )
    parser.add_argument(
        "--source-json",
        type=Path,
        help="Explicit 06_xy_*.json override. If omitted, the newest valid saved XY scan is used.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Resolve the source scan and validate/print the mask and motion plan without hardware access.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip only the initial physical-motion confirmation; operator flux-change prompts remain active.",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _parse_utc(value: Any, fallback: float) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass
    return datetime.fromtimestamp(fallback, tz=timezone.utc)


def _array_path(json_path: Path, summary: dict[str, Any]) -> Path:
    # Prefer the relocatable basename saved beside the JSON.  Absolute paths in
    # metadata may refer to an older host or to the machine on which a copy was made.
    name = summary.get("array_file")
    if isinstance(name, str) and name.strip():
        candidate = json_path.parent / Path(name).name
        if candidate.is_file():
            return candidate
    for suffix in (".fits", ".npz"):
        candidate = json_path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No matching FITS/NPZ array file beside {json_path}")


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name], dtype=float) for name in archive.files}
    if path.suffix.lower() != ".fits":
        raise ValueError(f"Unsupported XY array format {path.suffix!r}: {path}")
    try:
        from astropy.io import fits
    except ImportError as error:
        raise RuntimeError("astropy is required to read the saved XY FITS matrix") from error
    arrays: dict[str, np.ndarray] = {}
    with fits.open(path, memmap=False) as hdus:
        for hdu in hdus:
            long_name = hdu.header.get("LONGNAME")
            if long_name and hdu.data is not None:
                arrays[str(long_name).strip()] = np.asarray(hdu.data, dtype=float)
    return arrays


def load_source_xy(json_path: Path, required_channels: Iterable[str]) -> SourceXY:
    json_path = json_path.expanduser().resolve()
    summary = load_json(json_path)
    if summary.get("kind") != "xy":
        raise ValueError(f"{json_path} has kind={summary.get('kind')!r}, not 'xy'")
    array_path = _array_path(json_path, summary)
    arrays = _load_arrays(array_path)
    axes = summary.get("axes", {})
    x = np.asarray(arrays.get("axis__x", axes.get("x", [])), dtype=float)
    y = np.asarray(arrays.get("axis__y", axes.get("y", [])), dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) < 2 or len(y) < 2:
        raise ValueError(f"{json_path} does not contain valid 1-D x/y axes")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError(f"{json_path} contains non-finite x/y coordinates")
    if not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)):
        raise ValueError(f"{json_path} x/y axes must be strictly increasing")
    maps: dict[str, np.ndarray] = {}
    for channel in required_channels:
        key = f"map__{channel}"
        if key not in arrays:
            raise ValueError(f"{array_path} is missing {key!r}")
        data = np.asarray(arrays[key], dtype=float)
        if data.shape != (len(y), len(x)):
            raise ValueError(
                f"{key} shape {data.shape} does not match y/x axes {(len(y), len(x))}"
            )
        if not np.any(np.isfinite(data)):
            raise ValueError(f"{key} contains no finite samples")
        maps[channel] = data
    completed = _parse_utc(summary.get("completed_utc"), json_path.stat().st_mtime)
    return SourceXY(json_path, array_path, completed, summary, x, y, maps)


def _candidate_xy_jsons(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Configured alignment output_root does not exist: {root}")
    candidates: list[Path] = []
    for path in root.rglob("06_xy_*.json"):
        parent = path.parent.name
        if parent == "Alignment_output_individual" or parent.startswith("Alignment_outputs_"):
            if XY_NAME.match(path.name):
                candidates.append(path)
    return candidates


def latest_source_xy(root: Path, required_channels: Iterable[str]) -> SourceXY:
    """Return the newest readable saved XY JSON/array pair under output_root.

    Ranking uses ``completed_utc`` from each JSON, with JSON modification time
    only as a fallback.  A newer incomplete or corrupt pair is skipped.
    """
    ranked: list[tuple[datetime, float, Path]] = []
    for json_path in _candidate_xy_jsons(root):
        try:
            summary = load_json(json_path)
            if summary.get("kind") != "xy":
                continue
            completed = _parse_utc(summary.get("completed_utc"), json_path.stat().st_mtime)
            ranked.append((completed, json_path.stat().st_mtime, json_path))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    errors: list[str] = []
    for _, _, json_path in sorted(ranked, reverse=True):
        try:
            return load_source_xy(json_path, required_channels)
        except Exception as error:
            errors.append(f"{json_path}: {error}")
    detail = "" if not errors else "\nRejected candidates:\n  " + "\n  ".join(errors[:8])
    raise FileNotFoundError(f"No valid 06_xy JSON/array pair found below {root}.{detail}")


def resolve_source_xy(
    instrument: dict[str, Any],
    scan: dict[str, Any],
    scan_config_path: Path,
    required_channels: Iterable[str],
    cli_override: Path | None,
) -> SourceXY:
    if cli_override is not None:
        return load_source_xy(cli_override, required_channels)
    configured = scan.get("source_xy_json", "latest")
    if configured in (None, "", "latest", "auto"):
        return latest_source_xy(alignment_root(instrument), required_channels)
    path = Path(str(configured)).expanduser()
    if not path.is_absolute():
        path = scan_config_path.parent / path
    return load_source_xy(path, required_channels)


def validate_scan_config(scan: dict[str, Any]) -> dict[str, Any]:
    if int(scan.get("schema_version", 0)) != 1:
        raise ValueError("mount_xy_hdr_scan schema_version must be 1")
    segment = int(scan.get("injected_segment", -1))
    if segment not in INJECTED_SEGMENTS:
        raise ValueError(f"injected_segment must be one of {INJECTED_SEGMENTS}")
    extent = scan.get("scan_extent", {})
    step = float(extent.get("step_pulses", 0))
    if not np.isfinite(step) or step <= 0:
        raise ValueError("scan_extent.step_pulses must be positive and finite")
    offset_keys = (
        "x_min_offset_pulses",
        "x_max_offset_pulses",
        "y_min_offset_pulses",
        "y_max_offset_pulses",
    )
    supplied_offsets = [key in extent for key in offset_keys]
    if any(supplied_offsets) and not all(supplied_offsets):
        raise ValueError(
            "scan_extent must provide all four min/max offset fields or none of them"
        )
    if all(supplied_offsets):
        offsets = {key: float(extent[key]) for key in offset_keys}
        if not all(np.isfinite(value) for value in offsets.values()):
            raise ValueError("scan_extent min/max offsets must be finite")
        if offsets["x_min_offset_pulses"] >= offsets["x_max_offset_pulses"]:
            raise ValueError("scan_extent x_min_offset_pulses must be below x_max_offset_pulses")
        if offsets["y_min_offset_pulses"] >= offsets["y_max_offset_pulses"]:
            raise ValueError("scan_extent y_min_offset_pulses must be below y_max_offset_pulses")
    else:
        for key in ("x_span_factor", "y_span_factor"):
            if not np.isfinite(float(extent.get(key, 0))) or float(extent.get(key, 0)) <= 0:
                raise ValueError(f"scan_extent.{key} must be positive and finite")
    mask = scan.get("mask", {})
    threshold = float(mask.get("threshold_fraction_of_peak", 0))
    if not 0 < threshold < 1:
        raise ValueError("mask.threshold_fraction_of_peak must be in (0, 1)")
    percentile = float(mask.get("background_percentile", -1))
    if not 0 <= percentile < 50:
        raise ValueError("mask.background_percentile must be in [0, 50)")
    if float(mask.get("dilation_pulses", -1)) < 0:
        raise ValueError("mask.dilation_pulses must be non-negative")
    overlap = scan.get("segment_overlap_mask", {})
    if bool(overlap.get("enabled", False)):
        if scan.get("dm_state", "selected_segment_only") != "selected_segment_only":
            raise ValueError(
                "segment_overlap_mask requires dm_state='selected_segment_only'"
            )
        if int(overlap.get("source_segment", -1)) != segment:
            raise ValueError(
                "segment_overlap_mask.source_segment must equal injected_segment"
            )
        targets = overlap.get("target_segments")
        if not isinstance(targets, list) or not targets:
            raise ValueError("segment_overlap_mask.target_segments must be a non-empty list")
        target_segments = [int(value) for value in targets]
        if len(set(target_segments)) != len(target_segments):
            raise ValueError("segment_overlap_mask.target_segments contains duplicates")
        unknown_targets = set(target_segments).difference(INJECTED_SEGMENTS)
        if unknown_targets or segment in target_segments:
            raise ValueError(
                "segment_overlap_mask targets must be other injected segments; "
                f"received {target_segments}"
            )
        offsets = overlap.get("target_offsets_pulses")
        if not isinstance(offsets, dict):
            raise ValueError("segment_overlap_mask.target_offsets_pulses must be an object")
        for target in target_segments:
            value = offsets.get(str(target), offsets.get(target))
            try:
                array = np.asarray(value, dtype=float)
            except (TypeError, ValueError):
                array = np.asarray([], dtype=float)
            if array.shape != (2,) or not np.all(np.isfinite(array)):
                raise ValueError(
                    "segment_overlap_mask.target_offsets_pulses must provide finite "
                    f"[dx, dy] for target segment {target}"
                )
        overlap_threshold = float(
            overlap.get("threshold_fraction_of_peak", mask.get("threshold_fraction_of_peak", 0))
        )
        if not 0 < overlap_threshold < 1:
            raise ValueError(
                "segment_overlap_mask.threshold_fraction_of_peak must be in (0, 1)"
            )
        overlap_percentile = float(
            overlap.get("background_percentile", mask.get("background_percentile", -1))
        )
        if not 0 <= overlap_percentile < 50:
            raise ValueError(
                "segment_overlap_mask.background_percentile must be in [0, 50)"
            )
        if float(overlap.get("dilation_pulses", -1)) < 0:
            raise ValueError("segment_overlap_mask.dilation_pulses must be non-negative")
    acquisition = scan.get("acquisition", {})
    if int(acquisition.get("nframes", 0)) < 1:
        raise ValueError("acquisition.nframes must be at least 1")
    if float(acquisition.get("settle_s", -1)) < 0:
        raise ValueError("acquisition.settle_s must be non-negative")
    if int(acquisition.get("checkpoint_every", 0)) < 1:
        raise ValueError("acquisition.checkpoint_every must be at least 1")
    guard = scan.get("saturation_guard", {})
    if bool(guard.get("enabled", True)):
        if float(guard.get("adc_max_adu", 0)) <= 0:
            raise ValueError("saturation_guard.adc_max_adu must be positive")
        if not 0 < float(guard.get("fraction", 0)) < 1:
            raise ValueError("saturation_guard.fraction must be in (0, 1)")
        if guard.get("regions", "full_frame") not in {"all_photometry", "full_frame"}:
            raise ValueError(
                "saturation_guard.regions must be 'all_photometry' or 'full_frame'"
            )
    ratio = scan.get("normalization", {}).get("high_to_reference_flux_ratio")
    if ratio is not None and (not np.isfinite(float(ratio)) or float(ratio) <= 0):
        raise ValueError("normalization.high_to_reference_flux_ratio must be null or positive")
    if scan.get("dm_state", "selected_segment_only") not in {
        "selected_segment_only",
        "source_scan_state",
    }:
        raise ValueError("dm_state must be 'selected_segment_only' or 'source_scan_state'")
    output_folder = scan.get("output_folder", "Alignment_output_individual")
    if (
        not isinstance(output_folder, str)
        or not output_folder.strip()
        or Path(output_folder).name != output_folder
        or output_folder in {".", ".."}
    ):
        raise ValueError("output_folder must be one simple directory name below the dated output root")
    return scan


def mask_channels(scan: dict[str, Any]) -> tuple[str, ...]:
    configured = scan.get("mask", {}).get("channels", "all_photometry")
    if configured == "all_photometry":
        return tuple(PHOT_BOXES)
    if not isinstance(configured, list) or not configured:
        raise ValueError("mask.channels must be 'all_photometry' or a non-empty list")
    channels = tuple(str(value) for value in configured)
    unknown = set(channels).difference(PHOT_BOXES)
    if unknown:
        raise ValueError(f"Unknown photometry mask channels: {sorted(unknown)}")
    return channels


def source_channel_center(source: SourceXY, channel: str) -> tuple[float, float]:
    for item in source.summary.get("estimates", {}).get("channels", []):
        if item.get("channel") == channel:
            x, y = float(item["x"]), float(item["y"])
            if np.isfinite(x) and np.isfinite(y):
                return x, y
    z = np.asarray(source.maps[channel], dtype=float)
    background = float(np.nanpercentile(z, 5))
    weights = np.maximum(np.nan_to_num(z - background, nan=0.0), 0.0)
    if float(weights.sum()) <= 0:
        iy, ix = np.unravel_index(np.nanargmax(z), z.shape)
        return float(source.x[ix]), float(source.y[iy])
    xx, yy = np.meshgrid(source.x, source.y)
    return float(np.sum(weights * xx) / np.sum(weights)), float(np.sum(weights * yy) / np.sum(weights))


def centered_grid(center: float, span: float, step: float) -> np.ndarray:
    count = int(round(span / step))
    if count < 2:
        raise ValueError("HDR scan extent must contain at least three points")
    if count % 2:
        count += 1
    integer_motion = np.isclose(step, round(step), atol=1e-9)
    midpoint = float(round(center)) if integer_motion else float(center)
    start = midpoint - 0.5 * count * step
    values = start + np.arange(count + 1, dtype=float) * step
    return np.round(values) if integer_motion else values


def offset_grid(center: float, minimum_offset: float, maximum_offset: float, step: float) -> np.ndarray:
    """Build an asymmetric grid around a fitted centre using explicit offsets."""
    integer_motion = np.isclose(step, round(step), atol=1e-9)
    midpoint = float(round(center)) if integer_motion else float(center)
    start = midpoint + float(minimum_offset)
    stop = midpoint + float(maximum_offset)
    count = int(np.floor((stop - start) / step + 1e-9))
    if count < 2:
        raise ValueError("HDR scan extent must contain at least three points")
    values = start + np.arange(count + 1, dtype=float) * step
    return np.round(values) if integer_motion else values


def _normalised_bright_mask(
    data: np.ndarray,
    threshold: float,
    background_percentile: float,
) -> tuple[np.ndarray, dict[str, float]]:
    finite = np.asarray(data, dtype=float)
    background = float(np.nanpercentile(finite, background_percentile))
    peak = float(np.nanmax(finite))
    dynamic = peak - background
    if not np.isfinite(dynamic) or dynamic <= 0:
        raise ValueError("Reference XY map has no positive dynamic range")
    normalised = np.maximum((finite - background) / dynamic, 0.0)
    return np.isfinite(normalised) & (normalised >= threshold), {
        "background": background,
        "peak": peak,
        "dynamic_range": dynamic,
    }


def build_exclusion_mask(
    source: SourceXY,
    target_x: np.ndarray,
    target_y: np.ndarray,
    channels: Sequence[str],
    *,
    threshold: float,
    background_percentile: float,
    dilation_pulses: float,
    allow_boundary_contact: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source_bright = np.zeros((len(source.y), len(source.x)), dtype=bool)
    channel_stats: dict[str, dict[str, float]] = {}
    for channel in channels:
        bright, stats = _normalised_bright_mask(
            source.maps[channel], threshold, background_percentile
        )
        source_bright |= bright
        channel_stats[channel] = stats
    boundary_count = int(
        np.count_nonzero(source_bright[0])
        + np.count_nonzero(source_bright[-1])
        + np.count_nonzero(source_bright[:, 0])
        + np.count_nonzero(source_bright[:, -1])
    )
    if boundary_count and not allow_boundary_contact:
        raise ValueError(
            "The chosen bright-mask threshold touches the reference XY boundary; "
            "the scan does not prove where that flux falls below threshold. Increase "
            "mask.threshold_fraction_of_peak or explicitly set "
            "allow_threshold_touching_source_boundary=true after reviewing the risk."
        )
    source_points_yx = np.argwhere(source_bright)
    if len(source_points_yx) == 0:
        raise ValueError("The source bright mask is empty")
    bright_xy = np.column_stack(
        (source.x[source_points_yx[:, 1]], source.y[source_points_yx[:, 0]])
    )
    excluded = _points_exclusion_mask(
        bright_xy, target_x, target_y, dilation_pulses=dilation_pulses
    )
    metadata = {
        "channels": list(channels),
        "threshold_fraction_of_peak": float(threshold),
        "background_percentile": float(background_percentile),
        "dilation_pulses": float(dilation_pulses),
        "source_bright_points": int(np.count_nonzero(source_bright)),
        "threshold_boundary_contact_points": boundary_count,
        "target_excluded_points": int(np.count_nonzero(excluded)),
        "target_safe_points": int(np.count_nonzero(~excluded)),
        "channel_statistics": channel_stats,
    }
    return excluded, source_bright, metadata


def _points_exclusion_mask(
    bright_xy: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    *,
    dilation_pulses: float,
) -> np.ndarray:
    """Project measured bright points onto a target grid with a safety margin."""
    target_xx, target_yy = np.meshgrid(target_x, target_y)
    target_points = np.column_stack((target_xx.ravel(), target_yy.ravel()))
    distances, _ = cKDTree(bright_xy).query(target_points, k=1)
    return (distances <= float(dilation_pulses) + 1e-9).reshape(target_yy.shape)


def build_segment_overlap_masks(
    source: SourceXY,
    target_x: np.ndarray,
    target_y: np.ndarray,
    *,
    source_segment: int,
    target_segments: Sequence[int],
    target_offsets_pulses: dict[str, Any],
    threshold: float,
    background_percentile: float,
    dilation_pulses: float,
    allow_boundary_contact: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Mask where one flat segment would illuminate other injected microlenses.

    Each target uses that target photometric channel's measured low-flux bright
    footprint from the newest XY reference scan.  The footprint is translated
    by the configured mount-space offset from the flat source segment to the
    target microlens.  This retains measured microlens aberration and aperture
    shape instead of replacing it with an ideal circular or hexagonal mask.
    """
    masks: dict[str, np.ndarray] = {}
    targets_metadata: dict[str, Any] = {}
    for target_segment in target_segments:
        target_segment = int(target_segment)
        channel = SEGMENT_TO_PHOT[target_segment]
        bright, statistics = _normalised_bright_mask(
            source.maps[channel], threshold, background_percentile
        )
        boundary_count = int(
            np.count_nonzero(bright[0])
            + np.count_nonzero(bright[-1])
            + np.count_nonzero(bright[:, 0])
            + np.count_nonzero(bright[:, -1])
        )
        if boundary_count and not allow_boundary_contact:
            raise ValueError(
                f"The segment-overlap template for {channel} touches the reference XY "
                "boundary. The low-flux scan does not bound the unsafe footprint; "
                "review the threshold or acquire a wider reference XY scan."
            )
        source_points_yx = np.argwhere(bright)
        if len(source_points_yx) == 0:
            raise ValueError(f"The segment-overlap template for {channel} is empty")
        template_xy = np.column_stack(
            (source.x[source_points_yx[:, 1]], source.y[source_points_yx[:, 0]])
        )
        configured_offset = target_offsets_pulses.get(
            str(target_segment), target_offsets_pulses.get(target_segment)
        )
        offset = np.asarray(configured_offset, dtype=float)
        translated_xy = template_xy + offset[None, :]
        name = f"segment{source_segment}_on_microlens{target_segment}"
        target_mask = _points_exclusion_mask(
            translated_xy,
            target_x,
            target_y,
            dilation_pulses=dilation_pulses,
        )
        target_center = np.asarray(source_channel_center(source, channel)) + offset
        masks[name] = target_mask
        targets_metadata[str(target_segment)] = {
            "component": name,
            "template_channel": channel,
            "offset_pulses": offset.tolist(),
            "predicted_overlap_center": {
                "x": float(target_center[0]),
                "y": float(target_center[1]),
            },
            "source_bright_points": int(np.count_nonzero(bright)),
            "threshold_boundary_contact_points": boundary_count,
            "target_excluded_points": int(np.count_nonzero(target_mask)),
            "channel_statistics": statistics,
        }
    return masks, {
        "enabled": True,
        "source_segment": int(source_segment),
        "target_segments": [int(value) for value in target_segments],
        "threshold_fraction_of_peak": float(threshold),
        "background_percentile": float(background_percentile),
        "dilation_pulses": float(dilation_pulses),
        "targets": targets_metadata,
    }


def _neighbours(node: tuple[int, int], shape: tuple[int, int]) -> Iterable[tuple[int, int]]:
    iy, ix = node
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        ny, nx = iy + dy, ix + dx
        if 0 <= ny < shape[0] and 0 <= nx < shape[1]:
            yield ny, nx


def _shortest_safe_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    safe: np.ndarray,
) -> list[tuple[int, int]]:
    if start == goal:
        return [start]
    queue: list[tuple[int, int, tuple[int, int]]] = []
    heapq.heappush(queue, (0, 0, start))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    cost = {start: 0}
    while queue:
        _, current_cost, current = heapq.heappop(queue)
        if current == goal:
            path = [current]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            path.reverse()
            return path
        if current_cost != cost.get(current):
            continue
        for neighbour in _neighbours(current, safe.shape):
            if not safe[neighbour]:
                continue
            next_cost = current_cost + 1
            if next_cost >= cost.get(neighbour, sys.maxsize):
                continue
            cost[neighbour] = next_cost
            came_from[neighbour] = current
            heuristic = abs(neighbour[0] - goal[0]) + abs(neighbour[1] - goal[1])
            heapq.heappush(queue, (next_cost + heuristic, next_cost, neighbour))
    raise ValueError("Safe scan grid is disconnected; no mask-respecting mount route exists")


def safe_serpentine_route(excluded: np.ndarray) -> tuple[
    tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]
]:
    """Cover every safe point while every physical step stays on a safe grid edge."""
    safe = ~np.asarray(excluded, dtype=bool)
    targets: list[tuple[int, int]] = []
    for iy in range(safe.shape[0]):
        columns = range(safe.shape[1]) if iy % 2 == 0 else range(safe.shape[1] - 1, -1, -1)
        targets.extend((iy, ix) for ix in columns if safe[iy, ix])
    if not targets:
        raise ValueError("The exclusion mask leaves no safe acquisition point")
    route = [targets[0]]
    acquired = [targets[0]]
    seen = {targets[0]}
    current = targets[0]
    for target in targets[1:]:
        if target in seen:
            continue
        if abs(current[0] - target[0]) + abs(current[1] - target[1]) == 1:
            path = [current, target]
        else:
            path = _shortest_safe_path(current, target, safe)
        for node in path[1:]:
            route.append(node)
            if node not in seen:
                acquired.append(node)
                seen.add(node)
        current = target
    if len(seen) != int(np.count_nonzero(safe)):
        raise AssertionError("Internal error: safe route did not cover the complete scan mask")
    return tuple(route), tuple(acquired)


def make_plan(source: SourceXY, scan: dict[str, Any], channels: Sequence[str]) -> ScanPlan:
    segment = int(scan["injected_segment"])
    selected_channel = SEGMENT_TO_PHOT[segment]
    center_x, center_y = source_channel_center(source, selected_channel)
    extent = scan["scan_extent"]
    step = float(extent["step_pulses"])
    if "x_min_offset_pulses" in extent:
        x = offset_grid(
            center_x,
            float(extent["x_min_offset_pulses"]),
            float(extent["x_max_offset_pulses"]),
            step,
        )
        y = offset_grid(
            center_y,
            float(extent["y_min_offset_pulses"]),
            float(extent["y_max_offset_pulses"]),
            step,
        )
    else:
        x = centered_grid(
            center_x,
            float(np.ptp(source.x)) * float(extent["x_span_factor"]),
            step,
        )
        y = centered_grid(
            center_y,
            float(np.ptp(source.y)) * float(extent["y_span_factor"]),
            step,
        )
    mask = scan["mask"]
    excluded, source_bright, metadata = build_exclusion_mask(
        source,
        x,
        y,
        channels,
        threshold=float(mask["threshold_fraction_of_peak"]),
        background_percentile=float(mask["background_percentile"]),
        dilation_pulses=float(mask["dilation_pulses"]),
        allow_boundary_contact=bool(mask.get("allow_threshold_touching_source_boundary", False)),
    )
    mask_components = {"direct_reference_bright": excluded.copy()}
    metadata["direct_target_excluded_points"] = int(np.count_nonzero(excluded))
    overlap = scan.get("segment_overlap_mask", {})
    if bool(overlap.get("enabled", False)):
        overlap_masks, overlap_metadata = build_segment_overlap_masks(
            source,
            x,
            y,
            source_segment=int(overlap["source_segment"]),
            target_segments=[int(value) for value in overlap["target_segments"]],
            target_offsets_pulses=overlap["target_offsets_pulses"],
            threshold=float(
                overlap.get("threshold_fraction_of_peak", mask["threshold_fraction_of_peak"])
            ),
            background_percentile=float(
                overlap.get("background_percentile", mask["background_percentile"])
            ),
            dilation_pulses=float(overlap["dilation_pulses"]),
            allow_boundary_contact=bool(
                overlap.get(
                    "allow_threshold_touching_source_boundary",
                    mask.get("allow_threshold_touching_source_boundary", False),
                )
            ),
        )
        for name, overlap_mask in overlap_masks.items():
            mask_components[name] = overlap_mask
            excluded |= overlap_mask
        metadata["segment_overlap"] = overlap_metadata
    else:
        metadata["segment_overlap"] = {"enabled": False}
    metadata["target_excluded_points"] = int(np.count_nonzero(excluded))
    metadata["target_safe_points"] = int(np.count_nonzero(~excluded))
    route, acquisition_order = safe_serpentine_route(excluded)
    return ScanPlan(
        x=x,
        y=y,
        excluded=excluded,
        route=route,
        acquisition_order=acquisition_order,
        center_x=center_x,
        center_y=center_y,
        source_bright=source_bright,
        mask_components=mask_components,
        mask_metadata=metadata,
    )


def validate_mount_plan(plan: ScanPlan, hardware: InstrumentHardware) -> None:
    limits = hardware.mount_limits()
    for name, values in (("x", plan.x), ("y", plan.y)):
        lower, upper = limits[name]
        if float(values[0]) < lower or float(values[-1]) > upper:
            raise ValueError(
                f"HDR {name} grid {values[0]:g}..{values[-1]:g} exceeds configured limits "
                f"[{lower:g}, {upper:g}]"
            )


def _source_dm_state(source: SourceXY) -> dict[int, SegmentPTT]:
    metadata = source.summary.get("metadata", {})
    for key in ("state_at_completion", "state_after_review"):
        values = metadata.get(key, {}).get("dm")
        if isinstance(values, dict) and len(values) == 37:
            return {int(index): SegmentPTT(**ptt) for index, ptt in values.items()}
    raise ValueError(f"{source.json_path} does not contain a complete saved DM state")


def scan_dm_state(
    instrument: dict[str, Any], source: SourceXY, scan: dict[str, Any]
) -> dict[int, SegmentPTT]:
    if scan.get("dm_state", "selected_segment_only") == "source_scan_state":
        return _source_dm_state(source)
    segment = int(scan["injected_segment"])
    state = segment_state(instrument, "tilted_away")
    state[segment] = segment_state(instrument, "zero_ptt")[segment]
    return state


def validate_emergency_tilt_state(instrument: dict[str, Any]) -> None:
    """Refuse physical motion unless the protective all-tilted map is validated."""
    definition = instrument.get("dm_states", {}).get("tilted_away", {})
    if not bool(definition.get("validated", False)):
        raise RuntimeError(
            "dm_states.tilted_away is not marked validated in instrument.json. The HDR scan "
            "will not rely on an unverified DM state for detector protection. Confirm the "
            "measured all-tilted PTT map, then set its validated field to true."
        )
    state = segment_state(instrument, "tilted_away")
    values = np.asarray(
        [[ptt.piston_um, ptt.tip_mrad, ptt.tilt_mrad] for ptt in state.values()],
        dtype=float,
    )
    if values.shape != (37, 3) or not np.all(np.isfinite(values)):
        raise RuntimeError("The validated tilted-away state must define 37 finite PTT commands")


def _average_frame(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(frames, dtype=float)
    if values.ndim == 2:
        return values, values[None, ...]
    if values.ndim == 3:
        return np.nanmean(values, axis=0), values
    raise ValueError(f"Expected a 2-D frame or 3-D frame cube, got {values.shape}")


def _guard_raw_max(frames: np.ndarray, regions: str) -> float:
    if regions == "full_frame":
        return float(np.nanmax(frames))
    if regions != "all_photometry":
        raise ValueError("saturation_guard.regions must be 'all_photometry' or 'full_frame'")
    maxima = []
    for box in PHOT_BOXES.values():
        y0, y1, x0, x1 = map(int, box)
        maxima.append(float(np.nanmax(frames[..., y0:y1, x0:x1])))
    return max(maxima)


def _guarded_apapane_frames(
    hardware: InstrumentHardware,
    *,
    nframes: int,
    guard: dict[str, Any],
    guard_limit: float,
    emergency_dm_state: dict[int, SegmentPTT],
    label: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Acquire one probe frame before committing to the remaining frame batch.

    A tripped guard immediately commands the complete tilted-away DM state. The
    method never changes APAPANE; both calls are passive SHM reads.
    """
    _, probe_frames = _average_frame(hardware.apapane_frames(1))
    regions = str(guard.get("regions", "all_photometry"))
    maximum = _guard_raw_max(probe_frames, regions)
    if bool(guard.get("enabled", True)) and maximum >= guard_limit:
        hardware.set_dm(emergency_dm_state)
        raise SaturationDetected(
            f"APAPANE probe reached {maximum:.1f} ADU, above the {guard_limit:.1f} ADU "
            f"guard during {label}; all DM segments were immediately tilted away"
        )
    if nframes == 1:
        return np.asarray(probe_frames[0], dtype=float), probe_frames, maximum
    _, remaining_frames = _average_frame(hardware.apapane_frames(nframes - 1))
    frames = np.concatenate((probe_frames, remaining_frames), axis=0)
    maximum = _guard_raw_max(frames, regions)
    if bool(guard.get("enabled", True)) and maximum >= guard_limit:
        hardware.set_dm(emergency_dm_state)
        raise SaturationDetected(
            f"APAPANE reached {maximum:.1f} ADU, above the {guard_limit:.1f} ADU guard "
            f"during {label}; all DM segments were immediately tilted away"
        )
    return np.nanmean(frames, axis=0), frames, maximum


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


def _next_output_base(root: Path, folder: str) -> Path:
    day = root / datetime.now().astimezone().strftime("%Y%m%d") / folder
    day.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^hdr_xy_(\d+)\.(?:json|fits|png)$")
    indices = []
    for path in day.glob("hdr_xy_*.*"):
        match = pattern.match(path.name)
        if match:
            indices.append(int(match.group(1)))
    return day / f"hdr_xy_{max(indices, default=0) + 1}"


def _reference_on_target(source: SourceXY, plan: ScanPlan, channel: str) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        (source.y, source.x),
        source.maps[channel],
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    xx, yy = np.meshgrid(plan.x, plan.y)
    return interpolator(np.column_stack((yy.ravel(), xx.ravel()))).reshape(xx.shape)


def _fits_arrays(
    source: SourceXY,
    plan: ScanPlan,
    acquired: np.ndarray,
    actual_x: np.ndarray,
    actual_y: np.ndarray,
    raw_max: np.ndarray,
    flux: dict[str, np.ndarray],
    flux_std: dict[str, np.ndarray],
    ratio: float | None,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "axis__x": plan.x,
        "axis__y": plan.y,
        "mask__excluded": plan.excluded.astype(np.uint8),
        "mask__acquired": acquired.astype(np.uint8),
        "axis__actual_x": actual_x,
        "axis__actual_y": actual_y,
        "map__raw_max_adu": raw_max,
        "source_axis__x": source.x,
        "source_axis__y": source.y,
        "source_mask__bright": plan.source_bright.astype(np.uint8),
    }
    for name, component in plan.mask_components.items():
        arrays[f"mask__{name}"] = component.astype(np.uint8)
    for channel in PHOT_BOXES:
        arrays[f"map__{channel}"] = flux[channel]
        arrays[f"std__{channel}"] = flux_std[channel]
        if channel in source.maps:
            arrays[f"source_map__{channel}"] = source.maps[channel]
        if ratio is not None:
            normalised = flux[channel] / ratio
            arrays[f"normalised_map__{channel}"] = normalised
            if channel in source.maps:
                reference = _reference_on_target(source, plan, channel)
                hdr = normalised.copy()
                use_reference = plan.excluded & np.isfinite(reference)
                hdr[use_reference] = reference[use_reference]
                arrays[f"hdr_map__{channel}"] = hdr
    return arrays


def _write_fits_atomic(path: Path, arrays: dict[str, np.ndarray], header: dict[str, Any]) -> None:
    try:
        from astropy.io import fits
    except ImportError as error:
        raise RuntimeError("astropy is required to save the HDR scan as FITS") from error
    primary = fits.PrimaryHDU()
    for key, value in header.items():
        primary.header[key] = value
    hdus = [primary]
    for index, (name, values) in enumerate(arrays.items(), start=1):
        hdu = fits.ImageHDU(np.asarray(values), name=f"DATA{index:03d}")
        hdu.header["LONGNAME"] = name
        hdus.append(hdu)
    temporary = path.with_name(f".{path.name}.tmp")
    fits.HDUList(hdus).writeto(temporary, overwrite=True)
    os.replace(temporary, path)


def _plot_result(
    path: Path,
    source: SourceXY,
    plan: ScanPlan,
    acquired: np.ndarray,
    flux: dict[str, np.ndarray],
    raw_max: np.ndarray,
    selected_channel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    source_map = np.asarray(source.maps[selected_channel], dtype=float)
    source_background = float(np.nanpercentile(source_map, 5))
    source_positive = np.maximum(source_map - source_background, 0.0)
    high_positive = np.where(acquired, np.maximum(flux[selected_channel], 0.0), np.nan)

    def norm_for(data: np.ndarray) -> LogNorm | None:
        positive = data[np.isfinite(data) & (data > 0)]
        if not len(positive):
            return None
        vmax = float(np.nanmax(positive))
        vmin = max(float(np.nanpercentile(positive, 5)), vmax * 1e-6, np.finfo(float).tiny)
        return LogNorm(vmin=vmin, vmax=max(vmax, vmin * 1.01))

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    source_image = axes[0, 0].pcolormesh(
        source.x, source.y, source_positive, shading="auto", cmap="magma", norm=norm_for(source_positive)
    )
    axes[0, 0].contour(source.x, source.y, plan.source_bright, levels=[0.5], colors="cyan")
    axes[0, 0].set_title(f"Reference {selected_channel}; cyan = source bright mask")
    figure.colorbar(source_image, ax=axes[0, 0], label="dark-subtracted integrated flux")

    plan_image = axes[0, 1].pcolormesh(
        plan.x, plan.y, plan.excluded.astype(float), shading="nearest", cmap="gray_r", vmin=0, vmax=1
    )
    axes[0, 1].scatter(plan.center_x, plan.center_y, marker="+", color="red", s=80)
    component_colours = {
        "direct_reference_bright": "cyan",
        "segment11_on_microlens20": "lime",
        "segment11_on_microlens31": "deepskyblue",
    }
    legend_handles = []
    for name, component in plan.mask_components.items():
        if np.any(component) and np.any(~component):
            colour = component_colours.get(name, "orange")
            axes[0, 1].contour(
                plan.x,
                plan.y,
                component,
                levels=[0.5],
                colors=[colour],
                linewidths=1.5,
            )
            legend_handles.append(Line2D([], [], color=colour, label=name.replace("_", " ")))
    if legend_handles:
        axes[0, 1].legend(handles=legend_handles, loc="best", fontsize=8)
    axes[0, 1].set_title("Planned grid: white excluded; outlines show mask source")
    figure.colorbar(plan_image, ax=axes[0, 1], ticks=[0, 1])

    high_image = axes[1, 0].pcolormesh(
        plan.x, plan.y, high_positive, shading="auto", cmap="viridis", norm=norm_for(high_positive)
    )
    axes[1, 0].set_title(f"High-flux {selected_channel}")
    figure.colorbar(high_image, ax=axes[1, 0], label="dark-subtracted integrated flux")

    raw_image = axes[1, 1].pcolormesh(plan.x, plan.y, raw_max, shading="auto", cmap="plasma")
    axes[1, 1].set_title("Maximum raw APAPANE pixel in guarded regions")
    figure.colorbar(raw_image, ax=axes[1, 1], label="ADU")
    for axis in axes.ravel():
        axis.set_xlabel("Mount X [pulses]")
        axis.set_ylabel("Mount Y [pulses]")
        axis.set_aspect("equal", adjustable="box")
    figure.suptitle("GLINT masked HDR-like mount X/Y scan")
    figure.savefig(path, dpi=170, facecolor="white")
    plt.close(figure)


def _print_plan(
    source: SourceXY,
    plan: ScanPlan,
    scan: dict[str, Any],
    channels: Sequence[str],
) -> None:
    acquisition = scan["acquisition"]
    print("\nResolved HDR mount scan")
    print(f"  source JSON        : {source.json_path}")
    print(f"  source arrays      : {source.array_path}")
    print(f"  source completed   : {source.completed_utc.isoformat()}")
    print(f"  injected segment   : {scan['injected_segment']} ({SEGMENT_TO_PHOT[int(scan['injected_segment'])]})")
    print(f"  mask channels      : {', '.join(channels)}")
    print(
        f"  exclusion          : >= {100 * float(scan['mask']['threshold_fraction_of_peak']):.3g}% "
        f"of each background-subtracted peak, dilated by {float(scan['mask']['dilation_pulses']):g} pulses"
    )
    print(f"  center              : x={plan.center_x:.3f}, y={plan.center_y:.3f}")
    print(
        f"  grid                : x={plan.x[0]:g}..{plan.x[-1]:g}, "
        f"y={plan.y[0]:g}..{plan.y[-1]:g}, step={float(scan['scan_extent']['step_pulses']):g}"
    )
    overlap = plan.mask_metadata.get("segment_overlap", {})
    if overlap.get("enabled"):
        print("  flat-segment overlap masks:")
        for target in overlap["target_segments"]:
            details = overlap["targets"][str(target)]
            center = details["predicted_overlap_center"]
            offset = details["offset_pulses"]
            print(
                f"    segment {overlap['source_segment']} on microlens {target}: "
                f"center=({center['x']:.3f}, {center['y']:.3f}), "
                f"offset=({offset[0]:+.3f}, {offset[1]:+.3f}), "
                f"excluded={details['target_excluded_points']} points"
            )
    print(
        f"  points              : {len(plan.acquisition_order)} acquired, "
        f"{int(np.count_nonzero(plan.excluded))} excluded, {len(plan.route)} safe motion nodes"
    )
    minimum_seconds = len(plan.acquisition_order) * float(acquisition["settle_s"])
    print(f"  lower-bound time    : {minimum_seconds / 60:.1f} min plus camera and mount motion")


def _confirm(prompt: str, expected: str) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError(f"Interactive confirmation required: {prompt}")
    answer = input(prompt).strip()
    if answer != expected:
        raise RuntimeError(f"Confirmation failed; expected {expected!r}")


def _snapshot_metadata(snapshot: Any) -> dict[str, Any]:
    return {
        "mount": {key: float(value) for key, value in snapshot.mount.items()},
        "conex": {key: float(value) for key, value in snapshot.conex.items()},
        "dm": dm_state_to_dict(snapshot.dm),
    }


def _restore_safely(
    hardware: InstrumentHardware,
    instrument: dict[str, Any],
    snapshot: Any,
    *,
    prompt_before_restore: bool,
) -> None:
    # Suppress the selected beam before moving back through any bright position;
    # restore the original DM PTT only after the mount is home again.
    try:
        hardware.set_dm(segment_state(instrument, "tilted_away"))
    except Exception as error:
        print(f"WARNING: could not park DM tilted-away before restoration: {error}", file=sys.stderr)
    if prompt_before_restore and sys.stdin.isatty():
        while input("Lower the source flux, then type SAFE to restore the pre-scan state: ").strip() != "SAFE":
            print("Waiting for SAFE so the original DM state is not restored at high source flux.")
    timeout = float(instrument["safety"]["motion_timeout_s"])
    # Keep the away PTT state during mount travel.  Voltage and original PTT
    # restoration are deliberately last.
    hardware.move_mount(snapshot.mount, timeout)
    hardware.move_conex(snapshot.conex)
    hardware.set_dm_volts(snapshot.dm_volts)
    hardware.set_dm(snapshot.dm)


def run(args: argparse.Namespace) -> Path | None:
    instrument_path = args.instrument_config.expanduser().resolve()
    scan_path = args.scan_config.expanduser().resolve()
    instrument = load_config(instrument_path, simulation=False)
    scan = validate_scan_config(load_json(scan_path))
    channels = mask_channels(scan)
    required_channels = tuple(dict.fromkeys((*channels, *PHOT_BOXES.keys())))
    source = resolve_source_xy(
        instrument, scan, scan_path, required_channels, args.source_json
    )
    plan = make_plan(source, scan, channels)
    _print_plan(source, plan, scan, channels)
    if args.check_only:
        return None
    validate_emergency_tilt_state(instrument)
    if not args.yes:
        _confirm("Proceed with physical DM and mount commands? Type YES: ", "YES")

    dark = load_saved_dark(instrument, "apapane")
    if dark is None:
        raise RuntimeError(
            f"No readable shared APAPANE dark at {alignment_root(instrument) / 'darks'}; "
            "acquire one with the current detector settings before this scan"
        )
    dark_path = alignment_root(instrument) / "darks" / "apapane_dark.fits"
    if not dark_path.is_file():
        dark_path = alignment_root(instrument) / "darks" / "apapane_dark.npy"

    output_base = _next_output_base(
        alignment_root(instrument), str(scan.get("output_folder", "Alignment_output_individual"))
    )
    json_path = output_base.with_suffix(".json")
    fits_path = output_base.with_suffix(".fits")
    png_path = output_base.with_suffix(".png")
    shape = plan.excluded.shape
    acquired = np.zeros(shape, dtype=bool)
    actual_x = np.full(shape, np.nan)
    actual_y = np.full(shape, np.nan)
    raw_max = np.full(shape, np.nan)
    flux = {channel: np.full(shape, np.nan) for channel in PHOT_BOXES}
    flux_std = {channel: np.full(shape, np.nan) for channel in PHOT_BOXES}
    ratio_value = scan.get("normalization", {}).get("high_to_reference_flux_ratio")
    ratio = None if ratio_value is None else float(ratio_value)
    started = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "hdr_xy",
        "status": "starting",
        "started_utc": started.isoformat(),
        "instrument_config": str(instrument_path),
        "scan_config": str(scan_path),
        "scan_settings": scan,
        "source_xy": {
            "json": str(source.json_path),
            "array": str(source.array_path),
            "completed_utc": source.completed_utc.isoformat(),
            "json_sha256": _sha256(source.json_path),
            "array_sha256": _sha256(source.array_path),
            "selection": "explicit_override" if args.source_json is not None else str(scan.get("source_xy_json", "latest")),
        },
        "selected_channel": SEGMENT_TO_PHOT[int(scan["injected_segment"])],
        "axes": {"x": plan.x.tolist(), "y": plan.y.tolist()},
        "center": {"x": plan.center_x, "y": plan.center_y},
        "mask": plan.mask_metadata,
        "planned_acquisition_points": len(plan.acquisition_order),
        "planned_safe_motion_nodes": len(plan.route),
        "acquired_points": 0,
        "outputs": {"json": json_path.name, "fits": fits_path.name, "figure": png_path.name},
        "apapane_dark": {"path": str(dark_path), "sha256": _sha256(dark_path)},
    }
    _atomic_json(json_path, payload)

    hardware: InstrumentHardware | None = None
    snapshot = None
    command_log_start = 0
    failure: BaseException | None = None
    acquisition = scan["acquisition"]
    checkpoint_every = int(acquisition["checkpoint_every"])
    selected_channel = SEGMENT_TO_PHOT[int(scan["injected_segment"])]
    guard = scan["saturation_guard"]
    guard_limit = (
        float(guard["adc_max_adu"]) * float(guard["fraction"])
        if bool(guard.get("enabled", True))
        else float("inf")
    )

    def checkpoint(status: str) -> None:
        arrays = _fits_arrays(
            source, plan, acquired, actual_x, actual_y, raw_max, flux, flux_std, ratio
        )
        _write_fits_atomic(
            fits_path,
            arrays,
            {
                "SCANTYPE": "HDR_XY",
                "STATUS": status,
                "SEGMENT": int(scan["injected_segment"]),
                "MASKFRAC": float(scan["mask"]["threshold_fraction_of_peak"]),
                "DATE-OBS": started.isoformat(),
            },
        )
        payload["status"] = status
        payload["acquired_points"] = int(np.count_nonzero(acquired))
        _atomic_json(json_path, payload)

    try:
        hardware = InstrumentHardware(instrument)
        validate_mount_plan(plan, hardware)
        command_log_start = hardware.command_log_marker()
        fixed_roll = float(instrument["mount_control"]["fixed_roll"])
        hardware.move_mount(
            {"roll": fixed_roll}, float(instrument["safety"]["motion_timeout_s"])
        )
        snapshot = hardware.snapshot()
        payload["pre_scan_state"] = _snapshot_metadata(snapshot)
        payload["status"] = "positioning_at_low_flux"
        _atomic_json(json_path, payload)
        emergency_dm_state = segment_state(instrument, "tilted_away")
        # No injected segment remains flat while the mount travels to the first
        # scan coordinate.  This is deliberately done before any X/Y motion.
        hardware.set_dm(emergency_dm_state)
        first_iy, first_ix = plan.acquisition_order[0]
        hardware.move_mount(
            {"roll": fixed_roll, "x": float(plan.x[first_ix]), "y": float(plan.y[first_iy])},
            float(instrument["safety"]["motion_timeout_s"]),
        )
        if bool(acquisition.get("pause_for_flux_change", True)):
            _confirm(
                "Mount is at the first safe point with all DM segments tilted away. "
                "Confirm the source is at LOW/reference flux, then type LOW: ",
                "LOW",
            )
        # Only now is the selected injected segment made flat.  Check one frame
        # at the low/reference setting before asking the operator to raise flux.
        hardware.set_dm(scan_dm_state(instrument, source, scan))
        time.sleep(float(acquisition["settle_s"]))
        _, _, pilot_max = _guarded_apapane_frames(
            hardware,
            nframes=1,
            guard=guard,
            guard_limit=guard_limit,
            emergency_dm_state=emergency_dm_state,
            label="low-flux pilot at the first safe position",
        )
        payload["low_flux_pilot"] = {
            "x": float(plan.x[first_ix]),
            "y": float(plan.y[first_iy]),
            "raw_max_adu": float(pilot_max),
            "guard_limit_adu": None if not np.isfinite(guard_limit) else float(guard_limit),
        }
        _atomic_json(json_path, payload)
        if bool(acquisition.get("pause_for_flux_change", True)):
            _confirm(
                f"Segment {int(scan['injected_segment'])} is flat at the safe start and the "
                f"low-flux probe peaked at {pilot_max:.1f} ADU. Increase flux gradually, "
                "then type READY: ",
                "READY",
            )
        payload["status"] = "running"
        _atomic_json(json_path, payload)

        seen: set[tuple[int, int]] = set()
        total = len(plan.acquisition_order)
        report_every = max(1, total // 25)
        for node in plan.route:
            iy, ix = node
            if node in seen:
                # This is a safe transit node used to route around the bright hole.
                hardware.move_mount(
                    {"roll": fixed_roll, "x": float(plan.x[ix]), "y": float(plan.y[iy])},
                    float(instrument["safety"]["motion_timeout_s"]),
                )
                continue
            hardware.move_mount(
                {"roll": fixed_roll, "x": float(plan.x[ix]), "y": float(plan.y[iy])},
                float(instrument["safety"]["motion_timeout_s"]),
            )
            time.sleep(float(acquisition["settle_s"]))
            mean_frame, frames, point_raw_max = _guarded_apapane_frames(
                hardware,
                nframes=int(acquisition["nframes"]),
                guard=guard,
                guard_limit=guard_limit,
                emergency_dm_state=emergency_dm_state,
                label=f"HDR acquisition at x={plan.x[ix]:g}, y={plan.y[iy]:g}",
            )
            corrected_frames = frames - dark[None, ...]
            corrected_mean = mean_frame - dark
            raw_max[iy, ix] = point_raw_max
            readback = hardware.get_mount()
            actual_x[iy, ix] = float(readback["x"])
            actual_y[iy, ix] = float(readback["y"])
            for channel, box in PHOT_BOXES.items():
                flux[channel][iy, ix] = integrate_box(corrected_mean, box)
                y0, y1, x0, x1 = map(int, box)
                per_frame = corrected_frames[:, y0:y1, x0:x1].sum(axis=(1, 2))
                flux_std[channel][iy, ix] = float(np.std(per_frame, ddof=1)) if len(per_frame) > 1 else 0.0
            acquired[iy, ix] = True
            seen.add(node)
            count = len(seen)
            if count == 1 or count == total or count % report_every == 0:
                print(
                    f"  [{count:4d}/{total}] x={plan.x[ix]:g}, y={plan.y[iy]:g}, "
                    f"{selected_channel}={flux[selected_channel][iy, ix]:.6g}, raw_max={raw_max[iy, ix]:.1f}"
                )
            if count % checkpoint_every == 0:
                checkpoint("running")
        checkpoint("acquisition_complete")
    except BaseException as error:
        failure = error
        payload["status"] = "failed"
        payload["error"] = f"{type(error).__name__}: {error}"
        if hardware is not None:
            try:
                checkpoint("failed")
            except Exception as save_error:
                payload["checkpoint_error"] = f"{type(save_error).__name__}: {save_error}"
                _atomic_json(json_path, payload)
    finally:
        if hardware is not None and snapshot is not None:
            try:
                _restore_safely(
                    hardware,
                    instrument,
                    snapshot,
                    prompt_before_restore=bool(acquisition.get("pause_before_restore", True)),
                )
                payload["restored_pre_scan_state"] = True
                payload["state_after_restore"] = {
                    "mount": hardware.get_mount(),
                    "conex": hardware.get_conex(),
                    "dm": dm_state_to_dict(hardware.get_dm()),
                }
            except BaseException as restore_error:
                payload["restored_pre_scan_state"] = False
                payload["restore_error"] = f"{type(restore_error).__name__}: {restore_error}"
                if failure is None:
                    failure = restore_error
            try:
                log = hardware.export_command_log(output_base.parent, command_log_start)
                if log is not None:
                    payload["command_log"] = log
            except Exception as log_error:
                payload["command_log_error"] = f"{type(log_error).__name__}: {log_error}"
        if hardware is not None:
            hardware.close()

    try:
        if fits_path.is_file():
            _plot_result(
                png_path, source, plan, acquired, flux, raw_max, selected_channel
            )
    except Exception as plot_error:
        payload["plot_error"] = f"{type(plot_error).__name__}: {plot_error}"
    payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
    payload["duration_s"] = (
        datetime.fromisoformat(payload["completed_utc"]) - started
    ).total_seconds()
    if failure is None:
        payload["status"] = "completed"
    else:
        payload["status"] = "failed"
    _atomic_json(json_path, payload)
    if failure is not None:
        raise failure
    print(f"\nSaved HDR scan: {json_path.parent}")
    return json_path.parent


def main() -> None:
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("\nInterrupted by operator.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
