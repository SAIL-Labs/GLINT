"""Validated detector boxes loaded from the central configuration file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BOX_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "detector_boxes.json"


def _validate_box(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{label} must be [y0, y1, x0, x1]")
    try:
        y0, y1, x0, x1 = map(int, value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain four integer coordinates") from error
    if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
        raise ValueError(f"{label} contains invalid detector bounds: {value}")
    return [y0, y1, x0, x1]


def load_detector_boxes(path: str | Path | None = None) -> dict[str, Any]:
    """Read and validate one complete detector-box snapshot from JSON.

    APAPANE Live calls this for every acquired update, so JSON changes affect
    both its overlays and integrations without restarting the GUI.
    """
    source = Path(path).expanduser().resolve() if path is not None else BOX_CONFIG_PATH
    with source.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("convention") != "[y0, y1, x0, x1]":
        raise ValueError("detector_boxes.json convention must be [y0, y1, x0, x1]")
    for section in ("photometry", "null", "fringe_tracking"):
        if not isinstance(payload.get(section), dict) or not payload[section]:
            raise ValueError(f"detector_boxes.json requires a non-empty {section} object")
    photometry = {
        str(name): _validate_box(box, f"photometry.{name}")
        for name, box in payload["photometry"].items()
    }
    null = {
        str(name): _validate_box(box, f"null.{name}")
        for name, box in payload["null"].items()
    }
    fringe_tracking: dict[str, dict[str, list[int]]] = {}
    for name, channels in payload["fringe_tracking"].items():
        if not isinstance(channels, dict) or set(channels) != {"FT1", "FT2"}:
            raise ValueError(f"fringe_tracking.{name} must contain exactly FT1 and FT2")
        fringe_tracking[str(name)] = {
            channel: _validate_box(box, f"fringe_tracking.{name}.{channel}")
            for channel, box in channels.items()
        }
    if set(null) != set(fringe_tracking):
        missing_ft = sorted(set(null) - set(fringe_tracking))
        missing_null = sorted(set(fringe_tracking) - set(null))
        raise ValueError(
            "Null and fringe-tracking keys must match exactly; "
            f"missing FT entries={missing_ft}, missing null entries={missing_null}"
        )
    return {
        "convention": payload["convention"],
        "photometry": photometry,
        "null": null,
        "fringe_tracking": fringe_tracking,
    }


_BOX_CONFIG = load_detector_boxes()

PHOT_BOXES = _BOX_CONFIG["photometry"]
NULL_BOXES = _BOX_CONFIG["null"]
FRINGE_TRACKING_BOXES = _BOX_CONFIG["fringe_tracking"]

INJECTED_SEGMENTS = (11, 20, 31)
SEGMENT_TO_PHOT = {11: "Phot 1", 20: "Phot 2", 31: "Phot 3"}
BASELINE_TO_NULL = {(11, 20): "N1 / 11-20", (11, 31): "N2 / 11-31"}

DIFFERENTIAL_BASELINE_TO_NULL = dict(BASELINE_TO_NULL)


def differential_null_boxes(box_config: dict[str, Any] | None = None) -> dict[tuple[int, int], list[int]]:
    """Resolve active differential-scan baselines from one box snapshot."""
    null = (box_config or load_detector_boxes())["null"]
    return {
        baseline: list(null[name]) for baseline, name in DIFFERENTIAL_BASELINE_TO_NULL.items()
    }


DIFFERENTIAL_NULL_BOXES = differential_null_boxes(_BOX_CONFIG)


def integrate_box(frame, box):
    """Integrate a detector box after validating its bounds."""
    y0, y1, x0, x1 = map(int, box)
    if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
        raise ValueError(f"Invalid box {box}")
    if y1 > frame.shape[-2] or x1 > frame.shape[-1]:
        raise ValueError(f"Box {box} exceeds frame shape {frame.shape}")
    return float(frame[y0:y1, x0:x1].sum())
