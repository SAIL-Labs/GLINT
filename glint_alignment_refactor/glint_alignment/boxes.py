"""Detector extraction boxes loaded from the central configuration folder."""

import json
from pathlib import Path


BOX_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "detector_boxes.json"
with BOX_CONFIG_PATH.open(encoding="utf-8") as stream:
    _BOX_CONFIG = json.load(stream)

PHOT_BOXES = _BOX_CONFIG["photometry"]
NULL_BOXES = _BOX_CONFIG["null"]
FRINGE_TRACKING_BOXES = _BOX_CONFIG["fringe_tracking"]

INJECTED_SEGMENTS = (11, 20, 31)
SEGMENT_TO_PHOT = {11: "Phot 1", 20: "Phot 2", 31: "Phot 3"}
BASELINE_TO_NULL = {(11, 20): "N1 / 11-20", (11, 31): "N3 / 11-31"}


def integrate_box(frame, box):
    """Integrate a detector box after validating its bounds."""
    y0, y1, x0, x1 = map(int, box)
    if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
        raise ValueError(f"Invalid box {box}")
    if y1 > frame.shape[-2] or x1 > frame.shape[-1]:
        raise ValueError(f"Box {box} exceeds frame shape {frame.shape}")
    return float(frame[y0:y1, x0:x1].sum())
