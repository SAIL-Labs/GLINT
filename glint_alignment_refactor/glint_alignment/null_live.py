from __future__ import annotations

import numpy as np

from .boxes import integrate_box, load_detector_boxes


DISPLAY_XMIN = 200
DISPLAY_XMAX = 320
DEFAULT_REFRESH_SECONDS = 10.0
DEFAULT_NFRAMES = 25


def format_flux(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{int(round(float(value))):,}"


def format_depth(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{float(value):.4f}"


def process_apapane_frame(frames: np.ndarray, dark: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(frames, dtype=float)
    if values.ndim == 2:
        values = values[None, :, :]
    if values.ndim != 3:
        raise ValueError(f"Expected APAPANE frame cube, got shape {values.shape}")
    image = np.nanmean(values, axis=0)
    if dark is not None:
        dark = np.asarray(dark, dtype=float)
        if dark.shape != image.shape:
            raise ValueError(f"APAPANE dark shape {dark.shape} does not match image {image.shape}")
        image = image - dark
    return image


def collect_null_statistics(
    image: np.ndarray,
    detector_boxes: dict | None = None,
) -> dict[str, dict[str, float]]:
    """Integrate one validated detector-box snapshot from the central JSON."""
    box_config = detector_boxes or load_detector_boxes()
    photometry_boxes = box_config["photometry"]
    null_boxes = box_config["null"]
    fringe_tracking_boxes = box_config["fringe_tracking"]
    statistics: dict[str, dict[str, float]] = {}
    for name, box in photometry_boxes.items():
        statistics[name] = {"flux": float(integrate_box(image, box))}
    for name, box in null_boxes.items():
        null_flux = float(integrate_box(image, box))
        ft1 = float(integrate_box(image, fringe_tracking_boxes[name]["FT1"]))
        ft2 = float(integrate_box(image, fringe_tracking_boxes[name]["FT2"]))
        denominator = ft1 + ft2
        statistics[name] = {
            "null_flux": null_flux,
            "ft1_flux": ft1,
            "ft2_flux": ft2,
            "ft_sum": denominator,
            "null_depth": float(null_flux / denominator) if np.isfinite(denominator) and denominator != 0 else np.nan,
        }
    return statistics
