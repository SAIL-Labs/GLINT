from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .boxes import integrate_box, load_detector_boxes


DISPLAY_XMIN = 200
DISPLAY_XMAX = 320
DEFAULT_REFRESH_SECONDS = 10.0
DEFAULT_NFRAMES = 25
DEFAULT_WAVELENGTH_MAP_FILENAME = "pixel_to_wavelength.json"


def resolve_wavelength_map_path(config: dict) -> Path:
    """Resolve the APAPANE wavelength map beside the active JSON config.

    ``null_live.wavelength_map`` may name a different file manually. If that
    setting is absent, the canonical ``pixel_to_wavelength.json`` is used. A
    single suffixed copy such as ``pixel_to_wavelength(2).json`` is accepted so
    downloaded calibration files do not need a code change.
    """
    configured = str(
        config.get("null_live", {}).get(
            "wavelength_map",
            DEFAULT_WAVELENGTH_MAP_FILENAME,
        )
    ).strip()
    if not configured:
        raise ValueError("null_live.wavelength_map must not be empty")
    path = Path(configured).expanduser()
    if not path.is_absolute():
        config_path = config.get("_config_path")
        if not config_path:
            raise ValueError("The active configuration has no _config_path")
        path = Path(config_path).expanduser().resolve().parent / path
    path = path.resolve()
    if path.exists():
        return path
    if configured == DEFAULT_WAVELENGTH_MAP_FILENAME:
        matches = sorted(path.parent.glob("pixel_to_wavelength*.json"))
        if len(matches) == 1:
            return matches[0].resolve()
        if len(matches) > 1:
            names = ", ".join(candidate.name for candidate in matches)
            raise ValueError(
                "Several pixel-to-wavelength maps were found; set "
                f"null_live.wavelength_map explicitly in the active config: {names}"
            )
    raise FileNotFoundError(f"APAPANE pixel-to-wavelength map not found: {path}")


def wavelength_markers(calibration: dict) -> list[dict[str, float | str]]:
    """Flatten all calibrated photometry, null and FT trace coordinates."""
    markers: list[dict[str, float | str]] = []

    def append_channel(group: str, channel: str, samples: dict) -> None:
        if not isinstance(samples, dict) or not samples:
            raise ValueError(f"Wavelength calibration {group}.{channel} must contain samples")
        for wavelength_text, position in samples.items():
            try:
                wavelength = float(wavelength_text)
                x = float(position["x"])
                y = float(position["y"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid wavelength coordinate at {group}.{channel}.{wavelength_text}"
                ) from error
            if not np.all(np.isfinite([wavelength, x, y])):
                raise ValueError(
                    f"Non-finite wavelength coordinate at {group}.{channel}.{wavelength_text}"
                )
            markers.append(
                {
                    "group": group,
                    "channel": channel,
                    "wavelength_nm": wavelength,
                    "x": x,
                    "y": y,
                }
            )

    for group in ("photometry", "null"):
        channels = calibration.get(group)
        if not isinstance(channels, dict) or not channels:
            raise ValueError(f"Wavelength calibration is missing {group}")
        for channel, samples in channels.items():
            append_channel(group, str(channel), samples)

    fringe_tracking = calibration.get("fringe_tracking")
    if not isinstance(fringe_tracking, dict) or not fringe_tracking:
        raise ValueError("Wavelength calibration is missing fringe_tracking")
    for baseline, channels in fringe_tracking.items():
        if not isinstance(channels, dict) or not channels:
            raise ValueError(f"Wavelength calibration fringe_tracking.{baseline} is empty")
        for channel, samples in channels.items():
            append_channel("fringe_tracking", f"{baseline} {channel}", samples)
    return markers


def load_wavelength_calibration(config: dict) -> dict:
    """Load and validate the wavelength map used by the Null Live overlay."""
    path = resolve_wavelength_map_path(config)
    with path.open(encoding="utf-8") as stream:
        calibration = json.load(stream)
    try:
        shape = tuple(int(value) for value in calibration["detector_shape_yx"])
        wavelength_range = tuple(float(value) for value in calibration["calibrated_range_nm"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Wavelength calibration requires detector_shape_yx and calibrated_range_nm"
        ) from error
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("detector_shape_yx must contain two positive integers")
    if len(wavelength_range) != 2 or not 0 < wavelength_range[0] < wavelength_range[1]:
        raise ValueError("calibrated_range_nm must contain two increasing positive wavelengths")
    markers = wavelength_markers(calibration)
    for marker in markers:
        wavelength = float(marker["wavelength_nm"])
        x = float(marker["x"])
        y = float(marker["y"])
        if not wavelength_range[0] <= wavelength <= wavelength_range[1]:
            raise ValueError(f"Wavelength {wavelength:g} nm lies outside calibrated_range_nm")
        if not (0 <= x < shape[1] and 0 <= y < shape[0]):
            raise ValueError(
                f"Calibrated point ({x:g}, {y:g}) lies outside detector_shape_yx={shape}"
            )
    calibration["_path"] = str(path)
    return calibration


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
