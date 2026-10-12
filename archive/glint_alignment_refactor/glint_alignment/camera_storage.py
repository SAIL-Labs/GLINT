from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CAMERA_LABELS = {
    "palila": "Palila",
    "apapane": "APAPANE",
    "focal": "Focal plane",
    "pupil": "Pupil plane",
}


def safe_filename(value: str) -> str:
    value = Path(value.strip()).stem
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not value:
        raise ValueError("Filename is empty after removing unsupported characters")
    return value


def resolve_camera_directory(requested_directory: str, config: dict) -> Path:
    try:
        configured_root = str(config["camera_frame_root"])
    except KeyError as error:
        raise ValueError("camera_frame_root must be defined in the active JSON configuration") from error
    value = requested_directory.strip() or configured_root
    directory = Path(value).expanduser()
    if not directory.is_absolute():
        directory = Path(config["_root"]) / directory
    return directory.resolve()


def save_camera_frame(
    frame: np.ndarray,
    camera: str,
    requested_name: str,
    config: dict,
    requested_directory: str = "",
) -> Path:
    """Save a camera frame to the operator-selected directory and basename."""
    if camera not in CAMERA_LABELS:
        raise ValueError(f"Unknown camera {camera!r}")
    now = datetime.now(timezone.utc)
    directory = resolve_camera_directory(requested_directory, config)
    directory.mkdir(parents=True, exist_ok=True)
    basename = safe_filename(requested_name) if requested_name.strip() else f"{camera}_{now.strftime('%Y%m%d_%H%M%S_%f')}"
    try:
        from astropy.io import fits

        path = directory / f"{basename}.fits"
        fits.PrimaryHDU(np.asarray(frame)).writeto(path, overwrite=False)
    except ImportError:
        path = directory / f"{basename}.npy"
        if path.exists():
            raise FileExistsError(path)
        np.save(path, np.asarray(frame), allow_pickle=False)
    return path
