from __future__ import annotations

import json
import re
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .models import ScanResult, dm_state_to_dict


REVIEW_FIGURE_KINDS = {"bulk", "pitch_yaw", "xy", "tip_tilt", "null"}
SCAN_FILE_NAMES = {
    "focal_dark": "01_focal_dark",
    "apapane_dark": "02_apapane_dark",
    "bulk": "03_bulk_alignment",
    "set_z": "04_set_z",
    "pitch_yaw": "05_pitch_yaw",
    "xy": "06_xy",
    "tip_tilt": "07_tip_tilt",
    "null": "08_null",
}
_LOCK = threading.RLock()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def alignment_root(config: dict) -> Path:
    root = Path(config["output_root"]).expanduser()
    if not root.is_absolute():
        root = Path(config["_root"]) / root
    return root.resolve()


def begin_alignment(config: dict, *, now: datetime | None = None) -> Path:
    """Create and activate the next Alignment_outputs_N directory."""
    # Folder dates follow the telescope computer's local observing date.
    timestamp = now or datetime.now().astimezone()
    day = alignment_root(config) / timestamp.strftime("%Y%m%d")
    with _LOCK:
        day.mkdir(parents=True, exist_ok=True)
        indices = []
        for path in day.glob("Alignment_outputs_*"):
            try:
                indices.append(int(path.name.rsplit("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        directory = day / f"Alignment_outputs_{max(indices, default=0) + 1}"
        directory.mkdir(parents=False, exist_ok=False)
        config["_alignment_session_dir"] = str(directory)
    return directory


def active_alignment(config: dict, *, create: bool = True) -> Path | None:
    value = config.get("_alignment_session_dir")
    if value:
        path = Path(value)
        if path.is_dir():
            return path
    return begin_alignment(config) if create else None


def individual_alignment_directory(config: dict, *, now: datetime | None = None) -> Path:
    """Return the persistent per-day destination for operator-run scans."""
    timestamp = now or datetime.now().astimezone()
    directory = alignment_root(config) / timestamp.strftime("%Y%m%d") / "Alignment_output_individual"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _next_scan_stem(result: ScanResult, directory: Path) -> str:
    """Allocate an always-suffixed scan name using files already on disk."""
    base = SCAN_FILE_NAMES.get(result.kind, _safe_name(result.kind))
    pattern = re.compile(rf"^{re.escape(base)}_(\d+)\.json$")
    indices = []
    for path in directory.glob(f"{base}_*.json"):
        match = pattern.match(path.name)
        if match:
            indices.append(int(match.group(1)))
    return f"{base}_{max(indices, default=0) + 1}"


def _json_safe_config(config: dict) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in config.items() if not str(key).startswith("_")}


def _write_arrays(path: Path, result: ScanResult, decision: str) -> tuple[str, str]:
    arrays = {f"axis__{key}": np.asarray(value) for key, value in result.axes.items()}
    arrays.update({f"map__{key}": np.asarray(value) for key, value in result.maps.items()})
    try:
        from astropy.io import fits

        hdus = [fits.PrimaryHDU()]
        hdus[0].header["SCANTYPE"] = result.kind
        hdus[0].header["DECISION"] = decision
        hdus[0].header["DATE-OBS"] = result.completed_utc
        for index, (name, array) in enumerate(arrays.items(), start=1):
            hdu = fits.ImageHDU(np.asarray(array), name=f"DATA{index:03d}")
            hdu.header["LONGNAME"] = name
            hdus.append(hdu)
        fits.HDUList(hdus).writeto(path.with_suffix(".fits"), overwrite=True)
        return path.with_suffix(".fits").name, "FITS"
    except ImportError:
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        return path.with_suffix(".npz").name, "NPZ fallback (install astropy for FITS)"


def _save_dark(result: ScanResult, hardware, config: dict, decision: str) -> Path:
    camera = result.kind.removesuffix("_dark")
    directory = alignment_root(config) / "darks"
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / f"{camera}_dark"
    result.metadata["review_decision"] = decision
    result.metadata["state_after_review"] = {
        "mount": hardware.get_mount(),
        "conex": hardware.get_conex(),
        "dm": dm_state_to_dict(hardware.get_dm()),
    }
    dark = np.asarray(result.maps["dark"])
    try:
        from astropy.io import fits

        fits.PrimaryHDU(dark).writeto(base.with_suffix(".fits"), overwrite=True)
        array_name, array_format = base.with_suffix(".fits").name, "FITS"
    except ImportError:
        np.save(base.with_suffix(".npy"), dark, allow_pickle=False)
        array_name, array_format = base.with_suffix(".npy").name, "NPY fallback (install astropy for FITS)"
    summary = result.json_summary()
    summary.update({"array_file": array_name, "array_format": array_format, "updated_in_place": True})
    json_path = base.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    result.metadata["saved_files"] = {"json": str(json_path), "array": str(directory / array_name)}
    return directory


def load_saved_dark(config: dict, camera: str) -> np.ndarray | None:
    """Load the latest shared dark, allowing a partial workflow after restart."""
    if camera not in {"focal", "apapane"}:
        raise ValueError(f"Unsupported alignment dark camera {camera!r}")
    base = alignment_root(config) / "darks" / f"{camera}_dark"
    fits_path = base.with_suffix(".fits")
    npy_path = base.with_suffix(".npy")
    if fits_path.is_file():
        try:
            from astropy.io import fits
        except ImportError:
            return None
        return np.asarray(fits.getdata(fits_path), dtype=float)
    if npy_path.is_file():
        return np.asarray(np.load(npy_path, allow_pickle=False), dtype=float)
    return None


def save_result(result: ScanResult, hardware, config: dict, decision: str) -> Path:
    """Save a reviewed scan flat inside the current alignment-output folder.

    Darks are the exception: each camera dark is overwritten under the shared
    ``alignment_scan/darks`` directory.
    """
    with _LOCK:
        if result.kind.endswith("_dark"):
            return _save_dark(result, hardware, config, decision)
        scope = result.metadata.get("run_scope")
        if scope is None:
            scope = "workflow" if active_alignment(config, create=False) is not None else "individual"
            result.metadata["run_scope"] = scope
        if scope == "workflow":
            directory = active_alignment(config, create=True)
            assert directory is not None
        elif scope == "individual":
            directory = individual_alignment_directory(config)
        else:
            raise ValueError(f"Unknown scan run scope {scope!r}")
        stem = _next_scan_stem(result, directory)
        base = directory / stem
        result.metadata["review_decision"] = decision
        result.metadata["state_after_review"] = {
            "mount": hardware.get_mount(),
            "conex": hardware.get_conex(),
            "dm": dm_state_to_dict(hardware.get_dm()),
        }
        command_log = hardware.export_command_log(
            directory,
            int(result.metadata.get("command_log_start_sequence", 0)),
        )
        if command_log is not None:
            result.metadata["command_log"] = command_log
        array_name, array_format = _write_arrays(base, result, decision)
        summary = result.json_summary()
        summary.update({"array_file": array_name, "array_format": array_format})
        json_path = base.with_suffix(".json")
        with json_path.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
        result.metadata["saved_files"] = {"json": str(json_path), "array": str(directory / array_name)}
        return directory


def save_review_figure(result: ScanResult, directory: Path, figure) -> Path | None:
    """Save the exact GUI review figure beside its FITS and JSON files."""
    if result.kind not in REVIEW_FIGURE_KINDS:
        return None
    files = result.metadata.get("saved_files", {})
    json_path = Path(files.get("json", ""))
    if not json_path.is_file():
        return None
    figure_path = json_path.with_suffix(".png")
    figure.savefig(figure_path, dpi=160, facecolor=figure.get_facecolor())
    with json_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    summary["review_figure_file"] = figure_path.name
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    files["figure"] = str(figure_path)
    return figure_path


def finalize_result(result: ScanResult, hardware, config: dict, decision: str) -> Path:
    """Finalize the already-saved review record without allocating a new stem."""
    files = result.metadata.get("saved_files", {})
    json_path = Path(files.get("json", ""))
    if not json_path.is_file():
        return save_result(result, hardware, config, decision)
    with _LOCK:
        with json_path.open("r", encoding="utf-8") as stream:
            existing = json.load(stream)
        result.metadata["review_decision"] = decision
        result.metadata["state_after_review"] = {
            "mount": hardware.get_mount(),
            "conex": hardware.get_conex(),
            "dm": dm_state_to_dict(hardware.get_dm()),
        }
        command_log = hardware.export_command_log(
            json_path.parent,
            int(result.metadata.get("command_log_start_sequence", 0)),
        )
        if command_log is not None:
            result.metadata["command_log"] = command_log
        summary = result.json_summary()
        for key in ("array_file", "array_format", "review_figure_file"):
            if key in existing:
                summary[key] = existing[key]
        with json_path.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
    return json_path.parent


def save_final_configuration(
    config: dict,
    hardware,
    engine,
    *,
    workflow_status: dict[str, str] | None = None,
    workflow_range: dict[str, Any] | None = None,
) -> Path:
    """Write the final hardware and selected operational configuration."""
    directory = active_alignment(config, create=False)
    if directory is None:
        raise RuntimeError("No automatic alignment output directory is active")
    payload = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "mount": hardware.get_mount(),
        "conex": hardware.get_conex(),
        "dm": dm_state_to_dict(hardware.get_dm()),
        "rest_of_dm": engine.rest_dm_selection(),
        "workflow": {
            "status": deepcopy(workflow_status or {}),
            "requested_range": deepcopy(workflow_range or {}),
        },
        "application_configuration": _json_safe_config(config),
    }
    path = directory / "final_configuration.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    return path
