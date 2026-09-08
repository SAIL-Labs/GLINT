#!/usr/bin/env python3
"""Standalone 37-segment GLINT quadrature tip/tilt diagnostic.

Every segment is rastered individually while the other 36 segments remain at
one common piston/tip/tilt state.  At every tip/tilt point, the scanned segment
is sampled at optical phases 0, pi/2, pi and 3pi/2.  Because the DM is
reflective, the physical surface-piston commands are 0, lambda/8, lambda/4 and
3*lambda/8, corresponding to optical-path differences 0, lambda/4, lambda/2
and 3*lambda/4.

The script attaches to the already-running APAPANE and DM PTT SHMs; it does not
initialise either device and does not move the mount or steering mirrors.

Telescope-owned paths and SHM names come from the GLINT alignment refactor's
``config/instrument.json``.
Experiment settings come from ``tiptilt_map_each_segment_quadrature.json``
beside this script.  The complete live DM state is captured before the first
command and is always restored on success, error, or Ctrl+C.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


N_DM_SEGMENTS = 37
PTT_WIDTH = 3
DEFAULT_WAVELENGTH_NM = 1500.0
OUTPUT_SUBDIRECTORY = "standalone_tiptilt_maps/each_segment_quadrature"
DEFAULT_INSTRUMENT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "glint_alignment_refactor"
    / "config"
    / "instrument.json"
)
DEFAULT_SCAN_CONFIG = (
    Path(__file__).resolve().parents[0]
    / "config"
    / "tiptilt_map_each_segment_quadrature.json"
)

@dataclass(frozen=True)
class BoxSpec:
    identifier: str
    section: str
    name: str
    channel: str | None
    bounds: tuple[int, int, int, int]

    @property
    def height(self) -> int:
        return self.bounds[1] - self.bounds[0]

    @property
    def width(self) -> int:
        return self.bounds[3] - self.bounds[2]

    def as_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "section": self.section,
            "name": self.name,
            "channel": self.channel,
            "bounds": list(self.bounds),
        }


@dataclass(frozen=True)
class QuadratureStep:
    index: int
    phase_deg: float
    opd_um: float
    surface_piston_um: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "index": self.index,
            "phase_deg": self.phase_deg,
            "optical_path_difference_um": self.opd_um,
            "dm_surface_piston_um": self.surface_piston_um,
        }


class PersistentDMLogger:
    """Append standalone DM commands to the refactor's persistent DM log."""

    def __init__(self, instrument_config: dict[str, Any], config_root: Path) -> None:
        settings = instrument_config.get("command_logging", {})
        self.enabled = bool(settings.get("enabled", True))
        self._stream = None
        self._sequence = 0
        self.path: Path | None = None
        if not self.enabled:
            return
        configured_root = settings.get("root")
        if not configured_root:
            raise ValueError("instrument.json command_logging.root is required")
        root = resolve_path(configured_root, config_root)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "dm_commands.jsonl"
        self.path.touch(exist_ok=True)
        self._sequence = self._last_sequence()
        self._stream = self.path.open("a", encoding="utf-8")

    def _last_sequence(self) -> int:
        if self.path is None:
            return 0
        try:
            with self.path.open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 65536))
                for line in reversed(stream.read().decode("utf-8", errors="ignore").splitlines()):
                    try:
                        return int(json.loads(line)["sequence"])
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        continue
        except OSError:
            pass
        return 0

    def record(
        self,
        action: str,
        command: dict[str, Any],
        *,
        targets: dict[str, Any] | None = None,
        outcome: str = "succeeded",
        error: str | None = None,
    ) -> None:
        if not self.enabled or self._stream is None:
            return
        self._sequence += 1
        record: dict[str, Any] = {
            "sequence": self._sequence,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "instrument-standalone-tiptilt",
            "record_type": "command",
            "device": "dm",
            "action": action,
            "command": command,
            "outcome": outcome,
        }
        if targets is not None:
            record["targets"] = targets
        if error is not None:
            record["error"] = str(error)
        self._stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        self._stream.flush()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.flush()
            self._stream.close()
            self._stream = None


class DMController:
    """Fresh-read DM PTT operations matching the refactor instrument backend."""

    def __init__(self, shm: Any, logger: PersistentDMLogger) -> None:
        self.shm = shm
        self.logger = logger

    def snapshot(self) -> np.ndarray:
        state = np.asarray(self.shm.get_data(), dtype=float).reshape(N_DM_SEGMENTS, PTT_WIDTH)
        if not np.all(np.isfinite(state)):
            raise RuntimeError("dmptt contains non-finite values")
        return state.copy()

    def set_full(
        self,
        state: np.ndarray,
        *,
        action: str,
        target_summary: dict[str, Any],
    ) -> None:
        target = validate_dm_state(state)
        try:
            current = np.asarray(self.shm.get_data()).reshape(N_DM_SEGMENTS, PTT_WIDTH)
            command = target.astype(current.dtype, copy=False)
            self.shm.set_data(command.reshape(N_DM_SEGMENTS * PTT_WIDTH))
            readback = np.asarray(self.shm.get_data(), dtype=float).reshape(N_DM_SEGMENTS, PTT_WIDTH)
            if not np.allclose(readback, target, atol=1e-6):
                raise RuntimeError("complete dmptt readback does not match the requested state")
            self.logger.record(
                action,
                {"interface": "dmptt fresh full-state write", "shape": [111]},
                targets=target_summary,
            )
        except BaseException as exc:
            self.logger.record(
                action,
                {"interface": "dmptt fresh full-state write", "shape": [111]},
                targets=target_summary,
                outcome="failed",
                error=str(exc),
            )
            raise

    def set_segment(self, segment: int, piston_um: float, tip_mrad: float, tilt_mrad: float) -> None:
        segment = validate_segment(segment)
        target = np.asarray([piston_um, tip_mrad, tilt_mrad], dtype=float)
        if not np.all(np.isfinite(target)):
            raise ValueError(f"segment {segment} PTT command contains non-finite values")
        target_dict = {
            str(segment): {
                "piston_um": float(target[0]),
                "tip_mrad": float(target[1]),
                "tilt_mrad": float(target[2]),
            }
        }
        try:
            current = np.asarray(self.shm.get_data()).reshape(N_DM_SEGMENTS, PTT_WIDTH).copy()
            current[segment] = target.astype(current.dtype, copy=False)
            self.shm.set_data(current.reshape(N_DM_SEGMENTS * PTT_WIDTH))
            readback = np.asarray(self.shm.get_data(), dtype=float).reshape(N_DM_SEGMENTS, PTT_WIDTH)
            if not np.allclose(readback[segment], target, atol=1e-6):
                raise RuntimeError(
                    f"segment {segment} readback {readback[segment].tolist()} does not match "
                    f"command {target.tolist()}"
                )
            self.logger.record(
                "standalone_set_segment_ptt",
                {"interface": "dmptt fresh read-modify-write", "segment": segment},
                targets=target_dict,
            )
        except BaseException as exc:
            self.logger.record(
                "standalone_set_segment_ptt",
                {"interface": "dmptt fresh read-modify-write", "segment": segment},
                targets=target_dict,
                outcome="failed",
                error=str(exc),
            )
            raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instrument-config",
        type=Path,
        default=DEFAULT_INSTRUMENT_CONFIG,
        help="Path to the aligned GUI instrument.json.",
    )
    parser.add_argument(
        "--scan-config",
        type=Path,
        default=DEFAULT_SCAN_CONFIG,
        help="Path to the quadrature scan JSON (defaults beside this script).",
    )
    parser.add_argument(
        "--wavelength-nm",
        type=float,
        default=None,
        help=(
            "Central wavelength in nm. If omitted during a real run, prompt using "
            "central_wavelength_nm from the scan JSON (1500 nm fallback)."
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate configuration and estimate storage without attaching to hardware.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the final interactive confirmation.",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    with source.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a JSON object")
    return value


def prompt_finite_float(label: str, current: Any) -> float:
    """Prompt for one finite float, using the configured value when left blank."""
    try:
        default = float(current)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must have a finite numeric default in the scan config") from exc
    if not np.isfinite(default):
        raise ValueError(f"{label} must have a finite numeric default in the scan config")
    while True:
        entered = input(f"  {label} [{default:g}]: ").strip()
        if not entered:
            return default
        try:
            value = float(entered)
        except ValueError:
            print("    Enter a number, or press Enter to keep the value shown.")
            continue
        if np.isfinite(value):
            return value
        print("    Enter a finite number.")


def validate_wavelength_nm(value: Any) -> float:
    try:
        wavelength_nm = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("central wavelength must be a finite positive number in nm") from exc
    if not np.isfinite(wavelength_nm) or wavelength_nm <= 0:
        raise ValueError("central wavelength must be a finite positive number in nm")
    return wavelength_nm


def quadrature_steps(wavelength_nm: float) -> tuple[QuadratureStep, ...]:
    """Return OPD quadratures and the corresponding reflective-DM commands."""
    wavelength_um = validate_wavelength_nm(wavelength_nm) / 1000.0
    return tuple(
        QuadratureStep(
            index=index,
            phase_deg=90.0 * index,
            opd_um=wavelength_um * index / 4.0,
            surface_piston_um=wavelength_um * index / 8.0,
        )
        for index in range(4)
    )


def prompt_and_update_quadrature_settings(
    scan_path: Path,
    scan_config: dict[str, Any],
    wavelength_override_nm: float | None,
) -> dict[str, Any]:
    """Collect launch-time settings and persist them atomically."""
    rest = scan_config.get("rest_of_dm")
    if not isinstance(rest, dict):
        raise ValueError("rest_of_dm must be an object in the scan config")

    print("Enter quadrature scan settings (press Enter to keep each current value):")
    updated_rest = {
        "piston_um": prompt_finite_float("rest_of_dm.piston_um", rest.get("piston_um")),
        "tip_mrad": prompt_finite_float("rest_of_dm.tip_mrad", rest.get("tip_mrad")),
        "tilt_mrad": prompt_finite_float("rest_of_dm.tilt_mrad", rest.get("tilt_mrad")),
    }
    if wavelength_override_nm is None:
        configured_wavelength = scan_config.get(
            "central_wavelength_nm", DEFAULT_WAVELENGTH_NM
        )
        wavelength_nm = validate_wavelength_nm(
            prompt_finite_float("central_wavelength_nm", configured_wavelength)
        )
    else:
        wavelength_nm = validate_wavelength_nm(wavelength_override_nm)

    updated = dict(scan_config)
    updated["scan_kind"] = "each_segment_tiptilt_quadrature"
    updated["rest_of_dm"] = updated_rest
    updated["central_wavelength_nm"] = wavelength_nm
    updated["output_subdirectory"] = OUTPUT_SUBDIRECTORY
    # A single scanned-segment piston is not applicable: wavelength determines
    # all four absolute quadrature commands.
    updated.pop("scanned_segment", None)
    atomic_json(scan_path, updated)
    print(f"Updated scan config: {scan_path}")
    return updated


def resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def instrument_root(config_path: Path) -> Path:
    path = config_path.expanduser().resolve()
    return path.parent.parent if path.parent.name == "config" else path.parent


def resolve_detector_boxes_path(instrument: dict[str, Any], instrument_path: Path) -> Path:
    root = instrument_root(instrument_path)
    configured = instrument.get("calibration_assets", {}).get("detector_boxes")
    if configured:
        path = resolve_path(configured, root)
    else:
        # Current refactor versions keep detector_boxes.json beside instrument.json.
        path = instrument_path.expanduser().resolve().parent / "detector_boxes.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Detector-box configuration not found: {path}. Add "
            "calibration_assets.detector_boxes to instrument.json or place "
            "detector_boxes.json beside it."
        )
    return path


def resolve_output_root(instrument: dict[str, Any], instrument_path: Path) -> Path:
    configured = instrument.get("output_root")
    if not configured:
        raise ValueError("instrument.json output_root is required")
    return resolve_path(configured, instrument_root(instrument_path))


def resolve_apapane_dark(instrument: dict[str, Any], instrument_path: Path) -> Path:
    dark_root = resolve_output_root(instrument, instrument_path) / "darks"
    for name in ("apapane_dark.fits", "apapane_dark.npy"):
        candidate = dark_root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No saved APAPANE dark found in {dark_root}; acquire GUI workflow step 2 first."
    )


def load_dark(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        dark = np.load(path, allow_pickle=False)
    else:
        try:
            from astropy.io import fits
        except ImportError as exc:
            raise RuntimeError("astropy is required to load the configured APAPANE dark") from exc
        dark = fits.getdata(path)
    dark = np.asarray(dark, dtype=float)
    if dark.ndim == 3:
        dark = np.median(dark, axis=0)
    if dark.ndim != 2 or not np.all(np.isfinite(dark)):
        raise ValueError(f"APAPANE dark must be one finite 2-D image, got {dark.shape}")
    return dark


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_grid(start: float, stop: float, step: float, label: str) -> np.ndarray:
    start, stop, step = float(start), float(stop), float(step)
    if not np.all(np.isfinite([start, stop, step])) or step <= 0 or stop < start:
        raise ValueError(f"{label} requires finite start <= stop and step > 0")
    intervals = (stop - start) / step
    rounded = int(round(intervals))
    if not np.isclose(intervals, rounded, rtol=0.0, atol=1e-9):
        raise ValueError(f"{label} range {stop - start} is not an integer multiple of step {step}")
    return start + np.arange(rounded + 1, dtype=float) * step


def load_detector_boxes(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("convention") != "[y0, y1, x0, x1]":
        raise ValueError("detector_boxes.json must use convention [y0, y1, x0, x1]")
    for section in ("photometry", "null", "fringe_tracking"):
        if not isinstance(payload.get(section), dict):
            raise ValueError(f"detector_boxes.json requires object {section!r}")
    return payload


def validate_bounds(value: Any, identifier: str) -> tuple[int, int, int, int]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{identifier} must be [y0, y1, x0, x1]")
    y0, y1, x0, x1 = map(int, value)
    if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
        raise ValueError(f"{identifier} has invalid bounds {value}")
    return y0, y1, x0, x1


def selected_box_specs(detector: dict[str, Any], selection: dict[str, Any]) -> list[BoxSpec]:
    specs: list[BoxSpec] = []
    for section in ("photometry", "null"):
        names = selection.get(section, [])
        if not isinstance(names, list):
            raise ValueError(f"saved_boxes.{section} must be a list")
        for name in names:
            if name not in detector[section]:
                raise ValueError(f"Unknown detector box {section}/{name}")
            specs.append(
                BoxSpec(
                    identifier=f"{section}/{name}",
                    section=section,
                    name=str(name),
                    channel=None,
                    bounds=validate_bounds(detector[section][name], f"{section}/{name}"),
                )
            )
    fringe_selection = selection.get("fringe_tracking", {})
    if not isinstance(fringe_selection, dict):
        raise ValueError("saved_boxes.fringe_tracking must be an object")
    for baseline, channels in fringe_selection.items():
        if baseline not in detector["fringe_tracking"]:
            raise ValueError(f"Unknown fringe-tracking baseline {baseline!r}")
        if not isinstance(channels, list):
            raise ValueError(f"saved_boxes.fringe_tracking.{baseline} must be a list")
        for channel in channels:
            if channel not in detector["fringe_tracking"][baseline]:
                raise ValueError(f"Unknown fringe-tracking box {baseline}/{channel}")
            specs.append(
                BoxSpec(
                    identifier=f"fringe_tracking/{baseline}/{channel}",
                    section="fringe_tracking",
                    name=str(baseline),
                    channel=str(channel),
                    bounds=validate_bounds(
                        detector["fringe_tracking"][baseline][channel],
                        f"fringe_tracking/{baseline}/{channel}",
                    ),
                )
            )
    identifiers = [spec.identifier for spec in specs]
    if not specs:
        raise ValueError("saved_boxes must select at least one detector box")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("saved_boxes contains duplicate entries")
    return specs


def validate_boxes_against_dark(specs: Iterable[BoxSpec], dark: np.ndarray) -> None:
    for spec in specs:
        y0, y1, x0, x1 = spec.bounds
        if y1 > dark.shape[0] or x1 > dark.shape[1]:
            raise ValueError(f"{spec.identifier} bounds {spec.bounds} exceed dark shape {dark.shape}")


def validate_segment(segment: int) -> int:
    segment = int(segment)
    if segment not in range(N_DM_SEGMENTS):
        raise ValueError(f"DM segment must be in [0, 36], got {segment}")
    return segment


def validate_dm_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=float).reshape(N_DM_SEGMENTS, PTT_WIDTH)
    if not np.all(np.isfinite(state)):
        raise ValueError("DM state contains non-finite values")
    return state


def ptt_dict(values: Iterable[float]) -> dict[str, float]:
    piston, tip, tilt = map(float, values)
    return {"piston_um": piston, "tip_mrad": tip, "tilt_mrad": tilt}


def signed_value(value: float) -> str:
    return f"{float(value):+0.3f}"


def fixed_run_prefix(now: datetime, rest_ptt: np.ndarray, scan_label: str) -> str:
    return (
        f"{now:%Y%m%d}_{now:%H%M%S}"
        f"_tip{signed_value(rest_ptt[1])}"
        f"_tilt{signed_value(rest_ptt[2])}"
        f"_piston{signed_value(rest_ptt[0])}"
        f"_{scan_label}"
    )


def unique_directory(parent: Path, name: str) -> Path:
    candidate = parent / name
    suffix = 2
    while candidate.exists():
        candidate = parent / f"{name}_{suffix:02d}"
        suffix += 1
    return candidate


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def frame_cube(values: Any, nframes: int, frame_shape: tuple[int, int]) -> np.ndarray:
    frames = np.asarray(values)
    if frames.ndim == 2:
        if nframes != 1:
            raise RuntimeError(
                f"APAPANE multi_recv_data({nframes}) returned one averaged 2-D image; "
                "individual-frame box storage requires a 3-D frame cube"
            )
        frames = frames[None, :, :]
    elif frames.ndim == 3 and frames.shape[0] != nframes and frames.shape[-1] == nframes:
        frames = np.moveaxis(frames, -1, 0)
    if frames.ndim != 3 or frames.shape[0] != nframes or tuple(frames.shape[1:]) != tuple(frame_shape):
        raise RuntimeError(
            f"Expected APAPANE cube ({nframes}, {frame_shape[0]}, {frame_shape[1]}), got {frames.shape}"
        )
    return frames


def acquire_box_crops(
    apapane: Any,
    dark: np.ndarray,
    nframes: int,
    specs: list[BoxSpec],
) -> dict[str, np.ndarray]:
    frames = frame_cube(apapane.multi_recv_data(int(nframes)), nframes, dark.shape)
    corrected = np.asarray(frames, dtype=np.float32) - np.asarray(dark, dtype=np.float32)[None, :, :]
    crops: dict[str, np.ndarray] = {}
    for spec in specs:
        y0, y1, x0, x1 = spec.bounds
        crops[spec.identifier] = corrected[:, y0:y1, x0:x1].copy()
    return crops


def allocate_raster_arrays(
    specs: list[BoxSpec],
    ntip: int,
    ntilt: int,
    nframes: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    crops = {
        spec.identifier: np.empty((ntip, ntilt, nframes, spec.height, spec.width), dtype=np.float32)
        for spec in specs
    }
    fluxes = {
        spec.identifier: np.empty((ntip, ntilt, nframes), dtype=np.float64)
        for spec in specs
    }
    return crops, fluxes


def grid_indices(ntip: int, ntilt: int, serpentine: bool) -> Iterable[tuple[int, int]]:
    for tip_index in range(ntip):
        tilt_indices = range(ntilt) if not serpentine or tip_index % 2 == 0 else range(ntilt - 1, -1, -1)
        for tilt_index in tilt_indices:
            yield tip_index, tilt_index


def scan_quadrature_raster(
    *,
    apapane: Any,
    dm: DMController,
    segment: int,
    phase_steps: tuple[QuadratureStep, ...],
    tips: np.ndarray,
    tilts: np.ndarray,
    dark: np.ndarray,
    nframes: int,
    specs: list[BoxSpec],
    settle_s: float,
    serpentine: bool,
) -> tuple[
    list[dict[str, np.ndarray]],
    list[dict[str, np.ndarray]],
]:
    phase_crops: list[dict[str, np.ndarray]] = []
    phase_fluxes: list[dict[str, np.ndarray]] = []
    for _ in phase_steps:
        crops, fluxes = allocate_raster_arrays(specs, len(tips), len(tilts), nframes)
        phase_crops.append(crops)
        phase_fluxes.append(fluxes)
    total = len(tips) * len(tilts)
    report_every = max(1, total // 20)
    for completed, (tip_index, tilt_index) in enumerate(
        grid_indices(len(tips), len(tilts), serpentine), start=1
    ):
        # Interleave all four phases at each tip/tilt point. This minimises
        # temporal-drift contamination in the later quadrature differences.
        for phase_index, phase_step in enumerate(phase_steps):
            dm.set_segment(
                segment,
                phase_step.surface_piston_um,
                tips[tip_index],
                tilts[tilt_index],
            )
            time.sleep(settle_s)
            crops = acquire_box_crops(apapane, dark, nframes, specs)
            for spec in specs:
                crop = crops[spec.identifier]
                phase_crops[phase_index][spec.identifier][tip_index, tilt_index] = crop
                phase_fluxes[phase_index][spec.identifier][tip_index, tilt_index] = crop.sum(
                    axis=(1, 2), dtype=np.float64
                )
        if completed == 1 or completed == total or completed % report_every == 0:
            print(
                f"  segment {segment:02d}: {completed:4d}/{total} tip/tilt points "
                f"x {len(phase_steps)} phases"
            )
    return phase_crops, phase_fluxes


def save_raster_fits(
    path: Path,
    *,
    scan_type: str,
    segment: int,
    repeat_index: int,
    nframes: int,
    tips: np.ndarray,
    tilts: np.ndarray,
    rest_ptt: np.ndarray,
    scanned_piston_um: float,
    initial_dm: np.ndarray,
    specs: list[BoxSpec],
    crops: dict[str, np.ndarray],
    fluxes: dict[str, np.ndarray],
    dark_path: Path,
    detector_boxes_path: Path,
    primary_metadata: dict[str, Any] | None = None,
    extra_comments: Iterable[str] = (),
) -> None:
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise RuntimeError("astropy is required to save the standalone FITS products") from exc

    primary = fits.PrimaryHDU()
    primary.header["SCANTYPE"] = scan_type
    primary.header["DATE-OBS"] = datetime.now(timezone.utc).isoformat()
    primary.header["SEGMENT"] = int(segment)
    primary.header["REPEAT"] = int(repeat_index)
    primary.header["NFRAMES"] = int(nframes)
    primary.header["RESTPST"] = float(rest_ptt[0])
    primary.header["RESTTIP"] = float(rest_ptt[1])
    primary.header["RESTTILT"] = float(rest_ptt[2])
    primary.header["SCANPST"] = float(scanned_piston_um)
    primary.header.add_comment("All crop cubes are dark-subtracted and preserve every acquired frame.")
    primary.header.add_comment(f"APAPANE dark: {dark_path}")
    primary.header.add_comment(f"Detector boxes: {detector_boxes_path}")
    for key, value in (primary_metadata or {}).items():
        primary.header[str(key).upper()[:8]] = value
    for comment in extra_comments:
        primary.header.add_comment(str(comment))
    hdus: list[Any] = [
        primary,
        fits.ImageHDU(np.asarray(tips, dtype=np.float64), name="TIP_AXIS"),
        fits.ImageHDU(np.asarray(tilts, dtype=np.float64), name="TILT_AXIS"),
        fits.ImageHDU(np.asarray(initial_dm, dtype=np.float32), name="INITIAL_DM"),
    ]
    for index, spec in enumerate(specs, start=1):
        crop_hdu = fits.ImageHDU(np.asarray(crops[spec.identifier], dtype=np.float32), name=f"BOX{index:03d}")
        crop_hdu.header["BOXNAME"] = spec.identifier
        crop_hdu.header["Y0"], crop_hdu.header["Y1"], crop_hdu.header["X0"], crop_hdu.header["X1"] = spec.bounds
        crop_hdu.header["DIMS"] = "tip,tilt,frame,y,x"
        flux_hdu = fits.ImageHDU(np.asarray(fluxes[spec.identifier], dtype=np.float64), name=f"FLUX{index:03d}")
        flux_hdu.header["BOXNAME"] = spec.identifier
        flux_hdu.header["DIMS"] = "tip,tilt,frame"
        mean_hdu = fits.ImageHDU(
            np.asarray(fluxes[spec.identifier].mean(axis=2), dtype=np.float64),
            name=f"MEAN{index:03d}",
        )
        mean_hdu.header["BOXNAME"] = spec.identifier
        mean_hdu.header["DIMS"] = "tip,tilt"
        hdus.extend((crop_hdu, flux_hdu, mean_hdu))

    temporary = path.with_name(path.name + ".partial")
    try:
        fits.HDUList(hdus).writeto(temporary, overwrite=True, checksum=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_scan_config(scan: dict[str, Any], expected_kind: str) -> dict[str, Any]:
    allowed_kinds = {expected_kind}
    if expected_kind == "each_segment_tiptilt_quadrature":
        # Accept a copied legacy config on first launch; the interactive update
        # rewrites it with the quadrature scan kind before hardware attachment.
        allowed_kinds.add("each_segment_tiptilt")
    if scan.get("scan_kind") not in allowed_kinds:
        raise ValueError(f"scan_kind must be one of {sorted(allowed_kinds)!r}")
    nframes = int(scan.get("nframes", 0))
    repeats = int(scan.get("repeats", 0))
    settle_s = float(scan.get("dm_settle_s", -1))
    initial_settle_s = float(scan.get("initial_settle_s", -1))
    if nframes <= 0 or repeats <= 0:
        raise ValueError("nframes and repeats must be positive integers")
    if settle_s < 0 or initial_settle_s < 0:
        raise ValueError("settling times must be non-negative")
    grid = scan.get("grid", {})
    tips = make_grid(grid.get("tip", {}).get("start"), grid.get("tip", {}).get("stop"), grid.get("tip", {}).get("step"), "tip grid")
    tilts = make_grid(grid.get("tilt", {}).get("start"), grid.get("tilt", {}).get("stop"), grid.get("tilt", {}).get("step"), "tilt grid")
    if expected_kind in {"each_segment_tiptilt", "each_segment_tiptilt_quadrature"}:
        rest_cfg = scan.get("rest_of_dm", {})
        rest_ptt = np.asarray(
            [rest_cfg.get("piston_um"), rest_cfg.get("tip_mrad"), rest_cfg.get("tilt_mrad")],
            dtype=float,
        )
        if rest_ptt.shape != (3,) or not np.all(np.isfinite(rest_ptt)):
            raise ValueError("rest_of_dm requires finite piston_um, tip_mrad and tilt_mrad")
        scan["_rest_ptt"] = rest_ptt
    elif expected_kind == "dm_wide_tiptilt":
        rastered_piston = float(scan.get("rastered_segments", {}).get("piston_um"))
        if not np.isfinite(rastered_piston):
            raise ValueError("rastered_segments.piston_um must be finite")
        scan["_rastered_piston"] = rastered_piston
    else:
        raise ValueError(f"Unsupported standalone scan kind {expected_kind!r}")
    output_subdirectory = scan.get("output_subdirectory")
    if not isinstance(output_subdirectory, str) or not output_subdirectory.strip():
        raise ValueError("output_subdirectory must be a non-empty relative path")
    subdir = Path(output_subdirectory)
    if subdir.is_absolute() or ".." in subdir.parts:
        raise ValueError("output_subdirectory must stay below instrument.json output_root")
    scan["_tips"] = tips
    scan["_tilts"] = tilts
    scan["_nframes"] = nframes
    scan["_repeats"] = repeats
    scan["_settle_s"] = settle_s
    scan["_initial_settle_s"] = initial_settle_s
    return scan


def estimated_crop_bytes(
    specs: list[BoxSpec],
    npoints: int,
    nframes: int,
    rasters: int,
    repeats: int,
) -> int:
    pixels = sum(spec.height * spec.width for spec in specs)
    crop_bytes = rasters * repeats * npoints * nframes * pixels * np.dtype(np.float32).itemsize
    flux_bytes = rasters * repeats * npoints * nframes * len(specs) * np.dtype(np.float64).itemsize
    return int(crop_bytes + flux_bytes)


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TiB"


def public_scan_config(scan: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in scan.items() if not key.startswith("_")}


def print_preflight(
    *,
    instrument_path: Path,
    scan_path: Path,
    detector_path: Path,
    dark_path: Path,
    output_parent: Path,
    scan: dict[str, Any],
    specs: list[BoxSpec],
    estimate: int,
    rasters: int,
) -> None:
    print("GLINT standalone tip/tilt preflight")
    print(f"  instrument config : {instrument_path}")
    print(f"  scan config       : {scan_path}")
    print(f"  detector boxes    : {detector_path}")
    print(f"  APAPANE dark      : {dark_path}")
    print(f"  output parent     : {output_parent}")
    print(f"  raster count      : {rasters} x {scan['_repeats']} repeat(s)")
    print(f"  tip points        : {len(scan['_tips'])}")
    print(f"  tilt points       : {len(scan['_tilts'])}")
    print(f"  frames / point    : {scan['_nframes']}")
    print(f"  saved boxes       : {[spec.identifier for spec in specs]}")
    print(f"  estimated payload : {human_bytes(estimate)} before FITS overhead")
    print("  final DM state    : mandatory restoration of the launch-time live snapshot")


def attach_devices(instrument: dict[str, Any]) -> tuple[Any, Any]:
    try:
        from pyMilk.interfacing.shm import SHM
    except ImportError as exc:
        raise RuntimeError("pyMilk is required on the telescope to attach to APAPANE and dmptt") from exc
    shm_config = instrument.get("instrument", {}).get("shm", {})
    dm_config = instrument.get("instrument", {}).get("dm", {})
    apapane_name = shm_config.get("apapane")
    dmptt_name = dm_config.get("ptt_shm")
    if not apapane_name or not dmptt_name:
        raise ValueError("instrument.json must define instrument.shm.apapane and instrument.dm.ptt_shm")
    return SHM(str(apapane_name)), SHM(str(dmptt_name))


def run(scan_args: argparse.Namespace) -> Path | None:
    instrument_path = scan_args.instrument_config.expanduser().resolve()
    scan_path = scan_args.scan_config.expanduser().resolve()
    instrument = load_json(instrument_path)
    scan_config = load_json(scan_path)
    scan_config["output_subdirectory"] = OUTPUT_SUBDIRECTORY
    scan_config.pop("scanned_segment", None)
    if scan_args.check_only:
        wavelength_nm = validate_wavelength_nm(
            scan_args.wavelength_nm
            if scan_args.wavelength_nm is not None
            else scan_config.get("central_wavelength_nm", DEFAULT_WAVELENGTH_NM)
        )
    else:
        scan_config = prompt_and_update_quadrature_settings(
            scan_path,
            scan_config,
            scan_args.wavelength_nm,
        )
        wavelength_nm = validate_wavelength_nm(scan_config["central_wavelength_nm"])
    scan = validate_scan_config(scan_config, "each_segment_tiptilt_quadrature")
    phase_steps = quadrature_steps(wavelength_nm)
    detector_path = resolve_detector_boxes_path(instrument, instrument_path)
    detector = load_detector_boxes(detector_path)
    specs = selected_box_specs(detector, scan.get("saved_boxes", {}))
    dark_path = resolve_apapane_dark(instrument, instrument_path)
    dark = load_dark(dark_path)
    validate_boxes_against_dark(specs, dark)
    output_parent = resolve_output_root(instrument, instrument_path) / scan["output_subdirectory"]
    npoints = len(scan["_tips"]) * len(scan["_tilts"])
    estimate = estimated_crop_bytes(
        specs,
        npoints,
        scan["_nframes"],
        N_DM_SEGMENTS * len(phase_steps),
        scan["_repeats"],
    )
    print_preflight(
        instrument_path=instrument_path,
        scan_path=scan_path,
        detector_path=detector_path,
        dark_path=dark_path,
        output_parent=output_parent,
        scan=scan,
        specs=specs,
        estimate=estimate,
        rasters=N_DM_SEGMENTS * len(phase_steps),
    )
    print(f"  central wavelength: {wavelength_nm:g} nm")
    print("  quadrature steps  :")
    for phase_step in phase_steps:
        print(
            f"    {phase_step.phase_deg:5.0f} deg: "
            f"OPD={phase_step.opd_um:.6f} um, "
            f"DM surface piston={phase_step.surface_piston_um:.6f} um"
        )
    if scan_args.check_only:
        return None
    output_parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_parent).free
    reserve = float(scan.get("minimum_free_space_gb", 1.0)) * 1024**3
    if estimate + reserve > free_bytes:
        raise RuntimeError(
            f"Insufficient free space: need approximately {human_bytes(estimate + int(reserve))}, "
            f"but {human_bytes(free_bytes)} is available"
        )
    if not scan_args.yes:
        answer = input("Proceed with physical quadrature DM commands? Type YES: ").strip()
        if answer != "YES":
            print("Cancelled before hardware attachment.")
            return None

    now = datetime.now().astimezone()
    prefix = fixed_run_prefix(now, scan["_rest_ptt"], "each_segment_quadrature")
    run_directory = unique_directory(output_parent / f"{now:%Y%m%d}", prefix)
    run_directory.mkdir(parents=True)
    summary_path = run_directory / f"{run_directory.name}_summary.json"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "starting",
        "scan_type": "each_segment_tiptilt_quadrature",
        "started_local": now.isoformat(),
        "instrument_config": str(instrument_path),
        "scan_config": str(scan_path),
        "scan_settings": public_scan_config(scan),
        "central_wavelength_nm": wavelength_nm,
        "quadrature_convention": {
            "requested_phase_opd": "[0, lambda/4, lambda/2, 3*lambda/4]",
            "dm_command": "reflective surface displacement = OPD/2",
            "double_pass_reflection": True,
        },
        "quadrature_steps": [phase_step.as_dict() for phase_step in phase_steps],
        "detector_boxes_file": str(detector_path),
        "saved_boxes": [spec.as_dict() for spec in specs],
        "apapane_dark_file": str(dark_path),
        "apapane_dark_sha256": sha256_file(dark_path),
        "estimated_payload_bytes": estimate,
        "outputs": [],
    }
    atomic_json(summary_path, summary)

    logger = PersistentDMLogger(instrument, instrument_root(instrument_path))
    apapane = None
    dm: DMController | None = None
    initial_dm: np.ndarray | None = None
    failure: BaseException | None = None
    restoration_error: BaseException | None = None
    try:
        apapane_shm, dmptt_shm = attach_devices(instrument)
        apapane = apapane_shm
        dm = DMController(dmptt_shm, logger)
        initial_dm = dm.snapshot()
        summary["initial_dm_ptt"] = initial_dm.tolist()
        summary["status"] = "running"
        atomic_json(summary_path, summary)
        rest_state = np.tile(scan["_rest_ptt"], (N_DM_SEGMENTS, 1))
        serpentine = bool(scan.get("serpentine", True))

        for repeat_index in range(1, scan["_repeats"] + 1):
            for segment in range(N_DM_SEGMENTS):
                print(f"\nRepeat {repeat_index}/{scan['_repeats']}; segment {segment}/36")
                dm.set_full(
                    rest_state,
                    action="standalone_park_all_segments",
                    target_summary={"all_segments": ptt_dict(scan["_rest_ptt"])},
                )
                time.sleep(scan["_initial_settle_s"])
                phase_crops, phase_fluxes = scan_quadrature_raster(
                    apapane=apapane,
                    dm=dm,
                    segment=segment,
                    phase_steps=phase_steps,
                    tips=scan["_tips"],
                    tilts=scan["_tilts"],
                    dark=dark,
                    nframes=scan["_nframes"],
                    specs=specs,
                    settle_s=scan["_settle_s"],
                    serpentine=serpentine,
                )
                for phase_index, phase_step in enumerate(phase_steps):
                    filename = (
                        f"{run_directory.name}_segment{segment:02d}"
                        f"_repeat{repeat_index:02d}_phase{int(phase_step.phase_deg):03d}.fits"
                    )
                    file_path = run_directory / filename
                    save_raster_fits(
                        file_path,
                        scan_type="EACHQUAD",
                        segment=segment,
                        repeat_index=repeat_index,
                        nframes=scan["_nframes"],
                        tips=scan["_tips"],
                        tilts=scan["_tilts"],
                        rest_ptt=scan["_rest_ptt"],
                        scanned_piston_um=phase_step.surface_piston_um,
                        initial_dm=initial_dm,
                        specs=specs,
                        crops=phase_crops[phase_index],
                        fluxes=phase_fluxes[phase_index],
                        dark_path=dark_path,
                        detector_boxes_path=detector_path,
                        primary_metadata={
                            "WAVELEN": float(wavelength_nm),
                            "PHASEDEG": float(phase_step.phase_deg),
                            "OPDUM": float(phase_step.opd_um),
                            "SURFPST": float(phase_step.surface_piston_um),
                            "DBLPASS": True,
                        },
                        extra_comments=(
                            "SCANPST/SURFPST are physical reflective-DM surface displacement.",
                            "OPDUM is twice the surface displacement and sets optical phase.",
                            "The four phases were interleaved at every tip/tilt point.",
                        ),
                    )
                    summary["outputs"].append(
                        {
                            "segment": segment,
                            "repeat": repeat_index,
                            "phase_index": phase_step.index,
                            "phase_deg": phase_step.phase_deg,
                            "optical_path_difference_um": phase_step.opd_um,
                            "dm_surface_piston_um": phase_step.surface_piston_um,
                            "fits": filename,
                        }
                    )
                    atomic_json(summary_path, summary)
                    print(f"  saved {file_path}")
    except BaseException as exc:
        failure = exc
        summary["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if dm is not None and initial_dm is not None:
            try:
                dm.set_full(
                    initial_dm,
                    action="standalone_restore_launch_dm",
                    target_summary={"source": "launch-time live dmptt snapshot", "segments": 37},
                )
                summary["dm_restored"] = True
            except BaseException as exc:
                restoration_error = exc
                summary["dm_restored"] = False
                summary["restoration_error"] = f"{type(exc).__name__}: {exc}"
        summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
        if failure is None and restoration_error is None:
            summary["status"] = "completed"
        atomic_json(summary_path, summary)
        logger.close()

    if restoration_error is not None:
        raise RuntimeError(f"CRITICAL: launch-time DM restoration failed: {restoration_error}") from restoration_error
    if failure is not None:
        raise failure
    print(f"\nCompleted. Launch-time DM state restored. Outputs: {run_directory}")
    return run_directory


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted; restoration was attempted before exit.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
