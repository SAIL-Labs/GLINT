from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .models import AxisGrid, SegmentPTT


PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None, *, simulation: bool = False) -> dict[str, Any]:
    name = "simulator.json" if simulation else "instrument.json"
    config_path = Path(path).expanduser().resolve() if path else PACKAGE_ROOT / "config" / name
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    config["_config_path"] = str(config_path)
    config["_root"] = str(PACKAGE_ROOT)
    return config


def save_config(config: dict[str, Any]) -> Path:
    """Atomically persist the active configuration without runtime-only keys."""
    try:
        config_path = Path(config["_config_path"]).expanduser().resolve()
    except (KeyError, TypeError) as error:
        raise ValueError("The active configuration has no writable _config_path") from error
    payload = {key: deepcopy(value) for key, value in config.items() if not str(key).startswith("_")}
    config_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{config_path.name}.",
        suffix=".tmp",
        dir=config_path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, config_path)
    except Exception:
        try:
            Path(temporary_name).unlink(missing_ok=True)
        finally:
            raise
    return config_path


def validate_scan_defaults(config: dict[str, Any], mount_limits: dict[str, tuple[float, float]] | None = None) -> None:
    """Validate persisted scan defaults before the GUI replaces the active set."""
    scans = config.get("scans")
    if not isinstance(scans, dict):
        raise ValueError("scans must be a configuration object")
    required = {"focal_dark", "apapane_dark", "bulk", "set_z", "pitch_yaw", "xy", "tip_tilt", "null"}
    missing = required.difference(scans)
    if missing:
        raise ValueError(f"Missing scan defaults: {', '.join(sorted(missing))}")

    for kind in required:
        for key in ("nframes", "numavg", "max_iterations"):
            if key in scans[kind] and int(scans[kind][key]) < 1:
                raise ValueError(f"{kind}.{key} must be at least 1")
        if "settle_s" in scans[kind]:
            settle_s = float(scans[kind]["settle_s"])
            if not np.isfinite(settle_s) or settle_s < 0:
                raise ValueError(f"{kind}.settle_s must be a finite non-negative value")

    bulk = scans["bulk"]
    crop = bulk.get("crop")
    if not isinstance(crop, list) or len(crop) != 4:
        raise ValueError("bulk.crop must be a JSON list [y0, y1, x0, x1]")
    y0, y1, x0, x1 = map(int, crop)
    if min(y0, x0) < 0 or y1 <= y0 or x1 <= x0:
        raise ValueError("bulk.crop must contain increasing non-negative detector bounds")
    for key in ("centroid_threshold",):
        value = float(bulk[key])
        if not 0 < value <= 1:
            raise ValueError(f"bulk.{key} must be in (0, 1]")
    for key in ("tolerance_px", "trust_region_px", "max_condition"):
        if not np.isfinite(float(bulk[key])) or float(bulk[key]) <= 0:
            raise ValueError(f"bulk.{key} must be a positive finite value")
    if int(bulk.get("max_backtracks", 0)) < 0:
        raise ValueError("bulk.max_backtracks must be zero or greater")
    perturbation = np.asarray(bulk.get("response_perturbation"), dtype=float)
    if perturbation.shape != (2,) or not np.all(np.isfinite(perturbation)) or np.any(perturbation == 0):
        raise ValueError("bulk.response_perturbation must contain two finite non-zero values")
    grid_axes = {
        "pitch_yaw": ("pitch", "yaw"),
        "xy": ("x", "y"),
        "tip_tilt": ("tip", "tilt"),
        "null": ("piston",),
    }
    limits = mount_limits or {
        axis: tuple(map(float, bounds))
        for axis, bounds in config.get("mount_control", {}).get("limits", {}).items()
    }
    for kind, axes in grid_axes.items():
        for axis in axes:
            try:
                values = AxisGrid(**scans[kind][axis]).values()
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid {kind}.{axis} grid: {error}") from error
            if kind in {"pitch_yaw", "xy"}:
                if axis not in limits:
                    raise ValueError(f"No configured mount limits are available for {axis}")
                lower, upper = limits[axis]
                if float(values[0]) < float(lower) or float(values[-1]) > float(upper):
                    raise ValueError(
                        f"{kind}.{axis} runs from {values[0]:g} to {values[-1]:g}, outside [{lower:g}, {upper:g}]"
                    )

    z = float(scans["set_z"]["z"])
    if "z" in limits and not float(limits["z"][0]) <= z <= float(limits["z"][1]):
        raise ValueError(f"set_z.z {z:g} is outside [{limits['z'][0]:g}, {limits['z'][1]:g}]")
    if float(scans["pitch_yaw"].get("agreement_pulses", 0)) <= 0:
        raise ValueError("pitch_yaw.agreement_pulses must be positive")
    if float(scans["xy"].get("agreement_pulses", 0)) <= 0:
        raise ValueError("xy.agreement_pulses must be positive")
    com_threshold = float(scans["xy"].get("com_threshold", 0))
    if not 0 < com_threshold <= 1:
        raise ValueError("xy.com_threshold must be in (0, 1]")
    null = scans["null"]
    period_bounds = np.asarray(null.get("period_bounds_um", (1.2, 2.0)), dtype=float)
    if period_bounds.shape != (2,) or not np.all(np.isfinite(period_bounds)) or not 0 < period_bounds[0] < period_bounds[1]:
        raise ValueError("null.period_bounds_um must contain two increasing positive OPD periods")
    min_fit_r2 = float(null.get("min_fit_r2", 0.35))
    if not np.isfinite(min_fit_r2) or not 0 <= min_fit_r2 <= 1:
        raise ValueError("null.min_fit_r2 must lie in [0, 1]")
    max_invalid_fraction = float(null.get("max_invalid_fraction", 0.05))
    if not np.isfinite(max_invalid_fraction) or not 0 <= max_invalid_fraction < 1:
        raise ValueError("null.max_invalid_fraction must lie in [0, 1)")


def segment_state(config: dict[str, Any], name: str) -> dict[int, SegmentPTT]:
    source = config["dm_states"][name]
    default = source.get("default", {})
    overrides = source.get("segments", {})
    state: dict[int, SegmentPTT] = {}
    for segment in range(37):
        values = deepcopy(default)
        values.update(overrides.get(str(segment), {}))
        state[segment] = SegmentPTT(
            piston_um=float(values.get("piston_um", 0.0)),
            tip_mrad=float(values.get("tip_mrad", 0.0)),
            tilt_mrad=float(values.get("tilt_mrad", 0.0)),
        )
    return state


def rest_dm_maps(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the configured named maps available for non-injected segments."""
    maps = config.get("rest_of_dm", {}).get("maps", {})
    if not isinstance(maps, dict) or not maps:
        raise ValueError("rest_of_dm.maps must contain at least one named DM map")
    return maps


def rest_dm_state(
    config: dict[str, Any],
    name: str,
    *,
    uniform_piston_um: float | None = None,
    uniform_tip_mrad: float | None = None,
    uniform_tilt_mrad: float | None = None,
) -> dict[int, SegmentPTT]:
    """Build one complete named rest-of-DM state.

    A map normally references a calibrated entry in ``dm_states``. Maps marked
    ``editable_uniform_ptt`` additionally support session-only uniform piston,
    tip and tilt overrides. The legacy ``editable_uniform_tip_tilt`` flag is
    still accepted and preserves the map's configured per-segment pistons.
    """
    maps = rest_dm_maps(config)
    if name not in maps:
        raise ValueError(f"Unknown rest-of-DM map {name!r}")
    definition = maps[name]
    state_name = definition.get("dm_state")
    if state_name not in config.get("dm_states", {}):
        raise ValueError(f"Rest-of-DM map {name!r} references missing dm_states entry {state_name!r}")
    state = segment_state(config, str(state_name))
    editable_ptt = bool(definition.get("editable_uniform_ptt", False))
    editable_tip_tilt = editable_ptt or bool(definition.get("editable_uniform_tip_tilt", False))
    if editable_ptt:
        piston = float(
            definition.get("default_piston_um", 0.0)
            if uniform_piston_um is None
            else uniform_piston_um
        )
        for ptt in state.values():
            ptt.piston_um = piston
    if editable_tip_tilt:
        tip = float(definition.get("default_tip_mrad", 0.0) if uniform_tip_mrad is None else uniform_tip_mrad)
        tilt = float(definition.get("default_tilt_mrad", 0.0) if uniform_tilt_mrad is None else uniform_tilt_mrad)
        for ptt in state.values():
            ptt.tip_mrad = tip
            ptt.tilt_mrad = tilt
    return state


def validate_config(config: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for key in ("output_root", "camera_frame_root"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            warnings.append(f"{key} must be a non-empty path in the active JSON configuration.")
    command_root = config.get("command_logging", {}).get("root")
    if config.get("command_logging", {}).get("enabled", True) and (
        not isinstance(command_root, str) or not command_root.strip()
    ):
        warnings.append("command_logging.root must be a non-empty path in the active JSON configuration.")
    config_directory = (Path(config.get("_root", PACKAGE_ROOT)) / "config").resolve()
    response_path = config.get("calibration_assets", {}).get("bulk_response_matrix")
    if not response_path:
        warnings.append("calibration_assets.bulk_response_matrix is missing; bulk alignment is unavailable.")
    else:
        resolved_response = Path(str(response_path)).expanduser()
        if not resolved_response.is_absolute():
            resolved_response = Path(config.get("_root", PACKAGE_ROOT)) / resolved_response
        resolved_response = resolved_response.resolve()
        try:
            resolved_response.relative_to(config_directory)
        except ValueError:
            warnings.append(
                f"The bulk response matrix must be stored inside the configuration folder: {config_directory}"
            )
        else:
            if not resolved_response.is_file():
                warnings.append(f"The configured bulk response matrix does not exist: {resolved_response}")
    logo_path = config.get("gui", {}).get("brand_logo")
    if not logo_path:
        warnings.append("gui.brand_logo is missing; the GLINT GUI cannot display its configured logo.")
    else:
        resolved_logo = Path(str(logo_path)).expanduser()
        if not resolved_logo.is_absolute():
            resolved_logo = Path(config.get("_root", PACKAGE_ROOT)) / resolved_logo
        if not resolved_logo.is_file():
            warnings.append(f"The configured GUI logo does not exist: {resolved_logo}")
    away = config["dm_states"]["tilted_away"]
    if not away.get("validated", False):
        warnings.append("Tilted-away DM values are marked unvalidated; verify all 37 segment values on the instrument.")
    flat_volts = config.get("dm_voltage_states", {}).get("flat", {})
    if config.get("mode") == "instrument":
        instrument = config.get("instrument", {})
        dm_streams = instrument.get("dm", {})
        for key in ("ptt_shm", "voltage_shm"):
            if not isinstance(dm_streams.get(key), str) or not dm_streams[key].strip():
                warnings.append(f"instrument.dm.{key} must name an existing SHM stream.")
        mount = instrument.get("mount", {})
        if not isinstance(mount.get("port"), str) or not mount["port"].strip():
            warnings.append("instrument.mount.port must name the Suruga serial device.")
        try:
            if int(mount.get("baudrate", 0)) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            warnings.append("instrument.mount.baudrate must be a positive integer.")
        for key in ("read_timeout_s", "write_timeout_s"):
            try:
                timeout = float(mount.get(key, 0))
                if not np.isfinite(timeout) or timeout <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                warnings.append(f"instrument.mount.{key} must be a positive finite number.")
        try:
            poll_interval = float(mount.get("poll_interval_s", 0.02))
            if not np.isfinite(poll_interval) or poll_interval <= 0:
                raise ValueError
        except (TypeError, ValueError):
            warnings.append("instrument.mount.poll_interval_s must be a positive finite number.")
        if not flat_volts.get("calibration_file"):
            warnings.append("dm_voltage_states.flat.calibration_file is missing; F0 is unavailable.")
        else:
            flat_path = Path(str(flat_volts["calibration_file"])).expanduser()
            if not flat_path.is_absolute():
                flat_path = Path(config.get("_root", PACKAGE_ROOT)) / flat_path
            if not flat_path.is_file():
                warnings.append(f"The configured DM flat-voltage file does not exist: {flat_path}")
        if not flat_volts.get("validated", False):
            warnings.append("The configured DM flat-voltage file is unvalidated; F0 remains safeguarded.")
    limits = config.get("conex", {}).get("limits", {})
    if limits.get("u") is None or limits.get("v") is None:
        warnings.append("Conex absolute limits are not configured; bulk alignment will use only its 100-pixel trust region.")
    fixed_roll = config.get("mount_control", {}).get("fixed_roll")
    if fixed_roll is None:
        warnings.append("mount_control.fixed_roll is missing; mount scans cannot enforce a common roll value.")
    else:
        try:
            if not np.isfinite(float(fixed_roll)):
                raise ValueError
        except (TypeError, ValueError):
            warnings.append("mount_control.fixed_roll must be a finite number.")
    mount_limits = config.get("mount_control", {}).get("limits", {})
    for axis in ("pitch", "roll", "yaw", "x", "z", "y"):
        bounds = mount_limits.get(axis)
        if not isinstance(bounds, list) or len(bounds) != 2:
            warnings.append(f"mount_control.limits.{axis} must contain [minimum, maximum].")
            continue
        try:
            lower, upper = map(float, bounds)
            if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
                raise ValueError
            if axis == "roll" and fixed_roll is not None and not lower <= float(fixed_roll) <= upper:
                warnings.append(f"Fixed roll {fixed_roll} is outside the configured roll limits {bounds}.")
        except (TypeError, ValueError):
            warnings.append(f"mount_control.limits.{axis} must contain finite, increasing numbers.")
    if not config.get("mount_control", {}).get("validated", False):
        warnings.append("The configured fixed roll is marked unvalidated; verify it before mount motion.")
    if not config.get("scans", {}).get("set_z", {}).get("validated", False):
        warnings.append("The default Set mount Z target is marked unvalidated; verify it before applying Z.")
    mirrors = config.get("steering_mirrors", {})
    devices = mirrors.get("devices", {})
    if mirrors.get("controlled_device") != "glint_steering2":
        warnings.append("steering_mirrors.controlled_device must be glint_steering2.")
    if devices.get("glint_steering1", {}).get("command_enabled", False):
        warnings.append("glint_steering1 must remain command-disabled.")
    for mirror in ("glint_steering1", "glint_steering2"):
        recovery = devices.get(mirror, {}).get("recovery_position", {})
        try:
            if not all(np.isfinite(float(recovery[axis])) for axis in ("u", "v")):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            warnings.append(f"{mirror} needs a finite recovery_position with u and v.")
    try:
        maps = rest_dm_maps(config)
        default_map = config.get("rest_of_dm", {}).get("default_map")
        if default_map not in maps:
            warnings.append("rest_of_dm.default_map does not name one of rest_of_dm.maps.")
        for map_name, definition in maps.items():
            state_name = definition.get("dm_state")
            if state_name not in config.get("dm_states", {}):
                warnings.append(f"Rest-of-DM map {map_name!r} references missing DM state {state_name!r}.")
            elif not config["dm_states"][state_name].get("validated", False):
                warnings.append(
                    f"Rest-of-DM map {map_name!r} uses unvalidated DM state {state_name!r}; "
                    "do not select it on the instrument until calibrated."
                )
    except ValueError as error:
        warnings.append(str(error))
    return warnings
