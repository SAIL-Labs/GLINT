from __future__ import annotations

import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.signal import find_peaks

from .analysis import (
    agreement,
    fit_elliptical_gaussian,
    hotspot,
    solve_common_pistons,
    threshold_center_of_mass,
)
from .boxes import (
    INJECTED_SEGMENTS,
    PHOT_BOXES,
    SEGMENT_TO_PHOT,
    differential_null_boxes,
    integrate_box,
)
from .config import rest_dm_maps, rest_dm_state, segment_state
from .hardware import AlignmentHardware
from .models import AxisGrid, ScanResult, SegmentPTT, dm_state_to_dict
from .storage import load_saved_dark


class ScanCancelled(RuntimeError):
    pass


class AlignmentEngine:
    """All acquisition, analysis, restore, and apply logic independent of the GUI."""

    def __init__(
        self,
        hardware: AlignmentHardware,
        config: dict,
        *,
        progress: Callable[[str, int, int], None] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.hardware = hardware
        self.config = config
        self.progress = progress or (lambda message, current, total: None)
        self.log = log or (lambda message: None)
        self._abort = threading.Event()
        self.focal_dark: np.ndarray | None = load_saved_dark(config, "focal")
        self.apapane_dark: np.ndarray | None = load_saved_dark(config, "apapane")
        rest_config = config.get("rest_of_dm", {})
        self.active_rest_dm_map = str(rest_config.get("default_map", "all_tilted"))
        definition = rest_dm_maps(config)[self.active_rest_dm_map]
        self.active_rest_dm_piston_um = (
            float(definition.get("default_piston_um", 0.0))
            if definition.get("editable_uniform_ptt", False)
            else None
        )
        editable_tip_tilt = bool(
            definition.get("editable_uniform_ptt", False)
            or definition.get("editable_uniform_tip_tilt", False)
        )
        self.active_rest_dm_tip_mrad = (
            float(definition.get("default_tip_mrad", 0.0))
            if editable_tip_tilt
            else None
        )
        self.active_rest_dm_tilt_mrad = (
            float(definition.get("default_tilt_mrad", 0.0))
            if editable_tip_tilt
            else None
        )

    def rest_dm_selection(self) -> dict:
        definition = rest_dm_maps(self.config)[self.active_rest_dm_map]
        return {
            "map": self.active_rest_dm_map,
            "label": definition.get("label", self.active_rest_dm_map),
            "dm_state": definition["dm_state"],
            "uniform_piston_um": self.active_rest_dm_piston_um,
            "uniform_tip_mrad": self.active_rest_dm_tip_mrad,
            "uniform_tilt_mrad": self.active_rest_dm_tilt_mrad,
        }

    def _rest_dm_state(self) -> dict[int, SegmentPTT]:
        return rest_dm_state(
            self.config,
            self.active_rest_dm_map,
            uniform_piston_um=self.active_rest_dm_piston_um,
            uniform_tip_mrad=self.active_rest_dm_tip_mrad,
            uniform_tilt_mrad=self.active_rest_dm_tilt_mrad,
        )

    def _scan_away_state(self) -> dict[int, SegmentPTT]:
        """Combine the selected non-injected map with injected away values."""
        state = self._rest_dm_state()
        injected_away = segment_state(self.config, "tilted_away")
        for segment in INJECTED_SEGMENTS:
            state[segment] = deepcopy(injected_away[segment])
        return state

    def set_rest_dm_map(
        self,
        name: str,
        *,
        uniform_piston_um: float | None = None,
        uniform_tip_mrad: float | None = None,
        uniform_tilt_mrad: float | None = None,
    ) -> dict:
        """Physically apply and select a named map for non-injected segments."""
        self.reset_abort()
        definition = rest_dm_maps(self.config).get(name)
        if definition is None:
            raise ValueError(f"Unknown rest-of-DM map {name!r}")
        editable_ptt = bool(definition.get("editable_uniform_ptt", False))
        editable_tip_tilt = editable_ptt or bool(definition.get("editable_uniform_tip_tilt", False))
        if editable_ptt and uniform_piston_um is None:
            uniform_piston_um = float(definition.get("default_piston_um", 0.0))
        if editable_tip_tilt and (uniform_tip_mrad is None or uniform_tilt_mrad is None):
            raise ValueError("Editable REST OF DM map requires both tip and tilt values")
        mapped = rest_dm_state(
            self.config,
            name,
            uniform_piston_um=uniform_piston_um,
            uniform_tip_mrad=uniform_tip_mrad,
            uniform_tilt_mrad=uniform_tilt_mrad,
        )
        target = self.hardware.get_dm()
        for segment in range(37):
            if segment not in INJECTED_SEGMENTS:
                target[segment] = deepcopy(mapped[segment])
        self.hardware.set_dm(target)
        self._settle()
        self.active_rest_dm_map = name
        self.active_rest_dm_piston_um = float(uniform_piston_um) if editable_ptt else None
        self.active_rest_dm_tip_mrad = float(uniform_tip_mrad) if editable_tip_tilt else None
        self.active_rest_dm_tilt_mrad = float(uniform_tilt_mrad) if editable_tip_tilt else None
        return self.rest_dm_selection()

    def apply_dm_preset(self, preset: str) -> None:
        """Apply one operator-requested whole-DM preset through the hardware boundary."""
        self.reset_abort()
        if preset == "zero_ptt":
            self.hardware.set_dm(segment_state(self.config, "zero_ptt"))
        elif preset == "zero_volts":
            self.hardware.zero_dm_volts()
        elif preset == "flat_volts":
            flat = self.config.get("dm_voltage_states", {}).get("flat", {})
            if not flat.get("validated", False):
                raise RuntimeError(
                    "The configured DM flat-voltage state is not validated; verify its config entry before F0"
                )
            self.hardware.flatten_dm_volts()
        else:
            raise ValueError(f"Unknown DM preset {preset!r}")
        self._settle()

    def fixed_roll(self) -> float:
        try:
            return float(self.config["mount_control"]["fixed_roll"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("mount_control.fixed_roll must be configured as a number") from error

    def ensure_fixed_roll(self) -> None:
        self._move_mount({"roll": self.fixed_roll()})

    def validate_scan_parameters(self, kind: str, parameters: dict) -> None:
        """Validate complete scan grids before any hardware state is changed."""
        grid_axes = {
            "pitch_yaw": ("pitch", "yaw"),
            "xy": ("x", "y"),
            "tip_tilt": ("tip", "tilt"),
            "null": ("piston",),
        }
        for axis in grid_axes.get(kind, ()):
            try:
                grid = AxisGrid(**parameters[axis])
                values = grid.values()
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid {axis} scan grid: {error}") from error
            if kind in {"pitch_yaw", "xy"}:
                limits = self.hardware.mount_limits()
                if axis not in limits:
                    raise ValueError(f"No configured mount limit is available for {axis}")
                lower, upper = limits[axis]
                start = float(values[0])
                endpoint = float(values[-1])
                if start < lower or endpoint > upper:
                    raise ValueError(
                        f"{axis.title()} scan {start:g} to {endpoint:g} exceeds configured limits "
                        f"[{lower:g}, {upper:g}]. The endpoint is start + range "
                        f"({grid.start:g} + {grid.span:g})."
                    )
        for key in ("nframes", "numavg", "max_iterations"):
            if key in parameters and int(parameters[key]) < 1:
                raise ValueError(f"{key} must be at least 1")
        if "settle_s" in parameters:
            settle_s = float(parameters["settle_s"])
            if not np.isfinite(settle_s) or settle_s < 0:
                raise ValueError("settle_s must be a finite non-negative value")
        if kind == "null":
            period_bounds = np.asarray(parameters.get("period_bounds_um", (1.2, 2.0)), dtype=float)
            if (
                period_bounds.shape != (2,)
                or not np.all(np.isfinite(period_bounds))
                or not 0 < period_bounds[0] < period_bounds[1]
            ):
                raise ValueError("period_bounds_um must contain two increasing positive OPD periods")
            min_fit_r2 = float(parameters.get("min_fit_r2", 0.35))
            if not np.isfinite(min_fit_r2) or not 0 <= min_fit_r2 <= 1:
                raise ValueError("min_fit_r2 must lie in [0, 1]")
            max_invalid_fraction = float(parameters.get("max_invalid_fraction", 0.05))
            if not np.isfinite(max_invalid_fraction) or not 0 <= max_invalid_fraction < 1:
                raise ValueError("max_invalid_fraction must lie in [0, 1)")

    def set_dm_segment(self, segment: int, piston_um: float, tip_mrad: float, tilt_mrad: float) -> None:
        """Apply one edited segment while preserving every other live segment."""
        if segment not in range(37):
            raise ValueError("DM segment must be in the range 0..36")
        values = np.asarray([piston_um, tip_mrad, tilt_mrad], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Piston, tip, and tilt must be finite numbers")
        state = self.hardware.get_dm()
        state[segment] = SegmentPTT(float(piston_um), float(tip_mrad), float(tilt_mrad))
        self.hardware.set_dm(state)
        self._settle()

    def jog_mount(self, axis: str, step: float) -> dict[str, float]:
        """Move one mount axis relative to its current readback with safeguards."""
        if axis not in {"pitch", "yaw", "roll", "x", "y", "z"}:
            raise ValueError(f"Unknown mount axis {axis!r}")
        if axis == "roll":
            raise ValueError(f"Roll is fixed at {self.fixed_roll()} and cannot be jogged")
        if not np.isfinite(float(step)) or float(step) == 0:
            raise ValueError("Mount step must be a finite non-zero number")
        current = self.hardware.get_mount()
        self._move_mount({axis: float(current[axis]) + float(step)})
        return self.hardware.get_mount()

    def abort(self) -> None:
        self._abort.set()
        self.hardware.stop()

    def reset_abort(self) -> None:
        self._abort.clear()

    def _check_abort(self) -> None:
        if self._abort.is_set():
            raise ScanCancelled("Scan aborted by user")

    def _settle(self, delay_s: float | None = None) -> None:
        """Wait for optical/DM settling after a verified command.

        Scan-specific delays are explicit in the JSON configuration.  The
        global safety value remains the fallback for manual actions and older
        configuration files.
        """
        delay = float(self.config["safety"]["settle_s"] if delay_s is None else delay_s)
        if delay < 0:
            raise ValueError("Settling time cannot be negative")
        time.sleep(delay * float(self.config.get("time_scale", 1.0)))
        self._check_abort()

    def _scan_settle(self, parameters: dict) -> float:
        return float(parameters.get("settle_s", self.config["safety"]["settle_s"]))

    def _snapshot_metadata(self) -> dict:
        return {
            "mount": self.hardware.get_mount(),
            "conex": self.hardware.get_conex(),
            "dm": dm_state_to_dict(self.hardware.get_dm()),
            "hardware_mode": self.hardware.mode,
        }

    @staticmethod
    def _average_frame(frames: np.ndarray) -> np.ndarray:
        """Accept both legacy 2-D multi_recv_data output and frame cubes."""
        values = np.asarray(frames, dtype=float)
        if values.ndim == 2:
            return values
        if values.ndim == 3:
            return np.nanmean(values, axis=0)
        raise ValueError(f"Expected a 2-D frame or 3-D frame cube, got {values.shape}")

    def acquire_dark(self, camera: str, nframes: int = 25) -> ScanResult:
        started = time.perf_counter()
        command_log_start = self.hardware.command_log_marker()
        self.reset_abort()
        if camera not in {"focal", "apapane"}:
            raise ValueError("camera must be focal or apapane")
        self.log(f"Acquiring {nframes} {camera} dark frames")
        self.hardware.set_dark_mode(True)
        try:
            frames = self.hardware.focal_frames(nframes) if camera == "focal" else self.hardware.apapane_frames(nframes)
        finally:
            self.hardware.set_dark_mode(False)
        self._check_abort()
        values = np.asarray(frames, dtype=float)
        dark = values if values.ndim == 2 else np.median(values, axis=0)
        if dark.ndim != 2:
            raise ValueError(f"Expected a 2-D dark or 3-D dark cube, got {values.shape}")
        if camera == "focal":
            self.focal_dark = dark
        else:
            self.apapane_dark = dark
        result = ScanResult(
            kind=f"{camera}_dark",
            maps={"dark": dark},
            metadata={"nframes": nframes, "state_at_completion": self._snapshot_metadata()},
        )
        result.metadata["runtime"] = {"acquisition_s": time.perf_counter() - started}
        result.metadata["command_log_start_sequence"] = command_log_start
        return result

    def _guarded(
        self,
        operation: Callable[[], ScanResult],
        *,
        keep_dm_ptt_on_success: bool = False,
        keep_dm_volts_on_success: bool = False,
    ) -> ScanResult:
        guarded_started = time.perf_counter()
        command_log_start = self.hardware.command_log_marker()
        self.reset_abort()
        # Refresh all mount axes before trusting the verified-position cache.
        # The pre-scan recovery target deliberately uses the configured fixed
        # roll even if an external process changed roll while this GUI was idle.
        before = self.hardware.snapshot()
        before.mount["roll"] = self.fixed_roll()
        try:
            self.ensure_fixed_roll()
            operation_started = time.perf_counter()
            result = operation()
            operation_elapsed = time.perf_counter() - operation_started
        except BaseException:
            self.log("Restoring pre-scan state")
            self.hardware.restore(before, float(self.config["safety"]["motion_timeout_s"]))
            raise
        successful_dm = self.hardware.get_dm() if keep_dm_ptt_on_success else None
        successful_volts = self.hardware.get_dm_volts() if keep_dm_volts_on_success else None
        self.hardware.restore(before, float(self.config["safety"]["motion_timeout_s"]))
        if successful_volts is not None:
            self.hardware.set_dm_volts(successful_volts)
        if successful_dm is not None:
            self.hardware.set_dm(successful_dm)
        result.metadata["pre_scan_state"] = {
            "mount": before.mount,
            "conex": before.conex,
            "dm": dm_state_to_dict(before.dm),
        }
        result.metadata["dm_ptt_retained_after_scan"] = keep_dm_ptt_on_success
        result.metadata["state_at_completion"] = self._snapshot_metadata()
        result.metadata["command_log_start_sequence"] = command_log_start
        runtime = result.metadata.setdefault("runtime", {})
        runtime["scan_operation_s"] = operation_elapsed
        runtime["guarded_total_s"] = time.perf_counter() - guarded_started
        return result

    def restore_review_state(self, result: ScanResult) -> None:
        """Restore a DM state deliberately retained while a result was reviewed."""
        if not result.metadata.get("dm_ptt_retained_after_scan", False):
            return
        saved = result.metadata["pre_scan_state"]["dm"]
        state = {int(segment): SegmentPTT(**values) for segment, values in saved.items()}
        self.hardware.set_dm(state)
        self._settle()

    def bulk_align(self) -> ScanResult:
        if self.focal_dark is None:
            raise RuntimeError("Acquire a focal-plane dark before bulk alignment")

        def operation() -> ScanResult:
            bulk = self.config["scans"]["bulk"]
            settle_s = self._scan_settle(bulk)
            self.hardware.zero_dm_volts()
            reference_bright, reference_dark = self.hardware.focal_reference()
            goal = hotspot(reference_bright - reference_dark, bulk["crop"], bulk["centroid_threshold"])
            current = self.hardware.get_conex()
            theta = np.asarray([current["u"], current["v"]], dtype=float)
            response = self._load_response_matrix(bulk)
            history: list[dict] = []
            stalled = 0
            for iteration in range(int(bulk["max_iterations"]) + 1):
                self._check_abort()
                image = self._average_frame(self.hardware.focal_frames(int(bulk["nframes"]))) - self.focal_dark
                position = hotspot(image, bulk["crop"], bulk["centroid_threshold"])
                error = goal - position
                norm = float(np.linalg.norm(error))
                history.append(
                    {"iteration": iteration, "u": theta[0], "v": theta[1], "x": position[0], "y": position[1], "error_px": norm}
                )
                self.progress(f"Bulk alignment: {norm:.2f} px", iteration, int(bulk["max_iterations"]))
                if norm <= float(bulk["tolerance_px"]):
                    break
                if iteration == int(bulk["max_iterations"]):
                    break
                if response is None or np.linalg.cond(response) > float(bulk["max_condition"]) or stalled >= 2:
                    response = self._symmetric_response(theta, bulk)
                    stalled = 0
                step = np.linalg.pinv(response) @ error
                predicted = float(np.linalg.norm(response @ step))
                trust = float(bulk["trust_region_px"])
                if predicted > trust:
                    step *= trust / predicted
                accepted = False
                for backtrack in range(int(bulk["max_backtracks"]) + 1):
                    fraction = 0.5**backtrack
                    trial_theta = theta + fraction * step
                    self._check_conex_limits(trial_theta)
                    self.hardware.move_conex({"u": trial_theta[0], "v": trial_theta[1]})
                    self._settle(settle_s)
                    trial_image = self._average_frame(self.hardware.focal_frames(int(bulk["nframes"]))) - self.focal_dark
                    trial_position = hotspot(trial_image, bulk["crop"], bulk["centroid_threshold"])
                    trial_error = float(np.linalg.norm(goal - trial_position))
                    if trial_error < norm:
                        dtheta = trial_theta - theta
                        dposition = trial_position - position
                        denominator = float(dtheta @ dtheta)
                        if denominator > 0:
                            response = response + np.outer(dposition - response @ dtheta, dtheta) / denominator
                        theta = trial_theta
                        stalled = stalled + 1 if trial_error > 0.95 * norm else 0
                        accepted = True
                        break
                if not accepted:
                    self.hardware.move_conex({"u": theta[0], "v": theta[1]})
                    stalled += 1
            final_error = history[-1]["error_px"]
            return ScanResult(
                kind="bulk",
                axes={"iteration": np.arange(len(history))},
                maps={"current_psf": image, "reference_psf": reference_bright - reference_dark},
                estimates={"goal_xy": goal, "history": history, "response_matrix": response},
                recommendation={
                    "u": float(theta[0]),
                    "v": float(theta[1]),
                    "error_px": final_error,
                    "passed": bool(final_error <= float(bulk["tolerance_px"])),
                },
                metadata={
                    "dm_voltage_state": "zero_volts",
                    "parameters": bulk,
                    "sample_count": len(history),
                    "settle_s_per_motion": settle_s,
                },
            )

        return self._guarded(operation, keep_dm_volts_on_success=True)

    def _load_response_matrix(self, bulk: dict) -> np.ndarray | None:
        path = bulk.get("response_matrix")
        if path:
            path = Path(path)
            if not path.is_absolute():
                path = Path(self.config["_root"]) / path
            if path.exists():
                matrix = np.asarray(np.load(path), dtype=float)
                if matrix.shape == (2, 2):
                    return matrix
        initial = bulk.get("initial_response_matrix")
        return None if initial is None else np.asarray(initial, dtype=float)

    def _symmetric_response(self, theta: np.ndarray, bulk: dict) -> np.ndarray:
        response = np.zeros((2, 2), dtype=float)
        perturb = np.asarray(bulk["response_perturbation"], dtype=float)
        for axis in range(2):
            positions = []
            for sign in (-1.0, 1.0):
                trial = theta.copy()
                trial[axis] += sign * perturb[axis]
                self._check_conex_limits(trial)
                self.hardware.move_conex({"u": trial[0], "v": trial[1]})
                self._settle(self._scan_settle(bulk))
                frame = self._average_frame(self.hardware.focal_frames(int(bulk["nframes"]))) - self.focal_dark
                positions.append(hotspot(frame, bulk["crop"], bulk["centroid_threshold"]))
            response[:, axis] = (positions[1] - positions[0]) / (2 * perturb[axis])
        self.hardware.move_conex({"u": theta[0], "v": theta[1]})
        self._settle(self._scan_settle(bulk))
        return response

    def _check_conex_limits(self, theta: np.ndarray) -> None:
        limits = self.config.get("conex", {}).get("limits", {})
        for index, axis in enumerate(("u", "v")):
            bounds = limits.get(axis)
            if bounds is not None and not (bounds[0] <= theta[index] <= bounds[1]):
                raise ValueError(f"Conex {axis} target {theta[index]} outside configured limits {bounds}")

    def pitch_yaw_scan(self, parameters: dict | None = None) -> ScanResult:
        params = deepcopy(self.config["scans"]["pitch_yaw"])
        params.update(parameters or {})
        self.validate_scan_parameters("pitch_yaw", params)

        def operation() -> ScanResult:
            self.hardware.set_dm(segment_state(self.config, "zero_ptt"))
            result = self._mount_map("pitch_yaw", "pitch", "yaw", params, fit="gaussian")
            result.metadata["dm_ptt_state"] = "zero_ptt"
            return result

        return self._guarded(operation)

    def _xy_dm_state(self) -> dict[int, SegmentPTT]:
        """Return the complete DM state used for X/Y injection scans."""
        state = self._rest_dm_state()
        zero = segment_state(self.config, "zero_ptt")
        for segment in INJECTED_SEGMENTS:
            state[segment] = deepcopy(zero[segment])
        return state

    def xy_scan(self, parameters: dict | None = None) -> ScanResult:
        params = deepcopy(self.config["scans"]["xy"])
        params.update(parameters or {})
        self.validate_scan_parameters("xy", params)

        def operation() -> ScanResult:
            self.hardware.set_dm(self._xy_dm_state())
            result = self._mount_map("xy", "x", "y", params, fit="com")
            result.metadata["dm_ptt_state"] = "selected_rest_of_dm_except_injected_zero_ptt"
            result.metadata["rest_of_dm"] = self.rest_dm_selection()
            return result

        return self._guarded(operation, keep_dm_ptt_on_success=True)

    def _mount_map(self, kind: str, x_axis: str, y_axis: str, params: dict, fit: str) -> ScanResult:
        if self.apapane_dark is None:
            raise RuntimeError("Acquire an APAPANE dark before mount scans")
        xv = AxisGrid(**params[x_axis]).values()
        yv = AxisGrid(**params[y_axis]).values()
        maps = {name: np.zeros((len(yv), len(xv)), dtype=float) for name in PHOT_BOXES}
        actual_x = np.zeros((len(yv), len(xv)), dtype=float)
        actual_y = np.zeros((len(yv), len(xv)), dtype=float)
        total = len(xv) * len(yv)
        count = 0
        settle_s = self._scan_settle(params)
        for iy, y in enumerate(yv):
            # A serpentine raster avoids the long return to the first X value at
            # every row boundary. The array indices remain in ascending physical
            # coordinates, so fitting and saved products are unchanged.
            x_indices = range(len(xv)) if iy % 2 == 0 else range(len(xv) - 1, -1, -1)
            for ix in x_indices:
                x = xv[ix]
                readback = self._move_mount({x_axis: x, y_axis: y})
                self._settle(settle_s)
                actual_x[iy, ix] = readback[x_axis]
                actual_y[iy, ix] = readback[y_axis]
                frame = self._average_frame(self.hardware.apapane_frames(int(params["nframes"]))) - self.apapane_dark
                for name, box in PHOT_BOXES.items():
                    maps[name][iy, ix] = integrate_box(frame, box)
                count += 1
                self.progress(f"{kind}: {x_axis}={x:.3f}, {y_axis}={y:.3f}", count, total)
        estimates = []
        for name in PHOT_BOXES:
            estimate = (
                fit_elliptical_gaussian(maps[name], xv, yv)
                if fit == "gaussian"
                else threshold_center_of_mass(maps[name], xv, yv, float(params["com_threshold"]))
            )
            estimate["channel"] = name
            estimates.append(estimate)
        check = agreement(estimates, float(params["agreement_pulses"]))
        return ScanResult(
            kind=kind,
            axes={x_axis: xv, y_axis: yv, f"actual_{x_axis}": actual_x, f"actual_{y_axis}": actual_y},
            maps=maps,
            estimates={"channels": estimates, "agreement": check},
            recommendation={x_axis: check["x"], y_axis: check["y"], "passed": check["passed"]},
            metadata={
                "parameters": params,
                "sample_count": total,
                "settle_s_per_sample": settle_s,
                "raster_path": "serpentine",
            },
        )

    def tip_tilt_scan(self, parameters: dict | None = None) -> ScanResult:
        params = deepcopy(self.config["scans"]["tip_tilt"])
        params.update(parameters or {})
        self.validate_scan_parameters("tip_tilt", params)

        def operation() -> ScanResult:
            if self.apapane_dark is None:
                raise RuntimeError("Acquire an APAPANE dark before tip/tilt scans")
            tips = AxisGrid(**params["tip"]).values()
            tilts = AxisGrid(**params["tilt"]).values()
            maps: dict[str, np.ndarray] = {}
            estimates: dict[str, dict] = {}
            total = len(INJECTED_SEGMENTS) * len(tips) * len(tilts)
            count = 0
            away = self._scan_away_state()
            zero = segment_state(self.config, "zero_ptt")
            settle_s = self._scan_settle(params)
            for segment in INJECTED_SEGMENTS:
                state = deepcopy(away)
                state[segment] = deepcopy(zero[segment])
                self.hardware.set_dm(state)
                data = np.zeros((len(tilts), len(tips)), dtype=float)
                for iy, tilt in enumerate(tilts):
                    tip_indices = range(len(tips)) if iy % 2 == 0 else range(len(tips) - 1, -1, -1)
                    for ix in tip_indices:
                        tip = tips[ix]
                        state[segment] = SegmentPTT(0.0, float(tip), float(tilt))
                        self.hardware.set_dm(state)
                        self._settle(settle_s)
                        frame = self._average_frame(self.hardware.apapane_frames(int(params["nframes"]))) - self.apapane_dark
                        data[iy, ix] = integrate_box(frame, PHOT_BOXES[SEGMENT_TO_PHOT[segment]])
                        count += 1
                        self.progress(f"Tip/tilt segment {segment}", count, total)
                key = str(segment)
                maps[key] = data
                estimates[key] = fit_elliptical_gaussian(data, tips, tilts)
            return ScanResult(
                kind="tip_tilt",
                axes={"tip": tips, "tilt": tilts},
                maps=maps,
                estimates=estimates,
                recommendation={
                    "segments": {
                        key: {"piston_um": 0.0, "tip_mrad": value["x"], "tilt_mrad": value["y"]}
                        for key, value in estimates.items()
                    },
                    "passed": not any(bool(value.get("boundary")) for value in estimates.values()),
                },
                metadata={
                    "parameters": params,
                    "rest_of_dm": self.rest_dm_selection(),
                    "sample_count": total,
                    "settle_s_per_sample": settle_s,
                    "raster_path": "serpentine",
                },
            )

        return self._guarded(operation)

    def null_scan(self, parameters: dict | None = None) -> ScanResult:
        params = deepcopy(self.config["scans"]["null"])
        params.update(parameters or {})
        self.validate_scan_parameters("null", params)

        def operation() -> ScanResult:
            if self.apapane_dark is None:
                raise RuntimeError("Acquire an APAPANE dark before null scans")
            minimum = float(params["piston"]["start"])
            maximum = minimum + float(params["piston"]["span"])
            piston_step = float(params["piston"]["step"])
            differential_positions = AxisGrid(
                start=minimum - maximum,
                span=2.0 * (maximum - minimum),
                step=piston_step,
            ).values()

            # The one-segment-at-a-time trajectory changes which segment moves at
            # differential piston zero.  Make that turning point an explicit
            # sample even if a future configuration does not land on it exactly.
            zero_tolerance = 1e-12 * max(1.0, abs(maximum - minimum))
            zero_hits = np.flatnonzero(np.isclose(differential_positions, 0.0, atol=zero_tolerance, rtol=0.0))
            if len(zero_hits):
                differential_positions[int(zero_hits[0])] = 0.0
            else:
                differential_positions = np.sort(np.append(differential_positions, 0.0))

            # Scan one physical baseline at a time.  For every measured step only
            # ONE of the two baseline pistons is allowed to change:
            #   leg 1: A moves minimum -> maximum while B stays at maximum;
            #   leg 2: A stays at maximum while B moves maximum -> minimum.
            # The third injected segment remains fixed at its zero-PTT reference.
            baselines = [(11, 31), (11, 20)]
            maps: dict[str, np.ndarray] = {}
            estimates: dict[str, dict] = {}
            total = len(baselines) * int(params["numavg"]) * len(differential_positions)
            progress_count = 0
            current = self.hardware.get_dm()
            reference = segment_state(self.config, "zero_ptt")
            base_state = self._rest_dm_state()
            scan_boxes = differential_null_boxes()

            # Preserve the optimized tip/tilt of all three injected segments.
            # Only their pistons are manipulated during the null scan.
            for segment in INJECTED_SEGMENTS:
                base_state[segment] = deepcopy(current[segment])
                base_state[segment].piston_um = float(reference[segment].piston_um)

            settle_s = self._scan_settle(params)
            simulator_selector = getattr(self.hardware, "set_null_scan_baseline", None)

            # Put only the NON-INJECTED segments into the selected REST OF DM
            # state.  The three injected segments are deliberately left untouched
            # here so that their piston changes can subsequently be issued one
            # segment at a time.
            state = self.hardware.get_dm()
            for segment in range(37):
                if segment not in INJECTED_SEGMENTS:
                    state[segment] = deepcopy(base_state[segment])
            self.hardware.set_dm(state)
            self._settle(settle_s)

            def command_one_piston(segment: int, target_um: float) -> bool:
                """Change exactly one injected piston in one DM command.

                The hardware interface receives a full DM state, but every other
                state entry is kept identical to the last commanded state.  Thus
                only ``segment`` has a changed piston target in this command.
                Returns True when a physical piston change was commanded.
                """
                nonlocal state
                target = float(target_um)
                current_piston = float(state[segment].piston_um)
                if np.isclose(current_piston, target, atol=1e-12, rtol=0.0):
                    return False
                state[segment].piston_um = target
                self.hardware.set_dm(state)
                self._settle(settle_s)
                return True

            try:
                for baseline in baselines:
                    a, b = baseline
                    fixed_segment = next(
                        segment for segment in INJECTED_SEGMENTS if segment not in baseline
                    )
                    box = scan_boxes[baseline]

                    # Keep the non-participating injected segment fixed throughout
                    # this baseline.  If a previous baseline left it elsewhere,
                    # park it with a single-segment piston command.
                    command_one_piston(
                        fixed_segment,
                        float(base_state[fixed_segment].piston_um),
                    )

                    if simulator_selector is not None:
                        simulator_selector(baseline)

                    piston_pairs = np.asarray(
                        [
                            self._pair_pistons_for_differential(
                                float(differential),
                                minimum,
                                maximum,
                                float(reference[a].piston_um),
                                float(reference[b].piston_um),
                            )
                            for differential in differential_positions
                        ],
                        dtype=float,
                    )

                    # Hard trajectory checks.  These make a regression to the old
                    # simultaneous cross-sweep fail immediately instead of silently
                    # changing the physical scan.
                    if not (
                        np.isclose(piston_pairs[0, 0], minimum, atol=1e-12)
                        and np.isclose(piston_pairs[0, 1], maximum, atol=1e-12)
                        and np.isclose(piston_pairs[-1, 0], maximum, atol=1e-12)
                        and np.isclose(piston_pairs[-1, 1], minimum, atol=1e-12)
                    ):
                        raise RuntimeError(
                            f"Null {a}-{b} trajectory endpoints are incorrect: "
                            f"first=({piston_pairs[0, 0]:g}, {piston_pairs[0, 1]:g}), "
                            f"last=({piston_pairs[-1, 0]:g}, {piston_pairs[-1, 1]:g})"
                        )
                    pair_changes = ~np.isclose(
                        np.diff(piston_pairs, axis=0),
                        0.0,
                        atol=1e-12,
                        rtol=0.0,
                    )
                    if np.any(np.sum(pair_changes, axis=1) > 1):
                        raise RuntimeError(
                            f"Null {a}-{b} trajectory would move both baseline segments in one measured step"
                        )

                    repetitions: list[list[float]] = []
                    best_per_repeat: list[dict[str, float | int | str]] = []
                    for repeat in range(int(params["numavg"])):
                        # Reposition to the first sample WITHOUT ever changing two
                        # injected pistons in the same command.  B is placed at MAX
                        # first, then A at MIN.  The first flux sample is acquired
                        # only after this start state has been reached.
                        start_a = float(piston_pairs[0, 0])
                        start_b = float(piston_pairs[0, 1])
                        command_one_piston(b, start_b)
                        command_one_piston(a, start_a)

                        fluxes: list[float] = []
                        previous_a = start_a
                        previous_b = start_b

                        for index, differential in enumerate(differential_positions):
                            piston_a = float(piston_pairs[index, 0])
                            piston_b = float(piston_pairs[index, 1])

                            changed_a = not np.isclose(previous_a, piston_a, atol=1e-12, rtol=0.0)
                            changed_b = not np.isclose(previous_b, piston_b, atol=1e-12, rtol=0.0)
                            if changed_a and changed_b:
                                raise RuntimeError(
                                    f"Null {a}-{b} attempted simultaneous piston motion at "
                                    f"differential={float(differential):g} um"
                                )

                            moved_segment: int | None = None
                            if changed_a:
                                command_one_piston(a, piston_a)
                                moved_segment = a
                            elif changed_b:
                                command_one_piston(b, piston_b)
                                moved_segment = b

                            # The inactive injected segment must remain parked for
                            # the complete baseline measurement.
                            if not np.isclose(
                                float(state[fixed_segment].piston_um),
                                float(base_state[fixed_segment].piston_um),
                                atol=1e-12,
                                rtol=0.0,
                            ):
                                raise RuntimeError(
                                    f"Fixed segment {fixed_segment} changed during null baseline {a}-{b}"
                                )

                            fluxes.append(self._null_flux(box, int(params["nframes"])))
                            progress_count += 1
                            motion_text = (
                                "start position"
                                if moved_segment is None
                                else f"only p{moved_segment} moved"
                            )
                            self.progress(
                                (
                                    f"Null {a}-{b}, repeat {repeat + 1}: "
                                    f"p{a}={piston_a:+.4f} um, "
                                    f"p{b}={piston_b:+.4f} um, "
                                    f"p{fixed_segment}={state[fixed_segment].piston_um:+.4f} um fixed; "
                                    f"{motion_text}"
                                ),
                                progress_count,
                                total,
                            )
                            previous_a = piston_a
                            previous_b = piston_b

                        repetitions.append(fluxes)
                        best_per_repeat.append(
                            self._parabolic_minimum(
                                differential_positions,
                                np.asarray(fluxes, dtype=float),
                            )
                        )

                    repetition_array = np.asarray(repetitions, dtype=float)
                    mean_flux = np.mean(repetition_array, axis=0)
                    # Select from the curve shown in the review.  Averaging the
                    # independently selected positions is invalid for a periodic
                    # scan: different repeats may choose neighbouring fringe orders
                    # and their arithmetic mean can lie on a maximum.
                    candidates = self._null_minimum_candidates(
                        differential_positions,
                        mean_flux,
                        period_bounds_um=params.get("period_bounds_um", (1.2, 2.0)),
                    )
                    candidate = candidates[0]
                    combined_best = {
                        "best_diff_piston_um": candidate["delta_piston_um"],
                        "best_opd_um": candidate["opd_um"],
                        "best_flux": candidate["metric"],
                        "sample_min_diff_piston_um": candidate["sample_diff_piston_um"],
                        "sample_min_flux": candidate["sample_flux"],
                        "sample_min_index": candidate["sample_index"],
                        "method_used": candidate["refinement_method"],
                    }
                    key = f"{a}-{b}"
                    maps[key] = repetition_array
                    estimates[key] = {
                        "model": "measured raw-flux minimum",
                        "fit_ok": True,
                        "fit_warning": "",
                        "method": "parabolic three-sample minimum with measured-sample fallback",
                        "detector_box": list(box),
                        "opd_um": 2.0 * differential_positions,
                        "differential_piston_um": differential_positions,
                        "piston_first_um": piston_pairs[:, 0],
                        "piston_second_um": piston_pairs[:, 1],
                        "mean_metric": mean_flux,
                        "best_per_repeat": best_per_repeat,
                        "averaged_curve_minimum": combined_best,
                        # Candidate zero is the deepest minimum of the averaged
                        # curve. Further entries are neighbouring fringe minima,
                        # ordered by distance from that minimum. They are used only
                        # when the deepest-minimum pair cannot be represented inside
                        # all three DM piston ranges.
                        "candidates": candidates,
                        "best": candidate,
                        "scan_segments": [a, b],
                        "fixed_segment": fixed_segment,
                        "fixed_piston_um": float(base_state[fixed_segment].piston_um),
                        "motion_rule": "one injected baseline segment changes per measured step",
                    }
            finally:
                if simulator_selector is not None:
                    simulator_selector(None)

            solution = self._solve_null_pistons_with_neighbours(
                estimates["11-20"]["candidates"],
                estimates["11-31"]["candidates"],
                minimum,
                maximum,
            )
            return ScanResult(
                kind="null",
                axes={
                    "delta_piston_um": differential_positions,
                    "opd_um": 2.0 * differential_positions,
                },
                maps=maps,
                estimates=estimates,
                recommendation={**solution, "passed": bool(solution["feasible"])},
                metadata={
                    "parameters": params,
                    "baseline_20_31": "disabled",
                    "rest_of_dm": self.rest_dm_selection(),
                    "sample_count": total,
                    "settle_s_per_sample": settle_s,
                    "scan_sequence": (
                        "One baseline at a time; first segment moves min-to-max while second stays at max, "
                        "then first stays at max while second moves max-to-min; third injected segment fixed"
                    ),
                    "motion_invariant": "at most one injected baseline piston changes between measured samples",
                    "opd_definition": "2 * (piston_first - piston_second)",
                    "measurement_definition": "dark-subtracted integrated raw null-channel flux",
                    "injected_dm_state": (
                        "all three retain current optimized tip/tilt; third piston held at zero-PTT reference"
                    ),
                },
            )

        return self._guarded(operation)

    @staticmethod
    def _pair_pistons_for_differential(
        differential: float,
        minimum: float,
        maximum: float,
        reference_a: float,
        reference_b: float,
    ) -> tuple[float, float]:
        """Return the one-segment-at-a-time piston pair for one baseline.

        Let ``d = pA - pB`` and ``span = maximum - minimum``.

        * For ``-span <= d <= 0``: B is held at ``maximum`` and only A moves
          from ``minimum`` to ``maximum``.
        * For ``0 <= d <= +span``: A is held at ``maximum`` and only B moves
          from ``maximum`` to ``minimum``.

        The full differential range is therefore unchanged, but adjacent scan
        samples never require both baseline pistons to move simultaneously.
        ``reference_a`` and ``reference_b`` remain in the signature for backward
        compatibility and intentionally do not affect the scan trajectory.
        """
        del reference_a, reference_b
        span = float(maximum - minimum)
        if not np.isfinite(span) or span <= 0:
            raise ValueError("Null-scan maximum piston must be greater than minimum piston")

        differential_min = -span
        differential_max = span
        tolerance = 1e-12 * max(1.0, abs(span))
        d = float(differential)
        if d < differential_min - tolerance or d > differential_max + tolerance:
            raise ValueError(f"Differential piston {d:g} is outside the feasible range")
        d = float(np.clip(d, differential_min, differential_max))

        if d <= 0.0:
            # Leg 1: B remains at MAX; A alone moves MIN -> MAX.
            piston_a = maximum + d
            piston_b = maximum
        else:
            # Leg 2: A remains at MAX; B alone moves MAX -> MIN.
            piston_a = maximum
            piston_b = maximum - d

        return float(piston_a), float(piston_b)

    @staticmethod
    def _parabolic_minimum(x: np.ndarray, y: np.ndarray) -> dict[str, float | int | str]:
        """Working-script minimum: sample minimum plus a local 3-point parabola."""
        values_x = np.asarray(x, dtype=float)
        values_y = np.asarray(y, dtype=float)
        if values_x.shape != values_y.shape or values_x.ndim != 1 or len(values_x) == 0:
            raise ValueError("Null coordinates and flux must be equal non-empty 1-D arrays")
        index = int(np.nanargmin(values_y))
        return AlignmentEngine._parabolic_minimum_at_index(values_x, values_y, index)

    @classmethod
    def _null_minimum_candidates(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        *,
        period_bounds_um: tuple[float, float] | list[float] = (1.2, 2.0),
    ) -> list[dict[str, float | int | str]]:
        """Return the deepest and neighbouring minima of an averaged null scan.

        ``x`` is differential DM piston, while the configured period bounds
        are optical-path difference. Peak separation is therefore converted
        by OPD = 2 * differential piston. A small prominence requirement
        rejects sample-scale noise without imposing a sinusoidal fit.
        """
        values_x = np.asarray(x, dtype=float)
        values_y = np.asarray(y, dtype=float)
        if values_x.shape != values_y.shape or values_x.ndim != 1 or len(values_x) == 0:
            raise ValueError("Null coordinates and flux must be equal non-empty 1-D arrays")
        if not np.all(np.isfinite(values_x)) or not np.all(np.isfinite(values_y)):
            raise ValueError("Null coordinates and averaged flux must be finite")

        global_index = int(np.argmin(values_y))
        if len(values_x) > 1:
            step = float(np.median(np.diff(values_x)))
            if not np.isfinite(step) or step <= 0:
                raise ValueError("Null differential-piston coordinates must be strictly increasing")
            lower_opd_period = float(period_bounds_um[0])
            # Half of the shortest expected differential-piston fringe period
            # prevents small ripples from masquerading as adjacent fringes.
            minimum_distance = max(1, int(round(lower_opd_period / (4.0 * step))))
        else:
            minimum_distance = 1
        flux_span = float(np.ptp(values_y))
        prominence = 0.03 * flux_span if flux_span > 0 else None
        indices, _ = find_peaks(
            -values_y,
            distance=minimum_distance,
            prominence=prominence,
        )
        index_list = [int(index) for index in indices]
        if global_index not in index_list:
            index_list.append(global_index)
        index_list.sort()
        global_position = index_list.index(global_index)

        candidates: list[dict[str, float | int | str]] = []
        for position, index in enumerate(index_list):
            refined = cls._parabolic_minimum_at_index(values_x, values_y, index)
            order = abs(position - global_position)
            side = "deepest" if order == 0 else ("left" if position < global_position else "right")
            candidates.append(
                {
                    "opd_um": 2.0 * float(refined["best_diff_piston_um"]),
                    "delta_piston_um": float(refined["best_diff_piston_um"]),
                    "metric": float(refined["best_flux"]),
                    "sample_index": int(index),
                    "sample_diff_piston_um": float(values_x[index]),
                    "sample_flux": float(values_y[index]),
                    "refinement_method": str(refined["method_used"]),
                    "neighbour_order": int(order),
                    "neighbour_side": side,
                }
            )
        candidates.sort(key=lambda item: (int(item["neighbour_order"]), float(item["metric"])))
        return candidates

    @staticmethod
    def _parabolic_minimum_at_index(
        x: np.ndarray,
        y: np.ndarray,
        index: int,
    ) -> dict[str, float | int | str]:
        """Refine one selected sample minimum without changing fringe order."""
        values_x = np.asarray(x, dtype=float)
        values_y = np.asarray(y, dtype=float)
        if values_x.shape != values_y.shape or values_x.ndim != 1 or len(values_x) == 0:
            raise ValueError("Null coordinates and flux must be equal non-empty 1-D arrays")
        if index < 0 or index >= len(values_x):
            raise IndexError("Null minimum index is outside the sampled curve")
        best_x = float(values_x[index])
        best_y = float(values_y[index])
        method = "sample_min"
        if 0 < index < len(values_x) - 1:
            local_x = values_x[index - 1 : index + 2]
            local_y = values_y[index - 1 : index + 2]
            if np.all(np.isfinite(local_x)) and np.all(np.isfinite(local_y)):
                curvature, linear, constant = np.polyfit(local_x, local_y, 2)
                if curvature > 0:
                    vertex = float(-linear / (2.0 * curvature))
                    if local_x[0] <= vertex <= local_x[-1]:
                        best_x = vertex
                        best_y = float(curvature * vertex**2 + linear * vertex + constant)
                        method = "parabolic"
        return {
            "best_diff_piston_um": best_x,
            "best_opd_um": 2.0 * best_x,
            "best_flux": best_y,
            "sample_min_diff_piston_um": float(values_x[index]),
            "sample_min_flux": float(values_y[index]),
            "sample_min_index": index,
            "method_used": method,
        }

    @staticmethod
    def _solve_null_pistons_with_neighbours(
        candidates_1120: list[dict],
        candidates_1131: list[dict],
        minimum: float,
        maximum: float,
    ) -> dict[str, object]:
        """Use deepest minima first, expanding to neighbours only if required."""
        primary_1120 = [item for item in candidates_1120 if int(item.get("neighbour_order", 0)) == 0]
        primary_1131 = [item for item in candidates_1131 if int(item.get("neighbour_order", 0)) == 0]
        primary = solve_common_pistons(primary_1120, primary_1131, minimum, maximum)
        if primary["feasible"]:
            return {
                **primary,
                "fallback_attempted": False,
                "fallback_used": False,
                "selection_mode": "deepest averaged-curve minima",
                "selected_neighbour_order": {"11-20": 0, "11-31": 0},
            }

        maximum_order = max(
            [int(item.get("neighbour_order", 0)) for item in candidates_1120 + candidates_1131],
            default=0,
        )
        for order in range(1, maximum_order + 1):
            pool_1120 = [item for item in candidates_1120 if int(item.get("neighbour_order", 0)) <= order]
            pool_1131 = [item for item in candidates_1131 if int(item.get("neighbour_order", 0)) <= order]
            solution = solve_common_pistons(pool_1120, pool_1131, minimum, maximum)
            if solution["feasible"]:
                selected_orders = {
                    "11-20": next(
                        int(item.get("neighbour_order", 0))
                        for item in pool_1120
                        if np.isclose(
                            float(item["delta_piston_um"]),
                            float(solution["delta_11_20_um"]),
                            atol=1e-10,
                        )
                    ),
                    "11-31": next(
                        int(item.get("neighbour_order", 0))
                        for item in pool_1131
                        if np.isclose(
                            float(item["delta_piston_um"]),
                            float(solution["delta_11_31_um"]),
                            atol=1e-10,
                        )
                    ),
                }
                return {
                    **solution,
                    "fallback_attempted": True,
                    "fallback_used": True,
                    "fallback_level": int(order),
                    "fallback_reason": primary["reason"],
                    "selection_mode": "automatic neighbouring-minimum DM-range fallback",
                    "selected_neighbour_order": selected_orders,
                }

        return {
            "feasible": False,
            "passed": False,
            "reason": (
                f"{primary['reason']} The deepest minimum and {maximum_order} neighbouring "
                "fringe order(s) were checked automatically."
            ),
            "fallback_attempted": True,
            "fallback_used": False,
            "fallback_level": int(maximum_order),
            "selection_mode": "no feasible measured minimum",
        }

    def _null_flux(self, box: list[int], nframes: int) -> float:
        frame = self._average_frame(self.hardware.apapane_frames(nframes)) - self.apapane_dark
        return float(integrate_box(frame, box))

    def _validated_mount_targets(self, targets: dict[str, float]) -> dict[str, float]:
        targets = {axis: float(value) for axis, value in targets.items()}
        fixed_roll = self.fixed_roll()
        if "roll" in targets and not np.isclose(targets["roll"], fixed_roll, atol=1e-9):
            raise ValueError(f"Roll is fixed at {fixed_roll}; target {targets['roll']} is not permitted")
        targets["roll"] = fixed_roll
        limits = self.hardware.mount_limits()
        for axis, target in targets.items():
            if axis not in limits:
                raise ValueError(f"No mount limit is available for axis {axis}")
            lower, upper = limits[axis]
            if not lower <= float(target) <= upper:
                raise ValueError(f"{axis} target {target} outside [{lower}, {upper}]")
        return targets

    def _move_mount(self, targets: dict[str, float]) -> dict[str, float]:
        targets = self._validated_mount_targets(targets)
        return self.hardware.move_mount(targets, float(self.config["safety"]["motion_timeout_s"]))

    def set_z_scan(self, parameters: dict | None = None) -> ScanResult:
        command_log_start = self.hardware.command_log_marker()
        params = deepcopy(self.config["scans"]["set_z"])
        params.update(parameters or {})
        z = float(params["z"])
        targets = self._validated_mount_targets({"z": z})
        result = ScanResult(
            kind="set_z",
            recommendation={"z": z, "roll": targets["roll"], "passed": True},
            metadata={"parameters": params, "state_before_review": self._snapshot_metadata()},
        )
        result.metadata["command_log_start_sequence"] = command_log_start
        return result

    def apply(self, result: ScanResult) -> None:
        """Apply a reviewed recommendation and its complete associated DM state."""
        rec = result.recommendation
        if not bool(rec.get("passed", True)):
            raise ValueError("Recommendation failed its safeguard and cannot be applied")
        if result.kind == "bulk":
            self.hardware.move_conex({"u": rec["u"], "v": rec["v"]})
        elif result.kind == "set_z":
            self._move_mount({"z": rec["z"]})
            result.metadata["applied_fixed_roll"] = self.fixed_roll()
        elif result.kind == "pitch_yaw":
            self.hardware.set_dm(segment_state(self.config, "zero_ptt"))
            self._move_mount({"pitch": rec["pitch"], "yaw": rec["yaw"]})
            result.metadata["applied_dm_ptt_state"] = "zero_ptt"
        elif result.kind == "xy":
            self.hardware.set_dm(self._xy_dm_state())
            self._move_mount({"x": rec["x"], "y": rec["y"]})
            result.metadata["applied_dm_ptt_state"] = "selected_rest_of_dm_except_injected_zero_ptt"
            result.metadata["rest_of_dm"] = self.rest_dm_selection()
        elif result.kind == "tip_tilt":
            state = self._rest_dm_state()
            for key, values in rec["segments"].items():
                state[int(key)] = SegmentPTT(**values)
            self.hardware.set_dm(state)
            result.metadata["applied_dm_ptt_state"] = "selected_rest_of_dm_except_injected_recommended_tip_tilt"
            result.metadata["rest_of_dm"] = self.rest_dm_selection()
        elif result.kind == "null":
            current = self.hardware.get_dm()
            state = self._rest_dm_state()
            for segment in INJECTED_SEGMENTS:
                state[segment] = deepcopy(current[segment])
            for key, piston in rec["pistons_um"].items():
                state[int(key)].piston_um = float(piston)
            self.hardware.set_dm(state)
            result.metadata["applied_dm_ptt_state"] = "selected_rest_of_dm_except_injected_recommended_ptt"
            result.metadata["rest_of_dm"] = self.rest_dm_selection()
        else:
            raise ValueError(f"No apply action for {result.kind}")
        self._settle()
