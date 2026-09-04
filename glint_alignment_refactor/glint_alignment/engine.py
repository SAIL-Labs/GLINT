from __future__ import annotations

import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np

from .analysis import (
    agreement,
    fit_elliptical_gaussian,
    fit_null_envelope,
    hotspot,
    solve_common_pistons,
    threshold_center_of_mass,
)
from .boxes import (
    BASELINE_TO_NULL,
    FRINGE_TRACKING_BOXES,
    INJECTED_SEGMENTS,
    NULL_BOXES,
    PHOT_BOXES,
    SEGMENT_TO_PHOT,
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

    def _settle(self) -> None:
        delay = float(self.config["safety"]["settle_s"])
        time.sleep(delay * float(self.config.get("time_scale", 1.0)))
        self._check_abort()

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
        result.metadata["command_log_start_sequence"] = command_log_start
        return result

    def _guarded(
        self,
        operation: Callable[[], ScanResult],
        *,
        keep_dm_ptt_on_success: bool = False,
        keep_dm_volts_on_success: bool = False,
    ) -> ScanResult:
        command_log_start = self.hardware.command_log_marker()
        self.reset_abort()
        # Every scan snapshot and restoration uses the same configured roll.
        self.ensure_fixed_roll()
        before = self.hardware.snapshot()
        try:
            result = operation()
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
                    self._settle()
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
                metadata={"dm_voltage_state": "zero_volts"},
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
                self._settle()
                frame = self._average_frame(self.hardware.focal_frames(int(bulk["nframes"]))) - self.focal_dark
                positions.append(hotspot(frame, bulk["crop"], bulk["centroid_threshold"]))
            response[:, axis] = (positions[1] - positions[0]) / (2 * perturb[axis])
        self.hardware.move_conex({"u": theta[0], "v": theta[1]})
        self._settle()
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
        for iy, y in enumerate(yv):
            for ix, x in enumerate(xv):
                self._move_mount({x_axis: x, y_axis: y})
                self._settle()
                readback = self.hardware.get_mount()
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
            metadata={"parameters": params},
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
            for segment in INJECTED_SEGMENTS:
                state = deepcopy(away)
                state[segment] = deepcopy(zero[segment])
                self.hardware.set_dm(state)
                data = np.zeros((len(tilts), len(tips)), dtype=float)
                for iy, tilt in enumerate(tilts):
                    for ix, tip in enumerate(tips):
                        state[segment] = SegmentPTT(0.0, float(tip), float(tilt))
                        self.hardware.set_dm(state)
                        self._settle()
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
                metadata={"parameters": params, "rest_of_dm": self.rest_dm_selection()},
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
            ascending = AxisGrid(**params["piston"]).values()
            descending = ascending[-2::-1]
            baselines = [(11, 20), (11, 31)]
            maps: dict[str, np.ndarray] = {}
            estimates: dict[str, dict] = {}
            total = len(baselines) * int(params["numavg"]) * (len(ascending) + len(descending))
            progress_count = 0
            current = self.hardware.get_dm()
            away = self._scan_away_state()
            for baseline in baselines:
                a, b = baseline
                null_name = BASELINE_TO_NULL[baseline]
                deltas: list[float] = []
                repetitions: list[list[float]] = []
                for repeat in range(int(params["numavg"])):
                    metrics: list[float] = []
                    repeat_deltas: list[float] = []
                    state = deepcopy(away)
                    for seg in baseline:
                        state[seg] = deepcopy(current[seg])
                    state[a].piston_um = minimum
                    state[b].piston_um = maximum
                    for piston_a in ascending:
                        state[a].piston_um = float(piston_a)
                        self.hardware.set_dm(state)
                        self._settle()
                        metrics.append(self._null_metric(null_name, int(params["nframes"])))
                        repeat_deltas.append(float(state[a].piston_um - state[b].piston_um))
                        progress_count += 1
                        self.progress(f"Null {a}-{b}, repeat {repeat + 1}", progress_count, total)
                    for piston_b in descending:
                        state[b].piston_um = float(piston_b)
                        self.hardware.set_dm(state)
                        self._settle()
                        metrics.append(self._null_metric(null_name, int(params["nframes"])))
                        repeat_deltas.append(float(state[a].piston_um - state[b].piston_um))
                        progress_count += 1
                        self.progress(f"Null {a}-{b}, repeat {repeat + 1}", progress_count, total)
                    if not deltas:
                        deltas = repeat_deltas
                    repetitions.append(metrics)
                mean_metric = np.mean(repetitions, axis=0)
                opd = 2.0 * np.asarray(deltas)
                fit = fit_null_envelope(opd, mean_metric)
                key = f"{a}-{b}"
                maps[key] = np.asarray(repetitions)
                estimates[key] = fit
                estimates[key]["opd_um"] = opd
                estimates[key]["mean_metric"] = mean_metric
            solution = solve_common_pistons(
                estimates["11-20"]["candidates"],
                estimates["11-31"]["candidates"],
                minimum,
                maximum,
            )
            return ScanResult(
                kind="null",
                axes={"delta_piston_um": np.asarray(deltas), "opd_um": 2 * np.asarray(deltas)},
                maps=maps,
                estimates=estimates,
                recommendation={**solution, "passed": bool(solution["feasible"])},
                metadata={
                    "parameters": params,
                    "baseline_20_31": "disabled",
                    "rest_of_dm": self.rest_dm_selection(),
                },
            )

        return self._guarded(operation)

    def _null_metric(self, null_name: str, nframes: int) -> float:
        frame = self._average_frame(self.hardware.apapane_frames(nframes)) - self.apapane_dark
        null_flux = integrate_box(frame, NULL_BOXES[null_name])
        ft = FRINGE_TRACKING_BOXES[null_name]
        denominator = integrate_box(frame, ft["FT1"]) + integrate_box(frame, ft["FT2"])
        return float(null_flux / max(denominator, 1e-12))

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

    def _move_mount(self, targets: dict[str, float]) -> None:
        targets = self._validated_mount_targets(targets)
        self.hardware.move_mount(targets, float(self.config["safety"]["motion_timeout_s"]))

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
