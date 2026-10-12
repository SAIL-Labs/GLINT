from __future__ import annotations

from copy import deepcopy
from threading import RLock

import numpy as np

from .boxes import BASELINE_TO_NULL, FRINGE_TRACKING_BOXES, NULL_BOXES, PHOT_BOXES, differential_null_boxes
from .command_log import CommandLogger
from .config import segment_state
from .hardware import AlignmentHardware
from .models import SegmentPTT, dm_state_to_dict, zero_dm_state


class SimulatedHardware(AlignmentHardware):
    """Deterministic synthetic instrument for workflow and UI testing."""

    mode = "simulation"

    def __init__(self, config: dict) -> None:
        self.config = config
        self.command_logger = CommandLogger(config, self.mode)
        self.rng = np.random.default_rng(int(config.get("simulation", {}).get("seed", 4217)))
        self.lock = RLock()
        self.camera_lock = RLock()
        self.dark_mode = False
        self.mount = {
            "x": 3038.0,
            "y": 3442.0,
            "z": 1800.0,
            "pitch": 300.0,
            "yaw": -100.0,
            "roll": float(config.get("mount_control", {}).get("fixed_roll", 0.0)),
        }
        configured_limits = config.get("mount_control", {}).get("limits", {})
        self.limits = {
            axis: tuple(map(float, configured_limits[axis]))
            for axis in ("pitch", "roll", "yaw", "x", "z", "y")
        }
        initial_state = config.get("simulation", {}).get("initial_dm_state", "zero_ptt")
        self.dm = segment_state(config, initial_state) if initial_state else zero_dm_state()
        self.dm_volts = np.full(111, 0.25, dtype=np.float32)
        self.conex = {"u": 0.0, "v": 0.0}
        self.conex_optimum = np.asarray([0.08, -0.04])
        self.focal_response = np.asarray([[-243.065, -1752.007], [1867.755, -134.803]])
        self.goal_xy = np.asarray([545.0, 525.0])
        self.py_centers = {
            11: (535.0, 145.0),
            20: (515.0, 160.0),
            31: (500.0, 135.0),
        }
        self.xy_centers = {
            11: (3037.0, 3442.0),
            20: (3045.0, 3435.0),
            31: (3032.0, 3448.0),
        }
        self.tt_centers = {
            11: (0.20, 0.02),
            20: (-0.02, -0.10),
            31: (-0.09, 0.24),
        }
        self.null_deltas = {(11, 20): 0.32, (11, 31): -0.46}
        self.active_null_scan_baseline: tuple[int, int] | None = None
        self.active_null_scan_box: list[int] | None = None
        self.noise = float(config.get("simulation", {}).get("noise", 1.0))

    def set_dark_mode(self, enabled: bool) -> None:
        self.dark_mode = bool(enabled)

    def _stack(self, frame: np.ndarray, nframes: int) -> np.ndarray:
        # A small independent scalar fluctuation is enough to test averaging without
        # allocating thousands of full random detector arrays during a null scan.
        count = 1 if self.config.get("simulation", {}).get("fast_average", True) else nframes
        scales = 1.0 + self.rng.normal(0.0, 0.001 * self.noise / np.sqrt(max(nframes, 1)), size=(count, 1, 1))
        return (frame[None, :, :] * scales).astype(np.float32)

    def focal_frames(self, nframes: int) -> np.ndarray:
        with self.camera_lock:
            return self._focal_frames_unlocked(nframes)

    def _focal_frames_unlocked(self, nframes: int) -> np.ndarray:
        yy, xx = np.indices((850, 850))
        dark = 80.0 + 0.01 * xx + 0.015 * yy
        if self.dark_mode:
            frame = dark + self.rng.normal(0, self.noise, dark.shape)
        else:
            theta = np.asarray([self.conex["u"], self.conex["v"]])
            center = self.goal_xy + self.focal_response @ (theta - self.conex_optimum)
            psf = 18000 * np.exp(-0.5 * (((xx - center[0]) / 7.0) ** 2 + ((yy - center[1]) / 9.0) ** 2))
            frame = dark + psf + self.rng.normal(0, self.noise, dark.shape)
        return self._stack(frame, nframes)

    def focal_reference(self) -> tuple[np.ndarray, np.ndarray]:
        yy, xx = np.indices((850, 850))
        dark = 80.0 + 0.01 * xx + 0.015 * yy
        bright = dark + 18000 * np.exp(
            -0.5 * (((xx - self.goal_xy[0]) / 7.0) ** 2 + ((yy - self.goal_xy[1]) / 9.0) ** 2)
        )
        return bright.astype(np.float32), dark.astype(np.float32)

    def palila_frames(self, nframes: int) -> np.ndarray:
        yy, xx = np.indices((320, 320))
        radius = np.hypot(xx - 160.0, yy - 160.0)
        airy_like = 12000 * np.sinc(radius / 8.0) ** 2
        frame = 55.0 + airy_like + self.rng.normal(0, self.noise, radius.shape)
        return self._stack(frame, nframes)

    def pupil_frames(self, nframes: int) -> np.ndarray:
        yy, xx = np.indices((420, 420))
        dx, dy = xx - 210.0, yy - 210.0
        radius = np.hypot(dx, dy)
        pupil = (radius <= 160) & (radius >= 48)
        spiders = (np.abs(dx) < 4) | (np.abs(dy) < 4)
        frame = 70.0 + 5500.0 * (pupil & ~spiders)
        frame += 350.0 * np.cos(dx / 13.0) * np.cos(dy / 15.0) * pupil
        frame += self.rng.normal(0, self.noise, frame.shape)
        return self._stack(frame, nframes)

    def camera_frames(self, camera: str, nframes: int = 1) -> np.ndarray:
        readers = {
            "palila": self.palila_frames,
            "apapane": self.apapane_frames,
            "focal": self.focal_frames,
            "pupil": self.pupil_frames,
        }
        with self.camera_lock:
            try:
                frames = readers[camera](nframes)
            except KeyError as error:
                raise ValueError(f"Unknown simulated camera {camera!r}") from error
            # Scan averaging may use the fast one-frame shortcut, but an operator's
            # explicit camera capture must produce the requested cube depth.
            if frames.shape[0] == 1 and nframes > 1:
                scales = 1.0 + self.rng.normal(0.0, 0.001 * self.noise, size=(nframes, 1, 1))
                frames = (frames[0][None, :, :] * scales).astype(np.float32)
            return frames

    @staticmethod
    def _hex_response(x: float, y: float, cx: float, cy: float) -> float:
        dx, dy = (x - cx) / 36.0, (y - cy) / 36.0
        radius = max(abs(dx), abs(0.5 * dx + 0.8660254 * dy), abs(-0.5 * dx + 0.8660254 * dy))
        return float(np.exp(-(radius**4)))

    def _phot_flux(self, segment: int) -> float:
        px, py = self.py_centers[segment]
        sx, sy = (175.0, 145.0) if segment != 20 else (150.0, 185.0)
        py_gain = np.exp(-0.5 * (((self.mount["pitch"] - px) / sx) ** 2 + ((self.mount["yaw"] - py) / sy) ** 2))
        cx, cy = self.xy_centers[segment]
        xy_gain = self._hex_response(self.mount["x"], self.mount["y"], cx, cy)
        tx, ty = self.tt_centers[segment]
        ptt = self.dm[segment]
        tt_gain = np.exp(-0.5 * (((ptt.tip_mrad - tx) / 0.38) ** 2 + ((ptt.tilt_mrad - ty) / 0.32) ** 2))
        if abs(ptt.tip_mrad) >= 3 or abs(ptt.tilt_mrad) >= 3:
            tt_gain *= 0.002
        return float(130000 * py_gain * xy_gain * tt_gain + 20)

    def _put_box(self, frame: np.ndarray, box: list[int], total_flux: float) -> None:
        y0, y1, x0, x1 = box
        width = x1 - x0
        spectrum = 0.75 + 0.25 * np.cos(np.linspace(-np.pi, np.pi, width))
        spectrum /= spectrum.sum()
        frame[y0:y1, x0:x1] += total_flux * spectrum[None, :] / (y1 - y0)

    def apapane_frames(self, nframes: int) -> np.ndarray:
        with self.camera_lock:
            return self._apapane_frames_unlocked(nframes)

    def set_null_scan_baseline(self, baseline: tuple[int, int] | None) -> None:
        """Select the telescope-validated raw-flux box for a simulated scan."""
        self.active_null_scan_baseline = None if baseline is None else tuple(map(int, baseline))
        self.active_null_scan_box = (
            None
            if self.active_null_scan_baseline is None
            else differential_null_boxes()[self.active_null_scan_baseline]
        )

    def _apapane_frames_unlocked(self, nframes: int) -> np.ndarray:
        yy, xx = np.indices((280, 280))
        dark = 95.0 + 0.012 * xx + 0.008 * yy
        frame = dark.copy()
        if not self.dark_mode:
            for segment, name in zip((11, 20, 31), PHOT_BOXES):
                self._put_box(frame, PHOT_BOXES[name], self._phot_flux(segment))
            if self.active_null_scan_baseline is not None:
                baseline = self.active_null_scan_baseline
                a, b = baseline
                delta = self.dm[a].piston_um - self.dm[b].piston_um
                opd = 2 * delta
                target = 2 * self.null_deltas[baseline]
                period = 1.55 if baseline == (11, 20) else 1.65
                coherence_width = 5.6 if baseline == (11, 20) else 6.2
                envelope = 0.65 + 0.35 * np.sinc((opd - target) / coherence_width)
                flux = 60000.0 - 50000.0 * envelope * np.cos(2 * np.pi * (opd - target) / period)
                assert self.active_null_scan_box is not None
                self._put_box(frame, self.active_null_scan_box, flux)
            else:
                # Keep the established dashboard/null-live simulation when a
                # differential scan is not in progress.
                for baseline, name in BASELINE_TO_NULL.items():
                    a, b = baseline
                    delta = self.dm[a].piston_um - self.dm[b].piston_um
                    opd = 2 * delta
                    target = 2 * self.null_deltas[baseline]
                    period = 1.55 if baseline == (11, 20) else 1.65
                    coherence_width = 5.6 if baseline == (11, 20) else 6.2
                    envelope = np.sinc((opd - target) / coherence_width)
                    metric = 0.16 + 0.115 * envelope * np.sin(
                        2 * np.pi * (opd - target) / period - np.pi / 2
                    )
                    ft_total_each = 8000.0
                    self._put_box(frame, FRINGE_TRACKING_BOXES[name]["FT1"], ft_total_each)
                    self._put_box(frame, FRINGE_TRACKING_BOXES[name]["FT2"], ft_total_each)
                    self._put_box(frame, NULL_BOXES[name], metric * 2 * ft_total_each)
            frame += self.rng.normal(0, self.noise, frame.shape)
        else:
            frame += self.rng.normal(0, self.noise, frame.shape)
        return self._stack(frame, nframes)

    def get_mount(self) -> dict[str, float]:
        with self.lock:
            return dict(self.mount)

    def move_mount(self, targets: dict[str, float], timeout_s: float) -> dict[str, float]:
        del timeout_s
        try:
            with self.lock:
                for axis, value in targets.items():
                    low, high = self.limits[axis]
                    if not low <= float(value) <= high:
                        raise ValueError(f"Simulated {axis} limit violation")
                changed = {
                    axis: float(value)
                    for axis, value in targets.items()
                    if not np.isclose(self.mount[axis], float(value), atol=1e-12)
                }
                for axis, value in changed.items():
                    self.mount[axis] = value
                readback = {axis: float(self.mount[axis]) for axis in targets}
            if changed:
                self.command_logger.record(
                    "mount",
                    "move_absolute",
                    command=[
                        {
                            "method": "set_pos",
                            "axis_name": axis,
                            "axis_number": {"pitch": 1, "roll": 2, "yaw": 3, "x": 4, "z": 5, "y": 6}[axis],
                            "position": float(value),
                        }
                        for axis, value in changed.items()
                    ],
                    targets=changed,
                    readback=readback,
                    details={
                        "requested_targets": targets,
                        "unchanged_axes_skipped": sorted(set(targets) - set(changed)),
                    },
                )
            return readback
        except BaseException as error:
            self.command_logger.record(
                "mount",
                "move_absolute",
                command={"simulation": True},
                targets=targets,
                outcome="failed",
                error=str(error),
            )
            raise

    def mount_limits(self) -> dict[str, tuple[float, float]]:
        return dict(self.limits)

    def get_dm(self) -> dict[int, SegmentPTT]:
        with self.lock:
            return deepcopy(self.dm)

    def set_dm(self, state: dict[int, SegmentPTT]) -> None:
        if sorted(state) != list(range(37)):
            raise ValueError("DM state must contain segments 0..36")
        with self.lock:
            self.dm = deepcopy(state)
        self.command_logger.record(
            "dm",
            "set_ptt",
            command={"interface": "simulated_dmptt.set_data", "shape": [111]},
            targets=dm_state_to_dict(state),
            readback={"matched": True},
        )

    def zero_dm_volts(self) -> None:
        self.dm_volts[:] = 0.0
        self.command_logger.record(
            "dm",
            "zero_volts",
            command={"interface": "simulated_DM.sendzeros"},
            targets={"all_actuators_volts": 0.0},
        )

    def flatten_dm_volts(self) -> None:
        definition = self.config.get("dm_voltage_states", {}).get("flat", {})
        voltage = float(definition.get("all_actuators_volts", 0.25))
        self.dm_volts[:] = voltage
        self.command_logger.record(
            "dm",
            "flat_volts",
            command={"interface": "simulated_dmvolt.set_data", "shape": [111]},
            targets={"all_actuators_volts": voltage},
            readback={"matched": True},
        )

    def get_dm_volts(self) -> np.ndarray:
        return self.dm_volts.copy()

    def set_dm_volts(self, voltages: np.ndarray) -> None:
        values = np.asarray(voltages, dtype=np.float32).reshape(111)
        self.dm_volts = values.copy()
        self.command_logger.record(
            "dm",
            "set_volts",
            command={"interface": "simulated_dmvolt.set_data", "shape": [111]},
            targets=values,
            readback={"matched": True},
        )

    def get_conex(self) -> dict[str, float]:
        return dict(self.conex)

    def move_conex(self, targets: dict[str, float]) -> None:
        for axis in ("u", "v"):
            if axis in targets:
                self.conex[axis] = float(targets[axis])
                self.command_logger.record(
                    "glint_steering2",
                    "goto",
                    command=["glint_steering2", axis, "goto", str(float(targets[axis]))],
                    targets={axis: float(targets[axis])},
                    readback={axis: self.conex[axis]},
                )

    def stop(self) -> None:
        self.command_logger.record(
            "mount",
            "stop",
            command={"interface": "simulated_Mount.rstop"},
        )

    def close(self) -> None:
        self.command_logger.close()
