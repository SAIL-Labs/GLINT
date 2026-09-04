from __future__ import annotations

import importlib
import subprocess
import time
from pathlib import Path
from threading import RLock

import numpy as np

from .command_log import CommandLogger
from .hardware import AlignmentHardware
from .models import SegmentPTT, dm_state_from_array, dm_state_to_array, dm_state_to_dict
from .suruga_mount import SurugaMount


AXIS_NUMBER = {"pitch": 1, "roll": 2, "yaw": 3, "x": 4, "z": 5, "y": 6}


class InstrumentHardware(AlignmentHardware):
    """Adapter for the existing SHM streams, Suruga mount wrapper, and Conex CLI.

    Constructing this class attaches to the already-running DM/camera streams. It
    does not initialize the DM or Point Grey cameras.
    """

    mode = "instrument"

    def __init__(self, config: dict) -> None:
        self.config = config
        self.command_logger = CommandLogger(config, self.mode)
        self._camera_lock = RLock()
        self._mount_lock = RLock()
        self._mount_cache: dict[str, int] = {}
        self._conex_cache: dict[str, float] = {}
        SHM = importlib.import_module("pyMilk.interfacing.shm").SHM
        # Attach to the already-running DM service.  Publishing to these SHM
        # streams does not initialize or directly open the physical DM.
        dm_cfg = config["instrument"]["dm"]
        self.dmptt = SHM(str(dm_cfg["ptt_shm"]))
        self.dmvolt = SHM(str(dm_cfg["voltage_shm"]))
        self.limit_warnings: list[str] = []
        self.cameras = {}
        for camera, stream_name in config["instrument"]["shm"].items():
            try:
                self.cameras[camera] = SHM(stream_name)
            except Exception as error:
                self.cameras[camera] = None
                self.limit_warnings.append(f"Could not attach {camera} camera stream {stream_name!r}: {error}")
        self.apapane = self.cameras.get("apapane")
        self.focal = self.cameras.get("focal")
        if self.apapane is None or self.focal is None:
            raise RuntimeError("The APAPANE and focal SHM streams are required for alignment scans")
        mount_cfg = config["instrument"]["mount"]
        self.mount_device = SurugaMount(
            mount_cfg["port"],
            int(mount_cfg["baudrate"]),
            read_timeout_s=mount_cfg.get("read_timeout_s", 1.0),
            write_timeout_s=mount_cfg.get("write_timeout_s", 1.0),
        )
        self._mount_poll_interval_s = float(mount_cfg.get("poll_interval_s", 0.02))
        self._limits = self._query_limits()

    def focal_frames(self, nframes: int) -> np.ndarray:
        with self._camera_lock:
            return np.asarray(self.focal.multi_recv_data(int(nframes)), dtype=float)

    def apapane_frames(self, nframes: int) -> np.ndarray:
        with self._camera_lock:
            return np.asarray(self.apapane.multi_recv_data(int(nframes)), dtype=float)

    def camera_frames(self, camera: str, nframes: int = 1) -> np.ndarray:
        if camera not in {"palila", "apapane", "focal", "pupil"}:
            raise ValueError(f"Unknown camera {camera!r}")
        stream = self.cameras.get(camera)
        if stream is None:
            raise RuntimeError(f"The {camera} SHM stream is not available")
        with self._camera_lock:
            if camera == "apapane" or nframes > 1:
                return np.asarray(stream.multi_recv_data(int(nframes)), dtype=float)
            return np.asarray(stream.get_data(), dtype=float)[None, ...]

    def focal_reference(self) -> tuple[np.ndarray, np.ndarray]:
        try:
            from astropy.io import fits
        except ImportError as error:
            raise RuntimeError("astropy is required to read focal reference FITS files") from error
        reference = self.config["instrument"]["focal_reference"]
        return (
            np.asarray(fits.getdata(reference["bright"]), dtype=float),
            np.asarray(fits.getdata(reference["dark"]), dtype=float),
        )

    def get_mount(self) -> dict[str, float]:
        # One lock covers the complete six-axis read. This prevents the GUI from
        # interleaving status queries with an in-progress multi-axis move.
        with self._mount_lock:
            positions = {name: int(self.mount_device.get_pos(number)) for name, number in AXIS_NUMBER.items()}
            self._mount_cache.update(positions)
            return {name: float(value) for name, value in positions.items()}

    def move_mount(self, targets: dict[str, float], timeout_s: float) -> dict[str, float]:
        """Move only axes whose target differs from the last verified position.

        The former implementation re-commanded roll and both map axes at every
        raster point, polled every requested axis, then performed another full
        six-axis read in the engine.  The cache is populated only by controller
        readback or a successfully verified move; it is never updated merely
        because a command was sent.
        """
        integer_targets = {name: int(round(value)) for name, value in targets.items()}
        with self._mount_lock:
            try:
                # Unknown axes are read once. Subsequent scan points use only
                # readbacks already verified by this process.
                unknown = [name for name in integer_targets if name not in self._mount_cache]
                for name in unknown:
                    self._mount_cache[name] = int(self.mount_device.get_pos(AXIS_NUMBER[name]))
                changed = {
                    name: value
                    for name, value in integer_targets.items()
                    if self._mount_cache.get(name) != value
                }
                if not changed:
                    return {name: float(self._mount_cache[name]) for name in integer_targets}

                command = [
                    {
                        "method": "set_pos",
                        "axis_name": name,
                        "axis_number": AXIS_NUMBER[name],
                        "position": value,
                        "wire": f"AXI{AXIS_NUMBER[name]}:GOABS {value}",
                    }
                    for name, value in changed.items()
                ]
                for name, value in changed.items():
                    self.mount_device.set_pos(AXIS_NUMBER[name], value)

                deadline = time.monotonic() + timeout_s
                positions: dict[str, int] = {}
                while True:
                    moving = {
                        name: bool(self.mount_device.in_motion(AXIS_NUMBER[name]))
                        for name in changed
                    }
                    # Position queries are useful only after all commanded axes
                    # have stopped. Avoiding them while moving substantially cuts
                    # serial open/write/read/close transactions on the DS102.
                    if not any(moving.values()):
                        positions = {
                            name: int(self.mount_device.get_pos(AXIS_NUMBER[name]))
                            for name in changed
                        }
                        if all(positions[name] == value for name, value in changed.items()):
                            self._mount_cache.update(positions)
                            requested_readback = {
                                name: float(self._mount_cache[name]) for name in integer_targets
                            }
                            self.command_logger.record(
                                "mount",
                                "move_absolute",
                                command=command,
                                targets=changed,
                                readback=requested_readback,
                                details={
                                    "requested_targets": integer_targets,
                                    "unchanged_axes_skipped": sorted(set(integer_targets) - set(changed)),
                                },
                            )
                            return requested_readback
                    if time.monotonic() >= deadline:
                        self.mount_device.rstop()
                        for name in changed:
                            self._mount_cache.pop(name, None)
                        raise TimeoutError(
                            f"Mount did not reach {changed} within {timeout_s:.1f} s; readback={positions}"
                        )
                    time.sleep(self._mount_poll_interval_s)
            except BaseException as error:
                for name in integer_targets:
                    self._mount_cache.pop(name, None)
                self.command_logger.record(
                    "mount",
                    "move_absolute",
                    command={"interface": "SurugaMount.set_pos", "requested_targets": integer_targets},
                    targets=integer_targets,
                    outcome="failed",
                    error=str(error),
                )
                raise

    def _query_limits(self) -> dict[str, tuple[float, float]]:
        configured = self.config.get("mount_control", {}).get("limits", {})
        if all(name in configured for name in AXIS_NUMBER):
            limits = {name: tuple(map(float, configured[name])) for name in AXIS_NUMBER}
            for name, axis in AXIS_NUMBER.items():
                try:
                    ccw_enabled = bool(self.mount_device.lim_enabled(axis, "CCW"))
                    cw_enabled = bool(self.mount_device.lim_enabled(axis, "CW"))
                    if ccw_enabled and cw_enabled:
                        device_limits = sorted(
                            [
                                float(self.mount_device.get_lim(axis, "CCW")),
                                float(self.mount_device.get_lim(axis, "CW")),
                            ]
                        )
                        if not (
                            device_limits[0] <= limits[name][0]
                            and limits[name][1] <= device_limits[1]
                        ):
                            self.limit_warnings.append(
                                f"Configured {name} limits {limits[name]} extend beyond enabled device limits "
                                f"{tuple(device_limits)}; verify the controller before motion"
                            )
                    else:
                        self.limit_warnings.append(
                            f"{name} controller soft limit is disabled; using the configured safety limits {limits[name]}"
                        )
                except Exception as error:
                    self.limit_warnings.append(
                        f"Could not query {name} controller limits ({error}); using configured safety limits {limits[name]}"
                    )
            return limits

        # Backward-compatible fallback for older external configuration files.
        fallbacks = self.config["instrument"]["mount"].get("fallback_limits", {})
        limits: dict[str, tuple[float, float]] = {}
        for name, axis in AXIS_NUMBER.items():
            if name == "z":
                limits[name] = tuple(map(float, fallbacks.get("z", [1100, 2500])))
                continue
            try:
                ccw_enabled = bool(self.mount_device.lim_enabled(axis, "CCW"))
                cw_enabled = bool(self.mount_device.lim_enabled(axis, "CW"))
                if ccw_enabled and cw_enabled:
                    values = sorted(
                        [float(self.mount_device.get_lim(axis, "CCW")), float(self.mount_device.get_lim(axis, "CW"))]
                    )
                    limits[name] = (values[0], values[1])
                    continue
            except Exception as error:
                self.limit_warnings.append(f"Could not query {name} soft limits: {error}")
            fallback = fallbacks.get(name)
            if fallback is None:
                limits[name] = (-float("inf"), float("inf"))
                self.limit_warnings.append(f"{name} soft limit is disabled/unavailable; no fallback is configured")
            else:
                limits[name] = tuple(map(float, fallback))
                self.limit_warnings.append(f"{name} is using configured fallback limits {fallback}")
        return limits

    def mount_limits(self) -> dict[str, tuple[float, float]]:
        return dict(self._limits)

    def get_dm(self) -> dict[int, SegmentPTT]:
        return dm_state_from_array(self.dmptt.get_data())

    def set_dm(self, state: dict[int, SegmentPTT]) -> None:
        command = dm_state_to_array(state).reshape(111)
        try:
            self.dmptt.set_data(command)
            readback = np.asarray(self.dmptt.get_data(), dtype=float).reshape(111)
            if not np.allclose(readback, command, atol=1e-6):
                raise RuntimeError("DM PTT SHM readback does not match the published command")
            self.command_logger.record(
                "dm",
                "set_ptt",
                command={"interface": "dmptt.set_data", "shape": [111]},
                targets=dm_state_to_dict(state),
                readback={"matched": True},
            )
        except BaseException as error:
            self.command_logger.record(
                "dm",
                "set_ptt",
                command={"interface": "dmptt.set_data", "shape": [111]},
                targets=dm_state_to_dict(state),
                outcome="failed",
                error=str(error),
            )
            raise

    def zero_dm_volts(self) -> None:
        self._publish_dm_volts(
            np.zeros(111, dtype=np.float32),
            operation="zero_volts",
            command={"interface": "dmvolt.set_data", "shape": [111]},
        )

    def flatten_dm_volts(self) -> None:
        """Load and publish the calibrated flat-voltage vector named in the JSON config.

        This intentionally uses the old GUI's ``dmvolt.set_data`` interface but
        avoids the legacy ``DM.flatten()`` relative-path dependency. The flat
        calibration path is therefore visible and editable in ``instrument.json``.
        """
        definition = self.config.get("dm_voltage_states", {}).get("flat", {})
        configured_path = definition.get("calibration_file")
        if not configured_path:
            raise RuntimeError("dm_voltage_states.flat.calibration_file is not configured")
        path = Path(str(configured_path)).expanduser()
        if not path.is_absolute():
            path = Path(self.config["_root"]) / path
        try:
            values = np.asarray(np.loadtxt(path, dtype=np.float32), dtype=np.float32).reshape(-1)
        except Exception as error:
            raise RuntimeError(f"Could not load the configured DM flat file {path}: {error}") from error
        if values.shape != (111,) or not np.all(np.isfinite(values)):
            raise RuntimeError(f"Configured DM flat file must contain exactly 111 finite values: {path}")
        self._publish_dm_volts(
            values,
            operation="flat_volts",
            command={"interface": "dmvolt.set_data", "calibration_file": str(path)},
        )

    def get_dm_volts(self) -> np.ndarray:
        return np.asarray(self.dmvolt.get_data(), dtype=np.float32).reshape(111).copy()

    def set_dm_volts(self, voltages: np.ndarray) -> None:
        values = np.asarray(voltages, dtype=np.float32).reshape(111)
        self._publish_dm_volts(
            values,
            operation="set_volts",
            command={"interface": "dmvolt.set_data", "shape": [111]},
        )

    def _publish_dm_volts(self, values: np.ndarray, *, operation: str, command: dict) -> None:
        try:
            self.dmvolt.set_data(values)
            readback = np.asarray(self.dmvolt.get_data(), dtype=float).reshape(111)
            if not np.allclose(readback, values, atol=1e-6):
                raise RuntimeError("DM voltage SHM readback does not match the published command")
            self.command_logger.record(
                "dm",
                operation,
                command=command,
                targets=values,
                readback={"matched": True},
            )
        except BaseException as error:
            self.command_logger.record(
                "dm",
                operation,
                command=command,
                targets=values,
                outcome="failed",
                error=str(error),
            )
            raise

    def _steering2_executable(self) -> str:
        mirrors = self.config.get("steering_mirrors", {})
        controlled = mirrors.get("controlled_device", "glint_steering2")
        devices = mirrors.get("devices", {})
        if controlled != "glint_steering2":
            raise RuntimeError("Only glint_steering2 may be controlled by this alignment application")
        if bool(devices.get("glint_steering1", {}).get("command_enabled", False)):
            raise RuntimeError("glint_steering1 must remain command-disabled")
        executable = devices.get("glint_steering2", {}).get(
            "executable",
            self.config.get("conex", {}).get("executable", "glint_steering2"),
        )
        if Path(str(executable)).name != "glint_steering2":
            raise RuntimeError("The steering executable must be glint_steering2; steering mirror 1 is forbidden")
        return str(executable)

    def _conex_command(self, axis: str, action: str, value: float | None = None) -> str:
        executable = self._steering2_executable()
        command = [executable, axis, action]
        if value is not None:
            command.append(str(float(value)))
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
            output = completed.stdout.strip()
            self.command_logger.record(
                "glint_steering2",
                action,
                command=command,
                targets=None if value is None else {axis: float(value)},
                readback={"stdout": output},
                record_type="query" if action == "status" else "command",
            )
            return output
        except BaseException as error:
            self.command_logger.record(
                "glint_steering2",
                action,
                command=command,
                targets=None if value is None else {axis: float(value)},
                outcome="failed",
                record_type="query" if action == "status" else "command",
                error=str(error),
            )
            raise

    def get_conex(self) -> dict[str, float]:
        result = {}
        for axis in ("u", "v"):
            output = self._conex_command(axis, "status")
            try:
                result[axis] = float(output.split()[-1])
            except (IndexError, ValueError) as error:
                raise RuntimeError(f"Could not parse Conex {axis} status: {output!r}") from error
        self._conex_cache.update(result)
        return result

    def move_conex(self, targets: dict[str, float]) -> None:
        for axis in ("u", "v"):
            if axis in targets and not (
                axis in self._conex_cache
                and np.isclose(self._conex_cache[axis], float(targets[axis]), atol=1e-12)
            ):
                target = float(targets[axis])
                self._conex_command(axis, "goto", target)
                self._conex_cache[axis] = target

    def stop(self) -> None:
        try:
            with self._mount_lock:
                self.mount_device.rstop()
                self._mount_cache.clear()
            self.command_logger.record(
                "mount",
                "stop",
                command={"interface": "SurugaMount.rstop", "wire": "STOP 1"},
            )
        except Exception as error:
            self.command_logger.record(
                "mount",
                "stop",
                command={"interface": "SurugaMount.rstop", "wire": "STOP 1"},
                outcome="failed",
                error=str(error),
            )

    def close(self) -> None:
        try:
            self.mount_device.close()
        except Exception:
            pass
        self.command_logger.close()
