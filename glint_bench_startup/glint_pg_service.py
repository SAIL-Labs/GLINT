#!/usr/bin/env python3
"""Interactive GLINT Point Grey camera controller used by glintpg_ctrl.

The service intentionally lives outside camstack. It preserves the installed
camstack frame-grabber command, initializes the tmux shell through ``~/.bashrc``,
and repairs FlyCapture bus-generation error 20 by rebuilding only the stale
control handle.
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
from pathlib import Path
import signal
import threading
import traceback
from typing import Any


LOG = logging.getLogger("glint_pg_service")
CAMERAS: list[Any] = []
SHUTTING_DOWN = False


def is_invalid_generation(exc: BaseException) -> bool:
    """Return whether a PyCapture2 exception represents Fc2 error code 20."""
    text = repr(exc).lower().replace(" ", "")
    if "fc2error(20)" in text or "fc2error:20" in text:
        return True
    for attr in ("errorType", "error_type", "errorCode", "error_code"):
        try:
            if int(getattr(exc, attr)) == 20:
                return True
        except (AttributeError, TypeError, ValueError):
            pass
    for value in getattr(exc, "args", ()):
        try:
            if int(value) == 20:
                return True
        except (TypeError, ValueError):
            continue
    return False


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_camera_class() -> type[Any]:
    # These imports occur only inside the configured pycapture interpreter.
    import PyCapture2 as PC2
    from camstack.cams.flycapturecam import Chameleon3

    class GlintChameleon3(Chameleon3):
        """Chameleon3 with native backend startup and error-20 recovery."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._glint_reconnect_lock = threading.Lock()
            super().__init__(*args, **kwargs)

        def _prepare_backend_cmdline(self, reuse_shm: bool = False) -> None:
            # Let the telescope's installed camstack version build the native
            # frame-grabber command.  Only prepend the shell initialization that
            # is required in the fresh tmux shell used by camstack.
            super()._prepare_backend_cmdline(reuse_shm=reuse_shm)
            self.taker_tmux_command = (
                f"source ~/.bashrc; {self.taker_tmux_command}"
            )

        def _reconnect_control_handle(self) -> None:
            with self._glint_reconnect_lock:
                old_camera = self.fly_cam
                try:
                    if old_camera is not None:
                        old_camera.disconnect()
                except Exception:
                    pass

                bus = PC2.BusManager()
                serials = [
                    bus.getCameraSerialNumberFromIndex(index)
                    for index in range(bus.getNumOfCameras())
                ]
                if self.fly_number in serials:
                    uid = bus.getCameraFromSerialNumber(self.fly_number)
                else:
                    uid = bus.getCameraFromIndex(self.fly_number)

                camera = PC2.Camera()
                camera.connect(uid)
                self.fly_bus = bus
                self.fly_cam = camera

        def _recover_error20(self, exc: BaseException) -> bool:
            if not is_invalid_generation(exc):
                return False
            try:
                self._reconnect_control_handle()
                print(
                    f"GLINT_FC2_20_RECOVERED camera={self.NAME} serial={self.fly_number}",
                    flush=True,
                )
                return True
            except Exception as recovery_exc:
                print(
                    "GLINT_FC2_20_RECOVERY_FAILED "
                    f"camera={self.NAME} serial={self.fly_number}: {recovery_exc!r}",
                    flush=True,
                )
                return False

        def _fill_keywords(self) -> None:
            # Error 20 may first appear while initial keywords are populated,
            # before the polling thread exists.
            try:
                super()._fill_keywords()
            except Exception as exc:
                if not self._recover_error20(exc):
                    raise
                super()._fill_keywords()

        def poll_camera_for_keywords(self) -> None:
            try:
                super().poll_camera_for_keywords()
            except Exception as exc:
                if not self._recover_error20(exc):
                    raise
                # Confirm that the replacement handle is immediately usable.
                super().poll_camera_for_keywords()

    return GlintChameleon3


def release_cameras() -> None:
    global SHUTTING_DOWN
    if SHUTTING_DOWN:
        return
    SHUTTING_DOWN = True
    for camera in reversed(CAMERAS):
        try:
            name = getattr(camera, "NAME", "unknown")
            print(f"GLINT_CAMERA_RELEASE_BEGIN camera={name}", flush=True)
            camera.release()
            print(f"GLINT_CAMERA_RELEASED camera={name}", flush=True)
        except Exception as exc:
            print(f"GLINT_CAMERA_RELEASE_FAILED camera={name}: {exc!r}", flush=True)
    CAMERAS.clear()


def signal_handler(signum: int, _frame: Any) -> None:
    print(f"GLINT_CAMERA_SERVICE_SIGNAL signal={signum}", flush=True)
    release_cameras()
    raise SystemExit(128 + signum)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config.expanduser().resolve())
    camera_config = config["camera"]
    Camera = build_camera_class()

    # Leave SIGINT to the interactive Python prompt: an operator pressing
    # Ctrl+C to cancel one command must not shut down both cameras.
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, signal_handler)
    atexit.register(release_cameras)

    global pg1, pg2
    print("GLINT_CAMERA_SERVICE_START", flush=True)
    pg1_spec = camera_config["cameras"]["pg1"]
    print(
        f"GLINT_CAMERA_INIT_BEGIN camera=pg1 serial={pg1_spec['serial']}", flush=True
    )
    pg1 = Camera(
        pg1_spec["name"],
        pg1_spec["stream"],
        pg1_spec["mode"],
        int(pg1_spec["serial"]),
    )
    CAMERAS.append(pg1)
    print("GLINT_CAMERA_READY camera=pg1", flush=True)

    pg2_spec = camera_config["cameras"]["pg2"]
    print(
        f"GLINT_CAMERA_INIT_BEGIN camera=pg2 serial={pg2_spec['serial']}", flush=True
    )
    pg2 = Camera(
        pg2_spec["name"],
        pg2_spec["stream"],
        pg2_spec["mode"],
        int(pg2_spec["serial"]),
    )
    CAMERAS.append(pg2)
    print("GLINT_CAMERA_READY camera=pg2", flush=True)
    print(
        "GLINT_CAMERA_SERVICE_READY — pg1 and pg2 remain available in this prompt",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        print("GLINT_CAMERA_SERVICE_FAILED", flush=True)
        traceback.print_exc()
        release_cameras()
        raise
