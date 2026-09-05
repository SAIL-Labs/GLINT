#!/usr/bin/env python3
"""Restore the GLINT bench services and desktop after an scexao5 reboot.

This program is deliberately independent of the alignment GUI.  It owns only
the named tmux sessions in ``bench_startup.json`` and never commands a DM
shape, mount axis, or steering mirror.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


LOG = logging.getLogger("glint_bench_startup")
SESSION_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class StartupError(RuntimeError):
    """A failure that should stop automatic bench restoration."""


@dataclass(frozen=True)
class Window:
    window_id: str
    pid: int
    x: int
    y: int
    width: int
    height: int
    title: str


SHM_PROBE = r"""
import sys
import time
import numpy as np
from pyMilk.interfacing.shm import SHM

name, mode, timeout_text, expected_size_text = sys.argv[1:]
timeout = float(timeout_text)
expected_size = int(expected_size_text)
shm = SHM(name)
data = shm.get_data(check=False)
if mode == "live":
    first_acqtime = shm.IMAGE.md.acqtime
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(0.05)
        if shm.IMAGE.md.acqtime != first_acqtime:
            break
    else:
        raise TimeoutError(f"{name}: acquisition timestamp did not advance")
if expected_size >= 0 and np.asarray(data).size != expected_size:
    raise RuntimeError(
        f"{name}: expected {expected_size} elements, got {np.asarray(data).size}"
    )
print(tuple(np.asarray(data).shape))
"""


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    for key in ("host", "paths", "camera", "dm", "gui", "shm", "windows"):
        if key not in config:
            raise StartupError(f"Missing required configuration section: {key}")
    return config


def configured_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def setup_logging(log_path: Path, *, file_logging: bool = True) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if file_logging:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


def executable(value: str) -> str | None:
    if "/" in value:
        path = Path(value).expanduser()
        return str(path) if path.is_file() and os.access(path, os.X_OK) else None
    return shutil.which(value)


class Runner:
    def __init__(self, environment: dict[str, str]) -> None:
        self.environment = os.environ.copy()
        self.environment.update({str(k): str(v) for k, v in environment.items()})

    def run(
        self,
        args: Sequence[str],
        *,
        timeout: float | None = None,
        check: bool = False,
        quiet: bool = False,
        label: str | None = None,
        log_command: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        if log_command:
            LOG.info("$ %s", label or shlex.join(map(str, args)))
        result = subprocess.run(
            list(map(str, args)),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env=self.environment,
            check=False,
        )
        if not quiet and result.stdout.strip():
            LOG.info("stdout: %s", result.stdout.strip())
        if not quiet and result.returncode and result.stderr.strip():
            LOG.warning("stderr: %s", result.stderr.strip())
        if check and result.returncode:
            raise StartupError(
                f"Command failed ({result.returncode}): {label or shlex.join(args)}"
            )
        return result

    def popen(
        self,
        args: Sequence[str],
        *,
        stdout: Any = subprocess.DEVNULL,
        stderr: Any = subprocess.DEVNULL,
    ) -> subprocess.Popen[Any]:
        LOG.info("$ %s", shlex.join(map(str, args)))
        environment = self.environment.copy()
        # An xterm opened by a script launched inside tmux must not inherit the
        # parent tmux client identity.
        environment.pop("TMUX", None)
        return subprocess.Popen(
            list(map(str, args)),
            env=environment,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )


class Tmux:
    def __init__(self, runner: Runner, binary: str) -> None:
        self.runner = runner
        self.binary = binary

    @staticmethod
    def validate_name(name: str) -> None:
        if not SESSION_RE.fullmatch(name):
            raise StartupError(f"Unsafe tmux session name: {name!r}")

    def exists(self, name: str) -> bool:
        self.validate_name(name)
        result = self.runner.run(
            [self.binary, "has-session", "-t", name], quiet=True, log_command=False
        )
        return result.returncode == 0

    def create(self, name: str, command: Sequence[str]) -> None:
        self.validate_name(name)
        if self.exists(name):
            raise StartupError(f"Refusing to overwrite existing tmux session {name}")
        # tmux new-session accepts one shell-command string. shlex.join keeps
        # every path/argument intact while allowing tmux to invoke it through
        # the session's normal shell.
        shell_command = shlex.join(map(str, command))
        self.runner.run(
            [self.binary, "new-session", "-d", "-s", name, shell_command],
            check=True,
        )
        self.runner.run(
            [self.binary, "set-option", "-t", name, "remain-on-exit", "on"],
            quiet=True,
        )

    def kill(self, name: str) -> None:
        self.validate_name(name)
        if self.exists(name):
            LOG.warning("Killing owned tmux session: %s", name)
            self.runner.run(
                [self.binary, "kill-session", "-t", name], check=True, quiet=True
            )

    def capture(self, name: str, lines: int = 100) -> str:
        self.validate_name(name)
        if not self.exists(name):
            return ""
        result = self.runner.run(
            [self.binary, "capture-pane", "-p", "-S", f"-{int(lines)}", "-t", name],
            quiet=True,
        )
        return result.stdout

    def foreground(self, name: str) -> tuple[bool, str]:
        """Return (pane_alive, foreground command) for a one-pane session."""
        self.validate_name(name)
        if not self.exists(name):
            return False, ""
        result = self.runner.run(
            [
                self.binary,
                "list-panes",
                "-t",
                name,
                "-F",
                "#{pane_dead}|#{pane_current_command}",
            ],
            quiet=True,
            label=f"inspect tmux foreground: {name}",
        )
        if result.returncode or not result.stdout.strip():
            return False, ""
        dead_text, _, command = result.stdout.strip().splitlines()[0].partition("|")
        return dead_text != "1", command.strip()

    def running_expected(self, name: str, tokens: Iterable[str]) -> bool:
        alive, command = self.foreground(name)
        command_cf = command.casefold()
        return alive and any(str(token).casefold() in command_cf for token in tokens)


class WindowManager:
    def __init__(self, config: dict[str, Any], runner: Runner) -> None:
        self.config = config
        self.runner = runner
        self.wmctrl = config["paths"]["wmctrl"]
        self.xterm = config["paths"]["xterm"]

    def list_windows(self) -> list[Window]:
        result = self.runner.run(
            [self.wmctrl, "-l", "-p", "-G"],
            quiet=True,
            label="wmctrl: list windows",
            log_command=False,
        )
        windows: list[Window] = []
        for line in result.stdout.splitlines():
            parts = line.split(maxsplit=8)
            if len(parts) < 9:
                continue
            try:
                windows.append(
                    Window(
                        window_id=parts[0],
                        pid=int(parts[2]),
                        x=int(parts[3]),
                        y=int(parts[4]),
                        width=int(parts[5]),
                        height=int(parts[6]),
                        title=parts[8],
                    )
                )
            except ValueError:
                continue
        return windows

    def find(self, title_contains: str) -> Window | None:
        needle = title_contains.casefold()
        candidates = [w for w in self.list_windows() if needle in w.title.casefold()]
        return candidates[-1] if candidates else None

    def wait_for(self, title_contains: str, timeout: float) -> Window | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            found = self.find(title_contains)
            if found is not None:
                return found
            time.sleep(0.25)
        return None

    def wait_until_absent(self, title_contains: str, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.find(title_contains) is None:
                return True
            time.sleep(0.1)
        return self.find(title_contains) is None

    def place(self, logical_name: str, title_contains: str) -> Window | None:
        spec = self.config["windows"]["layout"][logical_name]
        window = self.wait_for(title_contains, self.config["windows"]["window_timeout_s"])
        if window is None:
            LOG.warning("Could not find window %s (%s)", logical_name, title_contains)
            return None
        x, y, width, height = map(int, spec["rect"])
        self.runner.run(
            [
                self.wmctrl,
                "-i",
                "-r",
                window.window_id,
                "-b",
                "remove,maximized_vert,maximized_horz",
            ],
            quiet=True,
            label=f"unmaximize {logical_name}",
        )
        self.runner.run(
            [
                self.wmctrl,
                "-i",
                "-r",
                window.window_id,
                "-b",
                "remove,fullscreen",
            ],
            quiet=True,
            label=f"remove fullscreen from {logical_name}",
        )
        result = self.runner.run(
            [
                self.wmctrl,
                "-i",
                "-r",
                window.window_id,
                "-e",
                f"0,{x},{y},{width},{height}",
            ],
            quiet=True,
            label=f"place {logical_name} at {x},{y} {width}x{height}",
        )
        if result.returncode:
            LOG.warning("Failed to place %s", logical_name)
        return window

    def close(self, window: Window, logical_name: str) -> None:
        self.runner.run(
            [self.wmctrl, "-i", "-c", window.window_id],
            quiet=True,
            label=f"close stale {logical_name}",
        )

    def raise_window(self, window: Window | None, logical_name: str) -> None:
        if window is None:
            return
        self.runner.run(
            [self.wmctrl, "-i", "-a", window.window_id],
            quiet=True,
            label=f"raise {logical_name}",
        )

    def ensure_xterm(
        self, logical_name: str, title: str, command: Sequence[str]
    ) -> Window | None:
        existing = self.find(title)
        if existing is None:
            font = self.config["windows"].get("xterm_font", "Monospace")
            font_size = str(self.config["windows"].get("xterm_font_size", 9))
            self.runner.popen(
                [
                    self.xterm,
                    "-T",
                    title,
                    "-fa",
                    font,
                    "-fs",
                    font_size,
                    "-xrm",
                    "XTerm*allowTitleOps: false",
                    "-e",
                    *command,
                ]
            )
        return self.place(logical_name, title)


class BenchRestorer:
    def __init__(
        self,
        config: dict[str, Any],
        config_path: Path,
        *,
        no_windows: bool,
        no_gui: bool,
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.no_windows = no_windows
        self.no_gui = no_gui
        self.runner = Runner(config.get("environment", {}))
        self.tmux = Tmux(self.runner, config["paths"]["tmux"])
        self.windows = WindowManager(config, self.runner)

    @property
    def camera_sessions(self) -> list[str]:
        camera = self.config["camera"]
        return [
            camera["controller_session"],
            camera["cameras"]["pg1"]["framegrab_session"],
            camera["cameras"]["pg2"]["framegrab_session"],
        ]

    def preflight(self) -> None:
        expected = str(self.config["host"])
        actual = socket.gethostname().split(".", 1)[0]
        if actual != expected:
            raise StartupError(f"This configuration is for {expected}, not {actual}")

        required = {
            "tmux": self.config["paths"]["tmux"],
            "bash": self.config["paths"]["bash"],
            "probe Python": self.config["paths"]["probe_python"],
            "pycapture Python": self.config["paths"]["pycapture_python"],
            "GUI Python": self.config["paths"]["gui_python"],
            "milk-makecsetandrt": self.config["paths"]["milk_makecsetandrt"],
            "hwint-bmc111": self.config["paths"]["dm_driver"],
        }
        if not self.no_windows:
            required.update(
                {
                    "wmctrl": self.config["paths"]["wmctrl"],
                    "xterm": self.config["paths"]["xterm"],
                    "shmImshow.py": self.config["paths"]["shm_viewer"],
                }
            )
        missing = [label for label, path in required.items() if executable(path) is None]
        if missing:
            raise StartupError("Missing executables: " + ", ".join(missing))

        service = configured_path(self.config["camera"]["service_script"], self.config_path)
        if not service.is_file():
            raise StartupError(f"Camera service script is missing: {service}")

        if not self.no_gui:
            workdir = Path(self.config["gui"]["working_directory"]).expanduser()
            script = workdir / self.config["gui"]["script"]
            gui_config = Path(self.config["gui"]["config"]).expanduser()
            if not workdir.is_dir():
                raise StartupError(f"GUI working directory is missing: {workdir}")
            if not script.is_file():
                raise StartupError(f"GUI launcher is missing: {script}")
            if not gui_config.is_file():
                raise StartupError(f"GUI configuration is missing: {gui_config}")

        self.validate_layout()
        LOG.info("Preflight passed on %s", actual)

    def validate_layout(self) -> None:
        screen_width, screen_height = map(int, self.config["windows"]["screen_size"])
        for name, spec in self.config["windows"]["layout"].items():
            rect = spec.get("rect", [])
            if len(rect) != 4:
                raise StartupError(f"Window {name} must have [x, y, width, height]")
            x, y, width, height = map(int, rect)
            if min(x, y) < 0 or min(width, height) <= 0:
                raise StartupError(f"Invalid rectangle for {name}: {rect}")
            if x + width > screen_width or y + height > screen_height:
                raise StartupError(f"Window {name} lies outside the configured screen")

    def probe_shm(
        self,
        name: str,
        *,
        live: bool,
        timeout: float,
        expected_size: int = -1,
    ) -> bool:
        mode = "live" if live else "read"
        result = self.runner.run(
            [
                self.config["paths"]["probe_python"],
                "-c",
                SHM_PROBE,
                name,
                mode,
                str(timeout),
                str(expected_size),
            ],
            timeout=max(2.0, 2.0 * timeout + 2.0),
            quiet=True,
            label=f"probe SHM {name} ({mode})",
            log_command=False,
        )
        return result.returncode == 0

    def wait_for_shms(
        self,
        names: Iterable[str],
        *,
        live: bool,
        timeout: float,
        expected_size: int = -1,
    ) -> bool:
        names = list(names)
        deadline = time.monotonic() + timeout
        last_missing = names
        while time.monotonic() < deadline:
            last_missing = [
                name
                for name in names
                if not self.probe_shm(
                    name,
                    live=live,
                    timeout=min(1.5, max(0.2, deadline - time.monotonic())),
                    expected_size=expected_size,
                )
            ]
            if not last_missing:
                return True
            time.sleep(0.5)
        LOG.error("Timed out waiting for SHMs: %s", ", ".join(last_missing))
        return False

    def controller_has_repeated_error20(self) -> bool:
        camera = self.config["camera"]
        pane = self.tmux.capture(camera["controller_session"], lines=120)
        failed_marker = "GLINT_FC2_20_RECOVERY_FAILED" in pane
        occurrences = pane.count("Fc2Error(20)") + pane.count("Fc2error(20)")
        threshold = int(camera.get("error20_occurrences_for_restart", 2))
        if failed_marker or occurrences >= threshold:
            LOG.error("Point Grey controller contains an unrecovered Fc2Error(20)")
            return True
        return False

    def camera_stack_healthy(self) -> bool:
        camera = self.config["camera"]
        if not all(self.tmux.exists(name) for name in self.camera_sessions):
            return False
        if not self.tmux.running_expected(
            camera["controller_session"], camera["controller_command_tokens"]
        ):
            return False
        for spec in camera["cameras"].values():
            if not self.tmux.running_expected(
                spec["framegrab_session"], camera["framegrab_command_tokens"]
            ):
                return False
        streams = [
            camera["cameras"]["pg1"]["stream"],
            camera["cameras"]["pg2"]["stream"],
        ]
        if not all(
            self.probe_shm(name, live=True, timeout=1.5) for name in streams
        ):
            return False
        return not self.controller_has_repeated_error20()

    def stop_camera_stack(self) -> None:
        camera = self.config["camera"]
        # The controller is stopped first so its signal handler can release
        # both control handles and their frame grabbers cleanly.
        self.tmux.kill(camera["controller_session"])
        time.sleep(1.0)
        self.tmux.kill(camera["cameras"]["pg1"]["framegrab_session"])
        self.tmux.kill(camera["cameras"]["pg2"]["framegrab_session"])
        time.sleep(float(camera["usb_release_s"]))

    def start_camera_stack(self) -> None:
        camera = self.config["camera"]
        service = configured_path(camera["service_script"], self.config_path)
        command = [
            self.config["paths"]["pycapture_python"],
            "-u",
            "-i",
            str(service),
            "--config",
            str(self.config_path),
        ]
        self.tmux.create(camera["controller_session"], command)

    def ensure_cameras(self, *, force_restart: bool) -> None:
        if not force_restart and self.camera_stack_healthy():
            LOG.info("Point Grey camera stack is already healthy; leaving it running")
            return

        camera = self.config["camera"]
        attempts = int(camera["restart_attempts"])
        streams = [
            camera["cameras"]["pg1"]["stream"],
            camera["cameras"]["pg2"]["stream"],
        ]
        for attempt in range(1, attempts + 1):
            LOG.info("Starting Point Grey camera stack (attempt %d/%d)", attempt, attempts)
            self.stop_camera_stack()
            self.start_camera_stack()
            healthy = self.wait_for_shms(
                streams, live=True, timeout=float(camera["startup_timeout_s"])
            )
            healthy = healthy and all(
                self.tmux.exists(name) for name in self.camera_sessions
            )
            if healthy:
                LOG.info(
                    "Camera streams started; observing one control-polling interval"
                )
                time.sleep(float(camera["stabilization_s"]))
                healthy = self.camera_stack_healthy()
            if healthy:
                LOG.info("Point Grey camera stack is healthy")
                return

            pane = self.tmux.capture(camera["controller_session"], lines=120)
            if pane.strip():
                LOG.error("glintpg_ctrl tail:\n%s", pane.rstrip())

        self.stop_camera_stack()
        raise StartupError(
            "Point Grey cameras failed after the configured retry count; "
            "the three camera tmux sessions were stopped"
        )

    def dm_healthy(self) -> bool:
        dm = self.config["dm"]
        if not self.tmux.running_expected(
            dm["session"], dm["driver_command_tokens"]
        ):
            return False
        return all(
            self.probe_shm(
                name,
                live=False,
                timeout=1.0,
                expected_size=int(dm["expected_elements"]),
            )
            for name in dm["required_shms"]
        )

    def ensure_dm(self) -> None:
        if self.dm_healthy():
            LOG.info("DM driver and control SHMs are already available")
            return

        dm = self.config["dm"]
        self.tmux.kill(dm["session"])
        inner_command = (
            "set -e; "
            f"{shlex.quote(self.config['paths']['milk_makecsetandrt'])} "
            f"$$ {shlex.quote(dm['cpu_set'])} {int(dm['realtime_priority'])}; "
            f"exec {shlex.quote(self.config['paths']['dm_driver'])} "
            + " ".join(shlex.quote(str(arg)) for arg in dm["driver_args"])
        )
        self.tmux.create(
            dm["session"],
            [self.config["paths"]["bash"], "-lc", inner_command],
        )
        streams_ready = self.wait_for_shms(
            dm["required_shms"],
            live=False,
            timeout=float(dm["startup_timeout_s"]),
            expected_size=int(dm["expected_elements"]),
        )
        if not streams_ready or not self.dm_healthy():
            pane = self.tmux.capture(dm["session"], lines=120)
            if pane.strip():
                LOG.error("glint_dm tail:\n%s", pane.rstrip())
            raise StartupError(
                "DM driver did not expose the configured SHMs. No DM state was sent."
            )
        LOG.info("DM driver is ready; the DM state was not changed")

    def verify_apapane(self) -> None:
        name = self.config["shm"]["apapane"]
        timeout = float(self.config["shm"]["live_probe_timeout_s"])
        if not self.probe_shm(name, live=True, timeout=timeout):
            raise StartupError(
                f"APAPANE SHM {name!r} is missing or not updating. "
                "APAPANE was not started and the alignment GUI will not be launched."
            )
        LOG.info("APAPANE SHM %s is live", name)

    def tmux_terminal(self, logical_name: str, session: str, title: str) -> Window | None:
        return self.windows.ensure_xterm(
            logical_name,
            title,
            [self.config["paths"]["tmux"], "attach-session", "-t", session],
        )

    def ensure_viewer(self, logical_name: str, stream: str, title_match: str) -> Window | None:
        existing = self.windows.find(title_match)
        if existing is None:
            self.runner.popen([self.config["paths"]["shm_viewer"], stream])
        return self.windows.place(logical_name, title_match)

    def gui_running(self) -> bool:
        return self.windows.find(self.config["gui"]["window_title_contains"]) is not None

    def launch_desktop(self) -> None:
        if self.no_windows:
            if not self.no_gui:
                self.launch_gui_without_terminal()
            return

        camera = self.config["camera"]
        dm = self.config["dm"]
        layout = self.config["windows"]["layout"]
        placed: dict[str, Window | None] = {}

        # Frame-grabber terminals are created first at exactly the same pixel
        # rectangles as their camera viewers. The viewers are raised last.
        placed["glintpg1_fgrab"] = self.tmux_terminal(
            "glintpg1_fgrab",
            camera["cameras"]["pg1"]["framegrab_session"],
            layout["glintpg1_fgrab"]["title"],
        )
        placed["glintpg2_fgrab"] = self.tmux_terminal(
            "glintpg2_fgrab",
            camera["cameras"]["pg2"]["framegrab_session"],
            layout["glintpg2_fgrab"]["title"],
        )
        placed["glint_dm"] = self.tmux_terminal(
            "glint_dm", dm["session"], layout["glint_dm"]["title"]
        )
        placed["glintpg_ctrl"] = self.tmux_terminal(
            "glintpg_ctrl",
            camera["controller_session"],
            layout["glintpg_ctrl"]["title"],
        )

        if not self.no_gui:
            placed["gui_terminal"] = self.ensure_gui_terminal()
        placed["general_terminal"] = self.ensure_general_terminal()

        placed["glintpg1_viewer"] = self.ensure_viewer(
            "glintpg1_viewer",
            camera["cameras"]["pg1"]["stream"],
            layout["glintpg1_viewer"]["title_contains"],
        )
        placed["glintpg2_viewer"] = self.ensure_viewer(
            "glintpg2_viewer",
            camera["cameras"]["pg2"]["stream"],
            layout["glintpg2_viewer"]["title_contains"],
        )

        if not self.no_gui:
            placed["glint_gui"] = self.windows.place(
                "glint_gui", self.config["gui"]["window_title_contains"]
            )

        # Explicit stacking for the only intentional overlaps.
        self.windows.raise_window(placed["glintpg1_fgrab"], "glintpg1_fgrab")
        self.windows.raise_window(placed["glintpg1_viewer"], "glintpg1_viewer")
        self.windows.raise_window(placed["glintpg2_fgrab"], "glintpg2_fgrab")
        self.windows.raise_window(placed["glintpg2_viewer"], "glintpg2_viewer")
        LOG.info("Desktop launch and placement completed")

    def ensure_gui_terminal(self) -> Window | None:
        gui = self.config["gui"]
        spec = self.config["windows"]["layout"]["gui_terminal"]
        workdir = Path(gui["working_directory"]).expanduser()
        launcher = workdir / gui["script"]
        gui_is_running = self.gui_running()
        command = (
            f"cd {shlex.quote(str(workdir))}; "
            f"{shlex.quote(self.config['paths']['gui_python'])} "
            f"{shlex.quote(str(launcher))} --config {shlex.quote(gui['config'])}; "
            f"exec {shlex.quote(self.config['paths']['bash'])} -i"
        )
        if gui_is_running:
            command = (
                f"cd {shlex.quote(str(workdir))}; "
                f"exec {shlex.quote(self.config['paths']['bash'])} -i"
            )
            LOG.info("GLINT GUI already exists; not starting a second instance")
        else:
            # If a previous GUI exited, its dedicated log xterm is deliberately
            # left at a shell prompt. Recreate only that owned window so the new
            # GUI command is actually run and its output remains visible.
            stale_terminal = self.windows.find(spec["title"])
            if stale_terminal is not None:
                self.windows.close(stale_terminal, "gui_terminal")
                if not self.windows.wait_until_absent(spec["title"], 3.0):
                    LOG.warning("The stale GUI terminal did not close in time")
        return self.windows.ensure_xterm(
            "gui_terminal",
            spec["title"],
            [self.config["paths"]["bash"], "-lc", command],
        )

    def ensure_general_terminal(self) -> Window | None:
        spec = self.config["windows"]["layout"]["general_terminal"]
        workdir = Path(self.config["gui"]["working_directory"]).expanduser()
        command = (
            f"cd {shlex.quote(str(workdir))}; "
            f"exec {shlex.quote(self.config['paths']['bash'])} -i"
        )
        return self.windows.ensure_xterm(
            "general_terminal",
            spec["title"],
            [self.config["paths"]["bash"], "-lc", command],
        )

    def launch_gui_without_terminal(self) -> None:
        gui = self.config["gui"]
        workdir = Path(gui["working_directory"]).expanduser()
        if self.gui_running():
            LOG.info("GLINT GUI is already running")
            return
        self.runner.popen(
            [
                self.config["paths"]["gui_python"],
                str(workdir / gui["script"]),
                "--config",
                gui["config"],
            ]
        )

    def status(self) -> bool:
        camera = self.config["camera"]
        dm = self.config["dm"]
        checks: list[tuple[str, bool]] = []
        for name in self.camera_sessions + [dm["session"]]:
            checks.append((f"tmux:{name}", self.tmux.exists(name)))
        checks.append(
            (
                "process:glintpg_ctrl",
                self.tmux.running_expected(
                    camera["controller_session"], camera["controller_command_tokens"]
                ),
            )
        )
        for key in ("pg1", "pg2"):
            checks.append(
                (
                    f"process:{camera['cameras'][key]['framegrab_session']}",
                    self.tmux.running_expected(
                        camera["cameras"][key]["framegrab_session"],
                        camera["framegrab_command_tokens"],
                    ),
                )
            )
        checks.append(
            (
                "process:hwint-bmc111",
                self.tmux.running_expected(
                    dm["session"], dm["driver_command_tokens"]
                ),
            )
        )
        checks.extend(
            [
                (
                    "SHM:glintpg1 live",
                    self.probe_shm(
                        camera["cameras"]["pg1"]["stream"], live=True, timeout=1.5
                    ),
                ),
                (
                    "SHM:glintpg2 live",
                    self.probe_shm(
                        camera["cameras"]["pg2"]["stream"], live=True, timeout=1.5
                    ),
                ),
                (
                    "SHM:apapane live",
                    self.probe_shm(
                        self.config["shm"]["apapane"], live=True, timeout=1.5
                    ),
                ),
            ]
        )
        checks.append(
            (
                "glintpg_ctrl no error 20",
                not self.controller_has_repeated_error20(),
            )
        )
        for name in dm["required_shms"]:
            checks.append(
                (
                    f"SHM:{name} readable",
                    self.probe_shm(
                        name,
                        live=False,
                        timeout=1.0,
                        expected_size=int(dm["expected_elements"]),
                    ),
                )
            )
        for label, ok in checks:
            LOG.info("%-30s %s", label, "OK" if ok else "MISSING/FAILED")
        return all(ok for _, ok in checks)

    def restore(self, *, force_camera_restart: bool) -> None:
        self.preflight()
        # APAPANE is an externally owned prerequisite. Fail before changing
        # any GLINT-owned session if it is unavailable.
        self.verify_apapane()
        self.ensure_cameras(force_restart=force_camera_restart)
        self.ensure_dm()
        self.launch_desktop()
        LOG.info("GLINT bench restoration completed successfully")

    def describe_plan(self) -> None:
        camera = self.config["camera"]
        LOG.info("DRY RUN — no tmux session, process, SHM, or window will be changed")
        LOG.info("Target host: %s; DISPLAY=%s", self.config["host"], self.config["environment"]["DISPLAY"])
        LOG.info("Camera controller: %s", camera["controller_session"])
        LOG.info("Camera streams: %s, %s", camera["cameras"]["pg1"]["stream"], camera["cameras"]["pg2"]["stream"])
        LOG.info("DM session: %s; DM shape command: none", self.config["dm"]["session"])
        LOG.info("APAPANE action: verify only")
        LOG.info("GUI directory: %s", self.config["gui"]["working_directory"])
        for name, spec in self.config["windows"]["layout"].items():
            LOG.info("Window %-20s rect=%s", name, spec["rect"])


def acquire_lock(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise StartupError("Another GLINT bench restoration is already running") from exc
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "bench_startup.json",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the configured plan without probing or changing anything"
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Read-only status report; perform no recovery"
    )
    parser.add_argument(
        "--force-camera-restart",
        action="store_true",
        help="Recreate only glintpg_ctrl and the two configured frame-grabber sessions",
    )
    parser.add_argument("--no-windows", action="store_true", help="Do not launch or arrange xterm/viewer windows")
    parser.add_argument("--no-gui", action="store_true", help="Do not launch the alignment GUI")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    try:
        config = load_config(config_path)
        log_path = Path(config["logging"]["path"]).expanduser()
        setup_logging(log_path, file_logging=not args.dry_run)
        restorer = BenchRestorer(
            config,
            config_path,
            no_windows=args.no_windows,
            no_gui=args.no_gui,
        )
        if args.dry_run:
            restorer.validate_layout()
            restorer.describe_plan()
            return 0
        lock_handle = acquire_lock(Path(config["locking"]["path"]).expanduser())
        try:
            if args.check_only:
                restorer.preflight()
                return 0 if restorer.status() else 1
            restorer.restore(force_camera_restart=args.force_camera_restart)
            return 0
        finally:
            lock_handle.close()
    except (StartupError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        LOG.error("GLINT bench restoration failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
