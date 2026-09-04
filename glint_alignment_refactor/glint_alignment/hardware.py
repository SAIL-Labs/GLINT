from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np

from .models import SegmentPTT, SystemSnapshot


Progress = Callable[[str, int, int], None]


class AlignmentHardware(ABC):
    """Small hardware boundary shared by the real and simulated backends."""

    mode = "abstract"

    @abstractmethod
    def focal_frames(self, nframes: int) -> np.ndarray: ...

    @abstractmethod
    def apapane_frames(self, nframes: int) -> np.ndarray: ...

    @abstractmethod
    def camera_frames(self, camera: str, nframes: int = 1) -> np.ndarray:
        """Read one of: palila, apapane, focal, or pupil."""

    @abstractmethod
    def focal_reference(self) -> tuple[np.ndarray, np.ndarray]: ...

    @abstractmethod
    def get_mount(self) -> dict[str, float]: ...

    @abstractmethod
    def move_mount(self, targets: dict[str, float], timeout_s: float) -> None: ...

    @abstractmethod
    def mount_limits(self) -> dict[str, tuple[float, float]]: ...

    @abstractmethod
    def get_dm(self) -> dict[int, SegmentPTT]: ...

    @abstractmethod
    def set_dm(self, state: dict[int, SegmentPTT]) -> None: ...

    @abstractmethod
    def zero_dm_volts(self) -> None: ...

    @abstractmethod
    def flatten_dm_volts(self) -> None:
        """Publish the configured calibrated flat-voltage command."""

    @abstractmethod
    def get_dm_volts(self) -> np.ndarray: ...

    @abstractmethod
    def set_dm_volts(self, voltages: np.ndarray) -> None: ...

    @abstractmethod
    def get_conex(self) -> dict[str, float]: ...

    @abstractmethod
    def move_conex(self, targets: dict[str, float]) -> None: ...

    def command_log_marker(self) -> int:
        logger = getattr(self, "command_logger", None)
        return 0 if logger is None else int(logger.marker())

    def export_command_log(self, directory: Path, start_sequence: int) -> dict | None:
        logger = getattr(self, "command_logger", None)
        return None if logger is None else logger.export_since(directory, start_sequence)

    def snapshot(self) -> SystemSnapshot:
        return SystemSnapshot(self.get_mount(), self.get_dm(), self.get_dm_volts(), self.get_conex())

    def restore(self, snapshot: SystemSnapshot, timeout_s: float = 3.0) -> None:
        self.set_dm(snapshot.dm)
        self.set_dm_volts(snapshot.dm_volts)
        self.move_mount(snapshot.mount, timeout_s)
        self.move_conex(snapshot.conex)

    def stop(self) -> None:
        """Best-effort stop. Backends may override."""

    def set_dark_mode(self, enabled: bool) -> None:
        """Simulation hook. On the instrument the operator blocks the beam manually."""

    def close(self) -> None:
        """Release backend-owned resources. Attached SHM streams are not destroyed."""
