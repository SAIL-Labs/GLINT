from __future__ import annotations

from threading import RLock
from typing import Callable, Protocol


class SerialConnection(Protocol):
    """Small subset of ``pyserial.Serial`` used by the Suruga controller."""

    is_open: bool

    def open(self) -> None: ...

    def close(self) -> None: ...

    def write(self, data: bytes) -> int: ...

    def read_until(self, expected: bytes = b"\n") -> bytes: ...


class SurugaMount:
    """Packaged DS102 serial adapter using the legacy GLINT command syntax.

    The legacy ``chipMountControl.Mount`` class mixed controller I/O with a
    hard-coded text log under ``/home/scexao/glint/control-code``.  This class
    deliberately performs I/O only.  ``InstrumentHardware`` owns structured
    command logging through the configured persistent log directory.
    """

    def __init__(
        self,
        port: str,
        baudrate: int,
        *,
        read_timeout_s: float | None = 1.0,
        write_timeout_s: float | None = 1.0,
        serial_factory: Callable[..., SerialConnection] | None = None,
    ) -> None:
        if serial_factory is None:
            try:
                from serial import Serial
            except ImportError as error:
                raise RuntimeError("pyserial is required for the instrument Suruga mount") from error
            serial_factory = Serial
        self._lock = RLock()
        self.serial = serial_factory(
            port=str(port),
            baudrate=int(baudrate),
            timeout=read_timeout_s,
            write_timeout=write_timeout_s,
        )

    def _ensure_open(self) -> None:
        if not bool(getattr(self.serial, "is_open", True)):
            self.serial.open()

    def read_command(self, command: str) -> str:
        """Send one query and return the ASCII reply without its terminator."""
        with self._lock:
            self._ensure_open()
            try:
                self.serial.write(f"{command}\r".encode("ascii"))
                response = self.serial.read_until(b"\r")
            finally:
                # Preserve the transaction lifetime of the working legacy
                # driver, which used ``with Serial(...)`` for every command.
                self.serial.close()
        if not response:
            raise TimeoutError(f"No response from Suruga controller for {command!r}")
        try:
            return response.rstrip(b"\r\n").decode("ascii")
        except UnicodeDecodeError as error:
            raise RuntimeError(f"Non-ASCII response from Suruga controller for {command!r}: {response!r}") from error

    def send_command(self, command: str) -> None:
        """Send one non-query command using the legacy carriage return."""
        with self._lock:
            self._ensure_open()
            try:
                self.serial.write(f"{command}\r".encode("ascii"))
            finally:
                self.serial.close()

    def get_pos(self, axis: int) -> int:
        return int(self.read_command(f"AXI{int(axis)}:POS?"))

    def set_pos(self, axis: int, position: int) -> None:
        self.send_command(f"AXI{int(axis)}:GOABS {int(position)}")

    def in_motion(self, axis: int) -> bool:
        return bool(int(self.read_command(f"AXI{int(axis)}:MOTION?")))

    def lim_enabled(self, axis: int, direction: str) -> bool:
        direction = str(direction).upper()
        if direction == "CW":
            command = f"AXI{int(axis)}:CWSLE?"
        elif direction == "CCW":
            command = f"AXI{int(axis)}:CCWSLE?"
        else:
            raise ValueError('direction must be "CW" or "CCW"')
        return bool(int(self.read_command(command)))

    def get_lim(self, axis: int, direction: str) -> int:
        direction = str(direction).upper()
        if direction == "CW":
            command = f"AXI{int(axis)}:CWSLP?"
        elif direction == "CCW":
            command = f"AXI{int(axis)}:CCWSLP?"
        else:
            raise ValueError('direction must be "CW" or "CCW"')
        return int(self.read_command(command))

    def rstop(self) -> None:
        self.send_command("STOP 1")

    def estop(self) -> None:
        self.send_command("STOP 0")

    def close(self) -> None:
        with self._lock:
            if bool(getattr(self.serial, "is_open", True)):
                self.serial.close()
