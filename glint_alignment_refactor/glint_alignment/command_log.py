from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEVICE_FILES = {
    "dm": "dm_commands.jsonl",
    "mount": "mount_commands.jsonl",
    "glint_steering2": "conex_commands.jsonl",
}


def _json_value(value: Any) -> Any:
    """Return JSON-safe command data without losing numeric arrays."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class CommandLogger:
    """Immediate, device-separated JSONL audit log for hardware commands."""

    def __init__(self, config: dict, mode: str) -> None:
        settings = config.get("command_logging", {})
        self.enabled = bool(settings.get("enabled", True))
        self.mode = str(mode)
        self._lock = threading.RLock()
        self._sequence = 0
        self._records: list[dict[str, Any]] = []
        self.session_directory: Path | None = None
        if not self.enabled:
            return

        configured_root = settings.get("root")
        if not configured_root:
            raise ValueError("command_logging.root must be defined in the active JSON configuration")
        root = Path(configured_root).expanduser()
        if not root.is_absolute():
            root = Path(config["_root"]) / root
        # These are deliberately persistent logs: every application launch and
        # every alignment appends to the same three files.  There are no daily,
        # session, or per-scan command-log copies.
        self.session_directory = root
        self.session_directory.mkdir(parents=True, exist_ok=True)
        for filename in DEVICE_FILES.values():
            (self.session_directory / filename).touch(exist_ok=True)
        self._sequence = self._last_persisted_sequence()

    def _last_persisted_sequence(self) -> int:
        last = 0
        assert self.session_directory is not None
        for filename in DEVICE_FILES.values():
            path = self.session_directory / filename
            try:
                with path.open("rb") as stream:
                    stream.seek(0, 2)
                    size = stream.tell()
                    stream.seek(max(0, size - 16384))
                    lines = stream.read().decode("utf-8", errors="ignore").splitlines()
                for line in reversed(lines):
                    try:
                        last = max(last, int(json.loads(line)["sequence"]))
                        break
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        continue
            except OSError:
                continue
        return last

    def marker(self) -> int:
        with self._lock:
            return self._sequence

    def record(
        self,
        device: str,
        action: str,
        *,
        command: Any,
        targets: Any | None = None,
        readback: Any | None = None,
        outcome: str = "succeeded",
        record_type: str = "command",
        error: str | None = None,
        details: Any | None = None,
    ) -> None:
        if not self.enabled:
            return
        if device not in DEVICE_FILES:
            raise ValueError(f"Unsupported command-log device {device!r}")
        with self._lock:
            self._sequence += 1
            record = {
                "sequence": self._sequence,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "mode": self.mode,
                "record_type": str(record_type),
                "device": device,
                "action": str(action),
                "command": _json_value(command),
                "outcome": str(outcome),
            }
            if targets is not None:
                record["targets"] = _json_value(targets)
            if readback is not None:
                record["readback"] = _json_value(readback)
            if error is not None:
                record["error"] = str(error)
            if details is not None:
                record["details"] = _json_value(details)
            self._records.append(record)
            line = json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
            assert self.session_directory is not None
            with (self.session_directory / DEVICE_FILES[device]).open("a", encoding="utf-8") as stream:
                stream.write(line)

    def export_since(self, directory: Path, start_sequence: int) -> dict[str, Any] | None:
        """Return the persistent-log range belonging to one scan.

        The trace is intentionally not copied into scan folders: the operator
        requested exactly three continuously appended device logs.
        """
        if not self.enabled:
            return None
        directory = Path(directory)
        with self._lock:
            records = [record for record in self._records if int(record["sequence"]) > int(start_sequence)]
            return {
                "persistent_directory": str(self.session_directory),
                "start_sequence_exclusive": int(start_sequence),
                "end_sequence_inclusive": self._sequence,
                "record_count": len(records),
                "files": dict(DEVICE_FILES),
            }
