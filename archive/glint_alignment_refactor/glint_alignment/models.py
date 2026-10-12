from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AxisGrid:
    """Inclusive scan coordinates expressed as start, span, and step."""

    start: float
    span: float
    step: float

    def values(self) -> np.ndarray:
        if self.step <= 0 or self.span < 0:
            raise ValueError("step must be positive and span must be non-negative")
        count = int(round(self.span / self.step))
        if not np.isclose(count * self.step, self.span, atol=1e-9):
            raise ValueError(f"span {self.span} is not an integer multiple of step {self.step}")
        return self.start + np.arange(count + 1, dtype=float) * self.step


@dataclass
class SegmentPTT:
    piston_um: float = 0.0
    tip_mrad: float = 0.0
    tilt_mrad: float = 0.0


def zero_dm_state() -> dict[int, SegmentPTT]:
    return {segment: SegmentPTT() for segment in range(37)}


def dm_state_to_dict(state: dict[int, SegmentPTT]) -> dict[str, dict[str, float]]:
    return {str(k): asdict(v) for k, v in sorted(state.items())}


def dm_state_from_array(values: np.ndarray) -> dict[int, SegmentPTT]:
    values = np.asarray(values, dtype=float).reshape(37, 3)
    return {i: SegmentPTT(*map(float, values[i])) for i in range(37)}


def dm_state_to_array(state: dict[int, SegmentPTT]) -> np.ndarray:
    return np.asarray(
        [[state[i].piston_um, state[i].tip_mrad, state[i].tilt_mrad] for i in range(37)],
        dtype=np.float32,
    )


@dataclass
class SystemSnapshot:
    mount: dict[str, float]
    dm: dict[int, SegmentPTT]
    dm_volts: np.ndarray
    conex: dict[str, float]


@dataclass
class ScanResult:
    kind: str
    axes: dict[str, np.ndarray] = field(default_factory=dict)
    maps: dict[str, np.ndarray] = field(default_factory=dict)
    estimates: dict[str, Any] = field(default_factory=dict)
    recommendation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    completed_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def json_summary(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": self.kind,
            "completed_utc": self.completed_utc,
            "axes": {k: np.asarray(v).tolist() for k, v in self.axes.items()},
            "estimates": _jsonable(self.estimates),
            "recommendation": _jsonable(self.recommendation),
            "metadata": _jsonable(self.metadata),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, SegmentPTT):
        return asdict(value)
    return value
