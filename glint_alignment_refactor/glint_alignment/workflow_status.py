from __future__ import annotations


class WorkflowStatus:
    """Pure workflow-state model used by the GUI and regression tests."""

    def __init__(self, steps) -> None:
        self.states = {step: "pending" for step in steps}
        self.xy_pitch_iteration = False
        self.iteration_stage: str | None = None
        self._before_run: dict[str, str] = {}

    def start(self, kind: str, *, allow_iteration: bool = True) -> None:
        self._before_run[kind] = self.states.get(kind, "pending")
        if (
            allow_iteration
            and kind == "pitch_yaw"
            and self.states.get("pitch_yaw") == "complete"
            and self.states.get("xy") == "complete"
        ):
            self.begin_xy_pitch_iteration()
        elif self.states.get(kind) != "iterating":
            self.states[kind] = "running"

    def reset(self) -> None:
        """Start a new complete alignment with every step pending again."""
        for step in self.states:
            self.states[step] = "pending"
        self.xy_pitch_iteration = False
        self.iteration_stage = None
        self._before_run.clear()

    def begin_xy_pitch_iteration(self) -> None:
        self.xy_pitch_iteration = True
        self.iteration_stage = "pitch_yaw"
        self.states["pitch_yaw"] = "iterating"
        self.states["xy"] = "iterating"

    def finish(self, kind: str, passed: bool = True) -> None:
        if self.xy_pitch_iteration and kind in {"pitch_yaw", "xy"}:
            if kind == self.iteration_stage and passed:
                self.states[kind] = "awaiting_confirmation"
            elif kind == self.iteration_stage:
                self.states[kind] = "iterating"
            return
        self.states[kind] = "awaiting_confirmation" if passed else "attention"

    def confirm(self, kind: str) -> str | None:
        """Confirm an applied result and return the next iterative scan, if any."""
        if self.states[kind] != "awaiting_confirmation":
            return None
        self.states[kind] = "complete"
        if not self.xy_pitch_iteration or kind != self.iteration_stage:
            return None
        if kind == "pitch_yaw":
            self.iteration_stage = "xy"
            self.states["xy"] = "iterating"
            return "xy"
        self.iteration_stage = None
        self.xy_pitch_iteration = False
        return None

    def restore(self, kind: str) -> None:
        """Return an unapplied reviewed result to a white pending state."""
        self.states[kind] = "pending"
        self._before_run.pop(kind, None)
        if not self.xy_pitch_iteration or kind != self.iteration_stage:
            return
        if kind == "pitch_yaw":
            # Both repeated alignment coordinates are unresolved when the
            # pitch/yaw half of an X/Y iteration is discarded.
            self.states["xy"] = "pending"
        self.xy_pitch_iteration = False
        self.iteration_stage = None

    def abort(self, kind: str) -> None:
        if self.xy_pitch_iteration and kind in {"pitch_yaw", "xy"}:
            self.states["pitch_yaw"] = "iterating"
            self.states["xy"] = "iterating"
            if kind == "xy" and self.iteration_stage == "xy":
                self.states["pitch_yaw"] = "complete"
        elif self.states.get(kind) == "running":
            self.states[kind] = self._before_run.get(kind, "pending")

    def state(self, kind: str) -> str:
        return self.states[kind]

    def invalidate_after(self, kind: str) -> None:
        """Mark every downstream result orange after an individual change."""
        steps = list(self.states)
        if kind not in self.states:
            raise KeyError(kind)
        for step in steps[steps.index(kind) + 1 :]:
            self.states[step] = "attention"
        self.xy_pitch_iteration = False
        self.iteration_stage = None

    def as_dict(self) -> dict[str, str]:
        return dict(self.states)
