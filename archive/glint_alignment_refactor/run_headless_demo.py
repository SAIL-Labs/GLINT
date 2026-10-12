#!/usr/bin/env python3
"""Run a complete synthetic sequence for regression/performance checks."""

import time

from glint_alignment.config import load_config
from glint_alignment.engine import AlignmentEngine
from glint_alignment.simulator import SimulatedHardware


def main() -> None:
    config = load_config(simulation=True)
    config["time_scale"] = 0.0
    hardware = SimulatedHardware(config)
    engine = AlignmentEngine(hardware, config)
    steps = [
        ("focal dark", lambda: engine.acquire_dark("focal", config["scans"]["focal_dark"]["nframes"]), False),
        ("APAPANE dark", lambda: engine.acquire_dark("apapane", config["scans"]["apapane_dark"]["nframes"]), False),
        ("bulk", engine.bulk_align, True),
        ("set Z", engine.set_z_scan, True),
        ("pitch/yaw", engine.pitch_yaw_scan, True),
        ("X/Y", engine.xy_scan, True),
        ("tip/tilt", engine.tip_tilt_scan, True),
        ("null", engine.null_scan, True),
    ]
    for label, action, apply in steps:
        start = time.perf_counter()
        result = action()
        if apply and result.recommendation.get("passed", False):
            engine.apply(result)
        print(f"{label:14s} {time.perf_counter() - start:7.3f} s  {result.recommendation}")


if __name__ == "__main__":
    main()
