#!/usr/bin/env python3
"""Thin non-GUI launcher for one scan using the same engine as the dashboard."""

import argparse

from glint_alignment.config import load_config
from glint_alignment.engine import AlignmentEngine
from glint_alignment.instrument import InstrumentHardware
from glint_alignment.simulator import SimulatedHardware
from glint_alignment.storage import save_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scan",
        choices=["focal_dark", "apapane_dark", "bulk", "set_z", "pitch_yaw", "xy", "tip_tilt", "null"],
    )
    parser.add_argument("--mode", choices=["simulation", "instrument"], default="simulation")
    parser.add_argument("--config")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    simulation = args.mode == "simulation"
    config = load_config(args.config, simulation=simulation)
    hardware = SimulatedHardware(config) if simulation else InstrumentHardware(config)
    engine = AlignmentEngine(
        hardware,
        config,
        progress=lambda message, current, total: print(f"[{current}/{total}] {message}"),
        log=print,
    )
    try:
        if args.scan == "focal_dark":
            result = engine.acquire_dark("focal", config["scans"]["focal_dark"]["nframes"])
        elif args.scan == "apapane_dark":
            result = engine.acquire_dark("apapane", config["scans"]["apapane_dark"]["nframes"])
        elif args.scan == "set_z":
            result = engine.set_z_scan()
            if args.apply:
                engine.apply(result)
        else:
            if args.scan == "bulk":
                input("Block the focal camera beam and press Enter to acquire the required dark…") if not simulation else None
                engine.acquire_dark("focal", config["scans"]["focal_dark"]["nframes"])
                input("Unblock the focal camera beam and press Enter to continue…") if not simulation else None
                result = engine.bulk_align()
            else:
                input("Block the APAPANE beam and press Enter to acquire the required dark…") if not simulation else None
                engine.acquire_dark("apapane", config["scans"]["apapane_dark"]["nframes"])
                input("Unblock the APAPANE beam and press Enter to continue…") if not simulation else None
                result = getattr(engine, f"{args.scan}_scan")()
            if args.apply:
                engine.apply(result)
        decision = "applied" if args.apply else "completed_not_applied"
        path = save_result(result, hardware, config, decision)
        print(f"Recommendation: {result.recommendation}")
        print(f"Saved: {path}")
    finally:
        hardware.close()


if __name__ == "__main__":
    main()
