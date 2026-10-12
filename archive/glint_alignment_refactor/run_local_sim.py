#!/usr/bin/env python3
"""Launch the alignment GUI against synthetic hardware."""

from glint_alignment.config import load_config
from glint_alignment.gui import run_gui
from glint_alignment.simulator import SimulatedHardware


def main() -> None:
    config = load_config(simulation=True)
    run_gui(SimulatedHardware(config), config)


if __name__ == "__main__":
    main()
