#!/usr/bin/env python3
"""Launch the alignment GUI and attach to the GLINT instrument services."""

import argparse

from glint_alignment.config import load_config
from glint_alignment.gui import run_gui
from glint_alignment.instrument import InstrumentHardware


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Optional instrument JSON configuration")
    args = parser.parse_args()
    config = load_config(args.config, simulation=False)
    run_gui(InstrumentHardware(config), config)


if __name__ == "__main__":
    main()
