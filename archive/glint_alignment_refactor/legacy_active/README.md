# Legacy active-code snapshot

This directory is a read-only reference snapshot of the Python entry points reachable from the uploaded `gui/run_gui.py` and `hotspotalignment/autoalign.py`, plus the active scan/reader scripts they launch and the control wrappers they import.

The new package does not import these files. They are retained to make hardware-command review and numerical comparisons straightforward. Amplitude scans, archived experiments, acquired observations, caches, and `.git` content are excluded by design.
