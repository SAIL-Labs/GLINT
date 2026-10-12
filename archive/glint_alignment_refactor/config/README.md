# GLINT default configuration assets

This directory is the single packaged location for configuration/default files used by both launch modes.

- `instrument.json` is the telescope configuration. Clicking **WORKFLOW** and pressing **UPDATE DEFAULTS** writes scan-default changes back to this file when the instrument GUI was launched from it.
- `simulator.json` is the local-demo configuration and is updated in the same way by the simulator GUI.
- `detector_boxes.json` contains the APAPANE photometry, null and fringe-tracking extraction boxes. The coordinate convention is `[y0, y1, x0, x1]`.
- `response_matrix.npy` is the sole two-axis Conex response matrix used by both modes. It is registered under `calibration_assets.bulk_response_matrix` and is intentionally absent from the Workflow defaults editor.
- `glint_ohia_logo.png` is the GUI footer artwork referenced by both mode JSON files.

All editable telescope paths and service names are stored in `instrument.json`: alignment output, default camera-frame destination, persistent command logs, DM and camera SHM stream names, Suruga serial device/timeouts, focal-reference FITS files, DM flat-voltage calibration and packaged calibration/GUI assets. The runtime no longer imports `shmDMcontrol.py`, `memsControl.py`, `apiDMcontrol.py`, or `chipMountControl.py`, and no external `control-code` directory is required. Python source contains no fallback destination for camera frames or command logs. A one-off camera destination selected in the GUI applies only to that capture and never rewrites `camera_frame_root`.

All DM named states, REST OF DM maps, mount limits, fixed roll, device paths, camera stream names, scan defaults, output roots and recovery positions are stored inside the active mode JSON. Runtime products, acquired darks and command logs are not defaults and are written to the configured output directories instead. The only additional operational values to inspect are the detector rectangles in `detector_boxes.json` and the response-matrix bytes in `response_matrix.npy`; both remain in this same `config/` directory.

Relative asset paths in a mode JSON are resolved from the repository root. The GUI saves default edits atomically to the exact JSON path recorded as `_config_path`; this includes a custom file supplied with `run_instrument.py --config ...`.

Before telescope use, manually verify every path/name in `instrument.json`, populate the segment-specific DM maps, provide Conex limits and validate the Set-Z target. The bulk response-matrix path and its values cannot be changed from the Workflow editor. Copy the real flat file to `config/32AW038_1500nm.txt` or change `dm_voltage_states.flat.calibration_file`; verify that it contains exactly 111 finite values in the established actuator order and only then set its `validated` field to `true`. F0 is blocked while it remains false.
