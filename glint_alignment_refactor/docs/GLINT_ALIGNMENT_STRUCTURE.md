# GLINT Alignment Refactor: Structure, Dependencies and File Storage

Version 0.4.8

## 1. Quick reference

- All packaged defaults and calibration metadata are in `config/`.
- The instrument GUI normally loads and updates `config/instrument.json`.
- The local simulator loads and updates `config/simulator.json`.
- Clicking **WORKFLOW** opens the persistent scan-default editor. **UPDATE DEFAULTS** validates the complete set, applies it immediately, and atomically rewrites the active JSON.
- Clicking **MOUNT CONFIGURATION** opens the live six-axis mount-control window. There is no separate Advanced button.
- Telescope alignment products are written below `/home/scexao/glint/alignment_scan` by default.
- Simulator products are written below `<refactor root>/alignment_scan` by default.
- Darks are shared and overwritten in `alignment_scan/darks/`.
- DM, mount and Conex commands append to three persistent JSONL logs in `alignment_scan/command_logs/`.

## 2. High-level architecture

1. A launcher loads one JSON configuration.
2. The selected hardware backend attaches either to telescope services or to the deterministic simulator.
3. The GUI collects settings and operator decisions.
4. `AlignmentEngine` performs scans through the hardware-neutral `AlignmentHardware` interface.
5. Analysis functions generate recommendations; the GUI never applies a recommendation until the operator confirms it.
6. Storage functions save arrays, JSON metadata, the exact review PNG, the final state and command-log sequence references.

The instrument and simulator use the same GUI, scan engine, analysis, storage, configuration validation and workflow state machine. Only the hardware adapter changes.

## 3. Repository structure

```text
glint_alignment_refactor/
  config/                         all packaged default/configuration assets
    instrument.json               telescope paths, safeguards, DM states and scan defaults
    simulator.json                local-demo defaults and synthetic-hardware settings
    detector_boxes.json           photometry/null/fringe-tracking extraction boxes
    response_matrix.npy           default 2 x 2 Conex response matrix
    README.md                     concise configuration-asset guide
  docs/
    GLINT_ALIGNMENT_STRUCTURE.md  source copy of this guide
  glint_alignment/
    analysis.py                   Gaussian/COM/null fits and common-piston solution
    boxes.py                      loads detector_boxes.json and integrates boxes
    camera_storage.py             manual camera-frame filenames and FITS/NPY writing
    command_log.py                three continuously appended device logs
    config.py                     load, validate and atomically save configuration
    engine.py                     scan sequencing, safeguards, restore and Apply logic
    gui.py                        dashboard, pop-ups, review figures and operator decisions
    hardware.py                   hardware-neutral abstract interface and snapshots
    instrument.py                 direct telescope SHM/serial/Conex adapter
    models.py                     grids, segment PTT, snapshots and scan results
    null_live.py                  APAPANE live statistics and null-depth calculation
    simulator.py                  deterministic local hardware model
    storage.py                    alignment sessions, scan products, darks and final JSON
    suruga_mount.py               packaged DS102 driver preserving legacy wire commands
    workflow_status.py            white/dot/tick/orange workflow state machine
  legacy_active/                  retained old scripts for traceability; never imported
  tests/test_core.py              simulator, persistence, storage and compatibility tests
  run_instrument.py               telescope GUI launcher
  run_local_sim.py                local GUI launcher
  run_headless_demo.py            complete non-GUI simulator sequence
  run_scan.py                     one-scan command-line launcher
  requirements.txt                Python dependencies
  pyproject.toml                  package metadata and Python dependency declarations
```

## 4. Central configuration folder

| File | Purpose | Changed by GUI? |
| --- | --- | --- |
| `config/instrument.json` | Every editable telescope path/service name, SHM names, mount mapping/limits, fixed roll, Conex rules, DM states, flat-voltage calibration reference, output paths and all scan defaults. | Scan defaults only, if it is the active file and **UPDATE DEFAULTS** is pressed. Other values are changed manually. |
| `config/simulator.json` | Equivalent defaults for the local demo plus simulator seed/noise/initial state. | Yes, when the simulator GUI updates defaults. |
| `config/detector_boxes.json` | Exact `[y0, y1, x0, x1]` boxes for Phot 1-3, N1-3 and FT1/FT2. | No; edit deliberately outside a running GUI. |
| `config/response_matrix.npy` | Default 2 x 2 focal-pixel/Conex response. | Read by bulk alignment; the Workflow editor can change its path or fallback matrix. |
| `config/README.md` | Operator summary of these assets. | No. |

DM maps are not separate hidden files. `zero_ptt`, `tilted_away`, `science_operation`, segment overrides and REST OF DM map definitions are all stored in the active mode JSON under `dm_states` and `rest_of_dm`.

Relative paths in a mode JSON are resolved from the refactor root. A custom instrument configuration passed with `--config` becomes the active file and receives future GUI default updates instead of `config/instrument.json`.

### External calibration/runtime references

The instrument JSON also points to telescope-owned files and services that cannot be bundled as generic defaults: focal-reference bright/dark FITS files, the 111-value DM flat-voltage calibration, DM/camera SHM streams and the Suruga serial device. Their paths or service names are recorded centrally in `config/instrument.json`, but the external files/services remain at their telescope locations. The only operational values outside that JSON are the detector rectangles and response-matrix bytes, both stored beside it under `config/`. No external `control-code` directory and no Python source edit are required to change a path or service name.

## 5. How persistent Workflow defaults are saved

1. Click **WORKFLOW** while no operation is running.
2. Select any of steps 1-8 in the pop-up.
3. Edit scalar fields normally. Fields marked `[JSON list]` must retain JSON list syntax, for example `[300, 800, 300, 800]`. Path fields are displayed read-only and must be changed manually in the active JSON.
4. Press **UPDATE DEFAULTS**.
5. Values are parsed using the type already stored in JSON: integer, floating point, Boolean, string or list.
6. The application validates all eight scans, not only the visible tab. Checks include positive frame/repetition counts, valid inclusive grids, mount endpoints, Z, crop ordering, thresholds and response-array shapes.
7. A complete public configuration is written to a temporary file beside the active JSON, flushed to disk and atomically moved over that JSON.
8. Runtime-only keys beginning with `_`, including the current alignment-session directory, are omitted.
9. Only after the disk write succeeds are the running GUI and engine switched to the new defaults.

If parsing, validation or writing fails, the previous on-disk and in-memory defaults remain active. Per-scan fields in the main panel remain temporary for that scan; use the Workflow pop-up when the change should survive a restart.

## 6. Launchers and configuration selection

| Command | Backend | Default configuration |
| --- | --- | --- |
| `python run_local_sim.py` | `SimulatedHardware` | `config/simulator.json` |
| `python run_instrument.py --config config/instrument.json` | `InstrumentHardware` | Explicit instrument JSON; omit `--config` to use the same default. |
| `python run_headless_demo.py` | Simulator, no GUI | `config/simulator.json` |
| `python run_scan.py <scan> --mode simulation` | One simulator scan | `config/simulator.json`, unless `--config` is supplied. |
| `python run_scan.py <scan> --mode instrument --config ...` | One telescope scan | Supplied or default instrument JSON. |

## 7. Dependencies

### Python and desktop dependencies

| Dependency | Used for |
| --- | --- |
| Python 3.10+ | Application runtime. |
| Tkinter / Tcl-Tk | GUI windows, controls and pop-ups; often installed as `python3-tk` on Linux. |
| NumPy >= 1.24 | Arrays, detector cubes, grids, state conversion and NPY/NPZ fallback. |
| SciPy >= 1.10 | Gaussian and null fitting, filtering and peak finding. |
| Matplotlib >= 3.7 | Scan review, DM state and Null Live figures. |
| Astropy >= 5.3 | FITS read/write. Without it, the simulator uses NPY/NPZ fallbacks; the instrument focal reference requires it. |
| pyserial >= 3.5 | Serial dependency of the telescope mount control stack. |

### Telescope-only services and interfaces

| External dependency | Interface used by this refactor |
| --- | --- |
| `pyMilk.interfacing.shm.SHM(name)` | Attach directly to the existing JSON-named `dmptt`, `dmvolt`, APAPANE, Palila, focal and pupil SHM streams. No legacy DM wrapper is imported. |
| Packaged `SurugaMount` | Opens the configured DS102 serial device and preserves the established `AXI...` and `STOP` wire commands for position/readback, motion state, controller-limit queries and stop. It has no private log path. |
| `glint_steering2` executable | `u/v status` and `u/v goto value`. `glint_steering1` is never commanded. |
| Focal-reference FITS pair | Bright and dark reference PSFs used by bulk alignment. |

The refactor does not initialize the DM or Point Grey cameras. It attaches to services that must already be running. The mount serial connection is opened by the instrument backend.

## 8. Mount configuration and control

Click **MOUNT CONFIGURATION** to open the six-axis controller. It shows live positions, configured limits and unit-specific jog steps. The logical/controller mapping is fixed:

Roll is displayed but locked to `mount_control.fixed_roll` (1400 by default). Every scan and jog uses the same safeguard path and includes the fixed roll. The configured JSON limits are authoritative; controller queries provide diagnostic warnings.

The instrument backend keeps a cache populated only by verified controller readback. A command already at its verified target is not sent again; moving axes are polled until stopped and read back once before acquisition. Pitch/yaw and X/Y use serpentine rasters. The GUI does not issue competing six-axis status requests while scans run. Serial command strings and the legacy open/write/read/close transaction lifetime are unchanged.

| Logical axis | Controller axis | Configured range |
| --- | ---: | ---: |
| Pitch | 1 | -500 to 800 pulses |
| Roll | 2 | 1000 to 2000 pulses |
| Yaw | 3 | -500 to 900 pulses |
| X | 4 | -7500 to 8000 microns |
| Z | 5 | 1100 to 2500 microns |
| Y | 6 | -4000 to 9000 microns |

## 9. Alignment output storage

### Automatic workflow

The instrument default root is `/home/scexao/glint/alignment_scan`. The simulator root is `<refactor root>/alignment_scan`.

Each press of automatic **RUN** creates:

```text
alignment_scan/YYYYMMDD/Alignment_outputs_N/
```

`N` increases for every automatic run that day, including partial ranges. A scan always receives a progressive suffix, including its first occurrence:

```text
03_bulk_alignment_1.json/.fits/.png
05_pitch_yaw_1.json/.fits/.png
05_pitch_yaw_2.json/.fits/.png
06_xy_1.json/.fits/.png
07_tip_tilt_1.json/.fits/.png
08_null_1.json/.fits/.png
final_configuration.json
```

`04_set_z_N` has JSON and FITS/NPZ but no plot. Darks use the shared location below instead of the session folder.

### Individual scans

An operator-run scan outside automatic RUN is saved under:

```text
alignment_scan/YYYYMMDD/Alignment_output_individual/
```

It uses the same numbered stem convention. Applying it turns downstream workflow states orange because the previous downstream results are no longer consistent.

### Darks

Focal and APAPANE alignment darks are updated in place:

```text
alignment_scan/darks/focal_dark.fits
alignment_scan/darks/focal_dark.json
alignment_scan/darks/apapane_dark.fits
alignment_scan/darks/apapane_dark.json
```

If Astropy is unavailable in the simulator, arrays use `.npy`. These shared latest darks allow a later partial workflow to reload them.

### Command logs

Exactly three files persist and append across scans, dates and application launches:

```text
alignment_scan/command_logs/dm_commands.jsonl
alignment_scan/command_logs/mount_commands.jsonl
alignment_scan/command_logs/conex_commands.jsonl
```

Each JSONL line contains a global sequence, UTC timestamp, mode, device, action, command, outcome and available target/readback/error data. Scan JSON records the relevant start/end sequence range and log paths; command records are not duplicated into scan folders.

### Manual camera frames

The persistent default folder is `camera_frame_root` in the active JSON. The supplied instrument value is `/home/scexao/glint/`; the simulator uses `<refactor root>/simulated_camera_frames`. Python has no fallback default. The operator may choose a one-off folder and basename in the GUI, but that does not change the JSON. A blank basename creates `<camera>_YYYYMMDD_HHMMSS_microseconds`. One frame is 2-D and multiple frames form a 3-D cube. FITS is used when Astropy is available, otherwise NPY.

## 10. What each saved scan contains

| Product | Contents |
| --- | --- |
| JSON | Scan kind/time, axes, fit estimates, recommendation, parameters, review decision, full post-review mount and Conex readback, all 37 DM segment piston/tip/tilt values, array filename/format, figure filename and command-log range. |
| FITS | Primary metadata plus one image HDU per scan axis/map. |
| NPZ | Simulator fallback containing the same named axis/map arrays if Astropy is unavailable. |
| PNG | Exact figure displayed in Scan Review for bulk, pitch/yaw, X/Y, tip/tilt and null scans. |
| `final_configuration.json` | Final mount, Conex and complete DM state; REST OF DM selection; all workflow statuses; requested scan range; complete public application configuration. |

Review products are first saved as `awaiting_review`. Apply, Redo or Restore/Exit updates the same JSON record rather than allocating another stem. Abort restores the captured pre-scan state and writes no scan product.

## 11. Configuration and output boundaries

- `config/` contains inputs/defaults and should be versioned/backed up.
- `alignment_scan/` contains acquired data and audit logs and should not be copied back into `config/`.
- `legacy_active/` is historical reference only.
- Telescope focal-reference FITS files, the DM flat-voltage file and running SHM/control services remain external; their authoritative paths/names are stored in `config/instrument.json`.
- `legacy_active/control-code/` is traceability material only. None of `shmDMcontrol.py`, `memsControl.py`, `apiDMcontrol.py`, or `chipMountControl.py` is imported at runtime.
- A scan JSON contains a snapshot of the public application configuration used for that run, but it is an output record, not the defaults file loaded at startup.

## 12. Pre-telescope checks

1. Back up `config/instrument.json`, `detector_boxes.json` and `response_matrix.npy` together.
2. Confirm that the active JSON path shown in the Workflow editor is the intended telescope file before pressing Update.
3. Validate all 37 segment-specific `tilted_away` and `science_operation` PTT values and their `validated` flags.
4. Confirm focal-reference FITS paths, the 111-value F0 calibration and its `validated` flag, response matrix, focal crop and Conex absolute limits.
5. Confirm mount axis mapping, ranges, fixed roll and Set Z.
6. Confirm the configured DM/camera SHM names, the Suruga serial device/timeouts and the `glint_steering2` executable. There is no control-code path.
7. Confirm output and command-log directories are writable.
8. Run the local simulator and the 48-test regression suite before telescope deployment.
9. Follow `TELESCOPE_INSTALL_CHECKLIST.md` for staged read-only and motion tests.
