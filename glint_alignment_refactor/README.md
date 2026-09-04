# GLINT automatic alignment

This refactor replaces the subprocess-driven, manual copy/paste scan workflow with one scan engine, one review GUI, and two hardware backends:

- `run_instrument.py` attaches to the existing GLINT SHM streams and the Suruga/Conex controls.
- `run_local_sim.py` uses synthetic cameras, mount, Conex mirrors, and a 37-segment DM. It exercises the same engine, fitters, safeguards, review buttons, restore behavior, and file writer as the instrument.

Version 0.4.8 replaces the null fit with a physical sinc coherence envelope multiplied by a sinusoid. Both baselines must now exhibit a coherent packet and valid FT normalization before piston values can be applied. The review displays every repetition, the mean, fit R² and FT scale, and the JSON/FITS products retain the underlying flux diagnostics. Null settling is restored to 0.1 s after the first telescope trace showed that the faster v0.4.7 cadence was not reliable enough to keep as a default. Mount-runtime optimizations and telescope command bytes are unchanged. See `docs/RUNTIME_AUDIT.md`.

The null model, exact two-leg command order, rejection rules and on-telescope troubleshooting steps are documented in `docs/NULL_SCAN_DIAGNOSTICS.md`.

The dashboard keeps the black retro styling, bright-green headings, compact workflow controls, balanced Mount/Status strip, editable DM inspector, live null pane, non-overlapping scan review plots, and permanent GLINT ALIGNMENT footer from v0.4.6. During DM-changing scans the PTT pane rereads SHM once per second. The GUI deliberately makes no background Suruga queries while a scan is running; mount readback refreshes at the scan/review boundary. All display-only behavior remains separate from hardware safeguards and reviewed Apply/Restore actions.

The old active scripts are retained under `legacy_active/` for traceability. They are not imported by the new package. Archived/amplitude-scan code, acquired data, and `.git` are intentionally omitted.

The runtime has no dependency on the legacy `control-code` directory. The instrument backend attaches directly to the JSON-named `dmptt` and `dmvolt` SHM streams and uses the packaged `glint_alignment.suruga_mount.SurugaMount` driver. That driver preserves the old DS102 wire commands and per-command serial lifetime but performs no file logging itself; all DM, mount and Conex records go through the three configured persistent JSONL logs.

## Try the local GUI

Python 3.10 or newer is recommended. Tkinter is included with most Python distributions; on Linux it may be packaged as `python3-tk`.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python run_local_sim.py
```

For a fast non-GUI regression/performance run:

```bash
python run_headless_demo.py
```

The simulator contains deliberately different channel centers, rotated/elliptical pitch-yaw maps, blurred hexagonal X/Y maps, individual segment tip/tilt maps, realistic dark/noise structure, and sinusoidal fringes within coherence envelopes. It starts in the synthetic `tilted_away` state so the DM vector display can be evaluated immediately. Its averaging shortcut is controlled by `simulation.fast_average` in `config/simulator.json`.

## Instrument launch

1. Install the Python requirements in the instrument environment.
2. Edit `config/instrument.json` and complete the preflight items below.
3. Make sure the existing DM and camera processes are already publishing `dmptt`, `dmvolt`, `apapane`, `palila`, `glintpg1`, and `glintpg2`. APAPANE and `glintpg2` are required for scans; unavailable Palila/pupil streams are reported but do not block alignment startup.
4. Launch only one process that owns the Suruga serial connection:

```bash
python run_instrument.py --config config/instrument.json
```

The instrument backend does **not** initialize the DM or either Point Grey camera. It attaches to existing streams. It does open the mount serial connection. A DM command is verified against SHM readback; because the downstream LUT/DM consumer is outside the uploaded code, that check cannot prove physical acceptance by the mirror controller.

## What each workflow button does

Successful scans return scanned mount/Conex coordinates to their pre-scan state before review. DM PTT is also restored except after X/Y, where the scan's required pattern is deliberately retained: injected segments are flat and all other segments retain the selected REST OF DM map. The user then chooses Apply, Redo, or Restore / Exit. Apply physically commands the full mount/DM configuration for that stage and refreshes both hardware panes from readback. Restore / Exit explicitly restores the pre-scan DM if X/Y retained its review pattern. **Abort + Restore** appears directly below the active scan's Run button only while that scan is executing. It issues a best-effort mount stop, restores mount, Conex, DM PTT, and the complete 111-element DM-voltage array, and writes no scan files. It is hidden and inactive at every other time, so it cannot reset the DM while the GUI is idle. A successful bulk scan deliberately retains the required zero-voltage state while its Conex recommendation waits for review; an aborted bulk scan restores the preceding voltages.

The compact automatic-workflow row contains **RUN**, a start spinbox, an arrow, and an end spinbox. Defaults 1 → 8 perform the complete alignment; ranges such as 3 → 7 run only that contiguous subset. Every automatic run still commands `mount_control.fixed_roll` (1400 pulses by default), physically commands the configured `workflow.start_dm_state` (`zero_ptt` by default), creates the next dated `Alignment_outputs_N` directory, refreshes both hardware panes, and resets all workflow colours to pending/white. Dark prompts are shown only when the selected range contains those steps. A range that skips a required dark is accepted only if its latest shared dark was loaded successfully. Every later mount command includes the fixed roll; a request for another roll is rejected. If startup or prerequisite validation fails, the workflow does not start.

There is no safeguard override. A failed recommendation cannot be applied: the choices are to redo the scan or restore/exit.

For every scan with a review stage, successful acquisition displays a green dot beside the workflow entry. This means the plots are ready but no recommendation has yet been applied. The dot becomes a green tick only after **Apply Recommended Values** succeeds. **Restore / Exit** discards the unapplied workflow result and returns that entry to pending white. Focal and APAPANE darks receive their ticks immediately because acquisition saves them directly and they have no Apply action.

### Persistent workflow defaults

Click the green **WORKFLOW** heading to open the complete default editor. Its left column selects one of the eight steps; the right side exposes every leaf stored under that scan in the active JSON, including bulk crop/response settings and list-valued fields. Lists use JSON syntax. Paths such as `bulk.response_matrix` are displayed but disabled and must be changed manually in the active JSON. **UPDATE DEFAULTS** parses values according to their existing JSON types, validates every scan grid and mount endpoint, checks counts, thresholds, crop and response-matrix shapes, then atomically replaces the active JSON. The new values are copied into the running engine immediately and are loaded again at the next launch. Invalid input or an unwritable configuration leaves both disk and in-memory defaults unchanged.

The destination is always the file used to launch the GUI: normally `config/instrument.json` or `config/simulator.json`, or the custom path supplied with `run_instrument.py --config ...`. Runtime-only keys such as the active alignment-output directory are never written into that file.

### REST OF DM maps

The DM pane's dropdown is populated from `rest_of_dm.maps` in the active JSON configuration. **Confirm REST OF DM** commands only the 34 non-injected segments and deliberately preserves the current physical piston/tip/tilt of injected segments 11, 20, and 31. The supplied names are:

- **Scan tilted-away**: the segment-specific `dm_states.tilted_away` calibration and the default scan map.
- **All flat**: the `dm_states.zero_ptt` map.
- **All tilted**: the `tilted_away` map with session-only uniform piston, tip and tilt entered in the pane.
- **Science operation**: the segment-specific `dm_states.science_operation` calibration.

The active selection remains in force for later X/Y, tip/tilt, and null scans and reviewed Apply commands. Tip/tilt and null still put the non-active injected inputs into their dedicated calibrated `tilted_away` positions. Named states are stored in JSON; editable all-tilted piston/tip/tilt values last for the current GUI session and are copied into each affected scan's metadata.

The three compact controls below REST OF DM act on the complete DM. **PTT0** publishes the configured `dm_states.zero_ptt` map. **V0** publishes 111 zero actuator voltages directly to the configured `dmvolt` SHM stream without changing the PTT table. **F0** publishes the 111-value calibrated flat-voltage file configured at `dm_voltage_states.flat.calibration_file` without changing PTT; it is blocked until that entry is marked `validated: true`. Every action asks for confirmation, uses SHM readback and enters the persistent DM command log.

### Mount control and Null Live

Click the green **MOUNT CONFIGURATION** heading to open the separate live six-axis controller. Selecting pitch, yaw, or roll shows configured CCW/CW limits and a pulse step; selecting X, Y, or Z shows minimum/maximum limits and a micron step. The defaults are 100 pulses for pitch/yaw, 5 µm for X/Y, and 100 µm for Z. The +/- buttons form an absolute target from current readback, run the same configured-limit validation as scans, include the fixed roll, issue the legacy command, and wait for readback. Roll is visible but deliberately cannot be jogged because the application invariant fixes it at 1400 pulses.

The **Null Live** tab takes one initial averaged APAPANE snapshot and then remains frozen. Press **START** above the flux summary to begin updates at the configured interval (10 seconds by default); the button becomes **STOP**, which pauses acquisition and retains the most recent frame. A scan or other camera operation always takes priority. The smaller image still overlays the exact Phot, N1/N2/N3, and FT1/FT2 boxes, while the wider tables show complete flux values and null depth to four decimal places. The simulator and instrument use this same view and calculation.

### 1. Focal-plane dark

The full workflow asks the operator to turn off/block the source once, acquires 5 focal-plane frames by default, takes their median, and updates `alignment_scan/darks/focal_dark.fits`. The source remains blocked for step 2. Running this dark individually retains the normal OFF/ON dialogs. The dialogs are present in both instrument and local-demo modes.

### 2. APAPANE dark

With the source still blocked, the GUI acquires 1000 APAPANE frames by default, takes their median, updates `alignment_scan/darks/apapane_dark.fits`, and prompts the operator to turn on/unblock the source. All following scans and the live-null pane subtract this in-memory dark.

### 3. Bulk alignment

The engine publishes zero DM voltages, measures the current dark-subtracted focal PSF, and aligns it to the two axes of `glint_steering2`. It starts with the stored response matrix and uses a full quasi-Newton step, a 100-pixel predicted-motion trust region, backtracking when a trial worsens the error, and a Broyden response update after accepted steps. A symmetric response recalibration is performed if the response is ill-conditioned or convergence stalls. Defaults are 2-pixel tolerance and 10 iterations. Review shows the reference, final trial PSF, and convergence curve. `glint_steering1` is configuration-only and is never invoked by the instrument backend.

### 4. Set mount Z

The operator confirms or edits the default absolute Z target from `scans.set_z.z`. Running this step validates Z against the configured 1100-2500-pulse limit and creates a review containing only `Z SET TO ...`; it does not move the mount during acquisition. **Apply Recommended Values** commands Z together with the configured fixed roll, waits for the normal mount readback, and then saves the reviewed result.

### 5. Pitch/yaw

The DM is set directly to zero PTT. The editable defaults are pitch `start=300, range=500, step=50` and yaw `start=-100, range=500, step=50`; endpoints are included. A redo rereads the current fields without mutating their configured defaults. Before the scan changes any hardware, each grid is rebuilt and checked as `endpoint = start + range`: pitch `300 + 500 = 800` is valid, while `300 + 550 = 850` is rejected with all three values and the configured bound in the error. At every point the engine commands only the axis or axes whose target changed, waits for their readback and 0.1 s settling, then acquires one APAPANE frame and integrates all three photometry boxes from that same image. The serpentine raster avoids a full-axis return at each row boundary. Each map is fitted by a bounded, rotated elliptical Gaussian with independent pitch/yaw sigma. Thresholded center of mass and then maximum provide fallbacks. The arithmetic mean always uses Phot 1, Phot 2, and Phot 3, and is accepted only if every channel lies within 150 pulses of it. Individual channels cannot be excluded in review. The review prints each photometric center above its map and the recommended three-channel mean across the top in green. Apply physically leaves every segment at its configured zero-PTT value while moving pitch/yaw.

### 6. X/Y

Non-injected segments use the active REST OF DM map; injected segments 11, 20, and 31 use their zero-PTT flat state. Editable defaults are X `2990, 100, 5` and Y `3390, 100, 5`, with inclusive endpoints. The command/readback/settle/acquire order is the same as pitch/yaw. A background-subtracted 60%-threshold center of mass is used because the expected lobe is a blurred hexagon; maximum is the fallback and boundary solutions are flagged. The review prints each photometric center above its map and the recommended three-channel mean across the top in green. After the scan finishes, this complete DM pattern remains physically commanded while the result is reviewed; it does not return the non-injected segments to flat. Apply retains the pattern while moving X/Y, whereas Restore / Exit returns to the state captured before the scan. The review includes an **Apply X/Y → Pitch/Yaw** option. Both workflow steps first turn orange. After the repeated pitch/yaw result is applied, only pitch/yaw becomes green and the GUI automatically runs/requires X/Y again. X/Y remains orange until its repeated recommendation is applied. If this loop was entered from the automatic **RUN** row, the queued tip/tilt and null scans are preserved and continue automatically afterward.

### 7. Tip/tilt

Segments 11, 20, and 31 are scanned one at a time. The active segment uses piston 0 µm and absolute tip/tilt coordinates from -1 to +1 mrad in 0.1 mrad steps relative to zero PTT. The other two injected segments use their dedicated calibrated `tilted_away` values; all non-injected segments use the active REST OF DM map. Ten frames per point are averaged. Phot 1/2/3 are paired with segments 11/20/31. Each map uses the bounded elliptical-Gaussian estimator; its fitted optimum is printed in green above that segment's map. Apply reconstructs the complete state: the three injected segments receive their fitted tip/tilt and every non-injected segment receives the selected map.

### 8. Null pistons

Only baselines 11–20 and 11–31 are scanned; 20–31 is disabled. For ordered pair `(a,b)`, `a` starts at -1.3 µm and `b` at +1.3 µm. The engine scans `a` upward while holding `b`, then holds `a` at maximum and scans `b` downward without duplicating the center point. Commands always precede acquisition. The path is repeated five times with ten frames per point. Null has an explicit 0.01 s post-command settling default, closer to the fast legacy cadence while remaining ten times more conservative than its 0.001 s inter-sample sleep.

The metric is `null / (FT1 + FT2)` using the configured boxes. The analysis fits a linear background plus sinusoid inside a Gaussian coherence envelope, finds all fitted troughs, validates them against nearby measured samples, and uses a local quadratic refinement. It then considers periodic candidate pairs and solves

```text
delta_11_20 = piston_11 - piston_20
delta_11_31 = piston_11 - piston_31
```

for a common set of three pistons within -1.3 to +1.3 µm. For every periodic candidate pair the absolute solution additionally enforces `piston_20 = -piston_31`; segment 11 takes the offset required to preserve both differentials. Null depth is primary and actuator margin breaks near-ties. Review shows the two measured curves, envelope fits, candidate troughs, chosen minima, and all three proposed pistons.

Apply preserves the fitted injected-segment tip/tilt from the preceding step, adds the three selected piston values, and keeps every non-injected segment in the active REST OF DM map. This complete state remains physically commanded after the last workflow step.

## Saving and recovery

Each automatic **RUN** creates `alignment_scan/YYYYMMDD/Alignment_outputs_N/`, where `N` is the next available integer for that local calendar date, even for a partial start/end range. Reviewed products are flat files and always use a two-digit workflow number plus a progressive suffix: `03_bulk_alignment_1`, `04_set_z_1`, `05_pitch_yaw_1`, `06_xy_1`, `07_tip_tilt_1`, or `08_null_1`; repeats become `_2`, `_3`, and so on. FITS/JSON and the exact displayed PNG are written as soon as acquisition completes, initially marked `awaiting_review`; Apply, Redo, or Restore/Exit finalizes the same JSON rather than allocating a duplicate. The simulator uses an NPZ fallback without Astropy. JSON records the decision, parameters, fits, recommendation, mount/Conex readback, and all 37 final DM piston/tip/tilt values. `final_configuration.json` records the complete applied hardware state, the requested range, and the status of all eight steps, including steps outside a partial range.

Running a scan from its individual workflow button while no automatic workflow is active saves it under `alignment_scan/YYYYMMDD/Alignment_output_individual/` with the same always-suffixed naming convention. This prevents an operator adjustment from modifying the preceding `Alignment_outputs_N` record. Once an individual recommendation is applied, every later workflow step turns orange because its earlier result is no longer downstream-consistent. Individual darks remain the shared overwrite-in-place exception described below.

Darks do not enter `Alignment_outputs_N`. They update fixed camera files under `alignment_scan/darks/`, so the most recent dark replaces the preceding dark for that camera. Aborted attempts create no scan products.

Commands append immediately to exactly three persistent files under `alignment_scan/command_logs/`: `dm_commands.jsonl`, `mount_commands.jsonl`, and `conex_commands.jsonl`. They are never split by scan, session, date, or application launch. Saved scan JSON records the relevant sequence interval and the persistent log paths. `glint_steering1` has no command log because this application never invokes it.

The mount mapping and authoritative configured safeguards are:

| Logical axis | Controller axis | Allowed pulses |
|---|---:|---:|
| Pitch | 1 | -500 to 800 |
| Roll | 2 | 1000 to 2000 |
| Yaw | 3 | -500 to 900 |
| X | 4 | -7500 to 8000 |
| Z | 5 | 1100 to 2500 |
| Y | 6 | -4000 to 9000 |

Controller soft-limit queries remain startup diagnostics. The scan engine rejects any target outside the configured bounds above before calling the legacy mount method.

## Manual camera frames

The left panel can acquire and save a user-selected number of Palila, APAPANE, focal-plane, or pupil-plane frames without leaving the alignment GUI. The count field is beside the camera selector. One frame is saved as a 2-D image; multiple frames are saved as a 3-D cube, while their mean is shown in the review tab. The persistent default destination is only `camera_frame_root` in the active JSON (`/home/scexao/glint/` in the supplied instrument configuration); Python contains no fallback destination. The operator can still choose a one-off folder and basename or use Browse, but that capture-specific choice does not rewrite the JSON default. A blank basename generates a timestamp.

The simulator generates all four camera types. When Astropy is not installed locally, its camera-frame writer uses `.npy` instead of FITS. Selecting or starting a different scan clears the preceding progress value, progress caption, and event-log text, and replaces the review display with the newly selected scan.

## Required instrument preflight

The code is complete enough to test locally, but these calibration inputs must be verified before moving the real instrument:

1. Replace the placeholder `tilted_away` default with measured, segment-specific piston/tip/tilt entries for all 37 segments and set `validated` to `true`.
2. Replace `dm_states.science_operation` with its measured segment-specific map before selecting it and set `validated` to `true`.
3. Confirm the supplied fixed roll of 1400 pulses and verify `scans.set_z.z`; Z remains explicitly marked unvalidated until checked for the observing setup.
4. Verify the focal reference FITS paths and focal crop.
5. Verify the supplied response matrix and 0.01 Conex response perturbations. Add absolute `u` and `v` limits to `config/instrument.json`; the 100-pixel trust region is not a substitute for travel limits.
6. Confirm the six configured mount ranges and compare them with every Suruga controller limit reported at startup. Resolve any warning that a configured range extends beyond an enabled controller limit before observing.
7. Confirm that the external DM LUT consumer reports rejected tip/tilt commands. This repository can verify only that the requested PTT array reached SHM.
8. Confirm `instrument.dm.ptt_shm` and `instrument.dm.voltage_shm` identify the existing 111-element streams. No `control-code` directory is used by the new runtime.
9. Confirm the Suruga serial device, baud rate and read/write timeouts. Only one process may own that serial device.
10. Confirm that `command_logging.root` is writable and has enough disk space. Verify the recovery coordinates `(0.239, 0.213)` for `glint_steering1` and `(0.058, -0.436)` for `glint_steering2`; the application stores both but is hard-coded to command only mirror 2.

See `docs/TELESCOPE_INSTALL_CHECKLIST.md` for the installation and staged hardware-test procedure.

## Detector boxes

All boxes use `[y0, y1, x0, x1]`. The exact photometry, null, and fringe-tracking boxes supplied for this refactor live in `glint_alignment/boxes.py`; each point uses one dark-subtracted frame for every channel integration.
