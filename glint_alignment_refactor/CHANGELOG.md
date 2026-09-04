# Changelog

## 0.4.8

- Replaced the Gaussian null-packet model with a bounded sinc coherence envelope multiplied by a sinusoid. The default physical OPD fringe-period interval is 1.2–2.0 µm and remains editable in the active configuration.
- Added multi-start fitting seeded from the Hilbert envelope and a physical-period FFT search. Fit quality is measured against a linear-drift-only model, preventing a slow trend from masquerading as a coherent fringe packet.
- Added a mandatory null safeguard: both 11–20 and 11–31 must pass the configured R² threshold and FT-denominator validity check before a common piston solution can be applied.
- Saved per-sample null, FT1, FT2 and FT1+FT2 flux diagnostics plus every individual repetition in the result data. The review graph now shows repetitions, the mean, sinc × sine fit, R², FT scale and an explicit rejected state.
- Restored null settling to 0.1 s after the first telescope trace showed that the v0.4.7 0.01 s optimization was not demonstrably repeatable on both baselines.
- Verified and regression-tested the scan sequence: segment A minimum→maximum with B at maximum, followed by B maximum-minus-one-step→minimum with A at maximum. OPD remains `2 × (piston A − piston B)`.

## 0.4.7

- Audited all eight workflow steps against the retained legacy scan scripts and documented sample counts, settling delays, and controller-transaction differences in `docs/RUNTIME_AUDIT.md`.
- Reduced a normal pitch/yaw or X/Y raster step from at least 15 Suruga transactions to 3 by skipping verified unchanged axes, polling only axes actually commanded, avoiding position queries while motion is active, returning verified readback directly to the engine, and suppressing competing GUI mount polls during scans.
- Added an atomic mount-communication lock and retained the exact legacy DS102 strings plus one-open/close-per-command behavior for compatibility.
- Changed pitch/yaw, X/Y, and tip/tilt maps to serpentine traversal, avoiding large end-of-row return moves without changing saved array orientation or fitting coordinates.
- Added explicit per-scan `settle_s` settings. Bulk, pitch/yaw, X/Y, and tip/tilt remain at 0.1 s; null now defaults to a conservative 0.01 s instead of inheriting 0.1 s, reducing its fixed settling budget from 105 s to 10.5 s for the default 1,050 samples.
- Kept persistent command-log streams open for the application lifetime while flushing every record immediately, removing per-DM-sample file-open overhead without weakening the audit trail.
- Added runtime and sample-count metadata to scan JSON plus regression tests for optimized Suruga command counts, raster readback behavior, and absence of background controller polling; all 51 tests pass.

## 0.4.6

- Removed all runtime imports of `shmDMcontrol.py`, `memsControl.py`, `apiDMcontrol.py`, and `chipMountControl.py`; `legacy_active/control-code/` is now traceability-only.
- Attached directly to the JSON-named `dmptt` and `dmvolt` SHM streams without opening or initializing the physical DM.
- Added a packaged `SurugaMount` driver that preserves the legacy DS102 command bytes while removing the hard-coded `/home/scexao/glint/control-code/command_log.txt` dependency.
- Routed mount, DM, and Conex audit records exclusively through the configured persistent JSONL command logs and included raw Suruga motion/stop strings in mount records.
- Moved the unvalidated default F0 reference to `config/32AW038_1500nm.txt`; the operator must supply and validate the actual 111-value calibration file.
- Added telescope preflight validation for DM SHM names, mount serial settings/timeouts, and the configured flat-file path, plus a staged installation checklist.
- Added byte-for-byte Suruga protocol tests and direct-SHM instrument-adapter tests; all 48 regression tests pass.

## 0.4.5

- Split the lower hardware strip evenly between Mount Configuration and Status; Status now fills its complete half with responsive wrapping, progress and the read-only event log.
- Extended the editable all-tilted REST OF DM control to uniform piston, tip and tilt while preserving injected segments 11, 20 and 31.
- Added guarded **PTT0**, **V0** and **F0** controls below REST OF DM. PTT0 publishes the configured `zero_ptt` state, V0 uses legacy `DM.sendzeros`, and F0 loads the explicit 111-value calibration path in the active JSON before publishing through legacy-compatible `dmvolt.set_data`.
- Forced ordinary decimal piston colourbar labels while retaining symmetric-log scaling.
- Added green review annotations containing the three photometric centers and their recommended mean for pitch/yaw and X/Y, plus the optimum tip/tilt above every injected-segment map.
- Removed Python-side fallback output paths: default camera-frame and command-log destinations must now be declared in the active JSON configuration. Persistent path fields are read-only in the Workflow editor and are changed manually in JSON. Expanded regression coverage to 47 passing tests.

## 0.4.4

- Made the **WORKFLOW** heading clickable and added a persistent editor for every leaf of all eight scan-default configurations. **UPDATE DEFAULTS** performs typed parsing and complete safeguard validation, writes the active mode JSON atomically, and applies the values immediately.
- Made **MOUNT CONFIGURATION** itself open the existing safeguarded six-axis controller and removed the separate **ADVANCED** button.
- Consolidated packaged default assets in `config/`: instrument and simulator JSON, Conex response matrix, detector extraction boxes, and an asset-location README. Detector boxes are no longer embedded in Python source.
- Added a complete architecture, dependencies, configuration, and file-storage guide in `docs/GLINT_ALIGNMENT_STRUCTURE.md` and as a separately rendered Word document.
- Preserved instrument/simulator behavior and legacy telescope command signatures; expanded regression coverage to 44 passing tests.

## 0.4.3

- Constrained the Status text widget's requested width and changed the lower hardware strip to an 80/20 Mount/Status split, leaving the complete **ADVANCED** control visible.
- Rebalanced Null Live in favour of the APAPANE image, added protected figure margins for its title, axes and tick labels, and slightly reduced the statistics-table font.
- Removed the separator immediately above the workflow controls, made **RUN** a solid green action button, and reduced workflow-cell height by approximately 20%.
- Kept instrument and simulator behavior identical and retained all legacy hardware command signatures; all 41 regression tests pass.

## 0.4.2

- Collapsed automatic workflow selection into one compact `RUN | start → end` row. The button is now **RUN**, with editable arrow spinboxes defaulting to 1 and 8.
- Reallocated the lower hardware strip from a 40/60 to a 62.5/37.5 Mount/Status split, reduced status wrapping, and widened **ADVANCED** so its label is not clipped.
- Replaced the linear piston display with a vivid blue/cyan–green–yellow/red diverging palette containing no black. Symmetric-log normalization expands near-zero piston differences while retaining zero PTT in dashboard green; both configurations expose `dm_display.piston_linthresh_um`.
- Rebalanced Null Live so the APAPANE image is smaller and the photometry/null tables are wider. Null depth now displays exactly four decimal places and the explanatory footer was removed.
- Null Live now acquires one initial APAPANE snapshot and remains frozen. The operator-aligned **START** button enables periodic updates and changes to **STOP** while active; scan/camera ownership still takes priority.
- Preserved instrument/simulator behavior and all legacy hardware command signatures; expanded regression coverage to 41 passing tests.

## 0.4.1

- Added selectable 1–8 start/end bounds to **Run Autoalign**. A partial automatic workflow still creates a numbered `Alignment_outputs_N` session, resets all workflow states, and writes every step's status plus the requested range into `final_configuration.json`.
- Added persistent-dark reload and clear prerequisite errors so a partial workflow may start after the dark steps only when the required shared focal/APAPANE dark already exists.
- Routed operator-run scans to the dated `Alignment_output_individual` directory, leaving the last automatic session immutable. Applying an individual result turns every downstream workflow step orange.
- Changed every scan product to an always-suffixed, two-digit name such as `05_pitch_yaw_1`, `05_pitch_yaw_2`, including scans performed only once.
- Added complete grid validation before hardware motion. Pitch start 300, range 500, step 50 is accepted through the inclusive 800 endpoint; invalid messages explicitly report the calculated `start + range` endpoint.
- Corrected controller axis 2 / roll to the supplied 1000–2000 pulse range in both backends and the instrument fallback while retaining fixed roll 1400.
- Replaced the three Null Live statistic cards with compact photometry and baseline tables containing Null, FT1, FT2, and null-depth columns.
- Preserved the v0.4.0 DM SHM, Suruga mount, camera SHM, and `glint_steering2` command interfaces and expanded regression coverage to 39 tests.

## 0.4.0

- Added operator-selectable camera-frame destinations and filenames, defaulting to the old instrument GUI folder `/home/scexao/glint/`.
- Made the selected DM segment's piston, tip, and tilt editable and physically applicable through the shared backend.
- Added the safeguarded Advanced Mount Control window with live six-axis readback, configured ranges, unit-specific steps, relative +/- motion, and an intentionally locked fixed-roll axis.
- Added the 10-second APAPANE **Null Live** tab with the supplied photometry, null, and fringe-tracking boxes and the exact `I_null / (FT1 + FT2)` statistic.
- Constrained null solutions so segments 20 and 31 are equally distant from zero piston while segment 11 preserves both selected differential pistons.
- Reworked automatic-alignment storage into `alignment_scan/YYYYMMDD/Alignment_outputs_N`, flat scan-numbered FITS/JSON/PNG products, repeat promotion to `_1`, `_2`, etc., shared overwrite-in-place darks, and a final configuration JSON.
- Reordered the first steps to focal dark, APAPANE dark, then bulk alignment. One light-source OFF prompt now covers both darks; the ON prompt follows APAPANE. Defaults are 5 focal frames and 1000 APAPANE frames.
- Replaced per-session/per-scan command traces with exactly three continuously appended files: `dm_commands.jsonl`, `mount_commands.jsonl`, and `conex_commands.jsonl`.
- Preserved and retested the legacy instrument DM SHM, Suruga mount, camera SHM, and steering-2 CLI signatures. The engine now also accepts either 2-D legacy `multi_recv_data` averages or 3-D camera cubes.

## 0.3.14

- Corrected the authoritative mount boundaries: controller axis 2 / roll is now -500 to 2000 pulses, and controller axis 5 / Z is now 1100 to 2500 pulses.
- Applied the same corrected limits to the instrument configuration, simulator configuration, legacy fallback table, documentation, and compatibility tests.
- Retained the fixed roll at 1400 pulses and default Z at 1800 pulses; both remain within their corrected ranges.

## 0.3.13

- Added immediate, timestamped JSONL command auditing with combined and device-separated session files for the DM, mount, `glint_steering1`, and `glint_steering2`. Each saved scan also receives its own sequence-bounded copies and `scan.json` references.
- Registered the recovery positions `glint_steering1=(0.239, 0.213)` and `glint_steering2=(0.058, -0.436)`. Mirror 1 is configuration-only; runtime guards forbid selecting or invoking its executable.
- Added authoritative mount limits using the legacy mapping pitch=1, roll=2, yaw=3, x=4, z=5, y=6. Controller-limit queries remain startup diagnostics.
- Set the fixed default roll to 1400 pulses.
- Applied the requested footer placement with `ALIEN_X_SHIFT = -10` and `alien.move("all", ALIEN_X_SHIFT, 0)`.
- Preserved the legacy SHM, `Mount.set_pos`, and `glint_steering2` CLI command signatures and expanded regression coverage to 29 tests.

## 0.3.12

- Shifted the footer alien face 10 pixels left and increased its right-side clipping margin.
- No other interface, workflow, scan, hardware, safety, or storage behavior changed.

## 0.3.11

- Restyled the permanent green footer with `GLINT` and `ALIGNMENT` on separate lines.
- Added a deterministic black alien-face mark to the right of the footer text, drawn directly by Tk so it does not depend on emoji fonts or platform-specific colour rendering.
- Made the Status event log read-only and changed its pointer to an arrow; the application unlocks it only while appending or clearing internal messages.
- No workflow, scan, hardware, safety, or storage behavior changed.

## 0.3.10

- Successful reviewed scans now show a green dot while their recommendation awaits the operator's decision.
- A green tick is shown only after **Apply Recommended Values** has physically succeeded and the applied result has been saved.
- **Restore / Exit** now explicitly returns the current scan's workflow entry to pending/white.
- Dark acquisitions remain immediately complete because they are acquired and saved directly and do not offer an Apply action.
- Preserved the staged orange/green behavior of the pitch/yaw–X/Y iteration while applying the same review-dot rule to each repeated scan.

## 0.3.9

- Removed the upper GLINT/mode banner and replaced it with a permanent bright-green **GLINT ALIGNMENT** footer at the bottom of the Scan Settings/Review column.
- Extended the Scan Settings/Review column through the full window height so long scan forms, review controls, and transient Abort + Restore control have more vertical room.
- Extended the Workflow/Camera column through the full window height so the camera-frame **ACQUIRE** button remains visible at the minimum supported window size.
- Reduced the lower hardware strip from 150 to 125 pixels and confined it beneath the central Scan Review/DM pane.
- Moved Mount Configuration to the left side of that lower strip and Status/progress/log to its right.
- Vertically and horizontally centred the segment inspector and REST OF DM controls in the space between the DM map and the pane's left boundary.

## 0.3.8

- Added a **REST OF DM** controller beside the live DM map. Operators can select a calibrated scan tilted-away map, all-flat map, editable uniform all-tilted map, or science-operation map and physically apply it to all non-injected segments while preserving segments 11, 20, and 31.
- Made the selected REST OF DM map the non-injected state used by X/Y, tip/tilt, and null scans and their reviewed Apply actions. The selection and optional uniform tip/tilt values are recorded in scan metadata.
- Added a six-axis live mount readback panel for pitch, yaw, roll, X, Y, and Z beside Status.
- Added a configured fixed-roll invariant. Every mount command includes `mount_control.fixed_roll`, nonmatching roll targets are rejected, and full-workflow startup establishes the configured value before snapshots and scans.
- Inserted a reviewed **Set mount Z** workflow step immediately after the APAPANE dark. Z is validated during acquisition, displayed as `Z SET TO ...`, and physically moved only when the operator applies it.
- Enabled the light-source OFF/blocked and ON/unblocked dialogs for focal and APAPANE darks in both the instrument GUI and the local simulator.
- Added compatibility regression coverage for the exact legacy mount, DM SHM, camera SHM, and Conex command signatures used by the instrument backend.

## 0.3.7

- Preserve the complete X/Y DM pattern after a successful scan while its result is being reviewed: injected segments 11, 20, and 31 remain flat and every other segment remains at its configured `tilted_away` value.
- Keep abort behavior unchanged: Abort + Restore still restores the complete state captured before the scan and saves no data.
- Keep Restore / Exit behavior unchanged by explicitly restoring the pre-scan DM state when leaving an unapplied X/Y review.
- Added regression coverage for the physical DM state immediately after X/Y acquisition and its explicit review-state restoration.

## 0.3.6

- Make every reviewed Apply action command the complete DM state associated with that alignment stage, rather than changing only the fitted fields on the restored pre-scan state.
- Pitch/yaw Apply leaves all 37 segments at `zero_ptt`; X/Y Apply leaves injected segments flat and every other segment at its configured `tilted_away` value.
- Tip/tilt Apply leaves all non-injected segments tilted away and commands the three injected segments to their fitted values.
- Null Apply preserves the injected segments' fitted tip/tilt, applies their selected pistons, and leaves every non-injected segment tilted away.
- Record the named applied DM pattern in scan metadata and refresh the pane from hardware readback after Apply.
- Continue the full-workflow controller after the final Apply so it reaches and reports the completed state even when its queue is empty.
- Added regression tests for the physical DM state retained after every Apply stage and final workflow completion.

## 0.3.5

- Physically command the configured `zero_ptt` DM state whenever **Run Full Workflow** starts, before any focal dark or alignment step.
- Stop workflow startup and report an error if that DM reset fails.
- Refresh the DM pane from live hardware every 0.5 s while a scan runs, then refresh again after the scan restores its pre-scan state.
- Added explicit simulator and instrument configuration for `workflow.start_dm_state` and the live-refresh interval.
- Added regression tests proving the commanded DM states for pitch/yaw, X/Y, tip/tilt, and null scans, including flat injected X/Y segments.
- Added regression tests for physical workflow-start reset and live DM-pane updates.

## 0.3.4

- Rebuilt the DM inspector as a wider three-row table with compact numeric formatting so values remain within their cells.
- Removed the “click another segment” caption.
- Moved the piston colourbar title above the scale and its tick labels to the left, with additional spacing from the pane boundary.
- Save the exact displayed review composite as `scan_review.png` for pitch/yaw, X/Y, tip/tilt, and null scan outputs, and reference it in `scan.json`.
- Reset every workflow step to pending/white when **Run Full Workflow** starts a new alignment cycle.
- Added regression coverage for workflow reset and saved review PNG/JSON metadata.

## 0.3.3

- Clear the progress value, progress caption, and event log whenever a scan is selected or started, eliminating stale focal/APAPANE dark messages.
- Added an interactive DM segment inspector in the previously unused left side of the DM view; clicking a segment displays its current piston, tip, and tilt values.
- Added an orange selection outline without changing injected-segment red outlines or the existing piston/tip/tilt encoding.
- Preserved completed workflow buttons in green while scans run by defining explicit disabled-state colours for completed and iterative steps.
- Added regression coverage for progress clearing and DM segment selection/table updates.

## 0.3.2

- Moved Status, the progress bar, progress text, and event log into a fixed bottom panel spanning beneath Scan Review/DM State and Scan Settings.
- Reduced the plot/settings vertical area so scan progress remains visible for scans with long parameter lists.
- Removed the photometry-channel selection controls; pitch/yaw and X/Y recommendations always use Phot 1, Phot 2, and Phot 3.
- Forced the camera selector field and dropdown list to use a black background with bright-green text.
- Reworked three-map reviews with independent colourbar axes and wider inter-panel spacing, and reserved explicit margins for all other scan layouts to prevent overlap.

## 0.3.1

- Rotated each DM segment 30° clockwise from the 0.3.0 rendering so every hexagon has flat top and bottom edges in both instrument and simulator views.

## 0.3.0

- Replaced DM tip/tilt arrows with directional black-to-piston-colour gradients; flat segments are uniform.
- Mapped zero piston to dashboard green, added a piston colourbar, outlined injected segments in red, and removed the DM plot title.
- Moved scan progress beneath Status.
- Made Abort + Restore appear beneath the selected scan's Run button only while a scan is active.
- Added an editable camera-frame count and changed the acquisition action to **ACQUIRE**.
- Saved multi-frame manual acquisitions as detector cubes and previewed their mean.
- Split pitch/yaw–X/Y iteration confirmation so pitch/yaw becomes green first and X/Y remains orange until repeated and applied.
- Preserved and automatically resumed the full-workflow queue through the pitch/yaw–X/Y iteration.
- Added regression coverage for exact simulator camera cube depth, staged iteration state, and the flat/directional DM colour encoding.

## 0.2.0

- Rotated every DM hexagon 30° clockwise to show flat top/bottom edges.
- Changed injected-segment outlines and dashboard headings to bright green.
- Added a cross-platform retro/monospace font preference.
- Rebuilt all Matplotlib views for a black background.
- Centered three-map reviews in compact square panels.
- Rebuilt bulk review as two stacked focal-plane views beside convergence.
- Normalized DM tip/tilt arrows and initialized the simulator with visible PTT.
- Removed every safeguard-override control and apply path.
- Added named Palila, APAPANE, focal-plane, and pupil-plane frame acquisition.
- Added camera/date destination routing and synthetic versions of all four cameras.
- Added green/orange workflow states for the X/Y–pitch/yaw iteration.
- Cleared stale progress text whenever the selected scan changes.
- Extended snapshots and abort restoration to include all 111 DM voltages.
- Added regression tests for camera capture, visible simulated tilt, iterative status, and bulk-abort voltage restoration.
