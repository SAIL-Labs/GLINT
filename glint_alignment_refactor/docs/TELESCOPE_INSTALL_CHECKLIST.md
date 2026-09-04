# GLINT Alignment v0.4.6 — Telescope Installation Checklist

Use this checklist before the first telescope alignment. Items marked **motion** or **DM command** change hardware state and should be performed only in an approved engineering window.

## 1. Install the package

- [ ] Copy the complete `glint_alignment_refactor` directory to its intended telescope location.
- [ ] Use Python 3.10 or newer and install `requirements.txt` in the intended environment.
- [ ] Confirm `python -c "import numpy, scipy, matplotlib, astropy, serial"` succeeds.
- [ ] Confirm Tkinter is available with `python -c "import tkinter"`.
- [ ] Run `python -m unittest discover -s tests -v`; all 48 tests must pass.
- [ ] Run `python run_headless_demo.py` and then `python run_local_sim.py` before attaching to telescope services.

The runtime does not import anything from `legacy_active/control-code/`. That directory may remain for traceability, but it is not an installation dependency.

## 2. Verify writable destinations in `config/instrument.json`

- [ ] Set `output_root` to the intended alignment-product directory.
- [ ] Set `camera_frame_root` to the default manual camera-frame directory.
- [ ] Set `command_logging.root` to the permanent command-log directory.
- [ ] Confirm the launch user can create files and directories beneath all three paths.
- [ ] Confirm sufficient disk space and the desired backup/retention policy.

The command logger must append to exactly:

```text
dm_commands.jsonl
mount_commands.jsonl
conex_commands.jsonl
```

No legacy `command_log.txt` should be created.

## 3. Verify the existing DM service and SHM streams

- [ ] Start the normal telescope DM control/LUT process before launching this GUI.
- [ ] Confirm `instrument.dm.ptt_shm` names the existing `dmptt` stream.
- [ ] Confirm `instrument.dm.voltage_shm` names the existing `dmvolt` stream.
- [ ] Confirm both streams contain exactly 111 values.
- [ ] Confirm the PTT ordering is 37 rows of `[piston, tip, tilt]` in the same segment order used by the legacy scans.
- [ ] Confirm the external DM consumer reports LUT/out-of-range rejection independently; GUI SHM readback proves publication only, not physical acceptance.

The GUI attaches to these streams directly. It does not call `apiDMcontrol`, open the BMC controller, or initialize the physical DM.

## 4. Verify DM configuration and calibration

- [ ] Replace the placeholder `dm_states.tilted_away` map with measured values, including segment-specific overrides where required.
- [ ] Replace the placeholder `dm_states.science_operation` map with measured values.
- [ ] Set each map's `validated` flag only after checking all 37 segments.
- [ ] Put the real 111-value flat-voltage calibration at `config/32AW038_1500nm.txt`, or change `dm_voltage_states.flat.calibration_file` to its actual absolute path.
- [ ] Confirm the flat file contains exactly 111 finite values in the established actuator order and expected voltage normalization.
- [ ] Leave `dm_voltage_states.flat.validated` false until the file and ordering are confirmed.
- [ ] **DM command:** In a safe state, test PTT0 first and confirm both SHM and the external DM controller response.
- [ ] **DM command:** Test V0 and F0 only after their physical consequences have been approved. V0 and F0 change actuator voltages without changing the PTT table.

## 5. Verify camera SHM streams and calibrations

- [ ] Confirm `apapane`, `palila`, `glintpg1`, and `glintpg2` match the active SHM names in `instrument.shm`.
- [ ] Confirm APAPANE and focal (`glintpg2`) are available; startup requires both.
- [ ] Confirm image dimensions contain every rectangle in `config/detector_boxes.json`.
- [ ] Verify all photometry, null and fringe-tracking boxes on a current APAPANE frame.
- [ ] Verify `instrument.focal_reference.bright` and `.dark` point to the correct FITS files.
- [ ] Verify the bulk-alignment crop contains the reference and live PSFs.

## 6. Verify the Suruga mount connection

- [ ] Confirm the axis mapping: 1 pitch, 2 roll, 3 yaw, 4 X, 5 Z, 6 Y.
- [ ] Confirm every `mount_control.limits` value against the current controller and mechanical setup.
- [ ] Confirm `mount_control.fixed_roll` is 1400 only if that is the approved operating value.
- [ ] Confirm `scans.set_z.z` and its `validated` flag for the installed chip/setup.
- [ ] Confirm `instrument.mount.port` resolves to the DS102 controller.
- [ ] Confirm baud rate 38400 and the configured read/write timeouts are suitable.
- [ ] Stop the old GUI or any other process that could own the Suruga serial port; only one process may use it.
- [ ] Launch the instrument GUI and inspect the startup position/limit readbacks before commanding motion.
- [ ] **Motion:** Test one small X or Y jog well inside both configured and controller limits, then verify displayed and physical readback.
- [ ] **Motion:** Verify Abort + Restore sends `STOP 1` and restores the captured positions during a controlled test.

The packaged driver sends the same established controller strings, including `AXI4:POS?`, `AXI4:GOABS ...`, `AXI4:MOTION?`, limit queries, and `STOP 1`.

## 7. Verify the steering mirror interface

- [ ] Confirm `glint_steering2` is available in `PATH`, or set its absolute executable path in `instrument.json`.
- [ ] Confirm `glint_steering2 u status` and `glint_steering2 v status` return a final token parseable as a float.
- [ ] Add verified absolute U/V bounds under `conex.limits`; do not rely only on the 100-pixel trust region.
- [ ] Confirm `steering_mirrors.controlled_device` remains `glint_steering2`.
- [ ] Confirm `glint_steering1.command_enabled` remains false.
- [ ] Verify both stored recovery coordinates with the current installation.
- [ ] **Motion:** Perform a very small mirror-2 move within the absolute limits and confirm the persistent Conex log and focal-plane response.

## 8. Verify scan defaults and data products

- [ ] Verify `config/response_matrix.npy`, the two response perturbations and their sign convention.
- [ ] Verify pitch/yaw, X/Y, tip/tilt and piston start/range/step values.
- [ ] Confirm pitch/yaw and X/Y endpoints are calculated as `start + range` and remain inside configured mount limits.
- [ ] Confirm focal dark count 5 and APAPANE dark count 1000 are suitable for the observing setup.
- [ ] Run a short approved engineering scan and confirm FITS, JSON and PNG outputs have the expected names.
- [ ] Confirm scan JSON contains all 37 final segment PTT values and the command-log sequence range.
- [ ] Confirm `final_configuration.json` is written after the selected workflow range completes.

## 9. Final readiness gate

- [ ] There are no startup warnings for paths, SHM names, fixed roll, Z, DM maps, F0 or Conex limits that have not been explicitly accepted.
- [ ] The three persistent command logs are writable and contain the staged test commands.
- [ ] Restore/abort behaviour has been tested with harmless motions before running a full scan.
- [ ] The operator knows that SHM readback is not physical DM verification.
- [ ] The telescope team has reviewed the remaining hardware-command and recovery risks before an executive release.
