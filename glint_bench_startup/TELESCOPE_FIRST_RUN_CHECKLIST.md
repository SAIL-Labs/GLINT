# GLINT bench supervisor — first telescope run

Use this checklist once before adopting the supervisor for normal recovery.

## 1. Configuration-only checks

- [ ] Package is installed at a stable location; suggested
      `/home/scexao/glint/glint/glint_bench_startup`.
- [ ] `config/bench_startup.json` points to the installed alignment-refactor
      directory and its `config/instrument.json`.
- [ ] `/usr/bin/xterm`, `/usr/bin/wmctrl`, `tmux`, `shmImshow.py`,
      `milk-makecsetandrt`, and `hwint-bmc111` paths still resolve.
- [ ] Camera serials remain `15322651` for PG1 and `15322653` for PG2.
- [ ] VNC display remains `:3` at 2560×1440.
- [ ] The log directory is writable by `scexao`.

## 2. Read-only checks

```bash
cd /home/scexao/glint/glint/glint_bench_startup
/home/scexao/miniforge3/bin/python restore_glint_bench.py --dry-run
/home/scexao/miniforge3/bin/python restore_glint_bench.py --check-only
```

- [ ] Dry run changes no session, SHM, process, or window.
- [ ] Check-only reports existing healthy services accurately.

## 3. Camera-only recovery validation

Perform this during engineering time, not during an active observation.
APAPANE must already be publishing its normal SHM; the supervisor will verify
it but will not start it.

```bash
/home/scexao/miniforge3/bin/python restore_glint_bench.py \
  --force-camera-restart --no-windows --no-gui
```

- [ ] Only `glintpg_ctrl`, `glintpg1_fgrab`, and `glintpg2_fgrab` are rebuilt.
- [ ] `glintpg1` and `glintpg2` publish changing frames.
- [ ] Both frame-grabber panes show the absolute pycapture Python, with no
      `conda activate` or `.bashrc` requirement.
- [ ] `pg1` and `pg2` are available after attaching to `glintpg_ctrl`.
- [ ] Waiting at least 20 seconds produces no repeating `Fc2Error(20)`.
- [ ] If one error 20 is deliberately encountered, the pane reports
      `GLINT_FC2_20_RECOVERED`; otherwise the supervisor performs only its one
      configured full retry.

## 4. DM validation

- [ ] The existing DM state has been recorded before testing.
- [ ] `glint_dm` runs `hwint-bmc111 -r -g` after applying `g_work`, priority 45.
- [ ] `dmptt` and `dmvolt` are readable with 111 elements each.
- [ ] No PTT0, V0, F0, flatten, zero-voltage, or other DM-state command appears
      in the startup log.
- [ ] Physical DM state is unchanged by the supervisor itself.

## 5. APAPANE and full desktop

- [ ] Start APAPANE using its normal independent procedure.
- [ ] Confirm `--check-only` reports `SHM:apapane live`.
- [ ] Run `restore_glint_bench.py` without optional flags.
- [ ] Alignment GUI launches only after both Point Grey cameras, DM services,
      and APAPANE pass their checks.
- [ ] `glintpg_ctrl` and `glint_dm` are visible in the upper-right.
- [ ] The GUI log and general terminals occupy the middle/lower right.
- [ ] PG1 and PG2 viewers cover the corresponding frame-grabber terminals.
- [ ] Minimizing a viewer exposes the correct frame-grabber terminal directly
      behind it.
- [ ] Re-running the supervisor does not duplicate healthy services or GUI.

## 6. Evidence to retain after the test

- [ ] Save the complete startup log:
      `/home/scexao/glint/glint/bench_startup_logs/glint_bench_startup.log`.
- [ ] Save `tmux capture-pane -p -S -100 -t glintpg_ctrl` if any camera recovery
      occurs.
- [ ] Record whether `dmptt` and `dmvolt` existed before or appeared after
      `hwint-bmc111` started.
- [ ] Record final window rectangles requiring adjustment; only edit the JSON
      `rect` values for the later layout pass.
