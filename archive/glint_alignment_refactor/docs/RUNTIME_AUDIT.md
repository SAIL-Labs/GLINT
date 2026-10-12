# GLINT scan runtime audit

Version 0.4.8 retains the v0.4.7 mount-runtime improvements and compares the refactored engine with the retained scripts under `legacy_active/`. Telescope wall time still depends on camera cadence, actual mount travel, serial latency and how many controller motion polls are required. The deterministic quantities below are the scan sample count and deliberate settling budget.

## Default workload and deliberate waits

| Step | Default samples or frames | Legacy timing behavior | v0.4.8 deliberate settling |
| --- | ---: | --- | ---: |
| Focal dark | 5 frames | Camera acquisition | None |
| APAPANE dark | 1,000 frames | Camera acquisition | None |
| Bulk | Up to 11 evaluation iterations, plus response trials if needed | Legacy `autoalign.py` used 0.6 s after trial moves | 0.1 s per Conex move |
| Set Z | One reviewed absolute move | Controller completion loop | Controller completion, then 0.1 s after Apply |
| Pitch/yaw | 11 × 11 = 121 points | Controller completion loop; no extra optical delay | 121 × 0.1 = 12.1 s |
| X/Y | 21 × 21 = 441 points | Controller completion loop; no extra optical delay | 441 × 0.1 = 44.1 s |
| Tip/tilt | 3 × 21 × 21 = 1,323 points | 0.1 s per sample | 1,323 × 0.1 = 132.3 s |
| Null | 2 baselines × 5 repeats × 105 points = 1,050 points | 0.001 s per inter-sample command, with 0.01 s initial commands; frames were acquired before the next labelled move | 1,050 × 0.1 = 105 s |

The saved legacy pitch/yaw parameter file currently contains a 50-pulse range with a 25-pulse step (9 points), whereas the approved refactor default is a 500-pulse range with a 50-pulse step (121 points). Those packaged defaults differ by 13.4× in point count. Runtime comparisons must use the same start, range, step and frame count.

## Significant discrepancy found

Before v0.4.7, every mount-map point supplied both scan axes and fixed roll to the hardware backend. The backend sent all three absolute commands, polled all three axes, read all three positions, and `_mount_map()` then read all six axes again. Even when only one coordinate changed, this required at least 15 serial transactions per sample, plus additional motion polls. The GUI independently requested another six-axis read every 0.5 s.

Version 0.4.7 retains the same `AXI<n>:GOABS`, `MOTION?`, `POS?`, and `STOP 1` bytes and the same per-command serial open/close lifetime, but:

- a verified readback cache removes unchanged pitch/yaw/X/Y/roll commands;
- only commanded axes are polled;
- position is queried only after motion reports stopped;
- verified positions are returned directly to the scan engine;
- a serpentine raster changes normally only one axis per point and removes full end-of-row returns;
- a transaction-level lock prevents interleaving with another GUI controller request;
- the GUI refreshes mount values only at scan/review boundaries.

A normal one-axis raster increment therefore uses 3 serial transactions: one absolute command, one motion query when the controller is already stopped, and one verified position query. More motion queries are made only while the controller reports motion. A two-axis initial move uses 6 minimum transactions. No safety limit, completion check, or 0.1 s mount settling delay was removed.

## Other scans

- Mount maps already acquire one APAPANE frame per point and integrate all three photometry boxes from that frame. The legacy scripts acquired three independent frames per point, so the refactor's camera path is not the source of the slowdown.
- Tip/tilt retains the legacy 0.1 s cadence. Its raster is now serpentine to reduce large DM jumps, but sample count and camera averaging are unchanged.
- Null uses the correct move → settle → acquire order. The retained legacy script acquired before issuing each labelled move, shifting saved samples by one command. Version 0.4.7 reduced the refactor delay to 0.01 s, but the first telescope result showed an unstable 11–31 trace; v0.4.8 therefore restores the conservative 0.1 s default until repeatability is measured. It remains editable in either JSON configuration.
- Command logs still flush every command immediately. Their three files are now opened once per application instead of once per command, which matters most for the 1,323-point tip/tilt and 1,050-point null scans.
- Scan JSON records measured `scan_operation_s`, `guarded_total_s`, sample count, and the configured per-sample settling interval for on-telescope comparison.

## Telescope verification

Run one short engineering raster before a full workflow and compare `runtime.scan_operation_s` in the saved JSON with the legacy wall time using identical parameters. Watch for residual image motion after the 0.1 s mount and null delays. If telescope testing establishes a shorter repeatable null delay, reduce only `scans.null.settle_s` and verify both baseline R² values and the saved FT diagnostics. Do not reduce controller polling or settling merely to match a previous runtime if readback or detector stability is not repeatable.
