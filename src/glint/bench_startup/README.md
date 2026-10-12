# GLINT bench startup supervisor

This standalone package restores the GLINT services and desktop on `scexao5`
after a reboot or power cycle. It is intentionally separate from
`glint_alignment_refactor`.

It manages only these tmux sessions:

- `glintpg_ctrl`
- `glintpg1_fgrab`
- `glintpg2_fgrab`
- `glint_dm`

It verifies `apapane` but never starts APAPANE. It starts the DM driver but
never sends a DM state, mount command, or steering-mirror command.

## What it does

1. Refuses to run a real restoration unless the hostname is `scexao5`.
2. Prevents two restorations from running simultaneously.
3. Leaves an already healthy Point Grey stack untouched.
4. Otherwise recreates the three Point Grey sessions, starting cameras with
   the established serial numbers and waiting for two new frames from each SHM.
5. Observes a complete camera keyword-poll interval. A repeated, unrecovered
   `Fc2Error(20)` causes one clean automatic retry.
6. Leaves an already healthy `glint_dm` untouched. Otherwise it starts:

   ```bash
   milk-makecsetandrt $$ g_work 45
   hwint-bmc111 -r -g
   ```

   The `$$` expansion occurs inside the `glint_dm` pane.
7. Requires readable 111-element `dmptt` and `dmvolt` SHMs. It does not write
   either stream.
8. Requires a live `apapane` SHM. If APAPANE is absent, restoration stops before
   changing the GLINT-owned sessions or launching the alignment GUI.
9. Opens the GLINT GUI, camera viewers, and requested terminal windows on
   `DISPLAY=:3`, then applies the configured 2560×1440 layout.
10. Places each frame-grabber terminal at the same rectangle as its camera
    viewer and raises the viewer last, hiding the terminal immediately behind
    the corresponding camera.

## Why the old frame-grabber workaround is removed

The current camstack FlyCapture backend sends this command to a fresh tmux shell:

```bash
conda activate pycapture; python -m camstack.acq.flycapture_grab
```

When that shell has not loaded conda initialization, no frame grabber starts and
camstack waits indefinitely for its SHM. The local `GlintChameleon3` adapter uses
the exact interpreter running `glintpg_ctrl` instead:

```text
/home/scexao/miniforge3/envs/pycapture/bin/python
```

Consequently the frame-grabber sessions do not depend on `source ~/.bashrc`.
The installed `/home/scexao/src/camstack` tree is not modified.

The adapter also recognizes FlyCapture error 20. It rebuilds the stale
BusManager/control-camera handle and retries the keyword poll once without
touching the frame-grabber handle. If that recovery fails or errors continue,
the outer supervisor performs one clean restart of all three camera sessions.

## Installation on scexao5

Copy this complete directory to a stable location, suggested:

```text
/home/scexao/glint/glint/glint_bench_startup
```

No `pip install` is needed. The supervisor uses the Python standard library;
the camera helper runs in the existing `pycapture` environment.

Before the first real run, inspect:

```text
config/bench_startup.json
```

All machine paths, session names, serial numbers, timeouts, GUI paths, SHM names,
and window rectangles are stored there.

## First telescope test

Run from an ordinary `scexao5` shell belonging to the normal `scexao` account.

1. Print the intended operations without probing hardware:

   ```bash
   cd /home/scexao/glint/glint/glint_bench_startup
   /home/scexao/miniforge3/bin/python restore_glint_bench.py --dry-run
   ```

2. With the established services still running, perform a read-only report:

   ```bash
   /home/scexao/miniforge3/bin/python restore_glint_bench.py --check-only
   ```

3. With APAPANE already running, and after a controlled Point Grey shutdown,
   test camera and DM restoration
   without opening desktop windows or the alignment GUI:

   ```bash
   /home/scexao/miniforge3/bin/python restore_glint_bench.py \
     --no-windows --no-gui
   ```

4. Confirm the following manually:

   - `glintpg1` and `glintpg2` both change frame at the expected rate;
   - attaching to `glintpg_ctrl` exposes `pg1` and `pg2`;
   - both frame-grabber panes use the pycapture interpreter directly;
   - `Fc2Error(20)` is either absent or followed by
     `GLINT_FC2_20_RECOVERED`;
   - `hwint-bmc111` remains alive in `glint_dm`;
   - no zero-PTT, zero-voltage, or flat command was sent;
   - APAPANE was checked but not started.

5. Run the complete restoration:

   ```bash
   /home/scexao/miniforge3/bin/python restore_glint_bench.py
   ```

## Normal commands

Complete idempotent restoration:

```bash
python restore_glint_bench.py
```

Read-only health report:

```bash
python restore_glint_bench.py --check-only
```

Force only the three camera sessions to be rebuilt:

```bash
python restore_glint_bench.py --force-camera-restart
```

Restore services without windows or GUI:

```bash
python restore_glint_bench.py --no-windows --no-gui
```

## Resulting layout

| Window | Rectangle `[x, y, width, height]` | Behaviour |
| --- | ---: | --- |
| GLINT GUI | `[0, 0, 1470, 940]` | Upper-left |
| PG1 frame grabber | `[0, 940, 680, 500]` | Behind PG1 viewer |
| PG1 viewer | `[0, 940, 680, 500]` | Lower-left, raised |
| PG2 frame grabber | `[680, 940, 790, 500]` | Behind PG2 viewer |
| PG2 viewer | `[680, 940, 790, 500]` | Lower-centre, raised |
| `glint_dm` | `[1470, 0, 420, 465]` | Upper-right, left part |
| `glintpg_ctrl` | `[1890, 0, 670, 465]` | Upper-right, right part |
| GUI log terminal | `[1470, 465, 1090, 450]` | Middle-right |
| General terminal | `[1470, 915, 1090, 525]` | Lower-right |

The only intentional overlaps are each viewer and its corresponding
frame-grabber terminal. Change the `rect` arrays in `bench_startup.json` to tune
the layout; no Python edit is required.

## Failure and safety behaviour

- Only the four configured GLINT tmux names can be killed or created.
- A malformed tmux name is rejected.
- Existing healthy services are not restarted.
- Camera recovery stops after two configured attempts; there is no endless loop.
- A failed camera recovery leaves the three Point Grey sessions stopped rather
  than half initialized.
- A missing APAPANE stream does not trigger an APAPANE command.
- A DM startup failure never triggers a DM state command.
- No USB reset, USB power cycle, `sudo`, broad `pkill`, reboot, mount motion, or
  steering-mirror command is implemented.
- The alignment GUI is never killed by the supervisor.

## Logs

The same append-only log is used across runs:

```text
/home/scexao/glint/glint/bench_startup_logs/glint_bench_startup.log
```

It records checks, owned-session restarts, commands, timeouts, and captured pane
tails on failures. It does not replace the alignment GUI command logs.
