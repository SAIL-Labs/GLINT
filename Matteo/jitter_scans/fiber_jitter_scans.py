#!/usr/bin/env python3
"""Record fixed-position fiber spectra until Ctrl+C; never command the mount.

Each sample averages 100 APAPANE frames, subtracts the saved dark, and sums
the detector-box rows to retain a 1D spectrum along detector X. Target one
sample per second; record actual batch start/midpoint/end times. Reuse the
focus scan's hardware adapter, saved-dark handling and SCExAO SSH helper.
The GLINT keep-alive republishes only the launch-time PTT, in a background
thread. No SEGMENT_PTT table, mount move, fixed-roll command or restoration
move is applied. Source frames and averaged images are not stored.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any

import numpy as np
from astropy.io import fits

SPECTRAL_BOX = (72, 77, 158, 267)  # [y0,y1,x0,x1], upper bounds excluded
NFRAMES = 100
SAMPLE_INTERVAL_S = 1.0
CHECKPOINT_EVERY = 30  # completed samples; Ctrl+C always saves immediately
GLINT_DM_KEEPALIVE_INTERVAL_MIN = 15.0
SCEXAO_DM_KEEPALIVE_ENABLED = True
SCEXAO_DM_KEEPALIVE_INTERVAL_MIN = 45.0
SCEXAO_DM_SSH_HOST = "sc6"
SCEXAO_DM_SSH_TIMEOUT_S = 30.0
OUTPUT_FOLDER = "Fiber_scan_jitter"

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
_helper_path = SCRIPT_DIRECTORY.parent / "Fiber_focus_scans" / "fiber_focus_scans.py"
_spec = importlib.util.spec_from_file_location("glint_jitter_focus_helpers", _helper_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load the existing focus helpers: {_helper_path}")
focus = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(focus)  # imports definitions only; does not open hardware
InstrumentHardware = focus.InstrumentHardware


class GLINTKeepAlive:
    """Refresh the exact captured PTT even while a camera read is blocked."""

    def __init__(self, hardware, state, interval_s):
        self.hardware, self.state = hardware, copy.deepcopy(state)
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="jitter-GLINT-keepalive", daemon=True)
        self._events = []
        self._error = None

    def _loop(self):
        while not self._stop.wait(self.interval_s):
            event = {"utc": datetime.now(timezone.utc).isoformat()}
            try:
                self.hardware.set_dm(self.state)
                event["status"] = "completed"
            except Exception as error:
                event.update(status="failed", error=f"{type(error).__name__}: {error}")
            with self._lock:
                self._events.append(event)
                if event["status"] == "failed":
                    self._error = event["error"]
            if event["status"] == "failed":
                return
            print(f"  GLINT keep-alive: launch-time PTT republished at {event['utc']}", flush=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join()

    def raise_if_failed(self):
        with self._lock:
            error = self._error
        if error:
            raise RuntimeError(f"GLINT keep-alive failed: {error}")

    def snapshot(self):
        with self._lock:
            return {"interval_min": self.interval_s/60, "state": "unchanged launch-time PTT",
                    "running": self._thread.is_alive(), "error": self._error,
                    "events": [dict(event) for event in self._events]}


def _scexao_settings():
    if not SCEXAO_DM_KEEPALIVE_ENABLED:
        return {"enabled": False}
    interval, timeout = float(SCEXAO_DM_KEEPALIVE_INTERVAL_MIN), float(SCEXAO_DM_SSH_TIMEOUT_S)
    host = str(SCEXAO_DM_SSH_HOST).strip()
    if not np.isfinite(interval) or interval <= 0 or not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("SCExAO keep-alive interval and timeout must be finite and positive")
    if not host or host.startswith("-") or any(c.isspace() for c in host):
        raise ValueError("SCEXAO_DM_SSH_HOST must be an SSH host/alias")
    return {"enabled": True, "interval_min": interval, "timeout_s": timeout,
            "ssh_host": host, "refresh_at_start": True,
            "ssh_argv": ["ssh", "-T", "-oBatchMode=yes", "-oConnectTimeout=10",
                         host, "dmdispzero", "00", "04"]}


def integrated_spectrum(frames, dark, box):
    """Average frames, subtract one dark, then SUM rows for each detector column."""
    averaged = np.mean(frames, axis=0, dtype=float)-dark
    y0, y1, x0, x1 = box
    return np.sum(averaged[y0:y1, x0:x1], axis=0, dtype=float)


def _next_output_base(root):
    directory = root / datetime.now().astimezone().strftime("%Y%m%d") / OUTPUT_FOLDER
    directory.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in directory.glob("fiber_jitter_*.*"):
        match = re.fullmatch(r"fiber_jitter_(\d+)\.(fits|json|png)", path.name)
        if match:
            numbers.append(int(match[1]))
    return directory/f"fiber_jitter_{max(numbers, default=0)+1}"


def _snapshot(hardware):
    mount = {key: float(value) for key, value in hardware.get_mount().items()}
    state = hardware.get_dm()
    values = np.array([[state[k].piston_um, state[k].tip_mrad, state[k].tilt_mrad] for k in range(37)])
    if not np.isfinite(values).all() or not all(np.isfinite(v) for v in mount.values()):
        raise ValueError("Mount and DM readbacks must be finite")
    result = {"mount": mount, "dm_ptt": focus.dm_state_to_dict(state)}
    try:
        volts = np.asarray(hardware.get_dm_volts(), dtype=float)
        if not np.isfinite(volts).all():
            raise ValueError("Nonfinite DM voltage readback")
        result["dm_volts"] = volts.tolist()
    except Exception as error:
        result["dm_volts_readback_error"] = f"{type(error).__name__}: {error}"
    return result, state


def write_outputs(base, samples, payload):
    """Atomic checkpoints contain only completed 100-frame batches."""
    box = payload["spectral_box_y0_y1_x0_x1"]
    spectra = np.array([sample["spectrum"] for sample in samples], dtype=float).reshape(-1, box[3]-box[2])
    primary = fits.PrimaryHDU()
    header = primary.header
    header["SCANTYPE"] = "FIBER_JITTER"
    header["STATUS"] = payload["status"]
    header["NFRAMES"] = payload["nframes"]
    header["CADENCE"] = payload["target_interval_s"]
    header["NSAMPLE"] = len(samples)
    header["DARKSUB"] = True
    header["ROWSUM"] = True
    header["MOUNTMOV"] = False
    for key, value in zip(("BOX_Y0", "BOX_Y1", "BOX_X0", "BOX_X1"), box):
        header[key] = value
    spectrum_hdu = fits.ImageHDU(spectra, name="SPECTRA")
    spectrum_hdu.header["NPORDER"] = "time,spectral_column"
    spectrum_hdu.header["BUNIT"] = "ADU/column/frame"
    spectrum_hdu.header["REDUCE"] = "mean_frames -> subtract_dark -> sum_rows"
    times = np.array([sample["time_s"] for sample in samples])
    hdus = [primary, spectrum_hdu, fits.ImageHDU(times, name="TIME_S"),
            fits.ImageHDU(np.arange(box[2], box[3]), name="SPECTRAL_X")]
    hdus[2].header["BUNIT"] = "s"
    hdus[2].header["TIMEDEF"] = "Batch midpoint elapsed from acquisition start"
    columns = [fits.Column(name=name, format="D", unit="s" if name != "RAW_MAX" else "ADU",
                           array=[sample[key] for sample in samples])
               for name, key in (("START_S", "start_s"), ("END_S", "end_s"),
                                 ("DURATION_S", "duration_s"), ("RAW_MAX", "raw_max"))]
    columns.append(fits.Column(name="UTC_MID", format="32A", array=[sample["utc"] for sample in samples]))
    hdus.append(fits.BinTableHDU.from_columns(columns, name="TIMING"))
    state = payload.get("pre_scan_state", {}).get("dm_ptt")
    if state:
        dm_hdu = fits.ImageHDU(np.array([[state[str(k)][name] for name in
                            ("piston_um", "tip_mrad", "tilt_mrad")] for k in range(37)]), name="DM_PTT")
        dm_hdu.header["PTTCOLS"] = "piston_um,tip_mrad,tilt_mrad"
        hdus.append(dm_hdu)
    encoded = json.dumps(payload, allow_nan=False, ensure_ascii=True)
    hdus.append(fits.BinTableHDU.from_columns([
        fits.Column(name="JSON", format=f"{len(encoded)}A", array=[encoded])], name="SCAN_SETTINGS"))
    path = base.with_suffix(".fits")
    temporary = path.with_name(f".{path.stem}.tmp.fits")
    fits.HDUList(hdus).writeto(temporary, overwrite=True, checksum=True)
    temporary.replace(path)
    focus._atomic_json(base.with_suffix(".json"), payload)


def display_flux(base, samples):
    import matplotlib.pyplot as plt
    if not samples:
        print("No completed batches to plot.")
        return
    figure, axis = plt.subplots(figsize=(10, 4), layout="constrained")
    axis.plot([s["time_s"] for s in samples], [float(np.sum(s["spectrum"])) for s in samples], lw=1)
    axis.set(xlabel="Elapsed time [s; batch midpoint]", ylabel="Integrated signal [ADU/frame]",
             title="Fixed-position fiber jitter: polychromatic flux")
    axis.grid(alpha=.25)
    try:
        figure.savefig(base.with_suffix(".png"), dpi=140)
        plt.show()
    finally:
        plt.close(figure)


def run(*, instrument_config=None, output_root=None, dark_path=None, nframes=NFRAMES,
        interval_s=SAMPLE_INTERVAL_S, box=SPECTRAL_BOX, checkpoint_every=CHECKPOINT_EVERY,
        show_plot=True, check_only=False):
    """Record from the terminal until Ctrl+C, save, then display total flux."""
    if isinstance(nframes, bool) or not isinstance(nframes, (int, np.integer)) or nframes < 1:
        raise ValueError("nframes must be a positive integer")
    if not np.isfinite(interval_s) or interval_s <= 0:
        raise ValueError("interval_s must be finite and positive")
    if not isinstance(checkpoint_every, int) or checkpoint_every < 1:
        raise ValueError("checkpoint_every must be a positive integer")
    bounds = np.asarray(box)
    if bounds.shape != (4,) or not np.isfinite(bounds).all() or not (bounds == np.rint(bounds)).all():
        raise ValueError("box must contain four finite integer bounds")
    box = tuple(map(int, bounds))
    if min(box[0], box[2]) < 0 or box[1] <= box[0] or box[3] <= box[2]:
        raise ValueError("Invalid detector box")
    if not np.isfinite(GLINT_DM_KEEPALIVE_INTERVAL_MIN) or GLINT_DM_KEEPALIVE_INTERVAL_MIN <= 0:
        raise ValueError("GLINT keep-alive interval must be finite and positive")
    remote_settings = _scexao_settings()
    config_path = Path(instrument_config or focus.DEFAULT_INSTRUMENT_CONFIG).expanduser().resolve()
    config = focus.load_config(config_path, simulation=False)
    root = Path(output_root).expanduser().resolve() if output_root is not None else focus._alignment_root(config)
    print(f"Fixed-position jitter: {nframes} frames/sample; target {interval_s:g} s; box {box}")
    print("No mount commands or new DM configuration; GLINT refresh uses current PTT.")
    print(f"Output: {root}/YYYYMMDD/{OUTPUT_FOLDER}/fiber_jitter_N.fits")
    if check_only:
        return None
    dark_file = Path(dark_path).expanduser().resolve() if dark_path is not None else focus._dark_path(root)
    dark = focus._load_dark(dark_file)
    if not np.isfinite(dark).all() or box[1] > dark.shape[0] or box[3] > dark.shape[1]:
        raise ValueError("Dark image is nonfinite or smaller than the selected box")
    base = _next_output_base(root)
    samples = []
    payload: dict[str, Any] = {
        "schema_version": 1, "kind": "fiber_jitter", "status": "starting",
        "started_utc": datetime.now(timezone.utc).isoformat(), "instrument_config": str(config_path),
        "instrument_config_snapshot": config, "output_directory": str(base.parent),
        "nframes": int(nframes), "target_interval_s": float(interval_s), "checkpoint_every": checkpoint_every,
        "spectral_box_y0_y1_x0_x1": list(box), "box_upper_bounds_excluded": True,
        "cube_axis_order": ["time", "spectral_column"], "spectral_units": "ADU/column/frame",
        "reduction": "mean_frames -> subtract_dark -> sum_rows", "mount_commands": False,
        "dark": {"path": str(dark_file), "sha256": focus._sha256(dark_file)},
        "saturation_guard": {"adc_max_adu": float(focus.ADC_MAX_ADU),
                             "fraction": float(focus.SATURATION_FRACTION)},
        "scexao_dm_keepalive": remote_settings, "acquired_samples": 0,
        "outputs": {"fits": base.with_suffix(".fits").name, "json": base.with_suffix(".json").name},
    }
    hardware = glint = remote = None
    failure = None

    def health():
        for worker in (glint, remote):
            if worker is not None:
                worker.raise_if_failed()

    def save():
        payload["acquired_samples"] = len(samples)
        if glint is not None:
            payload["glint_dm_keepalive"] = glint.snapshot()
        if remote is not None:
            payload["scexao_dm_keepalive"] = remote.snapshot()
        write_outputs(base, samples, payload)

    try:
        hardware = InstrumentHardware(config)
        payload["pre_scan_state"], state = _snapshot(hardware)
        glint = GLINTKeepAlive(hardware, state, GLINT_DM_KEEPALIVE_INTERVAL_MIN*60)
        glint.start()
        if remote_settings["enabled"]:
            remote = focus.SCExAODMKeepAlive(remote_settings)
            remote.start()
        started_utc = datetime.now(timezone.utc)
        clock = time.monotonic()
        next_due = clock
        payload.update(status="running", acquisition_started_utc=started_utc.isoformat())
        save()
        print("Recording; press Ctrl+C to save and finish.", flush=True)
        while True:
            health()
            while next_due > time.monotonic():
                time.sleep(min(.2, max(0., next_due-time.monotonic())))
                health()
            start = time.monotonic()
            frames = focus._frame_cube(hardware.apapane_frames(int(nframes)))
            end = time.monotonic()
            health()
            if frames.shape != (int(nframes), *dark.shape) or not np.isfinite(frames).all():
                raise ValueError(f"Invalid APAPANE batch: {frames.shape}; expected {(int(nframes), *dark.shape)}")
            maximum = float(np.max(frames))
            if maximum >= float(focus.ADC_MAX_ADU)*float(focus.SATURATION_FRACTION):
                raise focus.SaturationDetected(f"Raw APAPANE maximum {maximum:g} ADU reached saturation guard")
            spectrum = integrated_spectrum(frames, dark, box)
            midpoint = (start+end)/2-clock
            samples.append({"spectrum": spectrum, "time_s": midpoint, "start_s": start-clock,
                            "end_s": end-clock, "duration_s": end-start, "raw_max": maximum,
                            "utc": (started_utc+timedelta(seconds=midpoint)).isoformat()})
            if len(samples) % checkpoint_every == 0:
                save()
            if len(samples) == 1 or len(samples) % 10 == 0:
                print(f"  sample {len(samples)} | t={midpoint:.2f} s | total={spectrum.sum():.7g} ADU/frame", flush=True)
            # Preserve a 1-second target cadence without inventing samples or catch-up bursts.
            next_due = max(next_due+float(interval_s), time.monotonic())
    except KeyboardInterrupt:
        payload["status"] = "stopped"
        print("\nStopped by user; saving completed batches.", flush=True)
    except BaseException as error:
        failure = error
        payload.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        for worker in (remote, glint):
            if worker is not None:
                worker.stop()
                try:
                    worker.raise_if_failed()
                except Exception as error:
                    if failure is None:
                        failure = error
                        payload.update(status="failed", error=f"{type(error).__name__}: {error}")
        if hardware is not None:
            try:
                payload["post_scan_state"], _ = _snapshot(hardware)
            except Exception as error:
                payload["final_readback_error"] = f"{type(error).__name__}: {error}"
            try:
                hardware.close()
            except Exception as error:
                payload["close_error"] = f"{type(error).__name__}: {error}"
        payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    print(f"Saved {len(samples)} spectra: {base.with_suffix('.fits')}")
    if show_plot:
        display_flux(base, samples)
    if failure is not None:
        raise failure
    return base.with_suffix(".fits")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument-config", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dark-path", type=Path)
    parser.add_argument("--check-only", action="store_true", help="Print settings without opening hardware")
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
