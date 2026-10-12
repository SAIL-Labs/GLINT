#!/usr/bin/env python3
"""Standalone three-reference GLINT DM-wide tip/tilt diagnostic.

For each configured injected segment (normally 11, 20 and 31), its complete
aligned piston/tip/tilt is held at the live launch-time value while every other
DM segment is commanded together over the same tip/tilt raster.  This produces
three raster scans rather than 37 individual-segment scans.

The script attaches to already-running APAPANE and dmptt SHMs and always
restores the complete live launch-time DM state before exiting.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tiptilt_map_each_segment import (
    DEFAULT_INSTRUMENT_CONFIG,
    N_DM_SEGMENTS,
    BoxSpec,
    DMController,
    PersistentDMLogger,
    acquire_box_crops,
    allocate_raster_arrays,
    atomic_json,
    attach_devices,
    estimated_crop_bytes,
    grid_indices,
    human_bytes,
    instrument_root,
    load_dark,
    load_detector_boxes,
    load_json,
    print_preflight,
    public_scan_config,
    ptt_dict,
    resolve_apapane_dark,
    resolve_detector_boxes_path,
    resolve_output_root,
    save_raster_fits,
    selected_box_specs,
    sha256_file,
    signed_value,
    unique_directory,
    validate_boxes_against_dark,
    validate_scan_config,
    validate_segment,
)


DEFAULT_SCAN_CONFIG = (
    Path(__file__).resolve().parents[0]
    / "config"
    / "tiptilt_map_dm_wide.json"
)
DEFAULT_INJECTED_SEGMENTS = (11, 20, 31)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instrument-config",
        type=Path,
        default=DEFAULT_INSTRUMENT_CONFIG,
        help="Path to the aligned GUI instrument.json.",
    )
    parser.add_argument(
        "--scan-config",
        type=Path,
        default=DEFAULT_SCAN_CONFIG,
        help="Path to the separate DM-wide scan settings JSON.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate configuration and estimate storage without attaching to hardware.",
    )
    parser.add_argument("--yes", action="store_true", help="Skip the final interactive confirmation.")
    return parser.parse_args(argv)


def configured_injected_segments(instrument: dict[str, Any]) -> tuple[int, ...]:
    values = instrument.get("instrument", {}).get("injected_segments", DEFAULT_INJECTED_SEGMENTS)
    if not isinstance(values, list) and not isinstance(values, tuple):
        raise ValueError("instrument.injected_segments must be a list")
    segments = tuple(validate_segment(value) for value in values)
    if len(segments) != 3 or len(set(segments)) != 3:
        raise ValueError("instrument.injected_segments must contain three unique segment numbers")
    return segments


def raster_run_prefix(
    now: datetime,
    tips: np.ndarray,
    tilts: np.ndarray,
    piston_um: float,
) -> str:
    return (
        f"{now:%Y%m%d}_{now:%H%M%S}"
        f"_tip{signed_value(tips[0])}to{signed_value(tips[-1])}"
        f"_tilt{signed_value(tilts[0])}to{signed_value(tilts[-1])}"
        f"_piston{signed_value(piston_um)}"
        "_dm_wide"
    )


def scan_dm_wide_raster(
    *,
    apapane: Any,
    dm: DMController,
    held_segment: int,
    held_ptt: np.ndarray,
    rastered_piston_um: float,
    tips: np.ndarray,
    tilts: np.ndarray,
    dark: np.ndarray,
    nframes: int,
    specs: list[BoxSpec],
    settle_s: float,
    initial_settle_s: float,
    serpentine: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    crop_arrays, flux_arrays = allocate_raster_arrays(specs, len(tips), len(tilts), nframes)
    total = len(tips) * len(tilts)
    report_every = max(1, total // 20)
    for completed, (tip_index, tilt_index) in enumerate(
        grid_indices(len(tips), len(tilts), serpentine), start=1
    ):
        raster_ptt = np.asarray(
            [rastered_piston_um, tips[tip_index], tilts[tilt_index]], dtype=float
        )
        state = np.tile(raster_ptt, (N_DM_SEGMENTS, 1))
        state[held_segment] = held_ptt
        dm.set_full(
            state,
            action="standalone_set_dm_wide_tiptilt",
            target_summary={
                "held_segment": held_segment,
                "held_segment_ptt": ptt_dict(held_ptt),
                "other_36_segments": ptt_dict(raster_ptt),
            },
        )
        time.sleep(initial_settle_s if completed == 1 else settle_s)
        crops = acquire_box_crops(apapane, dark, nframes, specs)
        for spec in specs:
            crop = crops[spec.identifier]
            crop_arrays[spec.identifier][tip_index, tilt_index] = crop
            flux_arrays[spec.identifier][tip_index, tilt_index] = crop.sum(
                axis=(1, 2), dtype=np.float64
            )
        if completed == 1 or completed == total or completed % report_every == 0:
            print(f"  held segment {held_segment:02d}: {completed:4d}/{total} points")
    return crop_arrays, flux_arrays


def save_dm_wide_fits(
    path: Path,
    *,
    held_segment: int,
    held_ptt: np.ndarray,
    rastered_piston_um: float,
    repeat_index: int,
    nframes: int,
    tips: np.ndarray,
    tilts: np.ndarray,
    initial_dm: np.ndarray,
    specs: list[BoxSpec],
    crops: dict[str, np.ndarray],
    fluxes: dict[str, np.ndarray],
    dark_path: Path,
    detector_boxes_path: Path,
) -> None:
    """Use the shared atomic FITS writer with DM-wide metadata."""
    # The shared writer already stores every frame crop, flux cube, mean map,
    # axes, initial DM state, box coordinates and calibration paths.  Its
    # RESTPST header is the common piston of the 36 rastered segments; RESTTIP
    # and RESTTILT record the first raster point, while the complete axes define
    # their changing values.
    first_raster_ptt = np.asarray([rastered_piston_um, tips[0], tilts[0]], dtype=float)
    save_raster_fits(
        path,
        scan_type="DMWIDE",
        segment=held_segment,
        repeat_index=repeat_index,
        nframes=nframes,
        tips=tips,
        tilts=tilts,
        rest_ptt=first_raster_ptt,
        scanned_piston_um=float(held_ptt[0]),
        initial_dm=initial_dm,
        specs=specs,
        crops=crops,
        fluxes=fluxes,
        dark_path=dark_path,
        detector_boxes_path=detector_boxes_path,
        primary_metadata={
            "HELDSEG": int(held_segment),
            "HELDPST": float(held_ptt[0]),
            "HELDTIP": float(held_ptt[1]),
            "HELDTILT": float(held_ptt[2]),
            "RASTPST": float(rastered_piston_um),
        },
        extra_comments=(
            "HELDSEG remains at launch-time aligned PTT; all other 36 segments follow TIP_AXIS/TILT_AXIS.",
        ),
    )


def run(scan_args: argparse.Namespace) -> Path | None:
    instrument_path = scan_args.instrument_config.expanduser().resolve()
    scan_path = scan_args.scan_config.expanduser().resolve()
    instrument = load_json(instrument_path)
    scan = validate_scan_config(load_json(scan_path), "dm_wide_tiptilt")
    held_segments = configured_injected_segments(instrument)
    detector_path = resolve_detector_boxes_path(instrument, instrument_path)
    detector = load_detector_boxes(detector_path)
    specs = selected_box_specs(detector, scan.get("saved_boxes", {}))
    dark_path = resolve_apapane_dark(instrument, instrument_path)
    dark = load_dark(dark_path)
    validate_boxes_against_dark(specs, dark)
    output_parent = resolve_output_root(instrument, instrument_path) / scan["output_subdirectory"]
    npoints = len(scan["_tips"]) * len(scan["_tilts"])
    estimate = estimated_crop_bytes(
        specs,
        npoints,
        scan["_nframes"],
        len(held_segments),
        scan["_repeats"],
    )
    print_preflight(
        instrument_path=instrument_path,
        scan_path=scan_path,
        detector_path=detector_path,
        dark_path=dark_path,
        output_parent=output_parent,
        scan=scan,
        specs=specs,
        estimate=estimate,
        rasters=len(held_segments),
    )
    print(f"  held segments      : {held_segments}; PTT captured live at launch")
    if scan_args.check_only:
        return None
    output_parent.mkdir(parents=True, exist_ok=True)
    free_bytes = os.statvfs(output_parent).f_bavail * os.statvfs(output_parent).f_frsize
    reserve = float(scan.get("minimum_free_space_gb", 1.0)) * 1024**3
    if estimate + reserve > free_bytes:
        raise RuntimeError(
            f"Insufficient free space: need approximately {human_bytes(estimate + int(reserve))}, "
            f"but {human_bytes(free_bytes)} is available"
        )
    if not scan_args.yes:
        answer = input("Proceed with physical DM-wide commands? Type YES: ").strip()
        if answer != "YES":
            print("Cancelled before hardware attachment.")
            return None

    now = datetime.now().astimezone()
    prefix = raster_run_prefix(now, scan["_tips"], scan["_tilts"], scan["_rastered_piston"])
    run_directory = unique_directory(output_parent / f"{now:%Y%m%d}", prefix)
    run_directory.mkdir(parents=True)
    summary_path = run_directory / f"{run_directory.name}_summary.json"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "starting",
        "scan_type": "dm_wide_tiptilt",
        "started_local": now.isoformat(),
        "instrument_config": str(instrument_path),
        "scan_config": str(scan_path),
        "scan_settings": public_scan_config(scan),
        "held_segments": list(held_segments),
        "held_ptt_source": "launch-time live dmptt snapshot after instrument alignment",
        "detector_boxes_file": str(detector_path),
        "saved_boxes": [spec.as_dict() for spec in specs],
        "apapane_dark_file": str(dark_path),
        "apapane_dark_sha256": sha256_file(dark_path),
        "estimated_payload_bytes": estimate,
        "outputs": [],
    }
    atomic_json(summary_path, summary)

    logger = PersistentDMLogger(instrument, instrument_root(instrument_path))
    dm: DMController | None = None
    initial_dm: np.ndarray | None = None
    failure: BaseException | None = None
    restoration_error: BaseException | None = None
    try:
        apapane, dmptt_shm = attach_devices(instrument)
        dm = DMController(dmptt_shm, logger)
        initial_dm = dm.snapshot()
        summary["initial_dm_ptt"] = initial_dm.tolist()
        summary["held_segment_ptt"] = {
            str(segment): ptt_dict(initial_dm[segment]) for segment in held_segments
        }
        summary["status"] = "running"
        atomic_json(summary_path, summary)
        serpentine = bool(scan.get("serpentine", True))

        for repeat_index in range(1, scan["_repeats"] + 1):
            for held_segment in held_segments:
                held_ptt = initial_dm[held_segment].copy()
                print(
                    f"\nRepeat {repeat_index}/{scan['_repeats']}; holding aligned segment "
                    f"{held_segment} at {ptt_dict(held_ptt)}"
                )
                crops, fluxes = scan_dm_wide_raster(
                    apapane=apapane,
                    dm=dm,
                    held_segment=held_segment,
                    held_ptt=held_ptt,
                    rastered_piston_um=scan["_rastered_piston"],
                    tips=scan["_tips"],
                    tilts=scan["_tilts"],
                    dark=dark,
                    nframes=scan["_nframes"],
                    specs=specs,
                    settle_s=scan["_settle_s"],
                    initial_settle_s=scan["_initial_settle_s"],
                    serpentine=serpentine,
                )
                filename = (
                    f"{run_directory.name}_held{held_segment:02d}_repeat{repeat_index:02d}.fits"
                )
                file_path = run_directory / filename
                save_dm_wide_fits(
                    file_path,
                    held_segment=held_segment,
                    held_ptt=held_ptt,
                    rastered_piston_um=scan["_rastered_piston"],
                    repeat_index=repeat_index,
                    nframes=scan["_nframes"],
                    tips=scan["_tips"],
                    tilts=scan["_tilts"],
                    initial_dm=initial_dm,
                    specs=specs,
                    crops=crops,
                    fluxes=fluxes,
                    dark_path=dark_path,
                    detector_boxes_path=detector_path,
                )
                summary["outputs"].append(
                    {"held_segment": held_segment, "repeat": repeat_index, "fits": filename}
                )
                atomic_json(summary_path, summary)
                print(f"  saved {file_path}")
    except BaseException as exc:
        failure = exc
        summary["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if dm is not None and initial_dm is not None:
            try:
                dm.set_full(
                    initial_dm,
                    action="standalone_restore_launch_dm",
                    target_summary={"source": "launch-time live dmptt snapshot", "segments": 37},
                )
                summary["dm_restored"] = True
            except BaseException as exc:
                restoration_error = exc
                summary["dm_restored"] = False
                summary["restoration_error"] = f"{type(exc).__name__}: {exc}"
        summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
        if failure is None and restoration_error is None:
            summary["status"] = "completed"
        atomic_json(summary_path, summary)
        logger.close()

    if restoration_error is not None:
        raise RuntimeError(f"CRITICAL: launch-time DM restoration failed: {restoration_error}") from restoration_error
    if failure is not None:
        raise failure
    print(f"\nCompleted. Launch-time DM state restored. Outputs: {run_directory}")
    return run_directory


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted; restoration was attempted before exit.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
