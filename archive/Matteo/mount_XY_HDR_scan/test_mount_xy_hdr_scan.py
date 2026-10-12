from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mount_xy_hdr_scan import (
    SaturationDetected,
    _guarded_apapane_frames,
    build_exclusion_mask,
    latest_source_xy,
    safe_serpentine_route,
    validate_emergency_tilt_state,
)
from glint_alignment.models import SegmentPTT


def _write_xy(directory: Path, suffix: int, completed: str, peak: float = 10.0) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / f"06_xy_{suffix}"
    x = np.arange(5, dtype=float)
    y = np.arange(5, dtype=float)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    base = peak * np.exp(-0.5 * ((xx - 2) ** 2 + (yy - 2) ** 2))
    np.savez_compressed(
        stem.with_suffix(".npz"),
        axis__x=x,
        axis__y=y,
        **{f"map__Phot {index}": base * index for index in (1, 2, 3)},
    )
    payload = {
        "schema_version": 1,
        "kind": "xy",
        "completed_utc": completed,
        "axes": {"x": x.tolist(), "y": y.tolist()},
        "array_file": stem.with_suffix(".npz").name,
    }
    stem.with_suffix(".json").write_text(json.dumps(payload), encoding="utf-8")
    return stem.with_suffix(".json")


class LatestSourceTests(unittest.TestCase):
    def test_script_imports_refactor_from_separate_matteo_scan_folder(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            glint_root = Path(temporary) / "glint"
            scan_directory = glint_root / "Matteo" / "mount_XY_HDR_scan"
            scan_directory.mkdir(parents=True)
            (glint_root / "glint_alignment_refactor").symlink_to(
                project_root, target_is_directory=True
            )
            deployed_script = scan_directory / "mount_xy_hdr_scan.py"
            shutil.copy2(project_root / "mount_xy_hdr_scan.py", deployed_script)
            completed = subprocess.run(
                [sys.executable, str(deployed_script), "--help"],
                cwd=scan_directory,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_newest_valid_pair_wins_and_newer_incomplete_scan_is_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            older = _write_xy(
                root / "20260908" / "Alignment_outputs_9",
                2,
                "2026-09-08T23:00:00+00:00",
            )
            newer = _write_xy(
                root / "20260909" / "Alignment_output_individual",
                1,
                "2026-09-09T01:00:00+00:00",
            )
            incomplete = root / "20260910" / "Alignment_outputs_1" / "06_xy_1.json"
            incomplete.parent.mkdir(parents=True)
            incomplete.write_text(
                json.dumps(
                    {
                        "kind": "xy",
                        "completed_utc": "2026-09-10T01:00:00+00:00",
                        "array_file": "06_xy_1.fits",
                    }
                ),
                encoding="utf-8",
            )
            result = latest_source_xy(root, ("Phot 1", "Phot 2", "Phot 3"))
            self.assertEqual(result.json_path, newer.resolve())
            self.assertNotEqual(result.json_path, older.resolve())


class MaskAndRouteTests(unittest.TestCase):
    def test_unvalidated_emergency_tilt_state_is_rejected_before_motion(self):
        instrument = {
            "dm_states": {
                "tilted_away": {
                    "validated": False,
                    "default": {"piston_um": 0.0, "tip_mrad": 5.0, "tilt_mrad": 0.0},
                    "segments": {},
                }
            }
        }
        with self.assertRaisesRegex(RuntimeError, "not marked validated"):
            validate_emergency_tilt_state(instrument)

    def test_each_point_uses_one_probe_then_remaining_frames(self):
        class Hardware:
            def __init__(self):
                self.requests = []

            def apapane_frames(self, nframes):
                self.requests.append(nframes)
                return np.full((nframes, 4, 4), 12.0)

            def set_dm(self, state):
                raise AssertionError("Safe frames must not trigger emergency DM state")

        hardware = Hardware()
        _, frames, maximum = _guarded_apapane_frames(
            hardware,
            nframes=10,
            guard={"enabled": True, "regions": "full_frame"},
            guard_limit=80.0,
            emergency_dm_state={0: SegmentPTT()},
            label="test point",
        )
        self.assertEqual(hardware.requests, [1, 9])
        self.assertEqual(frames.shape, (10, 4, 4))
        self.assertEqual(maximum, 12.0)

    def test_saturated_probe_immediately_tilts_dm_and_skips_remaining_frames(self):
        emergency = {0: SegmentPTT(tip_mrad=5.0)}

        class Hardware:
            def __init__(self):
                self.requests = []
                self.commanded = None

            def apapane_frames(self, nframes):
                self.requests.append(nframes)
                return np.full((nframes, 4, 4), 90.0)

            def set_dm(self, state):
                self.commanded = state

        hardware = Hardware()
        with self.assertRaisesRegex(SaturationDetected, "immediately tilted away"):
            _guarded_apapane_frames(
                hardware,
                nframes=10,
                guard={"enabled": True, "regions": "full_frame"},
                guard_limit=80.0,
                emergency_dm_state=emergency,
                label="test point",
            )
        self.assertEqual(hardware.requests, [1])
        self.assertIs(hardware.commanded, emergency)

    def test_route_visits_every_safe_point_without_entering_or_crossing_mask(self):
        excluded = np.zeros((9, 11), dtype=bool)
        excluded[2:7, 4:7] = True
        route, acquisition = safe_serpentine_route(excluded)
        self.assertEqual(len(acquisition), int(np.count_nonzero(~excluded)))
        self.assertEqual(len(set(acquisition)), len(acquisition))
        self.assertTrue(all(not excluded[node] for node in route))
        for first, second in zip(route, route[1:]):
            self.assertEqual(abs(first[0] - second[0]) + abs(first[1] - second[1]), 1)

    def test_mask_rejects_threshold_contour_that_touches_reference_boundary(self):
        x = np.arange(3, dtype=float)
        y = np.arange(3, dtype=float)
        maps = {"Phot 1": np.ones((3, 3), dtype=float)}
        from mount_xy_hdr_scan import SourceXY

        source = SourceXY(
            Path("source.json"),
            Path("source.npz"),
            datetime.now(timezone.utc),
            {},
            x,
            y,
            maps,
        )
        with self.assertRaisesRegex(ValueError, "no positive dynamic range"):
            build_exclusion_mask(
                source,
                np.arange(5, dtype=float),
                np.arange(5, dtype=float),
                ("Phot 1",),
                threshold=0.05,
                background_percentile=5,
                dilation_pulses=1,
                allow_boundary_contact=False,
            )


if __name__ == "__main__":
    unittest.main()
