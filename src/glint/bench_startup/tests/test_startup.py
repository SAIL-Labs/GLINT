from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


startup = load_module("restore_glint_bench", ROOT / "restore_glint_bench.py")
pg_service = load_module("glint_pg_service", ROOT / "glint_pg_service.py")


class FakeFc2Error(Exception):
    pass


class ErrorWithCode(Exception):
    errorCode = 20


class StartupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = ROOT / "config" / "bench_startup.json"
        self.config = startup.load_config(self.config_path)

    def test_configuration_loads_and_has_expected_host(self) -> None:
        self.assertEqual(self.config["host"], "scexao5")
        self.assertEqual(self.config["environment"]["DISPLAY"], ":3")

    def test_only_expected_tmux_sessions_are_configured(self) -> None:
        camera = self.config["camera"]
        names = {
            camera["controller_session"],
            camera["cameras"]["pg1"]["framegrab_session"],
            camera["cameras"]["pg2"]["framegrab_session"],
            self.config["dm"]["session"],
        }
        self.assertEqual(
            names, {"glintpg_ctrl", "glintpg1_fgrab", "glintpg2_fgrab", "glint_dm"}
        )

    def test_service_path_resolves_inside_package(self) -> None:
        path = startup.configured_path(
            self.config["camera"]["service_script"], self.config_path
        )
        self.assertEqual(path, ROOT / "glint_pg_service.py")
        self.assertTrue(path.is_file())

    def test_viewers_exactly_cover_their_framegrab_terminals(self) -> None:
        layout = self.config["windows"]["layout"]
        self.assertEqual(layout["glintpg1_fgrab"]["rect"], layout["glintpg1_viewer"]["rect"])
        self.assertEqual(layout["glintpg2_fgrab"]["rect"], layout["glintpg2_viewer"]["rect"])

    def test_all_rectangles_fit_screen(self) -> None:
        restorer = startup.BenchRestorer(
            self.config, self.config_path, no_windows=False, no_gui=False
        )
        restorer.validate_layout()

    def test_only_viewer_framegrab_pairs_overlap(self) -> None:
        layout = self.config["windows"]["layout"]

        def overlaps(left, right) -> bool:
            lx, ly, lw, lh = layout[left]["rect"]
            rx, ry, rw, rh = layout[right]["rect"]
            return lx < rx + rw and rx < lx + lw and ly < ry + rh and ry < ly + lh

        allowed = {
            frozenset(("glintpg1_fgrab", "glintpg1_viewer")),
            frozenset(("glintpg2_fgrab", "glintpg2_viewer")),
        }
        names = list(layout)
        observed = {
            frozenset((left, right))
            for index, left in enumerate(names)
            for right in names[index + 1 :]
            if overlaps(left, right)
        }
        self.assertEqual(observed, allowed)

    def test_tmux_name_validation_rejects_shell_content(self) -> None:
        with self.assertRaises(startup.StartupError):
            startup.Tmux.validate_name("glintpg_ctrl; reboot")

    def test_error20_recognition(self) -> None:
        self.assertTrue(pg_service.is_invalid_generation(FakeFc2Error("Fc2Error(20)")))
        self.assertTrue(pg_service.is_invalid_generation(ErrorWithCode("generation")))
        self.assertFalse(pg_service.is_invalid_generation(RuntimeError("other failure")))

    def test_no_automatic_dm_state_is_configured(self) -> None:
        self.assertIsNone(self.config["dm"]["startup_state_command"])

    def test_apapane_has_no_start_command(self) -> None:
        self.assertEqual(set(self.config["shm"]), {"apapane", "live_probe_timeout_s", "comment"})

    def test_config_is_valid_json(self) -> None:
        with self.config_path.open("r", encoding="utf-8") as handle:
            json.load(handle)


if __name__ == "__main__":
    unittest.main()
