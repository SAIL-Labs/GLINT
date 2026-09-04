import queue
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import glint_alignment.instrument as instrument_module
import glint_alignment.gui as gui_module
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure

from glint_alignment.analysis import fit_null_envelope, solve_common_pistons
from glint_alignment.boxes import BOX_CONFIG_PATH, FRINGE_TRACKING_BOXES, NULL_BOXES, PHOT_BOXES
from glint_alignment.camera_storage import resolve_camera_directory, save_camera_frame
from glint_alignment.config import load_config, rest_dm_state, save_config, segment_state, validate_scan_defaults
from glint_alignment.engine import AlignmentEngine
from glint_alignment.engine import ScanCancelled
from glint_alignment.gui import (
    AlignmentGUI,
    GREEN,
    LABELS,
    assign_default_setting,
    dm_piston_mapping,
    dm_segment_rgba,
    flatten_default_settings,
    format_default_setting,
    format_piston_tick,
    parse_default_setting,
)
from glint_alignment.instrument import AXIS_NUMBER, InstrumentHardware
from glint_alignment.models import AxisGrid, ScanResult
from glint_alignment.null_live import collect_null_statistics, format_depth, process_apapane_frame
from glint_alignment.simulator import SimulatedHardware
from glint_alignment.suruga_mount import SurugaMount
from glint_alignment.storage import (
    begin_alignment,
    finalize_result,
    load_saved_dark,
    save_final_configuration,
    save_result,
    save_review_figure,
)
from glint_alignment.workflow_status import WorkflowStatus


class TestCore(unittest.TestCase):
    def setUp(self):
        self.config = load_config(simulation=True)
        self.config["command_logging"]["enabled"] = False
        self.config["time_scale"] = 0.0
        self.hardware = SimulatedHardware(self.config)
        self.engine = AlignmentEngine(self.hardware, self.config)

    def test_inclusive_grid(self):
        np.testing.assert_allclose(AxisGrid(300, 500, 50).values(), np.arange(300, 801, 50))
        self.assertEqual(len(AxisGrid(-1, 2, 0.1).values()), 21)
        with self.assertRaises(ValueError):
            AxisGrid(0, 1, 0.3).values()

    def test_default_configuration_assets_and_typed_editor_helpers(self):
        self.assertEqual(BOX_CONFIG_PATH.parent.name, "config")
        detector_boxes = json.loads(BOX_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(detector_boxes["photometry"], PHOT_BOXES)
        leaves = dict(flatten_default_settings({"axis": {"start": 1.5}, "crop": [1, 2, 3, 4]}))
        self.assertEqual(leaves[("axis", "start")], 1.5)
        self.assertEqual(parse_default_setting("7", 5), 7)
        self.assertEqual(parse_default_setting("2.5", 1.0), 2.5)
        self.assertEqual(parse_default_setting("false", True), False)
        self.assertEqual(parse_default_setting("[5, 6]", [1, 2]), [5, 6])
        self.assertEqual(format_default_setting([5, 6]), "[5, 6]")
        nested = {"axis": {"start": 1.0}}
        assign_default_setting(nested, ("axis", "start"), 3.0)
        self.assertEqual(nested["axis"]["start"], 3.0)
        self.assertNotIn("e", format_piston_tick(0.00002).lower())
        self.assertEqual(resolve_camera_directory("", self.config), Path(self.config["_root"]) / "simulated_camera_frames")
        without_camera_path = dict(self.config)
        without_camera_path.pop("camera_frame_root")
        with self.assertRaisesRegex(ValueError, "camera_frame_root"):
            resolve_camera_directory("", without_camera_path)

    def test_scan_defaults_are_validated_and_persisted_to_active_json(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "simulator.json"
            payload = {key: value for key, value in self.config.items() if not key.startswith("_")}
            path.write_text(json.dumps(payload), encoding="utf-8")
            config = load_config(path, simulation=True)
            config["scans"]["focal_dark"]["nframes"] = 9
            config["_alignment_session_dir"] = "/runtime/only"
            validate_scan_defaults(config)
            saved = save_config(config)
            self.assertEqual(saved, path.resolve())
            persisted = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["scans"]["focal_dark"]["nframes"], 9)
            self.assertFalse(any(key.startswith("_") for key in persisted))
            config["scans"]["pitch_yaw"]["pitch"]["start"] = 850.0
            with self.assertRaisesRegex(ValueError, "outside"):
                validate_scan_defaults(config)

    def test_workflow_defaults_update_is_immediate_and_persistent(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "simulator.json"
            payload = {key: value for key, value in self.config.items() if not key.startswith("_")}
            path.write_text(json.dumps(payload), encoding="utf-8")
            config = load_config(path, simulation=True)
            hardware = SimulatedHardware(config)
            gui = AlignmentGUI.__new__(AlignmentGUI)
            gui.config = config
            gui.hardware = hardware
            gui.engine = AlignmentEngine(hardware, config)
            variable = Mock()
            variable.get.return_value = "12"
            gui.workflow_default_fields = {"focal_dark": {("nframes",): (variable, 5)}}
            gui.current_kind = "focal_dark"
            gui.select_scan = Mock()
            gui.status = Mock()
            gui._append_log = Mock()
            gui._close_workflow_defaults = Mock()
            with patch.object(gui_module.messagebox, "showinfo"), patch.object(gui_module.messagebox, "showerror") as error:
                gui._update_workflow_defaults()
            self.assertFalse(error.called)
            self.assertEqual(gui.config["scans"]["focal_dark"]["nframes"], 12)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["scans"]["focal_dark"]["nframes"], 12)
            gui.select_scan.assert_called_once_with("focal_dark")

    def test_workflow_defaults_reject_path_changes_that_bypass_manual_json_editing(self):
        gui = AlignmentGUI.__new__(AlignmentGUI)
        gui.config = self.config
        gui.hardware = self.hardware
        gui.engine = self.engine
        original = self.config["scans"]["bulk"]["response_matrix"]
        variable = Mock()
        variable.get.return_value = "outside_config.npy"
        gui.workflow_default_fields = {"bulk": {("response_matrix",): (variable, original)}}
        gui.current_kind = "bulk"
        gui.status = Mock()
        gui._append_log = Mock()
        gui._close_workflow_defaults = Mock()
        with patch.object(gui_module.messagebox, "showerror") as error:
            gui._update_workflow_defaults()
        error.assert_called_once()
        self.assertIn("manually", error.call_args.args[1])
        self.assertEqual(self.config["scans"]["bulk"]["response_matrix"], original)

    def test_bulk_converges_and_restores_until_apply(self):
        initial = self.hardware.get_conex()
        self.engine.acquire_dark("focal", 25)
        result = self.engine.bulk_align()
        self.assertTrue(result.recommendation["passed"])
        self.assertLess(result.recommendation["error_px"], 2.0)
        self.assertEqual(self.hardware.get_conex(), initial)
        np.testing.assert_allclose(self.hardware.get_dm_volts(), 0.0)
        self.engine.apply(result)
        self.assertAlmostEqual(self.hardware.get_conex()["u"], 0.08, places=2)

    def test_mount_and_tip_tilt_recommendations(self):
        self.engine.acquire_dark("apapane", 25)
        py = self.engine.pitch_yaw_scan()
        self.assertTrue(py.recommendation["passed"])
        self.assertEqual([item["channel"] for item in py.estimates["channels"]], list(PHOT_BOXES))
        self.assertTrue(490 < py.recommendation["pitch"] < 550)
        self.assertTrue(120 < py.recommendation["yaw"] < 190)
        xy = self.engine.xy_scan()
        self.assertTrue(xy.recommendation["passed"])
        self.assertTrue(3025 < xy.recommendation["x"] < 3050)
        tt = self.engine.tip_tilt_scan()
        self.assertAlmostEqual(tt.recommendation["segments"]["11"]["tip_mrad"], 0.20, delta=0.03)

    def test_scan_specific_dm_states_are_commanded(self):
        self.engine.acquire_dark("apapane", 3)
        zero = segment_state(self.config, "zero_ptt")
        away = segment_state(self.config, "tilted_away")
        self.hardware.set_dm(zero)
        commanded = []
        original_set_dm = self.hardware.set_dm

        def record(state):
            commanded.append({segment: type(ptt)(ptt.piston_um, ptt.tip_mrad, ptt.tilt_mrad) for segment, ptt in state.items()})
            original_set_dm(state)

        self.hardware.set_dm = record

        self.engine.pitch_yaw_scan(
            {
                **self.config["scans"]["pitch_yaw"],
                "pitch": {"start": 500, "span": 100, "step": 50},
                "yaw": {"start": 100, "span": 100, "step": 50},
            }
        )
        self.assertEqual(commanded[0], zero)

        commanded.clear()
        xy_result = self.engine.xy_scan(
            {
                **self.config["scans"]["xy"],
                "x": {"start": 3030, "span": 20, "step": 10},
                "y": {"start": 3430, "span": 20, "step": 10},
            }
        )
        xy_state = commanded[0]
        for segment in (11, 20, 31):
            self.assertEqual(xy_state[segment], zero[segment])
        self.assertEqual(xy_state[0], away[0])
        self.assertEqual(self.hardware.get_dm(), xy_state)
        self.assertTrue(xy_result.metadata["dm_ptt_retained_after_scan"])

        self.engine.restore_review_state(xy_result)
        self.assertEqual(self.hardware.get_dm(), zero)

        commanded.clear()
        self.engine.tip_tilt_scan(
            {
                **self.config["scans"]["tip_tilt"],
                "tip": {"start": -0.1, "span": 0.2, "step": 0.1},
                "tilt": {"start": -0.1, "span": 0.2, "step": 0.1},
                "nframes": 1,
            }
        )
        first_tt_state = commanded[0]
        self.assertEqual(first_tt_state[11], zero[11])
        self.assertEqual(first_tt_state[20], away[20])
        self.assertEqual(first_tt_state[31], away[31])

        commanded.clear()
        self.engine.null_scan(
            {
                **self.config["scans"]["null"],
                "piston": {"start": -1.3, "span": 2.6, "step": 0.65},
                "nframes": 1,
                "numavg": 1,
            }
        )
        first_null_state = commanded[0]
        self.assertAlmostEqual(first_null_state[11].piston_um, -1.3)
        self.assertAlmostEqual(first_null_state[20].piston_um, 1.3)
        self.assertEqual(first_null_state[31], away[31])
        # Each 5-point ascending leg is followed by the other segment's
        # 4-point descending leg; the zero-OPD endpoint is not duplicated.
        self.assertEqual(len(commanded), 19)  # 18 scan points plus guarded pre-scan restore
        self.assertAlmostEqual(commanded[4][11].piston_um, 1.3)
        self.assertAlmostEqual(commanded[4][20].piston_um, 1.3)
        self.assertAlmostEqual(commanded[5][11].piston_um, 1.3)
        self.assertAlmostEqual(commanded[5][20].piston_um, 0.65)
        self.assertAlmostEqual(commanded[8][20].piston_um, -1.3)
        self.assertAlmostEqual(commanded[9][11].piston_um, -1.3)
        self.assertAlmostEqual(commanded[9][31].piston_um, 1.3)

    def test_apply_commands_and_retains_the_complete_stage_dm_state(self):
        zero = segment_state(self.config, "zero_ptt")
        away = segment_state(self.config, "tilted_away")
        self.hardware.set_dm(away)

        pitch_yaw = ScanResult(
            kind="pitch_yaw",
            recommendation={"pitch": 520.0, "yaw": 150.0, "passed": True},
        )
        self.engine.apply(pitch_yaw)
        self.assertEqual(self.hardware.get_dm(), zero)
        self.assertEqual(pitch_yaw.metadata["applied_dm_ptt_state"], "zero_ptt")

        xy = ScanResult(
            kind="xy",
            recommendation={"x": 3040.0, "y": 3440.0, "passed": True},
        )
        self.engine.apply(xy)
        xy_state = self.hardware.get_dm()
        for segment in (11, 20, 31):
            self.assertEqual(xy_state[segment], zero[segment])
        for segment in (0, 10, 36):
            self.assertEqual(xy_state[segment], away[segment])

        recommended = {
            "11": {"piston_um": 0.0, "tip_mrad": 0.20, "tilt_mrad": -0.10},
            "20": {"piston_um": 0.0, "tip_mrad": -0.15, "tilt_mrad": 0.25},
            "31": {"piston_um": 0.0, "tip_mrad": 0.05, "tilt_mrad": 0.30},
        }
        tip_tilt = ScanResult(
            kind="tip_tilt",
            recommendation={"segments": recommended, "passed": True},
        )
        self.engine.apply(tip_tilt)
        tip_tilt_state = self.hardware.get_dm()
        for segment in (0, 10, 36):
            self.assertEqual(tip_tilt_state[segment], away[segment])
        for key, values in recommended.items():
            segment = int(key)
            self.assertAlmostEqual(tip_tilt_state[segment].tip_mrad, values["tip_mrad"])
            self.assertAlmostEqual(tip_tilt_state[segment].tilt_mrad, values["tilt_mrad"])

        null = ScanResult(
            kind="null",
            recommendation={
                "pistons_um": {"11": -0.35, "20": 0.15, "31": 0.40},
                "passed": True,
            },
        )
        self.engine.apply(null)
        final_state = self.hardware.get_dm()
        for segment in (0, 10, 36):
            self.assertEqual(final_state[segment], away[segment])
        for key, piston in null.recommendation["pistons_um"].items():
            segment = int(key)
            self.assertAlmostEqual(final_state[segment].piston_um, piston)
            self.assertAlmostEqual(final_state[segment].tip_mrad, tip_tilt_state[segment].tip_mrad)
            self.assertAlmostEqual(final_state[segment].tilt_mrad, tip_tilt_state[segment].tilt_mrad)
        self.assertEqual(
            null.metadata["applied_dm_ptt_state"],
            "selected_rest_of_dm_except_injected_recommended_ptt",
        )

    def test_new_full_workflow_resets_physical_dm_to_zero_ptt(self):
        gui = AlignmentGUI.__new__(AlignmentGUI)
        gui.running = False
        gui.config = self.config
        gui.hardware = self.hardware
        gui.engine = self.engine
        gui.workflow = WorkflowStatus(LABELS)
        for kind in LABELS:
            gui.workflow.finish(kind, True)
        gui.status = Mock()
        gui.refresh_dm = Mock()
        gui.refresh_mount = Mock()
        gui._update_step_styles = Mock()
        gui._run_next_workflow_step = Mock()
        gui._append_log = Mock()
        self.hardware.set_dm(segment_state(self.config, "tilted_away"))

        with patch.object(gui_module.messagebox, "askokcancel", return_value=True), patch.object(
            gui_module, "begin_alignment", return_value=Path("alignment_output")
        ):
            gui.run_full_workflow()

        self.assertEqual(self.hardware.get_dm(), segment_state(self.config, "zero_ptt"))
        self.assertTrue(all(gui.workflow.state(kind) == "pending" for kind in LABELS))
        gui.refresh_dm.assert_called_once()
        gui.refresh_mount.assert_called_once()
        gui._run_next_workflow_step.assert_called_once()

    def test_dm_pane_refreshes_while_scan_is_running(self):
        gui = AlignmentGUI.__new__(AlignmentGUI)
        gui.events = queue.Queue()
        gui.config = self.config
        gui.running = True
        gui.scan_running = True
        gui.current_kind = "tip_tilt"
        gui._last_live_dm_refresh = 0.0
        gui.refresh_dm = Mock()
        gui.refresh_mount = Mock()
        gui._append_log = Mock()
        gui.root = Mock()

        gui._drain_events()

        gui.refresh_dm.assert_called_once()
        gui.refresh_mount.assert_not_called()
        gui.root.after.assert_called_once()

    def test_mount_scan_does_not_issue_background_suruga_status_reads(self):
        gui = AlignmentGUI.__new__(AlignmentGUI)
        gui.events = queue.Queue()
        gui.config = self.config
        gui.running = True
        gui.scan_running = True
        gui.current_kind = "pitch_yaw"
        gui._last_live_dm_refresh = 0.0
        gui.refresh_dm = Mock()
        gui.refresh_mount = Mock()
        gui._append_log = Mock()
        gui.root = Mock()

        gui._drain_events()

        gui.refresh_dm.assert_called_once()
        gui.refresh_mount.assert_not_called()

    def test_rest_of_dm_map_is_applied_physically_and_used_by_xy(self):
        current = segment_state(self.config, "zero_ptt")
        current[11].tip_mrad = 0.21
        current[20].tilt_mrad = -0.17
        current[31].piston_um = 0.33
        injected_before = {segment: current[segment] for segment in (11, 20, 31)}
        self.hardware.set_dm(current)

        selection = self.engine.set_rest_dm_map(
            "all_tilted",
            uniform_piston_um=0.42,
            uniform_tip_mrad=3.25,
            uniform_tilt_mrad=-1.5,
        )
        commanded = self.hardware.get_dm()
        self.assertEqual(selection["map"], "all_tilted")
        for segment in range(37):
            if segment in injected_before:
                self.assertEqual(commanded[segment], injected_before[segment])
            else:
                self.assertAlmostEqual(commanded[segment].piston_um, 0.42)
                self.assertAlmostEqual(commanded[segment].tip_mrad, 3.25)
                self.assertAlmostEqual(commanded[segment].tilt_mrad, -1.5)

        xy_state = self.engine._xy_dm_state()
        zero = segment_state(self.config, "zero_ptt")
        for segment in (11, 20, 31):
            self.assertEqual(xy_state[segment], zero[segment])
        for segment in (0, 10, 36):
            self.assertAlmostEqual(xy_state[segment].piston_um, 0.42)
            self.assertAlmostEqual(xy_state[segment].tip_mrad, 3.25)
            self.assertAlmostEqual(xy_state[segment].tilt_mrad, -1.5)

        self.engine.set_rest_dm_map("science_operation")
        science = rest_dm_state(self.config, "science_operation")
        after_science = self.hardware.get_dm()
        for segment in (0, 10, 36):
            self.assertEqual(after_science[segment], science[segment])
        for segment in (11, 20, 31):
            self.assertEqual(after_science[segment], injected_before[segment])

    def test_whole_dm_presets_use_configured_ptt_zero_voltage_and_flat_voltage_states(self):
        self.hardware.set_dm(segment_state(self.config, "tilted_away"))
        self.engine.apply_dm_preset("zero_ptt")
        self.assertEqual(self.hardware.get_dm(), segment_state(self.config, "zero_ptt"))

        self.hardware.set_dm_volts(np.full(111, 0.8, dtype=np.float32))
        self.engine.apply_dm_preset("zero_volts")
        np.testing.assert_allclose(self.hardware.get_dm_volts(), 0.0)

        self.engine.apply_dm_preset("flat_volts")
        np.testing.assert_allclose(
            self.hardware.get_dm_volts(),
            self.config["dm_voltage_states"]["flat"]["all_actuators_volts"],
        )
        self.config["dm_voltage_states"]["flat"]["validated"] = False
        with self.assertRaisesRegex(RuntimeError, "not validated"):
            self.engine.apply_dm_preset("flat_volts")

    def test_set_z_applies_only_after_review_and_enforces_fixed_roll(self):
        self.hardware.mount["roll"] = 123.0
        initial_z = self.hardware.get_mount()["z"]
        result = self.engine.set_z_scan({"z": 1900.0})
        self.assertEqual(self.hardware.get_mount()["z"], initial_z)
        self.assertEqual(result.recommendation["roll"], self.config["mount_control"]["fixed_roll"])

        self.engine.apply(result)

        mount = self.hardware.get_mount()
        self.assertEqual(mount["z"], 1900.0)
        self.assertEqual(mount["roll"], self.config["mount_control"]["fixed_roll"])
        with self.assertRaisesRegex(ValueError, "Roll is fixed"):
            self.engine._move_mount({"roll": self.config["mount_control"]["fixed_roll"] + 1})

    def test_set_z_is_after_apapane_dark_and_review_is_text_only(self):
        self.assertEqual(list(LABELS), [
            "focal_dark",
            "apapane_dark",
            "bulk",
            "set_z",
            "pitch_yaw",
            "xy",
            "tip_tilt",
            "null",
        ])
        self.assertEqual(self.config["scans"]["focal_dark"]["nframes"], 5)
        self.assertEqual(self.config["scans"]["apapane_dark"]["nframes"], 1000)
        result = self.engine.set_z_scan({"z": 1875})
        gui = AlignmentGUI.__new__(AlignmentGUI)
        gui.figure = Figure(figsize=(8.6, 4.8), dpi=100)
        gui.canvas = FigureCanvasAgg(gui.figure)
        gui._plot_result(result)
        text = [item.get_text() for axis in gui.figure.axes for item in axis.texts]
        self.assertEqual(text, ["Z SET TO 1875"])

    def test_dark_light_source_prompts_are_enabled_in_the_local_gui(self):
        gui = AlignmentGUI.__new__(AlignmentGUI)
        gui.running = False
        gui.engine = self.engine
        gui._parameters = Mock(return_value={"nframes": 25})
        for kind, camera_name in (("focal_dark", "focal-plane"), ("apapane_dark", "APAPANE")):
            gui.current_kind = kind
            with patch.object(gui_module.messagebox, "askokcancel", return_value=False) as prompt:
                gui.run_selected()
            prompt.assert_called_once()
            self.assertIn(camera_name, prompt.call_args.args[1])

        gui.hardware = self.hardware
        gui.config = self.config
        gui._set_running = Mock()
        gui.refresh_dm = Mock()
        gui.refresh_mount = Mock()
        gui.workflow = Mock()
        gui._update_step_styles = Mock()
        gui.progress = {}
        gui.progress_text = Mock()
        gui.status = Mock()
        gui._plot_dark_status = Mock()
        gui.workflow_queue = []
        gui.full_workflow_active = False
        for kind, camera_name in (("focal_dark", "Focal-Plane"), ("apapane_dark", "Apapane")):
            result = ScanResult(kind=kind, metadata={"nframes": 25})
            with patch.object(gui_module, "save_result", return_value=Path("saved")), patch.object(
                gui_module.messagebox, "showinfo"
            ) as prompt:
                gui._handle_result(result)
            prompt.assert_called_once()
            self.assertIn(camera_name, prompt.call_args.args[1])
            self.assertIn("Turn the light source ON", prompt.call_args.args[1])

    def test_instrument_adapter_uses_configured_shm_and_packaged_mount_driver(self):
        instrument_config = load_config(simulation=False)
        command_log_temp = tempfile.TemporaryDirectory()
        self.addCleanup(command_log_temp.cleanup)
        instrument_config["command_logging"] = {"enabled": True, "root": command_log_temp.name}
        flat_path = Path(command_log_temp.name) / "configured_flat.txt"
        np.savetxt(flat_path, np.full(111, 0.37, dtype=np.float32))
        instrument_config["dm_voltage_states"]["flat"] = {
            "calibration_file": str(flat_path),
            "validated": True,
        }
        shm_arrays = {
            "dmptt": np.zeros(111, dtype=np.float32),
            "dmvolt": np.full(111, 0.25, dtype=np.float32),
            "apapane": np.zeros((280, 280), dtype=np.float32),
            "palila": np.zeros((320, 320), dtype=np.float32),
            "glintpg1": np.zeros((420, 420), dtype=np.float32),
            "glintpg2": np.zeros((850, 850), dtype=np.float32),
        }

        class FakeSHM:
            created_names = []

            def __init__(self, name):
                self.name = name
                type(self).created_names.append(name)

            def get_data(self):
                return shm_arrays[self.name].copy()

            def set_data(self, values):
                shm_arrays[self.name] = np.asarray(values).copy()

            def multi_recv_data(self, nframes):
                return np.repeat(shm_arrays[self.name][None, ...], int(nframes), axis=0)

        class FakeMount:
            created_with = None

            def __init__(self, port, baudrate, *, read_timeout_s, write_timeout_s):
                type(self).created_with = (port, baudrate, read_timeout_s, write_timeout_s)
                self.positions = {axis: 0 for axis in AXIS_NUMBER.values()}
                self.closed = False

            def get_pos(self, axis):
                return self.positions[axis]

            def set_pos(self, axis, position):
                self.positions[axis] = int(position)

            def in_motion(self, axis):
                return False

            def rstop(self):
                return None

            def lim_enabled(self, axis, direction):
                return True

            def get_lim(self, axis, direction):
                return -5000 if direction == "CCW" else 5000

            def close(self):
                self.closed = True

        modules = {
            "pyMilk.interfacing.shm": SimpleNamespace(SHM=FakeSHM),
        }

        def import_runtime(name):
            return modules[name]

        def conex_command(command, **kwargs):
            self.assertTrue(kwargs["check"])
            self.assertTrue(kwargs["capture_output"])
            self.assertTrue(kwargs["text"])
            if command[2] == "status":
                return SimpleNamespace(stdout=f"{command[1]} position 0.125\n")
            return SimpleNamespace(stdout="")

        with patch.object(instrument_module.subprocess, "run", side_effect=conex_command) as run, patch.object(
            instrument_module.importlib, "import_module", side_effect=import_runtime
        ) as import_module, patch.object(instrument_module, "SurugaMount", FakeMount):
            hardware = InstrumentHardware(instrument_config)
            self.assertEqual(
                FakeMount.created_with,
                (
                    instrument_config["instrument"]["mount"]["port"],
                    instrument_config["instrument"]["mount"]["baudrate"],
                    instrument_config["instrument"]["mount"]["read_timeout_s"],
                    instrument_config["instrument"]["mount"]["write_timeout_s"],
                ),
            )
            import_module.assert_called_once_with("pyMilk.interfacing.shm")
            self.assertEqual(
                FakeSHM.created_names[:2],
                [
                    instrument_config["instrument"]["dm"]["ptt_shm"],
                    instrument_config["instrument"]["dm"]["voltage_shm"],
                ],
            )
            hardware.move_mount({"pitch": 12, "roll": 1400, "z": 1800}, timeout_s=1.0)
            self.assertEqual(hardware.get_mount()["pitch"], 12.0)
            self.assertEqual(hardware.get_mount()["z"], 1800.0)
            self.assertEqual(
                hardware.mount_limits(),
                {
                    "pitch": (-500.0, 800.0),
                    "roll": (1000.0, 2000.0),
                    "yaw": (-500.0, 900.0),
                    "x": (-7500.0, 8000.0),
                    "z": (1100.0, 2500.0),
                    "y": (-4000.0, 9000.0),
                },
            )
            hardware.set_dm(segment_state(instrument_config, "zero_ptt"))
            np.testing.assert_allclose(hardware.get_dm_volts(), 0.25)
            hardware.zero_dm_volts()
            np.testing.assert_allclose(hardware.get_dm_volts(), 0.0)
            hardware.flatten_dm_volts()
            np.testing.assert_allclose(hardware.get_dm_volts(), 0.37)
            self.assertEqual(hardware.apapane_frames(2).shape, (2, 280, 280))
            self.assertEqual(hardware.camera_frames("focal", 1).shape, (1, 850, 850))
            self.assertEqual(hardware.get_conex(), {"u": 0.125, "v": 0.125})
            hardware.move_conex({"u": 0.2, "v": -0.1})
            commands = [call.args[0] for call in run.call_args_list]
            self.assertIn(["glint_steering2", "u", "status"], commands)
            self.assertIn(["glint_steering2", "v", "goto", "-0.1"], commands)
            self.assertFalse(any("glint_steering1" in command for command in commands))
            logged_commands = []
            for filename in ("dm_commands.jsonl", "mount_commands.jsonl", "conex_commands.jsonl"):
                logged_commands.extend(
                    json.loads(line)
                    for line in (hardware.command_logger.session_directory / filename).read_text(encoding="utf-8").splitlines()
                )
            self.assertTrue(any(record["device"] == "dm" for record in logged_commands))
            self.assertTrue(any(record["device"] == "mount" for record in logged_commands))
            self.assertTrue(any(record["device"] == "glint_steering2" for record in logged_commands))
            self.assertFalse(
                any(
                    record["device"] == "glint_steering1" and record["record_type"] != "configuration"
                    for record in logged_commands
                )
            )
            call_count = run.call_count
            hardware.config["steering_mirrors"]["controlled_device"] = "glint_steering1"
            with self.assertRaisesRegex(RuntimeError, "Only glint_steering2"):
                hardware.move_conex({"u": 0.0})
            self.assertEqual(run.call_count, call_count)
            hardware.close()
            self.assertTrue(hardware.mount_device.closed)

    def test_packaged_suruga_driver_preserves_legacy_command_bytes(self):
        class FakeSerial:
            responses = {
                b"AXI4:POS?\r": b"1234\r",
                b"AXI2:MOTION?\r": b"0\r",
                b"AXI1:CWSLE?\r": b"1\r",
                b"AXI1:CCWSLP?\r": b"-500\r",
            }

            def __init__(self, **settings):
                self.settings = settings
                self.is_open = True
                self.writes = []
                self.last_write = None

            def open(self):
                self.is_open = True

            def close(self):
                self.is_open = False

            def write(self, data):
                self.last_write = bytes(data)
                self.writes.append(self.last_write)
                return len(data)

            def read_until(self, expected=b"\n"):
                self.asserted_terminator = expected
                return self.responses[self.last_write]

        mount = SurugaMount(
            "/dev/fake-suruga",
            38400,
            read_timeout_s=1.25,
            write_timeout_s=1.5,
            serial_factory=FakeSerial,
        )
        self.assertEqual(mount.get_pos(4), 1234)
        mount.set_pos(6, -4000)
        self.assertFalse(mount.in_motion(2))
        self.assertTrue(mount.lim_enabled(1, "CW"))
        self.assertEqual(mount.get_lim(1, "CCW"), -500)
        mount.rstop()
        mount.estop()
        self.assertEqual(
            mount.serial.writes,
            [
                b"AXI4:POS?\r",
                b"AXI6:GOABS -4000\r",
                b"AXI2:MOTION?\r",
                b"AXI1:CWSLE?\r",
                b"AXI1:CCWSLP?\r",
                b"STOP 1\r",
                b"STOP 0\r",
            ],
        )
        self.assertEqual(mount.serial.asserted_terminator, b"\r")
        self.assertEqual(mount.serial.settings["port"], "/dev/fake-suruga")
        self.assertEqual(mount.serial.settings["baudrate"], 38400)
        self.assertEqual(mount.serial.settings["timeout"], 1.25)
        self.assertEqual(mount.serial.settings["write_timeout"], 1.5)
        mount.close()
        self.assertFalse(mount.serial.is_open)

    def test_mount_move_skips_unchanged_axes_and_avoids_position_queries_while_moving(self):
        class CountingMount:
            def __init__(self):
                self.positions = {1: 0, 2: 1400, 3: 0}
                self.calls = []

            def set_pos(self, axis, position):
                self.calls.append(("set", axis, position))
                self.positions[axis] = position

            def in_motion(self, axis):
                self.calls.append(("motion", axis))
                return False

            def get_pos(self, axis):
                self.calls.append(("position", axis))
                return self.positions[axis]

            def rstop(self):
                self.calls.append(("stop",))

        hardware = InstrumentHardware.__new__(InstrumentHardware)
        hardware.mount_device = CountingMount()
        hardware._mount_lock = instrument_module.RLock()
        hardware._mount_cache = {"pitch": 0, "roll": 1400, "yaw": 0}
        hardware._mount_poll_interval_s = 0.0
        hardware.command_logger = Mock()

        readback = hardware.move_mount({"pitch": 50, "yaw": 25, "roll": 1400}, timeout_s=1.0)

        self.assertEqual(readback, {"pitch": 50.0, "yaw": 25.0, "roll": 1400.0})
        self.assertEqual(
            hardware.mount_device.calls,
            [
                ("set", 1, 50),
                ("set", 3, 25),
                ("motion", 1),
                ("motion", 3),
                ("position", 1),
                ("position", 3),
            ],
        )
        hardware.mount_device.calls.clear()
        hardware.move_mount({"pitch": 50, "yaw": 25, "roll": 1400}, timeout_s=1.0)
        self.assertEqual(hardware.mount_device.calls, [])
        hardware.command_logger.record.assert_called_once()

    def test_mount_raster_does_not_read_all_six_axes_at_every_sample(self):
        config = load_config(simulation=True)
        config["command_logging"]["enabled"] = False
        config["time_scale"] = 0.0

        class CountingHardware(SimulatedHardware):
            def __init__(self, settings):
                super().__init__(settings)
                self.mount_reads = 0

            def get_mount(self):
                self.mount_reads += 1
                return super().get_mount()

        hardware = CountingHardware(config)
        engine = AlignmentEngine(hardware, config)
        engine.apapane_dark = np.zeros((280, 280), dtype=float)
        result = engine.pitch_yaw_scan(
            {
                **config["scans"]["pitch_yaw"],
                "pitch": {"start": 300, "span": 100, "step": 50},
                "yaw": {"start": -100, "span": 100, "step": 50},
            }
        )

        self.assertEqual(result.metadata["sample_count"], 9)
        self.assertEqual(result.metadata["raster_path"], "serpentine")
        self.assertLessEqual(hardware.mount_reads, 2)

    def test_three_persistent_device_command_logs_without_per_scan_copies(self):
        with tempfile.TemporaryDirectory() as temp:
            config = load_config(simulation=True)
            config["time_scale"] = 0.0
            config["output_root"] = str(Path(temp) / "results")
            config["command_logging"] = {"enabled": True, "root": str(Path(temp) / "command_logs")}
            hardware = SimulatedHardware(config)
            engine = AlignmentEngine(hardware, config)

            result = engine.set_z_scan({"z": 1900.0})
            hardware.set_dm(segment_state(config, "zero_ptt"))
            engine.apply(result)
            hardware.move_conex({"u": 0.1, "v": -0.2})
            directory = save_result(result, hardware, config, "applied")

            session = hardware.command_logger.session_directory
            self.assertIsNotNone(session)
            self.assertEqual(
                {path.name for path in session.iterdir()},
                {"dm_commands.jsonl", "mount_commands.jsonl", "conex_commands.jsonl"},
            )
            for filename in ("dm_commands.jsonl", "mount_commands.jsonl", "conex_commands.jsonl"):
                self.assertTrue((session / filename).exists())

            all_records = []
            for filename in ("dm_commands.jsonl", "mount_commands.jsonl", "conex_commands.jsonl"):
                all_records.extend(json.loads(line) for line in (session / filename).read_text(encoding="utf-8").splitlines())
            self.assertTrue(any(record["device"] == "dm" for record in all_records))
            self.assertTrue(any(record["device"] == "mount" for record in all_records))
            self.assertTrue(any(record["device"] == "glint_steering2" for record in all_records))
            self.assertFalse(
                any(record["device"] == "glint_steering1" and record["record_type"] == "command" for record in all_records)
            )

            summary = json.loads((directory / "04_set_z_1.json").read_text(encoding="utf-8"))
            exported = summary["metadata"]["command_log"]["files"]
            self.assertEqual(set(exported.values()), {"dm_commands.jsonl", "mount_commands.jsonl", "conex_commands.jsonl"})
            self.assertFalse(any(path.name.startswith("command_log_") for path in directory.iterdir()))

            last_sequence = max(record["sequence"] for record in all_records)
            hardware_again = SimulatedHardware(config)
            self.assertEqual(hardware_again.command_logger.session_directory, session)
            hardware_again.move_mount({"x": 3040.0}, timeout_s=1.0)
            newest = json.loads((session / "mount_commands.jsonl").read_text(encoding="utf-8").splitlines()[-1])
            self.assertGreater(newest["sequence"], last_sequence)
            hardware.close()
            hardware_again.close()

    def test_null_fit_and_common_pistons(self):
        x = np.linspace(-5.2, 5.2, 209)
        y = 0.16 + 0.115 * np.sinc((x - 0.64) / 5.6) * np.sin(
            2 * np.pi * (x - 0.64) / 1.55 - np.pi / 2
        )
        fit = fit_null_envelope(x, y)
        self.assertEqual(fit["model"], "sinc × sine")
        self.assertTrue(fit["fit_ok"])
        self.assertGreater(fit["r2"], 0.99)
        self.assertAlmostEqual(fit["parameters"][5], 1.55, delta=0.02)
        self.assertAlmostEqual(fit["best"]["delta_piston_um"], 0.32, delta=0.04)
        solution = solve_common_pistons(
            [{"delta_piston_um": 0.32, "metric": 0.04}],
            [{"delta_piston_um": -0.46, "metric": 0.05}],
        )
        self.assertTrue(solution["feasible"])
        self.assertAlmostEqual(solution["pistons_um"]["20"], -solution["pistons_um"]["31"])
        self.assertAlmostEqual(solution["pistons_um"]["11"] - solution["pistons_um"]["20"], 0.32)
        self.assertAlmostEqual(solution["pistons_um"]["11"] - solution["pistons_um"]["31"], -0.46)
        for piston in solution["pistons_um"].values():
            self.assertTrue(-1.3 <= piston <= 1.3)

    def test_null_fit_rejects_a_drifting_nonfringing_trace(self):
        x = np.linspace(-5.2, 5.2, 209)
        rng = np.random.default_rng(3)
        y = 0.42 + 0.004 * x + rng.normal(0.0, 0.012, len(x))
        fit = fit_null_envelope(x, y)
        self.assertFalse(fit["fit_ok"])
        self.assertLess(fit["r2"], 0.35)
        self.assertIn("Poor sinc", fit["fit_warning"])

    def test_null_scan_refuses_solution_when_either_baseline_fit_fails(self):
        self.engine.acquire_dark("apapane", 1)
        good = {
            "model": "sinc × sine",
            "fit_ok": True,
            "fit_warning": "",
            "r2": 0.9,
            "fit_opd": np.asarray([-0.4, 0.0, 0.4]),
            "fit_metric": np.asarray([0.2, 0.1, 0.2]),
            "candidates": [{"opd_um": 0.0, "delta_piston_um": 0.0, "metric": 0.1}],
            "best": {"opd_um": 0.0, "delta_piston_um": 0.0, "metric": 0.1},
        }
        bad = {
            **good,
            "fit_ok": False,
            "fit_warning": "no coherent packet",
            "r2": 0.02,
            "candidates": [],
            "best": None,
        }
        parameters = {
            **self.config["scans"]["null"],
            "piston": {"start": -0.1, "span": 0.2, "step": 0.1},
            "nframes": 1,
            "numavg": 1,
        }
        with patch("glint_alignment.engine.fit_null_envelope", side_effect=[good, bad]):
            result = self.engine.null_scan(parameters)
        self.assertFalse(result.recommendation["passed"])
        self.assertFalse(result.recommendation["feasible"])
        self.assertIn("11-31", result.recommendation["reason"])
        self.assertIn("flux_diagnostics", result.estimates["11-20"])
        self.assertIn("11-20__denominator", result.maps)
        json.dumps(result.json_summary(), allow_nan=False)

    def test_individual_dm_apply_and_advanced_mount_jog(self):
        before = self.hardware.get_dm()
        self.engine.set_dm_segment(20, 0.17, -0.23, 0.41)
        after = self.hardware.get_dm()
        self.assertAlmostEqual(after[20].piston_um, 0.17)
        self.assertAlmostEqual(after[20].tip_mrad, -0.23)
        self.assertAlmostEqual(after[20].tilt_mrad, 0.41)
        self.assertEqual(after[19], before[19])

        old_x = self.hardware.get_mount()["x"]
        mount = self.engine.jog_mount("x", 5.0)
        self.assertEqual(mount["x"], old_x + 5.0)
        self.assertEqual(mount["roll"], self.config["mount_control"]["fixed_roll"])
        with self.assertRaisesRegex(ValueError, "Roll is fixed"):
            self.engine.jog_mount("roll", 100.0)

    def test_null_live_uses_supplied_box_sum_definition(self):
        image = np.zeros((280, 280), dtype=float)
        for box in PHOT_BOXES.values():
            y0, y1, x0, x1 = box
            image[y0:y1, x0:x1] = 3.0
        for name, box in NULL_BOXES.items():
            y0, y1, x0, x1 = box
            image[y0:y1, x0:x1] = 2.0
            for ft_box in FRINGE_TRACKING_BOXES[name].values():
                fy0, fy1, fx0, fx1 = ft_box
                image[fy0:fy1, fx0:fx1] = 1.0
        processed = process_apapane_frame(np.stack([image + 5.0, image + 5.0]), np.full_like(image, 5.0))
        statistics = collect_null_statistics(processed)
        for name in NULL_BOXES:
            self.assertAlmostEqual(statistics[name]["null_depth"], 1.0)

    def test_null_live_uses_compact_tables_instead_of_three_cards(self):
        image = np.ones((280, 320), dtype=float)
        statistics = collect_null_statistics(image)
        gui = AlignmentGUI.__new__(AlignmentGUI)
        gui.config = self.config
        gui.null_live_figure = Figure(figsize=(8.6, 4.8), dpi=100)
        gui.null_live_canvas = FigureCanvasAgg(gui.null_live_figure)
        gui._draw_null_live(image, statistics, 25)
        gui.null_live_canvas.draw()
        self.assertEqual(len(gui.null_live_figure.axes), 2)
        image_position = gui.null_live_figure.axes[0].get_position()
        stats_position = gui.null_live_figure.axes[1].get_position()
        self.assertGreater(image_position.width, stats_position.width)
        self.assertGreaterEqual(image_position.x0, 0.08)
        self.assertGreaterEqual(image_position.y0, 0.12)
        stats_axis = gui.null_live_figure.axes[1]
        self.assertEqual(len(stats_axis.tables), 2)
        null_table = stats_axis.tables[1]
        rendered = {cell.get_text().get_text() for cell in null_table.get_celld().values()}
        self.assertTrue({"BASELINE", "NULL", "FT1", "FT2", "DEPTH"}.issubset(rendered))
        self.assertEqual(format_depth(0.438721), "0.4387")
        self.assertFalse(any("NULL DEPTH =" in text.get_text() for text in stats_axis.texts))

    def test_null_live_acquires_once_until_operator_starts_updates(self):
        gui = AlignmentGUI.__new__(AlignmentGUI)
        gui._closing = False
        gui._null_live_enabled = False
        gui._null_live_initial_attempted = False
        gui.config = self.config
        gui.root = Mock()
        gui._request_null_live_frame = Mock(return_value=True)
        gui._schedule_null_live()
        gui._request_null_live_frame.assert_called_once()
        gui._schedule_null_live()
        gui._request_null_live_frame.assert_called_once()
        gui._null_live_enabled = True
        gui._schedule_null_live()
        self.assertEqual(gui._request_null_live_frame.call_count, 2)

    def test_alignment_session_repeat_names_darks_and_final_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            self.config["output_root"] = temp
            directory = begin_alignment(self.config)
            self.assertTrue(directory.name.startswith("Alignment_outputs_"))

            first = self.engine.set_z_scan({"z": 1800.0})
            second = self.engine.set_z_scan({"z": 1800.0})
            save_result(first, self.hardware, self.config, "applied")
            save_result(second, self.hardware, self.config, "applied")
            self.assertFalse((directory / "04_set_z.json").exists())
            self.assertTrue((directory / "04_set_z_1.json").exists())
            self.assertTrue((directory / "04_set_z_2.json").exists())

            plotted_first = ScanResult(kind="xy", axes={"x": np.arange(2)}, maps={"m": np.ones((2, 2))})
            plotted_second = ScanResult(kind="xy", axes={"x": np.arange(2)}, maps={"m": np.ones((2, 2))})
            save_result(plotted_first, self.hardware, self.config, "applied")
            figure = Figure(figsize=(2, 2))
            figure.add_subplot(111).plot([0, 1])
            save_review_figure(plotted_first, directory, figure)
            save_result(plotted_second, self.hardware, self.config, "applied")
            self.assertTrue((directory / "06_xy_1.png").exists())
            first_summary = json.loads((directory / "06_xy_1.json").read_text(encoding="utf-8"))
            self.assertEqual(first_summary["review_figure_file"], "06_xy_1.png")
            self.assertTrue((directory / "06_xy_2.json").exists())

            dark = self.engine.acquire_dark("focal", 2)
            save_result(dark, self.hardware, self.config, "acquired")
            dark_directory = Path(temp) / "darks"
            self.assertTrue((dark_directory / "focal_dark.json").exists())
            self.assertTrue((dark_directory / "focal_dark.fits").exists() or (dark_directory / "focal_dark.npy").exists())
            self.assertFalse(any(path.name.startswith("01_focal_dark") for path in directory.iterdir()))

            status = {kind: ("complete" if kind in {"bulk", "set_z"} else "pending") for kind in LABELS}
            requested_range = {"start": 3, "end": 4, "steps": ["bulk", "set_z"]}
            final = save_final_configuration(
                self.config,
                self.hardware,
                self.engine,
                workflow_status=status,
                workflow_range=requested_range,
            )
            payload = json.loads(final.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["dm"]), 37)
            self.assertEqual(payload["mount"]["roll"], self.config["mount_control"]["fixed_roll"])
            self.assertEqual(payload["workflow"]["status"], status)
            self.assertEqual(payload["workflow"]["requested_range"], requested_range)

            individual = self.engine.set_z_scan({"z": 1850.0})
            individual.metadata["run_scope"] = "individual"
            individual_directory = save_result(individual, self.hardware, self.config, "applied")
            self.assertEqual(individual_directory.name, "Alignment_output_individual")
            self.assertTrue((individual_directory / "04_set_z_1.json").exists())

    def test_storage_records_full_dm(self):
        self.engine.acquire_dark("apapane", 3)
        result = self.engine.xy_scan(
            {
                **self.config["scans"]["xy"],
                "x": {"start": 3025, "span": 20, "step": 10},
                "y": {"start": 3430, "span": 20, "step": 10},
            }
        )
        with tempfile.TemporaryDirectory() as temp:
            self.config["output_root"] = temp
            directory = save_result(result, self.hardware, self.config, "test")
            content = (Path(directory) / "06_xy_1.json").read_text(encoding="utf-8")
            self.assertIn('"36"', content)
            self.assertTrue((Path(directory) / "06_xy_1.npz").exists() or (Path(directory) / "06_xy_1.fits").exists())
            figure = Figure(figsize=(5, 3), dpi=80)
            figure.add_subplot(111).plot([0, 1], [0, 1])
            review_path = save_review_figure(result, directory, figure)
            self.assertEqual(review_path, Path(directory) / "06_xy_1.png")
            self.assertTrue(review_path.exists())
            updated_content = (Path(directory) / "06_xy_1.json").read_text(encoding="utf-8")
            self.assertIn('"review_figure_file": "06_xy_1.png"', updated_content)
            finalized_directory = finalize_result(result, self.hardware, self.config, "applied")
            self.assertEqual(finalized_directory, directory)
            finalized = json.loads((Path(directory) / "06_xy_1.json").read_text(encoding="utf-8"))
            self.assertEqual(finalized["metadata"]["review_decision"], "applied")
            self.assertFalse((Path(directory) / "06_xy_2.json").exists())

    def test_abort_restores_complete_state(self):
        self.engine.acquire_dark("apapane", 3)
        before = self.hardware.snapshot()

        def stop_after_three(message, current, total):
            del message, total
            if current == 3:
                self.engine.abort()

        self.engine.progress = stop_after_three
        with self.assertRaises(ScanCancelled):
            self.engine.xy_scan(
                {
                    **self.config["scans"]["xy"],
                    "x": {"start": 3025, "span": 20, "step": 10},
                    "y": {"start": 3430, "span": 20, "step": 10},
                }
            )
        self.assertEqual(self.hardware.get_mount(), before.mount)
        self.assertEqual(self.hardware.get_dm(), before.dm)
        np.testing.assert_allclose(self.hardware.get_dm_volts(), before.dm_volts)
        self.assertEqual(self.hardware.get_conex(), before.conex)

    def test_bulk_abort_restores_dm_voltages_and_conex(self):
        self.engine.acquire_dark("focal", 3)
        before = self.hardware.snapshot()

        def abort_on_first_measurement(message, current, total):
            del message, total
            if current == 0:
                self.engine.abort()

        self.engine.progress = abort_on_first_measurement
        with self.assertRaises(ScanCancelled):
            self.engine.bulk_align()
        np.testing.assert_allclose(self.hardware.get_dm_volts(), before.dm_volts)
        self.assertEqual(self.hardware.get_conex(), before.conex)
        self.assertEqual(self.hardware.get_dm(), before.dm)

    def test_simulator_camera_frames_and_visible_initial_tilt(self):
        self.assertGreater(max(abs(item.tip_mrad) + abs(item.tilt_mrad) for item in self.hardware.get_dm().values()), 0.0)
        for camera in ("palila", "apapane", "focal", "pupil"):
            frames = self.hardware.camera_frames(camera, 4)
            self.assertEqual(frames.ndim, 3)
            self.assertEqual(frames.shape[0], 4)
            self.assertGreater(frames.shape[-1], 200)
        with tempfile.TemporaryDirectory() as temp:
            self.config["camera_frame_root"] = temp
            frame = self.hardware.camera_frames("pupil", 1)[0]
            path = save_camera_frame(frame, "pupil", "test frame", self.config)
            self.assertTrue(path.exists())
            self.assertEqual(path.parent, Path(temp))
            custom = Path(temp) / "operator_selected"
            custom_path = save_camera_frame(frame, "pupil", "chosen", self.config, str(custom))
            self.assertEqual(custom_path.parent, custom)

    def test_xy_pitch_yaw_iteration_status(self):
        status = WorkflowStatus(("pitch_yaw", "xy"))
        status.finish("pitch_yaw", True)
        status.finish("xy", True)
        status.begin_xy_pitch_iteration()
        self.assertEqual(status.state("pitch_yaw"), "iterating")
        self.assertEqual(status.state("xy"), "iterating")
        status.finish("pitch_yaw", True)
        self.assertEqual(status.state("pitch_yaw"), "awaiting_confirmation")
        self.assertEqual(status.state("xy"), "iterating")
        self.assertEqual(status.confirm("pitch_yaw"), "xy")
        self.assertEqual(status.state("pitch_yaw"), "complete")
        self.assertEqual(status.state("xy"), "iterating")
        status.finish("xy", True)
        self.assertEqual(status.state("xy"), "awaiting_confirmation")
        self.assertIsNone(status.confirm("xy"))
        self.assertEqual(status.state("xy"), "complete")
        status.reset()
        self.assertEqual(status.state("pitch_yaw"), "pending")
        self.assertEqual(status.state("xy"), "pending")
        self.assertFalse(status.xy_pitch_iteration)

    def test_individual_scan_invalidates_every_downstream_step(self):
        status = WorkflowStatus(LABELS)
        for kind in LABELS:
            status.start(kind, allow_iteration=False)
            status.finish(kind, True)
            status.confirm(kind)
        status.start("pitch_yaw", allow_iteration=False)
        status.finish("pitch_yaw", True)
        status.confirm("pitch_yaw")
        status.invalidate_after("pitch_yaw")
        self.assertEqual(status.state("pitch_yaw"), "complete")
        self.assertEqual(status.state("bulk"), "complete")
        for kind in ("xy", "tip_tilt", "null"):
            self.assertEqual(status.state(kind), "attention")

    def test_gui_individual_apply_uses_downstream_invalidation_path(self):
        gui = AlignmentGUI.__new__(AlignmentGUI)
        gui.workflow = WorkflowStatus(LABELS)
        for kind in LABELS:
            gui.workflow.start(kind, allow_iteration=False)
            gui.workflow.finish(kind, True)
            gui.workflow.confirm(kind)
        gui.workflow.start("pitch_yaw", allow_iteration=False)
        gui.workflow.finish("pitch_yaw", True)
        gui.current_result = ScanResult(kind="pitch_yaw", metadata={"run_scope": "individual"})
        gui.figure = Mock()
        gui.status = Mock()
        gui._set_running = Mock()
        gui.refresh_dm = Mock()
        gui.refresh_mount = Mock()
        gui._update_rest_dm_control_state = Mock()
        gui._update_step_styles = Mock()
        with patch.object(gui_module, "save_review_figure"):
            gui._finish_apply(Path("individual"), None, "pitch_yaw")
        self.assertEqual(gui.workflow.state("pitch_yaw"), "complete")
        for kind in ("xy", "tip_tilt", "null"):
            self.assertEqual(gui.workflow.state(kind), "attention")
        gui.status.set.assert_called_with(
            "Individual scan applied and saved: individual. Downstream workflow results are now orange."
        )

    def test_partial_workflow_range_and_prerequisites(self):
        gui = AlignmentGUI.__new__(AlignmentGUI)
        gui.running = False
        gui.config = self.config
        gui.hardware = self.hardware
        gui.engine = self.engine
        gui.engine.focal_dark = np.zeros((850, 850))
        gui.engine.apapane_dark = np.zeros((280, 280))
        gui.workflow = WorkflowStatus(LABELS)
        gui.workflow_start = Mock()
        gui.workflow_start.get.return_value = "3"
        gui.workflow_end = Mock()
        gui.workflow_end.get.return_value = "7"
        gui.status = Mock()
        gui.refresh_dm = Mock()
        gui.refresh_mount = Mock()
        gui._update_step_styles = Mock()
        gui._run_next_workflow_step = Mock()
        gui._append_log = Mock()

        with patch.object(gui_module.messagebox, "askokcancel") as dark_prompt, patch.object(
            gui_module, "begin_alignment", return_value=Path("Alignment_outputs_9")
        ):
            gui.run_full_workflow()

        dark_prompt.assert_not_called()
        self.assertEqual(gui.workflow_queue, ["bulk", "set_z", "pitch_yaw", "xy", "tip_tilt"])
        self.assertEqual(gui.workflow_range["start"], 3)
        self.assertEqual(gui.workflow_range["end"], 7)
        self.assertTrue(all(gui.workflow.state(kind) == "pending" for kind in LABELS))

        gui.engine.focal_dark = None
        with self.assertRaisesRegex(ValueError, "no focal dark"):
            gui._validate_partial_workflow_prerequisites(["bulk"])

    def test_pitch_yaw_grid_is_validated_as_start_plus_range(self):
        parameters = json.loads(json.dumps(self.config["scans"]["pitch_yaw"]))
        parameters["pitch"] = {"start": 300.0, "span": 500.0, "step": 50.0}
        self.engine.validate_scan_parameters("pitch_yaw", parameters)
        parameters["pitch"]["span"] = 550.0
        with self.assertRaisesRegex(ValueError, r"Pitch scan 300 to 850.*start \+ range"):
            self.engine.validate_scan_parameters("pitch_yaw", parameters)

    def test_redo_validates_edited_pitch_yaw_fields_before_finalizing(self):
        gui = AlignmentGUI.__new__(AlignmentGUI)
        gui.current_result = ScanResult(kind="pitch_yaw")
        gui.engine = self.engine
        gui.hardware = self.hardware
        gui.config = self.config
        gui.figure = Mock()
        gui.run_selected = Mock()
        invalid = json.loads(json.dumps(self.config["scans"]["pitch_yaw"]))
        invalid["pitch"] = {"start": 300.0, "span": 550.0, "step": 50.0}
        gui._parameters = Mock(return_value=invalid)
        with patch.object(gui_module.messagebox, "showerror") as error_dialog, patch.object(
            gui_module, "finalize_result"
        ) as finalize:
            gui.redo()
        error_dialog.assert_called_once()
        self.assertIn("300 to 850", error_dialog.call_args.args[1])
        finalize.assert_not_called()
        gui.run_selected.assert_not_called()

    def test_saved_darks_reload_for_partial_workflows(self):
        with tempfile.TemporaryDirectory() as temp:
            self.config["output_root"] = temp
            result = self.engine.acquire_dark("apapane", 3)
            save_result(result, self.hardware, self.config, "acquired")
            loaded = load_saved_dark(self.config, "apapane")
            self.assertIsNotNone(loaded)
            np.testing.assert_allclose(loaded, result.maps["dark"])
            reloaded_engine = AlignmentEngine(self.hardware, self.config)
            np.testing.assert_allclose(reloaded_engine.apapane_dark, result.maps["dark"])

    def test_workflow_tick_requires_apply_and_restore_exit_returns_white(self):
        status = WorkflowStatus(("bulk",))
        status.start("bulk")
        status.finish("bulk", True)
        self.assertEqual(status.state("bulk"), "awaiting_confirmation")

        gui = AlignmentGUI.__new__(AlignmentGUI)
        gui.workflow = status
        gui.step_buttons = {"bulk": Mock()}
        gui._update_step_styles()
        self.assertEqual(gui.step_buttons["bulk"].configure.call_args.kwargs["style"], "Review.Step.TButton")
        self.assertTrue(gui.step_buttons["bulk"].configure.call_args.kwargs["text"].endswith("●"))

        status.confirm("bulk")
        gui._update_step_styles()
        self.assertEqual(status.state("bulk"), "complete")
        self.assertEqual(gui.step_buttons["bulk"].configure.call_args.kwargs["style"], "Done.Step.TButton")
        self.assertTrue(gui.step_buttons["bulk"].configure.call_args.kwargs["text"].endswith("✓"))

        status.start("bulk")
        status.finish("bulk", True)
        status.restore("bulk")
        gui._update_step_styles()
        self.assertEqual(status.state("bulk"), "pending")
        self.assertEqual(gui.step_buttons["bulk"].configure.call_args.kwargs["style"], "Step.TButton")
        self.assertEqual(gui.step_buttons["bulk"].configure.call_args.kwargs["text"], LABELS["bulk"])

    def test_exit_review_resets_the_current_workflow_step(self):
        gui = AlignmentGUI.__new__(AlignmentGUI)
        gui.current_result = ScanResult(kind="bulk", recommendation={"passed": True})
        gui.workflow = WorkflowStatus(LABELS)
        gui.workflow.start("bulk")
        gui.workflow.finish("bulk", True)
        gui.engine = Mock()
        gui.hardware = self.hardware
        gui.config = self.config
        gui.figure = Mock()
        gui.workflow_queue = ["apapane_dark"]
        gui.full_workflow_active = True
        gui.review_pending = True
        gui._update_step_styles = Mock()
        gui._clear_review = Mock()
        gui.status = Mock()
        gui.refresh_dm = Mock()
        gui.refresh_mount = Mock()
        gui._update_rest_dm_control_state = Mock()
        with patch.object(gui_module, "finalize_result", return_value=Path("saved")), patch.object(
            gui_module, "save_review_figure"
        ):
            gui.exit_review()

        self.assertEqual(gui.workflow.state("bulk"), "pending")
        self.assertEqual(gui.workflow_queue, [])
        self.assertFalse(gui.full_workflow_active)
        gui._update_step_styles.assert_called_once()

    def test_dm_gradient_is_uniform_when_flat_and_directional_when_tilted(self):
        piston_green = tuple(int(GREEN[index : index + 2], 16) / 255 for index in (1, 3, 5)) + (1.0,)
        flat = dm_segment_rgba(piston_green, 0.0, 0.0, 5.0, resolution=24)
        np.testing.assert_allclose(flat[:, :, :3], np.broadcast_to(piston_green[:3], flat[:, :, :3].shape))
        tipped = dm_segment_rgba(piston_green, 5.0, 0.0, 5.0, resolution=24)
        self.assertGreater(float(tipped[:, -1, :3].mean()), float(tipped[:, 0, :3].mean()))

    def test_dm_piston_palette_has_no_black_and_expands_small_values(self):
        cmap, norm, scale = dm_piston_mapping(np.asarray([-1.3, 0.0, 1.3]), 0.02)
        colours = np.asarray([cmap(value)[:3] for value in np.linspace(0.0, 1.0, 101)])
        self.assertGreater(float(colours.min()), 0.05)
        nonlinear_difference = float(norm(0.05) - norm(0.0))
        linear_difference = 0.05 / (2.0 * scale)
        self.assertGreater(nonlinear_difference, 3.0 * linear_difference)

    def test_full_workflow_queue_survives_xy_pitch_yaw_iteration(self):
        gui = AlignmentGUI.__new__(AlignmentGUI)
        gui.workflow = WorkflowStatus(LABELS)
        gui.workflow.finish("pitch_yaw", True)
        gui.workflow.finish("xy", True)
        gui.workflow_queue = ["tip_tilt", "null"]
        gui.full_workflow_active = True
        gui.status = Mock()
        gui.root = Mock()
        gui._set_running = Mock()
        gui.refresh_dm = Mock()
        gui._update_step_styles = Mock()
        gui.select_scan = Mock()
        gui.run_selected = Mock()
        gui._run_next_workflow_step = Mock()

        gui._finish_apply("scan", "pitch_yaw", "xy")
        self.assertEqual(gui.workflow_queue, ["tip_tilt", "null"])
        gui.workflow.finish("pitch_yaw", True)
        gui._finish_apply("scan", None, "pitch_yaw")
        self.assertEqual(gui.workflow.state("pitch_yaw"), "complete")
        self.assertEqual(gui.workflow.state("xy"), "iterating")
        gui.workflow.finish("xy", True)
        gui._finish_apply("scan", None, "xy")
        self.assertEqual(gui.workflow.state("xy"), "complete")
        self.assertEqual(gui.workflow_queue, ["tip_tilt", "null"])
        self.assertGreaterEqual(gui.root.after.call_count, 3)

    def test_full_workflow_finishes_after_applying_the_last_scan(self):
        gui = AlignmentGUI.__new__(AlignmentGUI)
        gui.workflow = WorkflowStatus(LABELS)
        gui.workflow_queue = []
        gui.full_workflow_active = True
        gui.status = Mock()
        gui.root = Mock()
        gui._set_running = Mock()
        gui.refresh_dm = Mock()
        gui._update_step_styles = Mock()
        gui._run_next_workflow_step = Mock()

        gui._finish_apply("scan", None, "null")

        gui.root.after.assert_called_once_with(250, gui._run_next_workflow_step)

    def test_idle_abort_is_a_no_op(self):
        gui = AlignmentGUI.__new__(AlignmentGUI)
        gui.running = False
        gui.scan_running = False
        gui.engine = Mock()
        gui.abort()
        gui.engine.abort.assert_not_called()

    def test_three_map_review_axes_and_colorbars_do_not_overlap(self):
        values = np.linspace(-1.0, 1.0, 11)
        maps = {name: np.ones((11, 11), dtype=float) * (index + 1) for index, name in enumerate(PHOT_BOXES)}
        estimates = [
            {"channel": name, "x": 0.0, "y": 0.0, "method": "test"}
            for name in PHOT_BOXES
        ]
        result = ScanResult(
            kind="pitch_yaw",
            axes={"pitch": values, "yaw": values},
            maps=maps,
            estimates={"channels": estimates, "agreement": {}},
            recommendation={"pitch": 0.0, "yaw": 0.0, "passed": True},
            metadata={},
        )
        gui = AlignmentGUI.__new__(AlignmentGUI)
        gui.config = self.config
        gui.figure = Figure(figsize=(8.6, 4.8), dpi=100)
        canvas = FigureCanvasAgg(gui.figure)
        gui.canvas = canvas
        gui._plot_result(result)
        canvas.draw()
        self.assertIn("OPTIMAL CENTRE", gui.figure._suptitle.get_text())
        plot_titles = [axis.get_title() for axis in gui.figure.axes if axis.get_title()]
        self.assertEqual(len(plot_titles), 3)
        self.assertTrue(all("PITCH=" in title and "YAW=" in title for title in plot_titles))
        bounds = [axis.get_position() for axis in gui.figure.axes]
        self.assertEqual(len(bounds), 6)
        for first_index, first in enumerate(bounds):
            for second in bounds[first_index + 1 :]:
                horizontal = min(first.x1, second.x1) - max(first.x0, second.x0)
                vertical = min(first.y1, second.y1) - max(first.y0, second.y0)
                self.assertFalse(horizontal > 1e-5 and vertical > 1e-5)

    def test_tip_tilt_review_prints_each_segment_optimum_above_its_map(self):
        values = np.linspace(-1.0, 1.0, 5)
        estimates = {
            "11": {"x": 0.12, "y": -0.21, "method": "gaussian"},
            "20": {"x": -0.15, "y": 0.22, "method": "gaussian"},
            "31": {"x": 0.04, "y": 0.31, "method": "gaussian"},
        }
        result = ScanResult(
            kind="tip_tilt",
            axes={"tip": values, "tilt": values},
            maps={segment: np.ones((5, 5)) for segment in estimates},
            estimates=estimates,
            recommendation={
                "segments": {
                    segment: {"piston_um": 0.0, "tip_mrad": value["x"], "tilt_mrad": value["y"]}
                    for segment, value in estimates.items()
                },
                "passed": True,
            },
        )
        gui = AlignmentGUI.__new__(AlignmentGUI)
        gui.config = self.config
        gui.figure = Figure(figsize=(8.6, 4.8), dpi=100)
        gui.canvas = FigureCanvasAgg(gui.figure)
        gui._plot_result(result)
        titles = [axis.get_title() for axis in gui.figure.axes if axis.get_title()]
        self.assertEqual(len(titles), 3)
        self.assertTrue(all("TIP=" in title and "TILT=" in title for title in titles))
        self.assertIn("EACH INJECTED SEGMENT", gui.figure._suptitle.get_text())

    def test_progress_area_is_cleared_at_scan_start(self):
        gui = AlignmentGUI.__new__(AlignmentGUI)
        gui.progress = {"value": 100}
        gui.progress_text = Mock()
        gui.log_text = Mock()
        gui._clear_progress_area(clear_log=True)
        self.assertEqual(gui.progress["value"], 0)
        gui.progress_text.set.assert_called_once_with("")
        gui.log_text.delete.assert_called_once_with("1.0", "end")
        self.assertEqual(
            [call.kwargs["state"] for call in gui.log_text.configure.call_args_list],
            ["normal", "disabled"],
        )

    def test_status_log_is_only_unlocked_for_internal_updates(self):
        gui = AlignmentGUI.__new__(AlignmentGUI)
        gui.log_text = Mock()
        gui._append_log("alignment message")
        self.assertEqual(
            [call.kwargs["state"] for call in gui.log_text.configure.call_args_list],
            ["normal", "disabled"],
        )
        gui.log_text.insert.assert_called_once_with("end", "alignment message\n")
        gui.log_text.see.assert_called_once_with("end")

    def test_dm_segment_click_selects_and_populates_inspector(self):
        gui = AlignmentGUI.__new__(AlignmentGUI)
        gui.config = self.config
        gui.hardware = self.hardware
        gui.dm_selected_segment = None
        gui.dm_figure = Figure(figsize=(8.6, 4.8), dpi=100)
        gui.dm_canvas = FigureCanvasAgg(gui.dm_figure)
        gui.dm_inspector_header = Mock()
        gui.dm_inspector_values = {
            "piston_um": Mock(),
            "tip_mrad": Mock(),
            "tilt_mrad": Mock(),
        }
        gui.refresh_dm()
        gui.dm_inspector_header.reset_mock()
        for variable in gui.dm_inspector_values.values():
            variable.reset_mock()
        gui.dm_canvas.draw()
        target = gui.dm_segment_patches[11].xy
        display_x, display_y = gui.dm_axes.transData.transform(target)
        event = MouseEvent("button_press_event", gui.dm_canvas, display_x, display_y, button=1)
        gui._on_dm_click(event)
        self.assertEqual(gui.dm_selected_segment, 11)
        gui.dm_inspector_header.set.assert_called_with("SEGMENT 11")
        self.assertTrue(all(variable.set.called for variable in gui.dm_inspector_values.values()))


if __name__ == "__main__":
    unittest.main()
