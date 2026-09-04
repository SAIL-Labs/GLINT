from __future__ import annotations

import json
import queue
import threading
import time
import traceback
from copy import deepcopy

import matplotlib.patheffects as path_effects
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, RegularPolygon
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tkinter import font as tkfont
from tkinter import filedialog, messagebox, ttk

from .boxes import FRINGE_TRACKING_BOXES, NULL_BOXES, PHOT_BOXES
from .camera_storage import CAMERA_LABELS, save_camera_frame
from .config import rest_dm_maps, save_config, segment_state, validate_config, validate_scan_defaults
from .engine import AlignmentEngine, ScanCancelled
from .models import ScanResult
from .null_live import collect_null_statistics, format_depth, format_flux, process_apapane_frame
from .storage import begin_alignment, finalize_result, save_final_configuration, save_result, save_review_figure
from .workflow_status import WorkflowStatus


BG = "#000000"
FG = "#f2f5f3"
MUTED = "#a3ada7"
GREEN = "#39ff88"
ORANGE = "#ff9f1c"
RED = "#ff4d6d"
GRID = "#48514c"
ALIEN_X_SHIFT = -10

LABELS = {
    "focal_dark": "1  Focal-plane dark",
    "apapane_dark": "2  APAPANE dark",
    "bulk": "3  Bulk alignment",
    "set_z": "4  Set mount Z",
    "pitch_yaw": "5  Pitch / yaw",
    "xy": "6  X / Y",
    "tip_tilt": "7  Segment tip / tilt",
    "null": "8  Null pistons",
}

CAMERA_DISPLAY_TO_KEY = {label: key for key, label in CAMERA_LABELS.items()}


def flatten_default_settings(mapping: dict, prefix: tuple[str, ...] = ()):
    """Yield scan-setting leaves while preserving their JSON order."""
    for key, value in mapping.items():
        path = prefix + (str(key),)
        if isinstance(value, dict):
            yield from flatten_default_settings(value, path)
        else:
            yield path, value


def format_default_setting(value) -> str:
    if isinstance(value, (list, dict)) or value is None:
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def parse_default_setting(text: str, original):
    """Parse one edited value using the type already present in the JSON."""
    text = text.strip()
    if isinstance(original, bool):
        if text.lower() not in {"true", "false"}:
            raise ValueError("expected true or false")
        return text.lower() == "true"
    if isinstance(original, int):
        number = float(text)
        if not np.isfinite(number) or not number.is_integer():
            raise ValueError("expected an integer")
        return int(number)
    if isinstance(original, float):
        number = float(text)
        if not np.isfinite(number):
            raise ValueError("expected a finite number")
        return number
    if isinstance(original, list):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError("expected a JSON list")
        return value
    if original is None:
        return json.loads(text)
    if isinstance(original, str):
        return text
    return json.loads(text)


def assign_default_setting(mapping: dict, path: tuple[str, ...], value) -> None:
    target = mapping
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def format_piston_tick(value: float, _position=None) -> str:
    """Format a piston tick as an ordinary decimal, never scientific notation."""
    if not np.isfinite(value):
        return ""
    text = np.format_float_positional(
        float(value),
        precision=6,
        unique=False,
        fractional=True,
        trim="-",
    )
    return "0" if text in {"-0", ""} else text


def dm_segment_rgba(
    piston_color: tuple[float, float, float, float],
    tip_mrad: float,
    tilt_mrad: float,
    full_scale_mrad: float,
    resolution: int = 96,
) -> np.ndarray:
    """Encode tip/tilt as a black-to-piston-colour directional gradient."""
    piston_rgb = np.asarray(piston_color[:3], dtype=float)
    magnitude = float(np.hypot(tip_mrad, tilt_mrad))
    strength = min(magnitude / max(float(full_scale_mrad), 1e-9), 1.0)
    coordinates = np.linspace(-1.0, 1.0, resolution)
    xx, yy = np.meshgrid(coordinates, coordinates)
    if magnitude <= 1e-9:
        mixture = np.ones_like(xx)
    else:
        direction = (xx * tip_mrad + yy * tilt_mrad) / magnitude
        directional_ramp = np.clip(0.5 + 0.5 * direction, 0.0, 1.0)
        mixture = (1.0 - strength) + strength * directional_ramp
    rgba = np.empty((resolution, resolution, 4), dtype=float)
    rgba[:, :, :3] = mixture[:, :, None] * piston_rgb
    rgba[:, :, 3] = 1.0
    return rgba


def dm_piston_mapping(
    piston_values: np.ndarray,
    linthresh_um: float = 0.02,
) -> tuple[LinearSegmentedColormap, SymLogNorm, float]:
    """Return a vivid zero-centred piston palette with expanded near-zero contrast."""
    values = np.asarray(piston_values, dtype=float)
    scale = max(float(np.max(np.abs(values))), 1.3)
    linthresh = min(max(float(linthresh_um), 1e-4), 0.25 * scale)
    cmap = LinearSegmentedColormap.from_list(
        "glint_piston_diverging",
        [
            (0.00, "#3157d5"),
            (0.35, "#58c6ff"),
            (0.50, GREEN),
            (0.65, "#ffd166"),
            (1.00, "#ff4d6d"),
        ],
    )
    norm = SymLogNorm(
        linthresh=linthresh,
        linscale=1.2,
        vmin=-scale,
        vmax=scale,
        base=10,
    )
    return cmap, norm, scale


class AlignmentGUI:
    def __init__(self, root: tk.Tk, hardware, config: dict) -> None:
        self.root = root
        self.hardware = hardware
        self.config = config
        self.events: queue.Queue = queue.Queue()
        self.engine = AlignmentEngine(hardware, config, progress=self._progress_from_worker, log=self._log_from_worker)
        self.current_kind = "focal_dark"
        self.current_result: ScanResult | None = None
        self.review_pending = False
        self.running = False
        self.scan_running = False
        self.full_workflow_active = False
        self.workflow_queue: list[str] = []
        self.workflow_selected_kinds: list[str] = []
        self.workflow_range: dict = {}
        self.workflow_start = tk.StringVar(value="1")
        self.workflow_end = tk.StringVar(value=str(len(LABELS)))
        self.fields: dict[str, tk.StringVar] = {}
        self.step_buttons: dict[str, ttk.Button] = {}
        self.workflow = WorkflowStatus(LABELS)
        self.status = tk.StringVar(value="Ready")
        self.progress_text = tk.StringVar(value="")
        self.camera_choice = tk.StringVar(value=CAMERA_LABELS["focal"])
        self.camera_nframes = tk.StringVar(value="1")
        self.camera_filename = tk.StringVar(value="")
        self.camera_directory = tk.StringVar(value=str(config["camera_frame_root"]))
        self.dm_selected_segment: int | None = None
        self.dm_segment_patches: dict[int, RegularPolygon] = {}
        self.dm_axes = None
        self.dm_inspector_header = tk.StringVar(value="SEGMENT —")
        self.dm_inspector_values = {
            "piston_um": tk.StringVar(value="—"),
            "tip_mrad": tk.StringVar(value="—"),
            "tilt_mrad": tk.StringVar(value="—"),
        }
        self.dm_apply_button = None
        self.mount_values = {
            axis: tk.StringVar(value="—") for axis in ("pitch", "yaw", "roll", "x", "y", "z")
        }
        maps = rest_dm_maps(config)
        self.rest_dm_label_to_key = {
            str(definition.get("label", name)): name for name, definition in maps.items()
        }
        active_rest = self.engine.rest_dm_selection()
        self.rest_dm_choice = tk.StringVar(value=str(active_rest["label"]))
        self.rest_dm_piston = tk.StringVar(
            value="" if active_rest["uniform_piston_um"] is None else str(active_rest["uniform_piston_um"])
        )
        self.rest_dm_tip = tk.StringVar(
            value="" if active_rest["uniform_tip_mrad"] is None else str(active_rest["uniform_tip_mrad"])
        )
        self.rest_dm_tilt = tk.StringVar(
            value="" if active_rest["uniform_tilt_mrad"] is None else str(active_rest["uniform_tilt_mrad"])
        )
        self.rest_dm_active = tk.StringVar(value=f"Active: {active_rest['label']}")
        self.dm_preset_buttons: dict[str, ttk.Button] = {}
        self._last_live_dm_refresh = 0.0
        self._null_live_acquiring = False
        self._null_live_enabled = False
        self._null_live_initial_attempted = False
        self._closing = False
        self.advanced_mount_window = None
        self.workflow_defaults_window = None
        self.workflow_default_fields = {}
        self.workflow_default_frames = {}
        self.workflow_default_buttons = {}
        self.advanced_axis = tk.StringVar(value="pitch")
        self.advanced_range = tk.StringVar(value="")
        self.advanced_step = tk.StringVar(value="100")
        self.advanced_step_label = tk.StringVar(value="Step (pulses)")
        self.advanced_minus_button = None
        self.advanced_plus_button = None
        self.retro_font = self._choose_retro_font()
        self._configure_window()
        self._build_layout()
        self.select_scan("focal_dark")
        self.refresh_dm()
        self.refresh_mount()
        self.root.after(75, self._drain_events)
        self.root.after(500, self._schedule_null_live)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        warnings = validate_config(config)
        warnings.extend(getattr(hardware, "limit_warnings", []))
        for warning in warnings:
            self._append_log(f"WARNING: {warning}")

    def _choose_retro_font(self) -> str:
        available = set(tkfont.families(self.root))
        for candidate in ("OCR A Extended", "Lucida Console", "Courier New", "DejaVu Sans Mono"):
            if candidate in available:
                return candidate
        return "TkFixedFont"

    def _configure_window(self) -> None:
        self.root.title(f"GLINT Alignment — {self.hardware.mode.title()}")
        self.root.geometry("1460x900")
        self.root.minsize(1180, 740)
        self.root.configure(bg=BG)
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        base_font = (self.retro_font, 9)
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG, font=base_font)
        style.configure("Header.TLabel", font=(self.retro_font, 14, "bold"), foreground=GREEN)
        style.configure("Muted.TLabel", foreground=MUTED, font=(self.retro_font, 8))
        style.configure("TButton", padding=7, font=base_font, background="#161616", foreground=FG)
        style.map("TButton", background=[("active", "#242424")], foreground=[("disabled", "#606660")])
        # Six pixels of vertical padding makes each workflow cell about 20%
        # shorter than the previous nine-pixel layout without reducing its
        # horizontal click target.
        style.configure("Step.TButton", anchor="w", padding=(9, 6))
        style.configure("Done.Step.TButton", foreground=GREEN)
        style.configure("Review.Step.TButton", foreground=GREEN)
        style.configure("Iterating.Step.TButton", foreground=ORANGE)
        style.map("Done.Step.TButton", foreground=[("disabled", GREEN), ("!disabled", GREEN)])
        style.map("Review.Step.TButton", foreground=[("disabled", GREEN), ("!disabled", GREEN)])
        style.map("Iterating.Step.TButton", foreground=[("disabled", ORANGE), ("!disabled", ORANGE)])
        style.configure("Accent.TButton", font=(self.retro_font, 9, "bold"), foreground=GREEN)
        style.configure(
            "Run.TButton",
            padding=7,
            font=(self.retro_font, 9, "bold"),
            foreground=BG,
            background=GREEN,
        )
        style.map(
            "Run.TButton",
            foreground=[("disabled", "#8ba797"), ("!disabled", BG)],
            background=[("disabled", "#174f30"), ("pressed", "#20d96e"), ("active", "#61ffa0")],
        )
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(12, 7), font=base_font, background="#111111", foreground=FG)
        style.map("TNotebook.Tab", background=[("selected", "#202020")], foreground=[("selected", GREEN)])
        style.configure("TEntry", fieldbackground="#101010", foreground=GREEN, insertcolor=GREEN, font=base_font)
        style.configure("TSpinbox", fieldbackground="#101010", foreground=GREEN, arrowcolor=GREEN, arrowsize=13, font=base_font)
        style.configure("TCombobox", fieldbackground="#101010", background="#101010", foreground=GREEN, font=base_font)
        style.configure(
            "Camera.TCombobox",
            fieldbackground=BG,
            background=BG,
            foreground=GREEN,
            arrowcolor=GREEN,
            selectbackground="#181818",
            selectforeground=GREEN,
            font=base_font,
        )
        style.map(
            "Camera.TCombobox",
            fieldbackground=[("readonly", BG), ("disabled", "#080808")],
            background=[("readonly", BG), ("active", "#181818"), ("disabled", "#080808")],
            foreground=[("readonly", GREEN), ("disabled", "#606660")],
            arrowcolor=[("readonly", GREEN), ("disabled", "#606660")],
        )
        self.root.option_add("*TCombobox*Listbox.background", BG)
        self.root.option_add("*TCombobox*Listbox.foreground", GREEN)
        self.root.option_add("*TCombobox*Listbox.selectBackground", "#181818")
        self.root.option_add("*TCombobox*Listbox.selectForeground", GREEN)
        style.configure("TProgressbar", troughcolor="#141414", background=GREEN)
        style.configure("TSeparator", background="#333333")

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)
        outer.rowconfigure(1, minsize=125)

        left = ttk.Frame(outer, padding=(0, 0, 12, 0), width=245)
        left.grid(row=0, column=0, rowspan=2, sticky="ns")
        left.grid_propagate(False)
        self.workflow_header_button = tk.Button(
            left,
            text="WORKFLOW",
            command=self.open_workflow_defaults,
            bg=BG,
            fg=GREEN,
            activebackground=BG,
            activeforeground=FG,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            anchor="w",
            cursor="hand2",
            font=(self.retro_font, 14, "bold"),
        )
        self.workflow_header_button.pack(fill="x", pady=(0, 8))
        for kind, label in LABELS.items():
            button = ttk.Button(left, text=label, style="Step.TButton", width=27, command=lambda k=kind: self.select_scan(k))
            button.pack(fill="x", pady=2)
            self.step_buttons[kind] = button
        range_row = ttk.Frame(left)
        range_row.pack(fill="x", pady=(10, 3))
        self.workflow_run_button = ttk.Button(
            range_row, text="RUN", style="Run.TButton", command=self.run_full_workflow
        )
        self.workflow_run_button.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.workflow_start_entry = ttk.Spinbox(
            range_row, from_=1, to=len(LABELS), textvariable=self.workflow_start, width=3, justify="center"
        )
        self.workflow_start_entry.pack(side="left", ipady=2)
        ttk.Label(range_row, text="→", style="Muted.TLabel").pack(side="left", padx=4)
        self.workflow_end_entry = ttk.Spinbox(
            range_row, from_=1, to=len(LABELS), textvariable=self.workflow_end, width=3, justify="center"
        )
        self.workflow_end_entry.pack(side="left", ipady=2)
        ttk.Separator(left).pack(fill="x", pady=12)
        ttk.Label(left, text="SAVE CAMERA FRAME", style="Header.TLabel").pack(anchor="w", pady=(0, 7))
        camera_row = ttk.Frame(left)
        camera_row.pack(fill="x", pady=2)
        camera_row.columnconfigure(0, weight=1)
        ttk.Label(camera_row, text="Camera", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(camera_row, text="Frames", style="Muted.TLabel").grid(row=0, column=1, sticky="w", padx=(7, 0))
        self.camera_combo = ttk.Combobox(
            camera_row,
            textvariable=self.camera_choice,
            values=list(CAMERA_DISPLAY_TO_KEY),
            state="readonly",
            style="Camera.TCombobox",
            width=17,
        )
        self.camera_combo.grid(row=1, column=0, sticky="ew")
        self.camera_frames_entry = ttk.Spinbox(
            camera_row,
            from_=1,
            to=10000,
            textvariable=self.camera_nframes,
            width=6,
        )
        self.camera_frames_entry.grid(row=1, column=1, sticky="ew", padx=(7, 0))
        ttk.Label(left, text="Optional filename", style="Muted.TLabel").pack(anchor="w", pady=(7, 1))
        ttk.Entry(left, textvariable=self.camera_filename).pack(fill="x", pady=2)
        ttk.Label(left, text="Destination folder", style="Muted.TLabel").pack(anchor="w", pady=(7, 1))
        camera_path_row = ttk.Frame(left)
        camera_path_row.pack(fill="x", pady=2)
        camera_path_row.columnconfigure(0, weight=1)
        ttk.Entry(camera_path_row, textvariable=self.camera_directory).grid(row=0, column=0, sticky="ew")
        self.camera_browse_button = ttk.Button(camera_path_row, text="…", width=3, command=self.choose_camera_directory)
        self.camera_browse_button.grid(row=0, column=1, padx=(5, 0))
        self.camera_button = ttk.Button(left, text="ACQUIRE", command=self.acquire_camera_frame)
        self.camera_button.pack(fill="x", pady=3)

        center = ttk.Frame(outer)
        center.grid(row=0, column=1, sticky="nsew")
        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)
        self.notebook = ttk.Notebook(center)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        self.result_tab = ttk.Frame(self.notebook)
        self.dm_tab = ttk.Frame(self.notebook)
        self.null_live_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.result_tab, text="SCAN REVIEW")
        self.notebook.add(self.dm_tab, text="DM STATE")
        self.notebook.add(self.null_live_tab, text="NULL LIVE")
        self.figure = Figure(figsize=(8.6, 6.2), dpi=100, facecolor=BG)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.result_tab)
        self.canvas.get_tk_widget().configure(bg=BG, highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.dm_tab.rowconfigure(0, weight=1)
        self.dm_tab.columnconfigure(1, weight=1)
        dm_sidebar = ttk.Frame(self.dm_tab, padding=(8, 10, 8, 8), width=215)
        dm_sidebar.grid(row=0, column=0, sticky="ns")
        dm_sidebar.grid_propagate(False)
        dm_sidebar.columnconfigure(0, weight=1)
        dm_sidebar.rowconfigure(0, weight=1)
        dm_controls = ttk.Frame(dm_sidebar)
        dm_controls.grid(row=0, column=0, sticky="ew", padx=2)
        ttk.Label(dm_controls, textvariable=self.dm_inspector_header, style="Header.TLabel").pack(
            anchor="w", pady=(0, 7)
        )
        inspector = tk.Frame(dm_controls, bg="#080808", highlightbackground=GRID, highlightthickness=1)
        inspector.pack(fill="x")
        inspector.columnconfigure(0, weight=1)
        inspector.columnconfigure(1, weight=1)
        inspector_rows = (
            ("Piston (µm)", "piston_um"),
            ("Tip (mrad)", "tip_mrad"),
            ("Tilt (mrad)", "tilt_mrad"),
        )
        for row, (label, key) in enumerate(inspector_rows):
            tk.Label(
                inspector,
                text=label,
                bg="#080808",
                fg=GREEN,
                anchor="w",
                padx=5,
                pady=5,
                font=(self.retro_font, 8),
            ).grid(row=row, column=0, sticky="ew")
            ttk.Entry(
                inspector,
                textvariable=self.dm_inspector_values[key],
                width=11,
            ).grid(row=row, column=1, sticky="ew", padx=5, pady=4)
        self.dm_apply_button = ttk.Button(
            dm_controls,
            text="APPLY SEGMENT VALUES",
            command=self.apply_dm_segment,
            state="disabled",
        )
        self.dm_apply_button.pack(fill="x", pady=(7, 0))

        ttk.Separator(dm_controls).pack(fill="x", pady=14)
        ttk.Label(dm_controls, text="REST OF DM", style="Header.TLabel").pack(anchor="w", pady=(0, 7))
        self.rest_dm_combo = ttk.Combobox(
            dm_controls,
            textvariable=self.rest_dm_choice,
            values=list(self.rest_dm_label_to_key),
            state="readonly",
            style="Camera.TCombobox",
            width=20,
        )
        self.rest_dm_combo.pack(fill="x", pady=(0, 7))
        self.rest_dm_combo.bind("<<ComboboxSelected>>", self._rest_dm_selection_changed)
        rest_piston_row = ttk.Frame(dm_controls)
        rest_piston_row.pack(fill="x", pady=2)
        ttk.Label(rest_piston_row, text="Piston (µm)").pack(side="left")
        self.rest_dm_piston_entry = ttk.Entry(rest_piston_row, textvariable=self.rest_dm_piston, width=9)
        self.rest_dm_piston_entry.pack(side="right")
        rest_tip_row = ttk.Frame(dm_controls)
        rest_tip_row.pack(fill="x", pady=2)
        ttk.Label(rest_tip_row, text="Tip (mrad)").pack(side="left")
        self.rest_dm_tip_entry = ttk.Entry(rest_tip_row, textvariable=self.rest_dm_tip, width=9)
        self.rest_dm_tip_entry.pack(side="right")
        rest_tilt_row = ttk.Frame(dm_controls)
        rest_tilt_row.pack(fill="x", pady=2)
        ttk.Label(rest_tilt_row, text="Tilt (mrad)").pack(side="left")
        self.rest_dm_tilt_entry = ttk.Entry(rest_tilt_row, textvariable=self.rest_dm_tilt, width=9)
        self.rest_dm_tilt_entry.pack(side="right")
        self.rest_dm_button = ttk.Button(dm_controls, text="CONFIRM REST OF DM", command=self.apply_rest_dm_map)
        self.rest_dm_button.pack(fill="x", pady=(8, 4))
        ttk.Label(dm_controls, textvariable=self.rest_dm_active, style="Muted.TLabel", wraplength=195).pack(anchor="w")
        preset_row = ttk.Frame(dm_controls)
        preset_row.pack(fill="x", pady=(9, 0))
        for column, (label, preset) in enumerate((("PTT0", "zero_ptt"), ("V0", "zero_volts"), ("F0", "flat_volts"))):
            button = ttk.Button(
                preset_row,
                text=label,
                command=lambda selected=preset: self.apply_dm_preset(selected),
                width=5,
            )
            button.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 3, 0))
            preset_row.columnconfigure(column, weight=1)
            self.dm_preset_buttons[preset] = button

        self.dm_figure = Figure(figsize=(7.2, 6.2), dpi=100, facecolor=BG)
        self.dm_canvas = FigureCanvasTkAgg(self.dm_figure, master=self.dm_tab)
        self.dm_canvas.get_tk_widget().configure(bg=BG, highlightthickness=0)
        self.dm_canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")
        self.dm_canvas.mpl_connect("button_press_event", self._on_dm_click)
        self._rest_dm_selection_changed()

        self.null_live_tab.rowconfigure(1, weight=1)
        self.null_live_tab.columnconfigure(0, weight=1)
        null_live_controls = ttk.Frame(self.null_live_tab, padding=(4, 4, 8, 2))
        null_live_controls.grid(row=0, column=0, sticky="ew")
        null_live_controls.columnconfigure(0, weight=5)
        null_live_controls.columnconfigure(1, weight=7)
        self.null_live_button = ttk.Button(
            null_live_controls,
            text="START",
            style="Accent.TButton",
            command=self.toggle_null_live,
            width=14,
        )
        self.null_live_button.grid(row=0, column=1, sticky="e")
        self.null_live_figure = Figure(figsize=(8.6, 6.2), dpi=100, facecolor=BG)
        self.null_live_canvas = FigureCanvasTkAgg(self.null_live_figure, master=self.null_live_tab)
        self.null_live_canvas.get_tk_widget().configure(bg=BG, highlightthickness=0)
        self.null_live_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self._plot_null_live_placeholder("Waiting for initial APAPANE frame…")

        right = ttk.Frame(outer, padding=(12, 0, 0, 0), width=310)
        right.grid(row=0, column=2, rowspan=2, sticky="nsew")
        right.grid_propagate(False)
        brand = tk.Frame(right, bg=GREEN, height=64)
        brand.pack(side="bottom", fill="x", pady=(12, 0))
        brand.pack_propagate(False)
        tk.Label(
            brand,
            text="GLINT\nALIGNMENT",
            bg=GREEN,
            fg=BG,
            justify="left",
            anchor="w",
            font=(self.retro_font, 12, "bold"),
            padx=14,
            pady=5,
        ).pack(side="left", fill="both", expand=True)
        alien = tk.Canvas(
            brand,
            # Extra canvas width is added on the right-anchored widget's left
            # side, moving the face away from the footer boundary and leaving
            # a safer clipping margin on the right.
            width=72,
            height=58,
            bg=GREEN,
            highlightthickness=0,
            borderwidth=0,
        )
        alien.pack(side="right", padx=(4, 10), pady=3)
        alien.create_polygon(
            31, 3,
            44, 7,
            52, 17,
            50, 30,
            43, 43,
            31, 54,
            19, 43,
            12, 30,
            10, 17,
            18, 7,
            fill=BG,
            outline=BG,
            smooth=True,
            splinesteps=24,
        )
        alien.create_polygon(18, 20, 29, 24, 25, 34, 17, 29, fill=GREEN, outline=GREEN, smooth=True)
        alien.create_polygon(44, 20, 33, 24, 37, 34, 45, 29, fill=GREEN, outline=GREEN, smooth=True)
        alien.create_arc(25, 36, 37, 45, start=200, extent=140, style="arc", outline=GREEN, width=2)
        alien.move("all", ALIEN_X_SHIFT, 0)
        right_content = ttk.Frame(right)
        right_content.pack(side="top", fill="both", expand=True)
        ttk.Label(right_content, text="SCAN SETTINGS", style="Header.TLabel").pack(anchor="w")
        self.parameter_frame = ttk.Frame(right_content)
        self.parameter_frame.pack(fill="x", pady=8)
        self.run_button = ttk.Button(
            right_content,
            text="RUN SELECTED SCAN",
            style="Accent.TButton",
            command=self.run_selected,
        )
        self.run_button.pack(fill="x", pady=4)
        self.abort_button = ttk.Button(right_content, text="ABORT + RESTORE", command=self.abort, state="disabled")
        ttk.Separator(right_content).pack(fill="x", pady=10)
        ttk.Label(right_content, text="REVIEW", style="Header.TLabel").pack(anchor="w")
        self.apply_button = ttk.Button(
            right_content,
            text="APPLY RECOMMENDED VALUES",
            command=self.apply_result,
            state="disabled",
        )
        self.apply_button.pack(fill="x", pady=2)
        self.xy_loop_button = ttk.Button(
            right_content,
            text="APPLY X/Y → PITCH/YAW",
            command=lambda: self.apply_result(next_kind="pitch_yaw"),
            state="disabled",
        )
        self.xy_loop_button.pack(fill="x", pady=2)
        self.redo_button = ttk.Button(right_content, text="REDO SCAN", command=self.redo, state="disabled")
        self.redo_button.pack(fill="x", pady=2)
        self.exit_button = ttk.Button(right_content, text="RESTORE / EXIT", command=self.exit_review, state="disabled")
        self.exit_button.pack(fill="x", pady=2)

        status_panel = ttk.Frame(outer, padding=(0, 8, 0, 0), height=125)
        status_panel.grid(row=1, column=1, sticky="nsew")
        status_panel.grid_propagate(False)
        # Mount readback and Status each own exactly half of the hardware strip.
        status_panel.columnconfigure(0, weight=1, minsize=0, uniform="hardware_strip")
        status_panel.columnconfigure(1, weight=1, minsize=0, uniform="hardware_strip")
        status_panel.rowconfigure(0, weight=1)

        mount_panel = ttk.Frame(status_panel, padding=(0, 0, 14, 0))
        mount_panel.grid(row=0, column=0, sticky="nsew")
        self.mount_header_button = tk.Button(
            mount_panel,
            text="MOUNT CONFIGURATION",
            command=self.open_advanced_mount,
            bg=BG,
            fg=GREEN,
            activebackground=BG,
            activeforeground=FG,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            anchor="w",
            cursor="hand2",
            font=(self.retro_font, 14, "bold"),
        )
        self.mount_header_button.pack(fill="x", pady=(0, 4))
        mount_grid = tk.Frame(mount_panel, bg=BG)
        mount_grid.pack(fill="both", expand=True)
        for index, axis in enumerate(("pitch", "yaw", "roll", "x", "y", "z")):
            row, column = divmod(index, 3)
            cell = tk.Frame(mount_grid, bg="#080808", highlightbackground=GRID, highlightthickness=1)
            cell.grid(row=row, column=column, sticky="nsew", padx=2, pady=2)
            mount_grid.columnconfigure(column, weight=1)
            mount_grid.rowconfigure(row, weight=1)
            tk.Label(
                cell,
                text=axis.upper(),
                bg="#080808",
                fg=GREEN,
                font=(self.retro_font, 8, "bold"),
            ).pack(anchor="center", pady=(2, 0))
            tk.Label(
                cell,
                textvariable=self.mount_values[axis],
                bg="#080808",
                fg=FG,
                font=(self.retro_font, 9),
            ).pack(anchor="center", pady=(0, 2))

        status_right = ttk.Frame(status_panel)
        status_right.grid(row=0, column=1, sticky="nsew")
        status_right.columnconfigure(0, weight=1)
        status_right.rowconfigure(4, weight=1)
        status_header = ttk.Frame(status_right)
        status_header.grid(row=0, column=0, sticky="ew")
        ttk.Label(status_header, text="STATUS", style="Header.TLabel").pack(side="left")
        self.status_message_label = ttk.Label(status_right, textvariable=self.status, anchor="w", justify="left")
        self.status_message_label.grid(row=1, column=0, sticky="ew", pady=(1, 0))
        status_right.bind(
            "<Configure>",
            lambda event: self.status_message_label.configure(wraplength=max(event.width - 8, 1)),
        )
        self.progress = ttk.Progressbar(status_right, mode="determinate", maximum=100)
        self.progress.grid(row=2, column=0, sticky="ew", pady=(5, 1))
        ttk.Label(status_right, textvariable=self.progress_text, style="Muted.TLabel").grid(
            row=3, column=0, sticky="ew", pady=(0, 3)
        )
        self.log_text = tk.Text(
            status_right,
            width=1,
            height=3,
            bg="#050505",
            fg="#d7ded9",
            insertbackground=FG,
            relief="flat",
            wrap="word",
            state="disabled",
            cursor="arrow",
            takefocus=0,
            font=(self.retro_font, 8),
        )
        self.log_text.grid(row=4, column=0, sticky="nsew")

    def open_workflow_defaults(self) -> None:
        """Open the persistent editor for every scan's configured defaults."""
        if self.running:
            messagebox.showinfo("Workflow defaults", "Wait for the active operation to finish before editing defaults.")
            return
        window = self.workflow_defaults_window
        if window is not None and window.winfo_exists():
            window.lift()
            window.focus_force()
            return
        window = tk.Toplevel(self.root)
        self.workflow_defaults_window = window
        window.title("Workflow default scan settings")
        window.geometry("850x660")
        window.minsize(760, 580)
        window.configure(bg=BG)
        window.protocol("WM_DELETE_WINDOW", self._close_workflow_defaults)
        body = ttk.Frame(window, padding=16)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="WORKFLOW DEFAULT SCAN SETTINGS", style="Header.TLabel").pack(anchor="w")
        config_path = str(self.config.get("_config_path", "active configuration JSON"))
        ttk.Label(
            body,
            text=(
                "Select a workflow step, edit its complete default configuration, then press UPDATE DEFAULTS. "
                "Path fields are read-only here and must be changed manually in the JSON. "
                f"Changes are used immediately and saved for future launches in:\n{config_path}"
            ),
            style="Muted.TLabel",
            wraplength=790,
        ).pack(fill="x", anchor="w", pady=(4, 12))

        content = ttk.Frame(body)
        content.pack(fill="both", expand=True)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)
        selector = ttk.Frame(content, padding=(0, 0, 12, 0), width=210)
        selector.grid(row=0, column=0, sticky="ns")
        selector.grid_propagate(False)
        editor = ttk.Frame(content)
        editor.grid(row=0, column=1, sticky="nsew")
        editor.columnconfigure(0, weight=1)
        editor.rowconfigure(0, weight=1)

        self.workflow_default_fields = {}
        self.workflow_default_frames = {}
        self.workflow_default_buttons = {}
        for kind, label in LABELS.items():
            button = ttk.Button(
                selector,
                text=label,
                style="Step.TButton",
                command=lambda selected=kind: self._show_workflow_default_scan(selected),
            )
            button.pack(fill="x", pady=2)
            self.workflow_default_buttons[kind] = button
            frame = ttk.Frame(editor, padding=(8, 4, 4, 4))
            frame.grid(row=0, column=0, sticky="nsew")
            frame.columnconfigure(1, weight=1)
            ttk.Label(frame, text=label.upper(), style="Header.TLabel").grid(
                row=0, column=0, columnspan=2, sticky="w", pady=(0, 10)
            )
            fields = {}
            for row_index, (path, value) in enumerate(flatten_default_settings(self.config["scans"][kind]), start=1):
                label_text = "  /  ".join(part.replace("_", " ").title() for part in path)
                manual_path = path[-1] in {"response_matrix"}
                if isinstance(value, list):
                    label_text += "  [JSON list]"
                elif isinstance(value, bool):
                    label_text += "  [true / false]"
                if manual_path:
                    label_text += "  [manual JSON only]"
                ttk.Label(frame, text=label_text).grid(row=row_index, column=0, sticky="w", padx=(0, 14), pady=4)
                variable = tk.StringVar(value=format_default_setting(value))
                entry = ttk.Entry(frame, textvariable=variable, width=42)
                entry.grid(row=row_index, column=1, sticky="ew", pady=4)
                if manual_path:
                    entry.configure(state="disabled")
                fields[path] = (variable, deepcopy(value))
            self.workflow_default_fields[kind] = fields
            self.workflow_default_frames[kind] = frame

        actions = ttk.Frame(body)
        actions.pack(fill="x", pady=(14, 0))
        ttk.Button(actions, text="CANCEL", command=self._close_workflow_defaults).pack(side="right")
        ttk.Button(
            actions,
            text="UPDATE DEFAULTS",
            style="Run.TButton",
            command=self._update_workflow_defaults,
        ).pack(side="right", padx=(0, 8))
        self._show_workflow_default_scan(self.current_kind)

    def _show_workflow_default_scan(self, kind: str) -> None:
        for name, button in self.workflow_default_buttons.items():
            button.configure(style="Review.Step.TButton" if name == kind else "Step.TButton")
        self.workflow_default_frames[kind].tkraise()

    def _close_workflow_defaults(self) -> None:
        if self.workflow_defaults_window is not None:
            self.workflow_defaults_window.destroy()
        self.workflow_defaults_window = None
        self.workflow_default_fields = {}
        self.workflow_default_frames = {}
        self.workflow_default_buttons = {}

    def _update_workflow_defaults(self) -> None:
        candidate_scans = deepcopy(self.config["scans"])
        try:
            for kind, fields in self.workflow_default_fields.items():
                for path, (variable, original) in fields.items():
                    if path[-1] in {"response_matrix"} and variable.get() != format_default_setting(original):
                        raise ValueError(
                            f"{kind}.{'/'.join(path)} is a path and must be changed manually in the active JSON"
                        )
                    try:
                        value = parse_default_setting(variable.get(), original)
                    except (TypeError, ValueError, json.JSONDecodeError) as error:
                        dotted = ".".join((kind,) + path)
                        raise ValueError(f"{dotted}: {error}") from error
                    assign_default_setting(candidate_scans[kind], path, value)
            candidate = deepcopy(self.config)
            candidate["scans"] = candidate_scans
            validate_scan_defaults(candidate, self.hardware.mount_limits())
            for kind, parameters in candidate_scans.items():
                self.engine.validate_scan_parameters(kind, parameters)
            path = save_config(candidate)
        except (OSError, TypeError, ValueError) as error:
            messagebox.showerror("Defaults not updated", str(error))
            return

        self.config["scans"] = candidate_scans
        self.select_scan(self.current_kind)
        self.status.set(f"Workflow defaults updated and saved: {path}")
        self._append_log(f"Workflow defaults updated in {path}")
        self._close_workflow_defaults()
        messagebox.showinfo(
            "Workflow defaults updated",
            f"The new defaults are active now and will be loaded on future launches.\n\nSaved to:\n{path}",
        )

    def select_scan(self, kind: str) -> None:
        if self.running:
            return
        self.current_kind = kind
        self.current_result = None
        self.review_pending = False
        self._clear_review()
        self._clear_progress_area(clear_log=True)
        for child in self.parameter_frame.winfo_children():
            child.destroy()
        self.fields.clear()
        params = self.config["scans"]
        definitions: list[tuple[str, str, float | int]] = []
        if kind in {"focal_dark", "apapane_dark"}:
            definitions = [("nframes", "Frames", params[kind]["nframes"])]
        elif kind == "bulk":
            definitions = [
                ("nframes", "Frames / measurement", params["bulk"]["nframes"]),
                ("max_iterations", "Maximum iterations", params["bulk"]["max_iterations"]),
                ("tolerance_px", "Tolerance (pixels)", params["bulk"]["tolerance_px"]),
            ]
        elif kind == "set_z":
            definitions = [("z", "Z position (pulses)", params["set_z"]["z"])]
        elif kind == "pitch_yaw":
            definitions = self._grid_definitions(params[kind], "pitch", "yaw")
        elif kind == "xy":
            definitions = self._grid_definitions(params[kind], "x", "y")
        elif kind == "tip_tilt":
            definitions = self._grid_definitions(params[kind], "tip", "tilt")
        elif kind == "null":
            piston = params[kind]["piston"]
            definitions = [
                ("piston.start", "Piston start (µm)", piston["start"]),
                ("piston.span", "Piston range (µm)", piston["span"]),
                ("piston.step", "Piston step (µm)", piston["step"]),
                ("nframes", "Frames / point", params[kind]["nframes"]),
                ("numavg", "Repeated scans", params[kind]["numavg"]),
            ]
        for key, label, value in definitions:
            row = ttk.Frame(self.parameter_frame)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label).pack(side="left")
            variable = tk.StringVar(value=str(value))
            ttk.Entry(row, textvariable=variable, width=11).pack(side="right")
            self.fields[key] = variable
        self.run_button.configure(text=f"RUN {LABELS[kind].split('  ', 1)[-1].upper()}")
        self.status.set(f"Ready: {LABELS[kind]}. Defaults loaded; confirm or edit settings.")
        self._plot_ready(kind)
        self._update_rest_dm_control_state()
        try:
            self.refresh_mount()
        except Exception as error:
            self._append_log(f"WARNING: mount display refresh failed: {error}")

    @staticmethod
    def _grid_definitions(params: dict, first: str, second: str) -> list[tuple[str, str, float | int]]:
        definitions = []
        for axis in (first, second):
            for key, label in (("start", "start"), ("span", "range"), ("step", "step")):
                definitions.append((f"{axis}.{key}", f"{axis.title()} {label}", params[axis][key]))
        if "nframes" in params:
            definitions.append(("nframes", "Frames / point", params["nframes"]))
        return definitions

    def _parameters(self) -> dict:
        if self.current_kind in {"focal_dark", "apapane_dark"}:
            return {"nframes": int(float(self.fields["nframes"].get()))}
        base = deepcopy(self.config["scans"][self.current_kind])
        for key, variable in self.fields.items():
            value = float(variable.get())
            if key in {"nframes", "max_iterations", "numavg"}:
                value = int(value)
            if "." in key:
                group, subkey = key.split(".")
                base[group][subkey] = value
            else:
                base[key] = value
        return base

    def run_full_workflow(self) -> None:
        if self.running:
            return
        try:
            start, end, selected = self._selected_workflow_range()
            self._validate_partial_workflow_prerequisites(selected)
        except ValueError as error:
            messagebox.showerror("Invalid workflow range", str(error))
            return
        darks = [kind for kind in selected if kind in {"focal_dark", "apapane_dark"}]
        if darks:
            if darks == ["focal_dark", "apapane_dark"]:
                prompt = (
                    "Turn OFF or block the light source. The focal-plane and APAPANE darks will now be "
                    "acquired consecutively; keep the source blocked until prompted to turn it on."
                )
            else:
                camera_name = "focal-plane" if darks[0] == "focal_dark" else "APAPANE"
                prompt = f"Turn OFF or block the light source reaching the {camera_name} camera, then press OK."
            if not messagebox.askokcancel("Prepare dark acquisition", prompt):
                return
        state_name = self.config.get("workflow", {}).get("start_dm_state", "zero_ptt")
        try:
            self.engine.ensure_fixed_roll()
            self.hardware.set_dm(segment_state(self.config, state_name))
            output_directory = begin_alignment(self.config)
        except Exception as error:
            self.status.set(f"Cannot start workflow: {error}")
            self._append_log(traceback.format_exc())
            messagebox.showerror("Workflow startup failed", str(error))
            return
        self.refresh_dm()
        self.refresh_mount()
        self.workflow.reset()
        self._update_step_styles()
        self.full_workflow_active = True
        self.workflow_selected_kinds = list(selected)
        self.workflow_range = {
            "start": start,
            "end": end,
            "steps": list(selected),
        }
        self.workflow_queue = list(selected)
        self.status.set(f"Automatic alignment steps {start}–{end}; output: {output_directory}")
        self._run_next_workflow_step()

    def _selected_workflow_range(self) -> tuple[int, int, list[str]]:
        start_var = getattr(self, "workflow_start", None)
        end_var = getattr(self, "workflow_end", None)
        start = int(start_var.get()) if start_var is not None else 1
        end = int(end_var.get()) if end_var is not None else len(LABELS)
        if not 1 <= start <= len(LABELS) or not 1 <= end <= len(LABELS):
            raise ValueError(f"Start and end must be between 1 and {len(LABELS)}.")
        if start > end:
            raise ValueError("Workflow start must not be after workflow end.")
        return start, end, list(LABELS)[start - 1 : end]

    def _validate_partial_workflow_prerequisites(self, selected: list[str]) -> None:
        if "bulk" in selected and "focal_dark" not in selected and self.engine.focal_dark is None:
            raise ValueError(
                "The selected range includes bulk alignment but no focal dark is loaded. "
                "Include step 1 or acquire a focal dark first."
            )
        apapane_users = {"pitch_yaw", "xy", "tip_tilt", "null"}
        if apapane_users.intersection(selected) and "apapane_dark" not in selected and self.engine.apapane_dark is None:
            raise ValueError(
                "The selected range includes an APAPANE-based scan but no APAPANE dark is loaded. "
                "Include step 2 or acquire an APAPANE dark first."
            )

    def _run_next_workflow_step(self) -> None:
        if not self.workflow_queue:
            self.full_workflow_active = False
            try:
                path = save_final_configuration(
                    self.config,
                    self.hardware,
                    self.engine,
                    workflow_status=self.workflow.as_dict(),
                    workflow_range=self.workflow_range,
                )
                self.status.set(f"Requested workflow range complete; final configuration saved: {path}")
                messagebox.showinfo("GLINT alignment", f"Requested alignment workflow complete.\n\nFinal configuration:\n{path}")
            except Exception as error:
                self.status.set(f"Workflow complete, but final configuration save failed: {error}")
                messagebox.showerror("Final configuration save failed", str(error))
            return
        self.select_scan(self.workflow_queue.pop(0))
        self.run_selected()

    def run_selected(self) -> None:
        if self.running:
            return
        try:
            parameters = self._parameters()
            self.engine.validate_scan_parameters(self.current_kind, parameters)
        except ValueError as error:
            messagebox.showerror("Invalid parameter", str(error))
            return
        if self.current_kind in {"focal_dark", "apapane_dark"} and not getattr(self, "full_workflow_active", False):
            camera_name = "focal-plane" if self.current_kind == "focal_dark" else "APAPANE"
            if not messagebox.askokcancel(
                "Dark acquisition",
                f"Turn OFF or block the light source reaching the {camera_name} camera, then press OK to acquire the dark.",
            ):
                return
        kind = self.current_kind
        run_scope = "workflow" if self.full_workflow_active else "individual"
        self._clear_progress_area(clear_log=True)
        self.workflow.start(kind, allow_iteration=self.full_workflow_active)
        self._update_step_styles()
        self._set_running(True, scan=True)
        self._last_live_dm_refresh = 0.0
        self.status.set(f"Running {LABELS[kind]}…")

        def task() -> None:
            try:
                if kind == "focal_dark":
                    result = self.engine.acquire_dark("focal", parameters["nframes"])
                elif kind == "apapane_dark":
                    result = self.engine.acquire_dark("apapane", parameters["nframes"])
                elif kind == "bulk":
                    self.config["scans"]["bulk"].update(parameters)
                    result = self.engine.bulk_align()
                else:
                    result = getattr(self.engine, f"{kind}_scan")(parameters)
                result.metadata["run_scope"] = run_scope
                self.events.put(("result", result))
            except ScanCancelled as error:
                self.events.put(("cancelled", str(error), kind))
            except Exception as error:
                self.events.put(("error", str(error), traceback.format_exc(), kind))

        threading.Thread(target=task, daemon=True).start()

    def acquire_camera_frame(self) -> None:
        if self.running:
            return
        camera = CAMERA_DISPLAY_TO_KEY[self.camera_choice.get()]
        requested_name = self.camera_filename.get()
        requested_directory = self.camera_directory.get()
        try:
            nframes = int(self.camera_nframes.get())
            if nframes < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid frame count", "Number of frames must be a positive integer.")
            return
        self._clear_progress_area(clear_log=True)
        self._set_running(True, scan=False)
        noun = "frame" if nframes == 1 else "frames"
        self.progress_text.set(f"Acquiring {nframes} {CAMERA_LABELS[camera]} {noun}…")
        self.status.set(f"Acquiring {CAMERA_LABELS[camera]} {noun}…")

        def task() -> None:
            try:
                frames = np.asarray(self.hardware.camera_frames(camera, nframes), dtype=float)
                if frames.ndim == 2:
                    frames = frames[None, :, :]
                saved_data = frames[0] if nframes == 1 else frames
                preview = np.mean(frames, axis=0)
                path = save_camera_frame(
                    saved_data,
                    camera,
                    requested_name,
                    self.config,
                    requested_directory=requested_directory,
                )
                self.events.put(("camera_saved", camera, preview, path, nframes))
            except Exception as error:
                self.events.put(("camera_error", str(error), traceback.format_exc()))

        threading.Thread(target=task, daemon=True).start()

    def choose_camera_directory(self) -> None:
        selected = filedialog.askdirectory(
            title="Select camera-frame destination",
            initialdir=self.camera_directory.get() or str(self.config["camera_frame_root"]),
        )
        if selected:
            self.camera_directory.set(selected)

    def _clear_progress_area(self, *, clear_log: bool) -> None:
        """Start each operation with no stale progress or acquisition text."""
        self.progress["value"] = 0
        self.progress_text.set("")
        if clear_log and hasattr(self, "log_text"):
            self.log_text.configure(state="normal")
            try:
                self.log_text.delete("1.0", "end")
            finally:
                self.log_text.configure(state="disabled")

    def abort(self) -> None:
        if self.running and self.scan_running:
            self.status.set("Aborting; restoring mount, Conex, DM PTT, and DM voltages…")
            self.abort_button.configure(state="disabled")
            self.engine.abort()

    def _set_running(self, running: bool, *, scan: bool = False) -> None:
        self.running = running
        self.scan_running = running and scan
        state = "disabled" if running else "normal"
        if hasattr(self, "workflow_header_button"):
            self.workflow_header_button.configure(state=state)
        if running and self.workflow_defaults_window is not None:
            self._close_workflow_defaults()
        self.run_button.configure(state=state)
        self.camera_button.configure(state=state)
        self.camera_combo.configure(state="disabled" if running else "readonly")
        self.camera_frames_entry.configure(state=state)
        self.camera_browse_button.configure(state=state)
        if hasattr(self, "workflow_run_button"):
            self.workflow_run_button.configure(state=state)
            self.workflow_start_entry.configure(state=state)
            self.workflow_end_entry.configure(state=state)
        for button in self.step_buttons.values():
            button.configure(state=state)
        if self.scan_running:
            self.abort_button.configure(state="normal")
            if not self.abort_button.winfo_manager():
                self.abort_button.pack(fill="x", pady=(0, 4), after=self.run_button)
        else:
            self.abort_button.configure(state="disabled")
            if self.abort_button.winfo_manager():
                self.abort_button.pack_forget()
        if running:
            self._clear_review()
        self._update_rest_dm_control_state()
        self._update_dm_inspector_control_state()
        self._update_advanced_mount_controls()
        self._update_null_live_button_state()

    def _progress_from_worker(self, message: str, current: int, total: int) -> None:
        self.events.put(("progress", message, current, total))

    def _log_from_worker(self, message: str) -> None:
        self.events.put(("log", message))

    def _drain_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "progress":
                    _, message, current, total = event
                    self.progress["value"] = 100 * current / max(total, 1)
                    self.progress_text.set(f"{message}  ({current}/{total})")
                elif event[0] == "log":
                    self._append_log(event[1])
                elif event[0] == "result":
                    self._handle_result(event[1])
                elif event[0] == "cancelled":
                    _, _, kind = event
                    self.workflow.abort(kind)
                    self._update_step_styles()
                    self._set_running(False)
                    self.workflow_queue.clear()
                    self.full_workflow_active = False
                    self.review_pending = False
                    self.progress["value"] = 0
                    self.progress_text.set("")
                    self.status.set("Aborted; complete pre-scan state restored; no scan files saved")
                    if kind in {"focal_dark", "apapane_dark"}:
                        messagebox.showinfo("Dark acquisition stopped", "Turn the light source ON or unblock the beam.")
                    self.refresh_dm()
                    self.refresh_mount()
                elif event[0] == "error":
                    _, message, details, kind = event
                    self.workflow.finish(kind, False)
                    self._update_step_styles()
                    self._set_running(False)
                    self.workflow_queue.clear()
                    self.full_workflow_active = False
                    self.review_pending = False
                    self.progress["value"] = 0
                    self.progress_text.set("")
                    self.status.set(f"Error in {LABELS[kind]}: {message}")
                    self._append_log(details)
                    messagebox.showerror("Scan failed", message)
                    if kind in {"focal_dark", "apapane_dark"}:
                        messagebox.showinfo("Dark acquisition stopped", "Turn the light source ON or unblock the beam.")
                    self.refresh_dm()
                    self.refresh_mount()
                elif event[0] == "applied":
                    self._finish_apply(*event[1:])
                elif event[0] == "rest_dm_applied":
                    self._handle_rest_dm_applied(event[1])
                elif event[0] == "rest_dm_error":
                    self._set_running(False)
                    self.status.set(f"REST OF DM command failed: {event[1]}")
                    self._append_log(event[2])
                    messagebox.showerror("REST OF DM failed", event[1])
                    self.refresh_dm()
                elif event[0] == "dm_segment_applied":
                    self._set_running(False)
                    self.status.set(f"DM segment {event[1]} values applied")
                    self.refresh_dm()
                elif event[0] == "dm_segment_error":
                    self._set_running(False)
                    self.status.set(f"DM segment command failed: {event[1]}")
                    self._append_log(event[2])
                    messagebox.showerror("DM segment command failed", event[1])
                    self.refresh_dm()
                elif event[0] == "dm_preset_applied":
                    self._set_running(False)
                    self.status.set(f"{event[2]} applied")
                    self.refresh_dm()
                elif event[0] == "dm_preset_error":
                    self._set_running(False)
                    self.status.set(f"DM preset command failed: {event[2]}")
                    self._append_log(event[3])
                    messagebox.showerror("DM preset command failed", event[2])
                    self.refresh_dm()
                elif event[0] == "mount_jogged":
                    self._set_running(False)
                    self.status.set(f"Mount {event[1]} moved by {event[2]:+g}")
                    self.refresh_mount()
                elif event[0] == "mount_jog_error":
                    self._set_running(False)
                    self.status.set(f"Advanced mount command failed: {event[1]}")
                    self._append_log(event[2])
                    messagebox.showerror("Advanced mount command failed", event[1])
                    self.refresh_mount()
                elif event[0] == "null_live":
                    self._null_live_acquiring = False
                    self._draw_null_live(*event[1:])
                    self._update_null_live_button_state()
                elif event[0] == "null_live_error":
                    self._null_live_acquiring = False
                    self._plot_null_live_placeholder(f"APAPANE live update failed\n{event[1]}")
                    self._update_null_live_button_state()
                elif event[0] == "camera_saved":
                    self._handle_camera_saved(*event[1:])
                elif event[0] == "camera_error":
                    self._set_running(False)
                    self.progress["value"] = 0
                    self.progress_text.set("")
                    self.status.set(f"Camera acquisition failed: {event[1]}")
                    self._append_log(event[2])
                    messagebox.showerror("Camera acquisition failed", event[1])
        except queue.Empty:
            pass
        refresh_interval = float(self.config.get("workflow", {}).get("live_dm_refresh_s", 0.5))
        now = time.monotonic()
        if self.running and self.scan_running and now - self._last_live_dm_refresh >= refresh_interval:
            self._last_live_dm_refresh = now
            try:
                self.refresh_dm()
                self.refresh_mount()
            except Exception as error:
                self._append_log(f"WARNING: live hardware display refresh failed: {error}")
        self.root.after(75, self._drain_events)

    def _handle_result(self, result: ScanResult) -> None:
        self._set_running(False)
        self.refresh_dm()
        self.refresh_mount()
        self.current_result = result
        passed = bool(result.recommendation.get("passed", True))
        self.workflow.finish(result.kind, passed)
        if result.kind.endswith("_dark") and passed:
            # Darks are acquired and saved directly; unlike alignment scans,
            # they do not have an Apply Recommended Values review action.
            self.workflow.confirm(result.kind)
            if result.metadata.get("run_scope") == "individual":
                self.workflow.invalidate_after(result.kind)
        self._update_step_styles()
        self.progress["value"] = 100
        if result.kind.endswith("_dark"):
            self.review_pending = False
            path = save_result(result, self.hardware, self.config, "acquired")
            camera_name = "focal-plane" if result.kind == "focal_dark" else "APAPANE"
            nframes = result.metadata["nframes"]
            self.progress_text.set(f"{nframes} {camera_name} dark frames acquired")
            self.status.set(f"{camera_name.title()} dark acquired and saved: {path}")
            self._plot_dark_status(result.kind, nframes)
            # During a full workflow both darks are consecutive: keep the source
            # blocked after step 1 and prompt to restore it only after step 2.
            keep_blocked_for_apapane = bool(
                self.full_workflow_active
                and result.kind == "focal_dark"
                and "apapane_dark" in self.workflow_queue
            )
            if not keep_blocked_for_apapane:
                messagebox.showinfo(
                    "Dark acquired",
                    f"{camera_name.title()} dark saved. Turn the light source ON or unblock the beam before continuing.",
                )
            if self.full_workflow_active:
                self.root.after(250, self._run_next_workflow_step)
            return
        self.review_pending = True
        self._plot_result(result)
        try:
            path = save_result(result, self.hardware, self.config, "awaiting_review")
            save_review_figure(result, path, self.figure)
        except Exception as error:
            self._append_log(f"ERROR: completed scan could not be saved before review: {error}")
            messagebox.showerror("Completed scan save failed", str(error))
        self.redo_button.configure(state="normal")
        self.exit_button.configure(state="normal")
        self._update_apply_state()
        self.status.set(
            "Review the plots, then apply, redo, or restore/exit."
            if passed
            else "Safeguard failed. Apply is disabled; redo the scan or restore/exit."
        )
        self._update_rest_dm_control_state()

    def _handle_camera_saved(self, camera: str, frame: np.ndarray, path, nframes: int) -> None:
        self._set_running(False)
        self.progress["value"] = 100
        noun = "frame" if nframes == 1 else "frames"
        self.progress_text.set(f"{nframes} {CAMERA_LABELS[camera]} {noun} acquired")
        self.status.set(f"{nframes} {CAMERA_LABELS[camera]} {noun} saved: {path}")
        self._plot_camera_frame(camera, frame)

    def _prepare_figure(self) -> None:
        self.figure.clear()
        self.figure.set_facecolor(BG)

    def _style_axis(self, ax, *, grid: bool = False) -> None:
        ax.set_facecolor(BG)
        ax.title.set_color(GREEN)
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        ax.tick_params(colors=FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#7b867f")
        if grid:
            ax.grid(color=GRID, alpha=0.45, linewidth=0.7)

    @staticmethod
    def _style_legend(legend) -> None:
        if legend is None:
            return
        legend.get_frame().set_facecolor("#080808")
        legend.get_frame().set_edgecolor(GREEN)
        for text in legend.get_texts():
            text.set_color(FG)

    def _style_colorbar(self, colorbar) -> None:
        colorbar.ax.tick_params(colors=FG, labelsize=7)
        colorbar.outline.set_edgecolor("#7b867f")

    def _plot_ready(self, kind: str) -> None:
        self._prepare_figure()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(BG)
        ax.axis("off")
        ax.text(0.5, 0.55, LABELS[kind].upper(), color=GREEN, fontsize=18, ha="center", fontfamily="monospace")
        ax.text(0.5, 0.46, "CONFIRM SETTINGS, THEN RUN", color=MUTED, fontsize=10, ha="center", fontfamily="monospace")
        self.canvas.draw_idle()

    def _plot_dark_status(self, kind: str, nframes: int) -> None:
        self._prepare_figure()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(BG)
        ax.axis("off")
        ax.text(0.5, 0.57, "✓", fontsize=76, color=GREEN, ha="center", va="center")
        ax.text(
            0.5,
            0.39,
            f"{kind.replace('_', ' ').upper()} — {nframes} FRAMES SAVED",
            fontsize=14,
            color=GREEN,
            ha="center",
            fontfamily="monospace",
        )
        self.canvas.draw_idle()

    def _plot_camera_frame(self, camera: str, frame: np.ndarray) -> None:
        self._prepare_figure()
        ax = self.figure.add_subplot(111)
        image = ax.imshow(frame, origin="lower", cmap="magma", aspect="equal")
        ax.set_title(f"{CAMERA_LABELS[camera]} frame")
        self._style_axis(ax)
        colorbar = self.figure.colorbar(image, ax=ax, shrink=0.78)
        self._style_colorbar(colorbar)
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    def _plot_result(self, result: ScanResult) -> None:
        self._prepare_figure()
        if result.kind == "set_z":
            ax = self.figure.add_subplot(111)
            ax.set_facecolor(BG)
            ax.axis("off")
            ax.text(
                0.5,
                0.54,
                f"Z SET TO {result.recommendation['z']:.0f}",
                color=GREEN,
                fontsize=20,
                ha="center",
                va="center",
                fontfamily="monospace",
            )
        elif result.kind == "bulk":
            grid = self.figure.add_gridspec(
                2,
                2,
                left=0.07,
                right=0.96,
                bottom=0.10,
                top=0.90,
                width_ratios=[1.0, 1.75],
                hspace=0.38,
                wspace=0.34,
            )
            ref_ax = self.figure.add_subplot(grid[0, 0])
            final_ax = self.figure.add_subplot(grid[1, 0])
            conv_ax = self.figure.add_subplot(grid[:, 1])
            y0, y1, x0, x1 = map(int, self.config["scans"]["bulk"]["crop"])
            ref_ax.imshow(result.maps["reference_psf"][y0:y1, x0:x1], origin="lower", cmap="magma", aspect="equal")
            final_ax.imshow(result.maps["current_psf"][y0:y1, x0:x1], origin="lower", cmap="magma", aspect="equal")
            ref_ax.set_title("Reference focal plane")
            final_ax.set_title("After alignment")
            history = result.estimates["history"]
            conv_ax.plot(
                [h["iteration"] for h in history],
                [h["error_px"] for h in history],
                "o-",
                color=GREEN,
                markerfacecolor=BG,
                markeredgewidth=1.5,
                linewidth=2,
                label="centroid error",
            )
            conv_ax.axhline(
                self.config["scans"]["bulk"]["tolerance_px"],
                color=ORANGE,
                linestyle="--",
                linewidth=1.5,
                label="tolerance",
            )
            conv_ax.set(xlabel="Iteration", ylabel="Centroid error (pixels)", title="Safeguarded convergence")
            self._style_axis(ref_ax)
            self._style_axis(final_ax)
            self._style_axis(conv_ax, grid=True)
            self._style_legend(conv_ax.legend(fontsize=8))
        elif result.kind in {"pitch_yaw", "xy", "tip_tilt"}:
            estimates = result.estimates["channels"] if result.kind in {"pitch_yaw", "xy"} else result.estimates
            xname, yname = list(result.axes)[:2]
            xv, yv = result.axes[xname], result.axes[yname]
            if result.kind in {"pitch_yaw", "xy"}:
                self.figure.suptitle(
                    f"OPTIMAL CENTRE  {xname.upper()}={result.recommendation[xname]:.2f}   "
                    f"{yname.upper()}={result.recommendation[yname]:.2f}",
                    color=GREEN,
                    fontsize=10,
                    fontfamily="monospace",
                    y=0.97,
                )
            else:
                self.figure.suptitle(
                    "OPTIMAL TIP / TILT FOR EACH INJECTED SEGMENT",
                    color=GREEN,
                    fontsize=10,
                    fontfamily="monospace",
                    y=0.97,
                )
            grid = self.figure.add_gridspec(
                1,
                len(result.maps),
                left=0.075,
                right=0.94,
                bottom=0.15,
                top=0.76,
                wspace=0.62,
            )
            for index, (name, data) in enumerate(result.maps.items()):
                ax = self.figure.add_subplot(grid[0, index])
                image = ax.imshow(
                    data,
                    origin="lower",
                    aspect="equal",
                    extent=[xv[0], xv[-1], yv[0], yv[-1]],
                    cmap="viridis",
                )
                estimate = estimates[index] if isinstance(estimates, list) else estimates[name]
                ax.plot(
                    estimate["x"],
                    estimate["y"],
                    marker="x",
                    color=RED,
                    ms=9,
                    mew=2,
                    linestyle="none",
                    label=estimate.get("method", "fit"),
                )
                if result.kind in {"pitch_yaw", "xy"}:
                    recommendation = result.recommendation
                    ax.plot(recommendation[xname], recommendation[yname], "o", color=GREEN, ms=5, label="average")
                    title = (
                        f"{name}\n{xname.upper()}={estimate['x']:.2f}   "
                        f"{yname.upper()}={estimate['y']:.2f}"
                    )
                else:
                    title = (
                        f"SEGMENT {name}\nTIP={estimate['x']:.3f}   "
                        f"TILT={estimate['y']:.3f} mrad"
                    )
                ax.set(xlabel=xname, ylabel=yname, title=title)
                ax.title.set_fontsize(8)
                ax.title.set_fontfamily("monospace")
                self._style_axis(ax)
                self._style_legend(ax.legend(fontsize=6, loc="best"))
                divider = make_axes_locatable(ax)
                colorbar_axis = divider.append_axes("right", size="4%", pad=0.07)
                colorbar = self.figure.colorbar(image, cax=colorbar_axis)
                self._style_colorbar(colorbar)
        elif result.kind == "null":
            grid = self.figure.add_gridspec(
                1,
                2,
                left=0.08,
                right=0.96,
                bottom=0.15,
                top=0.82,
                wspace=0.34,
            )
            for index, key in enumerate(("11-20", "11-31")):
                ax = self.figure.add_subplot(grid[0, index])
                fit = result.estimates[key]
                ax.plot(fit["opd_um"], fit["mean_metric"], ".", color=FG, ms=4, label="measured mean")
                ax.plot(fit["fit_opd"], fit["fit_metric"], "-", color=GREEN, lw=1.5, label="envelope × sinusoid")
                for candidate in fit["candidates"]:
                    ax.axvline(candidate["opd_um"], color="#bf7cff", alpha=0.22)
                if result.recommendation.get("feasible"):
                    delta = result.recommendation[f"delta_11_{key.split('-')[1]}_um"]
                    ax.axvline(2 * delta, color=ORANGE, linestyle="--", label="selected")
                ax.set(xlabel="OPD (µm)", ylabel="N / (FT1 + FT2)", title=f"Baseline {key}")
                self._style_axis(ax, grid=True)
                self._style_legend(ax.legend(fontsize=7))
            if result.recommendation.get("feasible"):
                pistons = result.recommendation["pistons_um"]
                title = self.figure.suptitle(
                    f"Chosen pistons  S11={pistons['11']:.3f}  S20={pistons['20']:.3f}  S31={pistons['31']:.3f} µm",
                    color=GREEN,
                    fontsize=11,
                    fontfamily="monospace",
                )
                title.set_y(0.94)
        self.canvas.draw_idle()

    def _update_apply_state(self) -> None:
        result = self.current_result
        enabled = bool(
            result is not None
            and not result.kind.endswith("_dark")
            and not self.running
            and result.recommendation.get("passed", True)
        )
        state = "normal" if enabled else "disabled"
        self.apply_button.configure(state=state)
        allow_loop = bool(
            enabled
            and result is not None
            and result.kind == "xy"
            and not self.workflow.xy_pitch_iteration
        )
        self.xy_loop_button.configure(state="normal" if allow_loop else "disabled")

    def apply_result(self, next_kind: str | None = None) -> None:
        if self.current_result is None or self.running:
            return
        result = self.current_result
        if not result.recommendation.get("passed", True):
            self.status.set("Safeguard failed. Values cannot be applied; redo or restore/exit.")
            return
        self._set_running(True, scan=False)
        self.status.set(f"Applying reviewed {LABELS[result.kind]} recommendation…")

        def task() -> None:
            try:
                self.engine.apply(result)
                path = finalize_result(result, self.hardware, self.config, "applied")
                self.events.put(("applied", path, next_kind, result.kind))
            except Exception as error:
                self.events.put(("error", str(error), traceback.format_exc(), result.kind))

        threading.Thread(target=task, daemon=True).start()

    def _finish_apply(self, path, next_kind: str | None, kind: str) -> None:
        self._set_running(False)
        self.review_pending = False
        current_result = getattr(self, "current_result", None)
        if current_result is not None and current_result.kind == kind:
            save_review_figure(current_result, path, self.figure)
        self.status.set(f"Values applied and scan saved: {path}")
        self.refresh_dm()
        self.refresh_mount()
        self._update_rest_dm_control_state()
        if current_result is not None and current_result.metadata.get("run_scope") == "individual":
            self.workflow.confirm(kind)
            self.workflow.invalidate_after(kind)
            self._update_step_styles()
            self.status.set(f"Individual scan applied and saved: {path}. Downstream workflow results are now orange.")
            return
        if next_kind:
            self.workflow.confirm(kind)
            self.workflow.begin_xy_pitch_iteration()
            self._update_step_styles()
            self.select_scan(next_kind)
            self.status.set("X/Y applied. Repeat pitch/yaw, then X/Y; each turns green only after its new values are applied.")
            if self.full_workflow_active:
                self.root.after(250, self.run_selected)
            return

        if self.workflow.xy_pitch_iteration and kind == self.workflow.iteration_stage:
            required_next = self.workflow.confirm(kind)
            self._update_step_styles()
            if required_next == "xy":
                self.select_scan("xy")
                self.status.set("Pitch/yaw applied. Pitch/yaw is green; X/Y remains orange until its repeat is applied.")
                if self.full_workflow_active:
                    self.root.after(250, self.run_selected)
                return
            if required_next is None and not self.workflow.xy_pitch_iteration:
                self.status.set("Pitch/yaw ↔ X/Y iteration complete; continuing the alignment workflow.")
                if self.full_workflow_active:
                    self.root.after(250, self._run_next_workflow_step)
                return

        self.workflow.confirm(kind)
        self._update_step_styles()
        if self.full_workflow_active:
            self.root.after(250, self._run_next_workflow_step)

    def redo(self) -> None:
        if self.current_result is None:
            return
        try:
            parameters = self._parameters()
            self.engine.validate_scan_parameters(self.current_result.kind, parameters)
        except ValueError as error:
            messagebox.showerror("Invalid redo parameters", str(error))
            return
        path = finalize_result(self.current_result, self.hardware, self.config, "redo_requested")
        save_review_figure(self.current_result, path, self.figure)
        self.run_selected()

    def exit_review(self) -> None:
        if self.current_result is None:
            return
        try:
            self.engine.restore_review_state(self.current_result)
        except Exception as error:
            self.status.set(f"Could not restore the pre-scan DM state: {error}")
            self._append_log(traceback.format_exc())
            messagebox.showerror("DM restore failed", str(error))
            return
        self.workflow.restore(self.current_result.kind)
        self._update_step_styles()
        path = finalize_result(self.current_result, self.hardware, self.config, "exit_without_apply")
        save_review_figure(self.current_result, path, self.figure)
        self.workflow_queue.clear()
        self.full_workflow_active = False
        self.review_pending = False
        self._clear_review()
        self.status.set(f"Pre-scan state retained; scan saved: {path}")
        self.refresh_dm()
        self.refresh_mount()
        self._update_rest_dm_control_state()

    def _clear_review(self) -> None:
        if hasattr(self, "apply_button"):
            self.apply_button.configure(state="disabled")
        if hasattr(self, "xy_loop_button"):
            self.xy_loop_button.configure(state="disabled")
            self.redo_button.configure(state="disabled")
            self.exit_button.configure(state="disabled")

    def _update_step_styles(self) -> None:
        for kind, button in self.step_buttons.items():
            state = self.workflow.state(kind)
            if state == "complete":
                suffix, style = "   ✓", "Done.Step.TButton"
            elif state == "awaiting_confirmation":
                suffix, style = "   ●", "Review.Step.TButton"
            elif state in {"iterating", "running", "attention"}:
                if state == "iterating":
                    suffix = "   ↻"
                elif state == "running":
                    suffix = "   ●"
                else:
                    suffix = "   !"
                style = "Iterating.Step.TButton"
            else:
                suffix, style = "", "Step.TButton"
            button.configure(text=LABELS[kind] + suffix, style=style)

    def _rest_dm_selection_changed(self, _event=None) -> None:
        label = self.rest_dm_choice.get()
        key = self.rest_dm_label_to_key.get(label)
        if key is None:
            return
        definition = rest_dm_maps(self.config)[key]
        editable_ptt = bool(definition.get("editable_uniform_ptt", False))
        editable_tip_tilt = editable_ptt or bool(definition.get("editable_uniform_tip_tilt", False))
        if editable_ptt:
            if key == self.engine.active_rest_dm_map and self.engine.active_rest_dm_piston_um is not None:
                piston = self.engine.active_rest_dm_piston_um
            else:
                piston = float(definition.get("default_piston_um", 0.0))
            self.rest_dm_piston.set(str(piston))
        else:
            self.rest_dm_piston.set("")
        if editable_tip_tilt:
            if key == self.engine.active_rest_dm_map and self.engine.active_rest_dm_tip_mrad is not None:
                tip = self.engine.active_rest_dm_tip_mrad
                tilt = self.engine.active_rest_dm_tilt_mrad
            else:
                tip = float(definition.get("default_tip_mrad", 0.0))
                tilt = float(definition.get("default_tilt_mrad", 0.0))
            self.rest_dm_tip.set(str(tip))
            self.rest_dm_tilt.set(str(tilt))
        else:
            self.rest_dm_tip.set("")
            self.rest_dm_tilt.set("")
        self._update_rest_dm_control_state()

    def _update_rest_dm_control_state(self) -> None:
        if not hasattr(self, "rest_dm_combo"):
            return
        available = not self.running and not self.review_pending
        self.rest_dm_combo.configure(state="readonly" if available else "disabled")
        label = self.rest_dm_choice.get()
        key = self.rest_dm_label_to_key.get(label)
        editable_ptt = bool(
            key is not None
            and rest_dm_maps(self.config)[key].get("editable_uniform_ptt", False)
            and available
        )
        editable_tip_tilt = bool(
            key is not None
            and (
                rest_dm_maps(self.config)[key].get("editable_uniform_ptt", False)
                or rest_dm_maps(self.config)[key].get("editable_uniform_tip_tilt", False)
            )
            and available
        )
        self.rest_dm_piston_entry.configure(state="normal" if editable_ptt else "disabled")
        self.rest_dm_tip_entry.configure(state="normal" if editable_tip_tilt else "disabled")
        self.rest_dm_tilt_entry.configure(state="normal" if editable_tip_tilt else "disabled")
        self.rest_dm_button.configure(state="normal" if available else "disabled")
        for button in getattr(self, "dm_preset_buttons", {}).values():
            button.configure(state="normal" if available else "disabled")

    def apply_rest_dm_map(self) -> None:
        if self.running:
            return
        if self.review_pending:
            messagebox.showerror("Review in progress", "Apply, redo, or exit the current scan review before changing REST OF DM.")
            return
        label = self.rest_dm_choice.get()
        key = self.rest_dm_label_to_key.get(label)
        if key is None:
            messagebox.showerror("Invalid DM map", f"Unknown REST OF DM selection: {label}")
            return
        definition = rest_dm_maps(self.config)[key]
        piston = tip = tilt = None
        editable_ptt = bool(definition.get("editable_uniform_ptt", False))
        editable_tip_tilt = editable_ptt or bool(definition.get("editable_uniform_tip_tilt", False))
        if editable_ptt:
            try:
                piston = float(self.rest_dm_piston.get())
            except ValueError:
                messagebox.showerror("Invalid DM map", "Piston must be a numeric value in micrometres.")
                return
        if editable_tip_tilt:
            try:
                tip = float(self.rest_dm_tip.get())
                tilt = float(self.rest_dm_tilt.get())
            except ValueError:
                messagebox.showerror("Invalid DM map", "Tip and tilt must be numeric values in milliradians.")
                return
        self._set_running(True, scan=False)
        self.status.set(f"Applying REST OF DM map: {label}…")

        def task() -> None:
            try:
                selection = self.engine.set_rest_dm_map(
                    key,
                    uniform_piston_um=piston,
                    uniform_tip_mrad=tip,
                    uniform_tilt_mrad=tilt,
                )
                self.events.put(("rest_dm_applied", selection))
            except Exception as error:
                self.events.put(("rest_dm_error", str(error), traceback.format_exc()))

        threading.Thread(target=task, daemon=True).start()

    def _handle_rest_dm_applied(self, selection: dict) -> None:
        self._set_running(False)
        self.rest_dm_active.set(f"Active: {selection['label']}")
        if selection.get("uniform_tip_mrad") is None:
            detail = ""
        else:
            piston = selection.get("uniform_piston_um")
            piston_detail = "" if piston is None else f"piston={piston:.3f} µm, "
            detail = (
                f" ({piston_detail}tip={selection['uniform_tip_mrad']:.3f} mrad, "
                f"tilt={selection['uniform_tilt_mrad']:.3f} mrad)"
            )
        self.status.set(f"REST OF DM applied to all non-injected segments: {selection['label']}{detail}")
        self.refresh_dm()
        self.refresh_mount()

    def apply_dm_preset(self, preset: str) -> None:
        if self.running or self.review_pending:
            return
        labels = {
            "zero_ptt": "PTT0 — configured zero DM PTT",
            "zero_volts": "V0 — zero DM voltages",
            "flat_volts": "F0 — configured calibrated flat voltages",
        }
        label = labels.get(preset)
        if label is None:
            messagebox.showerror("Unknown DM preset", preset)
            return
        if not messagebox.askokcancel("Apply DM preset", f"Apply {label} to the complete DM?"):
            return
        self._set_running(True, scan=False)
        self.status.set(f"Applying {label}…")

        def task() -> None:
            try:
                self.engine.apply_dm_preset(preset)
                self.events.put(("dm_preset_applied", preset, label))
            except Exception as error:
                self.events.put(("dm_preset_error", preset, str(error), traceback.format_exc()))

        threading.Thread(target=task, daemon=True).start()

    def _update_dm_inspector_control_state(self) -> None:
        if getattr(self, "dm_apply_button", None) is None:
            return
        available = (
            getattr(self, "dm_selected_segment", None) is not None
            and not getattr(self, "running", False)
            and not getattr(self, "review_pending", False)
        )
        self.dm_apply_button.configure(state="normal" if available else "disabled")

    def apply_dm_segment(self) -> None:
        if self.dm_selected_segment is None or self.running or self.review_pending:
            return
        try:
            piston = float(self.dm_inspector_values["piston_um"].get())
            tip = float(self.dm_inspector_values["tip_mrad"].get())
            tilt = float(self.dm_inspector_values["tilt_mrad"].get())
        except ValueError:
            messagebox.showerror("Invalid DM values", "Piston, tip, and tilt must be numeric values.")
            return
        segment = int(self.dm_selected_segment)
        self._set_running(True, scan=False)
        self.status.set(f"Applying segment {segment} PTT values…")

        def task() -> None:
            try:
                self.engine.set_dm_segment(segment, piston, tip, tilt)
                self.events.put(("dm_segment_applied", segment))
            except Exception as error:
                self.events.put(("dm_segment_error", str(error), traceback.format_exc()))

        threading.Thread(target=task, daemon=True).start()

    def open_advanced_mount(self) -> None:
        window = self.advanced_mount_window
        if window is not None and window.winfo_exists():
            window.lift()
            window.focus_force()
            return
        window = tk.Toplevel(self.root)
        self.advanced_mount_window = window
        window.title("Advanced mount control")
        window.geometry("560x430")
        window.minsize(520, 400)
        window.configure(bg=BG)
        window.protocol("WM_DELETE_WINDOW", self._close_advanced_mount)
        body = ttk.Frame(window, padding=16)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="ADVANCED MOUNT CONTROL", style="Header.TLabel").pack(anchor="w", pady=(0, 10))
        grid = tk.Frame(body, bg=BG)
        grid.pack(fill="x")
        self.advanced_axis_buttons = {}
        for index, axis in enumerate(("pitch", "yaw", "roll", "x", "y", "z")):
            row, column = divmod(index, 3)
            button = tk.Button(
                grid,
                textvariable=self.mount_values[axis],
                bg="#080808",
                fg=FG,
                activebackground="#202020",
                activeforeground=GREEN,
                highlightbackground=GRID,
                highlightthickness=1,
                relief="flat",
                font=(self.retro_font, 10),
                command=lambda selected=axis: self._select_advanced_axis(selected),
            )
            button.grid(row=2 * row + 1, column=column, sticky="nsew", padx=3, pady=(0, 6), ipady=7)
            tk.Label(
                grid,
                text=axis.upper(),
                bg=BG,
                fg=GREEN,
                font=(self.retro_font, 8, "bold"),
            ).grid(row=2 * row, column=column, sticky="ew")
            grid.columnconfigure(column, weight=1)
            self.advanced_axis_buttons[axis] = button

        ttk.Separator(body).pack(fill="x", pady=12)
        ttk.Label(body, textvariable=self.advanced_axis, style="Header.TLabel").pack(anchor="center")
        ttk.Label(body, textvariable=self.advanced_range).pack(anchor="center", pady=(5, 12))
        step_row = ttk.Frame(body)
        step_row.pack(anchor="center")
        ttk.Label(step_row, textvariable=self.advanced_step_label).pack(side="left", padx=(0, 10))
        self.advanced_step_entry = ttk.Entry(step_row, textvariable=self.advanced_step, width=12)
        self.advanced_step_entry.pack(side="left")
        move_row = ttk.Frame(body)
        move_row.pack(fill="x", pady=(14, 0))
        self.advanced_minus_button = ttk.Button(move_row, text="−", command=lambda: self.jog_advanced_mount(-1))
        self.advanced_minus_button.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.advanced_plus_button = ttk.Button(move_row, text="+", command=lambda: self.jog_advanced_mount(+1))
        self.advanced_plus_button.pack(side="left", fill="x", expand=True, padx=(4, 0))
        ttk.Label(
            body,
            text=f"ROLL is displayed but locked to the configured value {self.engine.fixed_roll():g}.",
            style="Muted.TLabel",
        ).pack(anchor="center", pady=(12, 0))
        self._select_advanced_axis("pitch")
        self.refresh_mount()

    def _close_advanced_mount(self) -> None:
        if self.advanced_mount_window is not None:
            self.advanced_mount_window.destroy()
        self.advanced_mount_window = None

    def _select_advanced_axis(self, axis: str) -> None:
        self.advanced_axis.set(axis)
        low, high = self.hardware.mount_limits()[axis]
        if axis in {"pitch", "yaw", "roll"}:
            self.advanced_range.set(f"CCW {low:g}    |    CW {high:g}")
            self.advanced_step_label.set("Step (pulses)")
        else:
            self.advanced_range.set(f"MIN {low:g}    |    MAX {high:g}")
            self.advanced_step_label.set("Step (micron)")
        default = self.config.get("mount_control", {}).get("advanced_default_steps", {}).get(axis, 100.0)
        self.advanced_step.set(f"{float(default):g}")
        if hasattr(self, "advanced_axis_buttons"):
            for name, button in self.advanced_axis_buttons.items():
                button.configure(fg=GREEN if name == axis else FG)
        self._update_advanced_mount_controls()

    def _update_advanced_mount_controls(self) -> None:
        if self.advanced_mount_window is None or not self.advanced_mount_window.winfo_exists():
            return
        enabled = not self.running and self.advanced_axis.get() != "roll"
        state = "normal" if enabled else "disabled"
        self.advanced_minus_button.configure(state=state)
        self.advanced_plus_button.configure(state=state)
        self.advanced_step_entry.configure(state=state)

    def jog_advanced_mount(self, direction: int) -> None:
        axis = self.advanced_axis.get()
        try:
            step = abs(float(self.advanced_step.get())) * int(direction)
            if step == 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid mount step", "Enter a positive, non-zero numeric step.")
            return
        self._set_running(True, scan=False)
        self.status.set(f"Moving mount {axis} by {step:+g}…")

        def task() -> None:
            try:
                self.engine.jog_mount(axis, step)
                self.events.put(("mount_jogged", axis, step))
            except Exception as error:
                self.events.put(("mount_jog_error", str(error), traceback.format_exc()))

        threading.Thread(target=task, daemon=True).start()

    def toggle_null_live(self) -> None:
        """Start or stop periodic APAPANE/null-statistics refreshes."""
        if self.running or self._null_live_acquiring:
            return
        self._null_live_enabled = not self._null_live_enabled
        self._update_null_live_button_state()
        if self._null_live_enabled:
            self.status.set("Null Live started; updating every configured refresh interval.")
            self._request_null_live_frame()
        else:
            self.status.set("Null Live stopped; the most recent APAPANE frame is retained.")

    def _update_null_live_button_state(self) -> None:
        if not hasattr(self, "null_live_button"):
            return
        self.null_live_button.configure(
            text="STOP" if self._null_live_enabled else "START",
            state="disabled" if self.running or self._null_live_acquiring else "normal",
        )

    def _request_null_live_frame(self) -> bool:
        if self.running or self._null_live_acquiring:
            return False
        self._null_live_initial_attempted = True
        self._null_live_acquiring = True
        self._update_null_live_button_state()
        nframes = int(self.config.get("null_live", {}).get("nframes", 25))

        def task() -> None:
            try:
                frames = self.hardware.camera_frames("apapane", nframes)
                image = process_apapane_frame(frames, self.engine.apapane_dark)
                stats = collect_null_statistics(image)
                self.events.put(("null_live", image, stats, nframes))
            except Exception as error:
                self.events.put(("null_live_error", str(error)))

        threading.Thread(target=task, daemon=True).start()
        return True

    def _schedule_null_live(self) -> None:
        if self._closing:
            return
        refresh_ms = max(1000, int(1000 * float(self.config.get("null_live", {}).get("refresh_s", 10.0))))
        initial_frame = not self._null_live_initial_attempted
        if initial_frame:
            self._null_live_initial_attempted = True
        should_acquire = initial_frame or self._null_live_enabled
        if should_acquire:
            self._request_null_live_frame()
        self.root.after(refresh_ms, self._schedule_null_live)

    def _plot_null_live_placeholder(self, text: str) -> None:
        if not hasattr(self, "null_live_figure"):
            return
        self.null_live_figure.clear()
        ax = self.null_live_figure.add_subplot(111)
        ax.set_facecolor(BG)
        ax.axis("off")
        ax.text(0.5, 0.5, text, color=MUTED, ha="center", va="center", fontfamily="monospace")
        self.null_live_canvas.draw_idle()

    def _draw_null_live(self, image: np.ndarray, statistics: dict, nframes: int) -> None:
        self.null_live_figure.clear()
        self.null_live_figure.set_facecolor(BG)
        grid = self.null_live_figure.add_gridspec(
            1,
            2,
            left=0.09,
            right=0.97,
            bottom=0.14,
            top=0.89,
            width_ratios=[1.60, 1.40],
            wspace=0.18,
        )
        image_ax = self.null_live_figure.add_subplot(grid[0, 0])
        stats_ax = self.null_live_figure.add_subplot(grid[0, 1])
        x0, x1 = self.config.get("null_live", {}).get("display_x", [200, image.shape[1]])
        x0, x1 = max(0, int(x0)), min(image.shape[1], int(x1))
        cropped = image[:, x0:x1]
        vmin, vmax = np.nanpercentile(cropped, [1, 99.5])
        image_ax.imshow(
            cropped,
            origin="lower",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            extent=[x0, x1, 0, image.shape[0]],
            aspect="auto",
        )
        overlay_groups = (
            (PHOT_BOXES, "cyan", lambda name: name),
            (NULL_BOXES, "magenta", lambda name: name),
        )
        for boxes, colour, labeler in overlay_groups:
            for name, box in boxes.items():
                y0, y1, bx0, bx1 = box
                image_ax.add_patch(Rectangle((bx0, y0), bx1 - bx0, y1 - y0, fill=False, edgecolor=colour, lw=1.5))
                image_ax.text(bx1 + 2, y0, labeler(name), color=colour, fontsize=6, va="bottom")
        for null_name, boxes in FRINGE_TRACKING_BOXES.items():
            short = null_name.split("/")[0].strip()
            for ft_name, box in boxes.items():
                y0, y1, bx0, bx1 = box
                image_ax.add_patch(Rectangle((bx0, y0), bx1 - bx0, y1 - y0, fill=False, edgecolor="lime", lw=1.1))
                image_ax.text(bx1 + 2, y0, f"{short} {ft_name}", color="lime", fontsize=6, va="bottom")
        image_ax.set(xlabel="x pixel", ylabel="y pixel")
        image_ax.set_title(f"APAPANE FRAME — AVG {nframes} FRAMES", fontsize=9, pad=6)
        image_ax.xaxis.label.set_size(8)
        image_ax.yaxis.label.set_size(8)
        self._style_axis(image_ax)

        stats_ax.set_facecolor(BG)
        stats_ax.axis("off")
        stats_ax.text(
            0.5,
            0.97,
            "LIVE FLUX SUMMARY",
            color=GREEN,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            fontfamily="monospace",
        )
        photometry_table = stats_ax.table(
            cellText=[[
                format_flux(statistics["Phot 1"]["flux"]),
                format_flux(statistics["Phot 2"]["flux"]),
                format_flux(statistics["Phot 3"]["flux"]),
            ]],
            colLabels=["PHOT 1", "PHOT 2", "PHOT 3"],
            cellLoc="center",
            colLoc="center",
            bbox=[0.01, 0.72, 0.98, 0.18],
        )
        null_rows = []
        for name in NULL_BOXES:
            values = statistics[name]
            short, baseline = [part.strip() for part in name.split("/", 1)]
            null_rows.append([
                f"{short}\n{baseline}",
                format_flux(values["null_flux"]),
                format_flux(values["ft1_flux"]),
                format_flux(values["ft2_flux"]),
                format_depth(values["null_depth"]),
            ])
        null_table = stats_ax.table(
            cellText=null_rows,
            colLabels=["BASELINE", "NULL", "FT1", "FT2", "DEPTH"],
            cellLoc="center",
            colLoc="center",
            bbox=[0.01, 0.18, 0.98, 0.48],
            colWidths=[0.22, 0.195, 0.195, 0.195, 0.195],
        )
        for table in (photometry_table, null_table):
            table.auto_set_font_size(False)
            table.set_fontsize(7.5)
            for (row, _column), cell in table.get_celld().items():
                cell.set_edgecolor(GRID)
                cell.set_linewidth(0.8)
                cell.set_facecolor("#101410" if row else "#173522")
                cell.get_text().set_color(GREEN if row == 0 else FG)
                cell.get_text().set_fontfamily("monospace")
                if row == 0:
                    cell.get_text().set_fontweight("bold")
        self.null_live_canvas.draw_idle()

    def refresh_mount(self) -> None:
        if not hasattr(self, "mount_values"):
            return
        mount = self.hardware.get_mount()
        for axis, variable in self.mount_values.items():
            variable.set(self._format_mount_value(mount[axis]))

    @staticmethod
    def _format_mount_value(value: float) -> str:
        value = float(value)
        return str(int(round(value))) if np.isclose(value, round(value), atol=1e-9) else f"{value:.3f}"

    def refresh_dm(self) -> None:
        state = self.hardware.get_dm()
        self.dm_selected_segment = getattr(self, "dm_selected_segment", None)
        self.dm_figure.clear()
        self.dm_figure.set_facecolor(BG)
        grid = self.dm_figure.add_gridspec(
            1,
            2,
            left=0.04,
            right=0.93,
            bottom=0.05,
            top=0.95,
            width_ratios=[4.35, 0.20],
            wspace=0.18,
        )
        ax = self.dm_figure.add_subplot(grid[0, 0])
        colorbar_ax = self.dm_figure.add_subplot(grid[0, 1])
        self.dm_axes = ax
        self.dm_segment_patches = {}
        ax.set_facecolor(BG)
        counts = [4, 5, 6, 7, 6, 5, 4]
        piston_values = np.asarray([state[index].piston_um for index in range(37)])
        full_scale_mrad = float(self.config.get("dm_display", {}).get("tip_tilt_full_scale_mrad", 5.0))
        cmap, norm, scale = dm_piston_mapping(
            piston_values,
            float(self.config.get("dm_display", {}).get("piston_linthresh_um", 0.02)),
        )
        segment = 0
        for column, count in enumerate(counts):
            x = 4.5 - 1.5 * column
            for row in range(count):
                y = (count - 1) * 0.88 - row * 1.76
                ptt = state[segment]
                piston_color = cmap(norm(ptt.piston_um))
                patch = RegularPolygon(
                    (x, y),
                    numVertices=6,
                    radius=1.0,
                    # Matplotlib's base hexagon is point-up in this view.
                    # Rotate every segment 30 degrees clockwise so its top and
                    # bottom edges are flat in both GUI backends.
                    orientation=-np.pi / 6,
                    facecolor="none",
                    edgecolor=RED if segment in {11, 20, 31} else "#b9c3bd",
                    linewidth=3.2 if segment in {11, 20, 31} else 1,
                    zorder=3,
                )
                ax.add_patch(patch)
                self.dm_segment_patches[segment] = patch
                gradient = dm_segment_rgba(
                    piston_color,
                    ptt.tip_mrad,
                    ptt.tilt_mrad,
                    full_scale_mrad,
                )
                image = ax.imshow(
                    gradient,
                    extent=(x - 1.0, x + 1.0, y - 1.0, y + 1.0),
                    origin="lower",
                    interpolation="bilinear",
                    zorder=1,
                )
                image.set_clip_path(patch)
                text = ax.text(
                    x,
                    y,
                    str(segment),
                    ha="center",
                    va="center",
                    color=FG,
                    fontsize=8,
                    weight="bold",
                    zorder=4,
                )
                text.set_path_effects([path_effects.withStroke(linewidth=2, foreground=BG)])
                if segment == self.dm_selected_segment:
                    ax.add_patch(
                        RegularPolygon(
                            (x, y),
                            numVertices=6,
                            radius=0.80,
                            orientation=-np.pi / 6,
                            facecolor="none",
                            edgecolor=ORANGE,
                            linewidth=1.8,
                            linestyle="--",
                            zorder=5,
                        )
                    )
                segment += 1
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6.5, 6.5)
        ax.set_aspect("equal")
        ax.axis("off")
        colorbar = self.dm_figure.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            cax=colorbar_ax,
        )
        near_zero = float(norm.linthresh)
        colorbar.set_ticks([-scale, -0.1, -near_zero, 0.0, near_zero, 0.1, scale])
        colorbar.ax.yaxis.set_major_formatter(FuncFormatter(format_piston_tick))
        colorbar.update_ticks()
        colorbar.ax.set_title("Piston (µm)", color=GREEN, fontfamily="monospace", fontsize=7, pad=6)
        colorbar.ax.yaxis.set_ticks_position("left")
        colorbar.ax.tick_params(labelleft=True, labelright=False, pad=3)
        self._style_colorbar(colorbar)
        self._update_dm_segment_inspector(state)
        self._update_dm_inspector_control_state()
        self.dm_canvas.draw_idle()

    def _update_dm_segment_inspector(self, state) -> None:
        if not hasattr(self, "dm_inspector_header"):
            return
        selected = self.dm_selected_segment
        if selected is None:
            self.dm_inspector_header.set("SEGMENT —")
            for variable in self.dm_inspector_values.values():
                variable.set("—")
            self._update_dm_inspector_control_state()
            return
        ptt = state[selected]
        self.dm_inspector_header.set(f"SEGMENT {selected}")
        self.dm_inspector_values["piston_um"].set(self._format_dm_value(ptt.piston_um))
        self.dm_inspector_values["tip_mrad"].set(self._format_dm_value(ptt.tip_mrad))
        self.dm_inspector_values["tilt_mrad"].set(self._format_dm_value(ptt.tilt_mrad))
        self._update_dm_inspector_control_state()

    @staticmethod
    def _format_dm_value(value: float) -> str:
        absolute = abs(float(value))
        if absolute >= 1000 or (0 < absolute < 0.001):
            return f"{value:.3e}"
        return f"{value:.4f}"

    def _on_dm_click(self, event) -> None:
        if event.inaxes is not self.dm_axes:
            return
        for segment, patch in self.dm_segment_patches.items():
            contains, _ = patch.contains(event)
            if contains:
                self.dm_selected_segment = segment
                self.refresh_dm()
                return

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        try:
            self.log_text.insert("end", message.rstrip() + "\n")
            self.log_text.see("end")
        finally:
            self.log_text.configure(state="disabled")

    def _close(self) -> None:
        self._closing = True
        if self.running:
            self.engine.abort()
        self.hardware.close()
        self.root.destroy()


def run_gui(hardware, config: dict) -> None:
    root = tk.Tk()
    AlignmentGUI(root, hardware, config)
    root.mainloop()
