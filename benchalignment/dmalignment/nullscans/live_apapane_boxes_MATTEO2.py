#!/usr/bin/env python3
"""
Live APAPANE box viewer with separate null-depth dashboard.

This robust version:
    - uses "FLUX" labels, not scientific notation or sigma/sum symbol
    - uses N1, N2, N3 labels for null channels
    - shows APAPANE live image in one window
    - shows photometry and null-depth dashboard in a second black-background window
    - DOES NOT show null depths on the APAPANE image
    - supports dark FITS cube with shape (N, ny, nx)
    - uses N = dark.shape[0] as the number of APAPANE frames to average
    - averages the N dark frames and subtracts averaged dark from averaged APAPANE frame
    - handles NaN safely
    - avoids fragile string reconstruction for FT1/FT2 fluxes

Null depth definition:

    null_depth = I_null / (I_FT1 + I_FT2)

Box convention:
    [y0, y1, x0, x1]
used as:
    image[y0:y1, x0:x1]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    from astropy.io import fits
except Exception:
    fits = None

try:
    from pyMilk.interfacing.shm import SHM
except Exception as exc:
    SHM = None
    SHM_IMPORT_ERROR = exc


# =============================================================================
# USER-EDITABLE PARAMETERS
# =============================================================================

# Set to the dark FITS cube path.
# Expected dark shape: (N, 256, 320), e.g. (25, 256, 320).
# If set to None, no dark subtraction is applied and DEFAULT_NFRAMES is used.
DARK_PATH = "/home/scexao/glint/darkMatteo.fits"

DEFAULT_SHM_NAME = "apapane"
DEFAULT_NFRAMES = 25
DEFAULT_REFRESH_SECONDS = 10.0

# Display crop for APAPANE image only.
# Flux extraction still uses the full frame and full-frame box coordinates.
DISPLAY_XMIN = 200
DISPLAY_XMAX = 320
DISPLAY_YMIN = None
DISPLAY_YMAX = None

# Convention: [y0, y1, x0, x1]
PHOT_BOXES = {
    "Phot 1": [241, 245, 229, 267],
    "Phot 2": [222, 226, 229, 267],
    "Phot 3": [203, 207, 229, 267],
}

# Null boxes. The keys are the dashboard labels.
NULL_BOXES = {
    "N1 / 11-31": [166, 170, 229, 267],
    "N2 / 11-20": [110, 114, 229, 267],
    "N3 / 20-31": [35, 39, 229, 267],
}

# PLACEHOLDERS: replace with correct fringe-tracking boxes.
# The top-level keys MUST match NULL_BOXES.
FRINGE_TRACKING_BOXES = {
    "N1 / 11-31": {
        "FT1": [185, 189, 229, 267],
        "FT2": [147, 151, 229, 267],
    },
    "N2 / 11-20": {
        "FT1": [129, 133, 229, 267],
        "FT2": [91, 95, 229, 267],
    },
    "N3 / 20-31": {
        "FT1": [54, 58, 229, 267],
        "FT2": [17, 21, 229, 267]
    },
}


# =============================================================================
# SAFE FORMATTERS
# =============================================================================

def fmt_flux(value) -> str:
    try:
        value = float(value)
    except Exception:
        return "nan"
    if not np.isfinite(value):
        return "nan"
    return f"{int(round(value)):,}"


def fmt_depth(value) -> str:
    try:
        value = float(value)
    except Exception:
        return "nan"
    if not np.isfinite(value):
        return "nan"
    return f"{value:.6g}"


# =============================================================================
# FRAME AND FLUX HELPERS
# =============================================================================

def connect_camera(shm_name: str):
    if SHM is None:
        raise RuntimeError(
            f"Could not import pyMilk SHM. Are you running on the instrument machine? "
            f"Original import error: {SHM_IMPORT_ERROR!r}"
        )
    return SHM(shm_name)


def get_averaged_frame(cam, nframes: int, delay: float = 0.0) -> np.ndarray:
    if nframes <= 0:
        raise ValueError("nframes must be > 0")

    frames = []
    for i in range(nframes):
        frame = np.array(cam.get_data(), dtype=float)
        if frame.ndim != 2:
            raise RuntimeError(f"Expected 2D APAPANE frame, got shape {frame.shape}")
        frames.append(frame)

        if delay > 0 and i < nframes - 1:
            time.sleep(delay)

    return np.nanmean(frames, axis=0)


def load_dark(path: str | None) -> tuple[np.ndarray | None, int | None]:
    """
    Load dark FITS.

    If dark is a cube with shape (N, ny, nx):
        return averaged dark and N.

    If dark is 2D:
        return dark and None.
    """
    if path is None:
        return None, None

    if fits is None:
        raise RuntimeError("astropy is required to read FITS dark frames.")

    dark_path = Path(path)
    if not dark_path.exists():
        raise FileNotFoundError(f"Dark file does not exist: {dark_path}")

    dark = np.array(fits.getdata(dark_path), dtype=float)

    if dark.ndim == 3:
        n_dark = int(dark.shape[0])
        dark_mean = np.nanmean(dark, axis=0)
        print(f"Loaded dark cube: {dark_path}")
        print(f"Dark cube shape: {dark.shape}; using NFRAMES={n_dark}")
        return dark_mean, n_dark

    if dark.ndim == 2:
        print(f"Loaded 2D dark frame: {dark_path}")
        print("WARNING: dark is 2D, so NFRAMES is not inferred from it.")
        return dark, None

    raise ValueError(f"Unsupported dark shape {dark.shape}; expected 2D or 3D FITS.")


def subtract_dark_if_needed(image: np.ndarray, dark: np.ndarray | None) -> np.ndarray:
    if dark is None:
        return image

    if dark.shape != image.shape:
        raise ValueError(f"Dark shape {dark.shape} does not match image shape {image.shape}")

    return image - dark


def box_flux(image: np.ndarray, box: list[int]) -> tuple[float, float]:
    y0, y1, x0, x1 = [int(v) for v in box]

    if y0 < 0 or x0 < 0 or y1 > image.shape[0] or x1 > image.shape[1]:
        raise ValueError(f"Box {box} outside image with shape {image.shape}")

    roi = image[y0:y1, x0:x1]
    return float(np.nansum(roi)), float(np.nanmean(roi))


def collect_fluxes(image: np.ndarray) -> tuple[dict, dict]:
    """
    Returns:
        fluxes:
            flat dict used for display labels.
        ft_components:
            nested dict:
                ft_components[null_name]["FT1"]
                ft_components[null_name]["FT2"]

    The nested dict avoids fragile string reconstruction.
    """
    fluxes = {}
    ft_components = {}

    for name, box in PHOT_BOXES.items():
        total, mean = box_flux(image, box)
        fluxes[name] = {"sum": total, "mean": mean, "kind": "phot"}

    for name, box in NULL_BOXES.items():
        total, mean = box_flux(image, box)
        fluxes[name] = {"sum": total, "mean": mean, "kind": "null"}

    for null_name, ft_boxes in FRINGE_TRACKING_BOXES.items():
        ft_components[null_name] = {}

        for ft_label, box in ft_boxes.items():
            display_name = f"{null_name} {ft_label}"
            total, mean = box_flux(image, box)

            ft_components[null_name][ft_label] = total

            fluxes[display_name] = {
                "sum": total,
                "mean": mean,
                "kind": "fringe",
                "null_name": null_name,
                "ft_label": ft_label,
            }

    return fluxes, ft_components


def compute_null_depths(fluxes: dict, ft_components: dict) -> dict[str, float]:
    depths = {}

    for null_name in NULL_BOXES:
        i_null = float(fluxes.get(null_name, {}).get("sum", np.nan))

        ft1 = float(ft_components.get(null_name, {}).get("FT1", np.nan))
        ft2 = float(ft_components.get(null_name, {}).get("FT2", np.nan))

        denom = ft1 + ft2

        if not np.isfinite(i_null) or not np.isfinite(denom) or denom == 0:
            depths[null_name] = np.nan
        else:
            depths[null_name] = i_null / denom

    return depths


# =============================================================================
# MAIN APAPANE FIGURE
# =============================================================================

def add_box(ax, box, label, edgecolor, flux_value=None, linewidth=1.8):
    y0, y1, x0, x1 = [int(v) for v in box]

    rect = Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        fill=False,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(rect)

    if flux_value is None:
        text = label
    else:
        text = f"{label}\nFLUX={fmt_flux(flux_value)}"

    text_x = x1 + 5
    text_y = y0

    txt = ax.text(
        text_x,
        text_y,
        text,
        color=edgecolor,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.5),
    )

    return rect, txt


def create_static_overlays(ax):
    artists = {}

    for label, box in PHOT_BOXES.items():
        rect, txt = add_box(ax, box, label, edgecolor="cyan", linewidth=1.8, flux_value=0)
        artists[label] = {"rect": rect, "text": txt}

    for label, box in NULL_BOXES.items():
        rect, txt = add_box(ax, box, label, edgecolor="magenta", linewidth=1.8, flux_value=0)
        artists[label] = {"rect": rect, "text": txt}

    for null_name, ft_boxes in FRINGE_TRACKING_BOXES.items():
        for ft_label, box in ft_boxes.items():
            display_name = f"{null_name} {ft_label}"
            short_label = f"{null_name.split('/')[0].strip()} {ft_label}"
            rect, txt = add_box(ax, box, short_label, edgecolor="lime", linewidth=1.2, flux_value=0)
            artists[display_name] = {"rect": rect, "text": txt}

    return artists


def update_flux_texts(artists, fluxes):
    for label, values in fluxes.items():
        if label not in artists:
            continue

        if values["kind"] == "fringe":
            null_label = values.get("null_name", label).split("/")[0].strip()
            ft_label = values.get("ft_label", "")
            display_label = f"{null_label} {ft_label}"
        else:
            display_label = label

        artists[label]["text"].set_text(
            f"{display_label}\nFLUX={fmt_flux(values['sum'])}"
        )


# =============================================================================
# DASHBOARD FIGURE
# =============================================================================

def create_dashboard():
    """
    Compact dashboard with:
        - one photometry box
        - one null-depth box
    """
    fig = plt.figure(figsize=(4.2, 6.5), facecolor="black")
    ax = fig.add_subplot(111)

    ax.set_facecolor("black")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    title = ax.text(
        0.5, 0.97,
        "GLINT LIVE",
        color="white",
        fontsize=16,
        fontweight="bold",
        ha="center",
        va="top",
    )

    # -----------------------------------------------------------------
    # PHOTOMETRY BOX
    # -----------------------------------------------------------------
    phot_box = Rectangle(
        (0.08, 0.58), 0.84, 0.28,
        facecolor="#2b1a00",
        edgecolor="orange",
        linewidth=3,
    )
    ax.add_patch(phot_box)

    phot_text = ax.text(
        0.12, 0.82,
        "",
        color="cyan",
        fontsize=20,
        fontweight="bold",
        ha="left",
        va="top",
        family="monospace",
    )

    # -----------------------------------------------------------------
    # NULL DEPTH BOX
    # -----------------------------------------------------------------
    null_box = Rectangle(
        (0.08, 0.18), 0.84, 0.28,
        facecolor="#16001f",
        edgecolor="purple",
        linewidth=3,
    )
    ax.add_patch(null_box)

    null_text = ax.text(
        0.12, 0.42,
        "",
        color="violet",
        fontsize=20,
        fontweight="bold",
        ha="left",
        va="top",
        family="monospace",
    )

    fig.tight_layout()

    return fig, ax, phot_text, null_text, title

def update_dashboard(
    phot_text,
    null_text,
    title,
    fluxes,
    null_depths,
    nframes,
):
    title.set_text(
        f"GLINT LIVE | {time.strftime('%H:%M:%S')}"
    )

    # -----------------------------------------------------------------
    # PHOTOMETRY
    # -----------------------------------------------------------------
    phot_text.set_color("yellow")

    phot_text.set_text(
        "PHOTOMETRY\n"
        f"P1  {fmt_flux(fluxes.get('Phot 1', {}).get('sum', np.nan))}\n"
        f"P2  {fmt_flux(fluxes.get('Phot 2', {}).get('sum', np.nan))}\n"
        f"P3  {fmt_flux(fluxes.get('Phot 3', {}).get('sum', np.nan))}"
    )

    # -----------------------------------------------------------------
    # NULL DEPTHS
    # -----------------------------------------------------------------
    null_text.set_color("#ff00ff")   # bright magenta / purple

    null_text.set_text(
        "NULL DEPTHS\n"
        f"N1  {fmt_depth(null_depths.get('N1 / 11-31', np.nan))}\n"
        f"N2  {fmt_depth(null_depths.get('N2 / 11-20', np.nan))}\n"
        f"N3  {fmt_depth(null_depths.get('N3 / 20-31', np.nan))}"
    )


def print_fluxes(fluxes: dict, ft_components: dict, null_depths: dict[str, float]):
    print("\nCurrent box fluxes")
    print("------------------")
    for label, values in fluxes.items():
        print(f"{label:26s} FLUX={fmt_flux(values['sum'])}  mean={fmt_depth(values['mean'])}")

    print("\nNull depths")
    print("-----------")
    for label, nd in null_depths.items():
        ft1 = ft_components.get(label, {}).get("FT1", np.nan)
        ft2 = ft_components.get(label, {}).get("FT2", np.nan)
        print(
            f"{label:26s} ND={fmt_depth(nd)} "
            f"I_null={fmt_flux(fluxes.get(label, {}).get('sum', np.nan))} "
            f"FT1+FT2={fmt_flux(ft1 + ft2)}"
        )


# =============================================================================
# LIVE LOOP
# =============================================================================

def acquire_processed_image(cam, dark, nframes, frame_delay):
    image = get_averaged_frame(cam, nframes, delay=frame_delay)
    image = subtract_dark_if_needed(image, dark)
    return image


def live_display(
    cam,
    dark,
    nframes: int,
    refresh_seconds: float,
    frame_delay: float,
    percentile_low: float,
    percentile_high: float,
    cmap: str,
):
    plt.ion()

    image = acquire_processed_image(cam, dark, nframes, frame_delay)
    fluxes, ft_components = collect_fluxes(image)
    null_depths = compute_null_depths(fluxes, ft_components)

    vmin = np.nanpercentile(image, percentile_low)
    vmax = np.nanpercentile(image, percentile_high)

    # Crop is display-only. Fluxes are still computed from the full image.
    y_min = 0 if DISPLAY_YMIN is None else int(DISPLAY_YMIN)
    y_max = image.shape[0] if DISPLAY_YMAX is None else int(DISPLAY_YMAX)
    x_min = 0 if DISPLAY_XMIN is None else int(DISPLAY_XMIN)
    x_max = image.shape[1] if DISPLAY_XMAX is None else int(DISPLAY_XMAX)

    image_display = image[y_min:y_max, x_min:x_max]

    fig_img, ax = plt.subplots(figsize=(5.0, 8.0))
    im = ax.imshow(
        image_display,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        extent=[x_min, x_max, y_min, y_max],
    )

    artists = create_static_overlays(ax)
    update_flux_texts(artists, fluxes)

    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    ax.set_title(f"Live APAPANE boxes | {time.strftime('%H:%M:%S')}")

    cbar = fig_img.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ADU")

    fig_img.tight_layout()
    fig_img.canvas.draw()
    fig_img.canvas.flush_events()

    fig_dash, ax_dash, phot_text, null_text, dash_title = create_dashboard()

    # Best-effort side-by-side placement. This works for many Qt/Tk backends;
    # it is safely ignored if the backend does not expose a window manager.
    try:
        fig_img.canvas.manager.window.move(50, 50)
        fig_dash.canvas.manager.window.move(900, 50)
    except Exception:
        pass

    update_dashboard(phot_text, null_text, dash_title, fluxes, null_depths, nframes)
    fig_dash.canvas.draw()
    fig_dash.canvas.flush_events()

    print_fluxes(fluxes, ft_components, null_depths)

    while plt.fignum_exists(fig_img.number) and plt.fignum_exists(fig_dash.number):
        time.sleep(refresh_seconds)

        image = acquire_processed_image(cam, dark, nframes, frame_delay)
        fluxes, ft_components = collect_fluxes(image)
        null_depths = compute_null_depths(fluxes, ft_components)

        vmin = np.nanpercentile(image, percentile_low)
        vmax = np.nanpercentile(image, percentile_high)

        image_display = image[y_min:y_max, x_min:x_max]
        im.set_data(image_display)
        im.set_clim(vmin, vmax)
        update_flux_texts(artists, fluxes)

        ax.set_title(
            f"Live APAPANE boxes | {time.strftime('%H:%M:%S')}"
        )

        update_dashboard(phot_text, null_text, dash_title, fluxes, null_depths, nframes)

        fig_img.canvas.draw_idle()
        fig_img.canvas.flush_events()

        fig_dash.canvas.draw_idle()
        fig_dash.canvas.flush_events()

        print_fluxes(fluxes, ft_components, null_depths)

    print("A figure was closed. Exiting live viewer.")


def main():
    parser = argparse.ArgumentParser(
        description="Live APAPANE frame display with separate null-depth dashboard."
    )
    parser.add_argument("--shm", default=DEFAULT_SHM_NAME)
    parser.add_argument(
        "--nframes",
        type=int,
        default=DEFAULT_NFRAMES,
        help="Fallback number of APAPANE frames if DARK_PATH is None or dark is 2D.",
    )
    parser.add_argument("--refresh", type=float, default=DEFAULT_REFRESH_SECONDS)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument(
        "--dark",
        default=DARK_PATH,
        help="Optional dark FITS path. Defaults to DARK_PATH at top of file.",
    )
    parser.add_argument("--pmin", type=float, default=1.0)
    parser.add_argument("--pmax", type=float, default=99.5)
    parser.add_argument("--cmap", default="gray")
    args = parser.parse_args()

    cam = connect_camera(args.shm)
    dark, n_dark = load_dark(args.dark)

    if dark is not None:
        print(f"Using averaged dark from: {args.dark}")

    nframes = n_dark if n_dark is not None else args.nframes

    live_display(
        cam=cam,
        dark=dark,
        nframes=nframes,
        refresh_seconds=args.refresh,
        frame_delay=args.delay,
        percentile_low=args.pmin,
        percentile_high=args.pmax,
        cmap=args.cmap,
    )


if __name__ == "__main__":
    main()
