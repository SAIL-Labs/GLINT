
'''
python make_segment_mosaic.py \
    --root-dir /home/scexao/glint/glintdata/benchalignment/dmalignment/nullscans \
    --year 2026 \
    --date 05-01 \
    --scan 16 \
    --mode average \
    --global-scale

'''

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from PIL import Image
import glint_paths


# ----------------------------
# Layout (top -> bottom rows)
# ----------------------------
SEGMENT_ROWS = [
    [15],
    [28, 22, 16, 9, 4],
    [33, 29, 23, 17, 10, 5, 0],
    [34, 30, 24, 18, 11, 6, 1],
    [35, 31, 25, 19, 12, 7, 2],
    [36, 32, 26, 20, 13, 8, 3],
    [27, 21, 14],
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create hex-grid mosaics directly from a nullscan FITS table."
    )

    p.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help="Root nullscan directory, e.g. /home/scexao/glint/glintdata/benchalignment/dmalignment/nullscans",
    )
    p.add_argument("--year", type=str, required=True, help="Year folder, e.g. 2026")
    p.add_argument("--date", type=str, required=True, help="Date folder, e.g. 04-21")
    p.add_argument("--scan", type=int, required=True, help="Scan number, e.g. 1")

    p.add_argument(
        "--fits-name",
        type=str,
        default=None,
        help="Optional FITS filename. Default: nullscan_table_<scan>.fits",
    )

    p.add_argument(
        "--background",
        type=str,
        default="white",
        choices=["black", "transparent", "white"],
        help="Mosaic background.",
    )

    p.add_argument(
        "--x-step-factor",
        type=float,
        default=0.95,
        help="Horizontal step as a fraction of tile width.",
    )
    p.add_argument(
        "--y-step-factor",
        type=float,
        default=0.9,
        help="Vertical step as a fraction of tile height.",
    )

    p.add_argument(
        "--tile-width",
        type=float,
        default=4.0,
        help="Width of each segment plot in inches.",
    )
    p.add_argument(
        "--tile-height",
        type=float,
        default=3.0,
        help="Height of each segment plot in inches.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for each segment plot.",
    )

    p.add_argument(
        "--normalise",
        action="store_true",
        help="Plot normalised flux instead of raw unnormalised flux.",
    )
    
    p.add_argument(
        "--mode",
        type=str,
        choices=["average", "index"],
        default="average",
        help="Plot either the average over AVG_INDEX or one particular AVG_INDEX.",
    )

    p.add_argument(
        "--avg-index",
        type=int,
        default=None,
        help="AVG_INDEX to plot when --mode index is used.",
    )

    p.add_argument(
        "--global-scale",
        action="store_true",
        help="Use global min/max flux across all segments for consistent y-axis scaling.",
    )

    return p.parse_args()


def bg_rgba(background: str) -> Tuple[int, int, int, int]:
    if background == "transparent":
        return (0, 0, 0, 0)
    if background == "white":
        return (255, 255, 255, 255)
    return (0, 0, 0, 255)

def compute_global_flux_range(data, phot, mode, avg_index):
    """
    Compute global min/max flux across all segments for one phot channel.
    """
    global_min = np.inf
    global_max = -np.inf

    for seg in range(37):
        opd, flux = get_plot_data(data, seg, phot, mode, avg_index)

        if flux is None or len(flux) == 0:
            continue

        global_min = min(global_min, np.nanmin(flux))
        global_max = max(global_max, np.nanmax(flux))

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        return None, None

    return global_min, global_max

def get_plot_data(data, segment: int, phot: int, mode: str, avg_index: int | None):
    """
    Return opd and flux arrays for one segment/phot pair.

    Modes
    -----
    average : average flux across all AVG_INDEX values at each OPD
    index   : use only rows matching the requested AVG_INDEX
    """
    mask = (data["SEGMENT"] == segment) & (data["PHOT_CHAN"] == phot)

    if np.sum(mask) == 0:
        return None, None

    sub = data[mask]

    opd = np.array(sub["OPD"], dtype=float)
    flux = np.array(sub["FLUX"], dtype=float)
    avg = np.array(sub["AVG_INDEX"], dtype=int)

    if mode == "index":
        if avg_index is None:
            raise ValueError("avg_index must be provided when mode='index'")

        idx_mask = avg == avg_index
        if np.sum(idx_mask) == 0:
            return None, None

        opd = opd[idx_mask]
        flux = flux[idx_mask]

        order = np.argsort(opd)
        return opd[order], flux[order]

    elif mode == "average":
        unique_opd = np.unique(opd)
        mean_flux = np.zeros_like(unique_opd, dtype=float)

        for i, opd_val in enumerate(unique_opd):
            mean_flux[i] = np.mean(flux[opd == opd_val])

        order = np.argsort(unique_opd)
        return unique_opd[order], mean_flux[order]

    else:
        raise ValueError(f"Unknown mode: {mode}")

def compute_positions(tile_w: int, tile_h: int, x_step_factor: float, y_step_factor: float):
    x_step = int(round(tile_w * x_step_factor))
    y_step = int(round(tile_h * y_step_factor))
    half_step = x_step // 2

    max_cols = max(len(r) for r in SEGMENT_ROWS)
    positions = {}

    for row_idx, row in enumerate(SEGMENT_ROWS):
        row_len = len(row)
        x0 = int(round((max_cols - row_len) * x_step / 2))
        y0 = row_idx * y_step

        if row_len % 2 == 0:
            x0 += half_step

        for col_idx, seg_id in enumerate(row):
            x = x0 + col_idx * x_step
            y = y0
            positions[seg_id] = (x, y)

    min_x = min(x for x, y in positions.values())
    min_y = min(y for x, y in positions.values())
    if min_x < 0 or min_y < 0:
        for seg_id, (x, y) in list(positions.items()):
            positions[seg_id] = (x - min_x, y - min_y)

    return positions


def load_scan_table(fits_path: Path):
    with fits.open(fits_path) as hdul:
        data = hdul["SCAN_DATA"].data
    return data


def make_segment_plot(
    data,
    segment: int,
    phot: int,
    normalise: bool,
    tile_width: float,
    tile_height: float,
    dpi: int,
    mode: str,
    avg_index: int | None,
    y_limits: tuple | None = None,
) -> Image.Image:
    opd, flux = get_plot_data(
        data=data,
        segment=segment,
        phot=phot,
        mode=mode,
        avg_index=avg_index,
    )

    fig, ax = plt.subplots(figsize=(tile_width, tile_height), dpi=dpi)

    if opd is None or flux is None or len(opd) == 0:
        ax.text(0.5, 0.5, f"Seg {segment}\nNo data", ha="center", va="center")
        ax.set_axis_off()
    else:
        if normalise and np.nanmax(flux) != 0:
            flux = flux / np.nanmax(flux)

        ax.plot(opd, flux, lw=1.5)
        if y_limits is not None:
            ax.set_ylim(y_limits)
        ax.set_title(f"Seg {segment}", fontsize=10)
        ax.set_xlabel("OPD (um)")
        ax.set_ylabel("Normalised flux" if normalise else "Flux")
        ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def make_mosaic_for_phot(
    data,
    out_path: Path,
    phot: int,
    background: str,
    x_step_factor: float,
    y_step_factor: float,
    tile_width: float,
    tile_height: float,
    dpi: int,
    normalise: bool,
    mode: str,
    avg_index: int | None,
    global_scale: bool,
) -> None:
    sample_tile = make_segment_plot(
        data=data,
        segment=0,
        phot=phot,
        normalise=normalise,
        tile_width=tile_width,
        tile_height=tile_height,
        dpi=dpi,
        mode=mode,
        avg_index=avg_index,
    )

    if global_scale:
        y_limits = compute_global_flux_range(data, phot, mode, avg_index)
    else:
        y_limits = None

    tile_w, tile_h = sample_tile.size

    positions = compute_positions(tile_w, tile_h, x_step_factor, y_step_factor)

    xs = [x for (x, y) in positions.values()]
    ys = [y for (x, y) in positions.values()]
    max_x = max(xs) + tile_w
    max_y = max(ys) + tile_h

    canvas = Image.new("RGBA", (max_x, max_y), bg_rgba(background))

    for seg_id in range(37):
        tile = make_segment_plot(
            data=data,
            segment=seg_id,
            phot=phot,
            normalise=normalise,
            tile_width=tile_width,
            tile_height=tile_height,
            dpi=dpi,
            mode=mode,
            avg_index=avg_index,
            y_limits=y_limits,
        )
        x, y = positions[seg_id]
        canvas.alpha_composite(tile, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if background != "transparent":
        rgb = Image.new("RGB", canvas.size, bg_rgba(background)[:3])
        rgb.paste(canvas, mask=canvas.split()[-1])
        rgb.save(out_path)
    else:
        canvas.save(out_path)

    print(f"[OK] Wrote {out_path}")


def main() -> None:
    args = parse_args()

    if args.mode == "index" and args.avg_index is None:
        raise ValueError("You must provide --avg-index when using --mode index")

    root_dir = args.root_dir.expanduser().resolve()
    scan_dir = root_dir / args.year / args.date / f"scan{args.scan}"

    fits_name = args.fits_name or f"nullscan_table_{args.scan}.fits"
    fits_path = scan_dir / fits_name

    if not fits_path.exists():
        raise FileNotFoundError(f"Could not find FITS file: {fits_path}")

    data = load_scan_table(fits_path)

    for phot in (1, 2, 3):
        if args.mode == "average":
            out_name = f"mosaic_phot{phot}_scan{args.scan}_avg.png"
        else:
            out_name = f"mosaic_phot{phot}_scan{args.scan}_avgidx{args.avg_index}.png"

        out_path = scan_dir / out_name

        make_mosaic_for_phot(
            data=data,
            out_path=out_path,
            phot=phot,
            background=args.background,
            x_step_factor=args.x_step_factor,
            y_step_factor=args.y_step_factor,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
            dpi=args.dpi,
            normalise=args.normalise,
            mode=args.mode,
            avg_index=args.avg_index,
            global_scale=args.global_scale,
        )


if __name__ == "__main__":
    main()