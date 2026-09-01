#!/usr/bin/env python3
"""
Read and analyse saved GLINT x/y mount raster scans.

This is the scan-computer version of the notebook analysis. It reads
year/date/iteration from scanparameters.json's "state" section, opens the
corresponding xyscan FITS files, sums each photometry cube into a 2D x/y
heatmap, estimates the centroid for each spectral channel, calculates a
common centroid, saves a PNG and JSON, and can optionally write the fitted
x/y position back into scanparameters.json.

Expected files:
    /home/scexao/glint/glintdata/benchalignment/chipmountalignment/xyscans/scanoutput/{year}/{date}/scan{iteration}/
        xyscan_spectra1_{iteration}.fits
        xyscan_spectra2_{iteration}.fits
        xyscan_spectra3_{iteration}.fits

Each FITS file should contain:
    PHOTOMETRY : image cube, shape (n_x*n_y, boxheight, boxwidth)
    MOUNT_POS  : array [x_positions, y_positions]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import glint_paths


DEFAULT_PARAM_FILE = str(glint_paths.CODE_ROOT / 'benchalignment' / 'chipmountalignment' / 'xyscans' / 'scanparameters.json')
DEFAULT_BASE_ROOT = str(glint_paths.data_dir('benchalignment', 'chipmountalignment', 'xyscans', 'scanoutput'))
DEFAULT_N_SPECTRA = 3


def load_params(param_file: str | Path) -> dict:
    """Load scanparameters.json (both "config" and "state" sections)."""
    with Path(param_file).open("r") as f:
        return json.load(f)


def save_json(data: dict, filename: str | Path) -> None:
    """Save dictionary as pretty JSON."""
    with Path(filename).open("w") as f:
        json.dump(data, f, indent=2)


def get_scan_dir(state: dict, base_root: str | Path, iteration: int | None = None) -> Path:
    """Build the scan directory from year/date/iteration in scanparameters.json's "state" section."""
    scan_iteration = int(state["iteration"] if iteration is None else iteration)
    return Path(base_root) / str(state["year"]) / str(state["date"]) / f"scan{scan_iteration}"


def read_xy_scan(
    scan_dir: str | Path,
    spectrum_index: int,
    iteration: int,
    filename_prefix: str = "xyscan",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read one spectral channel from an x/y scan FITS file.

    Returns
    -------
    photometry : np.ndarray
        Photometry cube, shape (n_x*n_y, boxheight, boxwidth).
    x_positions : np.ndarray
        Scanned x positions.
    y_positions : np.ndarray
        Scanned y positions.
    """
    scan_dir = Path(scan_dir)
    filename = scan_dir / f"{filename_prefix}_spectra{spectrum_index}_{iteration}.fits"

    if not filename.exists():
        raise FileNotFoundError(f"Could not find scan file: {filename}")

    with fits.open(filename) as hdul:
        photometry = np.asarray(hdul["PHOTOMETRY"].data, dtype=float)
        mount_positions = np.asarray(hdul["MOUNT_POS"].data, dtype=float)

    if mount_positions.shape[0] != 2:
        raise ValueError(
            f"Expected MOUNT_POS shape (2, nsteps), got {mount_positions.shape} in {filename}"
        )

    x_positions, y_positions = mount_positions
    return photometry, x_positions, y_positions


def make_heatmap(
    photometry: np.ndarray,
    n_x: int,
    n_y: int,
    wavelength_top: int | None = None,
    wavelength_bottom: int | None = None,
) -> np.ndarray:
    """
    Convert a photometry cube into a 2D x/y heatmap.

    The scan script loops over x first, then y, so the reshaped output is:
        heatmap[x_index, y_index]
    """
    expected_frames = n_x * n_y
    if photometry.shape[0] != expected_frames:
        raise ValueError(
            f"Photometry cube has {photometry.shape[0]} frames, but expected "
            f"{expected_frames} from n_x={n_x}, n_y={n_y}."
        )

    wl_top = 0 if wavelength_top is None else int(wavelength_top)
    wl_bottom = photometry.shape[1] if wavelength_bottom is None else int(wavelength_bottom)

    heatmap = np.zeros((n_x, n_y), dtype=float)
    frame = 0
    for i in range(n_x):
        for j in range(n_y):
            heatmap[i, j] = np.nansum(photometry[frame][wl_top:wl_bottom, :])
            frame += 1

    return heatmap


def centre_of_mass(image: np.ndarray) -> Tuple[float, float]:
    """
    Calculate centroid in heatmap pixel coordinates.

    Returns (y_pixel, x_pixel), matching the notebook convention where the
    plotted coordinate pair is written as (y, x).
    """
    img = np.asarray(image, dtype=float)
    img = img - np.nanmin(img)  # reduce bias from negative dark-subtraction offsets
    total = np.nansum(img)

    if total <= 0 or not np.isfinite(total):
        return np.nan, np.nan

    x_pix_grid = np.arange(img.shape[0])  # row index -> x axis
    y_pix_grid = np.arange(img.shape[1])  # column index -> y axis

    x_pix = np.nansum(x_pix_grid * np.nansum(img, axis=1)) / total
    y_pix = np.nansum(y_pix_grid * np.nansum(img, axis=0)) / total

    return float(y_pix), float(x_pix)


def pixel_to_xy(
    y_pix: float,
    x_pix: float,
    x_positions: np.ndarray,
    y_positions: np.ndarray,
) -> Tuple[float, float]:
    """
    Convert heatmap pixel coordinates to mount coordinates.

    Returns (x, y).
    """
    x = np.interp(x_pix, np.arange(len(x_positions)), x_positions)
    y = np.interp(y_pix, np.arange(len(y_positions)), y_positions)
    return float(x), float(y)


def find_centroid(heatmap: np.ndarray, x_positions: np.ndarray, y_positions: np.ndarray) -> dict:
    """Find centroid of one x/y heatmap."""
    y_pix, x_pix = centre_of_mass(heatmap)

    if not (np.isfinite(y_pix) and np.isfinite(x_pix)):
        max_x_i, max_y_j = np.unravel_index(np.nanargmax(heatmap), heatmap.shape)
        x_pix = float(max_x_i)
        y_pix = float(max_y_j)
        method_used = "max_fallback"
    else:
        method_used = "centroid"

    x, y = pixel_to_xy(y_pix, x_pix, x_positions, y_positions)

    return {
        "method_used": method_used,
        "x": x,
        "y": y,
        "pixel_x_index": float(x_pix),
        "pixel_y_index": float(y_pix),
        "peak_value": float(np.nanmax(heatmap)),
    }


def normalise_heatmaps(heatmaps: Dict[int, np.ndarray], mode: str) -> Dict[int, np.ndarray]:
    """
    Normalise heatmaps.

    mode options:
        none        : no normalisation
        individual  : each spectrum divided by its own max
        global      : all spectra divided by the global max
    """
    mode = mode.lower()

    if mode == "none":
        return {idx: img.copy() for idx, img in heatmaps.items()}

    if mode == "individual":
        output = {}
        for idx, img in heatmaps.items():
            max_value = np.nanmax(img)
            output[idx] = img / max_value if max_value != 0 else img.copy()
        return output

    if mode == "global":
        max_value = max(np.nanmax(img) for img in heatmaps.values())
        return {idx: (img / max_value if max_value != 0 else img.copy()) for idx, img in heatmaps.items()}

    raise ValueError("normalise must be one of: none, individual, global")


def common_centroid(centroids: Dict[int, dict]) -> dict:
    """Average the individual spectral-channel centroids."""
    xs = [centroids[idx]["x"] for idx in centroids]
    ys = [centroids[idx]["y"] for idx in centroids]
    return {
        "x": float(np.nanmean(xs)),
        "y": float(np.nanmean(ys)),
    }


def plot_heatmaps(
    heatmaps: Dict[int, np.ndarray],
    centroids: Dict[int, dict],
    common: dict,
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    output_png: str | Path,
    normalised: bool,
) -> None:
    """Save side-by-side x/y heatmaps with individual and common centroids."""
    spectra = list(heatmaps.keys())

    x_step = abs(x_positions[1] - x_positions[0]) if len(x_positions) > 1 else 1
    y_step = abs(y_positions[1] - y_positions[0]) if len(y_positions) > 1 else 1

    extent = [
        y_positions[0] - y_step / 2,
        y_positions[-1] + y_step / 2,
        x_positions[0] - x_step / 2,
        x_positions[-1] + x_step / 2,
    ]

    fig, axs = plt.subplots(1, len(spectra), figsize=(7 * len(spectra), 5), squeeze=False)
    axs = axs[0]

    for ax, idx in zip(axs, spectra):
        im = ax.imshow(heatmaps[idx], origin="lower", extent=extent, aspect="auto")
        ax.plot(centroids[idx]["y"], centroids[idx]["x"], "rx", markersize=8, label="individual centroid")
        ax.plot(common["y"], common["x"], "go", markersize=6, label="common centroid")
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        ax.set_title(f"Spectra {idx}: x={centroids[idx]['x']:.3f}, y={centroids[idx]['y']:.3f}")
        ax.legend(loc="best")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Normalised intensity" if normalised else "Summed intensity")

    fig.suptitle(f"Common centroid: x={common['x']:.3f}, y={common['y']:.3f}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.show()


def parse_spectra(spectrum_string: str) -> Tuple[int, ...]:
    """Parse comma-separated spectrum list, e.g. '1,2,3'."""
    return tuple(int(s.strip()) for s in spectrum_string.split(",") if s.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Find x/y centroids from saved GLINT x/y scans.")
    parser.add_argument("--param-file", default=DEFAULT_PARAM_FILE, help="Path to xyscans scanparameters.json")
    parser.add_argument("--base-root", default=DEFAULT_BASE_ROOT, help="Root directory containing xyscans")
    parser.add_argument("--iteration", type=int, default=None, help="Override iteration from JSON")
    parser.add_argument("--spectra", default="1,2,3", help="Comma-separated spectral channel list")
    parser.add_argument("--filename-prefix", default="xyscan", help="FITS filename prefix, default xyscan")
    parser.add_argument("--normalise", choices=["none", "individual", "global"], default="none", help="Normalisation used for plotting and centroiding")
    parser.add_argument("--wavelength-top", type=int, default=None, help="First spectral pixel to include")
    parser.add_argument("--wavelength-bottom", type=int, default=None, help="Last spectral pixel to include")
    parser.add_argument("--update-param-file", action="store_true", help="Write fitted x/y position back into scanparameters.json")
    args = parser.parse_args()

    params = load_params(args.param_file)
    state = params["state"]
    iteration = int(state["iteration"] if args.iteration is None else args.iteration)
    spectra = parse_spectra(args.spectra)
    scan_dir = get_scan_dir(state, args.base_root, iteration=iteration)

    if not scan_dir.exists():
        raise FileNotFoundError(f"Scan directory does not exist: {scan_dir}")

    print(f"Reading scans from: {scan_dir}")

    raw_heatmaps: Dict[int, np.ndarray] = {}
    x_positions_ref = None
    y_positions_ref = None

    for idx in spectra:
        photometry, x_positions, y_positions = read_xy_scan(
            scan_dir,
            spectrum_index=idx,
            iteration=iteration,
            filename_prefix=args.filename_prefix,
        )
        heatmap = make_heatmap(
            photometry,
            n_x=len(x_positions),
            n_y=len(y_positions),
            wavelength_top=args.wavelength_top,
            wavelength_bottom=args.wavelength_bottom,
        )
        raw_heatmaps[idx] = heatmap

        if x_positions_ref is None:
            x_positions_ref = x_positions
            y_positions_ref = y_positions
        else:
            if not np.allclose(x_positions_ref, x_positions) or not np.allclose(y_positions_ref, y_positions):
                raise ValueError(f"Spectra {idx} has different MOUNT_POS values to the first spectra file.")

    assert x_positions_ref is not None and y_positions_ref is not None

    heatmaps = normalise_heatmaps(raw_heatmaps, args.normalise)
    centroids = {
        idx: find_centroid(heatmaps[idx], x_positions_ref, y_positions_ref)
        for idx in spectra
    }
    common = common_centroid(centroids)

    results = {
        "year": state["year"],
        "date": state["date"],
        "iteration": iteration,
        "scan_dir": str(scan_dir),
        "normalise": args.normalise,
        "wavelength_top": args.wavelength_top,
        "wavelength_bottom": args.wavelength_bottom,
        "x": round(common["x"], 3),
        "y": round(common["y"], 3),
        "common_centroid": {
            "x": round(common["x"], 3),
            "y": round(common["y"], 3),
        },
        "spectra_centroids": {
            str(idx): {
                "x": round(centroids[idx]["x"], 3),
                "y": round(centroids[idx]["y"], 3),
                "method_used": centroids[idx]["method_used"],
                "pixel_x_index": centroids[idx]["pixel_x_index"],
                "pixel_y_index": centroids[idx]["pixel_y_index"],
                "peak_value": centroids[idx]["peak_value"],
            }
            for idx in spectra
        },
    }

    output_json = scan_dir / f"fittedoptimal_xy_{iteration}.json"
    output_png = scan_dir / f"xyscan_{iteration}.png"

    save_json(results, output_json)
    plot_heatmaps(
        heatmaps,
        centroids,
        common,
        x_positions_ref,
        y_positions_ref,
        output_png,
        normalised=(args.normalise != "none"),
    )

    print("\nOptimal common x/y position:")
    print(f'"x": {results["x"]}')
    print(f'"y": {results["y"]}')
    print(f"\nSaved results: {output_json}")
    print(f"Saved heatmaps: {output_png}")

    if args.update_param_file:
        # Written under "state.last_fit" to match scanparameters.json's
        # config/state split: xyscan.py never modifies "config", only this
        # script and xyscan.py's own bookkeeping touch "state".
        state["last_fit"] = {
            "x": results["x"],
            "y": results["y"],
            "xy_centroids": results["spectra_centroids"],
        }
        save_json(params, args.param_file)
        print(f"Updated x/y parameter file: {args.param_file}")


if __name__ == "__main__":
    main()
