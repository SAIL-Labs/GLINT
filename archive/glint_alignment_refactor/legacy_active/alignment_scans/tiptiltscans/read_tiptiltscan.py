#!/usr/bin/env python3
"""
Find optimal tip/tilt positions from GLINT tip-tilt scan FITS files.

This is the scan-computer version of the notebook analysis. It reads the date,
year, and iteration from scanparameters.json, opens the corresponding scan files,
sums each photometry cube into a 2D tip/tilt heatmap, estimates the optimum
position for each segment, saves heatmap PNGs, and writes the results to JSON.

Expected scan files are written by the scan script as:
    /home/scexao/glint/alignment_scans/tiptiltscans/{year}/{date}/scan{iteration}/
        tiptiltscan_seg11_{iteration}.fits
        tiptiltscan_seg20_{iteration}.fits
        tiptiltscan_seg31_{iteration}.fits

Each FITS file should contain:
    PHOTOMETRY  : image cube with shape (n_tip*n_tilt, boxheight, boxwidth)
    DM_POS      : array [tip_positions, tilt_positions]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib
# matplotlib.use("Agg")  # safe on observing/scanning computers without display
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


DEFAULT_PARAM_FILE = "/home/scexao/glint/alignment_scans/tiptiltscans/scanparameters.json"
DEFAULT_NULLSCAN_PARAM_FILE = "/home/scexao/glint/alignment_scans/nullscans/scanparameters.json"
DEFAULT_BASE_ROOT = "/home/scexao/glint/alignment_scans/tiptiltscans"
DEFAULT_SEGMENTS = (11, 20, 31)



def load_params(param_file: str | Path) -> dict:
    """Load scan parameters from JSON."""
    param_file = Path(param_file)
    with param_file.open("r") as f:
        return json.load(f)


def save_json(data: dict, filename: str | Path) -> None:
    """Save a dictionary as pretty JSON."""
    filename = Path(filename)
    with filename.open("w") as f:
        json.dump(data, f, indent=2)


def get_scan_dir(params: dict, base_root: str | Path, iteration: int | None = None) -> Path:
    """
    Build the scan directory from the JSON parameters.

    The scan script stores files under:
        base_root / year / date / scan{iteration}
    """
    year = params["year"]
    date = params["date"]
    scan_iteration = int(params["iteration"] if iteration is None else iteration)
    return Path(base_root) / str(year) / str(date) / f"scan{scan_iteration}"


def read_tiptilt_scan(scan_dir: str | Path, segment: int, iteration: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read one segment's tip/tilt scan FITS file.

    Returns
    -------
    photometry : np.ndarray
        Image cube, shape (n_tip*n_tilt, boxheight, boxwidth).
    tip_positions : np.ndarray
        Tip positions used in the scan.
    tilt_positions : np.ndarray
        Tilt positions used in the scan.
    """
    scan_dir = Path(scan_dir)
    filename = scan_dir / f"tiptiltscan_seg{segment}_{iteration}.fits"

    if not filename.exists():
        raise FileNotFoundError(f"Could not find scan file: {filename}")

    with fits.open(filename) as hdul:
        photometry = np.asarray(hdul["PHOTOMETRY"].data, dtype=float)
        dmpositions = np.asarray(hdul["DM_POS"].data, dtype=float)

    if dmpositions.shape[0] != 2:
        raise ValueError(f"Expected DM_POS shape (2, nsteps), got {dmpositions.shape} in {filename}")

    tip_positions, tilt_positions = dmpositions
    return photometry, tip_positions, tilt_positions


def make_heatmap(
    photometry: np.ndarray,
    n_tip: int,
    n_tilt: int,
    wavelength_top: int | None = None,
    wavelength_bottom: int | None = None,
) -> np.ndarray:
    """
    Convert a photometry cube into a 2D tip/tilt heatmap.

    The scan script loops over tip first, then tilt, so the reshaped output is:
        heatmap[tip_index, tilt_index]

    wavelength_top/bottom can be used to only sum over part of the spectral crop.
    If omitted, the full crop is summed.
    """
    expected_frames = n_tip * n_tilt
    if photometry.shape[0] != expected_frames:
        raise ValueError(
            f"Photometry cube has {photometry.shape[0]} frames, but expected "
            f"{expected_frames} from n_tip={n_tip}, n_tilt={n_tilt}."
        )

    wl_top = 0 if wavelength_top is None else int(wavelength_top)
    wl_bottom = photometry.shape[1] if wavelength_bottom is None else int(wavelength_bottom)

    heatmap = np.zeros((n_tip, n_tilt), dtype=float)
    frame = 0
    for i in range(n_tip):
        for j in range(n_tilt):
            heatmap[i, j] = np.sum(photometry[frame][wl_top:wl_bottom, :])
            frame += 1

    return heatmap


def centre_of_mass(image: np.ndarray) -> Tuple[float, float]:
    """
    Calculate centroid in pixel coordinates.

    Returns (x, y), where x is the tilt-axis pixel coordinate and y is the
    tip-axis pixel coordinate.
    """
    image = np.asarray(image, dtype=float)
    image = image - np.nanmin(image)  # prevents negative dark-subtraction offsets biasing centroid
    total = np.nansum(image)

    if total <= 0 or not np.isfinite(total):
        return np.nan, np.nan

    y = np.arange(image.shape[0])
    x = np.arange(image.shape[1])
    x_centroid = np.nansum(x * np.nansum(image, axis=0)) / total
    y_centroid = np.nansum(y * np.nansum(image, axis=1)) / total
    return float(x_centroid), float(y_centroid)


def two_d_gaussian(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """Rotated 2D Gaussian for fitting the heatmap peak."""
    x, y = coords
    xo = float(xo)
    yo = float(yo)
    sigma_x = abs(float(sigma_x))
    sigma_y = abs(float(sigma_y))

    a = np.cos(theta) ** 2 / (2 * sigma_x ** 2) + np.sin(theta) ** 2 / (2 * sigma_y ** 2)
    b = -np.sin(2 * theta) / (4 * sigma_x ** 2) + np.sin(2 * theta) / (4 * sigma_y ** 2)
    c = np.sin(theta) ** 2 / (2 * sigma_x ** 2) + np.cos(theta) ** 2 / (2 * sigma_y ** 2)

    g = offset + amplitude * np.exp(-(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2)))
    return g.ravel()


def gaussian_peak(image: np.ndarray, threshold_fraction: float = 0.7) -> Tuple[float, float]:
    """
    Fit a 2D Gaussian to the brightest part of a heatmap.

    Returns (x, y) in pixel coordinates. Falls back to NaN if fitting fails.
    """
    if not SCIPY_AVAILABLE:
        return np.nan, np.nan

    fit_image = np.asarray(image, dtype=float).copy()
    fit_image = fit_image - np.nanmin(fit_image)

    max_value = np.nanmax(fit_image)
    if not np.isfinite(max_value) or max_value <= 0:
        return np.nan, np.nan

    fit_image[fit_image < threshold_fraction * max_value] = 0

    y_size, x_size = fit_image.shape
    x, y = np.meshgrid(np.arange(x_size), np.arange(y_size))

    max_y, max_x = np.unravel_index(np.nanargmax(fit_image), fit_image.shape)
    initial_guess = (
        max_value,          # amplitude
        float(max_x),       # xo
        float(max_y),       # yo
        max(x_size / 4, 1), # sigma_x
        max(y_size / 4, 1), # sigma_y
        0.0,                # theta
        float(np.nanmedian(fit_image)),
    )

    try:
        popt, _ = curve_fit(two_d_gaussian, (x, y), fit_image.ravel(), p0=initial_guess, maxfev=10000)
        _, xo, yo, *_ = popt
        return float(xo), float(yo)
    except Exception:
        return np.nan, np.nan


def pixel_to_tiptilt(x_pix: float, y_pix: float, tip_positions: np.ndarray, tilt_positions: np.ndarray) -> Tuple[float, float]:
    """
    Convert heatmap pixel coordinates to physical tip/tilt coordinates.

    Returns (tip, tilt).
    """
    tip = np.interp(y_pix, np.arange(len(tip_positions)), tip_positions)
    tilt = np.interp(x_pix, np.arange(len(tilt_positions)), tilt_positions)
    return float(tip), float(tilt)


def find_optimum(
    heatmap: np.ndarray,
    tip_positions: np.ndarray,
    tilt_positions: np.ndarray,
    method: str = "gaussian",
    threshold_fraction: float = 0.7,
) -> dict:
    """
    Find the optimum tip/tilt position.

    method options:
        max       : choose the brightest heatmap pixel
        centroid  : centre of mass of the heatmap
        gaussian  : 2D Gaussian fit, falling back to centroid, then max
    """
    method = method.lower()

    max_i, max_j = np.unravel_index(np.nanargmax(heatmap), heatmap.shape)
    max_tip = float(tip_positions[max_i])
    max_tilt = float(tilt_positions[max_j])

    if method == "max":
        return {
            "method_used": "max",
            "tip": max_tip,
            "tilt": max_tilt,
            "pixel_x": float(max_j),
            "pixel_y": float(max_i),
            "peak_value": float(np.nanmax(heatmap)),
        }

    if method == "centroid":
        x_pix, y_pix = centre_of_mass(heatmap)
        used = "centroid"
    elif method == "gaussian":
        x_pix, y_pix = gaussian_peak(heatmap, threshold_fraction=threshold_fraction)
        used = "gaussian"
        if not (np.isfinite(x_pix) and np.isfinite(y_pix)):
            x_pix, y_pix = centre_of_mass(heatmap)
            used = "centroid_fallback"
    else:
        raise ValueError("method must be one of: max, centroid, gaussian")

    if not (np.isfinite(x_pix) and np.isfinite(y_pix)):
        x_pix, y_pix = float(max_j), float(max_i)
        used = "max_fallback"

    tip, tilt = pixel_to_tiptilt(x_pix, y_pix, tip_positions, tilt_positions)

    return {
        "method_used": used,
        "tip": tip,
        "tilt": tilt,
        "pixel_x": float(x_pix),
        "pixel_y": float(y_pix),
        "peak_value": float(np.nanmax(heatmap)),
    }


def normalise_heatmaps(heatmaps: Dict[int, np.ndarray], mode: str) -> Dict[int, np.ndarray]:
    """
    Normalise heatmaps for plotting and fitting.

    mode options:
        none        : no normalisation
        individual  : each segment divided by its own max
        global      : all segments divided by the global max
    """
    mode = mode.lower()
    if mode == "none":
        return {seg: img.copy() for seg, img in heatmaps.items()}

    if mode == "individual":
        output = {}
        for seg, img in heatmaps.items():
            max_value = np.nanmax(img)
            output[seg] = img / max_value if max_value != 0 else img.copy()
        return output

    if mode == "global":
        max_value = max(np.nanmax(img) for img in heatmaps.values())
        return {seg: (img / max_value if max_value != 0 else img.copy()) for seg, img in heatmaps.items()}

    raise ValueError("normalise must be one of: none, individual, global")

def apply_threshold_mask(image: np.ndarray, threshold_fraction: float) -> np.ndarray:
    """
    Apply threshold_fraction mask for visualisation (maximisation case).

    Keeps values within threshold_fraction of the peak, sets others to NaN.
    """
    img = np.asarray(image, dtype=float)
    max_val = np.nanmax(img)

    if not np.isfinite(max_val) or max_val == 0:
        return img

    mask = img >= max_val * (threshold_fraction)

    return np.where(mask, img, np.nan)

def plot_heatmaps(
    heatmaps: Dict[int, np.ndarray],
    optima: Dict[int, dict],
    tip_positions: np.ndarray,
    tilt_positions: np.ndarray,
    output_png: str | Path,
    normalised: bool,
    threshold_fraction: float,
) -> None:
    """Save side-by-side segment heatmaps with optimum positions marked."""
    segments = list(heatmaps.keys())

    tip_step = abs(tip_positions[1] - tip_positions[0]) if len(tip_positions) > 1 else 1
    tilt_step = abs(tilt_positions[1] - tilt_positions[0]) if len(tilt_positions) > 1 else 1

    extent = [
        tilt_positions[0] - tilt_step / 2,
        tilt_positions[-1] + tilt_step / 2,
        tip_positions[0] - tip_step / 2,
        tip_positions[-1] + tip_step / 2,
    ]

    fig, axs = plt.subplots(1, len(segments), figsize=(7 * len(segments), 5), squeeze=False)
    axs = axs[0]

    for ax, segment in zip(axs, segments):
        # Apply threshold masking for visualisation
        image = apply_threshold_mask(heatmaps[segment], threshold_fraction)
        im = ax.imshow(image, origin="lower", extent=extent, aspect="auto")
        ax.plot(optima[segment]["tilt"], optima[segment]["tip"], "rx", markersize=8, label="optimum")
        ax.set_xlabel("tilt")
        ax.set_ylabel("tip")
        ax.set_title(
            f"Segment {segment}: tip={optima[segment]['tip']:.3f}, "
            f"tilt={optima[segment]['tilt']:.3f}"
        )
        ax.legend(loc="best")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Normalised intensity" if normalised else "Summed intensity")

    fig.tight_layout()
    fig.savefig(output_png, dpi=200)

    plt.show()
    # plt.close(fig)


def parse_segments(segment_string: str) -> Tuple[int, ...]:
    """Parse comma-separated segment list, e.g. '11,20,31'."""
    return tuple(int(s.strip()) for s in segment_string.split(",") if s.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Find optimal tip/tilt positions from saved GLINT tip-tilt scans.")
    parser.add_argument("--param-file", default=DEFAULT_PARAM_FILE, help="Path to scanparameters.json")
    parser.add_argument("--base-root", default=DEFAULT_BASE_ROOT, help="Root directory containing tiptiltscans")
    parser.add_argument("--iteration", type=int, default=None, help="Override iteration from JSON")
    parser.add_argument("--segments", default="11,20,31", help="Comma-separated segment list")
    parser.add_argument("--method", choices=["max", "centroid", "gaussian"], default="gaussian", help="Optimum-finding method")
    parser.add_argument("--threshold-fraction", type=float, default=0.7, help="Threshold fraction for Gaussian fit")
    parser.add_argument("--normalise", choices=["none", "individual", "global"], default="individual", help="Normalisation used before optimum finding and plotting")
    parser.add_argument("--wavelength-top", type=int, default=None, help="First wavelength/spectral pixel to include")
    parser.add_argument("--wavelength-bottom", type=int, default=None, help="Last wavelength/spectral pixel to include")
    parser.add_argument("--update-param-file", action="store_true", help="Write best tips/tilts back into scanparameters.json")
    parser.add_argument("--nullscan-param-file", default=DEFAULT_NULLSCAN_PARAM_FILE, help="Path to nullscan scanparameters.json to update with fitted tips/tilts")
    args = parser.parse_args()

    params = load_params(args.param_file)
    iteration = int(params["iteration"] if args.iteration is None else args.iteration)
    segments = parse_segments(args.segments)
    scan_dir = get_scan_dir(params, args.base_root, iteration=iteration)

    if not scan_dir.exists():
        raise FileNotFoundError(f"Scan directory does not exist: {scan_dir}")

    print(f"Reading scans from: {scan_dir}")

    raw_heatmaps: Dict[int, np.ndarray] = {}
    tip_positions_ref = None
    tilt_positions_ref = None

    for segment in segments:
        photometry, tip_positions, tilt_positions = read_tiptilt_scan(scan_dir, segment, iteration)
        heatmap = make_heatmap(
            photometry,
            n_tip=len(tip_positions),
            n_tilt=len(tilt_positions),
            wavelength_top=args.wavelength_top,
            wavelength_bottom=args.wavelength_bottom,
        )
        raw_heatmaps[segment] = heatmap

        if tip_positions_ref is None:
            tip_positions_ref = tip_positions
            tilt_positions_ref = tilt_positions
        else:
            if not np.allclose(tip_positions_ref, tip_positions) or not np.allclose(tilt_positions_ref, tilt_positions):
                raise ValueError(f"Segment {segment} has different DM_POS values to the first segment.")

    assert tip_positions_ref is not None and tilt_positions_ref is not None

    heatmaps = normalise_heatmaps(raw_heatmaps, args.normalise)
    optima = {
        segment: find_optimum(
            heatmap,
            tip_positions_ref,
            tilt_positions_ref,
            method=args.method,
            threshold_fraction=args.threshold_fraction,
        )
        for segment, heatmap in heatmaps.items()
    }

    results = {
        "year": params["year"],
        "date": params["date"],
        "iteration": iteration,
        "scan_dir": str(scan_dir),
        "method_requested": args.method,
        "normalise": args.normalise,
        "wavelength_top": args.wavelength_top,
        "wavelength_bottom": args.wavelength_bottom,
        "tips": {str(seg): round(optima[seg]["tip"], 3) for seg in segments},
        "tilts": {str(seg): round(optima[seg]["tilt"], 3) for seg in segments},
        "details": {str(seg): optima[seg] for seg in segments},
    }

    output_json = scan_dir / f"fittedoptimal_tiptilt_{iteration}.json"
    output_png = scan_dir / f"tiptilt_heatmaps_{iteration}.png"

    save_json(results, output_json)
    plot_heatmaps(
        heatmaps,
        optima,
        tip_positions_ref,
        tilt_positions_ref,
        output_png,
        normalised=(args.normalise != "none"),
        threshold_fraction=args.threshold_fraction,
    )

    print("\nOptimal positions:")
    print(f'"tips": {json.dumps(results["tips"])}')
    print(f'"tilts": {json.dumps(results["tilts"])}')
    print(f"\nSaved results: {output_json}")
    print(f"Saved heatmaps: {output_png}")

    if args.update_param_file:
        # Update the tip-tilt scan parameter file too, if desired
        params["tips"] = results["tips"]
        params["tilts"] = results["tilts"]
        save_json(params, args.param_file)
        print(f"Updated tip-tilt parameter file: {args.param_file}")

        # Update the nullscan parameter file
        nullscan_param_file = Path(args.nullscan_param_file)

        if not nullscan_param_file.exists():
            raise FileNotFoundError(f"Could not find nullscan parameter file: {nullscan_param_file}")

        nullscan_params = load_params(nullscan_param_file)
        nullscan_params["tips"] = results["tips"]
        nullscan_params["tilts"] = results["tilts"]

        save_json(nullscan_params, nullscan_param_file)
        print(f"Updated nullscan parameter file: {nullscan_param_file}")
    
    


if __name__ == "__main__":
    main()
