#!/usr/bin/env python3
"""
Read and analyse GLINT pitch/yaw scan FITS files.

This is the scan-computer version of the local notebook analysis. It reads the
current year/date/iteration from scanparameters.json's "state" section, opens
the matching FITS files, turns each photometry cube into a 2D pitch/yaw
heatmap, estimates the best common centre, saves PNG/JSON outputs, and can
optionally write the best pitch/yaw back into scanparameters.json. The default
optimum is the centre of a fitted rotated elliptical Gaussian, not the
brightest sampled pixel.

Expected input files, as written by pitchyawscan.py:
    /home/scexao/glint/benchalignment/chipmountalignment/pitchyawscans/scanoutput/{year}/{date}/scan{iteration}/
        pitchyawscan_spectra1_{iteration}.fits
        pitchyawscan_spectra2_{iteration}.fits
        pitchyawscan_spectra3_{iteration}.fits

Each FITS file should contain:
    PHOTOMETRY : image cube with shape (n_pitch*n_yaw, boxheight, boxwidth)
    MOUNT_POS  : array [pitch_positions, yaw_positions]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


DEFAULT_PARAM_FILE = "/home/scexao/glint/benchalignment/chipmountalignment/pitchyawscans/scanparameters.json"
DEFAULT_BASE_ROOT = "/home/scexao/glint/benchalignment/chipmountalignment/pitchyawscans/scanoutput"
DEFAULT_SPECTRA = (1, 2, 3)


# Conversion constants copied from the notebook.
# These are mount-specific: degrees per pulse for pitch and yaw.
PITCH_DEG_PER_PULSE = 0.003
YAW_DEG_PER_PULSE = 0.0067


def load_json(filename: str | Path) -> dict:
    """Load a JSON file."""
    filename = Path(filename)
    with filename.open("r") as f:
        return json.load(f)


def save_json(data: dict, filename: str | Path) -> None:
    """Save a dictionary as pretty JSON."""
    filename = Path(filename)
    with filename.open("w") as f:
        json.dump(data, f, indent=2)


def get_scan_dir(state: dict, base_root: str | Path, iteration: int | None = None) -> Path:
    """Build the scan directory from year/date/iteration in scanparameters.json's "state" section."""
    scan_iteration = int(state["iteration"] if iteration is None else iteration)
    return Path(base_root) / str(state["year"]) / str(state["date"]) / f"scan{scan_iteration}"


def parse_spectra(spectra_string: str) -> Tuple[int, ...]:
    """Parse a comma-separated spectra list, e.g. '1,2,3'."""
    return tuple(int(s.strip()) for s in spectra_string.split(",") if s.strip())


def read_pitchyaw_scan(
    scan_dir: str | Path,
    spectrum: int,
    iteration: int,
    filename_prefix: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read one pitch/yaw scan FITS file.

    Returns
    -------
    photometry : ndarray
        Image cube, shape (n_pitch*n_yaw, boxheight, boxwidth).
    pitch_positions : ndarray
        Pitch positions used in the scan, in mount pulses.
    yaw_positions : ndarray
        Yaw positions used in the scan, in mount pulses.
    """
    scan_dir = Path(scan_dir)
    filename = scan_dir / f"{filename_prefix}_spectra{spectrum}_{iteration}.fits"

    if not filename.exists():
        raise FileNotFoundError(f"Could not find scan file: {filename}")

    with fits.open(filename) as hdul:
        photometry = np.asarray(hdul["PHOTOMETRY"].data, dtype=float)
        mount_positions = np.asarray(hdul["MOUNT_POS"].data, dtype=float)

    if mount_positions.shape[0] != 2:
        raise ValueError(f"Expected MOUNT_POS shape (2, nsteps), got {mount_positions.shape} in {filename}")

    pitch_positions, yaw_positions = mount_positions
    return photometry, pitch_positions, yaw_positions


def make_heatmap(
    photometry: np.ndarray,
    n_pitch: int,
    n_yaw: int,
    wavelength_top: int | None = None,
    wavelength_bottom: int | None = None,
) -> np.ndarray:
    """
    Convert a photometry cube into a 2D pitch/yaw heatmap.

    The scan script loops through pitch first, then yaw, so the heatmap indexing is:
        heatmap[pitch_index, yaw_index]
    """
    expected_frames = n_pitch * n_yaw
    if photometry.shape[0] != expected_frames:
        raise ValueError(
            f"Photometry cube has {photometry.shape[0]} frames, but expected "
            f"{expected_frames} from n_pitch={n_pitch}, n_yaw={n_yaw}."
        )

    wl_top = 0 if wavelength_top is None else int(wavelength_top)
    wl_bottom = photometry.shape[1] if wavelength_bottom is None else int(wavelength_bottom)

    heatmap = np.zeros((n_pitch, n_yaw), dtype=float)
    frame = 0
    for i in range(n_pitch):
        for j in range(n_yaw):
            heatmap[i, j] = np.nansum(photometry[frame][wl_top:wl_bottom, :])
            frame += 1

    return heatmap


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
        return {s: img.copy() for s, img in heatmaps.items()}

    if mode == "individual":
        output = {}
        for s, img in heatmaps.items():
            max_value = np.nanmax(img)
            output[s] = img / max_value if max_value != 0 else img.copy()
        return output

    if mode == "global":
        max_value = max(np.nanmax(img) for img in heatmaps.values())
        return {s: (img / max_value if max_value != 0 else img.copy()) for s, img in heatmaps.items()}

    raise ValueError("normalise must be one of: none, individual, global")


def centre_of_mass(image: np.ndarray) -> Tuple[float, float]:
    """
    Calculate centroid in pixel coordinates.

    Returns (x, y), where x is yaw pixel coordinate and y is pitch pixel coordinate.
    """
    image = np.asarray(image, dtype=float)
    image = image - np.nanmin(image)
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


def elliptical_gaussian_peak(
    image: np.ndarray,
    threshold_fraction: float = 0.35,
    fit_window_half_size: int | None = None,
) -> Tuple[float, float, dict]:
    """
    Fit a rotated elliptical 2D Gaussian to a pitch/yaw heatmap.

    The returned centre (x0, y0) is the fitted centre of the elliptical Gaussian,
    not the brightest sampled pixel. This gives a sub-grid estimate of the best
    pitch/yaw position and avoids choosing a noisy single-pixel maximum.

    Returns
    -------
    x0, y0 : float
        Fitted centre in heatmap pixel coordinates. x is the yaw axis, y is the
        pitch axis.
    info : dict
        Fit diagnostic information saved into the output JSON.
    """
    if not SCIPY_AVAILABLE:
        return np.nan, np.nan, {"fit_success": False, "reason": "scipy_unavailable"}

    raw = np.asarray(image, dtype=float)
    if raw.ndim != 2:
        return np.nan, np.nan, {"fit_success": False, "reason": "image_not_2d"}

    finite = np.isfinite(raw)
    if not np.any(finite):
        return np.nan, np.nan, {"fit_success": False, "reason": "no_finite_pixels"}

    # Robust background removal helps when the spectral crop has dark-subtraction
    # offsets or broad scattered light.
    background = float(np.nanpercentile(raw, 10))
    fit_image = raw - background
    fit_image[~finite] = 0.0
    fit_image[fit_image < 0] = 0.0

    peak = float(np.nanmax(fit_image))
    if not np.isfinite(peak) or peak <= 0:
        return np.nan, np.nan, {"fit_success": False, "reason": "non_positive_peak"}

    y_size, x_size = fit_image.shape
    max_y, max_x = np.unravel_index(np.nanargmax(fit_image), fit_image.shape)

    # Fit around the bright lobe by default, rather than the whole map. This
    # makes the fit less sensitive to parasitic light or secondary maxima.
    if fit_window_half_size is None:
        fit_window_half_size = max(3, int(np.ceil(max(x_size, y_size) / 4)))

    y0 = max(0, max_y - fit_window_half_size)
    y1 = min(y_size, max_y + fit_window_half_size + 1)
    x0 = max(0, max_x - fit_window_half_size)
    x1 = min(x_size, max_x + fit_window_half_size + 1)

    sub = fit_image[y0:y1, x0:x1].copy()

    local_peak = float(np.nanmax(sub))
    thresholded = sub.copy()
    thresholded[thresholded < threshold_fraction * local_peak] = 0.0
    if np.count_nonzero(thresholded) >= 8:
        sub = thresholded

    yy, xx = np.mgrid[y0:y1, x0:x1]

    total = float(np.nansum(sub))
    if total > 0:
        x_init = float(np.nansum(xx * sub) / total)
        y_init = float(np.nansum(yy * sub) / total)
    else:
        x_init = float(max_x)
        y_init = float(max_y)

    sx_init = max(1.0, (x1 - x0) / 4)
    sy_init = max(1.0, (y1 - y0) / 4)
    offset_init = float(np.nanmedian(raw[finite]))
    amp_init = peak

    initial_guess = (amp_init, x_init, y_init, sx_init, sy_init, 0.0, offset_init)

    lower_bounds = (0.0, x0 - 1.0, y0 - 1.0, 0.25, 0.25, -np.pi / 2, -np.inf)
    upper_bounds = (np.inf, x1 + 1.0, y1 + 1.0, max(x_size, 1) * 2, max(y_size, 1) * 2, np.pi / 2, np.inf)

    try:
        popt, _ = curve_fit(
            two_d_gaussian,
            (xx, yy),
            raw[y0:y1, x0:x1].ravel(),
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=20000,
        )
        amplitude, xo, yo, sigma_x, sigma_y, theta, offset = popt

        if not (np.isfinite(xo) and np.isfinite(yo)):
            raise RuntimeError("non-finite fitted centre")
        if xo < -0.5 or xo > x_size - 0.5 or yo < -0.5 or yo > y_size - 0.5:
            raise RuntimeError("fitted centre outside heatmap")

        info = {
            "fit_success": True,
            "amplitude": float(amplitude),
            "sigma_x_pix": float(abs(sigma_x)),
            "sigma_y_pix": float(abs(sigma_y)),
            "theta_rad": float(theta),
            "offset": float(offset),
            "fit_window": [int(y0), int(y1), int(x0), int(x1)],
        }
        return float(xo), float(yo), info

    except Exception as exc:
        return np.nan, np.nan, {"fit_success": False, "reason": repr(exc)}


# Backwards-compatible name used by older scripts/notebooks.
def gaussian_peak(image: np.ndarray, threshold_fraction: float = 0.35) -> Tuple[float, float]:
    x_pix, y_pix, _ = elliptical_gaussian_peak(image, threshold_fraction=threshold_fraction)
    return x_pix, y_pix


def pixel_to_mount_position(
    x_pix: float,
    y_pix: float,
    pitch_positions: np.ndarray,
    yaw_positions: np.ndarray,
) -> Tuple[float, float]:
    """Convert heatmap pixel coordinates to mount pulse positions. Returns (pitch, yaw)."""
    pitch = np.interp(y_pix, np.arange(len(pitch_positions)), pitch_positions)
    yaw = np.interp(x_pix, np.arange(len(yaw_positions)), yaw_positions)
    return float(pitch), float(yaw)


def find_optimum(
    heatmap: np.ndarray,
    pitch_positions: np.ndarray,
    yaw_positions: np.ndarray,
    method: str = "elliptical_gaussian",
    threshold_fraction: float = 0.35,
) -> dict:
    """
    Find the optimum pitch/yaw position for one spectrum.

    method options:
        max                  : brightest heatmap pixel
        centroid             : centre of mass of the heatmap
        gaussian             : alias for elliptical_gaussian
        elliptical_gaussian  : rotated elliptical 2D Gaussian fit; falls back
                               to centroid, then max if fitting fails
    """
    method = method.lower()
    if method == "gaussian":
        method = "elliptical_gaussian"

    max_i, max_j = np.unravel_index(np.nanargmax(heatmap), heatmap.shape)
    max_pitch = float(pitch_positions[max_i])
    max_yaw = float(yaw_positions[max_j])

    if method == "max":
        return {
            "method_used": "max",
            "pitch": max_pitch,
            "yaw": max_yaw,
            "pixel_x": float(max_j),
            "pixel_y": float(max_i),
            "peak_value": float(np.nanmax(heatmap)),
        }

    fit_info = None
    if method == "centroid":
        x_pix, y_pix = centre_of_mass(heatmap)
        used = "centroid"
    elif method == "elliptical_gaussian":
        x_pix, y_pix, fit_info = elliptical_gaussian_peak(
            heatmap,
            threshold_fraction=threshold_fraction,
        )
        used = "elliptical_gaussian"
        if not (np.isfinite(x_pix) and np.isfinite(y_pix)):
            x_pix, y_pix = centre_of_mass(heatmap)
            used = "centroid_fallback"
    else:
        raise ValueError("method must be one of: max, centroid, gaussian, elliptical_gaussian")

    if not (np.isfinite(x_pix) and np.isfinite(y_pix)):
        x_pix, y_pix = float(max_j), float(max_i)
        used = "max_fallback"

    pitch, yaw = pixel_to_mount_position(x_pix, y_pix, pitch_positions, yaw_positions)

    result = {
        "method_used": used,
        "pitch": pitch,
        "yaw": yaw,
        "pixel_x": float(x_pix),
        "pixel_y": float(y_pix),
        "peak_value": float(np.nanmax(heatmap)),
        "max_pixel_pitch": max_pitch,
        "max_pixel_yaw": max_yaw,
    }
    if method == "elliptical_gaussian":
        result["elliptical_gaussian_fit"] = fit_info
    return result


def mount_pulses_to_mrad(
    pitch_pulses: float,
    yaw_pulses: float,
    pitch_zero: float,
    yaw_zero: float,
) -> Tuple[float, float]:
    """
    Convert mount pulse offsets to mrad, using the notebook convention.

    The notebook divided the angle by 10 before converting to mrad. I have kept
    that factor here so this reproduces your existing plots.
    """
    pitch_mrad = ((pitch_pulses - pitch_zero) * PITCH_DEG_PER_PULSE * np.pi / 180) / 10 * 1000
    yaw_mrad = ((yaw_pulses - yaw_zero) * YAW_DEG_PER_PULSE * np.pi / 180) / 10 * 1000
    return float(pitch_mrad), float(yaw_mrad)


def plot_heatmaps(
    heatmaps: Dict[int, np.ndarray],
    optima: Dict[int, dict],
    common_centre: dict,
    pitch_positions: np.ndarray,
    yaw_positions: np.ndarray,
    output_png: str | Path,
    normalised: bool,
    show: bool = True,
) -> None:
    """Save side-by-side pitch/yaw heatmaps with fitted centres marked."""
    spectra = list(heatmaps.keys())

    pitch_step = abs(pitch_positions[1] - pitch_positions[0]) if len(pitch_positions) > 1 else 1
    yaw_step = abs(yaw_positions[1] - yaw_positions[0]) if len(yaw_positions) > 1 else 1

    extent = [
        yaw_positions[0] - yaw_step / 2,
        yaw_positions[-1] + yaw_step / 2,
        pitch_positions[0] - pitch_step / 2,
        pitch_positions[-1] + pitch_step / 2,
    ]

    fig, axs = plt.subplots(1, len(spectra), figsize=(7 * len(spectra), 5), squeeze=False)
    axs = axs[0]

    for ax, spectrum in zip(axs, spectra):
        im = ax.imshow(heatmaps[spectrum], origin="lower", extent=extent, aspect="auto")
        ax.plot(optima[spectrum]["yaw"], optima[spectrum]["pitch"], "rx", markersize=8, label="spectrum fit")
        ax.plot(common_centre["yaw"], common_centre["pitch"], "go", markersize=6, label="common centre")
        ax.set_xlabel("yaw (mount pulses)")
        ax.set_ylabel("pitch (mount pulses)")
        ax.set_title(
            f"Spectrum {spectrum}: pitch={optima[spectrum]['pitch']:.1f}, "
            f"yaw={optima[spectrum]['yaw']:.1f}"
        )
        ax.legend(loc="best")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Normalised intensity" if normalised else "Summed intensity")

    fig.suptitle(
        f"Common centre: pitch={common_centre['pitch']:.1f}, yaw={common_centre['yaw']:.1f}",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)

    # Preserve the original GUI behaviour: after the scan analysis finishes,
    # display the heatmap window automatically. block=True is intentional here
    # because some Matplotlib backends return immediately from bare plt.show()
    # when launched from a subprocess.
    if show:
        plt.show(block=True)
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Find optimal pitch/yaw centre from saved GLINT pitch/yaw scans.")
    parser.add_argument("--param-file", default=DEFAULT_PARAM_FILE, help="Path to pitch/yaw scanparameters.json")
    parser.add_argument("--base-root", default=DEFAULT_BASE_ROOT, help="Root directory containing pitchyawscans")
    parser.add_argument("--iteration", type=int, default=None, help="Override iteration from JSON")
    parser.add_argument("--spectra", default="1,2,3", help="Comma-separated spectra list")
    parser.add_argument("--filename-prefix", default="pitchyawscan", help="FITS filename prefix")
    parser.add_argument("--method", choices=["max", "centroid", "gaussian", "elliptical_gaussian"], default="elliptical_gaussian", help="Optimum-finding method. gaussian is an alias for elliptical_gaussian.")
    parser.add_argument("--threshold-fraction", type=float, default=0.35, help="Local threshold fraction used during the elliptical-Gaussian fit")
    parser.add_argument("--normalise", choices=["none", "individual", "global"], default="none", help="Normalisation before fitting/plotting")
    parser.add_argument("--wavelength-top", type=int, default=None, help="First wavelength/spectral pixel to include")
    parser.add_argument("--wavelength-bottom", type=int, default=None, help="Last wavelength/spectral pixel to include")
    parser.add_argument("--update-param-file", action="store_true", help="Write best pitch/yaw back into scanparameters.json")
    parser.add_argument("--show", dest="show", action="store_true", default=True, help="Display the resulting heatmap window after saving it. This is the default.")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Save the heatmap without opening a Matplotlib window.")
    args = parser.parse_args()

    params = load_json(args.param_file)
    state = params["state"]
    iteration = int(state["iteration"] if args.iteration is None else args.iteration)
    spectra = parse_spectra(args.spectra)
    scan_dir = get_scan_dir(state, args.base_root, iteration=iteration)

    if not scan_dir.exists():
        raise FileNotFoundError(f"Scan directory does not exist: {scan_dir}")

    print(f"Reading scans from: {scan_dir}")

    raw_heatmaps: Dict[int, np.ndarray] = {}
    pitch_positions_ref = None
    yaw_positions_ref = None

    for spectrum in spectra:
        photometry, pitch_positions, yaw_positions = read_pitchyaw_scan(
            scan_dir,
            spectrum=spectrum,
            iteration=iteration,
            filename_prefix=args.filename_prefix,
        )
        heatmap = make_heatmap(
            photometry,
            n_pitch=len(pitch_positions),
            n_yaw=len(yaw_positions),
            wavelength_top=args.wavelength_top,
            wavelength_bottom=args.wavelength_bottom,
        )
        raw_heatmaps[spectrum] = heatmap

        if pitch_positions_ref is None:
            pitch_positions_ref = pitch_positions
            yaw_positions_ref = yaw_positions
        else:
            if not np.allclose(pitch_positions_ref, pitch_positions) or not np.allclose(yaw_positions_ref, yaw_positions):
                raise ValueError(f"Spectrum {spectrum} has different MOUNT_POS values to the first spectrum.")

    assert pitch_positions_ref is not None and yaw_positions_ref is not None

    heatmaps = normalise_heatmaps(raw_heatmaps, args.normalise)
    optima = {
        spectrum: find_optimum(
            heatmap,
            pitch_positions_ref,
            yaw_positions_ref,
            method=args.method,
            threshold_fraction=args.threshold_fraction,
        )
        for spectrum, heatmap in heatmaps.items()
    }

    # Common centre = average of per-spectrum fitted centres, matching the notebook logic.
    common_pitch = float(np.nanmean([optima[s]["pitch"] for s in spectra]))
    common_yaw = float(np.nanmean([optima[s]["yaw"] for s in spectra]))
    common_pitch_mrad, common_yaw_mrad = mount_pulses_to_mrad(
        common_pitch,
        common_yaw,
        pitch_zero=float(pitch_positions_ref[0]),
        yaw_zero=float(yaw_positions_ref[0]),
    )

    common_centre = {
        "pitch": common_pitch,
        "yaw": common_yaw,
        "pitch_mrad_from_scan_start": common_pitch_mrad,
        "yaw_mrad_from_scan_start": common_yaw_mrad,
    }

    results = {
        "year": state["year"],
        "date": state["date"],
        "iteration": iteration,
        "scan_dir": str(scan_dir),
        "method_requested": args.method,
        "normalise": args.normalise,
        "wavelength_top": args.wavelength_top,
        "wavelength_bottom": args.wavelength_bottom,
        "common_centre": {
            "pitch": round(common_pitch, 3),
            "yaw": round(common_yaw, 3),
            "pitch_mrad_from_scan_start": round(common_pitch_mrad, 3),
            "yaw_mrad_from_scan_start": round(common_yaw_mrad, 3),
        },
        "per_spectrum": {str(s): optima[s] for s in spectra},
    }

    output_json = scan_dir / f"fittedoptimal_pitchyaw_{iteration}.json"
    output_png = scan_dir / f"pitchyaw_heatmaps_{iteration}.png"

    save_json(results, output_json)
    plot_heatmaps(
        heatmaps,
        optima,
        common_centre,
        pitch_positions_ref,
        yaw_positions_ref,
        output_png,
        normalised=(args.normalise != "none"),
        show=args.show,
    )

    print("\nOptimal common centre:")
    print(f"pitch = {results['common_centre']['pitch']}")
    print(f"yaw   = {results['common_centre']['yaw']}")
    print(f"pitch_mrad_from_scan_start = {results['common_centre']['pitch_mrad_from_scan_start']}")
    print(f"yaw_mrad_from_scan_start   = {results['common_centre']['yaw_mrad_from_scan_start']}")
    print(f"\nSaved results: {output_json}")
    print(f"Saved heatmaps: {output_png}")

    if args.update_param_file:
        # Written under "state.last_fit" to match scanparameters.json's
        # config/state split: pitchyawscan.py never modifies "config", only
        # this script and pitchyawscan.py's own bookkeeping touch "state".
        state["last_fit"] = results["common_centre"]
        save_json(params, args.param_file)
        print(f"Updated pitch/yaw parameter file: {args.param_file}")


if __name__ == "__main__":
    main()
