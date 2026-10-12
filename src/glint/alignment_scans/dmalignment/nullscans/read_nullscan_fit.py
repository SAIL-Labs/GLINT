"""
read_nullscan_fit.py

Example usage:

# Basic usage (fit a single null scan)
python read_nullscan_fit.py --date 09-01 --iteration 3

# With dark subtraction
python read_nullscan_fit.py --date 05-03 --iteration 12 --dark-iteration 13

# Run only one baseline
python read_nullscan_fit.py --date 05-03 --iteration 12 --baselines 11:31

# Run multiple specific baselines
python read_nullscan_fit.py --date 06-08 --iteration 5 --baselines 11:31 20:31

# Change wavelength (nm)
python read_nullscan_fit.py --date 05-03 --iteration 12 --wavelength-nm 1650

# Disable zoom (fit full OPD range)
python read_nullscan_fit.py --date 05-03 --iteration 12 --no-zoom

# Don’t save figuresr
python read_nullscan_fit.py --date 05-03 --iteration 12 --no-save

# Don’t display figures (useful for remote runs)
python read_nullscan_fit.py --date 05-03 --iteration 12 --no-show

Notes:
- Paths assume: /home/scexao/glint/glintdata/benchalignment/dmalignment/nullscans/{year}/{date}/scanX
- Dark subtraction requires matching OPD sampling between scans
- Baselines are formatted as "segment1:segment2" (e.g. 11:31)

"""

from pathlib import Path
import argparse

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import glint_paths


def sin_squared_model(x, A, f, x0, C):
    return A * (1 - np.cos(2 * np.pi * f * (x - x0))) / 2 + C


BASELINE_DEFAULTS = {
    ("11", "31"): dict(opd_guess=1.0, lower=2, upper=3),
    ("11", "20"): dict(opd_guess=4.0, lower=0.5, upper=1.5),
    ("20", "31"): dict(opd_guess=10.0, lower=-0.5, upper=1.0),
}


def load_movie_and_opd(path, baseline):
    file_path = path / f"avgmovie_{baseline[0]}:{baseline[1]}.fits"

    with fits.open(file_path) as hdul:
        movie = hdul["MOVIE"].data
        opd = hdul["METADATA"].data["OPD"]

    return movie, opd


def fit_and_plot_nullscan(
    year,
    date,
    iteration,
    baseline,
    dark_iteration=None,
    zoomed=True,
    wavelength_nm=1550,
    save=True,
    show=True,
):
    base_dir = glint_paths.data_dir('benchalignment', 'dmalignment', 'nullscans', 'scanoutputs', year, date)
    scan_path = base_dir / f"scan{iteration}"

    defaults = BASELINE_DEFAULTS[tuple(baseline)]
    lowerbound = defaults["lower"]
    upperbound = defaults["upper"]
    opd_guess = defaults["opd_guess"]

    if not zoomed:
        lowerbound = -2
        upperbound = 2

    movie, opd = load_movie_and_opd(scan_path, baseline)

    use_dark = dark_iteration is not None
    if use_dark:
        dark_path = base_dir / f"scan{dark_iteration}"
        dark_movie, dark_opd = load_movie_and_opd(dark_path, baseline)

        if not np.allclose(opd, dark_opd):
            raise ValueError(
                f"Dark OPD positions do not match scan OPD positions for baseline {baseline}."
            )

        movie = movie - dark_movie

    keep = (opd >= lowerbound) & (opd <= upperbound)
    movie = movie[keep]
    opd = opd[keep]

    summed_flux = movie.sum(axis=(1, 2))
    summed_flux_norm = summed_flux / np.max(summed_flux)

    wavelength_um = wavelength_nm / 1000
    initial_freq = 1 / wavelength_um

    p0 = [1, initial_freq, opd_guess, 0.1]

    plt.figure(figsize=(10, 5))

    try:
        popt, _ = curve_fit(
            sin_squared_model,
            opd,
            summed_flux_norm,
            p0=p0,
            maxfev=10000,
        )

        A_fit, f_fit, x0_fit, C_fit = popt

        opd_fit = np.linspace(opd.min(), opd.max(), 500)
        flux_fit = sin_squared_model(opd_fit, *popt)

        min_flux_fit = np.min(flux_fit)
        min_opd_fit = opd_fit[np.argmin(flux_fit)]

        max_flux_fit = np.max(flux_fit)
        visibility = (max_flux_fit - min_flux_fit) / (max_flux_fit + min_flux_fit)

        n_values = np.arange(-10, 11)
        troughs = x0_fit + n_values / f_fit
        troughs_in_range = troughs[
            (troughs >= opd.min()) & (troughs <= opd.max())
        ]

        for t in troughs_in_range:
            plt.axvline(t, color="blue", linestyle=":", alpha=0.5)
            plt.text(
                t,
                -0.057,
                f"{t:.2f}",
                ha="center",
                va="top",
                fontsize=8,
                color="blue",
                fontweight="bold",
                transform=plt.gca().get_xaxis_transform(),
            )

        plt.axvline(
            min_opd_fit,
            color="orange",
            linestyle="--",
            label=f"Best null = {min_opd_fit:.3f} um, visibility = {visibility:.3f}",
        )

        plt.plot(opd, summed_flux_norm, "o", markersize=3, color="red", label="Data")
        plt.plot(opd_fit, flux_fit, "--", color="black", label="Fit")

        print(f"\nBaseline {baseline[0]}:{baseline[1]}")
        print(f"Fit x0: {x0_fit:.3f} um")
        print(f"Best null OPD: {min_opd_fit:.3f} um")
        print(f"Minimum flux: {min_flux_fit:.3f}")
        print(f"Maximum flux: {max_flux_fit:.3f}")
        print(f"Visibility: {visibility:.3f}")

    except RuntimeError:
        plt.plot(opd, summed_flux_norm, "o-", color="red", label="Data, fit failed")
        print(f"Fit failed for baseline {baseline[0]}:{baseline[1]}")

    dark_label = f", dark scan {dark_iteration}" if use_dark else ""
    plt.xlabel("OPD (um)")
    plt.ylabel("Normalised summed intensity")
    plt.title(f"Flux vs OPD: baseline {baseline[0]}:{baseline[1]}, scan {iteration}{dark_label}")
    plt.grid(True)
    plt.legend()

    if save:
        dark_suffix = f"_dark{dark_iteration}" if use_dark else ""
        save_path = scan_path / f"fitted_nullscan_{baseline[0]}:{baseline[1]}_scan{iteration}{dark_suffix}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Fit averaged nullscan FITS files.")

    parser.add_argument("--year", default="2026")
    parser.add_argument("--date", required=True, help="Date folder, e.g. 05-03")
    parser.add_argument("--iteration", type=int, required=True, help="Null scan iteration to fit")

    parser.add_argument(
        "--dark-iteration",
        type=int,
        default=None,
        help="Optional dark scan iteration to subtract",
    )

    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["11:31", "11:20", "20:31"],
        help="Baselines to fit, e.g. 11:31 11:20 20:31",
    )

    parser.add_argument("--wavelength-nm", type=float, default=1550)
    parser.add_argument("--no-zoom", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-show", action="store_true")

    args = parser.parse_args()

    for baseline_str in args.baselines:
        baseline = baseline_str.split(":")

        fit_and_plot_nullscan(
            year=args.year,
            date=args.date,
            iteration=args.iteration,
            baseline=baseline,
            dark_iteration=args.dark_iteration,
            zoomed=not args.no_zoom,
            wavelength_nm=args.wavelength_nm,
            save=not args.no_save,
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()