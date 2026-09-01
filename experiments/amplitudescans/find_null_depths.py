import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm
import re
from scipy.optimize import curve_fit
import glint_paths

# -------- CONFIG --------

PARENT_DIR = str(glint_paths.DATA_ROOT / 'alignment_scans' / 'amplitudescans' / '2025' / '08-22' / 'scan1')
BASELINE = ["11", "31"]
TOP =57          # Top row for spectral region
BOTTOM = 59     # Bottom row for spectral region
OPD_RANGE = (-1, 0)
PEAKS = [21, 53, 84, 115, 147, 178]  # Spectral peak columns
VERTICAL_OFFSETS = [35, 32, 29, 24, 21, 19]  # Corresponding vertical positions
BOX_HALFWIDTH = 2
PHOTOMETRY_DEPTH = 60  # Number of rows for full photometric extraction

SAVE_NULLSCAN_PLOT = True
HEATMAP_OUTPUT = os.path.join(PARENT_DIR, f'null_depth_heatmap_{BASELINE[0]}_{BASELINE[1]}.png')

# -------- SIN^2 MODEL --------

def sin_squared_model(x, A, f, x0, C):
    return A * (1 - np.cos(2 * np.pi * f * (x - x0))) / 2 + C

# -------- LOAD DARK MOVIE --------

dark_path = os.path.join(PARENT_DIR, f'movie_{BASELINE[0]}:{BASELINE[1]}.fits')
try:
    with fits.open(dark_path) as dark_hdul:
        dark_movie = dark_hdul['MOVIE'].data
        dark_opd = dark_hdul['METADATA'].data['OPD']
except Exception as e:
    print(f"[ERROR] Failed to load dark movie: {e}")
    dark_movie = None
    dark_opd = None

# -------- PROCESS --------

seg1, seg2 = BASELINE
results = []

photometry_ratios = []
tilt1_vals = set()
tilt2_vals = set()

for subdir in sorted(os.listdir(PARENT_DIR)):
    full_path = os.path.join(PARENT_DIR, subdir)
    if not os.path.isdir(full_path):
        continue

    # Extract tilt1 and tilt2 values from folder name
    match = re.match(r'tilt1_([\-\d.]+)_tilt2_([\-\d.]+)', subdir)
    if not match:
        continue

    tilt1 = float(match.group(1))
    tilt2 = float(match.group(2))
    tilt1_vals.add(tilt1)
    tilt2_vals.add(tilt2)

    movie_path = os.path.join(full_path, f'movie{seg1}_{seg2}.fits')
    frame_path = os.path.join(full_path, 'apapane_frame.fits')
    if not os.path.isfile(movie_path) or not os.path.isfile(frame_path):
        print(f"[SKIP] Missing files in {subdir}")
        continue

    try:
        with fits.open(movie_path) as hdul:
            movie = hdul['MOVIE'].data
            opd = hdul['METADATA'].data['OPD']
    except Exception as e:
        print(f"[ERROR] Could not read FITS in {subdir}: {e}")
        continue

    # Mask OPD
    mask = (opd >= OPD_RANGE[0]) & (opd <= OPD_RANGE[1])
    if not np.any(mask):
        continue
    movie = movie[mask]
    opd = opd[mask]

    # Sum over cropped region
    summed_flux = movie[:, TOP:BOTTOM, :].sum(axis=(1, 2))
    normalised_flux = summed_flux/ np.max(summed_flux)

    # Prepare dark flux if available
    if dark_movie is not None and dark_opd is not None:
        dark_mask = (dark_opd >= OPD_RANGE[0]) & (dark_opd <= OPD_RANGE[1])
        dark_flux = dark_movie[dark_mask, TOP:BOTTOM, :].sum(axis=(1, 2))
        dark_flux_norm = dark_flux/ np.max(summed_flux)
        dark_opd_crop = dark_opd[dark_mask]
    else:
        dark_flux_norm = None
        dark_opd_crop = None

    # Fit sin^2 model
    try:
        wavelength_um = 1.6  # Approx central wavelength in microns
        initial_freq = 1 / wavelength_um
        p0 = [1.0, initial_freq, np.mean(opd), 0.1]
        popt, _ = curve_fit(sin_squared_model, opd, normalised_flux, p0=p0)
        A_fit, f_fit, x0_fit, C_fit = popt

        # Evaluate fit to find true minimum
        opd_dense = np.linspace(opd.min(), opd.max(), 1000)
        fit_curve = sin_squared_model(opd_dense, *popt)
        min_flux = np.min(fit_curve)
        min_opd = opd_dense[np.argmin(fit_curve)]

    except Exception as e:
        print(f"[WARN] Fit failed in {subdir}: {e}")
        min_flux = np.min(normalised_flux)
        min_opd = opd[np.argmin(normalised_flux)]

    # Compute photometric flux ratio from apapane_frame
    try:
        with fits.open(frame_path) as f:
            frame = f[0].data
        peak_11 = PEAKS[0]
        peak_31 = PEAKS[2]
        null = PEAKS[4]
        row_11 = VERTICAL_OFFSETS[0]
        row_31 = VERTICAL_OFFSETS[2]
        row_null = VERTICAL_OFFSETS[4]

        # Cropped region for wavelength
        flux_11 = np.sum(frame[TOP+row_11:BOTTOM+row_11, peak_11-BOX_HALFWIDTH:peak_11+BOX_HALFWIDTH])
        flux_31 = np.sum(frame[TOP+row_31:BOTTOM+row_31, peak_31-BOX_HALFWIDTH:peak_31+BOX_HALFWIDTH])
        flux_ratio = (flux_11 / flux_31) * 100 if flux_31 != 0 else np.nan

        # Full vertical integration
        flux_11_full = np.sum(frame[row_11:row_11+PHOTOMETRY_DEPTH, peak_11-BOX_HALFWIDTH:peak_11+BOX_HALFWIDTH])
        flux_31_full = np.sum(frame[row_31:row_31+PHOTOMETRY_DEPTH, peak_31-BOX_HALFWIDTH:peak_31+BOX_HALFWIDTH])
        flux_ratio_full = (flux_11_full / flux_31_full) * 100 if flux_31_full != 0 else np.nan

        # Full vertical integration
        flux_11_full = np.sum(frame[row_11:row_11+PHOTOMETRY_DEPTH, peak_11-BOX_HALFWIDTH:peak_11+BOX_HALFWIDTH])
        flux_null_full = np.sum(frame[row_null:row_null+PHOTOMETRY_DEPTH, null-BOX_HALFWIDTH:null+BOX_HALFWIDTH])
        flux_ratio_null = (flux_11_full / flux_null_full) * 100 if flux_null_full != 0 else np.nan



    except Exception as e:
        print(f"[WARN] Failed photometry in {subdir}: {e}")
        flux_ratio = np.nan

    results.append({
        'tilt1': tilt1,
        'tilt2': tilt2,
        'min_flux': min_flux,
        'min_opd': min_opd,
        'flux_ratio': flux_ratio,
        'flux_ratio_full': flux_ratio_full,
        'flux_ratio_null': flux_ratio_null,
        'folder': subdir
    })

    if SAVE_NULLSCAN_PLOT:
        plt.figure()
        plt.plot(opd, normalised_flux, 'o', color='C0', label='Data')
        if 'fit_curve' in locals():
            plt.plot(opd_dense, fit_curve, '-', color='k', label='Fit')
            plt.axvline(min_opd, color='red', linestyle='--', label=f"Min = {min_flux:.3f} at {min_opd:.3f} μm")
        if dark_flux_norm is not None:
            plt.plot(dark_opd_crop, dark_flux_norm, color='black', linestyle='--', label='Dark')
        plt.title(f"Nullscan: {subdir}")
        plt.xlabel("OPD (μm)")
        plt.ylabel("Normalised Flux")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(full_path, f'fitted_nullscan_{seg1}_{seg2}.png'))
        plt.close()

# -------- REPORT TOP 10 NULLS --------

results_sorted = sorted(results, key=lambda x: x['min_flux'])
print(f"\nTop 10 Deepest Nulls for Baseline {seg1}-{seg2} in OPD Range {OPD_RANGE}:")
for r in results_sorted[:10]:
    print(f"{r['folder']:35s} | Null depth: {r['min_flux']:.3f} at OPD = {r['min_opd']:.3f} | Photo Ratio: {r['flux_ratio']:.1f}% | Full Wavelength Photo Ratio: {r['flux_ratio_full']:.1f}% ")

# -------- HEATMAP --------

# Prepare 2D grid
tilt1_sorted = sorted(tilt1_vals)
tilt2_sorted = sorted(tilt2_vals)

heatmap = np.full((len(tilt2_sorted), len(tilt1_sorted)), np.nan)
textgrid = np.full((len(tilt2_sorted), len(tilt1_sorted)), '', dtype=object)

for r in results:
    i = tilt2_sorted.index(r['tilt2'])  # y-axis
    j = tilt1_sorted.index(r['tilt1'])  # x-axis
    heatmap[i, j] = r['min_flux']
    textgrid[i, j] = f"{r['flux_ratio']:.1f}%" if not np.isnan(r['flux_ratio']) else "--"

# Plot heatmap of null ratio (I_null / I_max)
null_ratio_map = np.full((len(tilt2_sorted), len(tilt1_sorted)), np.nan)
for r in results:
    i = tilt2_sorted.index(r['tilt2'])
    j = tilt1_sorted.index(r['tilt1'])
    null_ratio_map[i, j] = r['min_flux']

plt.figure(figsize=(8, 6))
im0 = plt.imshow(null_ratio_map, origin='lower', cmap='viridis',
                 extent=[min(tilt1_sorted), max(tilt1_sorted), min(tilt2_sorted), max(tilt2_sorted)],
                 aspect='auto')
plt.colorbar(im0, label='Null Ratio (I_min / I_max)')
plt.title(f"Null Ratio Heatmap: Baseline {seg1}-{seg2}")
plt.xlabel("Tilt 1 (Segment 11)")
plt.ylabel("Tilt 2 (Segment 31)")

plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(PARENT_DIR, f'null_ratio_heatmap_{seg1}_{seg2}.png'))
plt.show()


# Plot heatmap of visibility instead of null depth
visibility_map = np.full((len(tilt2_sorted), len(tilt1_sorted)), np.nan)
for r in results:
    i = tilt2_sorted.index(r['tilt2'])
    j = tilt1_sorted.index(r['tilt1'])
    I_min = r['min_flux']
    I_max = 1.0  # since normalised_flux was scaled to max=1
    V = (I_max - I_min) / (I_max + I_min)
    visibility_map[i, j] = V

plt.figure(figsize=(8, 6))
im = plt.imshow(visibility_map, origin='lower', cmap='viridis',
                extent=[min(tilt1_sorted), max(tilt1_sorted), min(tilt2_sorted), max(tilt2_sorted)],
                aspect='auto')
plt.colorbar(im, label='Visibility')
plt.title(f"Fringe Visibility Heatmap: Baseline {seg1}-{seg2}")
plt.xlabel("Tilt 1 (Segment 11)")
plt.ylabel("Tilt 2 (Segment 31)")

plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(PARENT_DIR, f'visibility_heatmap_{seg1}_{seg2}.png'))
plt.show()



# -------- SEPARATE HEATMAP FOR FLUX RATIO --------

flux_ratio_map = np.full((len(tilt2_sorted), len(tilt1_sorted)), np.nan)
for r in results:
    i = tilt2_sorted.index(r['tilt2'])
    j = tilt1_sorted.index(r['tilt1'])
    flux_ratio_map[i, j] = r['flux_ratio']

plt.figure(figsize=(8, 6))
im2 = plt.imshow(flux_ratio_map, origin='lower', cmap='plasma',
                 extent=[min(tilt1_sorted), max(tilt1_sorted), min(tilt2_sorted), max(tilt2_sorted)],
                 aspect='auto')
plt.colorbar(im2, label='Photometry Ratio (%): Segment 11 / Segment 31')
plt.title(f"Photometric Flux Ratio Heatmap: Baseline {seg1}-{seg2}")
plt.xlabel("Tilt 1 (Segment 11)")
plt.ylabel("Tilt 2 (Segment 31)")

plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(PARENT_DIR, f'photometry_ratio_heatmap_{seg1}_{seg2}.png'))
plt.show()


# -------- EXTRA HEATMAP FOR FULL-RANGE FLUX RATIO --------

flux_ratio_full_map = np.full((len(tilt2_sorted), len(tilt1_sorted)), np.nan)
for r in results:
    i = tilt2_sorted.index(r['tilt2'])
    j = tilt1_sorted.index(r['tilt1'])
    flux_ratio_full_map[i, j] = r['flux_ratio_full']

plt.figure(figsize=(8, 6))
im3 = plt.imshow(flux_ratio_full_map, origin='lower', cmap='plasma',
                 extent=[min(tilt1_sorted), max(tilt1_sorted), min(tilt2_sorted), max(tilt2_sorted)],
                 aspect='auto')
plt.colorbar(im3, label='Photometry Ratio (%): Segment 11 / Segment 31 (full height)')
plt.title(f"Full-Height Photometric Flux Ratio Heatmap: Baseline {seg1}-{seg2}")
plt.xlabel("Tilt 1 (Segment 11)")
plt.ylabel("Tilt 2 (Segment 31)")

plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(PARENT_DIR, f'photometry_ratio_fullrange_heatmap_{seg1}_{seg2}.png'))
plt.show()



# -------- EXTRA HEATMAP FOR FULL-RANGE FLUX RATIO --------

flux_ratio_null_map = np.full((len(tilt2_sorted), len(tilt1_sorted)), np.nan)
for r in results:
    i = tilt2_sorted.index(r['tilt2'])
    j = tilt1_sorted.index(r['tilt1'])
    flux_ratio_null_map[i, j] = r['flux_ratio_null']

plt.figure(figsize=(8, 6))
im3 = plt.imshow(flux_ratio_null_map, origin='lower', cmap='plasma',
                 extent=[min(tilt1_sorted), max(tilt1_sorted), min(tilt2_sorted), max(tilt2_sorted)],
                 aspect='auto')
plt.colorbar(im3, label='Photometry Ratio (%): Segment 11 / null (full height)')
plt.title(f"Full-Height Photometric Flux Ratio Heatmap: Baseline {seg1}-{seg2}")
plt.xlabel("Tilt 1 (Segment 11)")
plt.ylabel("Tilt 2 (Segment 31)")

plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(PARENT_DIR, f'photometry_ratio_null_fullrange_heatmap_{seg1}_{seg2}.png'))
plt.show()
