from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, hilbert


def threshold_center_of_mass(
    image: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    threshold_fraction: float = 0.60,
) -> dict[str, float | str | bool]:
    """Background-subtracted COM, intended for the blurred hexagonal XY lobes."""
    z = np.asarray(image, dtype=float)
    z = z - np.nanmin(z)
    peak = float(np.nanmax(z))
    mask = np.isfinite(z) & (z >= threshold_fraction * peak) if peak > 0 else np.zeros_like(z, bool)
    weights = np.where(mask, z, 0.0)
    total = float(weights.sum())
    if total <= 0:
        iy, ix = np.unravel_index(np.nanargmax(z), z.shape)
        method = "maximum"
        x0, y0 = float(x_values[ix]), float(y_values[iy])
    else:
        xx, yy = np.meshgrid(x_values, y_values)
        x0 = float((weights * xx).sum() / total)
        y0 = float((weights * yy).sum() / total)
        method = "thresholded center of mass"
    boundary = bool(
        x0 <= x_values[0] + 0.5 * (x_values[1] - x_values[0])
        or x0 >= x_values[-1] - 0.5 * (x_values[-1] - x_values[-2])
        or y0 <= y_values[0] + 0.5 * (y_values[1] - y_values[0])
        or y0 >= y_values[-1] - 0.5 * (y_values[-1] - y_values[-2])
    )
    return {"x": x0, "y": y0, "method": method, "boundary": boundary, "peak": peak}


def _elliptical_gaussian(coords, offset, amplitude, x0, y0, sx, sy, angle):
    x, y = coords
    ca, sa = np.cos(angle), np.sin(angle)
    xr = ca * (x - x0) + sa * (y - y0)
    yr = -sa * (x - x0) + ca * (y - y0)
    return offset + amplitude * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))


def fit_elliptical_gaussian(
    image: np.ndarray, x_values: np.ndarray, y_values: np.ndarray
) -> dict[str, float | str | bool]:
    """Bounded rotated Gaussian with independent x/y widths and COM fallback."""
    z = np.asarray(image, dtype=float)
    xx, yy = np.meshgrid(x_values, y_values)
    fallback = threshold_center_of_mass(z, x_values, y_values, 0.60)
    dx = max(float(np.ptp(x_values)), float(np.diff(x_values).mean()))
    dy = max(float(np.ptp(y_values)), float(np.diff(y_values).mean()))
    p0 = [float(np.nanmin(z)), float(np.ptp(z)), fallback["x"], fallback["y"], dx / 4, dy / 4, 0.0]
    lower = [
        float(np.nanmin(z) - np.ptp(z)),
        0.0,
        float(x_values[0]),
        float(y_values[0]),
        max(float(np.diff(x_values).mean()) / 3, 1e-6),
        max(float(np.diff(y_values).mean()) / 3, 1e-6),
        -np.pi / 2,
    ]
    upper = [
        float(np.nanmax(z)),
        max(float(np.ptp(z)) * 5, 1.0),
        float(x_values[-1]),
        float(y_values[-1]),
        dx * 2,
        dy * 2,
        np.pi / 2,
    ]
    try:
        popt, _ = curve_fit(
            _elliptical_gaussian,
            (xx.ravel(), yy.ravel()),
            z.ravel(),
            p0=p0,
            bounds=(lower, upper),
            maxfev=30000,
        )
        fitted = _elliptical_gaussian((xx, yy), *popt)
        ss_res = float(np.sum((z - fitted) ** 2))
        ss_tot = float(np.sum((z - np.mean(z)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        boundary = bool(
            np.isclose(popt[2], x_values[0], atol=np.diff(x_values).mean())
            or np.isclose(popt[2], x_values[-1], atol=np.diff(x_values).mean())
            or np.isclose(popt[3], y_values[0], atol=np.diff(y_values).mean())
            or np.isclose(popt[3], y_values[-1], atol=np.diff(y_values).mean())
        )
        if not np.isfinite(r2) or r2 < 0.20:
            raise RuntimeError(f"poor Gaussian fit R2={r2:.3f}")
        return {
            "x": float(popt[2]),
            "y": float(popt[3]),
            "sigma_x": float(popt[4]),
            "sigma_y": float(popt[5]),
            "angle_rad": float(popt[6]),
            "r2": r2,
            "method": "rotated elliptical Gaussian",
            "boundary": boundary,
        }
    except Exception as error:
        fallback["fit_warning"] = str(error)
        return fallback


def agreement(estimates: list[dict], max_distance: float) -> dict[str, object]:
    points = np.asarray([[item["x"], item["y"]] for item in estimates], dtype=float)
    average = points.mean(axis=0)
    distances = np.linalg.norm(points - average, axis=1)
    return {
        "x": float(average[0]),
        "y": float(average[1]),
        "distances": distances.tolist(),
        "max_distance": float(max_distance),
        "passed": bool(np.all(distances <= max_distance)),
    }


def hotspot(image: np.ndarray, crop: list[int] | tuple[int, int, int, int], fraction: float = 0.85) -> np.ndarray:
    """Return full-frame [x,y] hotspot using a high-threshold centroid."""
    y0, y1, x0, x1 = map(int, crop)
    z = np.asarray(image[y0:y1, x0:x1], dtype=float)
    z -= np.nanmedian(z)
    peak = float(np.nanmax(z))
    mask = z >= fraction * peak
    weights = np.where(mask, np.maximum(z, 0.0), 0.0)
    if weights.sum() <= 0:
        iy, ix = np.unravel_index(np.nanargmax(z), z.shape)
    else:
        iy_grid, ix_grid = np.indices(z.shape)
        ix = float((weights * ix_grid).sum() / weights.sum())
        iy = float((weights * iy_grid).sum() / weights.sum())
    return np.asarray([float(ix) + x0, float(iy) + y0])


def _null_sinc_model(x, c0, c1, amplitude, center, coherence_width, period, phase):
    """Rectangular-band coherence envelope multiplied by a monochromatic fringe.

    ``numpy.sinc(q)`` is ``sin(pi*q)/(pi*q)``, so ``coherence_width`` is the
    distance from the packet centre to the first zero of the envelope.
    """
    envelope = np.sinc((x - center) / coherence_width)
    return c0 + c1 * x + amplitude * envelope * np.sin(2 * np.pi * (x - center) / period + phase)


def fit_null_envelope(
    opd: np.ndarray,
    metric: np.ndarray,
    max_candidates: int = 10,
    *,
    period_bounds_um: tuple[float, float] = (1.2, 2.0),
    min_r2: float = 0.35,
) -> dict[str, object]:
    """Fit a sinc-envelope sinusoid and return measured-validated troughs.

    A successful numerical optimiser is not by itself a valid fringe packet.
    ``fit_ok`` therefore also requires the configured minimum R-squared.  The
    caller must not apply a piston solution when either baseline fails it.
    """
    raw_x = np.asarray(opd, dtype=float).ravel()
    raw_y = np.asarray(metric, dtype=float).ravel()
    if raw_x.shape != raw_y.shape:
        raise ValueError("OPD and null metric arrays must have identical shapes")
    finite = np.isfinite(raw_x) & np.isfinite(raw_y)
    x, y = raw_x[finite], raw_y[finite]
    if len(x) < 7:
        raise ValueError("At least seven finite null samples are required")
    order = np.argsort(x)
    x, y = x[order], y[order]
    if np.any(np.diff(x) <= 0):
        raise ValueError("Null OPD coordinates must be unique")

    span = float(np.ptp(x))
    dx = float(np.median(np.diff(x)))
    if not np.isfinite(span) or span <= 0 or not np.isfinite(dx) or dx <= 0:
        raise ValueError("Null OPD scan must have a positive span and spacing")
    period_low, period_high = map(float, period_bounds_um)
    period_low = max(period_low, 3.0 * dx)
    period_high = min(period_high, span)
    if not 0 < period_low < period_high:
        raise ValueError("period_bounds_um must overlap the sampled OPD range")
    if not 0.0 <= float(min_r2) <= 1.0:
        raise ValueError("min_r2 must lie in [0, 1]")

    smooth = gaussian_filter1d(y, sigma=max(0.8, len(y) / 250))
    slope0, intercept0 = np.polyfit(x, smooth, 1)
    detrended = smooth - (intercept0 + slope0 * x)
    analytic_amplitude = gaussian_filter1d(np.abs(hilbert(detrended)), sigma=max(1.0, len(y) / 100))
    center0 = float(x[int(np.argmax(analytic_amplitude))])
    amplitude0 = max(float(np.ptp(detrended)) / 2.0, 1e-9)

    # Restrict the FFT seed to the physical wavelength interval.  Without this
    # restriction a slow drift, such as the failed 11-31 trace seen on sky, can
    # be mistaken for a very long fringe period.
    freqs = np.fft.rfftfreq(len(x), d=dx)
    spectrum = np.abs(np.fft.rfft(detrended))
    valid_frequency = (freqs >= 1.0 / period_high) & (freqs <= 1.0 / period_low)
    if np.any(valid_frequency):
        selected = int(np.argmax(np.where(valid_frequency, spectrum, -np.inf)))
        period0 = float(1.0 / freqs[selected])
    else:
        period0 = 0.5 * (period_low + period_high)

    y_span = max(float(np.ptp(y)), 1e-9)
    width_low = max(period_low, 3.0 * dx)
    width_high = max(4.0 * span, width_low * 1.1)
    lower = [
        float(np.min(y) - y_span),
        -10.0 * y_span / span,
        0.0,
        float(x[0]),
        width_low,
        period_low,
        -2.0 * np.pi,
    ]
    upper = [
        float(np.max(y) + y_span),
        10.0 * y_span / span,
        5.0 * y_span,
        float(x[-1]),
        width_high,
        period_high,
        2.0 * np.pi,
    ]

    # A few starts avoid locking the packet centre to one arbitrary fringe
    # minimum.  The Hilbert-envelope maximum is the preferred centre seed.
    center_seeds = [center0, float(x[np.argmin(smooth)]), float(np.median(x))]
    width_seeds = [max(span / 2.0, width_low), max(span, width_low)]
    phase_seeds = [-np.pi / 2.0, 0.0, np.pi / 2.0]
    best_fit: tuple[float, np.ndarray] | None = None
    errors: list[str] = []
    for center_seed in center_seeds:
        for width_seed in width_seeds:
            for phase_seed in phase_seeds:
                p0 = [
                    float(intercept0),
                    float(slope0),
                    amplitude0,
                    float(np.clip(center_seed, x[0], x[-1])),
                    float(np.clip(width_seed, width_low, width_high)),
                    float(np.clip(period0, period_low, period_high)),
                    float(phase_seed),
                ]
                try:
                    popt, _ = curve_fit(
                        _null_sinc_model,
                        x,
                        y,
                        p0=p0,
                        bounds=(lower, upper),
                        maxfev=30000,
                    )
                    residual_sum = float(np.sum((y - _null_sinc_model(x, *popt)) ** 2))
                    if best_fit is None or residual_sum < best_fit[0]:
                        best_fit = (residual_sum, popt)
                except Exception as error:
                    errors.append(str(error))

    dense_x = np.linspace(x[0], x[-1], max(4000, len(x) * 20))
    optimiser_ok = best_fit is not None
    if optimiser_ok:
        ss_res, popt = best_fit
        fitted_samples = _null_sinc_model(x, *popt)
        dense_y = _null_sinc_model(dense_x, *popt)
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        total_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        linear_slope, linear_intercept = np.polyfit(x, y, 1)
        ss_linear = float(np.sum((y - (linear_intercept + linear_slope * x)) ** 2))
        # Judge whether the coherent packet explains structure beyond an
        # arbitrary straight drift.  A drifting, non-fringing trace can have a
        # deceptively high ordinary R-squared otherwise.
        r2 = float(1.0 - ss_res / ss_linear) if ss_linear > 0 else 0.0
        residual = y - fitted_samples
        rmse = float(np.sqrt(np.mean(residual**2)))
        model_baseline = popt[0] + popt[1] * x
        modulation_peak_to_peak = float(np.ptp(fitted_samples - model_baseline))
        period = float(popt[5])
        envelope_dense = np.abs(np.sinc((dense_x - popt[3]) / popt[4]))
    else:
        popt = None
        r2 = float("-inf")
        total_r2 = float("-inf")
        rmse = float(np.sqrt(np.mean((y - smooth) ** 2)))
        modulation_peak_to_peak = 0.0
        period = period0
        envelope_dense = np.ones_like(dense_x)
        dense_y = np.interp(dense_x, x, smooth)

    fit_ok = bool(optimiser_ok and np.isfinite(r2) and r2 >= float(min_r2))
    if not optimiser_ok:
        warning = "Sinc × sine optimiser failed" + (f": {errors[-1]}" if errors else "")
    elif not fit_ok:
        warning = f"Poor sinc × sine fit: R²={r2:.3f} is below {float(min_r2):.3f}"
    else:
        warning = ""

    minimum_distance = max(2, int(0.55 * period / span * len(dense_x)))
    trough_idx, _ = find_peaks(-dense_y, distance=minimum_distance)
    # Do not use sidelobes in which the fitted coherence amplitude has almost
    # vanished.  This still retains multiple periodic troughs near the packet.
    trough_idx = np.asarray([index for index in trough_idx if envelope_dense[index] >= 0.10], dtype=int)
    if len(trough_idx) == 0:
        trough_idx = np.asarray([int(np.argmin(dense_y))])
    candidates = []
    for index in trough_idx:
        candidate_x = float(dense_x[index])
        nearest = int(np.argmin(np.abs(x - candidate_x)))
        lo, hi = max(0, nearest - 1), min(len(x), nearest + 2)
        local_x, local_y = x[lo:hi], y[lo:hi]
        refined_x = candidate_x
        if len(local_x) == 3:
            q = np.polyfit(local_x, local_y, 2)
            vertex = -q[1] / (2 * q[0]) if q[0] > 0 else candidate_x
            if local_x.min() <= vertex <= local_x.max():
                refined_x = float(vertex)
        candidates.append(
            {
                "opd_um": refined_x,
                "delta_piston_um": refined_x / 2.0,
                "metric": float(np.min(local_y)),
                "fit_metric": float(dense_y[index]),
                "envelope_fraction": float(envelope_dense[index]),
            }
        )
    candidates.sort(key=lambda item: item["metric"])
    return {
        "model": "sinc × sine",
        "fit_ok": fit_ok,
        "fit_warning": warning,
        "r2": r2,
        "total_r2": total_r2,
        "rmse": rmse,
        "normalized_rmse": float(rmse / y_span),
        "modulation_peak_to_peak": modulation_peak_to_peak,
        "finite_sample_fraction": float(np.count_nonzero(finite) / len(raw_x)),
        "parameter_names": [
            "offset",
            "slope",
            "amplitude",
            "packet_center_um",
            "coherence_first_zero_um",
            "period_um",
            "phase_rad",
        ],
        "parameters": None if popt is None else [float(value) for value in popt],
        "fit_opd": dense_x,
        "fit_metric": dense_y,
        "candidates": candidates[:max_candidates],
        "best": candidates[0],
    }


def solve_common_pistons(
    candidates_1120: list[dict],
    candidates_1131: list[dict],
    minimum: float = -1.3,
    maximum: float = 1.3,
) -> dict[str, object]:
    """Choose periodic minima with segments 20 and 31 symmetric about zero.

    The measured convention is ``delta_11_20 = p11 - p20`` (and likewise for
    segment 31).  Requiring ``p20 = -p31`` therefore fixes all three absolute
    pistons for each pair of candidate differential pistons.
    """
    feasible: list[dict] = []
    scale20 = max((c["metric"] for c in candidates_1120), default=1.0) or 1.0
    scale31 = max((c["metric"] for c in candidates_1131), default=1.0) or 1.0
    for c20 in candidates_1120:
        d20 = float(c20["delta_piston_um"])
        for c31 in candidates_1131:
            d31 = float(c31["delta_piston_um"])
            p11 = 0.5 * (d20 + d31)
            p20 = 0.5 * (d31 - d20)
            p31 = -p20
            pistons = {11: p11, 20: p20, 31: p31}
            if any(p < minimum or p > maximum for p in pistons.values()):
                continue
            margin = min(min(p - minimum, maximum - p) for p in pistons.values())
            depth_score = float(c20["metric"]) / scale20 + float(c31["metric"]) / scale31
            feasible.append(
                {
                    "pistons_um": {str(k): float(v) for k, v in pistons.items()},
                    "delta_11_20_um": d20,
                    "delta_11_31_um": d31,
                    "symmetric_about_zero": True,
                    "margin_um": float(margin),
                    "depth_score": depth_score,
                    # Null depth is the primary criterion; actuator margin only
                    # breaks near-ties between periodic trough combinations.
                    "score": depth_score - 0.02 * margin,
                }
            )
    if not feasible:
        return {"feasible": False, "reason": "No common piston solution lies inside all three ranges."}
    feasible.sort(key=lambda item: item["score"])
    return {"feasible": True, **feasible[0], "alternatives_considered": len(feasible)}
