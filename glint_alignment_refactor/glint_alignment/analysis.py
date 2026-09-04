from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


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


def _null_model(x, c0, c1, amplitude, center, sigma, period, phase):
    envelope = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    return c0 + c1 * x + amplitude * envelope * np.cos(2 * np.pi * x / period + phase)


def fit_null_envelope(opd: np.ndarray, metric: np.ndarray, max_candidates: int = 10) -> dict[str, object]:
    """Fit a coherence-envelope sinusoid and return measured-validated trough candidates."""
    x = np.asarray(opd, dtype=float)
    y = np.asarray(metric, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    span = float(np.ptp(x))
    smooth = gaussian_filter1d(y, sigma=max(0.8, len(y) / 250))
    c0 = float(np.median(y))
    amp0 = max(float(np.ptp(y)) / 2, 1e-6)
    center0 = float(x[np.argmin(smooth)])
    dx = float(np.median(np.diff(x)))
    # FFT supplies a useful period seed while excluding the DC component.
    centered = smooth - np.mean(smooth)
    freqs = np.fft.rfftfreq(len(x), d=max(dx, 1e-9))
    spectrum = np.abs(np.fft.rfft(centered))
    valid = (freqs > 1 / max(span, dx)) & (freqs < 1 / max(4 * dx, 1e-9))
    period0 = float(1 / freqs[np.argmax(np.where(valid, spectrum, 0.0))]) if np.any(valid) else span / 6
    p0 = [c0, 0.0, amp0, center0, max(span / 3, dx * 4), np.clip(period0, 4 * dx, span), 0.0]
    lower = [float(np.min(y) - np.ptp(y)), -np.inf, -2 * max(np.ptp(y), 1e-6), x[0], 2 * dx, 3 * dx, -2 * np.pi]
    upper = [float(np.max(y) + np.ptp(y)), np.inf, 2 * max(np.ptp(y), 1e-6), x[-1], span * 2, span, 2 * np.pi]
    fit_ok = True
    warning = ""
    try:
        popt, _ = curve_fit(_null_model, x, y, p0=p0, bounds=(lower, upper), maxfev=60000)
        dense_x = np.linspace(x[0], x[-1], max(4000, len(x) * 20))
        dense_y = _null_model(dense_x, *popt)
    except Exception as error:
        fit_ok = False
        warning = str(error)
        popt = None
        dense_x = np.linspace(x[0], x[-1], max(4000, len(x) * 20))
        dense_y = np.interp(dense_x, x, smooth)
    trough_idx, _ = find_peaks(-dense_y, distance=max(2, int(len(dense_x) * 3 * dx / max(span, dx))))
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
        measured_metric = float(np.min(local_y))
        candidates.append(
            {
                "opd_um": refined_x,
                "delta_piston_um": refined_x / 2.0,
                "metric": measured_metric,
                "fit_metric": float(dense_y[index]),
            }
        )
    candidates.sort(key=lambda item: item["metric"])
    best = candidates[0]
    return {
        "fit_ok": fit_ok,
        "fit_warning": warning,
        "parameters": None if popt is None else [float(v) for v in popt],
        "fit_opd": dense_x,
        "fit_metric": dense_y,
        "candidates": candidates[:max_candidates],
        "best": best,
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
