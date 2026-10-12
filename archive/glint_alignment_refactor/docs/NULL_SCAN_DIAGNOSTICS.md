# Null piston scan: model and telescope diagnostics

## What the first telescope trace showed

- **11–20:** a high-contrast fringe packet is visible. Its strongest modulation lies mainly on the negative-OPD half of the scan; this is physically possible and is not by itself a command error.
- **11–31:** the samples do not show a repeatable coherent sinusoidal packet. The old fitter followed the slow trend and nevertheless supplied a candidate. That candidate was not safe to apply.

Changing a Gaussian envelope to sinc is physically appropriate for a roughly rectangular spectral band, but it cannot manufacture coherence in a poor measurement. Version 0.4.8 therefore couples the new model to mandatory data-quality rejection.

## Model

The fitted normalized null metric is

`c0 + c1*OPD + A*sinc((OPD-centre)/coherence_width)*sin(2*pi*(OPD-centre)/period + phase)`.

`coherence_width` is the distance from the packet centre to the first zero of the fitted sinc. The default accepted OPD period is 1.2–2.0 µm. The reported R² measures the extra structure explained by the coherent model relative to a linear-drift-only model; a sloping trace therefore does not pass merely because the fit reproduces its slope.

## Command and acquisition sequence

For each of 11–20 and 11–31, and for every repeat:

1. Put the first segment at minimum piston and the second at maximum piston, retaining their current tip/tilt values.
2. Move the first segment minimum → maximum. At every point: freshly read `dmptt`, replace only this segment's three PTT values, publish the array, verify the selected row, wait `scans.null.settle_s`, then acquire APAPANE frames.
3. Keep the first segment at maximum and move the second from maximum-minus-one-step → minimum. The duplicate zero-OPD endpoint is omitted.
4. Save `OPD = 2 * (piston_first - piston_second)`.

This order and the one-segment-change invariant are regression-tested. The SHM update deliberately matches the working standalone `shmDMcontrol.DM.set_segment` read/modify/write behavior without importing the legacy module.

## Apply safeguard

Apply remains disabled unless both baselines:

- produce a sinc × sine fit above `scans.null.min_fit_r2`;
- keep the fraction of non-positive or non-finite `FT1 + FT2` samples below `scans.null.max_invalid_fraction`; and
- yield a common three-segment solution within the configured piston range.

The review shows every repetition, their mean, the fitted curve, fit R² and median `FT1 + FT2`. A rejected scan displays a red rejection message and cannot send piston values to the DM.

## What is saved for diagnosis

The JSON estimate for each baseline contains the raw `null_flux`, `ft1_flux`, `ft2_flux`, denominator, normalized metric, R² and fit parameters. The FITS/NPZ array product also contains the individual metric repetitions and the four diagnostic-flux arrays. This permits separation of four common failure modes:

| Symptom | Most likely check |
| --- | --- |
| Raw null flux is smooth but normalized metric scatters | FT1/FT2 box flux is low, negative after dark subtraction, or varying strongly. |
| Raw null and FT fluxes all vary together | Source/seeing/camera variation; normalization may still recover the packet. |
| Only commands involving segment 31 fail to modulate | Verify segment 31 PTT is accepted at each piston; the DM's allowed tip/tilt depends on piston. SHM write readback alone may not prove the physical command was accepted by the downstream DM control. |
| Both baselines lag or repeat poorly | Increase `scans.null.settle_s`; v0.4.8 restores 0.1 s as the default. |
| N3 is bright but never fringes | Verify N3/11–31, FT1 and FT2 detector boxes against the live APAPANE image and confirm the correct beams are present. |
| Packet reaches a scan boundary | Shift or widen the piston start/range without exceeding the validated per-segment PTT region. |

For the next telescope run, first repeat 11–31 with the 0.1 s delay and inspect its saved raw N3, FT1 and FT2 arrays. If those are stable but there is still no fringe packet, test a few segment-31 piston commands manually while watching the DM controller's own range/error response and the live N3 signal.
