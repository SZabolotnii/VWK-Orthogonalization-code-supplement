# Synthetic H4 Production Report

*Дата: 2026-06-08 | reps=200 | n=2000 | noise_sd=0.25 | seed=20260608*

## Summary

Gate definition: VWK must beat the misspecified Gaussian/Wiener projection in at least one finite-moment non-Gaussian regime, while the Gaussian control should show no artificial gain.

- Gaussian control W/V = `1.000`.
- Best non-Gaussian W/V = `32.362` in `centered_exponential_skew`.
- Huber monomial is reported as a guardrail baseline, not as the H4 pass criterion.

## Results

| Regime | VWK MSE | Wiener MSE | Huber MSE | W/V | VWK beta2 coverage | VWK all-beta coverage |
|---|---:|---:|---:|---:|---:|---:|
| gaussian_control | 0.0032483 | 0.0032483 | 9.80348e-05 | 1.000 | 0.950 | 0.950 |
| centered_exponential_skew | 0.0159485 | 0.516133 | 9.37327e-05 | 32.362 | 0.875 | 0.915 |
| uniform_platykurtic | 0.00162222 | 0.00162222 | 0.000104216 | 1.000 | 0.925 | 0.938 |
| symmetric_contaminated | 0.0160327 | 0.0160327 | 9.92007e-05 | 1.000 | 0.885 | 0.912 |

## Reading

Current H4 support is asymmetric finite-moment only: the centered-exponential skew regime passes strongly, while Gaussian, platykurtic uniform, and symmetric contaminated controls do not show a W/V advantage in this setup.
Do not use this artifact to claim universal prediction superiority over robust or likelihood baselines.
