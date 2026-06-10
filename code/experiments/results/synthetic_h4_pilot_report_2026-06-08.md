# Synthetic H4 Pilot Report

*Дата: 2026-06-08 | reps=60 | n=800 | noise_sd=0.25 | seed=20260608*

## Summary

Gate definition: VWK must beat the misspecified Gaussian/Wiener projection in at least one finite-moment non-Gaussian regime, while the Gaussian control should show no artificial gain.

- Gaussian control W/V = `1.000`.
- Best non-Gaussian W/V = `17.966` in `centered_exponential_skew`.
- Huber monomial is reported as a guardrail baseline, not as the H4 pass criterion.

## Results

| Regime | VWK MSE | Wiener MSE | Huber MSE | W/V | VWK beta2 coverage | VWK all-beta coverage |
|---|---:|---:|---:|---:|---:|---:|
| gaussian_control | 0.00827246 | 0.00827246 | 0.000220225 | 1.000 | 0.917 | 0.917 |
| centered_exponential_skew | 0.0296987 | 0.533556 | 0.000248783 | 17.966 | 0.833 | 0.894 |
| uniform_platykurtic | 0.00337455 | 0.00337455 | 0.000278031 | 1.000 | 0.933 | 0.961 |
| symmetric_contaminated | 0.0467174 | 0.0467174 | 0.000279185 | 1.000 | 0.783 | 0.872 |

## Reading

Current H4 support is asymmetric finite-moment only: the centered-exponential skew regime passes strongly, while Gaussian, platykurtic uniform, and symmetric contaminated controls do not show a W/V advantage in this setup.
Do not use this artifact to claim universal prediction superiority over robust or likelihood baselines.
