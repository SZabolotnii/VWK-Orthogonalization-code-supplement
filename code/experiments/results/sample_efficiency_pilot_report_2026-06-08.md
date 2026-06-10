# Sample-Efficiency Pilot Report

*Дата: 2026-06-08 | reps=30 | n_grid=[600, 1200] | memory=3 | order=2 | noise_sd=0.25 | seed=20260608*

## Summary

Gate definition: the finite-memory H4 advantage should persist across sample sizes rather than appearing only at one chosen n.

- Minimum centered-exponential W/oracle across n = `3.401`.
- Minimum centered-exponential W/empirical across n = `5.174`.
- Maximum Gaussian-control |W/oracle - 1| across n = `0.000`.

## Results

| Regime | n | Oracle VWK MSE | Empirical VWK MSE | Wiener MSE | W/oracle | W/empirical | Emp/oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| gaussian_control | 600 | 0.0506217 | 0.036658 | 0.0506217 | 1.000 | 1.381 | 0.724 |
| centered_exponential_skew | 600 | 0.370457 | 0.243501 | 1.25985 | 3.401 | 5.174 | 0.657 |
| uniform_platykurtic | 600 | 0.0302608 | 0.0184159 | 0.0302608 | 1.000 | 1.643 | 0.609 |
| symmetric_contaminated | 600 | 0.127448 | 0.0794532 | 0.127448 | 1.000 | 1.604 | 0.623 |
| gaussian_control | 1200 | 0.0286366 | 0.0173681 | 0.0286366 | 1.000 | 1.649 | 0.607 |
| centered_exponential_skew | 1200 | 0.0867362 | 0.0455284 | 0.714668 | 8.240 | 15.697 | 0.525 |
| uniform_platykurtic | 1200 | 0.0134585 | 0.0102111 | 0.0134585 | 1.000 | 1.318 | 0.759 |
| symmetric_contaminated | 1200 | 0.0814404 | 0.0478014 | 0.0814404 | 1.000 | 1.704 | 0.587 |

## Reading

The sample-efficiency curve supports the same scoped H4 interpretation: asymmetric finite-moment inputs show a persistent misspecified-Wiener penalty, while the Gaussian control remains neutral.
This artifact is still synthetic finite-memory evidence, not a real-world or CF/3b claim.
