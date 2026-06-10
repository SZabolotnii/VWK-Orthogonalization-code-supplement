# Sample-Efficiency Production Report

*Дата: 2026-06-08 | reps=80 | n_grid=[600, 1200, 2500, 5000] | memory=3 | order=2 | noise_sd=0.25 | seed=20260608*

## Summary

Gate definition: the finite-memory H4 advantage should persist across sample sizes rather than appearing only at one chosen n.

- Minimum centered-exponential W/oracle across n = `4.164`.
- Minimum centered-exponential W/empirical across n = `6.198`.
- Maximum Gaussian-control |W/oracle - 1| across n = `0.000`.

## Results

| Regime | n | Oracle VWK MSE | Empirical VWK MSE | Wiener MSE | W/oracle | W/empirical | Emp/oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| gaussian_control | 600 | 0.0553125 | 0.0385655 | 0.0553125 | 1.000 | 1.434 | 0.697 |
| centered_exponential_skew | 600 | 0.230673 | 0.154992 | 0.960582 | 4.164 | 6.198 | 0.672 |
| uniform_platykurtic | 600 | 0.0303566 | 0.0217716 | 0.0303566 | 1.000 | 1.394 | 0.717 |
| symmetric_contaminated | 600 | 0.134022 | 0.083556 | 0.134022 | 1.000 | 1.604 | 0.623 |
| gaussian_control | 1200 | 0.0268312 | 0.0173127 | 0.0268312 | 1.000 | 1.550 | 0.645 |
| centered_exponential_skew | 1200 | 0.103497 | 0.0637084 | 0.746282 | 7.211 | 11.714 | 0.616 |
| uniform_platykurtic | 1200 | 0.0144139 | 0.0107833 | 0.0144139 | 1.000 | 1.337 | 0.748 |
| symmetric_contaminated | 1200 | 0.0990032 | 0.0596263 | 0.0990032 | 1.000 | 1.660 | 0.602 |
| gaussian_control | 2500 | 0.0118809 | 0.00920795 | 0.0118809 | 1.000 | 1.290 | 0.775 |
| centered_exponential_skew | 2500 | 0.0486043 | 0.0367289 | 0.671141 | 13.808 | 18.273 | 0.756 |
| uniform_platykurtic | 2500 | 0.00774189 | 0.00584615 | 0.00774189 | 1.000 | 1.324 | 0.755 |
| symmetric_contaminated | 2500 | 0.0608994 | 0.0338566 | 0.0608994 | 1.000 | 1.799 | 0.556 |
| gaussian_control | 5000 | 0.00586174 | 0.00416239 | 0.00586174 | 1.000 | 1.408 | 0.710 |
| centered_exponential_skew | 5000 | 0.0227698 | 0.0151246 | 0.604379 | 26.543 | 39.960 | 0.664 |
| uniform_platykurtic | 5000 | 0.00330522 | 0.00259361 | 0.00330522 | 1.000 | 1.274 | 0.785 |
| symmetric_contaminated | 5000 | 0.0251835 | 0.0168113 | 0.0251835 | 1.000 | 1.498 | 0.668 |

## Reading

The sample-efficiency curve supports the same scoped H4 interpretation: asymmetric finite-moment inputs show a persistent misspecified-Wiener penalty, while the Gaussian control remains neutral.
This artifact is still synthetic finite-memory evidence, not a real-world or CF/3b claim.
