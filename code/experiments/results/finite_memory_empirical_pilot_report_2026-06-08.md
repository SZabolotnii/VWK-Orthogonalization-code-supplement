# Finite-Memory Empirical-Moment Pilot Report

*Дата: 2026-06-08 | reps=40 | n=1200 | memory=3 | order=2 | noise_sd=0.25 | seed=20260608*

## Summary

Gate definition: a second-order finite-memory VWK system should keep the Gaussian control neutral, show a misspecified Wiener penalty on asymmetric finite-moment input, and remain stable when moments are estimated empirically from the input sample.

- Gaussian control W/oracle = `1.000`.
- Centered-exponential skew W/oracle = `9.506`.
- Centered-exponential skew W/empirical = `12.070`.
- Monomial LS and Huber are guardrails because they fit the same finite polynomial span directly.

## Results

| Regime | Oracle VWK MSE | Empirical VWK MSE | Wiener MSE | Monomial LS MSE | Huber MSE | W/oracle | W/empirical | Emp/oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gaussian_control | 0.0255584 | 0.0188696 | 0.0255584 | 0.000560223 | 0.00058584 | 1.000 | 1.354 | 0.738 |
| centered_exponential_skew | 0.0800451 | 0.0630408 | 0.760901 | 0.000524936 | 0.000545009 | 9.506 | 12.070 | 0.788 |
| uniform_platykurtic | 0.0113695 | 0.00908261 | 0.0113695 | 0.000509997 | 0.00054773 | 1.000 | 1.252 | 0.799 |
| symmetric_contaminated | 0.11783 | 0.0729305 | 0.11783 | 0.000534706 | 0.000552102 | 1.000 | 1.616 | 0.619 |

## Reading

The empirical-moment VWK estimator remains stable in this finite-memory gate. As in the one-lag H4 grid, the substantive W/V advantage is asymmetric finite-moment only; symmetric controls do not justify a symmetric non-Gaussian advantage claim.
Direct monomial LS and Huber guardrails are intentionally strong and should be reported as span-fitting baselines, not as the H4 target.
