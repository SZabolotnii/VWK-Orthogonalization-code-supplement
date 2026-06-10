# Finite-Memory Empirical-Moment Production Report

*Дата: 2026-06-08 | reps=150 | n=2500 | memory=3 | order=2 | noise_sd=0.25 | seed=20260608*

## Summary

Gate definition: a second-order finite-memory VWK system should keep the Gaussian control neutral, show a misspecified Wiener penalty on asymmetric finite-moment input, and remain stable when moments are estimated empirically from the input sample.

- Gaussian control W/oracle = `1.000`.
- Centered-exponential skew W/oracle = `13.457`.
- Centered-exponential skew W/empirical = `18.325`.
- Monomial LS and Huber are guardrails because they fit the same finite polynomial span directly.

## Results

| Regime | Oracle VWK MSE | Empirical VWK MSE | Wiener MSE | Monomial LS MSE | Huber MSE | W/oracle | W/empirical | Emp/oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gaussian_control | 0.0132205 | 0.00967447 | 0.0132205 | 0.000254143 | 0.000263734 | 1.000 | 1.367 | 0.732 |
| centered_exponential_skew | 0.0505975 | 0.0371562 | 0.6809 | 0.000251733 | 0.000261596 | 13.457 | 18.325 | 0.734 |
| uniform_platykurtic | 0.00805996 | 0.00565797 | 0.00805996 | 0.000247122 | 0.000260141 | 1.000 | 1.425 | 0.702 |
| symmetric_contaminated | 0.0519373 | 0.0327849 | 0.0519373 | 0.000244723 | 0.000253648 | 1.000 | 1.584 | 0.631 |

## Reading

The empirical-moment VWK estimator remains stable in this finite-memory gate. As in the one-lag H4 grid, the substantive W/V advantage is asymmetric finite-moment only; symmetric controls do not justify a symmetric non-Gaussian advantage claim.
Direct monomial LS and Huber guardrails are intentionally strong and should be reported as span-fitting baselines, not as the H4 target.
