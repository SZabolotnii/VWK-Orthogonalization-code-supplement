# Real-World VWK Screen

*Дата: 2026-06-08 | order=2 | min_n=200*

## Decision

candidate illustration available, but still not a theorem-supporting real-world claim

This screen is intentionally conservative. It tests scalar real-world illustrations from already-prepared local CSVs; it does not establish unbiased Volterra-kernel recovery because these datasets do not provide known Volterra kernels.

## Summary

- Screened candidates: `11`.
- Best W/V ratio: `4.093` for `sru_y1_dynamic`.
- Robust candidate illustrations: `4`.
- Wiener-gap-only diagnostics: `0`.

## Results

| Dataset | target | driver | n | skew | kurtosis | VWK MSE | Wiener MSE | LS MSE | Huber MSE | W/V | decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fremtpl2_severity_raw | ClaimAmount | DrivAge | 4000 | 32.613 | 1148.301 | 0.0784104 | 0.0744351 | 0.0784104 | 0.0712282 | 0.949 | no_flagship |
| fremtpl2_severity_log | log_claim_amount | Exposure | 4000 | -0.392 | 3.195 | 0.828967 | 0.830935 | 0.828967 | 0.829046 | 1.002 | no_flagship |
| gas_turbine_co_2015_raw | CO | TIT | 7384 | 9.912 | 197.398 | 0.275787 | 0.45438 | 0.275787 | 0.329211 | 1.648 | candidate_illustration |
| gas_turbine_nox_2015_raw | NOX | AT | 7384 | 0.727 | 6.673 | 1.08297 | 1.06459 | 1.08297 | 1.08113 | 0.983 | no_flagship |
| gas_turbine_co_all_raw | CO | TIT | 36733 | 7.893 | 159.369 | 0.410362 | 1.05809 | 0.410362 | 0.415731 | 2.578 | candidate_illustration |
| gas_turbine_nox_all_raw | NOX | AT | 36733 | 0.966 | 2.727 | 1.17784 | 1.1667 | 1.17784 | 1.10588 | 0.991 | no_flagship |
| sru_y1_static | y1 | u2 | 10081 | 7.553 | 95.701 | 1.4046 | 1.47242 | 1.4046 | 1.26786 | 1.048 | no_flagship |
| sru_y2_static | y2 | u3 | 10081 | 3.019 | 29.072 | 0.955207 | 0.930168 | 0.955207 | 0.910347 | 0.974 | no_flagship |
| sru_y1_dynamic | y1 | y1_lag1 | 10079 | 10.557 | 274.212 | 0.133954 | 0.548277 | 0.133954 | 0.132697 | 4.093 | candidate_illustration |
| sru_y2_dynamic | y2 | y2_lag1 | 10079 | 5.095 | 174.857 | 0.0600723 | 0.0871389 | 0.0600723 | 0.0604411 | 1.451 | candidate_illustration |
| concrete | strength | age | 1030 | 0.591 | -0.226 | 0.545752 | 0.567897 | 0.545752 | 0.523674 | 1.041 | no_flagship |

## Reading

The current 3a manuscript should not claim real-world unbiased identification from this screen. The synthetic finite-memory experiments remain the evidence base for H4; these real-world rows can be used only as diagnostic illustrations of the VWK-vs-misspecified-Wiener gap, not as theorem evidence.
