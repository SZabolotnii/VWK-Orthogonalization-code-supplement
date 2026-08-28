#!/usr/bin/env python3
"""E2 -- degree x sample-size grid for the empirically built basis (Rv1-C3, Rv2-C5).

One-lag (d=1) so the degree can be pushed to s=6. Per cell: orthogonality error of
the empirical basis measured under the TRUE law, cond of the empirical Hankel,
rate of numerical failure, and the relative coefficient error of the empirical
matched projection against the oracle-moment projection.
"""

from __future__ import annotations

import numpy as np

from common import (
    OUT, REGIMES, basis_orthogonality_error, build_vwk_basis, empirical_moments,
    fit_projection_model, hankel, sample_iid, write_csv,
)

DEGREES = [2, 3, 4, 5, 6]
NS = [250, 500, 1000, 2500, 5000, 20000]
REPS, NOISE, SEED = 200, 0.25, 20260827


def main() -> None:
    rows = []
    for r_off, regime in enumerate(REGIMES):
        for s in DEGREES:
            mom = regime.moments(s)
            true_basis = build_vwk_basis(mom, s)
            beta = np.asarray([0.10] + [((-1.0) ** k) * (0.20 + 0.03 * k) for k in range(1, s + 1)])
            for n in NS:
                rng = np.random.default_rng(SEED + 1000 * r_off + 37 * s + n)
                orth, cond, coef_err, fails = [], [], [], 0
                for _ in range(REPS):
                    x = sample_iid(rng, regime, n)
                    sig = true_basis.evaluate(x) @ beta
                    y = sig + rng.normal(scale=NOISE, size=n)
                    m_hat = empirical_moments(x, 2 * s)
                    try:
                        b_hat = build_vwk_basis(m_hat, s)
                    except np.linalg.LinAlgError:
                        fails += 1
                        continue
                    orth.append(basis_orthogonality_error(b_hat, mom))
                    cond.append(np.linalg.cond(hankel(m_hat, s)))
                    fo = fit_projection_model(x, y, true_basis, order=s)
                    fe = fit_projection_model(x, y, b_hat, order=s)
                    # compare fitted signals, which are basis-free
                    coef_err.append(np.sqrt(np.mean((fe.predict(x) - fo.predict(x)) ** 2) / np.mean(sig**2)))
                row = {
                    "regime": regime.name, "order": s, "n": n, "reps": REPS, "fail_rate": fails / REPS,
                    "orth_err_median": float(np.median(orth)), "orth_err_p90": float(np.quantile(orth, 0.9)),
                    "hankel_cond_median": float(np.median(cond)),
                    "rel_signal_err_emp_vs_oracle_median": float(np.median(coef_err)),
                    "rel_signal_err_p90": float(np.quantile(coef_err, 0.9)),
                }
                rows.append(row)
                print(f"{regime.name:28s} s={s} n={n:6d} orth_med={row['orth_err_median']:.2e} "
                      f"p90={row['orth_err_p90']:.2e} cond={row['hankel_cond_median']:.2e} "
                      f"relerr={row['rel_signal_err_emp_vs_oracle_median']:.3e} fail={row['fail_rate']:.2f}")
    write_csv(OUT / "e2_moment_sensitivity.csv", rows)


if __name__ == "__main__":
    main()
