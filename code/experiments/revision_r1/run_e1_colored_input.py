#!/usr/bin/env python3
"""E1 -- colored (AR(1)-copula) input, answers Rv1-C2.

Marginal law held exactly fixed across phi (Gaussian copula), so the population
matched basis is the same at every phi; only the dependence changes. Reports the
diagonal-estimator MSEs, the W/V ratio, the off-diagonal mass of the empirical
tensor Gram, and the full-solve (least squares in the VWK basis) MSE as the remedy.
"""

from __future__ import annotations

import numpy as np

from common import (
    OUT, REGIMES, align, canonical_beta, fit_projection_model, fit_vwk_volterra,
    fit_wiener_baseline, gram_offdiag, sample_ar1_copula, tensor_features, write_csv, build_vwk_basis,
)

PHIS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
N, D, S, NOISE, REPS, SEED = 2500, 3, 2, 0.25, 150, 20260827


def main() -> None:
    rows = []
    for r_off, regime in enumerate(REGIMES):
        mom = regime.moments(S)
        true_basis = build_vwk_basis(mom, S)
        for phi in PHIS:
            rng = np.random.default_rng(SEED + 1000 * r_off + int(phi * 100))
            acc = {k: [] for k in ("oracle", "emp", "wiener", "vwk_ls", "offdiag", "skew")}
            for _ in range(REPS):
                x = sample_ar1_copula(rng, regime, N, phi)
                feats, idx = tensor_features(x, true_basis, S, D)
                beta = canonical_beta(idx)
                sig = feats @ beta
                truth = align(x, sig, D)
                y = align(x, sig + rng.normal(scale=NOISE, size=sig.size), D)
                o = fit_vwk_volterra(x, y, order=S, memory=D, moments=mom)
                e = fit_vwk_volterra(x, y, order=S, memory=D, moments=None)
                w = fit_wiener_baseline(x, y, order=S, memory=D, variance=regime.variance)
                ls = fit_projection_model(x, y, true_basis, order=S, memory=D, method="least_squares")
                t = truth[D - 1:]
                acc["oracle"].append(np.mean((o.predict(x) - t) ** 2))
                acc["emp"].append(np.mean((e.predict(x) - t) ** 2))
                acc["wiener"].append(np.mean((w.predict(x) - t) ** 2))
                acc["vwk_ls"].append(np.mean((ls.predict(x) - t) ** 2))
                acc["offdiag"].append(gram_offdiag(feats))
                acc["skew"].append(float(np.mean(x**3) / np.mean(x**2) ** 1.5))
            m = {k: float(np.mean(v)) for k, v in acc.items()}
            rows.append({
                "regime": regime.name, "phi": phi, "n": N, "memory": D, "order": S, "reps": REPS,
                "marginal_skew": round(m["skew"], 4),
                "gram_offdiag_rel": round(m["offdiag"], 5),
                "oracle_vwk_mse": m["oracle"], "empirical_vwk_mse": m["emp"],
                "wiener_mse": m["wiener"], "vwk_full_ls_mse": m["vwk_ls"],
                "W_over_oracle": m["wiener"] / m["oracle"],
                "W_over_emp": m["wiener"] / m["emp"],
                "oracle_over_ls": m["oracle"] / m["vwk_ls"],
            })
            print(f"{regime.name:28s} phi={phi:.1f} skew={m['skew']:+.3f} offdiag={m['offdiag']:.4f} "
                  f"O={m['oracle']:.5f} W={m['wiener']:.5f} LS={m['vwk_ls']:.5f} W/O={m['wiener']/m['oracle']:.3f} "
                  f"O/LS={m['oracle']/m['vwk_ls']:.2f}")
    write_csv(OUT / "e1_colored_input.csv", rows)


if __name__ == "__main__":
    main()
