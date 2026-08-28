#!/usr/bin/env python3
"""E3 -- basis built for an assumed law, data from a slightly different one (Rv1-C5).

Two perturbation families on the centered-exponential base law:
  (a) eps-contamination: data ~ (1-eps) P + eps N(0, 3 sigma^2), basis built for P;
  (b) skew mis-set: data ~ P, basis built from P's moments with m3 scaled by (1 +/- 0.2).
Compared: mismatched matched-basis (diagonal), correctly matched (diagonal),
Wiener (diagonal), monomial full LS; plus cond of the empirical feature Gram in
the mismatched-VWK and monomial coordinates.
"""

from __future__ import annotations

import numpy as np

from common import (
    OUT, REGIMES, align, build_vwk_basis, canonical_beta, empirical_moments, fit_projection_model,
    fit_wiener_baseline, monomial_design, sample_iid, tensor_features, write_csv,
)

N, D, S, NOISE, REPS, SEED = 2500, 3, 2, 0.25, 150, 20260827
EPS = [0.0, 0.01, 0.02, 0.05, 0.10]
SKEW_SCALE = [0.8, 1.0, 1.2]
BASE = REGIMES[1]  # centered exponential


def one(rng, x, assumed_moments, true_moments):
    true_basis = build_vwk_basis(true_moments, S)
    feats, idx = tensor_features(x, true_basis, S, D)
    beta = canonical_beta(idx)
    sig = feats @ beta
    y = align(x, sig + rng.normal(scale=NOISE, size=sig.size), D)
    t = sig
    mis = fit_projection_model(x, y, build_vwk_basis(assumed_moments, S), order=S, memory=D)
    cor = fit_projection_model(x, y, true_basis, order=S, memory=D)
    w = fit_wiener_baseline(x, y, order=S, memory=D, variance=float(true_moments[2]))
    md = monomial_design(x, S, D)
    ls = md @ np.linalg.lstsq(md, y[D - 1:], rcond=None)[0]
    fm, _ = tensor_features(x, mis.basis, S, D)
    return {
        "mismatched_vwk_mse": np.mean((mis.predict(x) - t) ** 2),
        "correct_vwk_mse": np.mean((cor.predict(x) - t) ** 2),
        "wiener_mse": np.mean((w.predict(x) - t) ** 2),
        "monomial_ls_mse": np.mean((ls - t) ** 2),
        "cond_mismatched_vwk_gram": np.linalg.cond(fm.T @ fm / fm.shape[0]),
        "cond_monomial_gram": np.linalg.cond(md.T @ md / md.shape[0]),
    }


def main() -> None:
    rows = []
    base_mom = BASE.moments(S)
    # (a) contamination -- true moments of the mixture are known in closed form
    for k, eps in enumerate(EPS):
        rng = np.random.default_rng(SEED + k)
        from common import normal_moments
        true_mom = (1 - eps) * base_mom + eps * normal_moments(2 * S, variance=3.0)
        acc = []
        for _ in range(REPS):
            n_c = rng.binomial(N, eps)
            x = np.concatenate([sample_iid(rng, BASE, N - n_c), rng.normal(scale=np.sqrt(3.0), size=n_c)])
            rng.shuffle(x)
            acc.append(one(rng, x, base_mom, true_mom))
        m = {k2: float(np.mean([a[k2] for a in acc])) for k2 in acc[0]}
        rows.append({"family": "contamination", "param": eps, **m,
                     "mis_over_correct": m["mismatched_vwk_mse"] / m["correct_vwk_mse"],
                     "W_over_mis": m["wiener_mse"] / m["mismatched_vwk_mse"]})
        print(f"contam eps={eps:.2f} mis={m['mismatched_vwk_mse']:.5f} cor={m['correct_vwk_mse']:.5f} "
              f"W={m['wiener_mse']:.5f} LS={m['monomial_ls_mse']:.5f} condV={m['cond_mismatched_vwk_gram']:.2f} "
              f"condM={m['cond_monomial_gram']:.1f}")
    # (b) skew mis-set
    for k, sc in enumerate(SKEW_SCALE):
        rng = np.random.default_rng(SEED + 100 + k)
        assumed = base_mom.copy()
        assumed[3] = sc * assumed[3]
        acc = [one(rng, sample_iid(rng, BASE, N), assumed, base_mom) for _ in range(REPS)]
        m = {k2: float(np.mean([a[k2] for a in acc])) for k2 in acc[0]}
        rows.append({"family": "skew_scale", "param": sc, **m,
                     "mis_over_correct": m["mismatched_vwk_mse"] / m["correct_vwk_mse"],
                     "W_over_mis": m["wiener_mse"] / m["mismatched_vwk_mse"]})
        print(f"skew x{sc:.1f} mis={m['mismatched_vwk_mse']:.5f} cor={m['correct_vwk_mse']:.5f} "
              f"W={m['wiener_mse']:.5f} condV={m['cond_mismatched_vwk_gram']:.2f} condM={m['cond_monomial_gram']:.1f}")
    write_csv(OUT / "e3_law_mismatch.csv", rows)


if __name__ == "__main__":
    main()
