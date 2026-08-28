#!/usr/bin/env python3
"""E5 -- streaming update of the matched projection (Rv1-C10).

Two streaming schemes, both O(|I|) per sample with no matrix update:

  (A) naive: running mean of Y * Psi_alpha(x) under the CURRENT empirical basis,
      basis refreshed every K samples from online raw moments. Cheap, but the mean
      averages coordinates taken in slightly different bases, so it lags the batch
      estimate (reported: max |a_rec - a_batch| along the stream).
  (B) exact re-basing via Theorem 1: accumulate the basis-free monomial
      cross-correlations c_beta = E_n[Y x^beta] and the raw moments online; at
      any query time build T_s (O(s^3)) and map c -> a by the tensor product of
      T_s. Reproduces the batch diagonal projection to machine precision at
      every checkpoint.

Cost comparison is done on the UPDATE STEP ONLY with pre-evaluated feature rows
(the same p features for all schemes): O(p) running sums versus the O(p^2) RLS
covariance update in the monomial basis.
"""

from __future__ import annotations

import time

import numpy as np

from common import (
    OUT, REGIMES, build_vwk_basis, canonical_beta, fit_vwk_volterra, lag_matrix, monomial_design,
    multi_indices, sample_iid, tensor_features, write_csv,
)

N, D, S, NOISE, K, SEED = 50000, 3, 2, 0.25, 500, 20260827
CHECKPOINTS = [500, 1000, 2000, 5000, 10000, 20000, 50000]


def tensor_T(basis, idx):
    """Matrix M with a = M c, c_beta = E[Y x^beta], a_alpha = E[Y Psi_alpha]; from Theorem 1."""
    T = basis.coeffs  # psi_k = sum_j T[k, j] x^j
    p = len(idx)
    M = np.zeros((p, p))
    pos = {mi: i for i, mi in enumerate(idx)}
    for i, al in enumerate(idx):
        # Psi_alpha = prod_r psi_{alpha_r}(x_r) = sum over beta <= alpha of prod_r T[alpha_r, beta_r] x^beta
        from itertools import product
        for be in product(*[range(a + 1) for a in al]):
            coef = 1.0
            for r in range(D):
                coef *= T[al[r], be[r]]
            M[i, pos[tuple(be)]] += coef
    return M


def main() -> None:
    regime = REGIMES[1]
    rng = np.random.default_rng(SEED)
    mom = regime.moments(S)
    tb = build_vwk_basis(mom, S)
    x = sample_iid(rng, regime, N)
    feats_true, idx = tensor_features(x, tb, S, D)
    beta = canonical_beta(idx)
    sig = feats_true @ beta
    y = np.zeros(N)
    y[D - 1:] = sig + rng.normal(scale=NOISE, size=sig.size)
    lags = lag_matrix(x, D)
    p = len(idx)
    mono = monomial_design(x, S, D)  # basis-free feature rows x^beta
    powers = x[:, None] ** np.arange(2 * S + 1)[None, :]
    yt = y[D - 1:]

    rows = []
    ckpt = set(CHECKPOINTS)
    # ---- scheme A (naive running mean under refreshed basis) and scheme B (exact re-basing)
    a_naive = np.zeros(p)
    c_sum = np.zeros(p)  # sum of Y x^beta
    pow_sum = np.zeros(2 * S + 1)
    basis = None
    for t in range(N):
        pow_sum += powers[t]
        if (t + 1) % K == 0:
            basis = build_vwk_basis(pow_sum / (t + 1), S)
            M_cur = tensor_T(basis, idx)
        if t < D - 1:
            continue
        i = t - (D - 1)
        c_sum += yt[i] * mono[i]
        if basis is not None:
            # psi rows under the current basis, from the monomial row via T (O(p) with cached M)
            psi = M_cur @ mono[i]
            a_naive += (yt[i] * psi - a_naive) / (i + 1)
        if t + 1 in ckpt:
            batch = fit_vwk_volterra(x[: t + 1], y[: t + 1], order=S, memory=D, moments=None).coefficients
            b_now = build_vwk_basis(pow_sum / (t + 1), S)
            a_exact = tensor_T(b_now, idx) @ (c_sum / (i + 1))
            # Algorithm 1, step 3, literally: a_alpha = E_n[Y Psi_alpha] under the final basis
            f_now, _ = tensor_features(x[: t + 1], b_now, S, D)
            a_alg1 = f_now.T @ yt[: i + 1] / (i + 1)
            rows.append({
                "t": t + 1,
                "exact_max_abs_dev_vs_alg1": float(np.max(np.abs(a_exact - a_alg1))),
                "naive_max_abs_dev_vs_batch": float(np.max(np.abs(a_naive - batch))) if basis is not None else np.nan,
                "exact_max_abs_dev_vs_batch_empnorm": float(np.max(np.abs(a_exact - batch))),
                "rmse_naive_vs_truth": float(np.sqrt(np.mean((a_naive - beta) ** 2))),
                "rmse_exact_vs_truth": float(np.sqrt(np.mean((a_exact - beta) ** 2))),
                "rmse_batch_vs_truth": float(np.sqrt(np.mean((batch - beta) ** 2))),
            })

    # ---- update-step cost: O(p) running sums vs O(p^2) RLS, same pre-evaluated rows
    m = mono.shape[0]
    t0 = time.perf_counter()
    c = np.zeros(p); q = np.zeros(2 * S + 1)
    for i in range(m):
        c += yt[i] * mono[i]
        q += powers[i + D - 1]
    t_sums = time.perf_counter() - t0
    t0 = time.perf_counter()
    P = np.eye(p) * 1e4; th = np.zeros(p)
    for i in range(m):
        phi = mono[i]
        Pphi = P @ phi
        g = Pphi / (1.0 + phi @ Pphi)
        th += g * (yt[i] - phi @ th)
        P -= np.outer(g, Pphi)
    t_rls = time.perf_counter() - t0
    # sanity: RLS converges to the LS solution
    ls = np.linalg.lstsq(mono, yt, rcond=None)[0]
    rls_dev = float(np.max(np.abs(th - ls)))

    for r in rows:
        r.update({"update_cost_running_sums_s": t_sums, "update_cost_rls_s": t_rls,
                  "rls_vs_ls_max_dev": rls_dev, "refresh_K": K, "n_features": p, "n": N})
        print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"update cost: running sums {t_sums:.2f}s vs RLS {t_rls:.2f}s (ratio {t_rls / t_sums:.2f}); RLS-LS dev {rls_dev:.1e}")
    write_csv(OUT / "e5_streaming.csv", rows)


if __name__ == "__main__":
    main()
