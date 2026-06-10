"""De-confounding check: the W/V penalty is a diagonal-projection artifact.

The reported "VWK MSE" and "Wiener MSE" both come from DIAGONAL projection
(coordinate-wise b_k = <Y,phi_k>/<phi_k,phi_k>) in two bases that span the SAME
degree-<=s polynomial space. Full least squares is basis-invariant, so LS in the
VWK basis, the Wiener/Hermite basis, and the raw monomial basis must coincide and
dominate both diagonal projections. This script makes that explicit for the
centered-exponential row of Table 1.

Run:
    PYTHONPATH=code uv run --with numpy python code/experiments/verify_deconfound.py
"""

from __future__ import annotations

import numpy as np

from vwk.basis import build_vwk_basis, centered_exponential_moments, normal_moments

SEED = 20260608
BETA = np.array([0.2, 0.8, 0.6])  # signal in matched VWK coordinates


def diag_mse(feats, y, truth):
    coef = (feats.T @ y) / np.sum(feats * feats, axis=0)
    return float(np.mean((feats @ coef - truth) ** 2))


def ls_mse(feats, y, truth):
    coef = np.linalg.lstsq(feats, y, rcond=None)[0]
    return float(np.mean((feats @ coef - truth) ** 2))


def main(n: int = 2000, reps: int = 200, noise: float = 0.25) -> None:
    rng = np.random.default_rng(SEED)
    m = centered_exponential_moments(4)
    matched = build_vwk_basis(m, 2, name="vwk")
    wiener = build_vwk_basis(normal_moments(4, variance=1.0), 2, name="wiener")
    f = sum(BETA[k] * matched.coeffs[k] for k in range(3))

    acc = {k: [] for k in ["diag_vwk", "diag_wiener", "ls_vwk", "ls_wiener", "ls_power"]}
    for _ in range(reps):
        x = rng.exponential(1.0, n) - 1.0
        truth = np.polynomial.polynomial.polyval(x, f)
        y = truth + noise * rng.standard_normal(n)
        acc["diag_vwk"].append(diag_mse(matched.evaluate(x), y, truth))
        acc["diag_wiener"].append(diag_mse(wiener.evaluate(x), y, truth))
        acc["ls_vwk"].append(ls_mse(matched.evaluate(x), y, truth))
        acc["ls_wiener"].append(ls_mse(wiener.evaluate(x), y, truth))
        acc["ls_power"].append(ls_mse(np.vander(x, N=3, increasing=True), y, truth))
    means = {k: float(np.mean(v)) for k, v in acc.items()}

    print("Centered-exponential row, signal MSE (n=2000, reps=200, noise=0.25):")
    for k in ["diag_vwk", "diag_wiener", "ls_vwk", "ls_wiener", "ls_power"]:
        print(f"  {k:14s} {means[k]:.6f}")
    print(f"\n  diagonal W/V                 = {means['diag_wiener'] / means['diag_vwk']:.3f}")
    print(f"  max|LS across the 3 bases|   = "
          f"{max(abs(means['ls_vwk'] - means['ls_wiener']), abs(means['ls_vwk'] - means['ls_power'])):.2e}")
    print(f"  full-LS / diagonal-VWK       = {means['ls_vwk'] / means['diag_vwk']:.4f} "
          f"(full LS dominates both diagonal projections)")


if __name__ == "__main__":
    main()
