#!/usr/bin/env python3
"""Level-1 sanity checks for the Paper 3a VWK package."""

from __future__ import annotations

import numpy as np

from vwk import build_vwk_basis, centered_exponential_moments, normal_moments


def main() -> int:
    order = 5

    normal = build_vwk_basis(normal_moments(2 * order), order, name="normal")
    normal_gram_err = float(np.max(np.abs(normal.gram() - np.eye(order + 1))))

    exp_basis = build_vwk_basis(centered_exponential_moments(2 * order), order, name="centered-exp")
    exp_gram_err = float(np.max(np.abs(exp_basis.gram() - np.eye(order + 1))))

    checks = {
        "normal_gram_err": normal_gram_err,
        "centered_exp_gram_err": exp_gram_err,
        "normal_psi0_constant": float(abs(normal.coeffs[0, 0] - 1.0)),
        "normal_psi1_x": float(np.max(np.abs(normal.coeffs[1, :2] - np.array([0.0, 1.0])))),
    }

    for name, value in checks.items():
        print(f"{name}: {value:.3e}")

    passed = all(value < 1e-8 for value in checks.values())
    print("RESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
