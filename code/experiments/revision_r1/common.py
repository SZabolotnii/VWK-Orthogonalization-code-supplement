"""Shared pieces for the CSSP-R1 revision experiments (E1-E5).

Everything reuses ``code/vwk``; only the input-generation and metric helpers
that the revision needs are added here. Nothing in this file is quoted in the
paper -- the scripts write the artifacts that are.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np
from scipy import stats

_HERE = Path(__file__).resolve()
for _cand in (_HERE.parents[3] / "code", _HERE.parents[2]):  # track layout / supplement layout
    if (_cand / "vwk").is_dir():
        sys.path.insert(0, str(_cand))
        break

from vwk import (  # noqa: E402
    build_vwk_basis,
    centered_exponential_moments,
    empirical_moments,
    fit_projection_model,
    fit_vwk_volterra,
    fit_wiener_baseline,
    normal_moments,
    symmetric_gaussian_mixture_moments,
    uniform_moments,
)
from vwk.volterra import lag_matrix, multi_indices, tensor_features  # noqa: E402

OUT = Path(__file__).resolve().parent
HALF = sqrt(3.0)
MIX_W, MIX_LO, MIX_HI = 0.9, 0.25, 4.0
MIX_VAR = MIX_W * MIX_LO + (1 - MIX_W) * MIX_HI


@dataclass(frozen=True)
class Regime:
    name: str
    sampler: str
    variance: float

    def moments(self, order: int) -> np.ndarray:
        k = 2 * order
        if self.sampler == "normal":
            return normal_moments(k)
        if self.sampler == "centered_exp":
            return centered_exponential_moments(k)
        if self.sampler == "uniform":
            return uniform_moments(k, -HALF, HALF)
        if self.sampler == "mixture":
            return symmetric_gaussian_mixture_moments(k, weight=MIX_W, variance_low=MIX_LO, variance_high=MIX_HI)
        raise ValueError(self.sampler)


REGIMES = [
    Regime("gaussian_control", "normal", 1.0),
    Regime("centered_exponential_skew", "centered_exp", 1.0),
    Regime("uniform_platykurtic", "uniform", 1.0),
    Regime("symmetric_contaminated", "mixture", MIX_VAR),
]

# Mixture quantile: tabulated inverse of the CDF (used by the Gaussian copula).
_MIX_GRID = np.linspace(-9.0, 9.0, 40001)
_MIX_CDF = MIX_W * stats.norm.cdf(_MIX_GRID, scale=sqrt(MIX_LO)) + (1 - MIX_W) * stats.norm.cdf(
    _MIX_GRID, scale=sqrt(MIX_HI)
)


def sample_iid(rng: np.random.Generator, regime: Regime, n: int) -> np.ndarray:
    if regime.sampler == "normal":
        return rng.normal(size=n)
    if regime.sampler == "centered_exp":
        return rng.exponential(size=n) - 1.0
    if regime.sampler == "uniform":
        return rng.uniform(-HALF, HALF, size=n)
    if regime.sampler == "mixture":
        mask = rng.random(n) < MIX_W
        x = np.empty(n)
        x[mask] = rng.normal(scale=sqrt(MIX_LO), size=int(mask.sum()))
        x[~mask] = rng.normal(scale=sqrt(MIX_HI), size=int((~mask).sum()))
        return x
    raise ValueError(regime.sampler)


def quantile(regime: Regime, u: np.ndarray) -> np.ndarray:
    """Marginal quantile function F^{-1}(u) for the regime."""
    if regime.sampler == "normal":
        return stats.norm.ppf(u)
    if regime.sampler == "centered_exp":
        return -np.log1p(-u) - 1.0
    if regime.sampler == "uniform":
        return -HALF + 2 * HALF * u
    if regime.sampler == "mixture":
        return np.interp(u, _MIX_CDF, _MIX_GRID)
    raise ValueError(regime.sampler)


def sample_ar1_copula(rng: np.random.Generator, regime: Regime, n: int, phi: float, burn: int = 500) -> np.ndarray:
    """Stationary AR(1) Gaussian copula with the regime's marginal held EXACTLY fixed.

    z_t = phi z_{t-1} + sqrt(1-phi^2) e_t, e_t ~ N(0,1), u_t = F^{-1}(Phi(z_t)).
    phi = 0 reproduces the i.i.d. input; the marginal law -- hence the population
    Hankel matrix and the matched basis -- does not depend on phi.
    """
    e = rng.normal(size=n + burn)
    z = np.empty(n + burn)
    z[0] = e[0]
    s = sqrt(1.0 - phi * phi)
    for t in range(1, n + burn):
        z[t] = phi * z[t - 1] + s * e[t]
    u = stats.norm.cdf(z[burn:])
    u = np.clip(u, 1e-12, 1 - 1e-12)
    return quantile(regime, u)


def canonical_beta(indices: tuple[tuple[int, ...], ...]) -> np.ndarray:
    """The paper's finite-memory (d=3, s=2) coefficient vector, else a generic one."""
    canonical = {
        (0, 0, 0): 0.10, (0, 0, 1): 0.70, (0, 1, 0): -0.35, (1, 0, 0): 0.25,
        (0, 0, 2): 0.45, (0, 1, 1): -0.20, (0, 2, 0): 0.15, (1, 0, 1): 0.30,
        (1, 1, 0): -0.10, (2, 0, 0): 0.55,
    }
    if set(indices) == set(canonical):
        return np.asarray([canonical[i] for i in indices])
    return np.asarray(
        [0.10 if sum(i) == 0 else ((-1.0) ** c) * (0.20 + 0.03 * c) for c, i in enumerate(indices)]
    )


def align(x: np.ndarray, signal: np.ndarray, memory: int) -> np.ndarray:
    full = np.zeros_like(x)
    full[memory - 1:] = signal
    return full


def gram_offdiag(features: np.ndarray) -> float:
    """Relative off-diagonal mass ||G - I||_F / ||I||_F of the empirical feature Gram."""
    g = features.T @ features / features.shape[0]
    p = g.shape[0]
    return float(np.linalg.norm(g - np.eye(p)) / sqrt(p))


def basis_orthogonality_error(basis, true_moments: np.ndarray) -> float:
    """max |<psi_i, psi_j>_P - delta_ij| of a basis evaluated under the TRUE law P."""
    from vwk.basis import moment_inner

    k = basis.order + 1
    err = 0.0
    for i in range(k):
        for j in range(k):
            v = moment_inner(basis.coeffs[i], basis.coeffs[j], true_moments)
            err = max(err, abs(v - (1.0 if i == j else 0.0)))
    return err


def hankel(moments: np.ndarray, order: int) -> np.ndarray:
    return np.asarray([[moments[i + j] for j in range(order + 1)] for i in range(order + 1)])


def monomial_design(x: np.ndarray, order: int, memory: int) -> np.ndarray:
    lags = lag_matrix(x, memory)
    idx = multi_indices(order, memory)
    d = np.ones((lags.shape[0], len(idx)))
    for c, mi in enumerate(idx):
        for lag, deg in enumerate(mi):
            if deg:
                d[:, c] *= lags[:, lag] ** deg
    return d


def write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


__all__ = [
    "REGIMES", "Regime", "OUT", "align", "basis_orthogonality_error", "build_vwk_basis",
    "canonical_beta", "empirical_moments", "fit_projection_model", "fit_vwk_volterra",
    "fit_wiener_baseline", "gram_offdiag", "hankel", "lag_matrix", "monomial_design",
    "multi_indices", "normal_moments", "sample_ar1_copula", "sample_iid", "tensor_features",
    "write_csv",
]
