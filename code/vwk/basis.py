"""Moment-driven VWK basis construction.

Polynomials are represented by coefficient vectors in increasing powers:
``c[0] + c[1] x + ... + c[k] x**k``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, factorial

import numpy as np
from numpy.polynomial import polynomial as poly


@dataclass(frozen=True)
class Basis:
    """Oriented orthonormal polynomial basis in L2(P)."""

    coeffs: np.ndarray
    moments: np.ndarray
    normalized: bool = True
    name: str = "vwk"

    @property
    def order(self) -> int:
        return int(self.coeffs.shape[0] - 1)

    def polynomial(self, degree: int) -> np.ndarray:
        if degree < 0 or degree > self.order:
            raise ValueError(f"degree must be in [0, {self.order}]")
        return self.coeffs[degree].copy()

    def evaluate(self, x: np.ndarray | float, degree: int | None = None) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        if degree is not None:
            return poly.polyval(x_arr, self.coeffs[degree])
        values = [poly.polyval(x_arr, self.coeffs[k]) for k in range(self.order + 1)]
        return np.stack(values, axis=-1)

    def gram(self) -> np.ndarray:
        gram = np.empty((self.order + 1, self.order + 1), dtype=float)
        for i in range(self.order + 1):
            for j in range(self.order + 1):
                gram[i, j] = moment_inner(self.coeffs[i], self.coeffs[j], self.moments)
        return gram


def empirical_moments(x: np.ndarray, max_order: int) -> np.ndarray:
    """Return raw moments m_0..m_max_order from a sample."""

    if max_order < 0:
        raise ValueError("max_order must be non-negative")
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim != 1 or x_arr.size == 0:
        raise ValueError("x must be a non-empty one-dimensional array")
    moments = np.empty(max_order + 1, dtype=float)
    moments[0] = 1.0
    for r in range(1, max_order + 1):
        moments[r] = float(np.mean(x_arr**r))
    return moments


def normal_moments(max_order: int, mean: float = 0.0, variance: float = 1.0) -> np.ndarray:
    """Raw moments m_0..m_max_order for N(mean, variance)."""

    if variance <= 0:
        raise ValueError("variance must be positive")
    moments = np.zeros(max_order + 1, dtype=float)
    moments[0] = 1.0
    if max_order >= 1:
        moments[1] = float(mean)
    for n in range(2, max_order + 1):
        moments[n] = mean * moments[n - 1] + (n - 1) * variance * moments[n - 2]
    return moments


def centered_exponential_moments(max_order: int) -> np.ndarray:
    """Raw moments for X = E - 1, where E follows Exp(1)."""

    moments = np.empty(max_order + 1, dtype=float)
    for n in range(max_order + 1):
        total = 0.0
        for j in range(n + 1):
            total += comb(n, j) * ((-1.0) ** (n - j)) * factorial(j)
        moments[n] = total
    return moments


def uniform_moments(max_order: int, low: float, high: float) -> np.ndarray:
    """Raw moments for Uniform(low, high)."""

    if not low < high:
        raise ValueError("low must be less than high")
    moments = np.empty(max_order + 1, dtype=float)
    for n in range(max_order + 1):
        moments[n] = (high ** (n + 1) - low ** (n + 1)) / ((n + 1) * (high - low))
    return moments


def symmetric_gaussian_mixture_moments(
    max_order: int,
    *,
    weight: float = 0.9,
    variance_low: float = 0.25,
    variance_high: float = 4.0,
) -> np.ndarray:
    """Raw moments for a zero-mean two-component Gaussian mixture."""

    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    if variance_low <= 0 or variance_high <= 0:
        raise ValueError("variances must be positive")
    low = normal_moments(max_order, variance=variance_low)
    high = normal_moments(max_order, variance=variance_high)
    return weight * low + (1.0 - weight) * high


def moment_inner(c: np.ndarray, d: np.ndarray, moments: np.ndarray) -> float:
    """Inner product E[p(X) q(X)] from coefficient vectors and raw moments."""

    c_arr = np.asarray(c, dtype=float)
    d_arr = np.asarray(d, dtype=float)
    required = c_arr.size + d_arr.size - 1
    if moments.size < required:
        raise ValueError("not enough moments for requested inner product")
    total = 0.0
    for i, ci in enumerate(c_arr):
        if ci == 0:
            continue
        for j, dj in enumerate(d_arr):
            if dj != 0:
                total += ci * dj * moments[i + j]
    return float(total)


def build_vwk_basis(
    moments: np.ndarray,
    order: int,
    *,
    normalize: bool = True,
    orient: str = "positive_leading",
    name: str = "vwk",
    atol: float = 1e-12,
) -> Basis:
    """Build the Lemma 1 VWK basis by finite Gram-Schmidt on monomials."""

    if order < 0:
        raise ValueError("order must be non-negative")
    moments_arr = np.asarray(moments, dtype=float)
    if moments_arr.size < 2 * order + 1:
        raise ValueError(f"need moments m_0..m_{2 * order}")
    if not np.isclose(moments_arr[0], 1.0, atol=1e-8):
        raise ValueError("moments[0] must be 1 for a probability law")
    if orient not in {"positive_leading", "none"}:
        raise ValueError("orient must be 'positive_leading' or 'none'")

    coeffs = np.zeros((order + 1, order + 1), dtype=float)
    for k in range(order + 1):
        vec = np.zeros(order + 1, dtype=float)
        vec[k] = 1.0
        for j in range(k):
            denom = moment_inner(coeffs[j], coeffs[j], moments_arr)
            if abs(denom) < atol:
                raise np.linalg.LinAlgError(f"zero norm in basis vector {j}")
            vec -= moment_inner(vec, coeffs[j], moments_arr) / denom * coeffs[j]

        norm_sq = moment_inner(vec, vec, moments_arr)
        if norm_sq <= atol:
            raise np.linalg.LinAlgError(
                f"monomials are linearly dependent at degree {k}; norm={norm_sq:g}"
            )
        if normalize:
            vec /= np.sqrt(norm_sq)
        if orient == "positive_leading" and vec[k] < 0:
            vec *= -1.0
        coeffs[k] = vec

    return Basis(coeffs=coeffs, moments=moments_arr[: 2 * order + 1].copy(), normalized=normalize, name=name)
