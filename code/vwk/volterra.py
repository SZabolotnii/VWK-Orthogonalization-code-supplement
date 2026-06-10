"""Finite-memory Volterra feature maps and projection fits."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .basis import Basis, build_vwk_basis, empirical_moments, normal_moments


@dataclass(frozen=True)
class FitResult:
    """Result of a finite-memory projection fit."""

    coefficients: np.ndarray
    multi_indices: tuple[tuple[int, ...], ...]
    basis: Basis
    memory: int
    order: int
    method: str

    def predict(self, x: np.ndarray) -> np.ndarray:
        features, _ = tensor_features(x, self.basis, self.order, self.memory)
        return features @ self.coefficients


def multi_indices(order: int, memory: int) -> tuple[tuple[int, ...], ...]:
    if order < 0:
        raise ValueError("order must be non-negative")
    if memory <= 0:
        raise ValueError("memory must be positive")
    indices = []
    for idx in product(range(order + 1), repeat=memory):
        if sum(idx) <= order:
            indices.append(tuple(int(v) for v in idx))
    indices.sort(key=lambda item: (sum(item), item))
    return tuple(indices)


def lag_matrix(x: np.ndarray, memory: int) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if memory <= 0:
        raise ValueError("memory must be positive")
    if x_arr.size < memory:
        raise ValueError("x is shorter than memory")
    rows = []
    for t in range(memory - 1, x_arr.size):
        rows.append([x_arr[t - lag] for lag in range(memory)])
    return np.asarray(rows, dtype=float)


def tensor_features(
    x: np.ndarray, basis: Basis, order: int, memory: int
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Evaluate tensor-product basis features with total degree <= order."""

    lags = lag_matrix(x, memory)
    indices = multi_indices(order, memory)
    basis_values = [basis.evaluate(lags[:, lag]) for lag in range(memory)]
    features = np.ones((lags.shape[0], len(indices)), dtype=float)
    for col, idx in enumerate(indices):
        values = np.ones(lags.shape[0], dtype=float)
        for lag, degree in enumerate(idx):
            values *= basis_values[lag][:, degree]
        features[:, col] = values
    return features, indices


def fit_projection_model(
    x: np.ndarray,
    y: np.ndarray,
    basis: Basis,
    *,
    order: int,
    memory: int = 1,
    method: str = "projection",
) -> FitResult:
    """Fit coefficients by diagonal projection in the supplied basis."""

    y_arr = np.asarray(y, dtype=float)
    features, indices = tensor_features(x, basis, order, memory)
    target = y_arr[memory - 1 :]
    if target.shape[0] != features.shape[0]:
        raise ValueError("x and y lengths do not align")

    if method == "projection":
        denom = np.sum(features * features, axis=0)
        if np.any(denom <= 0):
            raise np.linalg.LinAlgError("feature has zero empirical norm")
        coefficients = (features.T @ target) / denom
    elif method == "least_squares":
        coefficients = np.linalg.lstsq(features, target, rcond=None)[0]
    else:
        raise ValueError("method must be 'projection' or 'least_squares'")

    return FitResult(
        coefficients=np.asarray(coefficients, dtype=float),
        multi_indices=indices,
        basis=basis,
        memory=memory,
        order=order,
        method=method,
    )


def fit_vwk_volterra(
    x: np.ndarray,
    y: np.ndarray,
    *,
    order: int,
    memory: int = 1,
    moments: np.ndarray | None = None,
    method: str = "projection",
) -> FitResult:
    """Fit a finite-memory VWK model using supplied or empirical moments."""

    if moments is None:
        moments = empirical_moments(np.asarray(x, dtype=float), 2 * order)
    basis = build_vwk_basis(moments, order, name="vwk")
    return fit_projection_model(x, y, basis, order=order, memory=memory, method=method)


def fit_wiener_baseline(
    x: np.ndarray,
    y: np.ndarray,
    *,
    order: int,
    memory: int = 1,
    variance: float | None = None,
    method: str = "projection",
) -> FitResult:
    """Fit a misspecified Gaussian/Wiener projection baseline."""

    x_arr = np.asarray(x, dtype=float)
    if variance is None:
        variance = float(np.mean((x_arr - np.mean(x_arr)) ** 2))
    moments = normal_moments(2 * order, mean=0.0, variance=variance)
    basis = build_vwk_basis(moments, order, name="wiener-gaussian")
    return fit_projection_model(x, y, basis, order=order, memory=memory, method=method)


def mean_squared_signal_error(fit: FitResult, x: np.ndarray, truth: np.ndarray) -> float:
    """MSE between fitted noiseless signal and a provided truth array."""

    pred = fit.predict(x)
    aligned_truth = np.asarray(truth, dtype=float)[fit.memory - 1 :]
    return float(np.mean((pred - aligned_truth) ** 2))
