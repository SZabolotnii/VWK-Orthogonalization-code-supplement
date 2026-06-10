"""Finite-memory VWK utilities for Paper 3a."""

from .basis import (
    Basis,
    build_vwk_basis,
    centered_exponential_moments,
    empirical_moments,
    normal_moments,
    symmetric_gaussian_mixture_moments,
    uniform_moments,
)
from .volterra import FitResult, fit_projection_model, fit_vwk_volterra, fit_wiener_baseline

__all__ = [
    "Basis",
    "FitResult",
    "build_vwk_basis",
    "centered_exponential_moments",
    "empirical_moments",
    "fit_projection_model",
    "fit_vwk_volterra",
    "fit_wiener_baseline",
    "normal_moments",
    "symmetric_gaussian_mixture_moments",
    "uniform_moments",
]
