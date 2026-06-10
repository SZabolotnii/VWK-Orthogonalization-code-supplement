#!/usr/bin/env python3
"""Synthetic H4 grid for finite-moment non-Gaussian VWK recovery.

The VWK model is compared with a misspecified Gaussian/Wiener diagonal projection.
An empirical Huber monomial baseline is reported as a guardrail: it can fit the
same finite polynomial space and therefore prevents overclaiming VWK as a
universal predictor. The H4 gate is about VWK vs misspecified Wiener.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path

import numpy as np

from vwk import (
    build_vwk_basis,
    centered_exponential_moments,
    fit_vwk_volterra,
    fit_wiener_baseline,
    normal_moments,
    symmetric_gaussian_mixture_moments,
    uniform_moments,
)
from vwk.volterra import mean_squared_signal_error

REPORT_DATE = "2026-06-08"


@dataclass(frozen=True)
class Regime:
    name: str
    moments: np.ndarray
    sampler: str
    variance: float


@dataclass(frozen=True)
class ReplicateResult:
    regime: str
    rep: int
    n: int
    noise_sd: float
    vwk_mse: float
    wiener_mse: float
    huber_mse: float
    mse_ratio: float
    vwk_beta0: float
    vwk_beta1: float
    vwk_beta2: float
    wiener_beta0: float
    wiener_beta1: float
    wiener_beta2: float
    vwk_beta0_se: float
    vwk_beta1_se: float
    vwk_beta2_se: float
    vwk_beta0_cover95: bool
    vwk_beta1_cover95: bool
    vwk_beta2_cover95: bool


@dataclass(frozen=True)
class RegimeSummary:
    name: str
    reps: int
    n: int
    noise_sd: float
    vwk_mse: float
    wiener_mse: float
    huber_mse: float
    mse_ratio: float
    vwk_beta2_bias: float
    wiener_beta2_bias: float
    vwk_beta2_coverage95: float
    vwk_all_beta_coverage95: float
    vwk_beta2_mean_ci_width95: float


def regimes(order: int) -> list[Regime]:
    unit_uniform_half_width = sqrt(3.0)
    mix_weight = 0.9
    mix_var_low = 0.25
    mix_var_high = 4.0
    mix_variance = mix_weight * mix_var_low + (1.0 - mix_weight) * mix_var_high
    return [
        Regime("gaussian_control", normal_moments(2 * order), "normal", 1.0),
        Regime("centered_exponential_skew", centered_exponential_moments(2 * order), "centered_exp", 1.0),
        Regime(
            "uniform_platykurtic",
            uniform_moments(2 * order, -unit_uniform_half_width, unit_uniform_half_width),
            "uniform_unit_var",
            1.0,
        ),
        Regime(
            "symmetric_contaminated",
            symmetric_gaussian_mixture_moments(
                2 * order, weight=mix_weight, variance_low=mix_var_low, variance_high=mix_var_high
            ),
            "gaussian_mixture",
            mix_variance,
        ),
    ]


def sample_regime(rng: np.random.Generator, regime: Regime, n: int) -> np.ndarray:
    if regime.sampler == "normal":
        return rng.normal(size=n)
    if regime.sampler == "centered_exp":
        return rng.exponential(scale=1.0, size=n) - 1.0
    if regime.sampler == "uniform_unit_var":
        return rng.uniform(-sqrt(3.0), sqrt(3.0), size=n)
    if regime.sampler == "gaussian_mixture":
        mask = rng.random(n) < 0.9
        x = np.empty(n, dtype=float)
        x[mask] = rng.normal(scale=sqrt(0.25), size=int(np.sum(mask)))
        x[~mask] = rng.normal(scale=sqrt(4.0), size=int(np.sum(~mask)))
        return x
    raise ValueError(f"unknown sampler: {regime.sampler}")


def monomial_design(x: np.ndarray, order: int) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    return np.column_stack([x_arr**k for k in range(order + 1)])


def huber_weights(residuals: np.ndarray, delta: float) -> np.ndarray:
    abs_res = np.abs(residuals)
    weights = np.ones_like(abs_res)
    mask = abs_res > delta
    weights[mask] = delta / abs_res[mask]
    return weights


def fit_huber_monomial(
    x: np.ndarray, y: np.ndarray, *, order: int, delta: float = 1.345, max_iter: int = 40
) -> np.ndarray:
    """Small IRLS Huber baseline for one-dimensional monomial features."""

    design = monomial_design(x, order)
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(max_iter):
        residuals = y - design @ coef
        scale = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
        if scale <= 1e-12:
            break
        weights = huber_weights(residuals / scale, delta)
        sqrt_w = np.sqrt(weights)
        next_coef = np.linalg.lstsq(design * sqrt_w[:, None], y * sqrt_w, rcond=None)[0]
        if np.linalg.norm(next_coef - coef) <= 1e-10 * (1.0 + np.linalg.norm(coef)):
            coef = next_coef
            break
        coef = next_coef
    return coef


def coefficient_se(y: np.ndarray, features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Ratio-estimator SE for population projection coefficients.

    The diagonal projection estimates beta_k = E[psi_k Y] / E[psi_k^2].
    With random X, empirical cross-products add variability beyond additive
    observation noise, so a residual-only conditional SE is too narrow.
    """

    y_arr = np.asarray(y, dtype=float)
    coef_arr = np.asarray(coefficients, dtype=float)
    denom_mean = np.mean(features * features, axis=0)
    if np.any(denom_mean <= 0):
        raise np.linalg.LinAlgError("non-positive feature norm in SE calculation")
    scores = features * y_arr[:, None] - coef_arr[None, :] * (features * features)
    return np.sqrt(np.var(scores, axis=0, ddof=1) / features.shape[0]) / denom_mean


def run_once(
    rng: np.random.Generator,
    regime: Regime,
    rep: int,
    n: int,
    noise_sd: float,
    beta: np.ndarray,
) -> ReplicateResult:
    order = beta.size - 1
    true_basis = build_vwk_basis(regime.moments, order, name=regime.name)

    x = sample_regime(rng, regime, n)
    psi = true_basis.evaluate(x)
    signal = psi @ beta
    y = signal + rng.normal(scale=noise_sd, size=n)

    vwk_fit = fit_vwk_volterra(x, y, order=order, moments=regime.moments, method="projection")
    wiener_fit = fit_wiener_baseline(x, y, order=order, variance=regime.variance, method="projection")
    huber_coef = fit_huber_monomial(x, y, order=order)
    huber_pred = monomial_design(x, order) @ huber_coef

    vwk_mse = mean_squared_signal_error(vwk_fit, x, signal)
    wiener_mse = mean_squared_signal_error(wiener_fit, x, signal)
    huber_mse = float(np.mean((huber_pred - signal) ** 2))
    se = coefficient_se(y, psi, vwk_fit.coefficients)
    cover = np.abs(vwk_fit.coefficients - beta) <= 1.96 * se

    return ReplicateResult(
        regime=regime.name,
        rep=rep,
        n=n,
        noise_sd=noise_sd,
        vwk_mse=vwk_mse,
        wiener_mse=wiener_mse,
        huber_mse=huber_mse,
        mse_ratio=wiener_mse / vwk_mse,
        vwk_beta0=float(vwk_fit.coefficients[0]),
        vwk_beta1=float(vwk_fit.coefficients[1]),
        vwk_beta2=float(vwk_fit.coefficients[2]),
        wiener_beta0=float(wiener_fit.coefficients[0]),
        wiener_beta1=float(wiener_fit.coefficients[1]),
        wiener_beta2=float(wiener_fit.coefficients[2]),
        vwk_beta0_se=float(se[0]),
        vwk_beta1_se=float(se[1]),
        vwk_beta2_se=float(se[2]),
        vwk_beta0_cover95=bool(cover[0]),
        vwk_beta1_cover95=bool(cover[1]),
        vwk_beta2_cover95=bool(cover[2]),
    )


def summarize_replicates(rows: list[ReplicateResult], beta: np.ndarray) -> RegimeSummary:
    if not rows:
        raise ValueError("rows must not be empty")
    vwk_mse = float(np.mean([row.vwk_mse for row in rows]))
    wiener_mse = float(np.mean([row.wiener_mse for row in rows]))
    huber_mse = float(np.mean([row.huber_mse for row in rows]))
    cover_matrix = np.asarray(
        [[row.vwk_beta0_cover95, row.vwk_beta1_cover95, row.vwk_beta2_cover95] for row in rows], dtype=float
    )
    return RegimeSummary(
        name=rows[0].regime,
        reps=len(rows),
        n=rows[0].n,
        noise_sd=rows[0].noise_sd,
        vwk_mse=vwk_mse,
        wiener_mse=wiener_mse,
        huber_mse=huber_mse,
        mse_ratio=wiener_mse / vwk_mse,
        vwk_beta2_bias=float(np.mean([row.vwk_beta2 - beta[2] for row in rows])),
        wiener_beta2_bias=float(np.mean([row.wiener_beta2 - beta[2] for row in rows])),
        vwk_beta2_coverage95=float(np.mean(cover_matrix[:, 2])),
        vwk_all_beta_coverage95=float(np.mean(cover_matrix)),
        vwk_beta2_mean_ci_width95=float(np.mean([2 * 1.96 * row.vwk_beta2_se for row in rows])),
    )


def run_grid(reps: int, n: int, noise_sd: float, seed: int) -> tuple[list[RegimeSummary], list[ReplicateResult]]:
    order = 2
    beta = np.array([0.2, 0.8, 0.6])
    summaries: list[RegimeSummary] = []
    all_rows: list[ReplicateResult] = []
    for offset, regime in enumerate(regimes(order)):
        rng = np.random.default_rng(seed + 1000 * offset)
        rows = [run_once(rng, regime, rep, n, noise_sd, beta) for rep in range(reps)]
        all_rows.extend(rows)
        summaries.append(summarize_replicates(rows, beta))
    return summaries, all_rows


def write_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        raise ValueError("cannot write empty rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    dictionaries = [asdict(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dictionaries[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(dictionaries)


def write_report(
    path: Path,
    summaries: list[RegimeSummary],
    *,
    label: str,
    reps: int,
    n: int,
    noise_sd: float,
    seed: int,
) -> None:
    gaussian = next(item for item in summaries if item.name == "gaussian_control")
    non_gaussian = [item for item in summaries if item.name != "gaussian_control"]
    best = max(non_gaussian, key=lambda item: item.mse_ratio)
    lines = [
        f"# Synthetic H4 {label.title()} Report",
        "",
        f"*Дата: {REPORT_DATE} | reps={reps} | n={n} | noise_sd={noise_sd} | seed={seed}*",
        "",
        "## Summary",
        "",
        "Gate definition: VWK must beat the misspecified Gaussian/Wiener projection in at least one finite-moment non-Gaussian regime, while the Gaussian control should show no artificial gain.",
        "",
        f"- Gaussian control W/V = `{gaussian.mse_ratio:.3f}`.",
        f"- Best non-Gaussian W/V = `{best.mse_ratio:.3f}` in `{best.name}`.",
        "- Huber monomial is reported as a guardrail baseline, not as the H4 pass criterion.",
        "",
        "## Results",
        "",
        "| Regime | VWK MSE | Wiener MSE | Huber MSE | W/V | VWK beta2 coverage | VWK all-beta coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item.name} | {item.vwk_mse:.6g} | {item.wiener_mse:.6g} | {item.huber_mse:.6g} | "
            f"{item.mse_ratio:.3f} | {item.vwk_beta2_coverage95:.3f} | {item.vwk_all_beta_coverage95:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "Current H4 support is asymmetric finite-moment only: the centered-exponential skew regime passes strongly, while Gaussian, platykurtic uniform, and symmetric contaminated controls do not show a W/V advantage in this setup.",
            "Do not use this artifact to claim universal prediction superiority over robust or likelihood baselines.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(summaries: list[RegimeSummary], reps: int, n: int, noise_sd: float) -> tuple[bool, bool]:
    print("Synthetic H4 grid: finite-moment inputs, order=2, memory=1")
    print(f"reps={reps} n={n} noise_sd={noise_sd:.3f}")
    print()
    print(
        "regime                         "
        "vwk_mse      wiener_mse   huber_mse    W/V    cov2   cov_all   bias2_vwk   bias2_wiener"
    )
    print("-" * 118)
    for summary in summaries:
        print(
            f"{summary.name:<30}"
            f"{summary.vwk_mse:11.5g} {summary.wiener_mse:12.5g} {summary.huber_mse:11.5g} "
            f"{summary.mse_ratio:6.2f} {summary.vwk_beta2_coverage95:7.2f} "
            f"{summary.vwk_all_beta_coverage95:9.2f} {summary.vwk_beta2_bias:11.5g} "
            f"{summary.wiener_beta2_bias:14.5g}"
        )

    gaussian = next(item for item in summaries if item.name == "gaussian_control")
    non_gaussian = [item for item in summaries if item.name != "gaussian_control"]
    h4_pass = any(item.mse_ratio > 1.2 for item in non_gaussian)
    gaussian_control_ok = 0.8 <= gaussian.mse_ratio <= 1.25
    print()
    print("Checks:")
    print(f"  Gaussian control W/V near 1     : {'PASS' if gaussian_control_ok else 'FAIL'} ({gaussian.mse_ratio:.3f})")
    print(f"  Any non-Gaussian W/V > 1.2      : {'PASS' if h4_pass else 'FAIL'}")
    print("  Huber baseline reported only as guardrail; it is not the H4 pass criterion.")
    print("RESULT:", "PASS" if (h4_pass and gaussian_control_ok) else "FAIL")
    return h4_pass, gaussian_control_ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Use the quick preregistered pilot grid.")
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--noise-sd", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--write", action="store_true", help="Write frozen CSV and Markdown report artifacts.")
    parser.add_argument("--out-dir", default="code/experiments/results", help="Output directory for --write.")
    args = parser.parse_args()

    reps = args.reps if args.reps is not None else (60 if args.pilot else 200)
    n = args.n if args.n is not None else (800 if args.pilot else 2000)

    summaries, replicate_rows = run_grid(reps, n, args.noise_sd, args.seed)
    h4_pass, gaussian_control_ok = print_summary(summaries, reps, n, args.noise_sd)

    if args.write:
        label = "pilot" if args.pilot else "production"
        out_dir = Path(args.out_dir)
        summary_path = out_dir / f"synthetic_h4_{label}_summary_{REPORT_DATE}.csv"
        replicate_path = out_dir / f"synthetic_h4_{label}_replicates_{REPORT_DATE}.csv"
        report_path = out_dir / f"synthetic_h4_{label}_report_{REPORT_DATE}.md"
        write_csv(summary_path, summaries)
        write_csv(replicate_path, replicate_rows)
        write_report(report_path, summaries, label=label, reps=reps, n=n, noise_sd=args.noise_sd, seed=args.seed)
        print()
        print("Wrote artifacts:")
        print(f"  {summary_path}")
        print(f"  {replicate_path}")
        print(f"  {report_path}")

    return 0 if (h4_pass and gaussian_control_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
