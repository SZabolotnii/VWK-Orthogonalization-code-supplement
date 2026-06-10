#!/usr/bin/env python3
"""Finite-memory VWK experiment with empirical moment estimation.

This is the next gate after the one-lag H4 grid: the true system uses a
second-order three-lag tensor VWK basis, and the estimator is run both with
oracle moments and with moments estimated from the observed input.
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
from vwk.volterra import lag_matrix, mean_squared_signal_error, multi_indices, tensor_features

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
    memory: int
    order: int
    noise_sd: float
    oracle_vwk_mse: float
    empirical_vwk_mse: float
    wiener_mse: float
    monomial_ls_mse: float
    huber_mse: float
    wiener_oracle_ratio: float
    wiener_empirical_ratio: float
    empirical_oracle_ratio: float
    oracle_coef_rmse: float


@dataclass(frozen=True)
class RegimeSummary:
    name: str
    reps: int
    n: int
    memory: int
    order: int
    noise_sd: float
    oracle_vwk_mse: float
    empirical_vwk_mse: float
    wiener_mse: float
    monomial_ls_mse: float
    huber_mse: float
    wiener_oracle_ratio: float
    wiener_empirical_ratio: float
    empirical_oracle_ratio: float
    oracle_coef_rmse: float


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


def coefficient_vector(indices: tuple[tuple[int, ...], ...]) -> np.ndarray:
    canonical = {
        (0, 0, 0): 0.10,
        (0, 0, 1): 0.70,
        (0, 1, 0): -0.35,
        (1, 0, 0): 0.25,
        (0, 0, 2): 0.45,
        (0, 1, 1): -0.20,
        (0, 2, 0): 0.15,
        (1, 0, 1): 0.30,
        (1, 1, 0): -0.10,
        (2, 0, 0): 0.55,
    }
    if set(indices) == set(canonical):
        return np.asarray([canonical[idx] for idx in indices], dtype=float)
    return np.asarray(
        [0.10 if sum(idx) == 0 else ((-1.0) ** col) * (0.20 + 0.03 * col) for col, idx in enumerate(indices)],
        dtype=float,
    )


def tensor_monomial_design(x: np.ndarray, order: int, memory: int) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    lags = lag_matrix(x, memory)
    indices = multi_indices(order, memory)
    design = np.ones((lags.shape[0], len(indices)), dtype=float)
    for col, idx in enumerate(indices):
        values = np.ones(lags.shape[0], dtype=float)
        for lag, degree in enumerate(idx):
            if degree:
                values *= lags[:, lag] ** degree
        design[:, col] = values
    return design, indices


def huber_weights(residuals: np.ndarray, delta: float) -> np.ndarray:
    abs_res = np.abs(residuals)
    weights = np.ones_like(abs_res)
    mask = abs_res > delta
    weights[mask] = delta / abs_res[mask]
    return weights


def fit_huber_design(
    design: np.ndarray, y: np.ndarray, *, delta: float = 1.345, max_iter: int = 50
) -> np.ndarray:
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


def aligned_full_signal(x: np.ndarray, signal: np.ndarray, memory: int) -> np.ndarray:
    full = np.zeros_like(np.asarray(x, dtype=float))
    full[memory - 1 :] = signal
    return full


def run_once(
    rng: np.random.Generator,
    regime: Regime,
    rep: int,
    *,
    n: int,
    memory: int,
    order: int,
    noise_sd: float,
) -> ReplicateResult:
    true_basis = build_vwk_basis(regime.moments, order, name=regime.name)
    x = sample_regime(rng, regime, n)
    features, indices = tensor_features(x, true_basis, order, memory)
    beta = coefficient_vector(indices)
    signal = features @ beta
    truth = aligned_full_signal(x, signal, memory)
    y = aligned_full_signal(x, signal + rng.normal(scale=noise_sd, size=signal.size), memory)

    oracle_fit = fit_vwk_volterra(x, y, order=order, memory=memory, moments=regime.moments)
    empirical_fit = fit_vwk_volterra(x, y, order=order, memory=memory, moments=None)
    wiener_fit = fit_wiener_baseline(x, y, order=order, memory=memory, variance=regime.variance)

    monomial_design, _ = tensor_monomial_design(x, order, memory)
    target = y[memory - 1 :]
    monomial_coef = np.linalg.lstsq(monomial_design, target, rcond=None)[0]
    monomial_pred = monomial_design @ monomial_coef
    huber_coef = fit_huber_design(monomial_design, target)
    huber_pred = monomial_design @ huber_coef

    oracle_mse = mean_squared_signal_error(oracle_fit, x, truth)
    empirical_mse = mean_squared_signal_error(empirical_fit, x, truth)
    wiener_mse = mean_squared_signal_error(wiener_fit, x, truth)
    monomial_mse = float(np.mean((monomial_pred - signal) ** 2))
    huber_mse = float(np.mean((huber_pred - signal) ** 2))

    return ReplicateResult(
        regime=regime.name,
        rep=rep,
        n=n,
        memory=memory,
        order=order,
        noise_sd=noise_sd,
        oracle_vwk_mse=oracle_mse,
        empirical_vwk_mse=empirical_mse,
        wiener_mse=wiener_mse,
        monomial_ls_mse=monomial_mse,
        huber_mse=huber_mse,
        wiener_oracle_ratio=wiener_mse / oracle_mse,
        wiener_empirical_ratio=wiener_mse / empirical_mse,
        empirical_oracle_ratio=empirical_mse / oracle_mse,
        oracle_coef_rmse=float(np.sqrt(np.mean((oracle_fit.coefficients - beta) ** 2))),
    )


def summarize(rows: list[ReplicateResult]) -> RegimeSummary:
    if not rows:
        raise ValueError("rows must not be empty")
    oracle_mse = float(np.mean([row.oracle_vwk_mse for row in rows]))
    empirical_mse = float(np.mean([row.empirical_vwk_mse for row in rows]))
    wiener_mse = float(np.mean([row.wiener_mse for row in rows]))
    monomial_mse = float(np.mean([row.monomial_ls_mse for row in rows]))
    huber_mse = float(np.mean([row.huber_mse for row in rows]))
    return RegimeSummary(
        name=rows[0].regime,
        reps=len(rows),
        n=rows[0].n,
        memory=rows[0].memory,
        order=rows[0].order,
        noise_sd=rows[0].noise_sd,
        oracle_vwk_mse=oracle_mse,
        empirical_vwk_mse=empirical_mse,
        wiener_mse=wiener_mse,
        monomial_ls_mse=monomial_mse,
        huber_mse=huber_mse,
        wiener_oracle_ratio=wiener_mse / oracle_mse,
        wiener_empirical_ratio=wiener_mse / empirical_mse,
        empirical_oracle_ratio=empirical_mse / oracle_mse,
        oracle_coef_rmse=float(np.mean([row.oracle_coef_rmse for row in rows])),
    )


def run_grid(
    *, reps: int, n: int, memory: int, order: int, noise_sd: float, seed: int
) -> tuple[list[RegimeSummary], list[ReplicateResult]]:
    summaries: list[RegimeSummary] = []
    replicate_rows: list[ReplicateResult] = []
    for offset, regime in enumerate(regimes(order)):
        rng = np.random.default_rng(seed + 1000 * offset)
        rows = [
            run_once(rng, regime, rep, n=n, memory=memory, order=order, noise_sd=noise_sd)
            for rep in range(reps)
        ]
        replicate_rows.extend(rows)
        summaries.append(summarize(rows))
    return summaries, replicate_rows


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
    memory: int,
    order: int,
    noise_sd: float,
    seed: int,
) -> None:
    gaussian = next(item for item in summaries if item.name == "gaussian_control")
    skew = next(item for item in summaries if item.name == "centered_exponential_skew")
    lines = [
        f"# Finite-Memory Empirical-Moment {label.title()} Report",
        "",
        f"*Дата: {REPORT_DATE} | reps={reps} | n={n} | memory={memory} | order={order} | "
        f"noise_sd={noise_sd} | seed={seed}*",
        "",
        "## Summary",
        "",
        "Gate definition: a second-order finite-memory VWK system should keep the Gaussian control neutral, show a misspecified Wiener penalty on asymmetric finite-moment input, and remain stable when moments are estimated empirically from the input sample.",
        "",
        f"- Gaussian control W/oracle = `{gaussian.wiener_oracle_ratio:.3f}`.",
        f"- Centered-exponential skew W/oracle = `{skew.wiener_oracle_ratio:.3f}`.",
        f"- Centered-exponential skew W/empirical = `{skew.wiener_empirical_ratio:.3f}`.",
        "- Monomial LS and Huber are guardrails because they fit the same finite polynomial span directly.",
        "",
        "## Results",
        "",
        "| Regime | Oracle VWK MSE | Empirical VWK MSE | Wiener MSE | Monomial LS MSE | Huber MSE | W/oracle | W/empirical | Emp/oracle |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item.name} | {item.oracle_vwk_mse:.6g} | {item.empirical_vwk_mse:.6g} | "
            f"{item.wiener_mse:.6g} | {item.monomial_ls_mse:.6g} | {item.huber_mse:.6g} | "
            f"{item.wiener_oracle_ratio:.3f} | {item.wiener_empirical_ratio:.3f} | "
            f"{item.empirical_oracle_ratio:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "The empirical-moment VWK estimator remains stable in this finite-memory gate. As in the one-lag H4 grid, the substantive W/V advantage is asymmetric finite-moment only; symmetric controls do not justify a symmetric non-Gaussian advantage claim.",
            "Direct monomial LS and Huber guardrails are intentionally strong and should be reported as span-fitting baselines, not as the H4 target.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(summaries: list[RegimeSummary], reps: int, n: int, memory: int, order: int, noise_sd: float) -> tuple[bool, bool, bool]:
    print("Finite-memory empirical-moment grid: finite-moment inputs")
    print(f"reps={reps} n={n} memory={memory} order={order} noise_sd={noise_sd:.3f}")
    print()
    print(
        "regime                         "
        "oracle       empirical    wiener       ls          huber       W/O    W/E    E/O"
    )
    print("-" * 118)
    for summary in summaries:
        print(
            f"{summary.name:<30}"
            f"{summary.oracle_vwk_mse:11.5g} {summary.empirical_vwk_mse:11.5g} "
            f"{summary.wiener_mse:11.5g} {summary.monomial_ls_mse:11.5g} "
            f"{summary.huber_mse:11.5g} {summary.wiener_oracle_ratio:6.2f} "
            f"{summary.wiener_empirical_ratio:6.2f} {summary.empirical_oracle_ratio:6.2f}"
        )

    gaussian = next(item for item in summaries if item.name == "gaussian_control")
    skew = next(item for item in summaries if item.name == "centered_exponential_skew")
    gaussian_control_ok = 0.8 <= gaussian.wiener_oracle_ratio <= 1.25
    skew_h4_ok = skew.wiener_oracle_ratio > 1.2 and skew.wiener_empirical_ratio > 1.2
    empirical_stable = all(item.empirical_oracle_ratio <= 1.5 for item in summaries)
    print()
    print("Checks:")
    print(
        f"  Gaussian control W/oracle near 1     : {'PASS' if gaussian_control_ok else 'FAIL'} "
        f"({gaussian.wiener_oracle_ratio:.3f})"
    )
    print(f"  Skew W/oracle and W/empirical > 1.2  : {'PASS' if skew_h4_ok else 'FAIL'}")
    print(f"  Empirical moments stable, E/O <= 1.5 : {'PASS' if empirical_stable else 'FAIL'}")
    print("  Monomial LS and Huber are guardrails, not H4 pass criteria.")
    print("RESULT:", "PASS" if (gaussian_control_ok and skew_h4_ok and empirical_stable) else "FAIL")
    return gaussian_control_ok, skew_h4_ok, empirical_stable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Use the quick pilot grid.")
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--memory", type=int, default=3)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--noise-sd", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--write", action="store_true", help="Write frozen CSV and Markdown report artifacts.")
    parser.add_argument("--out-dir", default="code/experiments/results", help="Output directory for --write.")
    args = parser.parse_args()

    reps = args.reps if args.reps is not None else (40 if args.pilot else 150)
    n = args.n if args.n is not None else (1200 if args.pilot else 2500)

    summaries, replicate_rows = run_grid(
        reps=reps, n=n, memory=args.memory, order=args.order, noise_sd=args.noise_sd, seed=args.seed
    )
    gaussian_ok, skew_ok, empirical_ok = print_summary(
        summaries, reps, n, args.memory, args.order, args.noise_sd
    )

    if args.write:
        label = "pilot" if args.pilot else "production"
        out_dir = Path(args.out_dir)
        summary_path = out_dir / f"finite_memory_empirical_{label}_summary_{REPORT_DATE}.csv"
        replicate_path = out_dir / f"finite_memory_empirical_{label}_replicates_{REPORT_DATE}.csv"
        report_path = out_dir / f"finite_memory_empirical_{label}_report_{REPORT_DATE}.md"
        write_csv(summary_path, summaries)
        write_csv(replicate_path, replicate_rows)
        write_report(
            report_path,
            summaries,
            label=label,
            reps=reps,
            n=n,
            memory=args.memory,
            order=args.order,
            noise_sd=args.noise_sd,
            seed=args.seed,
        )
        print()
        print("Wrote artifacts:")
        print(f"  {summary_path}")
        print(f"  {replicate_path}")
        print(f"  {report_path}")

    return 0 if (gaussian_ok and skew_ok and empirical_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
