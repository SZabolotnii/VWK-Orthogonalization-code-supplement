"""Conditioning of power, Wiener, and matched VWK coordinates.

The population VWK Gram is the identity by construction.  This script also
measures the finite-sample design Gram, which is the object relevant for
estimation, and compares fixed-relative and CV-tuned ridge fits in the power and
matched bases.

Run:
    PYTHONPATH=code uv run --with numpy python code/experiments/run_conditioning.py --write --empirical-gram --ridge-cv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from vwk.basis import (
    Basis,
    build_vwk_basis,
    centered_exponential_moments,
    normal_moments,
    symmetric_gaussian_mixture_moments,
    uniform_moments,
)

REPORT_DATE = "2026-06-09"
SEED = 20260608
REGIMES = ["gaussian", "centered_exponential", "uniform", "contaminated"]
ORDERS = [2, 3, 4, 5, 6]
RIDGE_GRID = np.array([0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], dtype=float)


@dataclass(frozen=True)
class ConditioningRow:
    regime: str
    order: int
    population_power_cond: float
    population_vwk_cond: float
    vwk_orthogonality_error: float
    empirical_n: int
    empirical_reps: int
    empirical_power_cond_median: float
    empirical_power_cond_p90: float
    empirical_vwk_cond_median: float
    empirical_vwk_cond_p90: float


@dataclass(frozen=True)
class RidgeRow:
    regime: str
    order: int
    n_train: int
    n_test: int
    reps: int
    noise_sd: float
    fixed_eta: float
    fixed_power_rmse: float
    fixed_vwk_rmse: float
    fixed_power_vwk_ratio: float
    cv_power_rmse: float
    cv_vwk_rmse: float
    cv_power_vwk_ratio: float
    cv_power_eta_median: float
    cv_vwk_eta_median: float


def regime_moments(name: str, max_order: int) -> np.ndarray:
    if name == "gaussian":
        return normal_moments(max_order, mean=0.0, variance=1.0)
    if name == "centered_exponential":
        return centered_exponential_moments(max_order)
    if name == "uniform":
        a = np.sqrt(3.0)
        return uniform_moments(max_order, low=-a, high=a)
    if name == "contaminated":
        return symmetric_gaussian_mixture_moments(max_order)
    raise ValueError(name)


def sample_regime(name: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if name == "gaussian":
        return rng.standard_normal(n)
    if name == "centered_exponential":
        return rng.exponential(1.0, n) - 1.0
    if name == "uniform":
        a = np.sqrt(3.0)
        return rng.uniform(-a, a, n)
    if name == "contaminated":
        comp = rng.random(n) < 0.9
        return np.where(comp, rng.normal(0.0, 0.5, n), rng.normal(0.0, 2.0, n))
    raise ValueError(name)


def hankel(moments: np.ndarray, order: int) -> np.ndarray:
    return np.array([[moments[i + j] for j in range(order + 1)] for i in range(order + 1)], dtype=float)


def power_design(x: np.ndarray, order: int) -> np.ndarray:
    return np.vander(np.asarray(x, dtype=float), N=order + 1, increasing=True)


def gram_cond(features: np.ndarray) -> float:
    gram = features.T @ features / features.shape[0]
    return float(np.linalg.cond(gram))


def conditioning_rows(empirical_n: int, empirical_reps: int) -> list[ConditioningRow]:
    rng = np.random.default_rng(SEED)
    rows: list[ConditioningRow] = []
    for name in REGIMES:
        moments = regime_moments(name, 2 * max(ORDERS))
        for order in ORDERS:
            basis = build_vwk_basis(moments, order, name="vwk")
            population_power_cond = float(np.linalg.cond(hankel(moments, order)))
            population_gram = basis.gram()
            population_vwk_cond = float(np.linalg.cond(population_gram))
            orth_error = float(np.max(np.abs(population_gram - np.eye(order + 1))))
            empirical_power: list[float] = []
            empirical_vwk: list[float] = []
            for _ in range(empirical_reps):
                x = sample_regime(name, empirical_n, rng)
                empirical_power.append(gram_cond(power_design(x, order)))
                empirical_vwk.append(gram_cond(basis.evaluate(x)))
            rows.append(
                ConditioningRow(
                    regime=name,
                    order=order,
                    population_power_cond=population_power_cond,
                    population_vwk_cond=population_vwk_cond,
                    vwk_orthogonality_error=orth_error,
                    empirical_n=empirical_n,
                    empirical_reps=empirical_reps,
                    empirical_power_cond_median=float(np.median(empirical_power)),
                    empirical_power_cond_p90=float(np.quantile(empirical_power, 0.9)),
                    empirical_vwk_cond_median=float(np.median(empirical_vwk)),
                    empirical_vwk_cond_p90=float(np.quantile(empirical_vwk, 0.9)),
                )
            )
    return rows


def solve_ridge(features: np.ndarray, y: np.ndarray, eta: float) -> np.ndarray:
    gram = features.T @ features
    penalty = eta * np.trace(gram) / features.shape[1]
    return np.linalg.solve(gram + penalty * np.eye(features.shape[1]), features.T @ y)


def choose_eta(train_x: np.ndarray, train_y: np.ndarray, basis: Basis | str, order: int) -> float:
    split = max(10, int(0.7 * train_x.size))
    x_fit, x_val = train_x[:split], train_x[split:]
    y_fit, y_val = train_y[:split], train_y[split:]
    fit_features = power_design(x_fit, order) if basis == "power" else basis.evaluate(x_fit)
    val_features = power_design(x_val, order) if basis == "power" else basis.evaluate(x_val)
    best_eta = float(RIDGE_GRID[0])
    best_mse = float("inf")
    for eta in RIDGE_GRID:
        coef = solve_ridge(fit_features, y_fit, float(eta))
        mse = float(np.mean((val_features @ coef - y_val) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_eta = float(eta)
    return best_eta


def ridge_rmse(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    truth_test: np.ndarray,
    basis: Basis | str,
    order: int,
    eta: float,
) -> float:
    train_features = power_design(train_x, order) if basis == "power" else basis.evaluate(train_x)
    test_features = power_design(test_x, order) if basis == "power" else basis.evaluate(test_x)
    coef = solve_ridge(train_features, train_y, eta)
    return float(np.sqrt(np.mean((test_features @ coef - truth_test) ** 2)))


def ridge_rows(n_train: int, n_test: int, reps: int, noise_sd: float, fixed_eta: float) -> list[RidgeRow]:
    rng = np.random.default_rng(SEED + 100000)
    rows: list[RidgeRow] = []
    for name in REGIMES:
        moments = regime_moments(name, 2 * max(ORDERS))
        for order in [3, 5]:
            basis = build_vwk_basis(moments, order, name="vwk")
            true_a = rng.standard_normal(order + 1)
            fixed_power: list[float] = []
            fixed_vwk: list[float] = []
            cv_power: list[float] = []
            cv_vwk: list[float] = []
            cv_power_eta: list[float] = []
            cv_vwk_eta: list[float] = []
            for _ in range(reps):
                train_x = sample_regime(name, n_train, rng)
                test_x = sample_regime(name, n_test, rng)
                truth_train = basis.evaluate(train_x) @ true_a
                truth_test = basis.evaluate(test_x) @ true_a
                train_y = truth_train + noise_sd * rng.standard_normal(n_train)

                fixed_power.append(ridge_rmse(train_x, train_y, test_x, truth_test, "power", order, fixed_eta))
                fixed_vwk.append(ridge_rmse(train_x, train_y, test_x, truth_test, basis, order, fixed_eta))

                eta_power = choose_eta(train_x, train_y, "power", order)
                eta_vwk = choose_eta(train_x, train_y, basis, order)
                cv_power_eta.append(eta_power)
                cv_vwk_eta.append(eta_vwk)
                cv_power.append(ridge_rmse(train_x, train_y, test_x, truth_test, "power", order, eta_power))
                cv_vwk.append(ridge_rmse(train_x, train_y, test_x, truth_test, basis, order, eta_vwk))

            fixed_power_rmse = float(np.mean(fixed_power))
            fixed_vwk_rmse = float(np.mean(fixed_vwk))
            cv_power_rmse = float(np.mean(cv_power))
            cv_vwk_rmse = float(np.mean(cv_vwk))
            rows.append(
                RidgeRow(
                    regime=name,
                    order=order,
                    n_train=n_train,
                    n_test=n_test,
                    reps=reps,
                    noise_sd=noise_sd,
                    fixed_eta=fixed_eta,
                    fixed_power_rmse=fixed_power_rmse,
                    fixed_vwk_rmse=fixed_vwk_rmse,
                    fixed_power_vwk_ratio=fixed_power_rmse / fixed_vwk_rmse,
                    cv_power_rmse=cv_power_rmse,
                    cv_vwk_rmse=cv_vwk_rmse,
                    cv_power_vwk_ratio=cv_power_rmse / cv_vwk_rmse,
                    cv_power_eta_median=float(np.median(cv_power_eta)),
                    cv_vwk_eta_median=float(np.median(cv_vwk_eta)),
                )
            )
    return rows


def write_csv(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dictionaries = [asdict(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dictionaries[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(dictionaries)


def print_conditioning(rows: list[ConditioningRow]) -> None:
    print("(A) Conditioning: population Hankel and finite-sample design Gram")
    print(
        f"{'regime':22s} {'s':>2s} {'pop power':>11s} {'emp power med':>14s} "
        f"{'emp VWK med':>12s} {'emp VWK p90':>11s}"
    )
    for row in rows:
        print(
            f"{row.regime:22s} {row.order:2d} {row.population_power_cond:11.3g} "
            f"{row.empirical_power_cond_median:14.3g} {row.empirical_vwk_cond_median:12.3g} "
            f"{row.empirical_vwk_cond_p90:11.3g}"
        )


def print_ridge(rows: list[RidgeRow]) -> None:
    print("\n(B) Ridge prediction RMSE; fixed eta and per-basis CV eta")
    print(
        f"{'regime':22s} {'s':>2s} {'fixed P/V':>10s} {'CV P/V':>8s} "
        f"{'CV power':>10s} {'CV VWK':>10s} {'eta P':>8s} {'eta V':>8s}"
    )
    for row in rows:
        print(
            f"{row.regime:22s} {row.order:2d} {row.fixed_power_vwk_ratio:10.2f} "
            f"{row.cv_power_vwk_ratio:8.2f} {row.cv_power_rmse:10.5f} "
            f"{row.cv_vwk_rmse:10.5f} {row.cv_power_eta_median:8.1e} {row.cv_vwk_eta_median:8.1e}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--empirical-gram", action="store_true", help="Accepted for audit readability; enabled by default.")
    parser.add_argument("--ridge-cv", action="store_true", help="Accepted for audit readability; enabled by default.")
    parser.add_argument("--empirical-n", type=int, default=2000)
    parser.add_argument("--empirical-reps", type=int, default=40)
    parser.add_argument("--ridge-n-train", type=int, default=2800)
    parser.add_argument("--ridge-n-test", type=int, default=1200)
    parser.add_argument("--ridge-reps", type=int, default=40)
    parser.add_argument("--noise-sd", type=float, default=0.25)
    parser.add_argument("--fixed-eta", type=float, default=1e-2)
    parser.add_argument("--out-dir", default="code/experiments/results")
    args = parser.parse_args()

    cond_rows = conditioning_rows(args.empirical_n, args.empirical_reps)
    ridge = ridge_rows(args.ridge_n_train, args.ridge_n_test, args.ridge_reps, args.noise_sd, args.fixed_eta)
    print_conditioning(cond_rows)
    print_ridge(ridge)

    if args.write:
        root = Path(__file__).resolve().parents[2]
        out_dir = root / args.out_dir
        cond_path = out_dir / f"conditioning_empirical_{REPORT_DATE}.csv"
        ridge_path = out_dir / f"ridge_cv_{REPORT_DATE}.csv"
        write_csv(cond_path, cond_rows)
        write_csv(ridge_path, ridge)
        print()
        print("Wrote artifacts:")
        print(f"  {cond_path.relative_to(root)}")
        print(f"  {ridge_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
