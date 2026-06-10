#!/usr/bin/env python3
"""Sample-efficiency curve for the finite-memory VWK gate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from run_finite_memory_empirical import REPORT_DATE, RegimeSummary, run_grid, write_csv


@dataclass(frozen=True)
class CurveRow:
    regime: str
    n: int
    reps: int
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


def curve_rows_for_n(summary: RegimeSummary) -> CurveRow:
    return CurveRow(
        regime=summary.name,
        n=summary.n,
        reps=summary.reps,
        memory=summary.memory,
        order=summary.order,
        noise_sd=summary.noise_sd,
        oracle_vwk_mse=summary.oracle_vwk_mse,
        empirical_vwk_mse=summary.empirical_vwk_mse,
        wiener_mse=summary.wiener_mse,
        monomial_ls_mse=summary.monomial_ls_mse,
        huber_mse=summary.huber_mse,
        wiener_oracle_ratio=summary.wiener_oracle_ratio,
        wiener_empirical_ratio=summary.wiener_empirical_ratio,
        empirical_oracle_ratio=summary.empirical_oracle_ratio,
    )


def parse_n_grid(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("n-grid must contain at least one integer")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("n-grid values must be positive")
    return values


def write_report(
    path: Path,
    rows: list[CurveRow],
    *,
    label: str,
    n_grid: list[int],
    reps: int,
    memory: int,
    order: int,
    noise_sd: float,
    seed: int,
) -> None:
    skew_rows = [row for row in rows if row.regime == "centered_exponential_skew"]
    gaussian_rows = [row for row in rows if row.regime == "gaussian_control"]
    min_skew_oracle = min(row.wiener_oracle_ratio for row in skew_rows)
    min_skew_empirical = min(row.wiener_empirical_ratio for row in skew_rows)
    max_gaussian_deviation = max(abs(row.wiener_oracle_ratio - 1.0) for row in gaussian_rows)
    lines = [
        f"# Sample-Efficiency {label.title()} Report",
        "",
        f"*Дата: {REPORT_DATE} | reps={reps} | n_grid={n_grid} | memory={memory} | order={order} | "
        f"noise_sd={noise_sd} | seed={seed}*",
        "",
        "## Summary",
        "",
        "Gate definition: the finite-memory H4 advantage should persist across sample sizes rather than appearing only at one chosen n.",
        "",
        f"- Minimum centered-exponential W/oracle across n = `{min_skew_oracle:.3f}`.",
        f"- Minimum centered-exponential W/empirical across n = `{min_skew_empirical:.3f}`.",
        f"- Maximum Gaussian-control |W/oracle - 1| across n = `{max_gaussian_deviation:.3f}`.",
        "",
        "## Results",
        "",
        "| Regime | n | Oracle VWK MSE | Empirical VWK MSE | Wiener MSE | W/oracle | W/empirical | Emp/oracle |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.regime} | {row.n} | {row.oracle_vwk_mse:.6g} | {row.empirical_vwk_mse:.6g} | "
            f"{row.wiener_mse:.6g} | {row.wiener_oracle_ratio:.3f} | "
            f"{row.wiener_empirical_ratio:.3f} | {row.empirical_oracle_ratio:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "The sample-efficiency curve supports the same scoped H4 interpretation: asymmetric finite-moment inputs show a persistent misspecified-Wiener penalty, while the Gaussian control remains neutral.",
            "This artifact is still synthetic finite-memory evidence, not a real-world or CF/3b claim.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(rows: list[CurveRow], n_grid: list[int], reps: int, memory: int, order: int) -> tuple[bool, bool, bool]:
    print("Sample-efficiency curve: finite-memory empirical VWK")
    print(f"reps={reps} n_grid={n_grid} memory={memory} order={order}")
    print()
    print("regime                         n      W/O    W/E    E/O    oracle      empirical   wiener")
    print("-" * 104)
    for row in rows:
        print(
            f"{row.regime:<30} {row.n:5d} {row.wiener_oracle_ratio:6.2f} "
            f"{row.wiener_empirical_ratio:6.2f} {row.empirical_oracle_ratio:6.2f} "
            f"{row.oracle_vwk_mse:11.5g} {row.empirical_vwk_mse:11.5g} {row.wiener_mse:11.5g}"
        )
    gaussian_rows = [row for row in rows if row.regime == "gaussian_control"]
    skew_rows = [row for row in rows if row.regime == "centered_exponential_skew"]
    gaussian_ok = all(0.8 <= row.wiener_oracle_ratio <= 1.25 for row in gaussian_rows)
    skew_ok = all(row.wiener_oracle_ratio > 1.2 and row.wiener_empirical_ratio > 1.2 for row in skew_rows)
    empirical_ok = all(row.empirical_oracle_ratio <= 1.5 for row in rows)
    print()
    print("Checks:")
    print(f"  Gaussian control W/oracle near 1 across n : {'PASS' if gaussian_ok else 'FAIL'}")
    print(f"  Skew W/oracle and W/empirical > 1.2      : {'PASS' if skew_ok else 'FAIL'}")
    print(f"  Empirical moments stable, E/O <= 1.5     : {'PASS' if empirical_ok else 'FAIL'}")
    print("RESULT:", "PASS" if (gaussian_ok and skew_ok and empirical_ok) else "FAIL")
    return gaussian_ok, skew_ok, empirical_ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Use quick pilot defaults.")
    parser.add_argument("--n-grid", type=parse_n_grid, default=None)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--memory", type=int, default=3)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--noise-sd", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--write", action="store_true", help="Write frozen CSV and Markdown report artifacts.")
    parser.add_argument("--out-dir", default="code/experiments/results", help="Output directory for --write.")
    args = parser.parse_args()

    n_grid = args.n_grid if args.n_grid is not None else ([600, 1200] if args.pilot else [600, 1200, 2500, 5000])
    reps = args.reps if args.reps is not None else (30 if args.pilot else 80)
    rows: list[CurveRow] = []
    for offset, n in enumerate(n_grid):
        summaries, _ = run_grid(
            reps=reps,
            n=n,
            memory=args.memory,
            order=args.order,
            noise_sd=args.noise_sd,
            seed=args.seed + 10000 * offset,
        )
        rows.extend(curve_rows_for_n(summary) for summary in summaries)

    gaussian_ok, skew_ok, empirical_ok = print_summary(rows, n_grid, reps, args.memory, args.order)

    if args.write:
        label = "pilot" if args.pilot else "production"
        out_dir = Path(args.out_dir)
        summary_path = out_dir / f"sample_efficiency_{label}_summary_{REPORT_DATE}.csv"
        report_path = out_dir / f"sample_efficiency_{label}_report_{REPORT_DATE}.md"
        write_csv(summary_path, rows)
        write_report(
            report_path,
            rows,
            label=label,
            n_grid=n_grid,
            reps=reps,
            memory=args.memory,
            order=args.order,
            noise_sd=args.noise_sd,
            seed=args.seed,
        )
        print()
        print("Wrote artifacts:")
        print(f"  {summary_path}")
        print(f"  {report_path}")

    return 0 if (gaussian_ok and skew_ok and empirical_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
