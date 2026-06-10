#!/usr/bin/env python3
"""Screen local real-world candidates for a VWK illustration.

The 3a preprint does not need a real-world flagship for its core theorem claim,
but the original problem statement lists a real-world demo as strong evidence.
This script performs a conservative local screen over already-prepared CSV
candidates and returns either a candidate illustration or an explicit no-claim
decision.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vwk import fit_vwk_volterra, fit_wiener_baseline

REPORT_DATE = "2026-06-08"


@dataclass(frozen=True)
class CandidateResult:
    dataset_id: str
    label: str
    path: str
    target: str
    driver: str
    n_total: int
    n_used: int
    train_n: int
    test_n: int
    residual_skew: float
    residual_excess_kurtosis: float
    vwk_test_mse: float
    wiener_test_mse: float
    monomial_ls_test_mse: float
    huber_test_mse: float
    wiener_vwk_ratio: float
    vwk_monomial_ratio: float
    decision: str
    note: str


def parse_formula(formula: str) -> tuple[str, list[str]]:
    if "~" not in formula:
        raise ValueError(f"formula lacks '~': {formula}")
    target, rhs = formula.split("~", 1)
    predictors = [term.strip() for term in rhs.split("+") if term.strip()]
    return target.strip(), predictors


def monomial_design(x: np.ndarray, order: int) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    return np.column_stack([x_arr**k for k in range(order + 1)])


def huber_weights(residuals: np.ndarray, delta: float) -> np.ndarray:
    abs_res = np.abs(residuals)
    weights = np.ones_like(abs_res)
    mask = abs_res > delta
    weights[mask] = delta / abs_res[mask]
    return weights


def fit_huber_monomial(x: np.ndarray, y: np.ndarray, *, order: int, delta: float = 1.345) -> np.ndarray:
    design = monomial_design(x, order)
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(40):
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


def standardized(values: np.ndarray, mean: float, sd: float) -> np.ndarray:
    if sd <= 1e-12:
        raise ValueError("standard deviation is too small")
    return (values - mean) / sd


def residual_moments(residuals: np.ndarray) -> tuple[float, float]:
    centered = residuals - np.mean(residuals)
    sd = float(np.sqrt(np.mean(centered**2)))
    if sd <= 1e-12:
        return 0.0, -3.0
    z = centered / sd
    return float(np.mean(z**3)), float(np.mean(z**4) - 3.0)


def candidate_numeric_frame(df: pd.DataFrame, target: str, predictors: list[str]) -> pd.DataFrame:
    keep = [target] + [col for col in predictors if col in df.columns]
    sub = df.loc[:, keep].copy()
    numeric_cols = [
        col
        for col in keep
        if col in sub.columns and pd.api.types.is_numeric_dtype(sub[col]) and sub[col].notna().sum() >= 100
    ]
    if target not in numeric_cols:
        raise ValueError(f"target is not numeric or has too few observations: {target}")
    return sub.loc[:, numeric_cols].replace([np.inf, -np.inf], np.nan).dropna()


def choose_driver(frame: pd.DataFrame, target: str, predictors: list[str], order: int) -> str:
    candidates = [col for col in predictors if col in frame.columns and col != target]
    if not candidates:
        raise ValueError("no numeric predictor candidates")
    n_train = max(20, int(0.7 * len(frame)))
    best_col = candidates[0]
    best_mse = float("inf")
    y = frame[target].to_numpy(dtype=float)
    y_train = y[:n_train]
    y_mean = float(np.mean(y_train))
    y_sd = float(np.std(y_train))
    y_std = standardized(y, y_mean, y_sd)
    for col in candidates:
        x = frame[col].to_numpy(dtype=float)
        x_train = x[:n_train]
        try:
            x_std = standardized(x, float(np.mean(x_train)), float(np.std(x_train)))
        except ValueError:
            continue
        coef = np.linalg.lstsq(monomial_design(x_std[:n_train], order), y_std[:n_train], rcond=None)[0]
        train_mse = float(np.mean((monomial_design(x_std[:n_train], order) @ coef - y_std[:n_train]) ** 2))
        if train_mse < best_mse:
            best_mse = train_mse
            best_col = col
    if not np.isfinite(best_mse):
        raise ValueError("all numeric predictors are degenerate")
    return best_col


def evaluate_candidate(root: Path, row: dict[str, str], *, order: int, min_n: int) -> CandidateResult:
    target, predictors = parse_formula(row["formula"])
    rel_path = row["path"]
    path = root / rel_path
    df = pd.read_csv(path)
    frame = candidate_numeric_frame(df, target, predictors)
    if len(frame) < min_n:
        raise ValueError(f"too few usable rows: {len(frame)} < {min_n}")
    driver = choose_driver(frame, target, predictors, order)

    y_raw = frame[target].to_numpy(dtype=float)
    x_raw = frame[driver].to_numpy(dtype=float)
    n_train = int(0.7 * len(frame))
    x_mean = float(np.mean(x_raw[:n_train]))
    x_sd = float(np.std(x_raw[:n_train]))
    y_mean = float(np.mean(y_raw[:n_train]))
    y_sd = float(np.std(y_raw[:n_train]))
    x = standardized(x_raw, x_mean, x_sd)
    y = standardized(y_raw, y_mean, y_sd)

    x_train = x[:n_train]
    y_train = y[:n_train]
    x_test = x[n_train:]
    y_test = y[n_train:]

    vwk_fit = fit_vwk_volterra(x_train, y_train, order=order, moments=None)
    wiener_fit = fit_wiener_baseline(x_train, y_train, order=order, variance=float(np.var(x_train)))
    mono_coef = np.linalg.lstsq(monomial_design(x_train, order), y_train, rcond=None)[0]
    huber_coef = fit_huber_monomial(x_train, y_train, order=order)

    vwk_pred = vwk_fit.predict(x_test)
    wiener_pred = wiener_fit.predict(x_test)
    mono_pred = monomial_design(x_test, order) @ mono_coef
    huber_pred = monomial_design(x_test, order) @ huber_coef

    vwk_mse = float(np.mean((vwk_pred - y_test) ** 2))
    wiener_mse = float(np.mean((wiener_pred - y_test) ** 2))
    mono_mse = float(np.mean((mono_pred - y_test) ** 2))
    huber_mse = float(np.mean((huber_pred - y_test) ** 2))
    train_resid = y_train - monomial_design(x_train, order) @ mono_coef
    skew, kurt = residual_moments(train_resid)

    ratio = wiener_mse / vwk_mse if vwk_mse > 0 else float("inf")
    vwk_mono_ratio = vwk_mse / mono_mse if mono_mse > 0 else float("inf")
    if ratio >= 1.2 and vwk_mse <= 1.25 * min(mono_mse, huber_mse):
        decision = "candidate_illustration"
        note = "VWK beats misspecified Wiener and remains close to direct span-fitting guardrails."
    elif ratio >= 1.2:
        decision = "wiener_gap_only"
        note = "VWK beats misspecified Wiener, but direct span-fitting guardrails are materially stronger."
    else:
        decision = "no_flagship"
        note = "No robust VWK-vs-Wiener real-world illustration under this scalar screen."

    return CandidateResult(
        dataset_id=row["id"],
        label=row["label"],
        path=rel_path,
        target=target,
        driver=driver,
        n_total=int(len(df)),
        n_used=int(len(frame)),
        train_n=int(n_train),
        test_n=int(len(frame) - n_train),
        residual_skew=skew,
        residual_excess_kurtosis=kurt,
        vwk_test_mse=vwk_mse,
        wiener_test_mse=wiener_mse,
        monomial_ls_test_mse=mono_mse,
        huber_test_mse=huber_mse,
        wiener_vwk_ratio=ratio,
        vwk_monomial_ratio=vwk_mono_ratio,
        decision=decision,
        note=note,
    )


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[CandidateResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dictionaries = [asdict(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dictionaries[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(dictionaries)


def write_report(path: Path, rows: list[CandidateResult], failures: list[str], *, order: int, min_n: int) -> None:
    best = max(rows, key=lambda item: item.wiener_vwk_ratio)
    robust = [row for row in rows if row.decision == "candidate_illustration"]
    gap_only = [row for row in rows if row.decision == "wiener_gap_only"]
    if robust:
        decision = "candidate illustration available, but still not a theorem-supporting real-world claim"
    elif gap_only:
        decision = "Wiener gaps exist, but guardrails are stronger; use as diagnostic only"
    else:
        decision = "no real-world flagship claim from local scalar screen"
    lines = [
        "# Real-World VWK Screen",
        "",
        f"*Дата: {REPORT_DATE} | order={order} | min_n={min_n}*",
        "",
        "## Decision",
        "",
        decision,
        "",
        "This screen is intentionally conservative. It tests scalar real-world illustrations from already-prepared local CSVs; it does not establish unbiased Volterra-kernel recovery because these datasets do not provide known Volterra kernels.",
        "",
        "## Summary",
        "",
        f"- Screened candidates: `{len(rows)}`.",
        f"- Best W/V ratio: `{best.wiener_vwk_ratio:.3f}` for `{best.dataset_id}`.",
        f"- Robust candidate illustrations: `{len(robust)}`.",
        f"- Wiener-gap-only diagnostics: `{len(gap_only)}`.",
        "",
        "## Results",
        "",
        "| Dataset | target | driver | n | skew | kurtosis | VWK MSE | Wiener MSE | LS MSE | Huber MSE | W/V | decision |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.dataset_id} | {row.target} | {row.driver} | {row.n_used} | "
            f"{row.residual_skew:.3f} | {row.residual_excess_kurtosis:.3f} | "
            f"{row.vwk_test_mse:.6g} | {row.wiener_test_mse:.6g} | "
            f"{row.monomial_ls_test_mse:.6g} | {row.huber_test_mse:.6g} | "
            f"{row.wiener_vwk_ratio:.3f} | {row.decision} |"
        )
    if failures:
        lines.extend(["", "## Skipped", ""])
        lines.extend(f"- {failure}" for failure in failures)
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "The current 3a manuscript should not claim real-world unbiased identification from this screen. The synthetic finite-memory experiments remain the evidence base for H4; these real-world rows can be used only as diagnostic illustrations of the VWK-vs-misspecified-Wiener gap, not as theorem evidence.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(rows: list[CandidateResult]) -> bool:
    print("Real-world VWK screen: local scalar candidates")
    print()
    print("dataset                         target        driver        W/V    decision")
    print("-" * 88)
    for row in rows:
        print(f"{row.dataset_id:<31} {row.target:<13} {row.driver:<13} {row.wiener_vwk_ratio:6.2f} {row.decision}")
    robust = [row for row in rows if row.decision == "candidate_illustration"]
    print()
    print(f"Robust candidate illustrations: {len(robust)}")
    print("RESULT:", "PASS_NO_CLAIM" if not robust else "PASS_CANDIDATE")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="shared/datasets/external/external_candidates.csv")
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--min-n", type=int, default=200)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--out-dir", default="paper-3-volterra-wiener-kunchenko/code/experiments/results")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    manifest_path = root / args.manifest
    rows: list[CandidateResult] = []
    failures: list[str] = []
    for item in read_manifest(manifest_path):
        try:
            rows.append(evaluate_candidate(root, item, order=args.order, min_n=args.min_n))
        except Exception as exc:  # noqa: BLE001 - report all screening failures.
            failures.append(f"{item.get('id', '<unknown>')}: {exc}")

    if not rows:
        raise RuntimeError("no real-world candidates could be evaluated")

    print_summary(rows)
    if args.write:
        out_dir = root / args.out_dir
        summary_path = out_dir / f"realworld_screen_summary_{REPORT_DATE}.csv"
        report_path = out_dir / f"realworld_screen_report_{REPORT_DATE}.md"
        write_csv(summary_path, rows)
        write_report(report_path, rows, failures, order=args.order, min_n=args.min_n)
        print()
        print("Wrote artifacts:")
        print(f"  {summary_path.relative_to(root)}")
        print(f"  {report_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
