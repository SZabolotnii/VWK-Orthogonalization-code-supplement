#!/usr/bin/env python3
"""Render the finite-memory sample-efficiency figure with bootstrap CI bands.

The committed figures (``paper-3a/figures/sample_efficiency_vwk{,_uk}.pdf``)
plotted point estimates only.  This script reproduces the exact production
point estimates (same seed scheme as ``run_sample_efficiency.py``: base seed
20260608, ``+10000`` per sample size, ``+1000`` per regime inside ``run_grid``)
and adds a 95% nonparametric bootstrap confidence band around each plotted
ratio.

Plotted quantity (unchanged from the committed figure and the report tables):
the *ratio of means* ``mean(Wiener MSE) / mean(VWK MSE)`` over the replications.
The band is a percentile bootstrap CI for that same ratio-of-means statistic,
resampling replication indices jointly so the paired (Wiener, VWK) draws stay
together.  For the Gaussian control the Wiener basis coincides with the VWK
oracle basis, so the per-replicate ratio is identically 1 and the band collapses
to the line (a correct, by-construction statement, not a noisy estimate).

Run from anywhere; paths are resolved relative to the repository root.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for sub in ("code", "code/experiments"):
    candidate = str(REPO_ROOT / sub)
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from run_finite_memory_empirical import run_grid  # noqa: E402

FIGURE_DATE = "2026-06-09"
BASE_SEED = 20260608
N_GRID = [600, 1200, 2500, 5000]
REPS = 80
MEMORY = 3
ORDER = 2
NOISE_SD = 0.25

# Unified, colour-blind-friendly palette (applied to BOTH language variants so
# the English and Ukrainian figures finally match).
TEAL = "#1b7f8c"   # skew, oracle moments
ORANGE = "#d8631a"  # skew, empirical moments
GRAY = "#6e6e6e"   # Gaussian control, oracle moments

LABELS = {
    "en": {
        "title": "Finite-memory sample-efficiency gate",
        "xlabel": "Sample size $n$",
        "ylabel": "Misspecified Wiener / VWK MSE ratio",
        "skew_oracle": "Skew, oracle moments",
        "skew_empirical": "Skew, empirical moments",
        "gauss": "Gaussian control, oracle",
        "ci": "95% bootstrap CI",
    },
    "uk": {
        "title": "Хибно заданий Вінер / узгоджений VWK",
        "xlabel": "Розмір вибірки $n$",
        "ylabel": "Відношення СКП",
        "skew_oracle": "асиметрія, точні моменти",
        "skew_empirical": "асиметрія, емпіричні моменти",
        "gauss": "гауссівський контроль, точні моменти",
        "ci": "95% бутстреп-ДІ",
    },
}


def bootstrap_ratio_ci(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    n_boot: int,
    rng: np.random.Generator,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the ratio-of-means statistic.

    ``numerator`` and ``denominator`` are paired per-replication arrays; the
    same resampled indices are applied to both so the pairing is preserved.
    Returns ``(point, lo, hi)`` where ``point`` is the ratio of the full-sample
    means and ``(lo, hi)`` are the percentile band endpoints.
    """
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    reps = numerator.size
    point = float(numerator.mean() / denominator.mean())
    idx = rng.integers(0, reps, size=(n_boot, reps))
    boot = numerator[idx].mean(axis=1) / denominator[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.percentile(boot, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return point, float(lo), float(hi)


def collect_series(n_boot: int, boot_seed: int) -> dict[str, dict[str, np.ndarray]]:
    """Run the production grid across the sample-size grid and assemble curves.

    Returns a dict keyed by series name; each value holds the x grid plus point
    estimate and CI endpoints for every sample size.
    """
    boot_rng = np.random.default_rng(boot_seed)
    series = {
        "skew_oracle": {"point": [], "lo": [], "hi": []},
        "skew_empirical": {"point": [], "lo": [], "hi": []},
        "gauss": {"point": [], "lo": [], "hi": []},
    }
    for offset, n in enumerate(N_GRID):
        _, replicate_rows = run_grid(
            reps=REPS,
            n=n,
            memory=MEMORY,
            order=ORDER,
            noise_sd=NOISE_SD,
            seed=BASE_SEED + 10000 * offset,
        )
        skew = [r for r in replicate_rows if r.regime == "centered_exponential_skew"]
        gauss = [r for r in replicate_rows if r.regime == "gaussian_control"]

        skew_w = np.array([r.wiener_mse for r in skew])
        skew_oracle = np.array([r.oracle_vwk_mse for r in skew])
        skew_emp = np.array([r.empirical_vwk_mse for r in skew])
        gauss_w = np.array([r.wiener_mse for r in gauss])
        gauss_oracle = np.array([r.oracle_vwk_mse for r in gauss])

        for key, num, den in (
            ("skew_oracle", skew_w, skew_oracle),
            ("skew_empirical", skew_w, skew_emp),
            ("gauss", gauss_w, gauss_oracle),
        ):
            point, lo, hi = bootstrap_ratio_ci(num, den, n_boot=n_boot, rng=boot_rng)
            series[key]["point"].append(point)
            series[key]["lo"].append(lo)
            series[key]["hi"].append(hi)

    for key in series:
        for stat in ("point", "lo", "hi"):
            series[key][stat] = np.array(series[key][stat])
    return series


def render(series: dict, lang: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    txt = LABELS[lang]
    x = np.array(N_GRID, dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    spec = [
        ("skew_empirical", ORANGE, "s", txt["skew_empirical"]),
        ("skew_oracle", TEAL, "o", txt["skew_oracle"]),
        ("gauss", GRAY, "^", txt["gauss"]),
    ]
    for key, color, marker, label in spec:
        s = series[key]
        ax.fill_between(x, s["lo"], s["hi"], color=color, alpha=0.18, linewidth=0)
        ax.plot(
            x,
            s["point"],
            marker=marker,
            color=color,
            lw=2.0,
            ms=7,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=label,
        )
    ax.axhline(1.0, ls="--", color="#9a9a9a", lw=1.0, zorder=1)

    ax.set_title(txt["title"], fontsize=13)
    ax.set_xlabel(txt["xlabel"], fontsize=12)
    ax.set_ylabel(txt["ylabel"], fontsize=12)
    ax.set_xticks(N_GRID)
    ax.set_xticklabels([str(n) for n in N_GRID])
    ax.set_xlim(N_GRID[0] - 150, N_GRID[-1] + 200)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, color="#dddddd", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles, plot_labels = ax.get_legend_handles_labels()
    band_proxy = plt.Rectangle((0, 0), 1, 1, fc="#888888", alpha=0.18, ec="none")
    ax.legend(
        handles + [band_proxy],
        plot_labels + [txt["ci"]],
        fontsize=10,
        loc="upper left",
        frameon=False,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def write_ci_csv(series: dict, path: Path, *, n_boot: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    series_label = {
        "skew_oracle": "centered_exponential_skew / W over oracle",
        "skew_empirical": "centered_exponential_skew / W over empirical",
        "gauss": "gaussian_control / W over oracle",
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["series", "n", "ratio_point", "ci_lo", "ci_hi", "reps", "n_boot", "ci_level"]
        )
        for key, label in series_label.items():
            s = series[key]
            for i, n in enumerate(N_GRID):
                writer.writerow(
                    [
                        label,
                        n,
                        f"{s['point'][i]:.6f}",
                        f"{s['lo'][i]:.6f}",
                        f"{s['hi'][i]:.6f}",
                        REPS,
                        n_boot,
                        0.95,
                    ]
                )
    print(f"  wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=4000, help="Bootstrap resamples.")
    parser.add_argument("--boot-seed", type=int, default=20260609)
    parser.add_argument(
        "--fig-dir",
        default=str(REPO_ROOT / "paper-3a" / "figures"),
        help="Output directory for the PDF figures.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(REPO_ROOT / "code" / "experiments" / "results"),
        help="Output directory for the CI provenance CSV.",
    )
    args = parser.parse_args()

    print("Running production grid (reps=80, n in {600,1200,2500,5000}) ...")
    series = collect_series(n_boot=args.n_boot, boot_seed=args.boot_seed)

    print("Series (ratio point [95% CI]):")
    for key in ("skew_oracle", "skew_empirical", "gauss"):
        s = series[key]
        cells = " ".join(
            f"n={n}:{s['point'][i]:.2f}[{s['lo'][i]:.2f},{s['hi'][i]:.2f}]"
            for i, n in enumerate(N_GRID)
        )
        print(f"  {key:15s} {cells}")

    fig_dir = Path(args.fig_dir)
    render(series, "en", fig_dir / "sample_efficiency_vwk.pdf")
    render(series, "uk", fig_dir / "sample_efficiency_vwk_uk.pdf")
    write_ci_csv(
        series,
        Path(args.results_dir) / f"sample_efficiency_ci_{FIGURE_DATE}.csv",
        n_boot=args.n_boot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
