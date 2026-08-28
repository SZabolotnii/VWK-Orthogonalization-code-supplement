#!/usr/bin/env python3
"""E4 -- non-Gaussian additive output noise at matched variance (Rv1-C7).

Repeats the finite-memory production grid (d=3, s=2, n=2500) with output noise
drawn from four laws scaled to sd 0.25. Reports MSEs, W/O, and -- the point of
the run -- the coefficient BIAS (mean over reps of a_hat - beta) versus the
coefficient SD, to show the noise law moves the variance, not the bias.
"""

from __future__ import annotations

import numpy as np

from common import (
    OUT, REGIMES, align, build_vwk_basis, canonical_beta, fit_vwk_volterra,
    fit_wiener_baseline, sample_iid, tensor_features, write_csv,
)

N, D, S, SD, REPS, SEED = 2500, 3, 2, 0.25, 150, 20260827


def noise(rng, law, size):
    if law == "gaussian":
        return rng.normal(scale=SD, size=size)
    if law == "centered_exponential":
        return SD * (rng.exponential(size=size) - 1.0)
    if law == "student_t3":
        return SD / np.sqrt(3.0) * rng.standard_t(3, size=size)
    if law == "contaminated_10pct":
        # 0.9 N(0,a) + 0.1 N(0,9a), variance = SD^2  ->  a = SD^2 / 1.8
        a = SD**2 / 1.8
        mask = rng.random(size) < 0.9
        e = np.empty(size)
        e[mask] = rng.normal(scale=np.sqrt(a), size=int(mask.sum()))
        e[~mask] = rng.normal(scale=np.sqrt(9 * a), size=int((~mask).sum()))
        return e
    raise ValueError(law)


LAWS = ["gaussian", "centered_exponential", "student_t3", "contaminated_10pct"]


def main() -> None:
    rows = []
    for r_off, regime in enumerate(REGIMES):
        mom = regime.moments(S)
        tb = build_vwk_basis(mom, S)
        for k, law in enumerate(LAWS):
            rng = np.random.default_rng(SEED + 1000 * r_off + k)
            o_mse, e_mse, w_mse, coefs, nsd, noise_proj = [], [], [], [], [], []
            for _ in range(REPS):
                x = sample_iid(rng, regime, N)
                feats, idx = tensor_features(x, tb, S, D)
                beta = canonical_beta(idx)
                sig = feats @ beta
                eps = noise(rng, law, sig.size)
                nsd.append(eps.std())
                y = align(x, sig + eps, D)
                o = fit_vwk_volterra(x, y, order=S, memory=D, moments=mom)
                e = fit_vwk_volterra(x, y, order=S, memory=D, moments=None)
                w = fit_wiener_baseline(x, y, order=S, memory=D, variance=regime.variance)
                o_mse.append(np.mean((o.predict(x) - sig) ** 2))
                e_mse.append(np.mean((e.predict(x) - sig) ** 2))
                w_mse.append(np.mean((w.predict(x) - sig) ** 2))
                coefs.append(o.coefficients - beta)
                noise_proj.append(feats.T @ eps / eps.size)  # E_n[eps Psi_alpha]: the noise's own contribution
            NP = np.asarray(noise_proj)
            np_bias = np.linalg.norm(NP.mean(0)); np_sd = np.sqrt(np.mean(NP.var(0)))
            C = np.asarray(coefs)
            bias = np.linalg.norm(C.mean(0))
            sd = np.sqrt(np.mean(C.var(0)))
            row = {
                "regime": regime.name, "noise_law": law, "noise_sd_measured": float(np.mean(nsd)),
                "oracle_vwk_mse": float(np.mean(o_mse)), "empirical_vwk_mse": float(np.mean(e_mse)),
                "wiener_mse": float(np.mean(w_mse)),
                "W_over_oracle": float(np.mean(w_mse) / np.mean(o_mse)),
                "coef_bias_norm": float(bias), "coef_sd_rms": float(sd),
                "bias_se_ratio": float(bias / (sd / np.sqrt(REPS))),
                "noiseproj_bias_norm": float(np_bias), "noiseproj_sd_rms": float(np_sd),
                "noiseproj_bias_se_ratio": float(np_bias / (np_sd / np.sqrt(REPS))),
                "noiseproj_sd_theory_iid": float(SD / np.sqrt(N - D + 1)),
            }
            rows.append(row)
            print(f"{regime.name:28s} {law:22s} O={row['oracle_vwk_mse']:.5f} W/O={row['W_over_oracle']:.3f} "
                  f"bias={bias:.4f} sd={sd:.4f} bias/se={row['bias_se_ratio']:.2f} | noise-proj bias={np_bias:.5f} sd={np_sd:.5f} "
                  f"bias/se={row['noiseproj_bias_se_ratio']:.2f} sd_theory={row['noiseproj_sd_theory_iid']:.5f}")
    write_csv(OUT / "e4_output_noise.csv", rows)


if __name__ == "__main__":
    main()
