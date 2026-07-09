# VWK Orthogonalization — Code & Lean Supplement

Reproducibility supplement for the paper **"Distribution-Matched Volterra
Identification under Non-Gaussian Input: A Closed-Form Skew Penalty for the
Wiener Cross-Correlation Estimator"** (Serhii V. Zabolotnii). It reproduces every
table, figure, and formal-proof claim in the paper.

## Contents
- `code/vwk/` — VWK basis construction (oriented Gram–Schmidt in L²(P)) + finite-memory
  Volterra projection (Python).
- `code/experiments/` — experiment & verification scripts (synthetic grids, conditioning
  and ridge diagnostics, penalty verification, sample-efficiency figure, real-world screen).
- `code/experiments/results/` — frozen CSV/Markdown outputs backing every reported table
  and figure (fixed seed `20260608`; reruns reproduce bit-for-bit).
- `vwk-lean/VWK/` — machine-checked **Lean 4** proofs (Mathlib v4.26.0) of the
  orthogonality/moment identities, including the arbitrary-`N` Binomial→Krawtchouk row.

## Reproduce (Python — requires Python 3 + NumPy)
```sh
PYTHONPATH=code python code/experiments/run_conditioning.py --write --empirical-gram --ridge-cv
PYTHONPATH=code python code/experiments/run_synthetic_h4.py
PYTHONPATH=code python code/experiments/run_finite_memory_empirical.py
PYTHONPATH=code python code/experiments/run_realworld_screen.py
python code/experiments/make_sample_efficiency_figure.py
PYTHONPATH=code python code/experiments/run_sanity.py
```

## Verify (Lean 4)
```sh
cd vwk-lean && lake build VWK
```
The proofs are **`sorry`-free**: `#print axioms` reports only
`[propext, Classical.choice, Quot.sound]` (no `sorryAx`).

## Reproduction map
| Manuscript artifact | Script |
|---|---|
| `tab:dgp` (data-generating spec) | regimes in `run_synthetic_h4.py` |
| `tab:conditioning` (incl. empirical VWK Gram), `tab:ridge` (fixed-η + CV-tuned) | `run_conditioning.py` |
| `tab:h4` | `run_synthetic_h4.py` |
| `tab:deconfound` | `verify_deconfound.py` |
| penalty proposition (W/V ≈ 32.36) | `verify_penalty_proposition.py` |
| `tab:finite-memory` | `run_finite_memory_empirical.py` |
| `fig:sample-efficiency` | `run_sample_efficiency.py` + `make_sample_efficiency_figure.py` |
| `tab:realworld` | `run_realworld_screen.py` |
| Lean orthogonality / Krawtchouk | `vwk-lean/VWK/*.lean` |

## License
MIT — see [`LICENSE`](LICENSE).

## Citation
If you use this code, please cite the paper: S. V. Zabolotnii, *Distribution-Matched
Volterra Identification under Non-Gaussian Input: A Closed-Form Skew Penalty for the
Wiener Cross-Correlation Estimator* (2026).
