# Paper 3a Experiment Suite Audit

*Date: 2026-06-08 | scope: moment-based finite-memory 3a | Lean included: True*

## Verdict

PASS

This audit covers the current required 3a experiment and verification gates. CF/3b, full formal VWK verification, and real-world unbiased kernel recovery are explicitly outside this suite.

## Command Gates

| Gate | Status | Seconds | Command |
|---|---|---:|---|
| G2 | PASS | 3.7 | `(cd . && python -m pytest code/tests -q)` |
| G3 | PASS | 0.3 | `(cd . && python code/experiments/run_sanity.py)` |
| G4 | PASS | 3.0 | `(cd . && python code/experiments/run_synthetic_h4.py --write)` |
| G7 | PASS | 7.4 | `(cd . && python code/experiments/run_finite_memory_empirical.py --write)` |
| G8 | PASS | 14.7 | `(cd . && python code/experiments/run_sample_efficiency.py --write)` |
| G9 | PASS | 1.7 | `(cd . && python code/experiments/run_realworld_screen.py --write)` |
| A1 | PASS | 0.2 | `(cd . && python verification/verify_askey_discrete.py)` |
| A2 | PASS | 0.2 | `(cd . && python verification/verify_askey_continuous.py)` |
| G6 | PASS | 21.0 | `(cd vwk-lean && lake build VWK)` |

## Artifact Gates

| Gate | Status | Artifact | Description |
|---|---|---|---|
| G1 | PASS | `theorem-proofs.md` | paper-level finite-memory proof layer |
| G5 | PASS | `manuscript-draft-uk.md` | Ukrainian 3a manuscript draft |
| G4 | PASS | `code/experiments/results/synthetic_h4_production_summary_2026-06-08.csv` | synthetic H4 production summary |
| G4 | PASS | `code/experiments/results/synthetic_h4_production_replicates_2026-06-08.csv` | synthetic H4 production replicate table |
| G4 | PASS | `code/experiments/results/synthetic_h4_production_report_2026-06-08.md` | synthetic H4 production report |
| G7 | PASS | `code/experiments/results/finite_memory_empirical_production_summary_2026-06-08.csv` | finite-memory empirical production summary |
| G7 | PASS | `code/experiments/results/finite_memory_empirical_production_replicates_2026-06-08.csv` | finite-memory empirical production replicate table |
| G7 | PASS | `code/experiments/results/finite_memory_empirical_production_report_2026-06-08.md` | finite-memory empirical production report |
| G8 | PASS | `code/experiments/results/sample_efficiency_production_summary_2026-06-08.csv` | sample-efficiency production summary |
| G8 | PASS | `code/experiments/results/sample_efficiency_production_report_2026-06-08.md` | sample-efficiency production report |
| G9 | PASS | `code/experiments/results/realworld_screen_summary_2026-06-08.csv` | real-world screen summary |
| G9 | PASS | `code/experiments/results/realworld_screen_report_2026-06-08.md` | real-world screen report |
| A1 | PASS | `verification/REPORT.txt` | stored discrete Askey report |
| A2 | PASS | `verification/REPORT_continuous.txt` | stored continuous Askey report |
| G6 | PASS | `vwk-lean/.lake/packages` | local Lean/Mathlib cache link |

## Claim Boundary

- The suite supports the finite-memory 3a claim: matched VWK projection versus misspecified Gaussian/Wiener projection under finite moments.
- The synthetic evidence remains scoped to asymmetric finite-moment inputs; symmetric controls do not justify a symmetric non-Gaussian advantage claim.
- The real-world screen is diagnostic only and does not establish unbiased Volterra-kernel recovery.
- CF/moment-free 3b experiments are not necessary for the current 3a workstream and were not run.

## Output Tails

### G2 - package tests for basis and finite-memory estimator

```text
......                                                                   [100%]
6 passed in 2.24s
```

### G3 - Level-1 sanity checks

```text
normal_gram_err: 3.553e-15
centered_exp_gram_err: 2.842e-14
normal_psi0_constant: 0.000e+00
normal_psi1_x: 0.000e+00
RESULT: PASS
```

### G4 - one-lag synthetic H4 production grid

```text
  Gaussian control W/V near 1     : PASS (1.000)
  Any non-Gaussian W/V > 1.2      : PASS
  Huber baseline reported only as guardrail; it is not the H4 pass criterion.
RESULT: PASS
Wrote artifacts:
  code/experiments/results/synthetic_h4_production_summary_2026-06-08.csv
  code/experiments/results/synthetic_h4_production_replicates_2026-06-08.csv
  code/experiments/results/synthetic_h4_production_report_2026-06-08.md
```

### G7 - second-order memory=3 empirical-moment production grid

```text
  Skew W/oracle and W/empirical > 1.2  : PASS
  Empirical moments stable, E/O <= 1.5 : PASS
  Monomial LS and Huber are guardrails, not H4 pass criteria.
RESULT: PASS
Wrote artifacts:
  code/experiments/results/finite_memory_empirical_production_summary_2026-06-08.csv
  code/experiments/results/finite_memory_empirical_production_replicates_2026-06-08.csv
  code/experiments/results/finite_memory_empirical_production_report_2026-06-08.md
```

### G8 - sample-efficiency production curve

```text
Checks:
  Gaussian control W/oracle near 1 across n : PASS
  Skew W/oracle and W/empirical > 1.2      : PASS
  Empirical moments stable, E/O <= 1.5     : PASS
RESULT: PASS
Wrote artifacts:
  code/experiments/results/sample_efficiency_production_summary_2026-06-08.csv
  code/experiments/results/sample_efficiency_production_report_2026-06-08.md
```

### G9 - real-world scalar diagnostic screen

```text
sru_y1_dynamic                  y1            y1_lag1         4.09 candidate_illustration
sru_y2_dynamic                  y2            y2_lag1         1.45 candidate_illustration
concrete                        strength      age             1.04 no_flagship
Robust candidate illustrations: 4
RESULT: PASS_CANDIDATE
Wrote artifacts:
  paper-3-volterra-wiener-kunchenko/code/experiments/results/realworld_screen_summary_2026-06-08.csv
  paper-3-volterra-wiener-kunchenko/code/experiments/results/realworld_screen_report_2026-06-08.md
```

### A1 - discrete Askey verification

```text
    Meixner    γ1 vs β→∞ (c=0.40)    2:1.5652  8:0.7826  32:0.3913  128:0.1957  512:0.0978   (emp==analytic ✓)
    [PASS] C5  all three discrete families → Hermite as index → ∞
==========================================================================
  RESULT: ALL CHECKS PASSED
    worst C1 (VWK==classical) = 3.65e-12   worst C2 = 2.22e-16   worst C4 = 5.58e-14
    worst X  (numpy QR cross) = 3.65e-12
    python 3.12.10    families: Krawtchouk, Charlier, Meixner   degrees 0..5
==========================================================================
```

### A2 - continuous Askey verification

```text
--------------------------------------------------------------------------
  Limit relations (Askey scheme top): Laguerre L^{(α)} → Hermite as α→∞;
  Jacobi → Hermite as α,β→∞ — continuous analogue of the discrete C5 (skew→0 ⇒ g→1).
==========================================================================
  RESULT: ALL CHECKS PASSED
    worst C1 = 3.93e-14   worst C2 = 4.44e-16   worst C4 = 3.44e-16
    numpy 2.4.6   families: Hermite, Laguerre, Legendre   degrees 0..5
==========================================================================
```

### G6 - Lean sorry-free fragment build

```text
ℹ [7746/7748] Replayed VWK.Krawtchouk
info: VWK/Krawtchouk.lean:83:0: 'VWK.Binom3.mass' depends on axioms: [propext, Classical.choice, Quot.sound]
info: VWK/Krawtchouk.lean:84:0: 'VWK.Binom3.orth_01' depends on axioms: [propext, Classical.choice, Quot.sound]
info: VWK/Krawtchouk.lean:85:0: 'VWK.Binom3.orth_02' depends on axioms: [propext, Classical.choice, Quot.sound]
info: VWK/Krawtchouk.lean:86:0: 'VWK.Binom3.orth_12' depends on axioms: [propext, Classical.choice, Quot.sound]
info: VWK/Krawtchouk.lean:87:0: 'VWK.Binom3.variance' depends on axioms: [propext, Classical.choice, Quot.sound]
info: VWK/Krawtchouk.lean:88:0: 'VWK.Binom3.central_three' depends on axioms: [propext, Classical.choice, Quot.sound]
Build completed successfully (7748 jobs).
```
