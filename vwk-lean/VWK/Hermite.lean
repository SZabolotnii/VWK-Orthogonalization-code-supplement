import Mathlib

/-!
# VWK.Hermite — Theorem 2 (reduction to Hermite), warm-up `k ≤ 2`

Closes target **L1-mini** of `../lean-handoff.md` §4.3 (Step A) and formalizes the
closed forms of `theorem-draft.md` §5.

The VWK reduction theorem states that, for `P = N(0, σ²)`, the cumulant-driven
Gram–Schmidt basis function `ψ_k` equals the scaled probabilist's Hermite
`ψ_k(x) = He_k(x/σ) / √(k!)`. Here we establish the **explicit closed forms** for
`k ≤ 2`:

* `ψ₀ = 1`                              (`scaledHermite_zero`)
* `ψ₁(x) = x / σ`                       (`scaledHermite_one`)
* `ψ₂(x) = (x² − σ²) / (σ² √2)`         (`scaledHermite_two`)

and the polynomial identity `He₂ = X² − 1` (`hermite_two`) underlying `ψ₂`.

These are **pure algebraic identities** — no integration. The companion claim that
`{ψ_k}` is *orthonormal* w.r.t. the Gaussian measure (which would identify these
functions with the actual Gram–Schmidt output) requires the Gaussian moments
`E[X³] = 0` and `E[X⁴] = 3σ⁴`. Mathlib v4.26 provides only the mean
(`integral_id_gaussianReal`) and variance (`variance_id_gaussianReal`); the 3rd/4th
moments are **not** available, so orthonormality is deferred — see `MATHLIB_AUDIT.md`
and `STATUS.md`.

`Polynomial.hermite : ℕ → ℤ[X]` is the probabilist's Hermite; it is evaluated at a
real argument through `Polynomial.aeval` (the `ℤ`-algebra map `ℤ[X] → ℝ`).
-/

namespace VWK

open Polynomial

/-- Probabilist's Hermite `He₂ = X² − 1`, obtained from the Mathlib recursion
`hermite_succ : hermite (n+1) = X * hermite n - derivative (hermite n)`. -/
theorem hermite_two : hermite 2 = X ^ 2 - 1 := by
  have h : hermite 2 = X * hermite 1 - derivative (hermite 1) := hermite_succ 1
  rw [h, hermite_one, derivative_X]
  ring

/-- Scaled probabilist Hermite as a real function: `x ↦ He_k(x/σ) / √(k!)`.
This is the candidate `k`-th VWK basis function `ψ_k` on `L²(N(0, σ²))` (Theorem 2). -/
noncomputable def scaledHermite (σ : ℝ) (k : ℕ) (x : ℝ) : ℝ :=
  (aeval (x / σ) (hermite k) : ℝ) / Real.sqrt (Nat.factorial k)

@[simp]
theorem scaledHermite_zero (σ x : ℝ) : scaledHermite σ 0 x = 1 := by
  simp [scaledHermite, hermite_zero]

@[simp]
theorem scaledHermite_one (σ x : ℝ) : scaledHermite σ 1 x = x / σ := by
  simp [scaledHermite]

/-- `ψ₂(x) = (x² − σ²) / (σ² √2)`, equivalently `(1/√2)·He₂(x/σ)`. -/
theorem scaledHermite_two (σ x : ℝ) (hσ : σ ≠ 0) :
    scaledHermite σ 2 x = (x ^ 2 - σ ^ 2) / (σ ^ 2 * Real.sqrt 2) := by
  have hs : Real.sqrt 2 ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr (by norm_num))
  have hfac : (Nat.factorial 2 : ℝ) = 2 := by norm_num [Nat.factorial]
  unfold scaledHermite
  rw [hermite_two, hfac, map_sub, map_pow, aeval_X, map_one]
  field_simp

end VWK
