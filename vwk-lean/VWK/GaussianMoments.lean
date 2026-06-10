import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Probability.Moments.Variance
import Mathlib.Probability.Moments.MGFAnalytic
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.IteratedDeriv.Defs
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Tactic

/-!
# VWK.GaussianMoments — explicit moments of the centered Gaussian

Closes the orthonormality blocker for `VWK.Hermite` (Theorem 2, L1-full): the inner products
`⟪ψ_i, ψ_j⟫` over `L²(N(0, σ²))` reduce to Gaussian moments, which Mathlib v4.26 does **not**
provide as explicit values (only mean + variance). See `MATHLIB_AUDIT.md` §2.3.

This module supplies them, moments-only (frozen decision §8.1 — no cumulants), at upstream-PR
quality (§8.3 — explicit statements):

* `integral_sq_gaussianReal       : E[X²] = v`   (`variance_of_integral_eq_zero`).
* `integral_cube_gaussianReal     : E[X³] = 0`   (symmetry `gaussianReal_map_neg`).
* `integral_pow_four_gaussianReal : E[X⁴] = 3v²` (MGF: `E[Xⁿ] = (dⁿ/dtⁿ mgf)(0)`).

Together these cover all six cross-terms of the `{ψ₀, ψ₁, ψ₂}` orthonormality (next: assemble
`∫ ψ_i·ψ_j dN = δ_ij`).
-/

open MeasureTheory ProbabilityTheory
open scoped NNReal

namespace VWK

/-- **Second moment of a centered Gaussian:** `∫ x², d N(0, v) = v`.

Since `N(0, v)` has mean `0` (`integral_id_gaussianReal`), the variance equals the second
moment (`variance_of_integral_eq_zero`); the variance is `v` (`variance_fun_id_gaussianReal`). -/
theorem integral_sq_gaussianReal (v : ℝ≥0) :
    ∫ x, x ^ 2 ∂(gaussianReal 0 v) = (v : ℝ) := by
  have hmean : ∫ x, x ∂(gaussianReal 0 v) = 0 := integral_id_gaussianReal
  have hv : variance (fun x => x) (gaussianReal 0 v) = ∫ x, x ^ 2 ∂(gaussianReal 0 v) :=
    variance_of_integral_eq_zero measurable_id.aemeasurable hmean
  rw [variance_fun_id_gaussianReal] at hv
  exact hv.symm

/-- **Third moment of a centered Gaussian vanishes:** `∫ x³, d N(0, v) = 0`.

`N(0, v)` is symmetric: `(N(0,v)).map (· ↦ -·) = N(0,v)` (`gaussianReal_map_neg`, since `-0 = 0`).
Hence `∫ x³ = ∫ (-x)³ = -∫ x³`, so the integral is its own negative and therefore `0`. -/
theorem integral_cube_gaussianReal (v : ℝ≥0) :
    ∫ x, x ^ 3 ∂(gaussianReal 0 v) = 0 := by
  have hsymm : (gaussianReal 0 v).map (fun x => -x) = gaussianReal 0 v := by
    rw [gaussianReal_map_neg, neg_zero]
  have hf : AEMeasurable (fun x : ℝ => -x) (gaussianReal 0 v) := measurable_neg.aemeasurable
  have hg : AEStronglyMeasurable (fun y : ℝ => y ^ 3) ((gaussianReal 0 v).map (fun x => -x)) :=
    (continuous_pow 3).aestronglyMeasurable
  have key : ∫ x, x ^ 3 ∂(gaussianReal 0 v) = -∫ x, x ^ 3 ∂(gaussianReal 0 v) := by
    conv_lhs => rw [← hsymm]
    rw [integral_map hf hg]
    have h3 : ∀ x : ℝ, (-x) ^ 3 = -(x ^ 3) := fun x => by ring
    simp_rw [h3]
    rw [integral_neg]
  linarith

/-- **Fourth moment of a centered Gaussian:** `∫ x⁴, d N(0, v) = 3v²`.

Via the moment-generating function: `mgf id N(0,v) t = exp(v t²/2)` (`mgf_id_gaussianReal`) and
`E[X⁴] = (d⁴/dt⁴ mgf)(0)` (`iteratedDeriv_mgf_zero`). The four derivatives of `t ↦ exp(c t²/2)`
(with `c = v`) are computed explicitly; evaluated at `t = 0` they give `3c² = 3v²`. -/
theorem integral_pow_four_gaussianReal (v : ℝ≥0) :
    ∫ x, x ^ 4 ∂(gaussianReal 0 v) = 3 * (v : ℝ) ^ 2 := by
  -- Step 1: the 4th moment is the 4th derivative of the mgf at 0.
  have h0 : (0 : ℝ) ∈ interior (integrableExpSet id (gaussianReal 0 v)) := by simp
  have hmom := iteratedDeriv_mgf_zero h0 4
  rw [mgf_id_gaussianReal] at hmom
  simp only [zero_mul, zero_add] at hmom
  set c : ℝ := (v : ℝ) with hc
  -- Step 2: compute iteratedDeriv 4 (fun t => exp (c t²/2)) 0 = 3 c².
  -- Base derivative: (exp (c t²/2))' = c t · exp (c t²/2).
  have hg : ∀ t : ℝ, HasDerivAt (fun s : ℝ => Real.exp (c * s ^ 2 / 2))
      (c * t * Real.exp (c * t ^ 2 / 2)) t := by
    intro t
    have hquad : HasDerivAt (fun s : ℝ => c * s ^ 2 / 2) (c * t) t := by
      have h := ((hasDerivAt_pow 2 t).const_mul c).div_const 2
      convert h using 1; push_cast; ring
    have h := hquad.exp
    convert h using 1; ring
  have hd1 : deriv (fun t : ℝ => Real.exp (c * t ^ 2 / 2))
      = fun t => c * t * Real.exp (c * t ^ 2 / 2) := funext fun t => (hg t).deriv
  -- First derivative function f₁ = c t · E; its derivative is (c + c² t²) E.
  have hf1 : ∀ t : ℝ, HasDerivAt (fun s : ℝ => c * s * Real.exp (c * s ^ 2 / 2))
      ((c + c ^ 2 * t ^ 2) * Real.exp (c * t ^ 2 / 2)) t := by
    intro t
    have hp : HasDerivAt (fun s : ℝ => c * s) c t := by simpa using (hasDerivAt_id t).const_mul c
    have h := hp.mul (hg t)
    convert h using 1; ring
  have hd2 : deriv (fun t : ℝ => c * t * Real.exp (c * t ^ 2 / 2))
      = fun t => (c + c ^ 2 * t ^ 2) * Real.exp (c * t ^ 2 / 2) := funext fun t => (hf1 t).deriv
  -- Second derivative function f₂ = (c + c² t²) E; its derivative is (3c² t + c³ t³) E.
  have hf2 : ∀ t : ℝ, HasDerivAt (fun s : ℝ => (c + c ^ 2 * s ^ 2) * Real.exp (c * s ^ 2 / 2))
      ((3 * c ^ 2 * t + c ^ 3 * t ^ 3) * Real.exp (c * t ^ 2 / 2)) t := by
    intro t
    have hp : HasDerivAt (fun s : ℝ => c + c ^ 2 * s ^ 2) (2 * c ^ 2 * t) t := by
      have h := ((hasDerivAt_pow 2 t).const_mul (c ^ 2)).const_add c
      convert h using 1; push_cast; ring
    have h := hp.mul (hg t)
    convert h using 1; ring
  have hd3 : deriv (fun t : ℝ => (c + c ^ 2 * t ^ 2) * Real.exp (c * t ^ 2 / 2))
      = fun t => (3 * c ^ 2 * t + c ^ 3 * t ^ 3) * Real.exp (c * t ^ 2 / 2) :=
    funext fun t => (hf2 t).deriv
  -- Third derivative function f₃ = (3c² t + c³ t³) E; its derivative is (3c² + 6c³ t² + c⁴ t⁴) E.
  have hf3 : ∀ t : ℝ, HasDerivAt
      (fun s : ℝ => (3 * c ^ 2 * s + c ^ 3 * s ^ 3) * Real.exp (c * s ^ 2 / 2))
      ((3 * c ^ 2 + 6 * c ^ 3 * t ^ 2 + c ^ 4 * t ^ 4) * Real.exp (c * t ^ 2 / 2)) t := by
    intro t
    have hp : HasDerivAt (fun s : ℝ => 3 * c ^ 2 * s + c ^ 3 * s ^ 3)
        (3 * c ^ 2 + 3 * c ^ 3 * t ^ 2) t := by
      have h := ((hasDerivAt_id t).const_mul (3 * c ^ 2)).add
        ((hasDerivAt_pow 3 t).const_mul (c ^ 3))
      convert h using 1; push_cast; ring
    have h := hp.mul (hg t)
    convert h using 1; ring
  have hd4 : deriv (fun t : ℝ => (3 * c ^ 2 * t + c ^ 3 * t ^ 3) * Real.exp (c * t ^ 2 / 2))
      = fun t => (3 * c ^ 2 + 6 * c ^ 3 * t ^ 2 + c ^ 4 * t ^ 4) * Real.exp (c * t ^ 2 / 2) :=
    funext fun t => (hf3 t).deriv
  -- Assemble the iterated derivative and evaluate at 0.
  have hcalc : iteratedDeriv 4 (fun t : ℝ => Real.exp (c * t ^ 2 / 2)) 0 = 3 * c ^ 2 := by
    have i1 : iteratedDeriv 1 (fun t : ℝ => Real.exp (c * t ^ 2 / 2))
        = fun t => c * t * Real.exp (c * t ^ 2 / 2) := by rw [iteratedDeriv_one]; exact hd1
    have i2 : iteratedDeriv 2 (fun t : ℝ => Real.exp (c * t ^ 2 / 2))
        = fun t => (c + c ^ 2 * t ^ 2) * Real.exp (c * t ^ 2 / 2) := by
      rw [iteratedDeriv_succ, i1]; exact hd2
    have i3 : iteratedDeriv 3 (fun t : ℝ => Real.exp (c * t ^ 2 / 2))
        = fun t => (3 * c ^ 2 * t + c ^ 3 * t ^ 3) * Real.exp (c * t ^ 2 / 2) := by
      rw [iteratedDeriv_succ, i2]; exact hd3
    have i4 : iteratedDeriv 4 (fun t : ℝ => Real.exp (c * t ^ 2 / 2))
        = fun t => (3 * c ^ 2 + 6 * c ^ 3 * t ^ 2 + c ^ 4 * t ^ 4) * Real.exp (c * t ^ 2 / 2) := by
      rw [iteratedDeriv_succ, i3]; exact hd4
    rw [i4]; simp [Real.exp_zero]
  -- Step 3: combine.
  rw [hcalc] at hmom
  simp only [Pi.pow_apply, id_eq] at hmom
  linarith [hmom]

end VWK
