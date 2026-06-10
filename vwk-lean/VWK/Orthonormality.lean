import VWK.Hermite
import VWK.GaussianMoments

/-!
# VWK.Orthonormality — orthonormality of the warm-up basis (L1-full, in progress)

Two reusable helpers concentrating the integrability + linearity plumbing:

* `integrable_pow_gaussianReal` — `fun x => xⁿ` is integrable w.r.t. `N(0, v)`.
* `integral_poly4_gaussianReal` — expectation of a degree-≤4 polynomial: reduces any such
  integral to the moments `E[X⁴]=3v²`, `E[X³]=0`, `E[X²]=v`, `E[X]=0`, `E[1]=1`.

These make each orthonormality cross-term `∫ ψ_i·ψ_j dN = δ_ij` a pointwise polynomial rewrite
followed by `integral_poly4_gaussianReal` + arithmetic (`σ² = v`).
-/

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace VWK

/-- Every monomial `xⁿ` is integrable w.r.t. a real Gaussian (all moments finite). -/
theorem integrable_pow_gaussianReal (v : ℝ≥0) (n : ℕ) :
    Integrable (fun x : ℝ => x ^ n) (gaussianReal 0 v) := by
  rcases Nat.eq_zero_or_pos n with hn | hn
  · subst hn; simp
  · have hml : MemLp id (n : ℝ≥0∞) (gaussianReal 0 v) := memLp_id_gaussianReal' _ (by simp)
    have hnorm : Integrable (fun x : ℝ => ‖x‖ ^ n) (gaussianReal 0 v) :=
      hml.integrable_norm_pow hn.ne'
    rw [← integrable_norm_iff (by fun_prop)]
    simpa [norm_pow] using hnorm

/-- **Expectation of a degree-≤4 polynomial under `N(0, v)`**, expressed via the moments
`E[X⁴]=3v²`, `E[X³]=0`, `E[X²]=v`, `E[X]=0`, `E[1]=1`. -/
theorem integral_poly4_gaussianReal (v : ℝ≥0) (a b c d e : ℝ) :
    ∫ x, (a * x ^ 4 + b * x ^ 3 + c * x ^ 2 + d * x + e) ∂(gaussianReal 0 v)
      = a * (3 * (v : ℝ) ^ 2) + c * (v : ℝ) + e := by
  have h4 : Integrable (fun x : ℝ => a * x ^ 4) (gaussianReal 0 v) :=
    (integrable_pow_gaussianReal v 4).const_mul a
  have h3 : Integrable (fun x : ℝ => b * x ^ 3) (gaussianReal 0 v) :=
    (integrable_pow_gaussianReal v 3).const_mul b
  have h2 : Integrable (fun x : ℝ => c * x ^ 2) (gaussianReal 0 v) :=
    (integrable_pow_gaussianReal v 2).const_mul c
  have h1 : Integrable (fun x : ℝ => d * x) (gaussianReal 0 v) := by
    simpa using (integrable_pow_gaussianReal v 1).const_mul d
  have h0 : Integrable (fun _ : ℝ => e) (gaussianReal 0 v) := integrable_const e
  have h43 : Integrable (fun x : ℝ => a * x ^ 4 + b * x ^ 3) (gaussianReal 0 v) := h4.add h3
  have h432 : Integrable (fun x : ℝ => a * x ^ 4 + b * x ^ 3 + c * x ^ 2) (gaussianReal 0 v) :=
    h43.add h2
  have h4321 : Integrable (fun x : ℝ => a * x ^ 4 + b * x ^ 3 + c * x ^ 2 + d * x)
      (gaussianReal 0 v) := h432.add h1
  rw [integral_add h4321 h0, integral_add h432 h1, integral_add h43 h2, integral_add h4 h3,
      integral_const_mul, integral_const_mul, integral_const_mul, integral_const_mul,
      integral_const, integral_pow_four_gaussianReal, integral_cube_gaussianReal,
      integral_sq_gaussianReal, integral_id_gaussianReal]
  simp

/-- **Orthonormality of the warm-up basis** `{ψ₀, ψ₁, ψ₂}` in `L²(N(0, v))` (L1-full, `k ≤ 2`).

For `σ > 0` with `σ² = v` (i.e. `σ` is the standard deviation), the scaled Hermite functions
`ψ_k = scaledHermite σ k` satisfy `∫ ψ_i·ψ_j dN(0,v) = δ_ij` for all `i, j ≤ 2`. Each cross-term
reduces, via the explicit forms (`scaledHermite_zero/one/two`), to a degree-≤4 polynomial whose
expectation is given by `integral_poly4_gaussianReal`.

Together with `VWK.Hermite` (the `ψ_k` equal scaled Hermite, Theorem 2 reduction) this is the
first **complete L1 block**: the warm-up basis is the orthonormal Gram–Schmidt basis of `L²(N(0,v))`. -/
theorem orthonormal_scaledHermite (v : ℝ≥0) (σ : ℝ) (hσ : 0 < σ) (hσ2 : σ ^ 2 = (v : ℝ)) :
    (∫ x, scaledHermite σ 0 x * scaledHermite σ 0 x ∂(gaussianReal 0 v) = 1) ∧
    (∫ x, scaledHermite σ 0 x * scaledHermite σ 1 x ∂(gaussianReal 0 v) = 0) ∧
    (∫ x, scaledHermite σ 0 x * scaledHermite σ 2 x ∂(gaussianReal 0 v) = 0) ∧
    (∫ x, scaledHermite σ 1 x * scaledHermite σ 1 x ∂(gaussianReal 0 v) = 1) ∧
    (∫ x, scaledHermite σ 1 x * scaledHermite σ 2 x ∂(gaussianReal 0 v) = 0) ∧
    (∫ x, scaledHermite σ 2 x * scaledHermite σ 2 x ∂(gaussianReal 0 v) = 1) := by
  have hσ0 : σ ≠ 0 := ne_of_gt hσ
  have hsne : Real.sqrt 2 ≠ 0 := by positivity
  have hsqrt2 : Real.sqrt 2 * Real.sqrt 2 = 2 := Real.mul_self_sqrt (by norm_num)
  -- pointwise: each product ψ_i·ψ_j as a degree-≤4 polynomial.
  have e00 : ∀ x : ℝ, scaledHermite σ 0 x * scaledHermite σ 0 x
      = (0 : ℝ) * x ^ 4 + 0 * x ^ 3 + 0 * x ^ 2 + 0 * x + 1 := by
    intro x; simp only [scaledHermite_zero]; ring
  have e01 : ∀ x : ℝ, scaledHermite σ 0 x * scaledHermite σ 1 x
      = (0 : ℝ) * x ^ 4 + 0 * x ^ 3 + 0 * x ^ 2 + (1 / σ) * x + 0 := by
    intro x; simp only [scaledHermite_zero, scaledHermite_one]; ring
  have e02 : ∀ x : ℝ, scaledHermite σ 0 x * scaledHermite σ 2 x
      = (0 : ℝ) * x ^ 4 + 0 * x ^ 3 + (1 / (σ ^ 2 * Real.sqrt 2)) * x ^ 2 + 0 * x
        + (-(1 / Real.sqrt 2)) := by
    intro x; simp only [scaledHermite_zero, scaledHermite_two σ x hσ0]; field_simp; ring
  have e11 : ∀ x : ℝ, scaledHermite σ 1 x * scaledHermite σ 1 x
      = (0 : ℝ) * x ^ 4 + 0 * x ^ 3 + (1 / σ ^ 2) * x ^ 2 + 0 * x + 0 := by
    intro x; simp only [scaledHermite_one]; field_simp; ring
  have e12 : ∀ x : ℝ, scaledHermite σ 1 x * scaledHermite σ 2 x
      = (0 : ℝ) * x ^ 4 + (1 / (σ ^ 3 * Real.sqrt 2)) * x ^ 3 + 0 * x ^ 2
        + (-(1 / (σ * Real.sqrt 2))) * x + 0 := by
    intro x; simp only [scaledHermite_one, scaledHermite_two σ x hσ0]; field_simp; ring
  have e22 : ∀ x : ℝ, scaledHermite σ 2 x * scaledHermite σ 2 x
      = (1 / (2 * σ ^ 4)) * x ^ 4 + 0 * x ^ 3 + (-(1 / σ ^ 2)) * x ^ 2 + 0 * x + (1 / 2) := by
    intro x
    have hden : σ ^ 2 * Real.sqrt 2 * (σ ^ 2 * Real.sqrt 2) = 2 * σ ^ 4 := by
      linear_combination σ ^ 4 * hsqrt2
    simp only [scaledHermite_two σ x hσ0]
    rw [div_mul_div_comm, hden]; field_simp; ring
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp_rw [e00]; rw [integral_poly4_gaussianReal]; ring
  · simp_rw [e01]; rw [integral_poly4_gaussianReal]; ring
  · simp_rw [e02]; rw [integral_poly4_gaussianReal, ← hσ2]; field_simp; ring
  · simp_rw [e11]; rw [integral_poly4_gaussianReal, ← hσ2]; field_simp; ring
  · simp_rw [e12]; rw [integral_poly4_gaussianReal]; ring
  · simp_rw [e22]; rw [integral_poly4_gaussianReal, ← hσ2]; field_simp; ring

end VWK
