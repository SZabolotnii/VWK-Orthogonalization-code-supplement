import Mathlib

/-!
# VWK.Krawtchouk — the finite-support VWK instance: Binomial(3, p) → Krawtchouk

The discrete (finite-support) sibling of `VWK/Orthonormality.lean`. Where the
Gaussian case needs measure theory (`gaussianReal`, Lebesgue `∫`), a FINITE support
turns the `L²(P)` inner product into a finite SUM, so the VWK / Gram-Schmidt
orthogonality (Lemma 1 / Theorem 2′, `theorem-draft.md` §5.1) becomes pure
real-field algebra. This is exactly the `§10.2` **L1′** point: Binomial→Krawtchouk
is the easiest nontrivial (non-Gaussian) instance — no `gaussianReal`, no
integrability, no `axiom of choice`, just `ring`.

For `Binomial(3, p)` (`q = 1 - p`) with weights `w_x = C(3,x) p^x q^{3-x}` on the
support `x ∈ {0,1,2,3}`, the monic degree-`0,1,2` orthogonal polynomials are the
(monic) Krawtchouk polynomials
  `π₀ = 1`,  `π₁(t) = t − μ`,  `π₂(t) = (t − μ)² − (μ₃/σ²)(t − μ) − σ²`,
with `μ = 3p`, `σ² = 3p(1−p)`, `μ₃ = 3p(1−p)(1−2p)`, so `μ₃/σ² = 1−2p` — the PMM2
`c₃` skew correction, absent in the Gaussian/Hermite case (`μ₃ = 0`). Everything
below is `sorry`-free by `ring`; `#print axioms` at the end confirms it.
-/

namespace VWK
namespace Binom3

/-- `Binomial(3, p)` weights at `x = 0,1,2,3` (`q = 1 - p`). -/
noncomputable def w0 (p : ℝ) : ℝ := (1 - p) ^ 3
noncomputable def w1 (p : ℝ) : ℝ := 3 * p * (1 - p) ^ 2
noncomputable def w2 (p : ℝ) : ℝ := 3 * p ^ 2 * (1 - p)
noncomputable def w3 (p : ℝ) : ℝ := p ^ 3

/-- Weighted inner product over the support `{0,1,2,3}`: `⟪f,g⟫ = Σ_x w_x f(x) g(x)`. -/
noncomputable def ip (p : ℝ) (f g : ℝ → ℝ) : ℝ :=
  w0 p * f 0 * g 0 + w1 p * f 1 * g 1 + w2 p * f 2 * g 2 + w3 p * f 3 * g 3

/-- The monic VWK / Krawtchouk polynomials of `Binomial(3, p)` at degrees `0,1,2`. -/
def π₀ : ℝ → ℝ := fun _ => 1
noncomputable def π₁ (p : ℝ) : ℝ → ℝ := fun t => t - 3 * p
noncomputable def π₂ (p : ℝ) : ℝ → ℝ :=
  fun t => (t - 3 * p) ^ 2 - (1 - 2 * p) * (t - 3 * p) - 3 * p * (1 - p)

/-- Total mass `= 1` (the binomial theorem for `N = 3`: `((1-p)+p)^3 = 1`). -/
theorem mass (p : ℝ) : ip p π₀ π₀ = 1 := by
  simp only [ip, π₀, w0, w1, w2, w3]; ring

/-- Mean `E[x] = 3p` — identifies `μ = 3p`. -/
theorem mean (p : ℝ) : ip p π₀ (fun t => t) = 3 * p := by
  simp only [ip, π₀, w0, w1, w2, w3]; ring

/-- Variance `σ² = E[(x−μ)²] = 3p(1−p)` — identifies the normalizing scale of `ψ₁`. -/
theorem variance (p : ℝ) : ip p (π₁ p) (π₁ p) = 3 * p * (1 - p) := by
  simp only [ip, π₁, w0, w1, w2, w3]; ring

/-- Third central moment `μ₃ = E[(x−μ)³] = 3p(1−p)(1−2p)`; hence `μ₃/σ² = 1−2p`,
the skew correction carried by `π₂` (and the `c₃` lever of PMM2). -/
theorem central_three (p : ℝ) :
    ip p π₀ (fun t => (t - 3 * p) ^ 3) = 3 * p * (1 - p) * (1 - 2 * p) := by
  simp only [ip, π₀, w0, w1, w2, w3]; ring

/-- **Orthogonality `π₀ ⟂ π₁`** — the VWK basis is orthogonal at degrees `0,1`. -/
theorem orth_01 (p : ℝ) : ip p π₀ (π₁ p) = 0 := by
  simp only [ip, π₀, π₁, w0, w1, w2, w3]; ring

/-- **Orthogonality `π₀ ⟂ π₂`**. -/
theorem orth_02 (p : ℝ) : ip p π₀ (π₂ p) = 0 := by
  simp only [ip, π₀, π₂, w0, w1, w2, w3]; ring

/-- **Orthogonality `π₁ ⟂ π₂`** — closes orthogonality of `{π₀,π₁,π₂}` (the binomial
instance of Lemma 1 / Theorem 2′, `k ≤ 2`). -/
theorem orth_12 (p : ℝ) : ip p (π₁ p) (π₂ p) = 0 := by
  simp only [ip, π₁, π₂, w0, w1, w2, w3]; ring

/-- The degree-2 monic Krawtchouk equals the Gram–Schmidt closed form built from the
binomial's own moments: `π₂(t) = (t−μ)² − (μ₃/σ²)(t−μ) − σ²` with `μ=3p`, `μ₃/σ²=1−2p`,
`σ²=3p(1−p)`. (For `μ₃=0` this collapses to the Hermite form `(t−μ)²−σ²`.) -/
theorem pi_two_eq (p t : ℝ) :
    π₂ p t = (t - 3 * p) ^ 2 - (1 - 2 * p) * (t - 3 * p) - 3 * p * (1 - p) := rfl

end Binom3
end VWK

-- sorry-free certificate: each proof depends only on the standard three axioms.
#print axioms VWK.Binom3.mass
#print axioms VWK.Binom3.orth_01
#print axioms VWK.Binom3.orth_02
#print axioms VWK.Binom3.orth_12
#print axioms VWK.Binom3.variance
#print axioms VWK.Binom3.central_three
