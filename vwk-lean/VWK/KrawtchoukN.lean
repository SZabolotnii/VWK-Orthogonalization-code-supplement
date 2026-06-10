import Mathlib

/-!
# VWK.KrawtchoukN — the finite-support VWK instance for *arbitrary* `N`: Binomial(N, p) → Krawtchouk

The general-`N` sibling of `VWK/Krawtchouk.lean` (which fixed `N = 3` and closed
everything by `ring`). Here `N` is a free natural number, so `ring` no longer reaches
the binomial sums; instead we ride Mathlib's **Bernstein-polynomial moment identities**.

`bernsteinPolynomial ℝ n ν = C(n,ν) Xᵛ (1−X)ⁿ⁻ᵛ` is exactly the `Binomial(n, X)` weight
with the success probability carried by the indeterminate `X`. Mathlib proves, as
identities in `ℝ[X]`,
  `bernsteinPolynomial.sum`       `∑ b = 1`                       (total mass),
  `bernsteinPolynomial.sum_smul`  `∑ ν • b = n • X`               (first moment),
  `bernsteinPolynomial.variance`  `∑ (n•X − ν)² b = n•X(1−X)`     (second central moment).
Evaluating each at `X = p` turns them into the real-number moments of `Binomial(n, p)`,
which is all the VWK / Gram–Schmidt construction (Lemma 1 / Theorem 2′, `theorem-draft.md`
§5.1) needs for the degree-`≤ 2` Krawtchouk basis.

For `Binomial(N, p)`: `μ = Np`, `σ² = Np(1−p)`, and the monic Krawtchouk polynomials are
  `K₀ = 1`, `K₁(t) = t − μ`, `K₂(t) = (t − μ)² − (1 − 2p)(t − μ) − σ²`,
where the skew coefficient `μ₃/σ² = 1 − 2p` is *independent of `N`* (it is the PMM2 `c₃`
lever, absent in the Gaussian/Hermite case). We prove sorry-free, for **all `N` and `p`**,
the **full** `{K₀,K₁,K₂}` orthogonality: mass `= 1`, mean `= Np`, `‖K₁‖² = Np(1−p)`, and
all three orthogonalities `K₀ ⟂ K₁` (`orth01`), `K₀ ⟂ K₂` (`orth02`), `K₁ ⟂ K₂` (`orth12`).

`K₀ ⟂ K₂` notably needs *no* third moment: the `μ₃` term of `K₂` rides `E[t − μ] = 0`.
The cross term `K₁ ⟂ K₂` does need the general-`N` third central moment
`μ₃ = Np(1−p)(1−2p)`; we obtain it from the **third Bernstein factorial moment**
`∑ ν(ν−1)(ν−2)·b = n(n−1)(n−2)·X³` (`sum_mul_mul_smul`), which Mathlib does *not* provide
(it stops at the second, `sum_mul_smul`). We prove it here by extending Mathlib's
MvPolynomial `pderiv` pattern to the third `∂/∂x` of `(x+y)ⁿ` — a candidate upstream PR —
and assemble `central3_N : ∑ (ν−Np)³·w = Np(1−p)(1−2p)`. (The fixed `N = 3` instance in
`VWK/Krawtchouk.lean` closes everything by `ring`; this module is its arbitrary-`N` lift.)
-/

open Polynomial Finset

namespace VWK
namespace BinomN

/-- `Binomial(n, p)` weight at value `ν`: the Bernstein polynomial evaluated at `p`. -/
noncomputable def wN (n ν : ℕ) (p : ℝ) : ℝ := (bernsteinPolynomial ℝ n ν).eval p

/-- Weighted inner product over the support `{0,…,n}`: `⟪f,g⟫ = Σ_ν w_ν f(ν) g(ν)`. -/
noncomputable def ipN (n : ℕ) (p : ℝ) (f g : ℝ → ℝ) : ℝ :=
  ∑ ν ∈ Finset.range (n + 1), wN n ν p * f (ν : ℝ) * g (ν : ℝ)

/-- The monic VWK / Krawtchouk polynomials of `Binomial(n, p)` at degrees `0,1,2`. -/
def K0 : ℝ → ℝ := fun _ => 1
noncomputable def K1 (n : ℕ) (p : ℝ) : ℝ → ℝ := fun t => t - n * p
noncomputable def K2 (n : ℕ) (p : ℝ) : ℝ → ℝ :=
  fun t => (t - n * p) ^ 2 - (1 - 2 * p) * (t - n * p) - n * p * (1 - p)

/-! ### Moments from the Bernstein identities (evaluated at `X = p`). -/

/-- Total mass `= 1` (Bernstein `sum` at `X = p`). -/
theorem mass_N (n : ℕ) (p : ℝ) : ∑ ν ∈ Finset.range (n + 1), wN n ν p = 1 := by
  simp only [wN, ← Polynomial.eval_finset_sum, bernsteinPolynomial.sum, Polynomial.eval_one]

/-- First moment `Σ ν w_ν = n p` (Bernstein `sum_smul` at `X = p`). -/
theorem mean_N (n : ℕ) (p : ℝ) :
    ∑ ν ∈ Finset.range (n + 1), (ν : ℝ) * wN n ν p = n * p := by
  have h := congrArg (Polynomial.eval p) (bernsteinPolynomial.sum_smul ℝ n)
  simp only [nsmul_eq_mul, Polynomial.eval_finset_sum, Polynomial.eval_mul,
    Polynomial.eval_natCast, Polynomial.eval_X] at h
  simpa [wN] using h

/-- Second central moment `Σ (np − ν)² w_ν = n p (1−p)` (Bernstein `variance` at `X = p`). -/
theorem var_N (n : ℕ) (p : ℝ) :
    ∑ ν ∈ Finset.range (n + 1), ((n : ℝ) * p - ν) ^ 2 * wN n ν p = n * p * (1 - p) := by
  have h := congrArg (Polynomial.eval p) (bernsteinPolynomial.variance ℝ n)
  simp only [nsmul_eq_mul, Polynomial.eval_finset_sum, Polynomial.eval_mul, Polynomial.eval_pow,
    Polynomial.eval_sub, Polynomial.eval_natCast, Polynomial.eval_X, Polynomial.eval_one] at h
  simpa [wN] using h

/-! ### VWK orthogonality at degrees `0,1,2` for arbitrary `N`. -/

/-- Total mass as an inner product: `⟪K₀,K₀⟫ = 1`. -/
theorem mass (n : ℕ) (p : ℝ) : ipN n p K0 K0 = 1 := by
  simp only [ipN, K0, mul_one]; exact mass_N n p

/-- `‖K₁‖² = ⟪K₁,K₁⟫ = n p (1−p)` — the squared norm of the linear Krawtchouk. -/
theorem norm1 (n : ℕ) (p : ℝ) : ipN n p (K1 n p) (K1 n p) = n * p * (1 - p) := by
  have e : ipN n p (K1 n p) (K1 n p)
      = ∑ ν ∈ Finset.range (n + 1), ((n : ℝ) * p - ν) ^ 2 * wN n ν p := by
    simp only [ipN, K1]; exact Finset.sum_congr rfl (fun ν _ => by ring)
  rw [e, var_N]

/-- **Orthogonality `K₀ ⟂ K₁`** for all `N`: `⟪K₀,K₁⟫ = E[t − μ] = 0`. -/
theorem orth01 (n : ℕ) (p : ℝ) : ipN n p K0 (K1 n p) = 0 := by
  have e : ipN n p K0 (K1 n p)
      = (∑ ν ∈ Finset.range (n + 1), (ν : ℝ) * wN n ν p)
        - (n * p) * (∑ ν ∈ Finset.range (n + 1), wN n ν p) := by
    simp only [ipN, K0, K1]
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl (fun ν _ => by ring)
  rw [e, mean_N, mass_N]; ring

/-- **Orthogonality `K₀ ⟂ K₂`** for all `N`. Needs only mass + mean + variance: the third
moment in `K₂` multiplies `E[t − μ] = 0`, so it drops out. -/
theorem orth02 (n : ℕ) (p : ℝ) : ipN n p K0 (K2 n p) = 0 := by
  have hmid : (∑ ν ∈ Finset.range (n + 1), ((ν : ℝ) - n * p) * wN n ν p) = 0 := by
    have e1 : (∑ ν ∈ Finset.range (n + 1), ((ν : ℝ) - n * p) * wN n ν p)
        = (∑ ν ∈ Finset.range (n + 1), (ν : ℝ) * wN n ν p)
          - (n * p) * (∑ ν ∈ Finset.range (n + 1), wN n ν p) := by
      rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl (fun ν _ => by ring)
    rw [e1, mean_N, mass_N]; ring
  have e : ipN n p K0 (K2 n p)
      = (∑ ν ∈ Finset.range (n + 1), ((n : ℝ) * p - ν) ^ 2 * wN n ν p)
        - (1 - 2 * p) * (∑ ν ∈ Finset.range (n + 1), ((ν : ℝ) - n * p) * wN n ν p)
        - (n * p * (1 - p)) * (∑ ν ∈ Finset.range (n + 1), wN n ν p) := by
    simp only [ipN, K0, K2]
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl (fun ν _ => by ring)
  rw [e, var_N, hmid, mass_N]; ring

/-! ### Stretch: the general-`N` third factorial moment, closing `K₁ ⟂ K₂`.

Mathlib stops at the *second* Bernstein factorial moment (`sum_mul_smul`). The third one,
`Σ ν(ν−1)(ν−2) • b = n(n−1)(n−2) • X³`, is what the degree-`(1,2)` cross term needs. We
prove it by faithfully extending Mathlib's `sum_mul_smul` proof to the *third* `∂/∂x` of
`(x+y)ⁿ`. -/

open MvPolynomial in
/-- Third Bernstein factorial moment (the third `x`-derivative of `(x+y)ⁿ`). -/
theorem sum_mul_mul_smul (n : ℕ) :
    (∑ ν ∈ Finset.range (n + 1), (ν * (ν - 1) * (ν - 2)) • bernsteinPolynomial ℝ n ν) =
      (n * (n - 1) * (n - 1 - 1)) • (Polynomial.X : ℝ[X]) ^ 3 := by
  let x : MvPolynomial Bool ℝ := MvPolynomial.X true
  let y : MvPolynomial Bool ℝ := MvPolynomial.X false
  have pderiv_true_x : pderiv true x = 1 := by rw [pderiv_X]; rfl
  have pderiv_true_y : pderiv true y = 0 := by rw [pderiv_X]; rfl
  let e : Bool → ℝ[X] := fun i => cond i Polynomial.X (1 - Polynomial.X)
  trans MvPolynomial.aeval e
      (pderiv true (pderiv true (pderiv true ((x + y) ^ n)))) * Polynomial.X ^ 3
  · have w : ∀ k : ℕ, (k * (k - 1) * (k - 2)) • bernsteinPolynomial ℝ n k =
        (n.choose k : ℝ[X]) * ((1 - Polynomial.X) ^ (n - k) *
          ((k : ℝ[X]) * ((↑(k - 1) : ℝ[X]) *
            ((↑(k - 1 - 1) : ℝ[X]) * Polynomial.X ^ (k - 1 - 1 - 1))))) * Polynomial.X ^ 3 := by
      rintro (_ | _ | _ | k)
      · simp
      · simp
      · simp
      · rw [bernsteinPolynomial]
        simp only [← natCast_mul, Nat.add_succ_sub_one, add_zero, pow_succ]
        push_cast
        ring
    rw [add_pow, map_sum (pderiv true), map_sum (pderiv true), map_sum (pderiv true),
      map_sum (MvPolynomial.aeval e), Finset.sum_mul]
    refine Finset.sum_congr rfl fun k _ => (w k).trans ?_
    simp only [x, y, e, pderiv_true_x, pderiv_true_y, smul_eq_mul, nsmul_eq_mul,
      Bool.cond_true, Bool.cond_false, add_zero, zero_add, mul_zero, smul_zero, mul_one,
      MvPolynomial.aeval_X,
      Derivation.leibniz_pow, Derivation.leibniz, Derivation.map_natCast, map_natCast, map_pow,
      map_mul]
  · simp only [x, y, e, (pderiv _).leibniz_pow,
      (pderiv true).map_add, pderiv_true_x, pderiv_true_y, smul_eq_mul, add_zero,
      mul_one, map_nsmul, map_pow, map_add, Bool.cond_true,
      Bool.cond_false, MvPolynomial.aeval_X, add_sub_cancel, one_pow, smul_smul,
      smul_one_mul]

/-- The `ℕ`-truncated falling factorial `ν(ν−1)` casts to the real `ν(ν−1)`
(agree at `ν = 0`, where both sides vanish). -/
theorem cast_ffact2 (ν : ℕ) : ((ν * (ν - 1) : ℕ) : ℝ) = (ν : ℝ) * ((ν : ℝ) - 1) := by
  rcases ν with _ | ν
  · norm_num
  · simp only [Nat.succ_sub_one]; push_cast; ring

/-- The `ℕ`-truncated falling factorial `ν(ν−1)(ν−2)` casts to the real `ν(ν−1)(ν−2)`
(agree at `ν = 0,1,2`, where both sides vanish). -/
theorem cast_ffact3 (ν : ℕ) :
    ((ν * (ν - 1) * (ν - 2) : ℕ) : ℝ) = (ν : ℝ) * ((ν : ℝ) - 1) * ((ν : ℝ) - 2) := by
  rcases ν with _ | _ | _ | ν
  · norm_num
  · norm_num
  · norm_num
  · simp only [Nat.succ_sub_succ]; push_cast; ring

/-- Second real factorial moment `Σ ν(ν−1) w_ν = n(n−1) p²` (Bernstein `sum_mul_smul`). -/
theorem fact2_N (n : ℕ) (p : ℝ) :
    ∑ ν ∈ Finset.range (n + 1), ((ν : ℝ) * ((ν : ℝ) - 1)) * wN n ν p
      = (n : ℝ) * ((n : ℝ) - 1) * p ^ 2 := by
  have h := congrArg (Polynomial.eval p) (bernsteinPolynomial.sum_mul_smul ℝ n)
  simp only [nsmul_eq_mul, Polynomial.eval_finset_sum, Polynomial.eval_mul, Polynomial.eval_pow,
    Polynomial.eval_natCast, Polynomial.eval_X] at h
  calc ∑ ν ∈ Finset.range (n + 1), ((ν : ℝ) * ((ν : ℝ) - 1)) * wN n ν p
      = ∑ ν ∈ Finset.range (n + 1), ((ν * (ν - 1) : ℕ) : ℝ) * wN n ν p :=
        Finset.sum_congr rfl (fun ν _ => by rw [cast_ffact2])
    _ = ((n * (n - 1) : ℕ) : ℝ) * p ^ 2 := by simp only [wN]; exact h
    _ = (n : ℝ) * ((n : ℝ) - 1) * p ^ 2 := by rw [cast_ffact2]

/-- Third real factorial moment `Σ ν(ν−1)(ν−2) w_ν = n(n−1)(n−2) p³`
(our `sum_mul_mul_smul`). -/
theorem fact3_N (n : ℕ) (p : ℝ) :
    ∑ ν ∈ Finset.range (n + 1), ((ν : ℝ) * ((ν : ℝ) - 1) * ((ν : ℝ) - 2)) * wN n ν p
      = (n : ℝ) * ((n : ℝ) - 1) * ((n : ℝ) - 2) * p ^ 3 := by
  have h := congrArg (Polynomial.eval p) (sum_mul_mul_smul n)
  simp only [nsmul_eq_mul, Polynomial.eval_finset_sum, Polynomial.eval_mul, Polynomial.eval_pow,
    Polynomial.eval_natCast, Polynomial.eval_X] at h
  calc ∑ ν ∈ Finset.range (n + 1), ((ν : ℝ) * ((ν : ℝ) - 1) * ((ν : ℝ) - 2)) * wN n ν p
      = ∑ ν ∈ Finset.range (n + 1), ((ν * (ν - 1) * (ν - 2) : ℕ) : ℝ) * wN n ν p :=
        Finset.sum_congr rfl (fun ν _ => by rw [cast_ffact3])
    _ = ((n * (n - 1) * (n - 2) : ℕ) : ℝ) * p ^ 3 := by simp only [wN]; exact h
    _ = (n : ℝ) * ((n : ℝ) - 1) * ((n : ℝ) - 2) * p ^ 3 := by rw [cast_ffact3]

/-- Centered first moment `Σ (ν − np) w_ν = 0`. -/
theorem mean_centered (n : ℕ) (p : ℝ) :
    ∑ ν ∈ Finset.range (n + 1), ((ν : ℝ) - n * p) * wN n ν p = 0 := by
  have e : (∑ ν ∈ Finset.range (n + 1), ((ν : ℝ) - n * p) * wN n ν p)
      = (∑ ν ∈ Finset.range (n + 1), (ν : ℝ) * wN n ν p)
        - (n * p) * (∑ ν ∈ Finset.range (n + 1), wN n ν p) := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl (fun ν _ => by ring)
  rw [e, mean_N, mass_N]; ring

/-- **Third central moment** `Σ (ν − np)³ w_ν = n p (1−p)(1−2p)`, assembled from the first
three factorial moments via `(t−a)³ = ν⁽³⁾ + (3−3a)ν⁽²⁾ + (1−3a+3a²)ν − a³`. -/
theorem central3_N (n : ℕ) (p : ℝ) :
    ∑ ν ∈ Finset.range (n + 1), ((ν : ℝ) - n * p) ^ 3 * wN n ν p
      = (n : ℝ) * p * (1 - p) * (1 - 2 * p) := by
  have e : (∑ ν ∈ Finset.range (n + 1), ((ν : ℝ) - n * p) ^ 3 * wN n ν p)
      = (∑ ν ∈ Finset.range (n + 1), ((ν : ℝ) * ((ν : ℝ) - 1) * ((ν : ℝ) - 2)) * wN n ν p)
        + (3 - 3 * (n * p)) * (∑ ν ∈ Finset.range (n + 1), ((ν : ℝ) * ((ν : ℝ) - 1)) * wN n ν p)
        + (1 - 3 * (n * p) + 3 * (n * p) ^ 2) * (∑ ν ∈ Finset.range (n + 1), (ν : ℝ) * wN n ν p)
        - (n * p) ^ 3 * (∑ ν ∈ Finset.range (n + 1), wN n ν p) := by
    rw [Finset.mul_sum, Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib,
      ← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl (fun ν _ => by ring)
  rw [e, fact3_N, fact2_N, mean_N, mass_N]; ring

/-- **Orthogonality `K₁ ⟂ K₂`** for all `N` — closes the full `{K₀,K₁,K₂}` orthogonality of
the general-`N` Krawtchouk basis. Needs the third central moment (`central3_N`). -/
theorem orth12 (n : ℕ) (p : ℝ) : ipN n p (K1 n p) (K2 n p) = 0 := by
  have e : ipN n p (K1 n p) (K2 n p)
      = (∑ ν ∈ Finset.range (n + 1), ((ν : ℝ) - n * p) ^ 3 * wN n ν p)
        - (1 - 2 * p) * (∑ ν ∈ Finset.range (n + 1), ((n : ℝ) * p - ν) ^ 2 * wN n ν p)
        - (n * p * (1 - p)) * (∑ ν ∈ Finset.range (n + 1), ((ν : ℝ) - n * p) * wN n ν p) := by
    simp only [ipN, K1, K2]
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl (fun ν _ => by ring)
  rw [e, central3_N, var_N, mean_centered]; ring

end BinomN
end VWK

-- sorry-free certificate: each proof depends only on the standard three axioms.
#print axioms VWK.BinomN.mass
#print axioms VWK.BinomN.norm1
#print axioms VWK.BinomN.orth01
#print axioms VWK.BinomN.orth02
#print axioms VWK.BinomN.sum_mul_mul_smul
#print axioms VWK.BinomN.central3_N
#print axioms VWK.BinomN.orth12
