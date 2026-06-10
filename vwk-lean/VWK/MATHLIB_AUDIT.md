# VWK — Mathlib Audit

*Версія: 0.2 | Дата: 2026-06-08 | Mathlib: v4.26.0 (commit `2df2f0150c275ad53cb3c90f7c98ec15a56a1a67`) | Lean: v4.26.0*

Виконано згідно з `../lean-handoff.md` §3.4. Джерело: повний source cloned Mathlib
(`grep` по `.lake/packages/mathlib/Mathlib/`). Легенда статусу: ✅ доступне напряму,
🟡 часткове / потребує деривації, ❌ відсутнє.

---

## 1. Зведена таблиця

| Інгредієнт | Статус | Модуль | Ключові декларації |
|---|---|---|---|
| Gram–Schmidt | ✅ | `Analysis.InnerProductSpace.GramSchmidtOrtho` | `gramSchmidt`, `gramSchmidtNormed`, `gramSchmidtOrthonormalBasis`, `gramSchmidt_orthogonal` |
| Hermite (probabilist's) | ✅ | `RingTheory.Polynomial.Hermite.Basic` | `hermite`, `hermite_zero/one/succ`, `hermite_monic`, `coeff_hermite_explicit`, `degree_hermite` |
| Hermite ↔ Gaussian density (Rodrigues) | ✅ | `RingTheory.Polynomial.Hermite.Gaussian` | `hermite_eq_deriv_gaussian`, `deriv_gaussian_eq_hermite_mul_gaussian` |
| Gaussian-міра | ✅ | `Probability.Distributions.Gaussian.Real` | `gaussianReal`, `gaussianPDFReal`, `IsGaussian`, `memLp_id_gaussianReal` |
| Gaussian мат. сподівання / дисперсія | ✅ | те саме | `integral_id_gaussianReal` (= μ), `variance_id_gaussianReal` (= v) |
| L²(P) як inner product space | ✅ | `MeasureTheory.Function.L2Space` | `instance L2.innerProductSpace`, `inner_def`, `integral_inner`, `MemLp.const_inner` |
| MGF / CGF | ✅ | `Probability.Moments.{Basic,MGFAnalytic,Tilted}` | `mgf`, `cgf`, `mgf_fun_id_gaussianReal` |
| **Вищі Gaussian-моменти E[X³], E[X⁴], E[X^{2n}]** | 🟡 | локально `VWK/GaussianMoments.lean` | `E[X²]=v`, `E[X³]=0`, `E[X⁴]=3v²` доведено; загальний `E[X^{2n}]` ще відсутній |
| **Ортогональність Hermite за Gaussian** (`∫ Hₘ·Hₙ dN = n!·δ`) | 🟡 | локально `VWK/Orthonormality.lean` | `{ψ₀,ψ₁,ψ₂}` доведено; довільний `k` ще відсутній |
| **Кумулянти cᵣ** (як коеф. ряду cgf) | ❌ | — | є лише cgf, не покоефіцієнтний розклад |
| **Volterra-функціонали, cross-кумулянти в L²(P^⊗n)** | ❌ | — | немає формального об'єкта |

---

## 2. Деталі

### 2.1. Gram–Schmidt (✅)
`gramSchmidt (𝕜) (f : ι → E) : ι → E` визначено для `[RCLike 𝕜] [NormedAddCommGroup E]
[InnerProductSpace 𝕜 E]`, індекс `[LinearOrder ι] [LocallyFiniteOrderBot ι] [WellFoundedLT ι]`.
- `ℕ` задовольняє вимоги на індекс ⇒ послідовність мономіалів `fun k : ℕ => …` валідна.
- `Lp ℝ 2 μ` має instance `L2.innerProductSpace` над `ℝ` (`RCLike ℝ`) ⇒ **`gramSchmidt ℝ`
  застосовний напряму до елементів `Lp ℝ 2 μ`, без uplift зі скінченновимірного підпростору**.
- Ортонормованість результату — теорема Mathlib (`gramSchmidt_orthogonal`, `gramSchmidtNormed`).
- ⚠ Mathlib `gramSchmidt` **не** фіксує знак provідного коефіцієнта; вимогу «leading coeff > 0»
  з Lemma 1 треба додати окремим orientation-аргументом.

### 2.2. Hermite (✅, але ℤ-коефіцієнти)
`Polynomial.hermite : ℕ → ℤ[X]` (probabilist's, монічний). Рекурсія
`hermite_succ : hermite (n+1) = X * hermite n - derivative (hermite n)`. Оскільки коефіцієнти
в `ℤ`, обчислення в точці `x : ℝ` — через `Polynomial.aeval` (ℤ-алгебра `ℤ[X] → ℝ`), а не
`.eval` (це коригує placeholder у handoff §4.2). Файл `Hermite/Gaussian.lean` дає формулу
Родрігеса — природний інгредієнт для майбутнього доведення ортогональності.

### 2.3. Gaussian-моменти (🟡 — частково закрито локально)
Доступні лише **μ** (`integral_id_gaussianReal`) і **дисперсія** (`variance_id_gaussianReal`).
`memLp_id_gaussianReal : MemLp id p (gaussianReal μ v)` гарантує **скінченність усіх моментів**
(інтегровність `xⁿ` — безкоштовна), але **значень** E[X³]=0, E[X⁴]=3σ⁴, загалом
E[X^{2n}]=(2n−1)!!·σ^{2n} немає. Це і є блокер ортонормованості `{ψ_k}` для `k ≥ 2`.

**Оновлення 2026-06-08.** Локально закрито потрібний warm-up блок:
`VWK/GaussianMoments.lean` доводить `E[X²]=v`, `E[X³]=0`, `E[X⁴]=3v²`, а
`VWK/Orthonormality.lean` збирає ортонормованість `{ψ₀,ψ₁,ψ₂}`. Відкритим лишається
загальний моментний результат `E[X^{2n}]` і довільне `k`.

**Шляхи деривації відсутніх моментів:**
- (A) **Через ортогональність Hermite за Gaussian** — довести `∫ Hₘ·Hₙ dN(0,σ²) = …`
  з формули Родрігеса (`hermite_eq_deriv_gaussian`) інтегруванням частинами n разів.
  Це закриває і моменти, і ортонормованість одразу; broadly useful ⇒ кандидат на upstream.
- (B) **Через MGF** — `mgf` гаусіани аналітична (`Moments.MGFAnalytic`); 4-та похідна в нулі дає
  E[X⁴]. Менш загально, але достатньо для k ≤ 4.
- (C) **Через raw Gaussian integrals** — `Analysis.SpecialFunctions.Gaussian.GaussianIntegral`
  (`integral_gaussian`, `integral_mul_cexp_neg_mul_sq`, `integral_gaussian_sq_complex`) + заміна
  змінних до pdf + IBP. Найбільш «ручний».

### 2.4. Кумулянти (❌)
`cgf` (cumulant generating function) є, але покоефіцієнтного розкладу `cᵣ` і моста
момент↔кумулянт (Bell-поліноми) немає. Підтверджує handoff §3.3.

### 2.5. L²(P) (✅)
`L2.innerProductSpace` + `inner_def` (⟪f,g⟫ = ∫ ⟪f a, g a⟫ dμ; для дійсних = ∫ f·g) роблять
`Lp ℝ 2 (gaussianReal 0 v)` готовим дійсним гільбертовим простором для формулювань
ортонормованості й gramSchmidt.

---

## 3. Наслідки для таргетів L1–L4

| Таргет | Здійсненність | Що блокує / що треба |
|---|---|---|
| **L1** Theorem 2 (reduction), `k ≤ 2` | 🟢 **ЗРОБЛЕНО sorry-free** (`Hermite.lean`, `GaussianMoments.lean`, `Orthonormality.lean`) | ще треба зв'язати closed forms з actual `gramSchmidt` output |
| **L2** Lemma 1 (GS у L²(P), general P) | 🟢 здійсненне | репрезентація `xᵏ` як `Lp`-елемента (з `memLp_id`-стилю); orientation для знака (§2.1) |
| **L3** Theorem 3 (kernel map, discrete s=2) | 🟢 здійсненне | чиста скінченновимірна лін. алгебра, Mathlib-gaps немає |
| **L4** Theorem 1 (повна загальність) | 🔴 довгострокове | кумулянти (§2.4) + Volterra-функціонали — обидва відсутні |

---

## 4. Відповіді на open questions handoff §4.5

1. **Чи `gramSchmidt` працює напряму для `Lp ℝ 2 μ`?** — **Так, напряму.** `Lp ℝ 2 μ` має
   `L2.innerProductSpace` над `ℝ`; `ℕ` — валідний індекс. Uplift зі скінченновимірного не потрібен.
2. **Чи нормалізація знака provідного коеф. автоматична?** — **Ні.** `gramSchmidtNormed` дає
   одиничну норму, але знак іде за порядком входу. «Leading coeff > 0» — додати явно.
3. **Direct computation vs `gramSchmidt`-machinery?** — Для L1-mini direct computation простіше
   (вже зроблено). Для L2 (general P) — використовувати `gramSchmidt` + його теорему ортогональності.

---

## 5. Action items

- [x] **GaussianMoments warm-up**: довести `E[X²]`, `E[X³]`, `E[X⁴]` і ортонормованість
      `{ψ₀,ψ₁,ψ₂}`.
- [ ] **GaussianMoments general**: довести `E[X^{2n}]` та/або ортогональність Hermite
      за Gaussian для довільного `k`. Кандидат на **Mathlib upstream**.
- [ ] L2: лема `xᵏ ∈ Lp (gaussianReal 0 v)` + репрезентація мономіалів.
- [ ] Рішення §8.1 (кумулянти): рекомендація — **bypass** (moments-only) до L4 (див. `STATUS.md`).
