# Status: 2026-05-23

*VWK Lean-формалізація. Протокол згідно з `../lean-handoff.md` §9. Lean v4.26.0 / Mathlib v4.26.0.*

---

## Done

- **Проєкт `vwk-lean/` піднято і будується.** `lake build VWK` — зелений; 2026-06-08
  команда завершилась як `Build completed successfully (7748 jobs)`. Пребілт Mathlib v4.26.0
  перевикористано з `локального кешу Mathlib (.lake/packages)`
  через локальний `.lake/packages` symlink, тому повна перебудова Mathlib не потрібна.
  Деталі — `README.md`.
- **Mathlib audit завершено** → `MATHLIB_AUDIT.md` (повна таблиця + шляхи деривації gaps).
- **L1-mini (Theorem 2, reduction to Hermite, `k ≤ 2`) — формалізовано БЕЗ `sorry`** у
  `VWK/Hermite.lean`:
  - `hermite_two : hermite 2 = X ^ 2 - 1` (з рекурсії `hermite_succ`);
  - `scaledHermite σ k x := aeval (x/σ) (hermite k) / √(k!)` — кандидат на `ψ_k`;
  - `scaledHermite_zero : ψ₀ = 1`;
  - `scaledHermite_one : ψ₁(x) = x/σ`;
  - `scaledHermite_two : ψ₂(x) = (x² − σ²)/(σ²√2)` (за `σ ≠ 0`).
  - **Перевірка sorry-free**: `#print axioms` для всіх чотирьох → лише
    `[propext, Classical.choice, Quot.sound]`, **без `sorryAx`**.
  - ⇒ Закрито acceptance criterion **L1-mini** (handoff §4.4).
- **Gaussian-моменти E[X²], E[X³], E[X⁴] — sorry-free** у `VWK/GaussianMoments.lean`:
  - `integral_sq_gaussianReal : ∫ x², d N(0,v) = v` (`variance_of_integral_eq_zero`
    + `variance_fun_id_gaussianReal`);
  - `integral_cube_gaussianReal : ∫ x³, d N(0,v) = 0` (симетрія `gaussianReal_map_neg`:
    `∫ x³ = ∫ (-x)³ = −∫ x³ ⇒ 0`);
  - `integral_pow_four_gaussianReal : ∫ x⁴, d N(0,v) = 3v²` (через MGF: `E[Xⁿ]=(dⁿ/dtⁿ mgf)(0)`
    `iteratedDeriv_mgf_zero`; `mgf = exp(v t²/2)`; 4 похідні обчислено явно, eval у 0 дає 3v²).
  Разом покривають **усі 6 cross-термів** ортонормованості `{ψ₀,ψ₁,ψ₂}`.
  (NB: цей файл наразі імпортує повний `Mathlib` заради надійності калькулюс-лем; звузити imports —
  cleanup-пункт перед upstream-PR.)
- **Ортонормованість `{ψ₀,ψ₁,ψ₂}` — sorry-free** у `VWK/Orthonormality.lean`:
  `orthonormal_scaledHermite (v) (σ) (hσ : 0<σ) (hσ2 : σ²=v) : ∫ ψ_i·ψ_j dN(0,v) = δ_ij` для всіх
  `i,j ≤ 2`. Збудовано на двох reusable-лемах: `integrable_pow_gaussianReal` (інтегровність
  мономіалів `xⁿ` через `memLp_id_gaussianReal`) та `integral_poly4_gaussianReal` (E[poly deg≤4]
  = `3a·v²+c·v+e` через моменти). **⇒ Повний L1-блок завершено** (reduction + ортонормованість, k≤2).
- **L1′ (Binomial→Krawtchouk) — sorry-free** у `VWK/Krawtchouk.lean` (2026-06-07): дискретний
  (finite-support) інстанс Theorem 2′ (theorem-draft §5.1). Для `Binomial(3,p)`: `mass=1`,
  `mean=3p`, `variance=3p(1−p)`, `central_three=3p(1−p)(1−2p)`, та ортогональність `{π₀,π₁,π₂}`
  (`orth_01/02/12 = 0`) — **усе через `ring`**, без measure theory (finite sum замість `∫`).
  `#print axioms` → `[propext, Classical.choice, Quot.sound]` (без `sorryAx`); `lake build VWK`
  зелений (48s). Підтверджує §10.2-claim, що finite support ⇒ найлегший нетривіальний Lean-інстанс
  (немає `gaussianReal`, інтегровності, AoC). Next: загальне-N (біноміальні суми) і Charlier/Meixner.
- **L1′ загальне-N (Binomial(N,p)→Krawtchouk, довільне N) — sorry-free** у `VWK/KrawtchoukN.lean`
  (2026-06-08): `N` — вільний параметр, тож `ring` уже не сягає сум; натомість підйом біноміальних
  моментів через Mathlib `bernsteinPolynomial.{sum, sum_smul, variance}` (тотожності в `ℝ[X]` з
  `X=p`, потім `eval p`). Доведено для всіх `N,p`: `mass=1`, `mean=Np`, `‖K₁‖²=Np(1−p)`
  (`var_N`), ортогональності `K₀⟂K₁` (`orth01`), `K₀⟂K₂` (`orth02` — без 3-го моменту: `μ₃`-член
  ×`E[t−μ]=0`), і **повна `{K₀,K₁,K₂}`** через `K₁⟂K₂` (`orth12`). Для останнього довів **третій
  факторіальний момент Бернштейна** `Σ ν(ν−1)(ν−2)•b = n(n−1)(n−2)•X³` (`sum_mul_mul_smul`) —
  лема, відсутня в Mathlib (там лише до 2-го, `sum_mul_smul`); розширив той самий MvPolynomial
  `pderiv`-патерн третьою `∂/∂x` від `(x+y)ⁿ` + helper-касти `cast_ffact2/3` (ℕ-зрізаний
  факторіал → ℝ). `central3_N`: `Σ(ν−Np)³w = Np(1−p)(1−2p)`. Усі `#print axioms` →
  `[propext, Classical.choice, Quot.sound]`; `lake build VWK` зелений, без warning'ів. **Кандидат на
  upstream-PR**: `sum_mul_mul_smul` (третій Bernstein factorial moment). Next: Charlier/Meixner
  (нескінченний носій → `tsum`, не finite sum).

## In progress / Next

- **L1-full** (arbitrary `k`): узагальнити закриту форму через `gramSchmidt` + reduction.
- **Ортонормованість `{ψ_k}` за `N(0,σ²)`** (перетворює closed forms на справжню
  ідентифікацію з GS-виходом) — наступний крок, але впирається у Blocked нижче.

## Blocked

- **Немає блокерів.** Повний L1-блок (Theorem 2 reduction `k≤2` + ортонормованість `{ψ₀,ψ₁,ψ₂}`)
  завершено sorry-free.

## Decisions (resolved 2026-05-23, С.В.) — FROZEN

Рішення по handoff §8 зафіксовано:
1. **§8.1 Кумулянти → BYPASS (moments-only).** Окремий модуль Cumulants відкладено до старту
   L4. L1–L3 ведемо без кумулянтів.
2. **§8.2 Time-domain → DISCRETE, finite-memory** як practical target для L3/L4. Continuous-time
   Volterra — поза scope (вимагає measure theory на функ. просторах, відсутню в Mathlib).
3. **§8.3 Upstream → UPSTREAM-QUALITY.** `GaussianMoments` та ортогональність Hermite за Gaussian
   розвивати чисто й загально, з наміром **PR у Mathlib** (мінімальні imports, загальні
   формулювання, doc-strings) — блокер L1 стає community-внеском.

## Mathlib audit findings (headline)

Повне — у `MATHLIB_AUDIT.md`. Головне:
- ✅ `gramSchmidt` працює **напряму** на `Lp ℝ 2 μ` (є `L2.innerProductSpace`; `ℕ` — валідний
  індекс). Uplift зі скінченновимірного **не** потрібен — це закриває open question handoff §4.5.1.
- ✅ Probabilist's `Polynomial.hermite` (ℤ[X]) + рекурсія + формула Родрігеса; обчислення в `ℝ`
  через `aeval` (не `.eval` — коригує placeholder handoff §4.2).
- ✅ L²(P) inner product, Gaussian-міра, μ та дисперсія.
- ❌ Вищі Gaussian-моменти, ортогональність Hermite за Gaussian, кумулянти `cᵣ`,
  Volterra-функціонали.

## Note до paper-side (handoff §10)

Жодного theorem statement не змінено. Єдина технічна корекція: у handoff §4.2 `scaledHermite`
використовував `(hermite k).eval (x/σ)`, але `hermite : ℕ → ℤ[X]` має цілі коефіцієнти, тож
коректна evaluation у `ℝ` — через `Polynomial.aeval`. Варто синхронізувати handoff/theorem-draft.

## Наступний конкретний крок

**Повний L1-блок (k≤2) завершено.** Опції наступного кроку:
1. **Зв'язати з Mathlib `gramSchmidt`**: довести, що `scaledHermite σ k` — це і є вихід
   `gramSchmidt ℝ (мономіали)` у `L²(N(0,σ²))` (через uniqueness GS + ортонормованість + degree).
   Це повністю замикає Theorem 2 для k≤2 як «GS-базис = scaled Hermite».
2. **L1-full arbitrary k**: узагальнити reduction на довільне k (індукція; потребує загальної
   ортогональності Hermite за Gaussian — кандидат на upstream-лему з `hermite_eq_deriv_gaussian`).
3. **L2 (Lemma 1)**: GS у `L²(P)` для general P (handoff §5).
4. **Cleanup**: звузити imports у `GaussianMoments.lean`/`Orthonormality.lean` (зараз повний Mathlib)
   перед upstream-PR (§8.3).
