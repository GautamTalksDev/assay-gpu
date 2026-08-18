# GEMM ABFT residual definition

This file is the residual specification. Cutoffs are not here. Cutoffs
come from `data/noisefloor/` under `docs/SPEC-NOISEFLOOR.md`. Kill-test
bars are not here. They stay in `README.md`.

The ones-sided checksum identity is unchanged:

```
e = ones(N)
(C @ e)   versus   (A @ (B @ e))
```

What changed is the number we divide by.

**The residual used by `check_gemm` and by noisefloor lookup is
`residual-v2`.** Every noisefloor sample and every `CheckResult` must
record that version string. `residual-v1` data is void. It must not be
looked up, pooled, or used as a threshold. A valid KT-1 evaluation
requires `residual-v2` measurements. This file does not evaluate KT-1.

`methodology-v1.json` is unchanged (`target_quantile`, `min_samples`).
Changing the residual statistic does not change how many samples a
quantile needs.

Code implementing v2 follows this argument. The PREDICTIONS section was
written before any residual-v2 sample existed.

## residual-v1 (defective, void)

Let `ce = C @ e` and `abe = A @ (B @ e)`, computed in the GEMM's native
dtype. Promote both vectors to float64 and sum with `numpy.sum`. Then

```
abs_residual = |sum(ce) - sum(abe)|
scale_v1     = max(|sum(ce)|, |sum(abe)|)
residual-v1  = 0                         if scale_v1 == +0 and abs_residual == 0
             = inf                       if scale_v1 == +0 and abs_residual != 0
             = abs_residual / scale_v1   otherwise
```

`scale_v1` is the max absolute **grand sum** `max(|1ᵀ C 1|, |1ᵀ A (B 1)|)`.
That is what `normalized_checksum_residual` in `src/assay/abft/gemm.py`
does. `check_gemm` used only the third value.

### Why the grand sum is the wrong normalizer

`abs_residual` is a linear functional of the error matrix
`E = C - A B` (plus rounding on the two checksum paths):

```
abs_residual ≈ |1ᵀ E 1|
```

Rounding error in a GEMM scales with the **magnitudes** of the summands,
not with the signed total of `C`. For `uniform_-1_1` factors, `1ᵀ C 1`
is a sum of `M·N` terms of random sign. Its typical size is far smaller
than `1ᵀ |C| 1`, and it can land near zero by cancellation on draws
where `|E|` is ordinary.

Dividing by that signed total mixes two unrelated quantities: an error
that tracks the size of the multiply-adds, and a denominator that tracks
how completely the random signs cancelled. The ratio is then heavy-tailed
even when both pieces are light-tailed.

This is not a property of the GPU. It is the normalizer.

### Measured evidence

Two runs. Same residual function. Different shapes. Same shape of tail.

**Tesla T4, W02 bfloat16, 4096³, n = 2000 independent `(A, B)`**
(reported `n = 4000` because `cublas` and `cublaslt` each recorded the
same `sample_index`; quantiles of a duplicated list are unchanged).
Only `residual-v1` was stored. Abs residual and `scale_v1` were not.

| statistic | residual-v1 |
| --- | --- |
| median | 0.00315 |
| p99 | 0.153 |
| p99.9 | 0.961 |
| max | 1.225 |
| max / median | 389 |

`running_p99` was flat from 100 to 2000. More samples of the same
statistic will not thin this tail.

**CPU bfloat16, 128³, n = 2000, W02 seed mixer** (`workload_id = 2`,
`case_index = 3`, `sample_index = 0..1999`). Same `vector_residual_parts`
as the detector. This is the run that stored abs residual and `scale_v1`.
It is not a T4 4096 characterization.

| quantity | median | max | max / median |
| --- | --- | --- | --- |
| abs residual | 1.020 | 6.113 | **6.00** |
| `scale_v1` | 323.8 | 1943 | **6.00** |
| residual-v1 | 0.00334 | 1.099 | **329** |

Two light-tailed pieces, one heavy-tailed quotient.

| link | value |
| --- | --- |
| Pearson(residual-v1, `scale_v1`) | −0.249 |
| Pearson(residual-v1, `1 / scale_v1`) | **0.794** |
| `n` with `scale_v1 == 0` | 0 |
| median `scale_v1` | 323.8 |
| median `scale_v1` among samples with residual-v1 ≥ p99 (21 samples) | 5.90 (**55×** smaller) |
| argmax residual-v1 | abs 2.17 (ordinary), scale 1.98 (tiny) |

The p99 tail of residual-v1 is the left tail of the grand sum, not a
hardware outlier. An evaluation of KT-1 against residual-v1 was
measuring that artifact.

## residual-v2 (proposed)

Numerator unchanged. Denominator replaced by the ones-sided **absolute**
checksum of the factors, accumulated in float64:

```
e          = ones(N)
scale_v2   = sum( |A| @ (|B| @ e) )     # entrywise abs; fp64 accumulation
residual-v2 = 0                         if scale_v2 == +0 and abs_residual == 0
            = inf                       if scale_v2 == +0 and abs_residual != 0
            = abs_residual / scale_v2   otherwise
```

`|A|` and `|B|` are entrywise. `|B| @ e` and `|A| @ (|B| @ e)` use the
same ones-sided pattern as the detector, on nonnegative matrices, with
the reduction promoted to float64 before the final sum the same way
`sum(ce)` is. Native-dtype accumulation of `scale_v2` is rejected:
bfloat16 cannot represent a 4096³ absolute checksum to more than a few
bits, and the point of the normalizer is a stable magnitude, not another
quantized GEMV.

`scale_v2 = eᵀ |A| |B| e = ∑_{i,k,j} |A_{ik}| |B_{kj}|`. Every term is
nonnegative. Random signs cannot drive it toward zero. For
`uniform_-1_1` it concentrates around `M·K·N / 4`.

Zero scale remains IEEE, not a cutoff. For the suite generator it occurs
only if a factor is all zeros.

### Why this bounds the checksum identity

Write `C = A B + E` in exact arithmetic on the stored factors. The
ones-sided check, if the two checksum paths were exact, would see

```
1ᵀ C e − 1ᵀ A (B e) = 1ᵀ E 1
```

so `abs_residual = |1ᵀ E 1| ≤ 1ᵀ |E| 1`.

For classical matrix multiplication in unit roundoff `u`, the
componentwise bound is `|E| ≤ γ_K |A| |B|` (Higham, *Accuracy and
Stability of Numerical Algorithms*; `γ_K = K u / (1 − K u)` when
`K u < 1`). Contracting with `e` on both sides:

```
|1ᵀ E 1| ≤ γ_K  eᵀ |A| |B| e  = γ_K · scale_v2
```

So `abs_residual / scale_v2` is the quantity the componentwise theory
actually controls. `γ_K` is **not** a detection threshold. At bfloat16
with `K = 4096`, `K u ≳ 1` and `γ_K` is not a useful number. Tensor-core
paths accumulate in a wider type and then store, so the constant in
front of `|A| |B|` is not the classical `γ_K` either. The structural
claim is the one we need: the checksum functional is bounded by a
nonnegative contraction of the factors, and that contraction is
`scale_v2`. The numerical cutoff still comes from measured
`residual-v2` on presumed-correct silicon.

Checksum-path rounding (`C @ e` and `A @ (B @ e)` in native dtype, then
an fp64 sum) adds a second error of the same family: it scales with
magnitudes along those GEMVs, not with the signed grand sum. It is
absorbed into the measured noisefloor. It is not absorbed by guessing
a `γ`.

`scale_v2` does not depend on `C`. A fault in the product cannot shrink
the denominator.

Algebraically `eᵀ |A| |B| e = (|A|ᵀ e) · (|B| e)`: column sums of `|A|`
dotted with row sums of `|B|`. That is two reductions and a dot, not a
pair of GEMVs, but it still reads every entry of `A` and `B`. FLOPs are
`O(M K + K N)`. Wall-clock versus the GEMM is a measurement, not a
flop-count argument, because the `|A|` pass is memory traffic.

## PREDICTIONS (written before any residual-v2 data)

These numbers are locked before `check_gemm` computes `scale_v2`. They
are not detection thresholds. If a later measurement disagrees, the
disagreement is the result. Do not replace `scale_v2` to chase them.

Target: W02, bfloat16, 4096³, n = 2000 independent `uniform_-1_1`
draws. Numerator is still `|sum(C e) − sum(A(B e))|`. Denominator is
`scale_v2 ≈ M K N / 4 = 17_179_869_184` with relative scatter
`O(1/√(M K N)) ≈ 4×10⁻⁶`, i.e. constant for this purpose.

**Predicted median of residual-v2: `1.3 × 10⁻⁸` (range `5 × 10⁻⁹` to
`2 × 10⁻⁷`).**

v1's median 0.00315 was `abs / scale_v1`. `scale_v1` is a cancelled
grand sum of typical size `∼ N √K / 3 ≈ 9×10⁴`. `scale_v2` is
`∼ M K N / 4 ≈ 1.7×10¹⁰`, about `2×10⁵` times larger, so the median
must drop by that factor if the numerator is unchanged.

The numerator is not a constant. Modeling store-to-bf16 as the leading
term, `std(C_{ij}) = √K / 3` and unit roundoff `u = 2⁻⁸` give
`std(1ᵀ E 1) ≈ N · u · √K / 3 = 341`. Median `|Gaussian|` is then
`∼ 230`. `230 / 1.718×10¹⁰ ≈ 1.3×10⁻⁸`.

The range is the checksum-path GEMVs (native bf16, not the GEMM) and
Tensor-Core accumulation, neither of which is in the 128³ CPU abs
series. They can move `abs` by a small integer factor. They cannot
restore a `10⁻³` median unless `scale_v2` is wrong.

**Predicted `max_over_median` of residual-v2: `6` (range `4` to `15`).**

`scale_v2` does not have a left tail. For n = 2000, `max/median` of the
ratio is `max/median` of the numerator. The 128³ CPU abs residual had
`max/median = 6.00`, which is the folded-Gaussian extreme-value ratio
(`max |N| ∼ 3.5σ`, median `0.67σ`). Tensor-Core blocking can correlate
rounding and fatten that a little. It does not produce the v1 ratio
389, which required a near-zero denominator. A value still in the
hundreds means the prediction is wrong.

**Qualitative claim: Pearson(residual-v2, `1/scale_v2`) is near zero
(`|r| < 0.2`), not 0.794.**

v1's 0.794 was the map `normalized ≈ c / scale_v1` on a wildly varying
`scale_v1`. `scale_v2` barely moves, so that correlation has nothing to
attach to. Residual-v2's tail, if any, is numerator cancellation in
`C e` versus `A(B e)`, not the normalizer. Pearson(residual-v2,
`scale_v2`) is also near zero; a mild negative would mean larger
factors slightly over-normalize, which is allowed.

## Candidates rejected

### Keep `scale_v1`, or `scale_v1` plus a floor

The measurements above are the rejection. Adding a guessed floor, or a
floor fitted from the same grand-sum samples, is still a rule about the
defective denominator. Forbidden as a guessed cutoff, and it does not
fix the mismatch with `|E|`.

### Unnormalized `abs_residual`

The noisefloor key is already per `(workload, dtype, shape)`, so a raw
absolute residual could be characterized. It is not invariant to input
magnitude. The suite is `uniform_-1_1` today; a later distribution, or
a caller that scales `A` and `B`, would move the statistic with no
fault. `scale_v2` tracks that magnitude. Abs residual stays a recorded
diagnostic. It is not the decision residual.

### `‖C e‖₁ = ∑_i |(C @ e)_i|`

No extra kernels: `ce` is already in hand. The left tail of a sum of
absolute row-sums is lighter than the left tail of the grand sum, so
this would probably kill the 389× blowup on `uniform_-1_1`.

It is still the wrong object. Row `i` of `C` can cancel in `∑_j C_{ij}`
even when the GEMM error in that row scales with `(|A| |B| e)_i`. The
denominator still uses `C`, so an exponent flip that inflates `|C|`
enlarges the scale and can hide the same fault the numerator is trying
to show. Rejected.

### `∑_{i,j} |C_{ij}| = eᵀ |C| e`

Nonnegative in `C`'s entries, so no signed grand-sum cancellation.
Still a function of `C`. Inner-product cancellation already makes
`|C_{ij}|` typically much smaller than `(|A| |B|)_{ij}` for
`uniform_-1_1`; that understates the a priori GEMM bound. A fault that
grows an entry grows the denominator. Rejected.

### Frobenius / spectral product `‖A‖ ‖B‖`

`|1ᵀ E 1| ≤ √(M N) ‖E‖_F`, and a normwise GEMM bound gives `‖E‖_F`
in terms of `‖A‖_F ‖B‖_F` (or the 2-norm, which needs an SVD). That
is a legal bound and it does not cancel.

It is the bound for a different functional. The detector contracts with
`e`, not with the Frobenius inner product. For rectangular `(M, K, N)`
the two scales differ by powers of the three dimensions; the noisefloor
is per-shape so that is survivable, but it is slack we do not need.
Spectral norm is not a checksum-path primitive. Rejected in favor of
the matching nonnegative contraction `eᵀ |A| |B| e`.

### `max(scale_v2, eᵀ |C| e)`

Puts `C` back in the denominator "for the extra matvec error." That
error is already in the measured numerator. The max reintroduces
fault-dependent scale. Rejected.

## What v2 does not change

- Ones-sided identity, `e = ones`. Weighted checksums are a different
  detector (`docs/ABFT.md`).
- `noisefloor-v1` methodology: same quantile, same `min_samples`, same
  INCONCLUSIVE-when-uncharacterized rule, same refusal to invent `atol`.
- KT-1 bars. This file does not pass, fail, or retune them.
- KT-2 bars. Whether `assay watch` can afford `scale_v2` is a measured
  cost, recorded below. It does not change the residual definition.

## Void data and mix-up rule

Any `run-*.json`, pilot JSON, or `CheckResult` produced under
`scale_v1` is **residual-v1** and is void as a detection threshold.
That includes the Tesla T4 W02 4096³ n = 2000 pilot: its normalized
distribution is the grand-sum artifact. It may be kept as evidence
that v1 is defective. It must not enter lookup.

Implemented:

- Every noisefloor sample records `residual_version: "residual-v2"`.
- Every `CheckResult` records the same field.
- Lookup ignores files that lack that version or that say `residual-v1`.
- Pilot JSON records `residual_version` at the top level.

`check_gemm` computes residual-v2. Lookup refuses residual-v1. The
PREDICTIONS section above was written before any residual-v2 noisefloor
existed.

## Measured cost of `eᵀ |A| |B| e`

CPU float32 ones-GEMM numbers for this workstation are in
`docs/ABFT.md` (development box, no NVIDIA GPU). Tesla T4 bfloat16
residual-v2 ratios, repeats = 8, are in the same file. Combined
checksum + normalizer is below 10% of GEMM only at 4096³ on that T4
(7.8%). That is not a KT-2 verdict. Residual-v2 remains the
characterization statistic either way.

## Pilot diagnostics (already in tree)

The duplicate-BLAS loop in `run_abft_pilot` is fixed: one
`preferred_blas_library`, `n == n_samples`, `sample_index` still unique
per draw.

The pilot still emits, for the same samples:

- `distribution` — residual-v2
- `distribution_abs` — unnormalized `abs_residual`
- `distribution_normalizer` — `scale_v2 = eᵀ |A| |B| e`
- `normalizer_link` — Pearson against scale and `1/scale`

Those fields are how the PREDICTIONS section is tested. They are not
a third residual.
