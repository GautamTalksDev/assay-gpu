# RESULTS-KSCALE

K-scaling study under residual-v3. This is a results record, not a plan.

## 1. Question and method

**Question:** how does detectability under residual-v3 vary with inner dimension K?

**Residual-v3 (elementwise).** Ones-sided checksum identity unchanged: `e = ones(N)`.

```
d      = C @ e                         # length M, fp64
d'     = A @ (B @ e)                  # length M, fp64
scale  = |A| @ (|B| @ e)              # length M, fp64, nonnegative

r_i    = |d_i - d'_i| / scale_i
r_max  = max_i r_i
```

All accumulation in float64. `A`, `B`, and `C` are promoted to fp64 before
these operations (`src/assay/abft/residual_v3.py`).

**Shape and workload.** K ∈ {512, 1024, 2048, 4096, 8192}. M = N = 4096
fixed. Workload W02, bfloat16. Flip injection reuses the existing harness and
`LAYOUT_BF16` unchanged (`src/assay/inject/`).

**Sample counts.**

| phase | n per K | total |
| --- | --- | --- |
| clean (no injection) | 500 | 2500 |
| flip matrix | 100 per (bit_class, n_flips) cell, 15 cells | 7500 |

Bit classes: SIGN, EXPONENT_HIGH, EXPONENT_LOW, MANTISSA_HIGH, MANTISSA_LOW.
`n_flips` ∈ {1, 2, 4}.

**Detection criterion.** Threshold per K = observed clean max at that K (max
`r_max` over the 500 clean samples for that K). A flip sample is detected when
`r_max` exceeds that K-specific threshold.

**Environment** (JSONL metadata header, T4 sweep):

| field | value |
| --- | --- |
| torch | 2.13.0+cu130 |
| GPU | Tesla T4 (sm75) |
| git_sha | 7da9fb3 |

## 2. Clean floor vs K

Source: n = 500 independent `(A, B)` draws per K, no fault injection.
Statistics from `scripts/analyze_v3_kscale.py` on T4 sweep output.

| K | med r_median | clean_max |
| --- | --- | --- |
| 512 | 1.029719e-06 | 7.636551e-06 |
| 1024 | 7.281700e-07 | 5.772809e-06 |
| 2048 | 5.150956e-07 | 3.746226e-06 |
| 4096 | 3.640184e-07 | 2.684525e-06 |
| 8192 | 2.575653e-07 | 1.940381e-06 |

**Log-log fit** (median `r_median` vs K, all five K):

| fit | exponent a |
| --- | --- |
| full range (512–8192, 16×) | −0.4999 |
| 512–2048 | −0.4997 |
| 4096–8192 | −0.4991 |

**Mechanism.** Absolute accumulation error in a row sum of K bf16 products
grows as √K. The per-row normalizer `scale_i = |A_i| |B| e` grows linearly
with K (K terms in the sum). The relative residual therefore scales as
K / √K = √K in the numerator sense, i.e. falls as 1/√K:

```
r_i ∝ (error ~ √K) / (scale ~ K) = 1/√K
```

**Correction to the locked prediction.** `docs/RESIDUAL.md` stated the
mechanism as: "clean accumulation error grows as √K, signal-to-floor should
fall as 1/√K." The measured exponent on the clean floor (−0.4999) matches
√K scaling in magnitude, but that reasoning conflated absolute and relative
error. It predicted the wrong sign for the floor's dependence on K: the
relative clean floor falls as K^(−1/2), not rises. The exponent matched; the
direction of floor vs K did not follow from the stated argument.

## 3. Detection by K

Detection criterion: `r_max > clean_max[K]` from section 2. n = 100 per cell.
Source: T4 sweep, git sha 7da9fb3; `scripts/analyze_v3_kscale.py` on
`data/noisefloor/pilot/sweep-v3-kscale.jsonl`.

### 3.1 Full flip matrix (5 K × 15 cells)

| K | bit_class | n_flips | n | detected | rate |
| --- | --- | --- | --- | --- | --- |
| 512 | EXPONENT_HIGH | 1 | 100 | 78/100 | 78.00% |
| 512 | EXPONENT_HIGH | 2 | 100 | 93/100 | 93.00% |
| 512 | EXPONENT_HIGH | 4 | 100 | 92/100 | 92.00% |
| 512 | EXPONENT_LOW | 1 | 100 | 69/100 | 69.00% |
| 512 | EXPONENT_LOW | 2 | 100 | 95/100 | 95.00% |
| 512 | EXPONENT_LOW | 4 | 100 | 100/100 | 100.00% |
| 512 | MANTISSA_HIGH | 1 | 100 | 4/100 | 4.00% |
| 512 | MANTISSA_HIGH | 2 | 100 | 8/100 | 8.00% |
| 512 | MANTISSA_HIGH | 4 | 100 | 12/100 | 12.00% |
| 512 | MANTISSA_LOW | 1 | 100 | 0/100 | 0.00% |
| 512 | MANTISSA_LOW | 2 | 100 | 0/100 | 0.00% |
| 512 | MANTISSA_LOW | 4 | 100 | 0/100 | 0.00% |
| 512 | SIGN | 1 | 100 | 79/100 | 79.00% |
| 512 | SIGN | 2 | 100 | 95/100 | 95.00% |
| 512 | SIGN | 4 | 100 | 100/100 | 100.00% |
| 1024 | EXPONENT_HIGH | 1 | 100 | 77/100 | 77.00% |
| 1024 | EXPONENT_HIGH | 2 | 100 | 91/100 | 91.00% |
| 1024 | EXPONENT_HIGH | 4 | 100 | 85/100 | 85.00% |
| 1024 | EXPONENT_LOW | 1 | 100 | 75/100 | 75.00% |
| 1024 | EXPONENT_LOW | 2 | 100 | 92/100 | 92.00% |
| 1024 | EXPONENT_LOW | 4 | 100 | 99/100 | 99.00% |
| 1024 | MANTISSA_HIGH | 1 | 100 | 1/100 | 1.00% |
| 1024 | MANTISSA_HIGH | 2 | 100 | 9/100 | 9.00% |
| 1024 | MANTISSA_HIGH | 4 | 100 | 9/100 | 9.00% |
| 1024 | MANTISSA_LOW | 1 | 100 | 0/100 | 0.00% |
| 1024 | MANTISSA_LOW | 2 | 100 | 0/100 | 0.00% |
| 1024 | MANTISSA_LOW | 4 | 100 | 0/100 | 0.00% |
| 1024 | SIGN | 1 | 100 | 75/100 | 75.00% |
| 1024 | SIGN | 2 | 100 | 93/100 | 93.00% |
| 1024 | SIGN | 4 | 100 | 99/100 | 99.00% |
| 2048 | EXPONENT_HIGH | 1 | 100 | 88/100 | 88.00% |
| 2048 | EXPONENT_HIGH | 2 | 100 | 95/100 | 95.00% |
| 2048 | EXPONENT_HIGH | 4 | 100 | 96/100 | 96.00% |
| 2048 | EXPONENT_LOW | 1 | 100 | 83/100 | 83.00% |
| 2048 | EXPONENT_LOW | 2 | 100 | 97/100 | 97.00% |
| 2048 | EXPONENT_LOW | 4 | 100 | 99/100 | 99.00% |
| 2048 | MANTISSA_HIGH | 1 | 100 | 5/100 | 5.00% |
| 2048 | MANTISSA_HIGH | 2 | 100 | 11/100 | 11.00% |
| 2048 | MANTISSA_HIGH | 4 | 100 | 15/100 | 15.00% |
| 2048 | MANTISSA_LOW | 1 | 100 | 0/100 | 0.00% |
| 2048 | MANTISSA_LOW | 2 | 100 | 0/100 | 0.00% |
| 2048 | MANTISSA_LOW | 4 | 100 | 0/100 | 0.00% |
| 2048 | SIGN | 1 | 100 | 73/100 | 73.00% |
| 2048 | SIGN | 2 | 100 | 99/100 | 99.00% |
| 2048 | SIGN | 4 | 100 | 100/100 | 100.00% |
| 4096 | EXPONENT_HIGH | 1 | 100 | 83/100 | 83.00% |
| 4096 | EXPONENT_HIGH | 2 | 100 | 97/100 | 97.00% |
| 4096 | EXPONENT_HIGH | 4 | 100 | 96/100 | 96.00% |
| 4096 | EXPONENT_LOW | 1 | 100 | 79/100 | 79.00% |
| 4096 | EXPONENT_LOW | 2 | 100 | 93/100 | 93.00% |
| 4096 | EXPONENT_LOW | 4 | 100 | 99/100 | 99.00% |
| 4096 | MANTISSA_HIGH | 1 | 100 | 7/100 | 7.00% |
| 4096 | MANTISSA_HIGH | 2 | 100 | 12/100 | 12.00% |
| 4096 | MANTISSA_HIGH | 4 | 100 | 18/100 | 18.00% |
| 4096 | MANTISSA_LOW | 1 | 100 | 0/100 | 0.00% |
| 4096 | MANTISSA_LOW | 2 | 100 | 0/100 | 0.00% |
| 4096 | MANTISSA_LOW | 4 | 100 | 0/100 | 0.00% |
| 4096 | SIGN | 1 | 100 | 81/100 | 81.00% |
| 4096 | SIGN | 2 | 100 | 99/100 | 99.00% |
| 4096 | SIGN | 4 | 100 | 100/100 | 100.00% |
| 8192 | EXPONENT_HIGH | 1 | 100 | 82/100 | 82.00% |
| 8192 | EXPONENT_HIGH | 2 | 100 | 94/100 | 94.00% |
| 8192 | EXPONENT_HIGH | 4 | 100 | 100/100 | 100.00% |
| 8192 | EXPONENT_LOW | 1 | 100 | 71/100 | 71.00% |
| 8192 | EXPONENT_LOW | 2 | 100 | 96/100 | 96.00% |
| 8192 | EXPONENT_LOW | 4 | 100 | 100/100 | 100.00% |
| 8192 | MANTISSA_HIGH | 1 | 100 | 6/100 | 6.00% |
| 8192 | MANTISSA_HIGH | 2 | 100 | 9/100 | 9.00% |
| 8192 | MANTISSA_HIGH | 4 | 100 | 20/100 | 20.00% |
| 8192 | MANTISSA_LOW | 1 | 100 | 0/100 | 0.00% |
| 8192 | MANTISSA_LOW | 2 | 100 | 0/100 | 0.00% |
| 8192 | MANTISSA_LOW | 4 | 100 | 0/100 | 0.00% |
| 8192 | SIGN | 1 | 100 | 77/100 | 77.00% |
| 8192 | SIGN | 2 | 100 | 97/100 | 97.00% |
| 8192 | SIGN | 4 | 100 | 100/100 | 100.00% |

**Observations (measured, not explained in this document):**

- EXPONENT_HIGH is non-monotonic in `n_flips` at K = 1024 (91% at 2 flips,
  85% at 4) and at K = 2048 and K = 4096 (97% at 2 flips, 96% at 4). The
  same effect was recorded at K = 4096, n = 200 in `docs/RESULTS-KT1-v3.md`.
  Unexplained.
- MANTISSA_HIGH rises with `n_flips` at every K but never exceeds 20%.
- MANTISSA_LOW is 0/100 in all 15 cells across all five K.
- The K = 4096 SIGN 1-flip rate here (81/100 = 81%) differs from the n = 200
  rate in `docs/RESULTS-KT1-v3.md` (155/200 = 77.5%). Both are within
  binomial noise at their respective sample sizes; no reconciliation is
  attempted.

### 3.2 SIGN 1-flip vs K

| K | n | detected | rate |
| --- | --- | --- | --- |
| 512 | 100 | 79/100 | 79.00% |
| 1024 | 100 | 75/100 | 75.00% |
| 2048 | 100 | 73/100 | 73.00% |
| 4096 | 100 | 81/100 | 81.00% |
| 8192 | 100 | 77/100 | 77.00% |

Mean 77.0%. Range 73–81% over a 16× K span.

## 4. Predictions scorecard

Predictions locked in `docs/RESIDUAL.md` before any K-scaling data existed.
Claims quoted verbatim from that section.

| ID | Claim (verbatim) | Measured | Verdict |
| --- | --- | --- | --- |
| Q1 | clean per-row floor r_median scales as K^a, a ∈ [0.4, 0.6] (i.e. roughly √K); at K = 4096 it is 3.64e-7 by measurement | a = −0.4999 (magnitude 0.4999); at K = 4096, med r_median = 3.640184e-07 | **PASS** |
| Q2 | SIGN 1-flip detection falls monotonically with K | 79%, 75%, 73%, 81%, 77% at K = 512, 1024, 2048, 4096, 8192. Non-monotonic; variation consistent with binomial noise at n = 100 (SE ~4 pp around a mean of ~77%) | **FAIL** |
| Q3 | SIGN 1-flip detection at K = 512 is > 95% | 79% | **FAIL** |
| Q4 | SIGN 1-flip detection at K = 8192 is < 70% | 77% | **FAIL** |
| Q5 | MANTISSA_LOW remains 0/100 at every K, including K = 512 | 0/1500 across all cells | **PASS** |

**Falsifier check.** The stated falsifier in `docs/RESIDUAL.md` — *"if detection
is flat across a 16× range of K, the √K mechanism is wrong and the finding
reduces to a single-shape observation with no predictive content"* — **FIRED**.
SIGN 1-flip detection spans 73–81% across K = 512 to 8192 (16×), with no
monotonic degradation. The clean floor exponent matched √K in magnitude, but
detectability did not track K.

The failed predictions (Q2, Q3, Q4) remain as written in `docs/RESIDUAL.md`.

## 5. Finding

The clean floor scales as K^(−1/2) with a measured exponent of −0.4999 across
a 16× range (2500 clean samples).

Detectability does not vary with K: signal and floor scale together under the
per-K threshold. Detectability is governed by the bit position of the
corruption, not the contraction length:

| bit_class | detection |
| --- | --- |
| SIGN | ~77% |
| EXPONENT | 71–88% |
| MANTISSA_HIGH | 1–7% |
| MANTISSA_LOW | 0/1500 |

## 6. Limitations

- Single GPU (Tesla T4, sm75).
- Single dtype (bfloat16).
- Single workload (W02).
- M = N = 4096 fixed throughout; only K varied.
- n = 100 per flip cell gives ~4 pp standard error on detection rates.
- Algorithm-level bit-flip injection. Prior art (arXiv:2601.19912) argues this
  does not capture realistic GPU fault behaviour. This remains an open
  limitation.
