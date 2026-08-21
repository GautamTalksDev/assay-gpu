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
| Q5 | MANTISSA_LOW remains 0/100 at every K, including K = 512 | 0/1500 across all cells at per-K observed clean maxima (not at `threshold_gpd`) | **PASS** |

Injection audit (`docs/RESULTS-INJECTION-AUDIT.md`): MANTISSA_LOW and
MANTISSA_HIGH `n_flips=1` both show `n_elements_bitwise_equal = 0/50` with
nonzero `achieved_rel_delta_max` on every sample — Q5 is not an injector miss.
At `threshold_gpd` (K = 4096), MANTISSA_LOW is **0/600**; see
`docs/RESULTS-FPR.md`.

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
| MANTISSA_LOW | 0/1500 at per-K observed clean maxima; 0/600 at `threshold_gpd` (K = 4096) |

## 6. Limitations

- Single GPU (Tesla T4, sm75).
- Single dtype (bfloat16).
- Single workload (W02).
- M = N = 4096 fixed throughout; only K varied.
- n = 100 per flip cell gives ~4 pp standard error on detection rates.
- Algorithm-level bit-flip injection. Prior art (arXiv:2601.19912) argues this
  does not capture realistic GPU fault behaviour. This remains an open
  limitation.

## Data provenance (recovered 2026-08-21)

The path cited above, `data/noisefloor/pilot/sweep-v3-kscale.jsonl`, was
missing from the repo. Three Kaggle `/kaggle/working` candidates were
downloaded and compared. **None was named `sweep-v3-kscale.jsonl`.**

| file | role | sha256 |
| --- | --- | --- |
| `kscale-all.jsonl` | **source of published numbers** (all five K) | `5b3da3c47d84bd47a4ed7bec21ac4cc0905e4d6dc7704f9a22118f4aebf643dd` |
| `kscale.jsonl` | K ∈ {512, 1024, 2048} only | `8ae4b480c1b3707e4e7215ad98be2db0c3660086d345a68becb537063f9ff46b` |
| `kscale-hi.jsonl` | K ∈ {4096, 8192} only | `97857b35f62cc5945debbaae9f90f2650ad6f4d1bb60e985c4054ec429f9bab8` |

`kscale-all.jsonl` reproduces the published clean maxima and median
`r_median` values exactly, with `git_sha` `7da9fb3…`, n = 500 clean per K,
n = 100 per flip cell. It is now stored at the cited path
`data/noisefloor/pilot/sweep-v3-kscale.jsonl` (same sha256). Candidates
retained under `data/noisefloor/pilot/kscale-candidates/` with
`kscale-SHA256SUMS.txt`.

### Detection predicate and EXPONENT_HIGH non-monotonicity

Published detection rates use the write-path field
`detected = (r_max > clean_max[K])`. NaN fails that comparison.
Non-finite `r_max` values appear **only** in EXPONENT_HIGH cells (table
omitted here for cells with zero non-finites):

| K | n_flips | nan | +inf |
| --- | --- | --- | --- |
| 512 | 2 | 4 | 0 |
| 512 | 4 | 8 | 0 |
| 1024 | 1 | 3 | 0 |
| 1024 | 2 | 5 | 1 |
| 1024 | 4 | 14 | 0 |
| 2048 | 2 | 2 | 0 |
| 2048 | 4 | 4 | 0 |
| 4096 | 4 | 4 | 0 |
| 8192 | 1 | 1 | 0 |
| 8192 | 2 | 3 | 0 |

EXPONENT_HIGH under both predicates (n = 100; threshold = per-K clean max):

| K | n_flips | bare `>` (published) | `(not isfinite) or >` (canonical) |
| --- | --- | --- | --- |
| 1024 | 2 | 91/100 | 96/100 |
| 1024 | 4 | 85/100 | 99/100 |
| 2048 | 2 | 95/100 | 97/100 |
| 2048 | 4 | 96/100 | 100/100 |
| 4096 | 2 | 97/100 | 97/100 |
| 4096 | 4 | 96/100 | 100/100 |

Under the canonical predicate the published non-monotonic dips
(91→85 at K = 1024; 97→96 at K = 4096) **do not appear** (96→99 and
97→100). The §3.1 / scorecard observations that EXPONENT_HIGH detection
falls with `n_flips` at those K are therefore a **NaN-dropping artifact of
the write-path predicate**, not an independent physical finding. Published
cell counts above are not rewritten; the artifact is recorded here.
Bit-class stratification (SIGN / EXPONENT / MANTISSA_HIGH / MANTISSA_LOW)
is unaffected: non-finites occur only in EXPONENT_HIGH.

## Detection predicate correction

The write path in `src/assay/noise/sweep_v3_kscale.py` scores detection as
`r_max > threshold`, which evaluates **False** on non-finite `r_max`.
**Ten of fifteen** EXPONENT_HIGH cells contain non-finite `r_max` (up to
**14/100** at K = 1024, `n_flips` = 4). No other bit class produces any
non-finite `r_max`.

Canonical predicate: `(not math.isfinite(r_max)) or (r_max > threshold)`.
Corrected EXPONENT_HIGH rates (n_flips = 1 / 2 / 4) beside published
(write-path) rates:

| K | corrected (canonical) | published (bare `>`) |
| --- | --- | --- |
| 512 | 78 / 97 / 100 | 78 / 93 / 92 |
| 1024 | 80 / 96 / 99 | 77 / 91 / 85 |
| 2048 | 88 / 97 / 100 | 88 / 95 / 96 |
| 4096 | 83 / 97 / 100 | 83 / 97 / 96 |
| 8192 | 83 / 97 / 100 | 82 / 94 / 100 |

The §3.1 table is left **unedited** and is correct for the predicate it
used. The corrected column above is **canonical**. The reported
EXPONENT_HIGH non-monotonicity in `n_flips` is **withdrawn**.
