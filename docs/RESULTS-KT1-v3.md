# RESULTS-KT1-v3

**STATUS: FAIL.** First valid KT-1 evaluation under residual-v3.

The bars are the same numbers as `README.md` at CP-0. They were not
moved. The detector was not retuned. No threshold was adjusted for
deployment. The detection criterion below uses the observed clean max
only as a measurement reference, not a derived FPR threshold.

Prior void evaluations: `docs/RESULTS-KT1-v2.md` (scalar reduction),
`docs/RESULTS-KT1.md` (null evaluation).

## 1. Method

### Residual-v3 (elementwise)

Ones-sided checksum identity unchanged: `e = ones(N)`.

```
d      = C @ e                         # length M, fp64
d'     = A @ (B @ e)                  # length M, fp64
scale  = |A| @ (|B| @ e)              # length M, fp64, nonnegative

r_i    = |d_i - d'_i| / scale_i
r_max  = max_i r_i
```

All accumulation in float64. `A`, `B`, and `C` are promoted to fp64
before these operations (`src/assay/abft/residual_v3.py`).

**Contrast with v2.** residual-v2 (`docs/RESIDUAL.md`) computed a single
scalar:

```
|sum(C @ e) - sum(A @ (B @ e))| / (eᵀ |A| |B| e)
```

That sums away M−1 degrees of freedom before comparing. Huang–Abraham
ABFT compares elementwise. v3 was introduced because v2's published
"structural invisibility" of SIGN and MANTISSA classes
(`docs/RESULTS-KT1-v2.md`, now void) may have been an artifact of the
reduction rather than a property of the ones-vector checksum.

### Environment

Recorded in JSONL metadata headers (`record_type: metadata`) on both
sweeps:

| field | value |
| --- | --- |
| torch | 2.13.0+cu130 |
| GPU | Tesla T4 (sm75) |
| git_sha | recorded in metadata header at measurement time |
| residual_version | v3 |

Clean sweep: `data/noisefloor/pilot/sweep-v3.jsonl`, n = 2000.
Flip sweep: `data/noisefloor/pilot/sweep-v3-flips.jsonl`, n = 200 per
cell.

### Workload and injection

- Workload W02, bfloat16, shape 4096×4096×4096.
- Factors: `gemm_numpy_pair`, case_index = 3, workload_id = 2,
  `sample_index` in 0..199 (same mixer as the v2 flip matrix).
- One GEMM per `sample_index`; every `(bit_class, n_flips)` cell flips
  that `C` after `torch.matmul` (`src/assay/inject/`, `LAYOUT_BF16`
  unchanged).
- Bit classes: SIGN, EXPONENT_HIGH, EXPONENT_LOW, MANTISSA_HIGH,
  MANTISSA_LOW. `n_flips` ∈ {1, 2, 4}.

## 2. Clean noise floor

Source: n = 2000 independent `(A, B)` draws, no fault injection.
Statistics from `scripts/analyze_v3.py` on `sweep-v3.jsonl`.

### r_median across samples

| statistic | value |
| --- | --- |
| median | 3.639779e-07 |
| p90 | 3.724246e-07 |
| p99 | 3.790080e-07 |
| p99.9 | 3.855132e-07 |
| max | 3.858686e-07 |

### r_max across samples

| statistic | value |
| --- | --- |
| median | 2.031933e-06 |
| max | 3.245921e-06 |

### Tail shape

```
max(r_max) / median(r_median) = 8.92
```

### Row-to-row correlation

From the 2000 × 256 matrix of per-row residuals (`r_rows` subsample):

| statistic | value |
| --- | --- |
| mean pairwise Pearson | 0.000126 |
| std | 0.022511 |
| range | −0.085284 to 0.098041 |

**What this rules out:** rows are effectively independent. The
per-row floor is not inflated by shared accumulation error across rows
(shared kernel tiling or shared reduction order would have produced
materially positive mean correlation). The SNR gain from elementwise
comparison is real, not a correlated-noise artifact.

## 3. Locked predictions scorecard

Predictions locked in `docs/RESIDUAL.md` before any v3 data existed.
Measured on T4, W02 bf16 4096³.

| ID | Claim (verbatim) | Measured | Verdict |
| --- | --- | --- | --- |
| P1 | clean per-row relative floor, median: 2e-6 to 8e-6 | 3.639779e-07 | **MISS** (below range; shape correct) |
| P2 | clean per-row max/median: 4 to 15 | 8.92 | **PASS** |
| P3 | SIGN 1-flip, median ratio to floor: 2 to 10 | 2.298 | **PASS** |
| P4 | MANTISSA_LOW 1-flip, median ratio to floor: still < 1 | 0/200 detected, max ratio 0.827 | **PASS** |
| P5 | EXPONENT_LOW 1-flip detection rate: > 70% | 136/200 = **68%** | **MISS** |

P5 is recorded as a miss. 136/200 = 68%, not 70%. Do not round up.

**Falsifier check:** P1 measured 3.64e-7, not near 5e-8 (v2 global
floor). The elementwise hypothesis is not falsified on SNR grounds.

## 4. Flip matrix

Detection criterion for this table: `r_max > 3.245921e-06`, the
observed clean max at n = 2000. Ratio = `r_max / 3.245921e-06`.
`n(ratio>1)` counts samples with ratio > 1 or non-finite.

Source: `scripts/analyze_v3_flips.py` on `sweep-v3-flips.jsonl`, T4.
n = 200 per cell.

| bit_class | n_flips | n | min ratio | median ratio | max ratio | n(ratio>1) |
| --- | --- | --- | --- | --- | --- | --- |
| EXPONENT_HIGH | 1 | 200 | 0.5536 | 2.036e+05 | inf | 165/200 |
| EXPONENT_HIGH | 2 | 200 | 0.5821 | 8.153e+09 | inf | 192/200 |
| EXPONENT_HIGH | 4 | 200 | 1.876 | 1.866e+19 | inf | 189/200 |
| EXPONENT_LOW | 1 | 200 | 0.5342 | 2.693 | 957.9 | 136/200 |
| EXPONENT_LOW | 2 | 200 | 0.5700 | 20.41 | 939.7 | 174/200 |
| EXPONENT_LOW | 4 | 200 | 0.6691 | 208.8 | 1088 | 195/200 |
| MANTISSA_HIGH | 1 | 200 | 0.5247 | 0.6294 | 1.517 | 10/200 |
| MANTISSA_HIGH | 2 | 200 | 0.5247 | 0.6382 | 2.622 | 17/200 |
| MANTISSA_HIGH | 4 | 200 | 0.5247 | 0.6543 | 1.601 | 23/200 |
| MANTISSA_LOW | 1 | 200 | 0.5247 | 0.6217 | 0.827 | 0/200 |
| MANTISSA_LOW | 2 | 200 | 0.5247 | 0.6217 | 0.827 | 0/200 |
| MANTISSA_LOW | 4 | 200 | 0.5247 | 0.6217 | 0.827 | 0/200 |
| SIGN | 1 | 200 | 0.5562 | 2.298 | 8.715 | 155/200 |
| SIGN | 2 | 200 | 0.5765 | 3.437 | 10.56 | 188/200 |
| SIGN | 4 | 200 | 1.061 | 4.159 | 10.50 | 200/200 |

**Observations (measured, not explained in this document):**

- MANTISSA_LOW is identical at 1, 2, and 4 flips to four significant
  figures (min 0.5247, median 0.6217, max 0.827). This is recorded as
  a hard floor, not a sampling coincidence.
- EXPONENT_HIGH at 4 flips detects 189/200, lower than 2 flips at
  192/200, despite a median ratio nine orders of magnitude higher at
  4 flips than at 2 flips. Flagged as unexplained.
- Min ratio 0.5247 is shared across all MANTISSA_HIGH and MANTISSA_LOW
  cells (six cells). Recorded without explanation.

### Exponent-flip detection at n_flips=1

Pooling EXPONENT_HIGH and EXPONENT_LOW:

```
detected = 165 + 136 = 301
total    = 200 + 200 = 400
TPR      = 301/400 = 75.25%
```

## 5. v2 vs v3 comparison (n_flips = 1)

Same workload, shape, injection harness, and n = 200 per cell. v2 used
scalar residual-v2 with clean max 5.57e-8; v3 uses elementwise
residual-v3 with clean max 3.245921e-06.

| metric | v2 (scalar) | v3 (elementwise) |
| --- | --- | --- |
| EXPONENT_HIGH | 138/200 | 165/200 |
| EXPONENT_LOW | 41/200 | 136/200 |
| pooled exponent | 179/400 = 44.75% | 301/400 = **75.25%** |
| SIGN | 0/200 | 155/200 |

## 6. KT-1 verdict

**FAIL.** Pooled exponent-bit detection at `n_flips = 1` is 75.25%
(301/400) against KT-1's ≥90% bar. The false-positive half of KT-1
(FPR < 1e-6 per GEMM at production shapes) remains **UNEVALUATED**; no
threshold was derived and no FPR was measured. This verdict rests on the
detection half alone.

## 7. Findings

**(a) v2 structural invisibility was a reduction artifact.** SIGN went
from 0/200 (v2) to 155/200 (v3) at one flip. The claim that ones-vector
ABFT structurally cannot detect SIGN flips was wrong. Retracted in
`docs/RESULTS-KT1-v2.md` (void).

**(b) MANTISSA_LOW is genuinely undetectable at this shape.** Min, median,
and max ratio (0.5247, 0.6217, 0.827) are identical across
`n_flips` ∈ {1, 2, 4} to four significant figures; 0/200 detected in
every cell. A mantissa-LSB perturbation lies below fp32 accumulation
error at K = 4096 regardless of flip count. This is the surviving
mechanism result.

**(c) MANTISSA_HIGH is marginal, not invisible.** 10/200 at 1 flip,
17/200 at 2 flips, 23/200 at 4 flips — not the structural zero of v2.

## 8. Limitations

- Single GPU model (Tesla T4, sm75).
- Single shape (4096×4096×4096).
- Single dtype (bfloat16).
- Single workload (W02).
- No FPR threshold derived; FPR half of KT-1 unevaluated.
- Detection expected to degrade as K grows — untested.

## Reproduce

```bash
uv run assay characterize --sweep-v3
uv run assay characterize --sweep-v3-flips
python scripts/analyze_v3.py data/noisefloor/pilot/sweep-v3.jsonl
python scripts/analyze_v3_flips.py data/noisefloor/pilot/sweep-v3-flips.jsonl
```
