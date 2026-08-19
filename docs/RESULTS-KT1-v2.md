# RESULTS-KT1-v2

STATUS: VOID -- INVALID RESIDUAL, NOT A KILL
This evaluation used residual-v2, which reduced the length-M checksum
vector to a scalar before comparison. That is not the Huang-Abraham
elementwise formulation (`docs/RESIDUAL.md`, residual-v3).
The "structural invisibility" of SIGN and MANTISSA_HIGH was an artifact
of that reduction, falsified by v3 (SIGN 0/200 → 155/200 at 1 flip).
The KT-1 bars (FPR < 1e-6 per GEMM at production shapes, >=90% exponent
flip detection) are UNCHANGED and UNEVALUATED pending v3 write-up.
This file is retained as evidence that the bars were not moved.
Superseded by `docs/RESULTS-KT1-v3.md` (to be written).

**STATUS: FAIL (VOID).** This was the first residual-v2 KT-1 evaluation.
It is void. Do not cite its mechanism claims.

The bars are the same numbers as `README.md` at CP-0. They were not
moved. The detector was not retuned. No threshold was adjusted.

## 1. Method

- **Residual:** residual-v2 (`docs/RESIDUAL.md`).
  Statistic: `max_i |row_i(C @ e) - row_i(A @ (B @ e))| / max_i |row_i(A @ (B @ e))|`,
  promoted to float64 before abs and division.
- **Clean floor:** T4 GPU, n=2000 independent (A, B) draws.
  Median residual: 1.07e-8. Max residual: 5.57e-8.
  Produced by `assay characterize --pilot`.
- **Injection grid:** n=200 per cell.
  Workload W02, bfloat16, shape 4096×4096×4096.
  Flips injected into C after GEMM (`src/assay/inject/`).
  Bit classes: SIGN, EXPONENT_HIGH, EXPONENT_LOW, MANTISSA_HIGH,
  MANTISSA_LOW. `n_flips` in {1, 2, 4}. Seeds: PCG64 from `BASE_SEED`.
- **Detection criterion:** sample residual exceeds the clean max (5.57e-8).
- **bf16 bit layout (16 bits):**

  | field | bits | position |
  | --- | --- | --- |
  | SIGN | 1 | [15] |
  | EXPONENT_HIGH | 4 | [14:11] — includes MSB (inf/NaN) |
  | EXPONENT_LOW | 4 | [10:7] — includes LSB of exponent |
  | MANTISSA_HIGH | 4 | [6:3] |
  | MANTISSA_LOW | 3 | [2:0] |

## 2. True-positive rate by bit_class and n_flips

Detection = sample residual > clean max (5.57e-8).

| bit_class | n_flips=1 | n_flips=2 | n_flips=4 | total |
| --- | --- | --- | --- | --- |
| SIGN | 0/200 | 0/200 | 0/200 | 0/600 |
| EXPONENT_HIGH | 138/200 | 183/200 | 199/200 | 520/600 |
| EXPONENT_LOW | 41/200 | 78/200 | 131/200 | 250/600 |
| MANTISSA_HIGH | 0/200 | 0/200 | 0/200 | 0/600 |
| MANTISSA_LOW | 0/200 | 0/200 | 0/200 | 0/600 |

### Exponent-flip detection at n_flips=1

KT-1 specifies detection of **exponent-bit flips**. Single-bit flips
(`n_flips=1`) are the canonical case.

Pooling EXPONENT_HIGH and EXPONENT_LOW at n_flips=1:

```
detected = 138 + 41 = 179
total    = 200 + 200 = 400
TPR      = 179/400 = 44.75%
```

Clopper–Pearson exact 95% two-sided confidence interval for a binomial
proportion with k=179, n=400:

```
lower = B.inv(alpha/2,     k,     n-k+1) = B.inv(0.025, 179, 222) ≈ 0.398
upper = B.inv(1 - alpha/2, k+1,   n-k)   = B.inv(0.975, 180, 221) ≈ 0.498

95% CI: [39.8%, 49.8%]
```

The entire confidence interval is below 50%. It does not contain 90%.

Even pooling all n_flips values:
`(138+183+199+41+78+131) / 1200 = 770/1200 = 64.2%`. Still below 90%.

## 3. Verdict

**FAIL** against KT-1's ≥90% exponent-flip detection bar.

| KT-1 clause (README, unchanged) | v2 result |
| --- | --- |
| ≥90% of exponent-bit flips detected | 179/400 = 44.75% at n_flips=1; 95% CI [39.8%, 49.8%] |
| FPR < 1e-6 per GEMM at production shapes | NOT ESTABLISHED (see §4) |

The detector does not meet the detection half of KT-1. The bars were
not moved. The detector was not retuned.

## 4. False-positive rate

**NOT ESTABLISHED.** No threshold was set from the clean floor — the
clean max was used only as a detection criterion for true-positive
measurement, not as a deployed decision boundary. No false-positive
rate was measured. The < 1e-6 FPR half of KT-1 is unevaluated.

FAIL rests on the detection half alone: the ones-vector checksum
cannot catch ≥90% of exponent-bit flips at this shape, so even a
perfect FPR would not save KT-1.

## 5. Mechanism

Detectability is governed by the **magnitude of the perturbation
relative to the row sum**, not by which IEEE 754 field the flipped bit
occupies.

A ones-vector checksum computes a row sum. For a 4096-element row of
bf16 values drawn from standard normal, the row sum is O(√4096) ≈ 64.
The residual is the difference of two such sums (one recomputed).

- **SIGN, MANTISSA_HIGH, MANTISSA_LOW:** 0 of 1800 samples across all
  cells exceeded the clean floor. Max ratio to clean max: 0.92.
  A sign flip changes one element by 2|x| ≈ O(1); mantissa flips
  change by O(ULP). Both are swamped by the row sum.
- **EXPONENT_HIGH at n_flips=1:** 138/200 caught. The MSB (bit 14)
  produces inf, always caught. Lower bits (11–13) scale the element
  by 2^k — caught only when 2^k exceeds the row-sum noise floor.
  Min ratio 0.013 shows that some lower-exponent-high flips vanish.
- **EXPONENT_LOW at n_flips=1:** 41/200 caught. Median ratio 0.288,
  max 13.9. Bits 7–8 add O(1) perturbation, invisible in a 4096-term
  sum; bits 9–10 add O(4)–O(16), sometimes caught. This is the
  four-bit mixture effect: roughly one quarter of random draws hit a
  high-magnitude bit and are detected, the rest are absorbed.

**Prediction (untested):** detection degrades as K (inner dimension)
grows, because the row sum grows as O(√K) while single-element
perturbations stay fixed. At K=16384 (common in transformer FFNs),
the ones-vector checksum should catch even fewer exponent-low flips.

## 6. Limitations

- **One GPU.** T4 only. No A100, H100, or other production GPU tested.
- **One dtype.** bfloat16 only. fp16, fp32 not evaluated under v2.
- **One shape.** 4096×4096×4096. Rectangular and smaller/larger shapes
  untested.
- **Flips in C, not an intermediate accumulator.** Real bit flips in
  hardware could corrupt partial sums during accumulation, which the
  checksum might detect differently. This injection model flips the
  final output.
- **n=200 per cell.** Adequate for the observed rates but not large
  enough to resolve sub-1% detection probabilities precisely.
- **Clean floor from pilot.** n=2000 is below noisefloor-v1's
  min_samples=100000. The clean max may not represent the true
  p99.999 tail.

## Reproduce

```bash
uv run assay characterize --pilot
# Then run the detection matrix at n=200 per cell on T4
uv run pytest -m gpu tests/test_detection_matrix.py
```

Confidence interval computation:

```python
from scipy.stats import beta
k, n, alpha = 179, 400, 0.05
lo = beta.ppf(alpha/2, k, n - k + 1)
hi = beta.ppf(1 - alpha/2, k + 1, n - k)
print(f"95% CI: [{lo:.3f}, {hi:.3f}]")
```
