# Detection matrix (CP-5)

Test fixture, not a product feature. Each measured cell corrupts
`C` after `C = A @ B` with `flip_random`, then calls `check_gemm`
against the committed `data/noisefloor/` (noisefloor-v1).

Residual: **residual-v2** (`docs/RESIDUAL.md`). residual-v1
values in prior revisions of this file are void.

Shape: 16 x 16 x 16 GEMM, factors from `gemm_numpy_pair` case 0.
Device: CPU. Backend: PyTorch. Flip counts: 1, 2, 4.
Noisefloor: `noisefloor-v1`, n_samples=0.
Do not compare these 16-cubed residuals to the T4 4096³ pilot
floor: `scale_v2` differs by `(4096/16)³`.

## What 'caught' means

`caught` is `CheckResult.status == FAIL`. INCONCLUSIVE is not a
catch. PASS is not a catch. This table does not invent a threshold
to pretend a FAIL happened.

Lookup reason on every measured cell: `no noisefloor measurements for this configuration`

## Summary of measured GEMM cells

Measured cells: 45.
Detector FAIL (caught): 0.
Checksum residual hex changed after the flip: 41.

| bit_class | n_cells | checksum_moved | caught |
| --- | --- | --- | --- |
| SIGN | 9 | 9 | 0 |
| EXPONENT_HIGH | 9 | 9 | 0 |
| EXPONENT_LOW | 9 | 9 | 0 |
| MANTISSA_HIGH | 9 | 8 | 0 |
| MANTISSA_LOW | 9 | 6 | 0 |

## Which classes the detector does not catch, and why

Caught (FAIL): 0 of 45. The detector did not
catch any flip. Two different reasons, do not mix them:

1. Uncharacterized noisefloor. Every cell is INCONCLUSIVE because
   n_samples=0. `check_gemm` is not allowed to FAIL.
   Residual-v2 scale does not depend on C. checksum_moved is the
   numerator. See the summary table for per-class counts.
   EXPONENT_HIGH includes Inf-class cells. Those still do not FAIL
   today. After a real floor exists, EXPONENT_HIGH is the class
   most likely to exceed sample max.

2. Checksum-invisible flips. The residual hex was unchanged in
   4 of 45 cells. Classes: MANTISSA_HIGH, MANTISSA_LOW.

- W01 float32 MANTISSA_LOW n_flips=1 residual `0x1.0e71be03fa96ap-29`
- W02 bfloat16 MANTISSA_HIGH n_flips=2 residual `0x1.0e6d15fa5ad9cp-15`
- W02 bfloat16 MANTISSA_LOW n_flips=1 residual `0x1.0e6d15fa5ad9cp-15`
- W02 bfloat16 MANTISSA_LOW n_flips=2 residual `0x1.0e6d15fa5ad9cp-15`

A ones-vector checksum sums each row of C. A MANTISSA_LOW flip
is a few ULPs in one element. Native-dtype `C @ e` can round
that delta away, so A@(B@e) vs C@e does not move.
Characterizing the GPU will not make those cells FAIL. This is
the asymmetry the bit_class API exists to record.

W04-W07 have no GEMM checksum. int8 has an injector layout and no
`check_gemm` path. Those are out of scope for CP-4, not a mantissa
finding.

## Applicability of workload x dtype

| workload | float32 | bfloat16 | float16 | int8 |
| --- | --- | --- | --- | --- |
| W01 | measured | not a suite dtype (suite uses float32) | not a suite dtype (suite uses float32) | check_gemm rejects int8; injector-only |
| W02 | not a suite dtype (suite uses bfloat16) | measured | not a suite dtype (suite uses bfloat16) | check_gemm rejects int8; injector-only |
| W03 | not a suite dtype (suite uses float16) | not a suite dtype (suite uses float16) | measured | check_gemm rejects int8; injector-only |
| W04 | no GEMM ABFT in CP-4 | no GEMM ABFT in CP-4 | no GEMM ABFT in CP-4 | check_gemm rejects int8; injector-only |
| W05 | no GEMM ABFT in CP-4 | no GEMM ABFT in CP-4 | no GEMM ABFT in CP-4 | check_gemm rejects int8; injector-only |
| W06 | no GEMM ABFT in CP-4 | no GEMM ABFT in CP-4 | no GEMM ABFT in CP-4 | check_gemm rejects int8; injector-only |
| W07 | no GEMM ABFT in CP-4 | no GEMM ABFT in CP-4 | no GEMM ABFT in CP-4 | check_gemm rejects int8; injector-only |

## Measured product: GEMM workload x native dtype x bit_class x n_flips

| workload | dtype | bit_class | n_flips | detector | checksum_moved | caught | residual_clean_hex | residual_flipped_hex | n_samples | spec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W01 | float32 | SIGN | 1 | INCONCLUSIVE | True | False | `0x1.0e71be03fa96ap-29` | `0x1.d7e69d4d4902ep-12` | 0 | noisefloor-v1 |
| W01 | float32 | SIGN | 2 | INCONCLUSIVE | True | False | `0x1.0e71be03fa96ap-29` | `0x1.0ec9692301762p-10` | 0 | noisefloor-v1 |
| W01 | float32 | SIGN | 4 | INCONCLUSIVE | True | False | `0x1.0e71be03fa96ap-29` | `0x1.b33773a3e4cb2p-8` | 0 | noisefloor-v1 |
| W01 | float32 | EXPONENT_HIGH | 1 | INCONCLUSIVE | True | False | `0x1.0e71be03fa96ap-29` | `0x1.acce1fce8d32ap-11` | 0 | noisefloor-v1 |
| W01 | float32 | EXPONENT_HIGH | 2 | INCONCLUSIVE | True | False | `0x1.0e71be03fa96ap-29` | `nan` | 0 | noisefloor-v1 |
| W01 | float32 | EXPONENT_HIGH | 4 | INCONCLUSIVE | True | False | `0x1.0e71be03fa96ap-29` | `0x1.435109c11b2eap+117` | 0 | noisefloor-v1 |
| W01 | float32 | EXPONENT_LOW | 1 | INCONCLUSIVE | True | False | `0x1.0e71be03fa96ap-29` | `0x1.63ff580ae43a7p-11` | 0 | noisefloor-v1 |
| W01 | float32 | EXPONENT_LOW | 2 | INCONCLUSIVE | True | False | `0x1.0e71be03fa96ap-29` | `0x1.b047404073721p-13` | 0 | noisefloor-v1 |
| W01 | float32 | EXPONENT_LOW | 4 | INCONCLUSIVE | True | False | `0x1.0e71be03fa96ap-29` | `0x1.30f04a7d8ee6ep-9` | 0 | noisefloor-v1 |
| W01 | float32 | MANTISSA_HIGH | 1 | INCONCLUSIVE | True | False | `0x1.0e71be03fa96ap-29` | `0x1.e0cc523fe4303p-14` | 0 | noisefloor-v1 |
| W01 | float32 | MANTISSA_HIGH | 2 | INCONCLUSIVE | True | False | `0x1.0e71be03fa96ap-29` | `0x1.f0def8c54f641p-21` | 0 | noisefloor-v1 |
| W01 | float32 | MANTISSA_HIGH | 4 | INCONCLUSIVE | True | False | `0x1.0e71be03fa96ap-29` | `0x1.e19a696d1bb86p-11` | 0 | noisefloor-v1 |
| W01 | float32 | MANTISSA_LOW | 1 | INCONCLUSIVE | False | False | `0x1.0e71be03fa96ap-29` | `0x1.0e71be03fa96ap-29` | 0 | noisefloor-v1 |
| W01 | float32 | MANTISSA_LOW | 2 | INCONCLUSIVE | True | False | `0x1.0e71be03fa96ap-29` | `0x1.ba4a041681d10p-24` | 0 | noisefloor-v1 |
| W01 | float32 | MANTISSA_LOW | 4 | INCONCLUSIVE | True | False | `0x1.0e71be03fa96ap-29` | `0x1.daafa42efbd5bp-24` | 0 | noisefloor-v1 |
| W02 | bfloat16 | SIGN | 1 | INCONCLUSIVE | True | False | `0x1.0e6d15fa5ad9cp-15` | `0x1.2da5ae1225c8bp-8` | 0 | noisefloor-v1 |
| W02 | bfloat16 | SIGN | 2 | INCONCLUSIVE | True | False | `0x1.0e6d15fa5ad9cp-15` | `0x1.f401b1f83aa54p-9` | 0 | noisefloor-v1 |
| W02 | bfloat16 | SIGN | 4 | INCONCLUSIVE | True | False | `0x1.0e6d15fa5ad9cp-15` | `0x1.5605f79315205p-8` | 0 | noisefloor-v1 |
| W02 | bfloat16 | EXPONENT_HIGH | 1 | INCONCLUSIVE | True | False | `0x1.0e6d15fa5ad9cp-15` | `0x1.122e99d6b88e9p+117` | 0 | noisefloor-v1 |
| W02 | bfloat16 | EXPONENT_HIGH | 2 | INCONCLUSIVE | True | False | `0x1.0e6d15fa5ad9cp-15` | `0x1.8e20993eccdd0p-13` | 0 | noisefloor-v1 |
| W02 | bfloat16 | EXPONENT_HIGH | 4 | INCONCLUSIVE | True | False | `0x1.0e6d15fa5ad9cp-15` | `0x1.37bdc072619e9p+55` | 0 | noisefloor-v1 |
| W02 | bfloat16 | EXPONENT_LOW | 1 | INCONCLUSIVE | True | False | `0x1.0e6d15fa5ad9cp-15` | `0x1.bef44b6f8f0b7p-12` | 0 | noisefloor-v1 |
| W02 | bfloat16 | EXPONENT_LOW | 2 | INCONCLUSIVE | True | False | `0x1.0e6d15fa5ad9cp-15` | `0x1.73ec874f671dap-5` | 0 | noisefloor-v1 |
| W02 | bfloat16 | EXPONENT_LOW | 4 | INCONCLUSIVE | True | False | `0x1.0e6d15fa5ad9cp-15` | `0x1.f1a8bf8e80144p-11` | 0 | noisefloor-v1 |
| W02 | bfloat16 | MANTISSA_HIGH | 1 | INCONCLUSIVE | True | False | `0x1.0e6d15fa5ad9cp-15` | `0x1.0147c87712e0fp-11` | 0 | noisefloor-v1 |
| W02 | bfloat16 | MANTISSA_HIGH | 2 | INCONCLUSIVE | False | False | `0x1.0e6d15fa5ad9cp-15` | `0x1.0e6d15fa5ad9cp-15` | 0 | noisefloor-v1 |
| W02 | bfloat16 | MANTISSA_HIGH | 4 | INCONCLUSIVE | True | False | `0x1.0e6d15fa5ad9cp-15` | `0x1.66b0b0b4f4f2ap-11` | 0 | noisefloor-v1 |
| W02 | bfloat16 | MANTISSA_LOW | 1 | INCONCLUSIVE | False | False | `0x1.0e6d15fa5ad9cp-15` | `0x1.0e6d15fa5ad9cp-15` | 0 | noisefloor-v1 |
| W02 | bfloat16 | MANTISSA_LOW | 2 | INCONCLUSIVE | False | False | `0x1.0e6d15fa5ad9cp-15` | `0x1.0e6d15fa5ad9cp-15` | 0 | noisefloor-v1 |
| W02 | bfloat16 | MANTISSA_LOW | 4 | INCONCLUSIVE | True | False | `0x1.0e6d15fa5ad9cp-15` | `0x1.33fc3c9603e9cp-13` | 0 | noisefloor-v1 |
| W03 | float16 | SIGN | 1 | INCONCLUSIVE | True | False | `0x1.2c7fa0c8b6891p-18` | `0x1.c8da08d78be93p-11` | 0 | noisefloor-v1 |
| W03 | float16 | SIGN | 2 | INCONCLUSIVE | True | False | `0x1.2c7fa0c8b6891p-18` | `0x1.07c98937646f7p-7` | 0 | noisefloor-v1 |
| W03 | float16 | SIGN | 4 | INCONCLUSIVE | True | False | `0x1.2c7fa0c8b6891p-18` | `0x1.9507061021a17p-8` | 0 | noisefloor-v1 |
| W03 | float16 | EXPONENT_HIGH | 1 | INCONCLUSIVE | True | False | `0x1.2c7fa0c8b6891p-18` | `0x1.69115a645e871p-11` | 0 | noisefloor-v1 |
| W03 | float16 | EXPONENT_HIGH | 2 | INCONCLUSIVE | True | False | `0x1.2c7fa0c8b6891p-18` | `0x1.750a7032c4336p-11` | 0 | noisefloor-v1 |
| W03 | float16 | EXPONENT_HIGH | 4 | INCONCLUSIVE | True | False | `0x1.2c7fa0c8b6891p-18` | `nan` | 0 | noisefloor-v1 |
| W03 | float16 | EXPONENT_LOW | 1 | INCONCLUSIVE | True | False | `0x1.2c7fa0c8b6891p-18` | `0x1.658bdb8204638p-10` | 0 | noisefloor-v1 |
| W03 | float16 | EXPONENT_LOW | 2 | INCONCLUSIVE | True | False | `0x1.2c7fa0c8b6891p-18` | `0x1.0478a0aac7056p-9` | 0 | noisefloor-v1 |
| W03 | float16 | EXPONENT_LOW | 4 | INCONCLUSIVE | True | False | `0x1.2c7fa0c8b6891p-18` | `0x1.6b976ccadadc5p-8` | 0 | noisefloor-v1 |
| W03 | float16 | MANTISSA_HIGH | 1 | INCONCLUSIVE | True | False | `0x1.2c7fa0c8b6891p-18` | `0x1.032e1446b7097p-14` | 0 | noisefloor-v1 |
| W03 | float16 | MANTISSA_HIGH | 2 | INCONCLUSIVE | True | False | `0x1.2c7fa0c8b6891p-18` | `0x1.f3942e80e2aa4p-14` | 0 | noisefloor-v1 |
| W03 | float16 | MANTISSA_HIGH | 4 | INCONCLUSIVE | True | False | `0x1.2c7fa0c8b6891p-18` | `0x1.d5876b399d363p-14` | 0 | noisefloor-v1 |
| W03 | float16 | MANTISSA_LOW | 1 | INCONCLUSIVE | True | False | `0x1.2c7fa0c8b6891p-18` | `0x1.a4b2ade5cc598p-18` | 0 | noisefloor-v1 |
| W03 | float16 | MANTISSA_LOW | 2 | INCONCLUSIVE | True | False | `0x1.2c7fa0c8b6891p-18` | `0x1.6899275741714p-19` | 0 | noisefloor-v1 |
| W03 | float16 | MANTISSA_LOW | 4 | INCONCLUSIVE | True | False | `0x1.2c7fa0c8b6891p-18` | `0x1.6899275741714p-19` | 0 | noisefloor-v1 |

## Reproduce

```bash
uv run pytest -m cpu tests/test_detection_matrix.py
```

The test re-measures and asserts this file matches. Do not edit
numbers by hand.

## W02 bfloat16 4096³ flip residuals vs T4 residual-v2 pilot floor

Not a KT-1 evaluation. No threshold is set. residual-v1 flip
residuals are void. A single `sample_index=0` draw per cell is
not evidence; cells below are n = 200 independent `(A, B)`.

Clean floor: Tesla T4 residual-v2 pilot, n = 2000, W02 bf16 4096³.

| statistic | value |
| --- | --- |
| n | 2000 |
| median | 1.07e-08 |
| p99.9 | 5.07e-08 |
| max | 5.57e-08 |

Ratio is residual-v2 / clean max (5.57e-08). `n(ratio>1)` is the count of samples with
ratio > 1 or non-finite, out of 200. Not a pass/fail rule.

bfloat16 bit_class → bit indices (bit 0 is IEEE LSB of the 16-bit
pattern). Exponent field is bits 14-7 (8 bits). Mantissa is 6-0
(7 bits). Split is ceil(field_width/2) on the high side
(`LAYOUT_BF16` in `src/assay/inject/bits.py`).

| bit_class | bits | field role |
| --- | --- | --- |
| SIGN | 15 | sign |
| EXPONENT_HIGH | 11, 12, 13, 14 | exponent bits [4:7], includes MSB |
| EXPONENT_LOW | 7, 8, 9, 10 | exponent bits [0:3], includes LSB |
| MANTISSA_HIGH | 3, 4, 5, 6 | mantissa bits [3:6] |
| MANTISSA_LOW | 0, 1, 2 | mantissa bits [0:2] |

`flip_random` draws uniformly among the bits in the class.
**EXPONENT_LOW includes the exponent LSB (bit 7).** Flipping only
that bit scales one element by 2x or 1/2. That is a small change to
a 4096-term ones-vector row sum: a structural property of this
checksum, not a detector bug. The same class also contains bits
8, 9, and 10 (element scale 4x, 16x, 256x), so a 1-flip cell is a
mixture of those four magnitudes.

SIGN is bit 15 only: `C_ij → -C_ij`, which adds `-2 C_ij` to the
row sum. The ones-vector can cancel that the same way it can
cancel a mantissa ULP.

Flips are injected into **C after the GEMM**. That is not an
intermediate accumulator SDC.

Factors: `gemm_numpy_pair` case_index=3, workload_id=2,
`sample_index` in 0..199 (same mixer as the pilot). One GEMM per
sample_index; every (bit_class, n_flips) reuses that C.

200-sample flip residuals: **UNMEASURED** on the writer
(no NVIDIA GPU). Reproduce on CUDA (~200 GEMMs at 4096³):

```bash
uv run pytest -m gpu tests/test_w02_4096_flips.py -s
```
