# Detection matrix (CP-5)

Test fixture, not a product feature. Each measured cell corrupts
`C` after `C = A @ B` with `flip_random`, then calls `check_gemm`
against the committed `data/noisefloor/` (noisefloor-v1).

Shape: 16 x 16 x 16 GEMM, factors from `gemm_numpy_pair` case 0.
Device: CPU. Backend: PyTorch. Flip counts: 1, 2, 4.
Noisefloor: `noisefloor-v1`, n_samples=0.

## What 'caught' means

`caught` is `CheckResult.status == FAIL`. INCONCLUSIVE is not a
catch. PASS is not a catch. This table does not invent a threshold
to pretend a FAIL happened.

Lookup reason on every measured cell: `no noisefloor measurements for this configuration`

## Summary of measured GEMM cells

Measured cells: 45.
Detector FAIL (caught): 0.
Checksum residual hex changed after the flip: 42.

| bit_class | n_cells | checksum_moved | caught |
| --- | --- | --- | --- |
| SIGN | 9 | 9 | 0 |
| EXPONENT_HIGH | 9 | 9 | 0 |
| EXPONENT_LOW | 9 | 9 | 0 |
| MANTISSA_HIGH | 9 | 9 | 0 |
| MANTISSA_LOW | 9 | 6 | 0 |

## Which classes the detector does not catch, and why

Caught (FAIL): 0 of 45. The detector did not
catch any flip. Two different reasons, do not mix them:

1. Uncharacterized noisefloor. Every cell is INCONCLUSIVE because
   n_samples=0. `check_gemm` is not allowed to FAIL.
   SIGN, EXPONENT_HIGH, EXPONENT_LOW, and MANTISSA_HIGH all moved
   the residual (9/9 each), including Inf-class EXPONENT_HIGH.
   Those would still not FAIL today. After a real floor exists,
   EXPONENT_HIGH is the class most likely to exceed sample max.

2. Checksum-invisible flips. The residual hex was unchanged in
   3 of 45 cells. Classes: MANTISSA_LOW.

- W01 float32 MANTISSA_LOW n_flips=1 residual `0x1.e9e38e83975ddp-23`
- W02 bfloat16 MANTISSA_LOW n_flips=1 residual `0x1.ea7f85601ea80p-9`
- W02 bfloat16 MANTISSA_LOW n_flips=2 residual `0x1.ea7f85601ea80p-9`

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
| W01 | float32 | SIGN | 1 | INCONCLUSIVE | True | False | `0x1.e9e38e83975ddp-23` | `0x1.ab67eee68c18dp-5` | 0 | noisefloor-v1 |
| W01 | float32 | SIGN | 2 | INCONCLUSIVE | True | False | `0x1.e9e38e83975ddp-23` | `0x1.b60d2005fcb65p-4` | 0 | noisefloor-v1 |
| W01 | float32 | SIGN | 4 | INCONCLUSIVE | True | False | `0x1.e9e38e83975ddp-23` | `0x1.8a2e3fe86aed1p-1` | 0 | noisefloor-v1 |
| W01 | float32 | EXPONENT_HIGH | 1 | INCONCLUSIVE | True | False | `0x1.e9e38e83975ddp-23` | `0x1.62bcfdd51ad86p-4` | 0 | noisefloor-v1 |
| W01 | float32 | EXPONENT_HIGH | 2 | INCONCLUSIVE | True | False | `0x1.e9e38e83975ddp-23` | `nan` | 0 | noisefloor-v1 |
| W01 | float32 | EXPONENT_HIGH | 4 | INCONCLUSIVE | True | False | `0x1.e9e38e83975ddp-23` | `0x1.0000000000000p+0` | 0 | noisefloor-v1 |
| W01 | float32 | EXPONENT_LOW | 1 | INCONCLUSIVE | True | False | `0x1.e9e38e83975ddp-23` | `0x1.426e52ec7a1f7p-4` | 0 | noisefloor-v1 |
| W01 | float32 | EXPONENT_LOW | 2 | INCONCLUSIVE | True | False | `0x1.e9e38e83975ddp-23` | `0x1.7e61c1584946dp-6` | 0 | noisefloor-v1 |
| W01 | float32 | EXPONENT_LOW | 4 | INCONCLUSIVE | True | False | `0x1.e9e38e83975ddp-23` | `0x1.b30995eb092fbp-3` | 0 | noisefloor-v1 |
| W01 | float32 | MANTISSA_HIGH | 1 | INCONCLUSIVE | True | False | `0x1.e9e38e83975ddp-23` | `0x1.adc0d468989adp-7` | 0 | noisefloor-v1 |
| W01 | float32 | MANTISSA_HIGH | 2 | INCONCLUSIVE | True | False | `0x1.e9e38e83975ddp-23` | `0x1.c1f93472e3229p-14` | 0 | noisefloor-v1 |
| W01 | float32 | MANTISSA_HIGH | 4 | INCONCLUSIVE | True | False | `0x1.e9e38e83975ddp-23` | `0x1.b431897235ef3p-4` | 0 | noisefloor-v1 |
| W01 | float32 | MANTISSA_LOW | 1 | INCONCLUSIVE | False | False | `0x1.e9e38e83975ddp-23` | `0x1.e9e38e83975ddp-23` | 0 | noisefloor-v1 |
| W01 | float32 | MANTISSA_LOW | 2 | INCONCLUSIVE | True | False | `0x1.e9e38e83975ddp-23` | `0x1.90961930c0052p-17` | 0 | noisefloor-v1 |
| W01 | float32 | MANTISSA_LOW | 4 | INCONCLUSIVE | True | False | `0x1.e9e38e83975ddp-23` | `0x1.adec51d2cd5dbp-17` | 0 | noisefloor-v1 |
| W02 | bfloat16 | SIGN | 1 | INCONCLUSIVE | True | False | `0x1.ea7f85601ea80p-9` | `0x1.12974aca93df8p-1` | 0 | noisefloor-v1 |
| W02 | bfloat16 | SIGN | 2 | INCONCLUSIVE | True | False | `0x1.ea7f85601ea80p-9` | `0x1.3b1984c558184p-2` | 0 | noisefloor-v1 |
| W02 | bfloat16 | SIGN | 4 | INCONCLUSIVE | True | False | `0x1.ea7f85601ea80p-9` | `0x1.8338f0c8eb783p-2` | 0 | noisefloor-v1 |
| W02 | bfloat16 | EXPONENT_HIGH | 1 | INCONCLUSIVE | True | False | `0x1.ea7f85601ea80p-9` | `0x1.0000000000000p+0` | 0 | noisefloor-v1 |
| W02 | bfloat16 | EXPONENT_HIGH | 2 | INCONCLUSIVE | True | False | `0x1.ea7f85601ea80p-9` | `0x1.6292f8ff0f284p-6` | 0 | noisefloor-v1 |
| W02 | bfloat16 | EXPONENT_HIGH | 4 | INCONCLUSIVE | True | False | `0x1.ea7f85601ea80p-9` | `0x1.0000000000000p+0` | 0 | noisefloor-v1 |
| W02 | bfloat16 | EXPONENT_LOW | 1 | INCONCLUSIVE | True | False | `0x1.ea7f85601ea80p-9` | `0x1.96dd61deaa18dp-5` | 0 | noisefloor-v1 |
| W02 | bfloat16 | EXPONENT_LOW | 2 | INCONCLUSIVE | True | False | `0x1.ea7f85601ea80p-9` | `0x1.3bac38161da32p+0` | 0 | noisefloor-v1 |
| W02 | bfloat16 | EXPONENT_LOW | 4 | INCONCLUSIVE | True | False | `0x1.ea7f85601ea80p-9` | `0x1.97e8242c4108cp-4` | 0 | noisefloor-v1 |
| W02 | bfloat16 | MANTISSA_HIGH | 1 | INCONCLUSIVE | True | False | `0x1.ea7f85601ea80p-9` | `0x1.bb12b27b80a83p-5` | 0 | noisefloor-v1 |
| W02 | bfloat16 | MANTISSA_HIGH | 2 | INCONCLUSIVE | True | False | `0x1.ea7f85601ea80p-9` | `0x1.ec572f7066923p-9` | 0 | noisefloor-v1 |
| W02 | bfloat16 | MANTISSA_HIGH | 4 | INCONCLUSIVE | True | False | `0x1.ea7f85601ea80p-9` | `0x1.46847ccb6eb0fp-4` | 0 | noisefloor-v1 |
| W02 | bfloat16 | MANTISSA_LOW | 1 | INCONCLUSIVE | False | False | `0x1.ea7f85601ea80p-9` | `0x1.ea7f85601ea80p-9` | 0 | noisefloor-v1 |
| W02 | bfloat16 | MANTISSA_LOW | 2 | INCONCLUSIVE | False | False | `0x1.ea7f85601ea80p-9` | `0x1.ea7f85601ea80p-9` | 0 | noisefloor-v1 |
| W02 | bfloat16 | MANTISSA_LOW | 4 | INCONCLUSIVE | True | False | `0x1.ea7f85601ea80p-9` | `0x1.13a4d1ac97b03p-6` | 0 | noisefloor-v1 |
| W03 | float16 | SIGN | 1 | INCONCLUSIVE | True | False | `0x1.101a5c2753cf5p-11` | `0x1.9dae79b3ca379p-4` | 0 | noisefloor-v1 |
| W03 | float16 | SIGN | 2 | INCONCLUSIVE | True | False | `0x1.101a5c2753cf5p-11` | `0x1.ddb8ade40bb12p-1` | 0 | noisefloor-v1 |
| W03 | float16 | SIGN | 4 | INCONCLUSIVE | True | False | `0x1.101a5c2753cf5p-11` | `0x1.6ec1211c01e9dp-1` | 0 | noisefloor-v1 |
| W03 | float16 | EXPONENT_HIGH | 1 | INCONCLUSIVE | True | False | `0x1.101a5c2753cf5p-11` | `0x1.46f2df87411a5p-4` | 0 | noisefloor-v1 |
| W03 | float16 | EXPONENT_HIGH | 2 | INCONCLUSIVE | True | False | `0x1.101a5c2753cf5p-11` | `0x1.380e3ce763f8ep-4` | 0 | noisefloor-v1 |
| W03 | float16 | EXPONENT_HIGH | 4 | INCONCLUSIVE | True | False | `0x1.101a5c2753cf5p-11` | `nan` | 0 | noisefloor-v1 |
| W03 | float16 | EXPONENT_LOW | 1 | INCONCLUSIVE | True | False | `0x1.101a5c2753cf5p-11` | `0x1.179093f36d0f0p-3` | 0 | noisefloor-v1 |
| W03 | float16 | EXPONENT_LOW | 2 | INCONCLUSIVE | True | False | `0x1.101a5c2753cf5p-11` | `0x1.d7b74c282d7e0p-3` | 0 | noisefloor-v1 |
| W03 | float16 | EXPONENT_LOW | 4 | INCONCLUSIVE | True | False | `0x1.101a5c2753cf5p-11` | `0x1.493be50095a82p-1` | 0 | noisefloor-v1 |
| W03 | float16 | MANTISSA_HIGH | 1 | INCONCLUSIVE | True | False | `0x1.101a5c2753cf5p-11` | `0x1.d560abc3d6f8dp-8` | 0 | noisefloor-v1 |
| W03 | float16 | MANTISSA_HIGH | 2 | INCONCLUSIVE | True | False | `0x1.101a5c2753cf5p-11` | `0x1.c45f060161bbep-7` | 0 | noisefloor-v1 |
| W03 | float16 | MANTISSA_HIGH | 4 | INCONCLUSIVE | True | False | `0x1.101a5c2753cf5p-11` | `0x1.a9292ffd72f3fp-7` | 0 | noisefloor-v1 |
| W03 | float16 | MANTISSA_LOW | 1 | INCONCLUSIVE | True | False | `0x1.101a5c2753cf5p-11` | `0x1.7cf1b4370eef1p-11` | 0 | noisefloor-v1 |
| W03 | float16 | MANTISSA_LOW | 2 | INCONCLUSIVE | True | False | `0x1.101a5c2753cf5p-11` | `0x1.466c02a80bb03p-12` | 0 | noisefloor-v1 |
| W03 | float16 | MANTISSA_LOW | 4 | INCONCLUSIVE | True | False | `0x1.101a5c2753cf5p-11` | `0x1.466c02a80bb03p-12` | 0 | noisefloor-v1 |

## Reproduce

```bash
uv run pytest -m cpu tests/test_detection_matrix.py
```

The test re-measures and asserts this file matches. Do not edit
numbers by hand.
