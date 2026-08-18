# RESULTS-KT1

STATUS: VOID -- NULL EVALUATION, NOT A KILL
This evaluation was run with zero GPU noisefloor samples. A detector that
is structurally incapable of returning FAIL cannot be measured for true
positive rate, and 45 CPU calls at 16x16x16 are not production shapes.
The evaluation additionally used residual-v1, a residual definition
since found defective (`docs/RESIDUAL.md`): the denominator was the
signed grand sum, whose cancellation artifact is not hardware noise.
A valid KT-1 evaluation requires residual-v2 noisefloor data.
The KT-1 bars (FPR < 1e-6 per GEMM at production shapes, >=90% exponent
flip detection) are UNCHANGED and UNEVALUATED.
This file is retained as evidence that the bars were not moved.
Superseded by RESULTS-KT1.md v2, to be written after CP-3 produces
>=1 characterized GPU model under residual-v2.

**Verdict: FAIL.** noisefloor-v1 `check_gemm` does not meet KT-1 as written
at CP-0: it has not shown a false-positive rate below `1e-6` per GEMM at
production shapes, and it caught `0` of `18` exponent-bit flips (`0%`,
not `>=90%`).

The bars are the same numbers as `README.md` at CP-0. They were not
moved. The detector was not retuned against this file.

## 1. Method

### Noise floor characterization

Spec: `noisefloor-v1` (`data/noisefloor/methodology-v1.json`,
`docs/SPEC-NOISEFLOOR.md`).

Tolerance, when it exists, is the empirical quantile at
`target_quantile = 99999/100000` of the normalized ones-checksum residual.
`min_samples = ceil(1/(1-p)) = 100000`. Distinct GPU models are not
pooled. Missing or under-sampled keys are UNCHARACTERIZED.
`check_gemm` returns INCONCLUSIVE, never FAIL, on UNCHARACTERIZED
configs (`src/assay/abft/check.py`).

Committed measurement files: **none**.
`data/noisefloor/index.json` lists `measured_gpu_models: []` and `runs: []`.
There is no `run-*.json` in the tree. Lookup for W02 bfloat16 4096 cubed
prints `n_samples=0`.

### GPUs used

None. This workstation had no NVIDIA GPU. Kaggle / Colab / HPC
characterization was never run. Budget spent: `$0` (`docs/BUDGET.md`).

### Sample counts

| population | n | source |
| --- | --- | --- |
| GPU characterization GEMMs (any model, any W01-W03 shape) | 0 | `data/noisefloor/` |
| CPU 16 x 16 x 16 clean `check_gemm` calls (pre-injection) | 45 | `docs/DETECTION_MATRIX.md` harness |
| CPU injected GEMMs in the detection matrix | 45 | same |
| of which exponent-bit class (`EXPONENT_HIGH` + `EXPONENT_LOW`) | 18 | 9 + 9 |

The 45 clean CPU calls are not production shapes. KT-1 asked for
production shapes (suite 512-8192). They are reported only so the
confidence interval has a real `n`, not an invented one.

### Detector configuration (v1, unchanged)

- Statistic: one-sided ABFT, `e = ones`, residual
  `|sum(C@e) - sum(A@(B@e))| / max(|sum(C@e)|, |sum(A@(B@e))|)` after
  promoting the two vectors to float64.
- Backend: PyTorch GEMV (the portable path). Triton was not used here.
- Decision: FAIL only if characterized and residual `>` every measured
  noisefloor sample. Uncharacterized -> INCONCLUSIVE.
- Threshold source: `data/noisefloor/` only. No `atol` / `rtol` in `src/`
  except IEEE `eps` in `ieee.py`.

### Injection methodology

`src/assay/inject/` is a test fixture. It is not imported by `assay run`
(enforced in `tests/test_inject.py`).

`flip_random(C, n_flips, bit_class, seed)` XOR-flips distinct
`(element, bit)` pairs in `C` after `C = A @ B`. A and B stay clean.
Bit classes: SIGN, EXPONENT_HIGH, EXPONENT_LOW, MANTISSA_HIGH,
MANTISSA_LOW. Seeds are PCG64 from `BASE_SEED`. Grid: W01 fp32, W02
bf16, W03 fp16; `n_flips` in `{1, 2, 4}`; shape 16 cubed; CPU.
`caught` means `CheckResult.status == FAIL`. INCONCLUSIVE is not a catch.
Full table: `docs/DETECTION_MATRIX.md`.

## 2. False-positive rate

A false positive is **FAIL on a GEMM with no injected fault**.
INCONCLUSIVE is not a false positive. PASS is not a false positive.

### Production-shape GPU assays

`n = 0` clean GPU GEMMs. The rate `k/n` is not defined. A confidence
interval is not defined. KT-1's "FPR < 1e-6 per GEMM at production
shapes" is **not demonstrated**.

### Supplementary: 45 CPU 16 cubed clean calls

`k = 0` FAIL, `n = 45`. Point estimate `k/n = 0`.

Clopper-Pearson 95% interval for a binomial proportion with `k = 0`
events in `n` trials is

```
[0, 1 - (alpha/2)**(1/n)]
```

with `alpha = 0.05` (two-sided 95%). This is the usual exact interval
when the count is zero; it uses only `n` and `alpha`, not a fitted
detector cutoff.

Computed from `n = 45`:

```
upper = 1 - (0.025)**(1/45) = 0x1.4260478469a00p-4   (about 0.0787)
```

One-sided 95% upper bound `1 - 0.05**(1/45) = 0x1.07cccaa26cd38p-4`
(about 0.0644).

Neither bound is below `1e-6`. To put a one-sided 95% Clopper-Pearson
upper bound under `1e-6` after zero FAILs would require

```
n >= ceil(log(0.05) / log(1 - 1e-6)) = 2995731
```

clean GEMMs. `min_samples = 100000` from noisefloor-v1 is smaller than
that, so even a completed characterization at the quantile spec would
not, by itself, have been a KT-1 FPR proof.

**FPR vs KT-1:** not shown to be `< 1e-6`.

## 3. True-positive rate by bit class

From `docs/DETECTION_MATRIX.md`. Detection = `caught` = FAIL.
`checksum_moved` is recorded beside it and is **not** the KT-1 TPR.

| bit_class | n | FAIL (caught) | TPR = caught/n | checksum_moved |
| --- | --- | --- | --- | --- |
| SIGN | 9 | 0 | 0/9 = 0 | 9/9 |
| EXPONENT_HIGH | 9 | 0 | 0/9 = 0 | 9/9 |
| EXPONENT_LOW | 9 | 0 | 0/9 = 0 | 9/9 |
| MANTISSA_HIGH | 9 | 0 | 0/9 = 0 | 9/9 |
| MANTISSA_LOW | 9 | 0 | 0/9 = 0 | 6/9 |
| exponent (HIGH+LOW) | 18 | 0 | 0/18 = 0 | 18/18 |
| all injected cells | 45 | 0 | 0/45 = 0 | 42/45 |

KT-1 asks for `>=90%` detection of **exponent-bit** flips. Combined
exponent TPR is `0/18 = 0`.

Clopper-Pearson 95% two-sided interval for that TPR, again `k = 0`,
`n = 18`:

```
upper = 1 - (0.025)**(1/18) = 0x1.7b7f99284ac58p-3   (about 0.185)
```

The interval is `[0, ~0.185]`. It does not contain `0.90`.
`P(0 FAIL in 18 | true TPR = 0.90) = 0.1**18`
(float64 print of that power: `1.000000000000001e-18`).
This configuration does not meet the 90% bar.

`checksum_moved` on exponent cells is 18/18. The ones-vector residual
usually changed, including `nan` on some EXPONENT_HIGH cells. That is
not a catch. v1 still returned INCONCLUSIVE because `n_samples = 0`.

## 4. Verdict

FAIL against KT-1 as written at CP-0.

| KT-1 clause (README, unchanged) | v1 result |
| --- | --- |
| FPR `< 1e-6` per GEMM at production shapes | not shown (`n = 0` GPU; CPU 16 cubed 95% CP upper `~0.0787`) |
| `>=90%` of exponent-bit flips detected | `0/18 = 0%` FAIL; 95% CP TPR upper `~0.185` |

No threshold in the detector was edited to improve these numbers.
noisefloor-v1 is unchanged. If a later detector is built, it is
**noisefloor-v2 / check_gemm v2**, and this file remains the v1 result.

## 5. Limitations

- **No GPU characterization.** Production shapes 512-8192, W01-W03, and
  every GPU model (T4, P100, A100, ...) are untested. There is no
  empirical p99.999 residual in this repository.
- **Shapes not in the injection grid.** 16 cubed CPU only.
  Rectangular suite shapes and 8192 were not injected.
- **Dtypes.** Matrix covers fp32, bf16, fp16 GEMM. int8 has an injector
  layout and no `check_gemm` path. fp64 GEMM is not a suite workload.
- **Workloads W04-W07.** No GEMM ABFT. SDPA, reductions, elementwise,
  decoder: not in KT-1's GEMM exponent-flip clause, and not measured as
  catches.
- **Mantissa-flip blind spot (measured, not a new cutoff).** 3 of 9
  MANTISSA_LOW cells kept the **same residual hex** as the clean GEMM:
  W01 fp32 `n_flips=1`, W02 bf16 `n_flips=1`, W02 bf16 `n_flips=2`
  (`docs/DETECTION_MATRIX.md`). Native-dtype `C @ e` rounded the ULP
  delta away. No residual magnitude was fitted as a "mantissa cutoff"
  and none will be back-ported into v1. Characterizing a GPU does not
  make a bitwise-identical residual FAIL.
- **INCONCLUSIVE vs FAIL.** v1 prefers a miss over a fabricated FAIL.
  That policy, plus zero `run-*.json`, makes exponent TPR as FAIL equal
  to zero even when the checksum moved. A v2 that FAILs on Inf without
  a floor would be a different detector; it is not this result.

## Reproduce

```bash
uv run assay characterize --lookup --workload W02 --dtype bfloat16 --m 4096 --k 4096 --n 4096
uv run pytest -m cpu tests/test_detection_matrix.py tests/test_inject.py tests/test_abft.py
```

Clopper-Pearson upper bound for zero events, two-sided 95%:

```bash
python -c "print((1 - (0.025)**(1/45)).hex()); print((1 - (0.025)**(1/18)).hex())"
```

Those two prints are the FPR (`n=45`) and exponent-TPR (`n=18`) upper
ends used above.
