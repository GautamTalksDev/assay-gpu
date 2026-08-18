# noisefloor-v1

Version: **noisefloor-v1**. Methodology file:
`data/noisefloor/methodology-v1.json`.

This document is the reproduction spec for empirical GPU noise-floor
measurement. It does **not** contain a residual magnitude. Magnitudes live
only in `data/noisefloor/**/run-*.json` written by `assay characterize`.

As of 2026-08-18 this repository contains **zero** `run-*.json` files.
`data/noisefloor/index.json` lists `measured_gpu_models: []`.

## What to state out loud (bf16 GEMM 4096 x 4096)

Command:

```bash
uv run assay characterize --lookup --workload W02 --dtype bfloat16 --m 4096 --k 4096 --n 4096
```

**Today:** verdict `INCONCLUSIVE`. status `uncharacterized`. n_samples `0`.
min_samples is `ceil(1/(1-p))` for `p = 99999/100000` from
`methodology-v1.json` (that is 100000). p99.999 residual: **not defined**.
How many samples produced it: **none**. Do not invent a number.

After a real GPU run with `--repeats` at least min_samples, the same command
prints `p_quantile_residual_hex` and `n_samples` from that GPU's files only.

## Goal

On hardware we have no independent reason to believe is faulty, measure the
distribution of deviation between GPU GEMM outputs and a defined-order
float64 CPU reference, plus the ABFT checksum residual, run-to-run bitwise
stability, cuBLAS library variation (where torch exposes it), and
clocks/power/temperature.

Every detector threshold in this project must be **read from those files**.
If a `(gpu_model, workload, dtype, shape)` key is missing or has
`n < min_samples`, the configuration is **UNCHARACTERIZED** and an assay
against it returns **INCONCLUSIVE**. Never pool GPUs. Never copy a T4
number onto an A100.

## Reproduce a measurement

Needs: NVIDIA GPU, CUDA torch, `nvidia-smi` on PATH, this repo, `uv sync`.

```bash
uv run assay characterize --repeats N --device 0
# optional filters:
uv run assay characterize --repeats N --workload W02 --m 4096 --k 4096 --n 4096
```

`--repeats` has **no default**. N is chosen by the operator. A tolerance at
quantile `99999/100000` is defined only for `N >= min_samples` computed from
that fraction. Smaller N still writes `run-*.json` (raw measurements) but
lookup stays INCONCLUSIVE.

Output path:

`data/noisefloor/<gpu_model>/cuda-<cuda>_driver-<driver>_tool-<tool>/run-<utc>-<digest>.json`

Keys: GPU model (from nvidia-smi name, else torch name), driver, CUDA
version, `assay-gpu` version.

`--include-large` adds shapes whose max dimension is greater than 4096
(the 8192 GEMMs). Default coverage is W01-W03 shapes with max dim <= 4096.

Kaggle dual-T4: run `--device 0` and `--device 1` separately. Same model,
two files. P100 and A100 are additional **models** for the three-model bar.

Hold Thunder Compute A100 credit for this characterization, as CP-1
documented. Log any signup credit in `docs/BUDGET.md`.

## Per-sample quantities

Recorded for each `(workload, shape, dtype, blas_library, repeat)`:

| Field | Definition |
| --- | --- |
| max_abs_error | max_i \|C_gpu[i] - C_fp64[i]\| after promoting C_gpu to float64 (fp16/bf16 finite values are exact in fp64) |
| max_rel_error | max \|C_gpu-C_fp64\|/\|C_fp64\| over elements where C_fp64 != +0. If every reference element is +0: 0 iff tensors equal, else inf |
| abft_residual_abs | \|sum(C_gpu in fp64) - checksum(A,B)\| |
| abft_residual_normalized | abs residual / max(\|checksum\|, \|sum(C)\|). If both sums are +0: 0 iff abs residual is 0, else inf |
| result_sha256 | SHA-256 of the GPU tensor in its native dtype (bitwise identity) |
| telemetry_before/after | nvidia-smi `clocks.sm`, `clocks.mem`, `power.draw`, `temperature.gpu` (null if the query fails) |
| blas_library | `torch.backends.cuda.preferred_blas_library` value actually selected |

Checksum (GEMM only):

```
checksum = sum_k (sum_i A[i,k]) * (sum_j B[k,j])
```

A and B are the seeded float64 factors from `gemm_numpy_pair`. Axis sums use
`numpy.sum(..., dtype=float64)`. The k-combination is a Python loop of
`multiply` then `add`. `sum(C)` uses `numpy.sum` on the promoted GPU result.

C_fp64 is `matmul_fp64` from `src/assay/reference/compute.py` (K-loop
multiply-then-add, no BLAS GEMM), computed once per shape and reused for
every repeat.

## Aggregates

Per `(gpu_model, workload, dtype, shape, blas_library)`:

- n
- bitwise_stable: all `result_sha256` identical
- run_to_run_max_abs_delta: max abs difference vs the first repeat, compared
  in float32 (magnitude of variation when bits differ)
- sample maxima of abs error, rel error, ABFT normalized residual

The p99.999 **tolerance** is not stored as a guessed constant. Lookup
computes the empirical quantile (inverse ECDF, order statistic at
`ceil(p*n)-1`) of `abft_residual_normalized` **only when** `n >= min_samples`.

## Quantile math

`p = 99999/100000` is read from `methodology-v1.json`.
`min_samples = ceil(1/(1-p))` is computed from that fraction in
`Methodology.min_samples`. It is not a residual cutoff.

## What is NOT covered (noisefloor-v1)

- Any GPU model, driver, CUDA version, or tool version without a `run-*.json`
- Pooling or transferring tolerances across GPU models
- cuBLASLt heuristic algorithm IDs, kernel autotune indices, and any
  selection torch does not expose via `preferred_blas_library`
- TF32 (the suite disables it; TF32 error is not this floor)
- W04 SDPA, W05 reductions, W06 elementwise, W07 decoder: no ABFT checksum
  in this version; do not use GEMM residuals as a proxy
- Shapes with max dimension > 4096 unless `--include-large` was used and a
  file exists
- Faulty-device behavior (this spec measures presumed-correct silicon)
- Multi-GPU collectives
- Integer and fp8 dtypes
- Windows / non-NVIDIA devices

## Three-model requirement

Definition of Done for this checkpoint requires measurements from at least
three distinct GPU **models**. The intended $0 set is Kaggle T4, Kaggle
P100, and student-credit A100. Until `index.json` lists three model keys,
the project has **not** met that bar. Empty index is honest. Fabricated
JSON is a bug.

## Assay rule (binding)

Code in `src/assay/noise/lookup.py`:

- No measurement file → `UNCHARACTERIZED` → verdict `INCONCLUSIVE`
- `n < min_samples` → `UNCHARACTERIZED` → `INCONCLUSIVE`
- More than one GPU model matched without `--gpu-model` → `INCONCLUSIVE`
  (refuses to pool)
- Otherwise the tolerance **is** the empirical quantile of the measured
  ABFT normalized residuals in those files

A hardcoded numerical tolerance anywhere else in `src/` is a bug.
