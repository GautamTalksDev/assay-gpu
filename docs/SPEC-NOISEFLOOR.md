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

`--repeats N` is N independent `(A, B)` draws (`sample_index` in the
seed). Same-input launches of sample 0 are `bitwise_stable` only.

A tolerance at quantile `99999/100000` is defined only for
`N >= min_samples`. That quantile is **UNJUSTIFIED** until the pilot is
read; `methodology-v1.json` is unchanged. Smaller N still writes
`run-*.json` but lookup stays INCONCLUSIVE. `--pilot` writes
`data/noisefloor/pilot/` and is never lookup input.

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

Recorded for each `(workload, shape, dtype, blas_library, sample_index)`:

| Field | Definition |
| --- | --- |
| residual_version | `residual-v2`. Lookup ignores `residual-v1` and unversioned samples. |
| abft_residual_normalized | residual-v2: \|sum(C@e) - sum(A@(B@e))\| / (eᵀ \|A\| \|B\| e). Same residual `check_gemm` uses. See `docs/RESIDUAL.md`. |
| backend | `pytorch` or `triton` (checksum reduction). Recorded so a floor cannot silently mix backends. |
| sample_index | Independent `(A, B)` draw index. Not a same-input repeat. |
| result_sha256 | SHA-256 of the GPU tensor in its native dtype (bitwise identity) |
| telemetry_before/after | nvidia-smi `clocks.sm`, `clocks.mem`, `power.draw`, `temperature.gpu` (null if the query fails) |
| blas_library | `torch.backends.cuda.preferred_blas_library` value actually selected |

A and B come from `gemm_numpy_pair(..., sample_index=..., workload_id=...)`.
Per-sample fp64 CPU `matmul_fp64` is not recorded (it would dominate wall
time at 4096 and is not the detector residual).

## Measured bitwise determinism (Tesla T4 only)

Source: `assay run --double` on a Tesla T4, driver **580.159.04**, CUDA
**13.0**, torch **2.13.0+cu130**. **44 of 44** suite cases reported
`all_bitwise_identical: true`: every W01–W03 GEMM shape in fp32 / bf16 /
fp16, all W04 SDPA shapes, W05 reductions, W06 transcendentals, W07
transformer. cuBLAS on sm75 picked a stable kernel every time.

That is a measured property of **this** (GPU model, driver, CUDA, torch)
tuple. It falsifies, on this hardware, the assumption that cuBLAS kernel
selection is not deterministic across runs. It does **not** generalize to
other GPU models, drivers, or torch builds until those are measured the
same way. Do not copy this sentence onto an A100.

## Sample population (implemented)

One sample is one independent `(A, B)` draw from `uniform_-1_1`. The
seed is a deterministic mix of `BASE_SEED`, workload id, shape index,
and **sample index** (`sample_factor_seed` in
`src/assay/reference/spec.py`). No wall-clock, no `hash()`.

`--repeats N` is N independent draws. Same-input launches (two GEMMs
of sample 0) populate `bitwise_stable` only. They are not quantile
inputs.

The ABFT residual recorded in `run-*.json` is residual-v2
(`docs/RESIDUAL.md`): `|sum(C@e)-sum(A@(B@e))| / (eᵀ |A| |B| e)`,
with `residual_version` on every sample. Lookup ignores residual-v1.
`backend` is recorded on every sample (PyTorch GEMV in this version).

## p99.999 is UNJUSTIFIED

`target_quantile = 99999/100000` and `min_samples = ceil(1/(1-p)) =
100000` were chosen at spec time **before any residual distribution
existed**. That arithmetic is the count needed to observe at least one
sample beyond p99.999; it is not a stable estimate of the tail, and it
was never reconciled with KT-1's FPR `< 1e-6` (p99.999 is 1e-5, ten
times looser). Do not change the quantile in
`methodology-v1.json` until the pilot is read.

Pilot (not a characterization): `assay characterize --pilot` writes
`data/noisefloor/pilot/`. Lookup never reads that directory.

## Open: checksum backend is not in the noisefloor key

`ones_matvec_pytorch` is `matrix @ ones` in the **matrix dtype** (fp32
GEMV / cuBLAS). `ones_matvec_triton` accumulates in **float64** then
casts back. On integer-valued fp32 inputs at 2048×2048, row sums are
~8.5e9, above the 2^24 exact-integer range, so the two backends
disagree while both being "correct" for their accumulation type.

`check_gemm` defaults to PyTorch but can be pointed at Triton. A
threshold calibrated with one backend does not transfer to the other.
Either force fp64 accumulation on both paths, or make `backend` part of
the noisefloor lookup key. Not implemented here; sampling gates it.

## Aggregates

Per `(gpu_model, workload, dtype, shape, blas_library)`:

- n
- bitwise_stable: two launches of **sample 0** (same A, B) have identical
  `result_sha256`. Independent draws are not expected to match.
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
