# Workload suite

Every workload is seeded and takes **zero free parameters** at call time.
This checkpoint **does not** compare GPU outputs to the fp64 goldens.

Inputs are drawn with `numpy.random.Generator(PCG64(seed))` from
`assay.reference.arrays`. Goldens live in `data/reference/` as `.npz` plus
`manifest.json`. The SHA-256 in the manifest is of the **array**
(`dtype.str|shape` + NUL + C-contiguous `tobytes()`), not the ZIP wrapper
(ZIP timestamps are not reproducible).

NumPy is pinned to **2.4.6** so Python 3.11 and 3.12 regenerate the same
hashes. Regenerating on another machine with that pin must match
`data/reference/manifest.json`.

## How to run

```bash
uv run assay reference generate
uv run pytest -m cpu
# GPU machine:
uv run assay run --double
uv run pytest -m gpu
```

`assay run --double` writes `data/workload/double_run.json`. That file is a
measurement: `bitwise_identical: false` is input to CP-3, not a test failure.

## W01 — GEMM fp32

- Shapes: square 512, 1024, 2048, 4096, 8192 and rectangular triples in
  `WORKLOAD_GEMM_SHAPES`.
- `torch.matmul` on `float32`.
- TF32 disabled (`allow_tf32=False`, `float32_matmul_precision="highest"`).
- Determinism: CUDA matmul is deterministic when
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` and
  `torch.use_deterministic_algorithms(True)` succeed. If torch raises, the
  reason is stored on the result and the op is rerun with the flag off.

## W02 — GEMM bf16 inputs, fp32 accumulate

- Same shapes as W01, `bfloat16` tensors.
- `allow_bf16_reduced_precision_reduction=False` so the accumulate stays
  fp32 as far as torch exposes that switch.
- Same determinism request as W01.

## W03 — GEMM fp16, fp16 accumulate (fragile)

- Same shapes, `float16` tensors.
- `allow_fp16_reduced_precision_reduction=True` (the fragile reduced-precision
  path). Torch does not always expose the actual ALU accumulate dtype; the
  flag is recorded in `backend`.
- Same determinism request as W01.

## W04 — Scaled dot-product attention

- Cases: `WORKLOAD_SDPA_SHAPES` `(batch, heads, seq, head_dim)`.
- `scaled_dot_product_attention(..., dropout_p=0.0, is_causal=False)`.
- Kernel: whatever torch reports via `flash_sdp_enabled`,
  `mem_efficient_sdp_enabled`, `math_sdp_enabled`. We do **not** silently
  force the math backend to look deterministic.

### Cannot be made deterministic (torch)

FlashAttention and memory-efficient SDPA are documented by PyTorch as not
guaranteeing bitwise determinism even with `use_deterministic_algorithms`.
If the flag raises at runtime, we record the exception text. The math
backend can be deterministic; we do not switch to it automatically because
that would hide the production kernel.

## W05 — Large reductions (sum, mean)

- Lengths `2**20` and `2**24`, fp32.
- `torch.sum` / `torch.mean`.
- Determinism: with `use_deterministic_algorithms(True)`, torch is supposed
  to use a deterministic reduction path. Without the flag, CUDA reductions
  may use atomics and disagree across runs. We request the flag and record
  failure instead of dropping the workload.

## W06 — Elementwise exp, tanh, rsqrt

- Shape `(4096, 4096)`.
- `exp`/`tanh` on `uniform_-1_1`; `rsqrt` on `uniform_1_4` (positive domain).
- Elementwise kernels are expected to be deterministic. If torch disagrees,
  that is recorded.

## W07 — Short decoder, open weights, greedy decode

- Architecture: vocab 128, d_model 64, 4 heads, 2 layers, ff 128, prompt 8,
  8 greedy steps. RMSNorm eps is `numpy.finfo(float32).eps` (IEEE-754), not
  a guessed `1e-5`.
- Weights are generated from `BASE_SEED` and stored in
  `data/reference/w07_weights.npz`. **No download, no Hugging Face, no
  network.** "Open weights" means they are in this repository.
- Attention: causal SDPA. GELU: erf (`approximate="none"`). Decode: argmax.
- Inherits SDPA determinism limits from W04 for the attention sub-op.

## fp64 CPU goldens (`src/assay/reference/`)

GEMM, SDPA, reductions, and the W07 forward are also computed in float64 on
CPU with an explicit K-loop (`multiply` then `add`, no BLAS `gemm`) so the
golden does not depend on OpenBLAS vs MKL. Those goldens are **not compared**
to GPU outputs in this checkpoint.

## Kernel names

`WorkloadResult.kernel` is filled only from torch APIs
(`preferred_blas_library`, SDPA enablement flags, or `torch.<op>`). If torch
does not expose the CUDA kernel symbol, the field is `None` or a string
stating that the API is absent. We do not parse `nvidia-smi` or nvprof
output to invent a name.

## Double-run record

`data/workload/double_run.json` (created on a GPU machine) fields:

- `all_bitwise_identical`
- per case: `bitwise_identical`, `deterministic_run1/2`,
  `nondeterminism_reason`, `kernel`

Do not assume two runs matched. Read the file.

## Docker vs CUDA torch

PyPI `torch` 2.13 Linux wheels include CUDA 13 libraries (multiple
hundreds of MB). Combined with `nvidia/cuda:12.4.1-runtime-ubuntu22.04`
(~2.3 GB) the image can exceed the CP-1 3 GB cap. Workloads must still be
run with a CUDA build of torch on a real GPU (`uv sync` on the instance,
then `uv run assay run --double`). Do not pretend a CPU-only torch run is
a GPU result.

