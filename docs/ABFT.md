# GEMM checksum detector (one-sided ABFT)

This is the detector for `C = A @ B`. It does **not** recompute the GEMM.
It compares two quantities that are equal in exact arithmetic:

```
e = ones(N)
(A @ (B @ e))   versus   (C @ e)
```

They differ by floating-point rounding and by corruption. The pass/fail
cutoffs are **not** in this file. They are read from `data/noisefloor/`
(`noisefloor-v1`). If that configuration is missing or under-sampled, the
detector returns **INCONCLUSIVE**, never FAIL.

API: `assay.abft.check_gemm(A, B, C, config) -> CheckResult`.

## What we take from prior art (and what we do not claim)

Huang and Abraham introduced algorithm-based fault tolerance for matrix
operations in 1984 (IEEE Transactions on Computers, C-33(6)). One-sided
checksums with a vector of ones are standard, not an assay-gpu invention.
The papers below are the ones this checkpoint was asked to cite honestly.

### ATTNChecker (PPoPP 2025)

Liang, Li, Ren, Li, Fang, Chen. *ATTNChecker: Highly-Optimized Fault
Tolerant Attention for Large Language Model Training.* PPoPP 2025.
DOI: 10.1145/3710848.3710870

ATTNChecker is ABFT **for attention**, aimed at INF / NaN / near-INF during
**training**, with fused GPU kernels and correction, at about 7% training
overhead in their evaluation. We take the reminder that checksum ABFT can
be made cheap enough to run on the critical path, and that the interesting
faults for LLMs are not only bit-flips in GEMM C. We do **not** implement
attention ABFT, we do not correct errors, and we do not fuse checksums into
FlashAttention-style kernels.

### ReaLM (DAC 2025)

Xie, Zhao, Wan, Zhang, Wang, Wang, Huang, Li. *ReaLM: Reliable and
Efficient Large Language Model Inference with Statistical Algorithm-Based
Fault Tolerance.* DAC 2025. arXiv:2503.24053.
Code: https://github.com/PKU-SEC-Lab/ReaLM_DAC25

ReaLM is algorithm/circuit co-design for **LLM inference accelerators**.
It uses statistical ABFT and custom detection circuits so that small,
frequent errors that the model can absorb do not trigger recovery. We take
the one-sided checksum / matrix-sum family (`e` a vector of ones) and the
idea that a raw checksum mismatch is not automatically a product failure.
We do **not** build circuits, we do not tune recovery for perplexity, and
we do not treat "the model will be fine" as a pass. assay-gpu is a
hardware assay, not an accuracy-preserving approximate runtime.

### ApproxABFT

Xue, Liu, Min, Luo, Han. *ApproxABFT: Approximate Algorithm-Based Fault
Tolerance for Neural Network Processing.* arXiv:2302.10469.

ApproxABFT relaxes classical ABFT so that matrix-sum deviation (MSD) below
a threshold does not invoke recovery, because neural nets tolerate small
errors. Thresholds there are chosen to protect **model accuracy** with
low overhead. We take MSD-style comparison of checksums. We do **not**
copy their T1/T2 heuristics or their accuracy-driven threshold search.
Our threshold is an empirical quantile of residuals measured on
presumed-correct silicon, stored in `data/noisefloor/`. The goal is
"is this GPU's GEMM outside the characterized noise," not "will ImageNet
top-1 still look OK."

### Intensity-guided ABFT (SC 2021)

Kosaian and Vinayak. *Arithmetic-Intensity-Guided Fault Tolerance for
Neural Network Inference on GPUs.* SC '21. DOI: 10.1145/3458817.3476184.
Code: https://github.com/Thesys-lab/arithmetic-intensity-guided-abft

They pick **global** vs **thread-level** ABFT per layer from arithmetic
intensity and the GPU's compute-to-memory ratio, implemented on CUTLASS,
to keep overhead low on mixed CNN/DLRM workloads. We take the fact that
checksum cost is a product constraint (see Kill Test 2 in `README.md`).
We do **not** implement thread-level fused ABFT, we do not switch schemes
per layer, and we are vendor-neutral at the API (PyTorch first, Triton
reduction second). CUTLASS-only fusion would fight that.

### What is different here

Not novelty of the checksum algebra. The delta we actually have:

1. **Productization of the threshold.** Detection bounds come from
   versioned `run-*.json` plus `methodology-v1.json`. No guessed `atol`.
2. **Vendor neutrality.** The portable path is PyTorch GEMV (`C @ e` and
   `A @ (B @ e)`). Triton is an optional faster reduction, not an NVIDIA
   library requirement at the API.
3. **Attestation-shaped verdicts.** Every `CheckResult` carries the
   residual, the threshold used, the noisefloor spec id, and the sample
   count. Uncharacterized configs are INCONCLUSIVE. That is for a later
   signed report, not for silently "fixing" a training step.
4. **We do not correct.** Detection only. Recompute-or-fail is a policy
   on top of this module, not inside it.

If a later paper does the same ones-vector check with a measured noise
floor, they are in the same family. Do not cite assay-gpu as the origin
of ABFT.

## The correlated-fault critique (why the checksum is a separate kernel)

The standard objection to ABFT: if the checksum is computed on the **same
execution units** as the GEMM, a fault can corrupt `C` and the checksum
**identically**, and the check passes. Fusing the checksum into the GEMM
that produced `C` makes that window large (same accumulators, same tensor
core instruction, same threadblock).

What this implementation does:

- The production GEMM `C = A @ B` is a separate launch (cuBLAS / PyTorch
  matmul). This detector never fuses checksum math into that kernel.
- `B @ e` and `C @ e` are a ones-vector reduction: PyTorch GEMV, or a
  Triton row-sum kernel compiled and launched on its own grid.
- `A @ (B @ e)` is a GEMV, not the original GEMM. Different shape, different
  kernel, different accumulation tree.
- `checksum_on_cpu=True` copies A, B, C and runs the PyTorch checksum on
  the host. That is the stronger isolation option for attestation. It does
  not protect against a fault that already wrote a consistent-but-wrong
  `C` into memory **and** a matching checksum; it does stop a GPU ALU
  fault from hitting both the GEMM and the check in one fused instruction.

What we still do not have:

- We do not pin the checksum to a different SM, a different device, or a
  different vendor library by default.
- A **persistent** SM or DRAM fault can still touch the GEMM at time t
  and the checksum at time t+1. Whole-chip, driver, and interconnect
  faults are outside this check.
- Weighted checksums (`e = [1, 2, 4, ...]`, Huang-Abraham) catch more
  structured error patterns than `e = ones`. This checkpoint uses ones,
  as specified.
- This is **not** a proof that a passing checksum means the GEMM is
  correct. It is a cheap filter whose false-FAIL rule is conservative.

The honest summary: we avoided the **fused same-instruction** failure
mode. We did not solve correlated faults in general. Prefer a miss over
a fabricated FAIL; that is why uncharacterized hardware cannot FAIL.

## Residual and the ambiguous band

Let `ce = C @ e` and `abe = A @ (B @ e)`. The decision residual is
**residual-v2** (`docs/RESIDUAL.md`):

```
|sum(ce) - sum(abe)| / (eᵀ |A| |B| e)
```

after promoting the checksum vectors to float64 and accumulating the
absolute-factor scale in float64. residual-v1 divided by the signed grand
sum and is void. Every `CheckResult` carries `residual_version`.

On a **characterized** `(gpu_model, workload, dtype, shape)`:

| residual R | verdict |
| --- | --- |
| R <= p-quantile of measured residuals | PASS |
| p-quantile < R <= sample max | INCONCLUSIVE (ambiguous band) |
| R > every measured sample | FAIL |
| R non-finite | FAIL |

The p-quantile and the sample max are both stored in the lookup from
`data/noisefloor/`. Neither is a constant in `src/`.

On **uncharacterized** configs (no `run-*.json`, `n < min_samples`, or
multiple GPU models pooled): **INCONCLUSIVE**, including when `C` is
obviously garbage. `min_samples` is `ceil(1/(1-p))` for
`p = 99999/100000` from `methodology-v1.json` (100000). As of 2026-08-18
the repo still has **zero** GPU runs, so every real `check_gemm` against
`data/noisefloor/` is INCONCLUSIVE.

`CheckResult` always includes: status, residual (decimal and hex),
threshold used (the quantile, or null), sample max, `noisefloor_spec_version`,
`n_samples`, `min_samples`.

## PyTorch path and Triton path

1. PyTorch (default, portable): `B @ e`, `A @ (B @ e)`, `C @ e` with
   `e = ones`. CPU and CUDA.
2. Triton: the ones-reduction (`B @ e` and `C @ e`) is a row-sum kernel
   with float64 accumulation, then a cast back to the tensor dtype.
   `A @ (B @ e)` stays a PyTorch GEMV. CUDA only. Triton is imported
   lazily; it is **not** added as a direct `pyproject.toml` dependency
   (the CUDA torch wheel already ships it). CPU CI does not launch it.

Tests marked `gpu` assert the two paths agree bitwise on ones-matrices
across `WORKLOAD_GEMM_SHAPES`, and that `check_gemm` residual hex matches
on an exact integer GEMM. Those tests have not been executed on this
workstation (no NVIDIA GPU).

## Checksum-only overhead (measured)

Command (no default for repeats):

```bash
uv run assay abft overhead --repeats N --m M --k K --n N --backend pytorch --device cpu
# CUDA, when a GPU exists:
uv run assay abft overhead --repeats N --m M --k K --n N --dtype bfloat16 --backend pytorch --device cuda
uv run assay abft overhead --repeats N --m M --k K --n N --backend triton --device cuda
```

This times `A @ B` against the ones-sided checksum and against
residual-v2 `eᵀ |A| |B| e`. It is **not** Kill Test 2 (end-to-end
transformer inference). Do not copy a CPU ratio onto a GPU.

### CPU (development workstation)

Measured 2026-08-18 on WSL2, no NVIDIA GPU, PyTorch 2.13.0+cu130 CPU
execution of the CUDA wheel, float32 ones GEMM:

| shape (M,K,N) | device | backend | repeats | gemm_s | checksum_s | checksum/gemm |
| --- | --- | --- | --- | --- | --- | --- |
| 32, 32, 32 | cpu | pytorch | 20 | 0.00000291 | 0.00000908 | 3.12063071 |
| 128, 128, 128 | cpu | pytorch | 20 | 0.00001554 | 0.00001021 | 0.65741694 |
| 512, 512, 512 | cpu | pytorch | 30 | 0.00085154 | 0.00004946 | 0.05808443 |

Read the 512 column as: on this CPU, the checksum path was about 5.8% of
the GEMM time. Small shapes are dominated by call overhead. That is a
measurement, not a spec.

### Tesla T4 (residual-v2, bfloat16, PyTorch)

Measured on a Tesla T4, residual-v2, bfloat16, PyTorch backend,
repeats = 8. Ratios are wall time over the GEMM. Combined is checksum
plus normalizer, each timed separately against the same GEMM.

| shape | checksum / GEMM | normalizer / GEMM | combined / GEMM |
| --- | --- | --- | --- |
| 512³ | 57.0% | 93.9% | 151% |
| 1024³ | 9.5% | 24.6% | 34.1% |
| 2048³ | 2.2% | 9.6% | 11.8% |
| 4096³ | 1.0% | 6.8% | 7.8% |

The combined figure is below 10% of GEMM only at 4096³ (7.8%) and would
be smaller at larger N. At 2048³ it is 11.8%. The normalizer costs about
3–7× the checksum at 1024³ and above and dominates the total. Overhead
falls roughly as 1/N: both checks are `O(M K + K N)` memory passes
against an `O(M K N)` GEMM.

That 10% comparison is this GEMM-only ratio on this GPU and dtype. It
is not Kill Test 2. KT-2 is e2e transformer inference. Its bar is
unchanged. residual-v2 has no noisefloor data yet. Triton: still
**UNMEASURED**.

## What this checkpoint does not do

- W04-W07 (SDPA, reductions, elementwise, decoder): no checksum here.
- Online correction or training-loop recovery.
- Thread-level / fused CUTLASS ABFT.
- A FAIL on any GPU in this repository today (no characterized floor).
- KT-1 (separation of bit-flips from noise) or KT-2 (10% e2e overhead).
  Those remain open until noisefloor data and a transformer-sized
  measurement exist.
