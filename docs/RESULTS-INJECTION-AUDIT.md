# RESULTS-INJECTION-AUDIT

**PASS.** Injected mantissa flips survive the working-dtype cast: deltas are
nonzero and `n_elements_bitwise_equal` is 0. MANTISSA_LOW remains a hard
negative: 0/1500 across the K-scaling study at per-K observed clean maxima,
and 0/600 at `threshold_gpd` (K = 4096) — not a discarded perturbation.

## Method

Hypothesis under test (CP-MA): MANTISSA_LOW hard-negative counts (0/1500 at
per-K clean maxima; 0/600 at `threshold_gpd`, K = 4096) with identical ratios
at `n_flips` ∈ {1, 2, 4} is the signature of a flip discarded by bf16 rounding
before the GEMM, not a real miss.

Measurement: `flip_random(..., verify=True)` from CP-MA. After the XOR and the
cast back to the working dtype, each perturbed element is compared as a raw
bit pattern (uint16 for bf16). Recorded fields:

| field | meaning |
| --- | --- |
| `n_elements_flipped` | unique elements targeted |
| `n_elements_bitwise_equal` | targeted elements identical post-cast |
| `achieved_rel_delta_max` | max \|new−old\|/\|old\| (0 if old == 0) |
| `achieved_rel_delta_median` | median of those relative deltas |

Grid: W02 layout, bf16, shape 4096×4096 (C-shaped), `n_flips=1`, `K=4096`,
n = 50 per class. Seeds: `_cell_seed` from `sweep_v3_flips` (same mixer as the
flip matrix). Control: MANTISSA_HIGH (known nonzero detection in K-scale).
Test: MANTISSA_LOW.

Host had no CUDA (`torch.cuda.device_count() == 0`), so this run is
injection-only: same injector and seeds as `assay characterize --sweep-v3-flips
--verify-injection`, no GEMM and no residual. Bit-survival is decided entirely
inside the injector cast path.

Source JSONL: `data/noisefloor/pilot/injection-audit.jsonl`.

## Numbers

### MANTISSA_HIGH (control)

| metric | value |
| --- | --- |
| n | 50 |
| elements targeted | 50 |
| `n_elements_bitwise_equal` | **0 / 50** |
| samples with `achieved_rel_delta_max == 0` | **0 / 50** |
| `achieved_rel_delta_max` min / median / max | 0.03175 / 0.09467 / 0.4923 |

Control fires: nonzero deltas, zero bitwise-equal. Harness is not broken.

### MANTISSA_LOW (test)

| metric | value |
| --- | --- |
| n | 50 |
| elements targeted | 50 |
| `n_elements_bitwise_equal` | **0 / 50** |
| samples with `achieved_rel_delta_max == 0` | **0 / 50** |
| `achieved_rel_delta_max` min / median / max | 0.004065 / 0.01258 / 0.03125 |

Flips stick. Relative deltas are smaller than MANTISSA_HIGH (as expected for
lower mantissa bits) but strictly positive on every sample.

## Verdict

**PASS.** MANTISSA_HIGH control shows nonzero deltas and
`n_elements_bitwise_equal = 0`. MANTISSA_LOW shows the same pattern. The
injected flip is not discarded by bf16 rounding in the injector. MANTISSA_LOW
remains a real hard negative: 0/600 at `threshold_gpd` (K = 4096), and
0/1500 across the K-scaling study scored at per-K observed clean maxima.
