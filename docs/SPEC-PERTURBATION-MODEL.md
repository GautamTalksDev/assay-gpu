# SPEC-PERTURBATION-MODEL

Version: **perturbation-v1**.

This is the perturbation model under which every published detection cell
in this repository was produced. The paper Method section points here.

## Definition

**Algorithm-level, post-GEMM, output-tensor bit flips.**

After `C = A @ B` completes, `flip_random` XOR-flips distinct
`(element, bit)` pairs inside `C` (the accumulated product tensor). The
residual is then computed with the original factors `A`, `B` and the
perturbed product.

| parameter | value |
| --- | --- |
| injection target | output tensor `C` only |
| bit classes | `SIGN`, `EXPONENT_HIGH`, `EXPONENT_LOW`, `MANTISSA_HIGH`, `MANTISSA_LOW` |
| `n_flips` | `{1, 2, 4}` |
| harness | `src/assay/inject/` (`flip_random`) |
| residual (current published cells) | residual-v3 unless a results doc states otherwise |

## What is NOT injected

Under perturbation-v1, the following are **never** flipped:

- operand **A**
- operand **B**
- accumulator intermediates inside the GEMM kernel
- instruction encoding
- control flow
- memory outside the output tensor `C`

There is therefore **no A-vs-B operand asymmetry** in any existing cell:
every recorded flip perturbs the residual numerator `|d − d'|` only.
Predictions that required operand-A vs operand-B contrast
(`docs/PREDICTIONS-MANTISSA-AB.md`) are untestable under this model;
see that file's OUTCOME section (falsified at precondition).

Operand-level injection is future work, not part of perturbation-v1.

## Results documents under this model

Every results document below was produced under perturbation-v1 (C-only
output flips), or under an earlier residual version of the same C-only
injection path. **No number in these documents is changed by this spec.**

| document | under perturbation-v1? |
| --- | --- |
| `docs/DETECTION_MATRIX.md` | yes (C after GEMM) |
| `docs/RESULTS-KT1.md` | yes (C after GEMM; null / early evaluation) |
| `docs/RESULTS-KT1-v2.md` | yes (C after GEMM; residual-v2; **VOID** as mechanism claim) |
| `docs/RESULTS-KT1-v3.md` | yes (C after GEMM; residual-v3) |
| `docs/RESULTS-KT2.md` | yes where flip cells appear; overhead cells are clean |
| `docs/RESULTS-KSCALE.md` | yes (C after GEMM; residual-v3) |
| `docs/RESULTS-FPR.md` | yes (re-score of C-flip matrix; no re-injection) |
| `docs/RESULTS-INJECTION-AUDIT.md` | yes (same injector path; cast-survival of flips in `C`) |

Published cells that a reader must not re-interpret as operand corruption
include, without changing their values: MANTISSA_LOW **0/1500** (per-K
clean maxima) and **0/600** (`threshold_gpd`); MANTISSA_HIGH **10/200**,
**17/200**, **24/200** at `threshold_gpd`; pooled exponent **304/400 = 76%**
at `threshold_gpd`.

## Relationship to arXiv:2601.19912

Chai et al. (arXiv:2601.19912) argue that algorithm-level fault injection
does not capture realistic GPU soft-error behaviour (instruction-level and
micro-architectural effects are invisible to output-tensor XOR models).

**That point is conceded.** This project's contribution is **detector
characterization under a defined perturbation model**, not fault
characterization in silicon. Fleet silent-corruption literature motivates
why a detector matters; it does not supply our measured rates, and our
rates do not estimate silicon fault incidence.
