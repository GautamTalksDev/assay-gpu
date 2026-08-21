# PREDICTIONS-MANTISSA-AB

Pre-registration for the A-vs-B mantissa split. Written before any A-vs-B
measurement exists. This document contains no measurement code and no data.
Locked 2026-08-21. Ancestor commit for the future measurement; nothing in
this file may be revised after that measurement is run.

## 1. BACKGROUND

Observation being explained: under residual-v3, MANTISSA_LOW is a hard
negative on two distinct threshold bases that must never be merged into one
number:

| count | threshold basis | source |
| --- | --- | --- |
| **0/1500** | per-K observed clean maxima | K-scaling study (`docs/RESULTS-KSCALE.md`) |
| **0/600** | `threshold_gpd = 3.144536e-06` at K = 4096 | FPR re-score (`docs/RESULTS-FPR.md`) |

Suspicion that bf16 rounding was discarding the injected flips before the
GEMM was cleared by `docs/RESULTS-INJECTION-AUDIT.md`: injected flips
demonstrably survive the cast (0/50 bitwise-equal, 0/50 with
`achieved_rel_delta_max == 0`) in both the MANTISSA_HIGH control and
MANTISSA_LOW. The floor is a real miss of a real perturbation, not an
injector artifact.

The current flip cells pool operand choice. They do not separate flips in
A from flips in B. That pooling is what this experiment breaks.

## 2. MECHANISM UNDER TEST

Residual-v3 per-row statistic (unchanged):

```
r_i = |d_i - d'_i| / scale_i
    d      = C @ e
    d'     = A @ (B @ e)
    scale  = |A_i| |B| e     (fp64)
```

- A flip in operand **A** perturbs both the numerator `|d - d'|` and the
  denominator `scale`. The two perturbations may partially or fully
  self-cancel in the ratio, leaving `r_max` at or below the clean floor.
- A flip in operand **B** perturbs the numerator only (relative to the
  self-cancellation path above). Detectability should therefore be higher
  for B than for A at the same bit class and flip count.

Therefore detectability should depend on which operand was flipped. The
pooled MANTISSA cells cannot test this. The A-vs-B split can.

## 3. PREDICTIONS

Primary evaluation cell unless noted: W02, bf16, K = 4096, T4, residual-v3,
`n_flips = 1`, `threshold_gpd = 3.144536e-06`, n = 200 independent draws per
`(bit_class, operand)` cell. Detection = `r_max > threshold_gpd`.

### P1 — MANTISSA_LOW A-flips stay at the floor

**Claim.** MANTISSA_LOW flips in operand A detect at **0/200**, or within
noise of zero: at most **1/200** (≤ 0.5%).

**Falsifier.** MANTISSA_LOW A-flip detections ≥ 2/200 at `threshold_gpd`.

### P2 — MANTISSA_LOW B-flips exceed A-flips by a locked margin

**Claim.** MANTISSA_LOW flips in operand B detect at a rate strictly greater
than A-flips, with minimum separation

```
rate_B - rate_A  ≥  0.05
```

(5 percentage points). At n = 200 this means at least 10 more detections in
the B cell than in the A cell (e.g. A = 0/200 and B ≥ 10/200).

**Falsifier.** `rate_B - rate_A < 0.05` at n = 200, including the case
`rate_B ≤ rate_A`.

### P3 — MANTISSA_HIGH shows the same directional split, higher absolute rate

**Claim.** MANTISSA_HIGH exhibits the same directional asymmetry
(`rate_B - rate_A ≥ 0.05` at n = 200, `n_flips = 1`), and both operand
rates are strictly higher than the corresponding MANTISSA_LOW rates:

```
rate_A(HIGH) > rate_A(LOW)
rate_B(HIGH) > rate_B(LOW)
```

This confirms the mechanism is normalizer self-cancellation, not a
MANTISSA_LOW-specific artifact.

**Falsifier.** Either the HIGH A-vs-B gap is `< 0.05`, or either HIGH
operand rate fails to exceed its LOW counterpart.

### P4 — Shared 0.5247 minimum is an A-flip clean-return signature

**Claim.** The shared minimum ratio 0.5247 across MANTISSA cells
(`docs/RESULTS-KT1-v3.md`) is the clean `r_max` for fixed seeds whose
perturbation never cleared the floor. Under the A-vs-B split, samples whose
flipped `r_max` is bitwise-identical to the paired clean `r_max` for the
same seed (sub-floor / null perturbation) appear predominantly in the
A-flip subset:

```
among MANTISSA_LOW samples with flipped r_max == clean r_max (same seed):
  fraction that are A-flips  ≥  0.80
```

**Falsifier.** That A-flip share is `< 0.80`, or such unchanged-`r_max`
events are majority-B / evenly split across operands.

## 4. THE NULL RESULT AND HOW IT WILL BE REPORTED

If **both** MANTISSA_LOW A-flips and MANTISSA_LOW B-flips are 0/n (or both
within noise of zero such that P2's separation cannot fire), the mechanism
hypothesis in §2 is **falsified**. The MANTISSA_LOW floor is then genuinely
unexplained by operand-dependent normalizer self-cancellation.

In that case the paper will state exactly that: the A-vs-B split was
pre-registered, run, and returned a null; the floor remains unexplained.
The experiment will not be omitted, re-thresholded, or re-defined after the
fact to rescue the mechanism story.

## 5. LOCKED PARAMETERS

| parameter | value |
| --- | --- |
| `threshold_gpd` | `3.144536e-06` (canonical, from the GPD fit) |
| residual | residual-v3, unchanged |
| K | 4096 |
| dtype | bf16 |
| GPU | T4 |
| workload | W02 |
| n per `(bit_class, operand)` cell | 200 |
| primary `n_flips` | 1 |

No threshold, formula, sample size, separation, or bar in this document may
be changed after the measurement is run. New data may falsify these
predictions; it may not rewrite them.

## OUTCOME — FALSIFIED AT PRECONDITION

Recorded 2026-08-21 (CP-INJ-1). Nothing above this section was edited.
The locked P2 separation (`rate_B - rate_A ≥ 0.05` at n = 200, `n_flips = 1`,
`threshold_gpd`) remains exactly as written.

The CP-AB2 recoverability check established that injection in this project
is **C-only**: flips are applied to the accumulated GEMM output, not to
operand A or operand B. Flip locations are discarded and never written to
JSONL; there is no operand field to recover. See
`docs/SPEC-PERTURBATION-MODEL.md` (perturbation-v1).

The predicted mechanism required an A-vs-B contrast in the residual
normalizer `|d − d'| / (|A_i||B|e)`. A flip in C perturbs the numerator
only, unconditionally. **P1–P4 are therefore untestable with the current
harness, not merely unmeasured.**

No measurement was run, no rates were computed, and **no PASS/FAIL is
claimed for P1–P4**. The predictions are falsified at their precondition.
This is the fourth registered prediction in this project to be scored as
written rather than rewritten.

The shared **0.5247** minimum across MANTISSA cells
(`docs/RESULTS-KT1-v3.md`) therefore remains **UNEXPLAINED**, and is
carried into the paper as an open observation rather than a resolved one.

Operand-level injection (flips in A or B) is legitimate future work; it is
not a prerequisite for publishing detector characterization under the
output-level model that was actually measured.

CP-AB2 and CP-AB3 are **VOID-PRECONDITION** (`docs/CHECKPOINTS-PRE-PAPER.md`).
