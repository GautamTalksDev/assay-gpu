# OUTLINE

Skeleton only. No prose in this session. Section slots for later writing.

## ABSTRACT

- **Sentence 1–2:** [TITLE CLAIM — fill later]
- **Sentence 3 (REQUIRED FRAMING — not deferred to Limitations):**
  This is a **detector characterization under an output-level (post-GEMM)
  algorithm-level perturbation**, not fault characterization in silicon.
  Preempt arXiv:2601.19912 (algorithm-level ≠ realistic GPU behaviour).
  Normative definition: `docs/SPEC-PERTURBATION-MODEL.md` (perturbation-v1).
- **Rest:** [headline numbers / KT-1 FAIL / scope one-liner — fill later]

## 1. INTRODUCTION

- Motivation: silent corruption / mercurial cores (cite Meta, Google)
- Gap: checksum / ABFT detector behaviour under controlled bit-class injection
- Contributions (map 1:1 to CAN CLAIM list in `CLAIMS.md`)
- Explicit non-claim: no silicon fault incidence; output-level only

## 2. BACKGROUND AND RELATED WORK

- Ones-sided checksum / Huang–Abraham ABFT
- Soft errors, SDC, mercurial cores
- Fault-injection abstraction levels (algorithm / instruction / µarch);
  cite arXiv:2601.19912
- TODO: remaining cites

## 3. METHOD

- Residual-v3 elementwise formulation (pointer only; formula from locked docs)
- ### 3.x Perturbation model (normative)
  - Pointer: `docs/SPEC-PERTURBATION-MODEL.md` (perturbation-v1)
  - Output-level (post-GEMM) algorithm-level bit-class injection only
- Threshold: `threshold_gpd` canonical (stated per-GEMM FPR)
- Workload lock: W02, bf16, T4, K-grid as measured
- Pre-registration discipline (predictions before measurement)

## 4. RESULTS

### 4.1 Noise floor vs K
- Slot for K^(−1/2) law (−0.4999, 16×)
- Support: `docs/RESULTS-KSCALE.md`

### 4.2 Detectability vs K (falsifier fired)
- Slot for K-invariant detectability
- Support: `docs/RESULTS-KSCALE.md`

### 4.3 Bit-class stratification
- Slot for bit position governs detectability
- Support: `docs/RESULTS-KSCALE.md`

### 4.4 MANTISSA_LOW hard negative
- Slot: both threshold bases named separately
- Support: `docs/RESULTS-KSCALE.md`, `docs/RESULTS-FPR.md`,
  `docs/RESULTS-INJECTION-AUDIT.md`

### 4.5 FPR bound and independence check
- Slot: variance ratio 0.9846; GPD path
- Support: `docs/RESULTS-FPR.md`

### 4.6 KT-1 scorecard (headline FAIL)
- Slot: 76% vs ≥90%
- Support: `docs/RESULTS-FPR.md`, `docs/RESULTS-KT1-v3.md`

### 4.7 Retractions
- (1) Scalar residual void
- (2) Mantissa near-miss rejected
- (3) A-vs-B normalizer self-cancellation falsified at precondition
- Support: `docs/RESULTS-KT1-v2.md`, `docs/RESULTS-KT1-v3.md`,
  `docs/RESULTS-INJECTION-AUDIT.md`, `docs/PREDICTIONS-MANTISSA-AB.md`

### 4.8 Falsified prediction: normalizer self-cancellation
- Support: `docs/PREDICTIONS-MANTISSA-AB.md`

## 5. SCOPE (limitations — not “future work”)

- T4 only; W02 only; bf16 only
- **Output-level injection only**; operand, accumulator, and instruction
  level are out of scope and named as future work
  → `docs/SPEC-PERTURBATION-MODEL.md`
- Shared **0.5247** minimum across MANTISSA cells is **unexplained**
  → `docs/RESULTS-KT1-v3.md`; `docs/PREDICTIONS-MANTISSA-AB.md` OUTCOME
- **p95 NOT EVALUABLE:** mean 205.3 exceedances/GEMM vs 64-value capture;
  19500/19500 GEMMs truncated; lost exceedances systematically the smaller
  ones. Data-collection limitation discovered after method lock — not a
  changed method; not a redefinition of F3.
  → `docs/RESULTS-FPR.md`
- Extrapolation: ~1e-2 observed exceedance probability → 2.44e-10
  (decades beyond data)
- Implied GPD upper endpoints (u − σ/ξ) differ **1.23×** (3.74e-06 from p99
  vs 4.60e-06 from p999) while reported quantiles agree at **1.06×** — hard
  ceiling less well determined than the sensitivity ratio alone suggests
- Backend disagreement (PyTorch fp32 GEMV vs Triton fp64) unresolved
  → `docs/SPEC-NOISEFLOOR.md`

## 6. DISCUSSION

- [fill later — map to CLAIMS only]

## 7. CONCLUSION

- [fill later]
