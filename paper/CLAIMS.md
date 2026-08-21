# CLAIMS

Inventory for the paper. Every claim names its supporting results document.
No prose. No new numbers.

## CAN CLAIM

1. **K^(−1/2) noise-floor law.** Clean per-row floor scales with measured
   exponent **−0.4999** over a **16×** range in K (512–8192).
   → `docs/RESULTS-KSCALE.md`

2. **Detectability is K-invariant.** The pre-registered falsifier (*if
   detection is flat across a 16× range of K, the √K mechanism is wrong…*)
   **fired** and was published; SIGN 1-flip stays ~73–81% across K.
   → `docs/RESULTS-KSCALE.md`

3. **Bit position, not contraction length, governs detectability.**
   Detection stratified by bit class (SIGN / EXPONENT / MANTISSA_HIGH /
   MANTISSA_LOW), not by K.
   → `docs/RESULTS-KSCALE.md`

4. **MANTISSA_LOW is a hard negative — two threshold bases, never merged.**
   - **0/1500** at per-K observed clean maxima (K-scaling).
   - **0/600** at `threshold_gpd` (K = 4096).
   Injection audit: flips survive bf16 cast (not discarded).
   → `docs/RESULTS-KSCALE.md`, `docs/RESULTS-FPR.md`,
     `docs/RESULTS-INJECTION-AUDIT.md`

5. **Per-GEMM FPR bound with independence checked, not asserted.**
   Variance ratio **0.9846** against Binomial(4096, 0.01); GPD-extrapolated
   `threshold_gpd`.
   → `docs/RESULTS-FPR.md`

6. **Three documented retractions (methodological caution).**
   - Scalar residual (residual-v2) evaluation voided; SIGN “structural
     invisibility” retracted under residual-v3.
     → `docs/RESULTS-KT1-v2.md` (void), `docs/RESULTS-KT1-v3.md`
   - Mantissa near-miss hypothesis (bf16 discarding flips) tested and
     rejected; hard negative stands.
     → `docs/RESULTS-INJECTION-AUDIT.md`
   - A-vs-B normalizer self-cancellation prediction falsified at its
     precondition when injection was found to be C-only.
     → `docs/PREDICTIONS-MANTISSA-AB.md`

7. **KT-1 FAIL at 76% against a pre-registered ≥90% bar** — headline, not
   buried. Canonical: **304/400 = 76%** at `threshold_gpd` (pilot clean-max
   figure 75.25% remains unedited in KT1-v3).
   → `docs/RESULTS-FPR.md`, `docs/RESULTS-KT1-v3.md`

8. **A registered prediction (A-vs-B normalizer self-cancellation) was
   falsified at its precondition** when the injection layer was found to
   be C-only. Reported as a method note.
   → `docs/PREDICTIONS-MANTISSA-AB.md`

9. **Perturbation model is fully specified and versioned** (perturbation-v1:
   algorithm-level, post-GEMM, output-tensor bit flips).
   → `docs/SPEC-PERTURBATION-MODEL.md`

## CANNOT CLAIM

1. **Anything about real GPU fault incidence, soft-error rates in silicon,
   or mercurial-core prevalence in deployed accelerators.** No claim is made
   about operand-level or instruction-level corruption. All reported
   detection rates are for **post-GEMM output corruption only**
   (perturbation-v1). Fleet SDC literature motivates the problem; it does
   not supply our measured rates.
   → `docs/SPEC-PERTURBATION-MODEL.md`;
     scope in `docs/RESULTS-KSCALE.md` limitations;
     framing in `paper/OUTLINE.md` ABSTRACT slot

## Citation hygiene (Method)

Normative references for the method are `docs/RESIDUAL.md` (residual-v3)
and `docs/SPEC-PERTURBATION-MODEL.md` (perturbation-v1) **ONLY**.
`docs/SPEC-NOISEFLOOR.md` is residual-v2, has zero `run-*.json` files,
lists `measured_gpu_models: []`, has not met its three-model bar, and
self-labels its target quantile **UNJUSTIFIED**. It describes a separate,
unfinished characterization effort and **MUST NOT** be cited by this paper.
