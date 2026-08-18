# Rented GPU Correctness Survey, 2026

**n = 0.** No GPU instance was rented. No signed report was archived.
This document is the publication contract: naming policy, method,
limitations, empty aggregates, and a CC-BY-4.0 dataset schema. It is
not a claim that thirty machines were assayed. A later fill of
`data/survey/reports/` must not rewrite these rules.

Tool: `assay-gpu` 0.0.0. Protocol: `docs/SURVEY.md`. Dataset:
`data/survey/` (CC-BY-4.0). Software remains Apache-2.0.

## Naming policy

This section is written before any findings.

- Aggregate statistics may name the **set** of providers surveyed.
  That set, in this release, is empty.
- No individual FAIL is attributed to a named provider. At n=30 we
  lack the reproductions to support it and the claim would be unsound.
  At n=0 the same ban holds.
- No comparative per-provider claim is made anywhere in this document.
  Not in findings. Not in anomalies. Not in a footnote.

Naming a provider for a single FAIL requires three reproductions on
the same provider and instance type. The budget does not permit a
third rental (`docs/SURVEY.md`). Therefore no provider is named for
an individual failure, including in this file.

The public table is `data/survey/summary.csv`. It has no provider
column. `named_in_publication` is always `false`. The operator ledger
`docs/BUDGET.md` and the lab notebook `docs/LAB_NOTEBOOK.txt` may name
providers for operational accounting. They are **not** part of the
CC-BY-4.0 dataset and they are not findings.

## Method

### What was rented

Nothing. `docs/BUDGET.md` has zero data rows. Cumulative spend
**$0.00**. `scripts/survey/preflight.sh` against `HARD_CEILING=75`
exits 0 with remaining $75.00.

| Quantity | Count | Source |
| --- | --- | --- |
| Rented instances | 0 | `docs/BUDGET.md` |
| Distinct providers surveyed | 0 | empty ledger |
| Distinct GPU models assayed | 0 | `data/survey/summary.csv` |
| Archived attestation-v1 reports | 0 | `data/survey/reports/` |
| Reproduction attempts | 0 | `data/survey/attempts.json` |

### How a rental would be run

Every future row follows `docs/SURVEY.md` without exception:

1. `./scripts/survey/preflight.sh`. Exit 1 ends the survey.
2. Cheapest tier on that provider with root access.
3. `assay run --thorough`, save a signed attestation-v1 report under
   `data/survey/reports/` with a filename that does not contain the
   provider, destroy the instance immediately, phone timer.
4. Log `docs/BUDGET.md` before the next rental.

FAIL or INCONCLUSIVE gets exactly one reproduction (same provider,
same instance type, cheapest tier). Never a third.

### When

No rental date exists. This text is dated 2026-08-18 from the
operator workstation (WSL2, no NVIDIA GPU). That machine is not a
survey instance.

### Tool and spec versions

| Item | Version |
| --- | --- |
| Package | `assay-gpu` 0.0.0 |
| Command | `assay run --thorough` (60-minute wall budget) |
| Report spec | `attestation-v1` (`docs/SPEC-ATTESTATION.md`) |
| Signature | Ed25519, `self-signed` only (v1 cannot issue `assay-signed`) |
| Workload suite | `workload-suite-v1` (`docs/WORKLOADS.md`) |
| Noise floor | `noisefloor-v1` (`docs/SPEC-NOISEFLOOR.md`) |
| Characterization files | none (`data/noisefloor/` has no `run-*.json`) |
| Goldens | `data/reference/` fp64 arrays, NumPy 2.4.6 |
| Public aggregate | `scripts/survey/collect.py` → `data/survey/summary.csv` |

`assay run` makes zero network requests. Reports are not uploaded.
`assay verify` is offline.

### Sampling frame (not the surveyed set)

The intended frame is 30–35 instances, ≥12 providers, ≥4 GPU models,
cash-or-credit ≤ $75 (`HARD_CEILING`). Breadth over repeats: twelve
providers with one instance each, not twenty instances on one
provider. Prefer T4 / A10 / A100 / H100 / RTX-class over a T4-only
pile. The names of providers in that frame are a shopping list in
`docs/SURVEY.md`. They are not a surveyed set. This document does not
treat them as observations.

Shared vs dedicated would be taken from `probe.appears_shared` on the
attestation body (MIG enabled, or other compute PIDs on the GPU).
That boolean is not a provider name. The released CSV in this version
does not yet emit it; with n=0 the shared/dedicated table is empty
either way.

## Limitations

This section is written before the findings section.

n=30 supports descriptive statistics, not inferential ones. You can
say "we observed X anomalies across 30 instances and 12 providers."
You cannot say "provider A is less reliable than provider B." Any
per-provider comparative claim is unsupported at this sample size and
asserting one is the fastest way to destroy the project's credibility.

Short spot runs miss time-dependent faults. Thermal drift and
degradation-related SDC need sustained load. You are sampling, not
monitoring.

Cheapest-tier bias. You are systematically testing the low end of
each provider's fleet. State it; do not speculate about the direction.

Signup-credit instances may not be representative. Some providers
route trial accounts to specific hardware pools.

Additionally, this release:

- Has **n=0**, which does not support even the descriptive sentence
  above. "We assayed 30 rented GPU instances across 12 providers and
  found no detectable numerical corruption" is a citable finding **if
  and only if** those thirty reports exist. They do not. A
  zero-anomaly result at n=30 is publishable and expected; a
  zero-assay result is not that result.
- Has an empty noisefloor (`measured_gpu_models: []`). On
  uncharacterized hardware `check_gemm` returns **INCONCLUSIVE**,
  never FAIL. Running the suite on a rented GPU today would not
  produce PASS. KT-1 is FAIL (`docs/RESULTS-KT1.md`): the detector
  has not shown FPR `< 1e-6` per GEMM at production shapes and caught
  `0%` of injected exponent-bit flips, not `>=90%`.
- Emits only **self-signed** reports. A provider grading its own
  hardware is not independent evidence (`docs/SPEC-ATTESTATION.md`).
- Uses a 60-minute `--thorough` budget. That is a wall-clock cap on
  the suite, not a soak.

## Aggregate findings

Naming policy and limitations precede this section. Nothing here
names a provider. Nothing here compares providers.

### Verdicts by GPU model

| gpu_model | n | PASS | FAIL | INCONCLUSIVE |
| --- | --- | --- | --- | --- |
| *(none)* | 0 | 0 | 0 | 0 |

Source: `data/survey/summary.csv` (header only).

### Verdicts by dtype

| dtype | n_cases | PASS | FAIL | INCONCLUSIVE |
| --- | --- | --- | --- | --- |
| float32 (W01) | 0 | 0 | 0 | 0 |
| bfloat16 (W02) | 0 | 0 | 0 | 0 |
| float16 (W03) | 0 | 0 | 0 | 0 |

No case rows exist. W04–W07 do not vote on GEMM ABFT in noisefloor-v1.

### Verdicts by shared vs dedicated

| appears_shared | n | PASS | FAIL | INCONCLUSIVE |
| --- | --- | --- | --- | --- |
| true | 0 | 0 | 0 | 0 |
| false | 0 | 0 | 0 | 0 |

### Residual distributions

No residuals. `data/noisefloor/` contains no `run-*.json`. No
attestation `cases[].residual_hex` values exist to histogram.
Percentiles are not defined. They are not inferred.

### One-sentence finding

We assayed 0 rented GPU instances across 0 providers and therefore
did not measure numerical corruption, nor the absence of it.

That is not the n=30 zero-anomaly sentence. Do not quote this release
as a bound on silent data corruption in the rental market.

## Anomalies

Every FAIL or INCONCLUSIVE, with reproduction status, anonymized by
provider. This table is empty.

| instance_index | gpu_model | status | attempt | reproduced | unreproduced | named_provider |
| --- | --- | --- | --- | --- | --- | --- |
| *(none)* | — | — | — | — | — | never |

`named_provider` is not a column in the released CSV. It is shown
here as **never** so the policy is visible on the page that would
otherwise be tempting to fill with a brand.

## Methodology critique

A hostile reviewer can attack this work, including a future n=30
fill, on the following grounds. These are ours to state first.

1. **This release has no field data.** A document titled as a 2026
   survey with n=0 is a methods paper plus an empty archive. Treating
   it as a market study is a misread we invite if the title is quoted
   without the first sentence.

2. **The detector cannot yet FAIL live hardware.** noisefloor-v1 has
   zero characterization files. Uncharacterized GEMMs are
   INCONCLUSIVE. KT-1 failed. A later archive of thirty INCONCLUSIVE
   reports is not thirty clean machines; it is thirty times "we did
   not have a threshold." Pass-rate tables would then overclaim.

3. **n=30 is descriptive only.** Even at target size there is no
   power for provider ranking, no confidence interval that licenses
   "A is worse than B," and no third reproduction to name a vendor
   for one FAIL. Reviewers who want a league table will not get one
   from this design. Offering one anyway would be unsound.

4. **One instance per provider is a survey of variance, not of a
   provider.** It is the right design for a rental-market snapshot
   and the wrong design for accusing a company. The naming policy
   exists because the design cannot support the accusation.

5. **Cheapest-tier and signup-credit bias are baked in.** The
   protocol requires the cheapest root SKU and treats promotional
   credit as survey budget. Findings, when they exist, describe that
   slice. They do not describe reserved A100s on committed contracts.
   Direction of bias is not estimated here.

6. **Spot duration is not monitoring.** `--thorough` is one hour of
   suite wall time, then destroy. Thermal drift, wear-out SDC, and
   neighbor noise on a shared MIG partition over days are out of
   scope. A clean hour does not imply a clean month.

7. **Self-signed attestations do not prove independent observation.**
   The operator generates the key, runs the binary, and signs the
   JSON. Tamper-evidence is real (`assay verify` rejects a
   hand-edited `status`). Independence is not. v1 has no issuance
   service.

8. **The analyst is the operator.** The same person writes the
   ledger, the notebook, and this file. There is no separate
   measurement lab. Probe fields (`dmi_sys_vendor`, GPU UUID) that
   could identify a provider are stripped from the public CSV; that
   is necessary and it also means outsiders cannot re-identify a row
   to audit the operator's classification.

9. **Shared vs dedicated is a heuristic.** `appears_shared` is MIG
   or extra compute PIDs at probe time. A quiet neighbor, a
   time-sliced hypervisor, or a lying vendor API will not show up.
   Virtualization flags (`hypervisor` CPU bit, known DMI vendors)
   are similarly incomplete.

10. **Residual distributions will be sparse even at n=30.** One
    thorough run per box, on uncharacterized GPUs, yields
    INCONCLUSIVE rows more often than FAIL rows. Histograms of
    checksum residuals without `min_samples` characterization are
    measurements without a threshold. They must not be dressed as
    detections.

11. **Selection of twelve providers is convenience sampling.**
    Signup-credit availability, geographic payment, and who still
    offers root on a $0.20/hr card decide who is in the set. The set
    is not a random sample of "the cloud."

12. **Destroy-immediately is a budget control, not a scientific
    instrument.** It prevents overnight billing. It also prevents
    follow-up on the same physical card. Reproduction is a second
    rental of the same SKU, not the same silicon.

13. **The empty CC-BY-4.0 archive is easy to over-cite.** License
    and schema are real. Counts of zero are real. Anyone quoting
    "dataset release, 2026" without n=0 is doing the damage this
    section exists to pre-empt.
