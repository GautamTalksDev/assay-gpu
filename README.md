# assay-gpu

**KT-1 status: UNEVALUATED (awaiting GPU characterization).** See `docs/RESULTS-KT1.md`.
**KT-2: FAIL.** See `docs/RESULTS-KT2.md`.

The detector was not retuned. The 10% overhead bar was not moved to 25%.
`assay watch` is **not a shipped product**. `assay run` remains the
acceptance test.

`assay run` still ships. On uncharacterized hardware it prints
measurements and returns **INCONCLUSIVE** (exit 2). It never guesses
PASS or FAIL. A false positive would tell a company its expensive GPU
is broken; a missed detection is preferred.

This workstation had no NVIDIA GPU, so the multi-provider Definition of
Done for CP-7 is **unmet**: four GPU models across three providers,
`docker run --gpus all <image> run` on a fresh instance, and a network
monitor proving zero egress were not executed here. CPU tests cover
exit codes, the uncharacterized banner, FAIL detail, and the
zero-network import guard.

## Kill Tests

  KT-1 (Feasibility): If no detection threshold exists that separates
       single-bit-flip corruption from floating-point non-associativity
       noise -- specifically, if we cannot achieve a false-positive rate
       below 1e-6 per GEMM at production shapes while catching >=90% of
       exponent-bit flips -- the detector adds nothing and this project
       stops.

  KT-2 (Overhead): If end-to-end assay-protected inference overhead exceeds
       10% on representative transformer workloads, nobody will run it
       continuously and the paid tier has no product. This project stops.

  KT-3 (Demand): If, within three months of publishing the Rented GPU
       Correctness Survey, fewer than three companies have contacted us
       unprompted asking for continuous or private attestation, there is
       no business here. Keep the tool free, take the reputation, move on.

KT-1 status: UNEVALUATED (awaiting GPU characterization).
KT-2 is FAIL (`docs/RESULTS-KT2.md`).
KT-3 was not opened. `assay watch` is cut as a product: no tokens/sec
delta exists on any GPU, so the 10% bar is unmet at every sampling rate.

## What this is

A hardware verification tool. A false positive tells a company its expensive
hardware is broken. All logic is deterministic and unit-testable. There are no
LLM calls, heuristic scores, machine-learning models, or unsolicited network
requests. Every numerical threshold must be traceable to measured data in this
repository.

`assay run` makes **zero network requests**: no telemetry, no version
check, no analytics, no phone-home. That guarantee is printed in
`assay --help`. `assay verify` is also offline: it checks an Ed25519
signature and internal consistency against any public key you pin.

What shipped: seeded fp64 goldens, seven GPU workloads, noisefloor-v1
lookup (empty), one-sided GEMM ABFT, a test-only bit-flip injector, the
detection matrix, `assay run`, and signed **attestation-v1** reports
(`docs/SPEC-ATTESTATION.md`). `data/noisefloor/` has no `run-*.json`.

## GPU access

Spend $0. See `docs/DEV_ENVIRONMENT.md`. Signup credits: `docs/BUDGET.md`.
No credit was consumed. The rented-GPU survey has **n=0** field
measurements. Publication contract: `docs/SURVEY-2026.md`. Protocol:
`docs/SURVEY.md`. Dataset: `data/survey/` (CC-BY-4.0).

## Tests

```bash
uv run pytest -m cpu    # CI, including KT-1 inputs (detection matrix, ABFT)
uv run pytest -m gpu    # real GPU: W01–W07 plus `assay run --quick --json`
```

Workloads: `docs/WORKLOADS.md`. Goldens: `data/reference/`.
Noise floor: `docs/SPEC-NOISEFLOOR.md` (noisefloor-v1).
GEMM checksum detector: `docs/ABFT.md`.
Fault-injection detection matrix: `docs/DETECTION_MATRIX.md`.
KT-1 evaluation: `docs/RESULTS-KT1.md`.
KT-2 evaluation: `docs/RESULTS-KT2.md`.
Attestation: `docs/SPEC-ATTESTATION.md` (attestation-v1).
Survey protocol: `docs/SURVEY.md`. Publication: `docs/SURVEY-2026.md`.
Lab notebook: `docs/LAB_NOTEBOOK.txt`. Unbuilt ideas: `IDEAS.md`.

```bash
uv run assay --help
uv run assay run                 # GPU; default 10-minute budget
uv run assay run --quick         # 2-minute budget
uv run assay run --thorough      # 60-minute budget
uv run assay run --json --report /tmp/assay.json --signing-key /tmp/operator.json
uv run assay keygen --out /tmp/operator.json
uv run assay verify /tmp/assay.json --pubkey /tmp/operator.json
uv run assay workload run --double    # GPU; bitwise identity, no goldens/ABFT
uv run assay characterize --lookup --workload W02 --dtype bfloat16 --m 4096 --k 4096 --n 4096
uv run assay characterize --repeats N   # GPU; writes data/noisefloor/**/run-*.json
uv run assay abft overhead --repeats N --m 512 --k 512 --n 512 --backend pytorch --device cpu
```

Exit codes for `assay run`: **0 PASS**, **1 FAIL**, **2 INCONCLUSIVE**,
**3 operational** (no GPU, bad flags, missing noisefloor methodology).

Docker (NVIDIA Container Toolkit):

```bash
docker run --gpus all <image> run
```

## Layout

```
src/assay/cli.py          CLI entrypoint (typer)
src/assay/probe/          environment + device introspection (no HTTP)
src/assay/run/            assay run session, budget, human/JSON render
src/assay/reference/      golden reference vectors, fp64 CPU compute
src/assay/workload/       deterministic GPU workload suite
src/assay/noise/          empirical noise floor characterization
src/assay/abft/           checksum detector
src/assay/inject/         fault injector (TEST FIXTURE ONLY)
src/assay/watch/          live module hooks (measurement harness; not shipped)
src/assay/report/         signed attestation-v1 + offline verify
scripts/survey/           preflight ceiling + anonymized collect (do not launch)
tests/
data/noisefloor/          measured noise floor data (version controlled)
data/reference/           fp64 golden arrays + manifest (SHA-256 of arrays)
data/survey/              CC-BY-4.0 dataset (empty reports; summary.csv)
docs/                     WORKLOADS.md, SPEC-NOISEFLOOR.md, ABFT.md,
                          DETECTION_MATRIX.md, RESULTS-KT1.md,
                          RESULTS-KT2.md, SPEC-ATTESTATION.md,
                          SURVEY.md, SURVEY-2026.md, LAB_NOTEBOOK.txt,
                          BUDGET.md
IDEAS.md                  feature ideas that were not built
```

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

## License

Apache License 2.0 for the software. See `LICENSE` and `NOTICE`.
The survey dataset (`data/survey/`) is CC-BY-4.0
(`data/survey/LICENSE`, `docs/SURVEY-2026.md`).
