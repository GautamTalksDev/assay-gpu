# assay-gpu

Deterministic GPU correctness assay. Packaged as `assay-gpu`; console script: `assay`.

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

## What this is

A hardware verification tool. A false positive tells a company its expensive
hardware is broken. All logic is deterministic and unit-testable. There are no
LLM calls, heuristic scores, machine-learning models, or unsolicited network
requests. Every numerical threshold must be traceable to measured data in this
repository.

This repository is a skeleton. Application logic is not present yet.

## GPU access

Spend $0. See `docs/DEV_ENVIRONMENT.md` for Kaggle (primary), Colab,
Lightning AI, university HPC, and the Docker path that Kaggle/Colab cannot
validate. Signup credits are survey budget; log them in `docs/BUDGET.md`.

## Tests

```bash
uv run pytest -m cpu    # CI
uv run pytest -m gpu    # real GPU, not CI
```

## Layout

```
src/assay/cli.py          CLI entrypoint (typer)
src/assay/probe/          environment + device introspection
src/assay/reference/      golden reference vectors, fp64 CPU compute
src/assay/workload/       deterministic GPU workload suite
src/assay/noise/          empirical noise floor characterization
src/assay/abft/           checksum detector
src/assay/inject/         fault injector (TEST FIXTURE ONLY)
src/assay/report/         attestation generation + signing
tests/
data/noisefloor/          measured noise floor data (version controlled)
docs/
```

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
