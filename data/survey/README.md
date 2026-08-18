# Survey dataset (CC-BY-4.0)

This directory is the public dataset for the Rented GPU Correctness
Survey. License: Creative Commons Attribution 4.0 International
(`LICENSE` in this directory, https://creativecommons.org/licenses/by/4.0/).

The `assay-gpu` software is Apache-2.0 (`LICENSE` at the repository
root). This dataset license does not change that.

Cite `docs/SURVEY-2026.md`. Quote n from `summary.csv`, not from
memory. This release has **zero** reports.

## Contents

| Path | What it is |
| --- | --- |
| `LICENSE` | CC-BY-4.0 legal code |
| `summary.csv` | Anonymized instance table (`scripts/survey/collect.py`) |
| `attempts.json` | Reproduction metadata keyed by `report_sha256` |
| `reports/` | Archived attestation-v1 JSON (empty in this release) |

Not in this dataset (provider names live there):

- `docs/BUDGET.md`
- `docs/LAB_NOTEBOOK.txt`

## Naming

`summary.csv` has no provider column. `named_in_publication` is
always `false`. Do not join this table to the ledger and publish the
join.
