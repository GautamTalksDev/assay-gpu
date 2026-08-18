# Rented GPU Correctness Survey

Phase D evidence collection. **Do not launch. Do not post. Do not tweet.**

This file is the rental protocol. `scripts/survey/preflight.sh` is the
budget gate. `scripts/survey/collect.py` builds the anonymized CSV.
`docs/LAB_NOTEBOOK.txt` is the operator log. Feature ideas go in
`IDEAS.md` and are not opened.

## Status (2026-08-18)

| Target | Have |
| --- | --- |
| ≥30 instances | **0** |
| ≥12 distinct providers | **0** |
| ≥4 distinct GPU models | **0** |
| archived signed reports | **0** |
| spend vs $75 | **$0.00** (`docs/BUDGET.md`) |
| anomalies with one reproduction | none (no assays) |
| consoles at zero active resources | no accounts opened |

Definition of Done is **unmet**. This checkpoint built the harness. It
did not rent. Publication text: `docs/SURVEY-2026.md` (naming policy
and limitations before findings; n=0 aggregates). Do not quote that
file as a thirty-instance market study.

## The one rule that makes $60 sufficient

12 providers × 1 instance beats 3 providers × 20 instances. Every time.
Without exception. The claim is variance across the rental market.
Twenty instances from one provider tells you about one provider.

When tempted to run "just one more" on a provider already in the
ledger, spend it on a provider that is not.

Signup credits are survey budget. Log a credit the day it is claimed.

## HARD_CEILING

```bash
./scripts/survey/preflight.sh
HARD_CEILING=75 ./scripts/survey/preflight.sh
```

Default `HARD_CEILING=75`. The script prints cumulative spend from
`docs/BUDGET.md` and **exits 1** if spend is at or past the ceiling, or
if the ledger is not reconciled to the cent. That refusal is the
survey ending: publish what you have. Do not rent.

Override the ledger path only in tests: `BUDGET_MD=/tmp/ledger.md`.

## Rental protocol (every time, without exception)

1. Check preflight. If it refuses, the survey is over. Publish what you
   have.
2. Rent the **cheapest** tier that provider offers with root access.
   This is a correctness test, not a performance test. A $0.20/hr card
   is as valid as a $3/hr card, and often more interesting.
3. Pull the container, run `assay run --thorough`, save the signed
   report under `data/survey/reports/` with a name that does **not**
   contain the provider (use `issued_at` + `run_id` prefix). **Destroy
   the instance immediately.** Set a phone timer. Forgetting a running
   instance overnight is the realistic way this budget blows up.
4. Log the row in `docs/BUDGET.md` **before** starting the next rental.
   Then append `docs/LAB_NOTEBOOK.txt`.

On the instance (after preflight has already passed on the operator
laptop):

```bash
uv run assay keygen --out /tmp/survey-operator.json
uv run assay run --thorough --report /tmp/report.json --signing-key /tmp/survey-operator.json
# copy /tmp/report.json off the box, then destroy the instance
```

There is no provider API destroy helper. Destroy in the provider
console. That omission is deliberate (`IDEAS.md`).

## Reproduction policy (constrained budget)

- A FAIL or INCONCLUSIVE gets **exactly one** reproduction attempt,
  same provider, same instance type, cheapest tier.
- Reproduced: it is a finding. Naming a provider publicly requires
  **three** reproductions. This budget will not produce three.
  Therefore: **no provider is named for an individual failure.**
  Aggregate statistics only.
- Not reproduced: published anonymized, flagged `unreproduced=true`.
- Never rented a third time.

Record the attempt number in `docs/BUDGET.md` (`attempt` = 1 or 2) and
in `data/survey/attempts.json` keyed by `report_sha256`.
`scripts/survey/collect.py` copies reproduction flags into the CSV and
forces `named_in_publication=false`.

## Anonymized CSV

```bash
uv run python scripts/survey/collect.py \
  --reports-dir data/survey/reports \
  --attempts data/survey/attempts.json \
  --out data/survey/summary.csv
```

The CSV has no provider, hostname, UUID, DMI, IP, or public-key
columns. GPU model is kept: the interesting variation is across
generations, not across brand pages.

## Breadth target (not a launch checklist)

Tier 1 (likely free or near-free via signup credit): RunPod, Vast.ai,
Thunder Compute, Paperspace, Modal, DigitalOcean (Hatch), Lambda,
Genesis Cloud.

Tier 2 (cheap spot, budget $3–6 each): TensorDock, Hyperstack,
CoreWeave, Nebius, Crusoe, Together, Fly.io GPUs, OVHcloud, Scaleway.

Tier 3 (hyperscaler new-user credit): use **one** of GCP or Azure, not
both.

Prefer T4 / A10 / A100 / H100 / RTX-class over a T4-only pile. A
survey of only T4s is a weaker claim because dtype behaviour differs
across generations.

Do not open these accounts in this checkpoint.

## What this is not

- Not a launch.
- Not a blog post.
- Not a tweet.
- Not a reason to widen KT-2.
- Not a third reproduction.
- Not a provider named for one FAIL.
