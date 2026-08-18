# Cloud budget ledger

HARD CEILING: $75 total (credit or cash, all providers combined).

When cumulative total would exceed $75, stop. Do not launch another instance.
`scripts/survey/preflight.sh` reads this table and **exits 1** if spend is
at or past `HARD_CEILING` (default 75). That refusal is the survey ending.

Signup credits are survey budget: a provider you have an account with is a
provider you can survey. Log a credit the day it is claimed, including how
much remains. Free-tier notebook hours (Kaggle, Colab, Lightning AI,
university HPC) are not cash and are not rows in this table.

No signup credit has been consumed as of 2026-08-18. Remaining against the
ceiling: $75.00. The CP-1 Docker GPU smoke test still needs a RunPod or Vast
new-account credit on a cheap GPU (3090/A4000); that row will be filled when
the account is opened, not estimated in advance.

Log a row **before** the next rental. `attempt` is 1 for a first assay, 2
for the single allowed reproduction. Never 3. Do not name a provider for
an individual FAIL; the public artifact is `scripts/survey/collect.py`.

Columns: date, provider, instance type, GPU model, credit or cash, amount spent, cumulative total, verdict, attempt, report path.

| date | provider | instance type | GPU model | credit or cash | amount spent | cumulative total | verdict | attempt | report path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

Reconciled cumulative total: **$0.00**. Preflight: OK. No instance has been
rented, so there is nothing to destroy.
