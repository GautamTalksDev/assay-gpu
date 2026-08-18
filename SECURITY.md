# Security Policy

## Supported versions

This project is a pre-logic skeleton. Treat the current tree as unsupported
for production use. Kill tests KT-1 and KT-2 have been evaluated and both
are FAIL (`docs/RESULTS-KT1.md`, `docs/RESULTS-KT2.md`). `assay watch` is
not a shipped product.

## Reporting a vulnerability

Do not open a public issue for security reports.

Email the maintainers with:

- a description of the issue
- steps to reproduce
- affected commit or release, if known
- any measured impact on false-positive or false-negative rates

We will acknowledge receipt and follow up with a remediation plan or a
reasoned decline. Do not disclose publicly until a fix is released or we
confirm the report is not a vulnerability.

Hardware-failure claims against real devices must be backed by recorded
measurements in `data/` and a reproducible command line. Unreproducible
reports will be closed.

Survey publications name **no provider** for an individual FAIL or
INCONCLUSIVE. Three reproductions would be required to name a provider;
this budget will not produce three. Aggregate statistics only
(`docs/SURVEY.md`, `docs/SURVEY-2026.md`, `scripts/survey/collect.py`).
