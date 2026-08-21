# Pre-paper checkpoint tracking

Minimal tracker for checkpoints that supersede or void earlier plans.
Does not replace results documents.

| ID | status | note |
| --- | --- | --- |
| CP-AB1 | DONE | `docs/PREDICTIONS-MANTISSA-AB.md` locked; commit `519e4cb` |
| CP-AB2 | **VOID-PRECONDITION** | Required recoverable A/B operand identity; harness is C-only. Outcome: `docs/PREDICTIONS-MANTISSA-AB.md` § OUTCOME — FALSIFIED AT PRECONDITION |
| CP-AB3 | **VOID-PRECONDITION** | Results write-up against an A-vs-B measurement that cannot exist under the current harness. Same outcome section. |
| CP-INJ-1 | DONE | Close out CP-AB1 as precondition falsification; add `docs/SPEC-PERTURBATION-MODEL.md` (perturbation-v1) |
| CP-INJ-2 | DONE | Propagate output-level framing into `paper/` scaffold; `main.tex` skeleton |
| CP-PAPER-0 | DONE | Skeleton + claim inventory under CP-INJ framing |
