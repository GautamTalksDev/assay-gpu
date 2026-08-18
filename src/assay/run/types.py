"""Records for one assay run. Verdicts come from noisefloor, not from guesses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from assay.abft.check import CheckStatus
from assay.probe.environment import EnvironmentProbe
from assay.run.budget import RunBudget


class ExitCode(IntEnum):
    PASS = 0
    FAIL = 1
    INCONCLUSIVE = 2
    OPERATIONAL = 3


class AssayOperationalError(Exception):
    """Run cannot proceed. Maps to exit code 3. Not a hardware FAIL."""


@dataclass(frozen=True, slots=True)
class CaseRecord:
    workload: str
    case: str
    shape: tuple[int, ...]
    dtype_name: str | None
    status: CheckStatus
    reason: str
    residual_hex: str | None
    residual_decimal: str | None
    threshold_hex: str | None
    threshold_decimal: str | None
    sample_max_hex: str | None
    n_samples: int | None
    min_samples: int | None
    noisefloor_spec_version: str | None
    golden_max_abs_error_hex: str | None
    wall_time_s: float
    skipped: bool
    skip_reason: str | None
    counts_toward_verdict: bool


@dataclass(frozen=True, slots=True)
class AssayResult:
    probe: EnvironmentProbe
    gpu_model: str | None
    noisefloor_status: str
    noisefloor_reason: str
    budget: RunBudget
    cases: tuple[CaseRecord, ...]
    status: CheckStatus
    elapsed_s: float


def overall_status(cases: tuple[CaseRecord, ...]) -> CheckStatus:
    """FAIL wins. PASS only if every voting GEMM case passed and none were skipped.

    W04-W07 have no GEMM ABFT in noisefloor-v1, so they are measurements
    and do not vote. A time-budget skip of a voting case is INCONCLUSIVE
    unless a FAIL already occurred.
    """
    if any(case.status is CheckStatus.FAIL and not case.skipped for case in cases):
        return CheckStatus.FAIL
    voting = tuple(case for case in cases if case.counts_toward_verdict)
    if any(case.skipped for case in voting):
        return CheckStatus.INCONCLUSIVE
    measured = tuple(case for case in voting if not case.skipped)
    if measured and all(case.status is CheckStatus.PASS for case in measured):
        return CheckStatus.PASS
    return CheckStatus.INCONCLUSIVE


def exit_code_for(status: CheckStatus) -> ExitCode:
    if status is CheckStatus.PASS:
        return ExitCode.PASS
    if status is CheckStatus.FAIL:
        return ExitCode.FAIL
    return ExitCode.INCONCLUSIVE
