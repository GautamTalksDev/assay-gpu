"""Run the seven workloads and optionally record a double-run bitwise check."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from assay.workload.context import require_cuda
from assay.workload.elementwise import run_w06
from assay.workload.gemm import run_w01, run_w02, run_w03
from assay.workload.reductions import run_w05
from assay.workload.report import WorkloadResult
from assay.workload.sdpa import run_w04
from assay.workload.transformer import run_w07

Runner = Callable[[], list[WorkloadResult]]

RUNNERS: tuple[tuple[str, Runner], ...] = (
    ("W01", run_w01),
    ("W02", run_w02),
    ("W03", run_w03),
    ("W04", run_w04),
    ("W05", run_w05),
    ("W06", run_w06),
    ("W07", run_w07),
)


@dataclass(frozen=True, slots=True)
class CaseBitwise:
    workload: str
    case: str
    shape: tuple[int, ...]
    bitwise_identical: bool
    deterministic_run1: bool
    deterministic_run2: bool
    nondeterminism_reason: str | None
    kernel: str | None


@dataclass(frozen=True, slots=True)
class DoubleRunReport:
    device_name: str
    torch_version: str
    cuda_version: str | None
    cases: tuple[CaseBitwise, ...]

    @property
    def all_bitwise_identical(self) -> bool:
        return all(case.bitwise_identical for case in self.cases)


def run_all() -> list[WorkloadResult]:
    require_cuda()
    out: list[WorkloadResult] = []
    for _, runner in RUNNERS:
        out.extend(runner())
    return out


def double_run() -> DoubleRunReport:
    """Run each case twice. Mismatch is recorded, not treated as a bug."""
    require_cuda()
    records: list[CaseBitwise] = []
    for _, runner in RUNNERS:
        first = runner()
        second = runner()
        if len(first) != len(second):
            msg = f"case count changed between runs: {len(first)} vs {len(second)}"
            raise RuntimeError(msg)
        for left, right in zip(first, second, strict=True):
            if left.case != right.case or left.workload != right.workload:
                msg = (
                    f"case order changed: {left.workload}/{left.case} "
                    f"vs {right.workload}/{right.case}"
                )
                raise RuntimeError(msg)
            records.append(
                CaseBitwise(
                    workload=left.workload,
                    case=left.case,
                    shape=left.shape,
                    bitwise_identical=bool(torch.equal(left.result, right.result)),
                    deterministic_run1=left.deterministic,
                    deterministic_run2=right.deterministic,
                    nondeterminism_reason=left.nondeterminism_reason
                    or right.nondeterminism_reason,
                    kernel=left.kernel,
                )
            )
            del left, right
        del first, second
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return DoubleRunReport(
        device_name=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        cases=tuple(records),
    )


def write_double_run_report(report: DoubleRunReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "device_name": report.device_name,
        "torch_version": report.torch_version,
        "cuda_version": report.cuda_version,
        "all_bitwise_identical": report.all_bitwise_identical,
        "note": (
            "bitwise_identical is a measurement. "
            "False is input to CP-3, not a test failure."
        ),
        "cases": [
            {
                "workload": case.workload,
                "case": case.case,
                "shape": list(case.shape),
                "bitwise_identical": case.bitwise_identical,
                "deterministic_run1": case.deterministic_run1,
                "deterministic_run2": case.deterministic_run2,
                "nondeterminism_reason": case.nondeterminism_reason,
                "kernel": case.kernel,
            }
            for case in report.cases
        ],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
