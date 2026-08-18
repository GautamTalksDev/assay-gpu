"""Deterministic GPU workload suite."""

from assay.workload.elementwise import run_w06
from assay.workload.gemm import run_w01, run_w02, run_w03
from assay.workload.reductions import run_w05
from assay.workload.report import WorkloadResult
from assay.workload.sdpa import run_w04
from assay.workload.suite import (
    DoubleRunReport,
    double_run,
    run_all,
    write_double_run_report,
)
from assay.workload.transformer import run_w07

__all__ = [
    "DoubleRunReport",
    "WorkloadResult",
    "double_run",
    "run_all",
    "run_w01",
    "run_w02",
    "run_w03",
    "run_w04",
    "run_w05",
    "run_w06",
    "run_w07",
    "write_double_run_report",
]
