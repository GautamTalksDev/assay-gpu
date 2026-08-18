"""Execute the shippable assay. Zero network I/O."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from assay.abft.check import CheckStatus, GemmCheckConfig, check_gemm
from assay.noise.errors import gpu_to_fp64, max_abs_error
from assay.noise.floats import encode_f64
from assay.noise.lookup import lookup_abft_tolerance
from assay.probe.environment import probe_environment
from assay.reference.spec import (
    CHARACTERIZATION_MAX_SIDE,
    REFERENCE_GEMM_SHAPES,
    WORKLOAD_GEMM_SHAPES,
)
from assay.run.budget import (
    RunBudget,
    gemm_shapes_for,
    reduce_lengths_for,
    sdpa_shapes_for,
)
from assay.run.types import (
    AssayOperationalError,
    AssayResult,
    CaseRecord,
    overall_status,
)
from assay.workload.context import gemm_flags, run_cuda_op
from assay.workload.elementwise import run_w06
from assay.workload.gemm import gemm_numpy_pair
from assay.workload.reductions import run_w05
from assay.workload.report import WorkloadResult
from assay.workload.sdpa import run_w04
from assay.workload.transformer import run_w07

ProgressFn = Callable[[str], None]

_GEMM_DTYPES: tuple[tuple[str, torch.dtype, bool, bool], ...] = (
    ("W01", torch.float32, False, False),
    ("W02", torch.bfloat16, False, False),
    ("W03", torch.float16, True, False),
)
_BUDGET_TIMEOUT = "time budget exhausted; remaining cases skipped"
_NO_ABFT = "no GEMM ABFT in noisefloor-v1; measurement only"


def _emit(progress: ProgressFn | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _within_budget(started: float, budget: RunBudget) -> bool:
    return (time.perf_counter() - started) < float(budget.seconds)


def skipped_case(
    workload: str,
    case: str,
    shape: tuple[int, ...],
    reason: str,
    *,
    counts_toward_verdict: bool,
) -> CaseRecord:
    return CaseRecord(
        workload=workload,
        case=case,
        shape=shape,
        dtype_name=None,
        status=CheckStatus.INCONCLUSIVE,
        reason=reason,
        residual_hex=None,
        residual_decimal=None,
        threshold_hex=None,
        threshold_decimal=None,
        sample_max_hex=None,
        n_samples=None,
        min_samples=None,
        noisefloor_spec_version=None,
        golden_max_abs_error_hex=None,
        wall_time_s=0.0,
        skipped=True,
        skip_reason=reason,
        counts_toward_verdict=counts_toward_verdict,
    )


def _load_gemm_golden(
    reference_dir: Path, shape: tuple[int, int, int]
) -> npt.NDArray[np.float64] | None:
    rows, inner, cols = shape
    path = reference_dir / f"gemm_m{rows}_k{inner}_n{cols}.npz"
    if not path.is_file():
        return None
    loaded = np.load(path)
    if "c" not in loaded:
        return None
    return np.ascontiguousarray(loaded["c"], dtype=np.dtype("<f8"))


@dataclass(frozen=True, slots=True)
class _GemmJob:
    workload: str
    case: str
    shape: tuple[int, int, int]
    case_index: int
    dtype: torch.dtype
    fp16_reduced: bool
    bf16_reduced: bool
    golden: npt.NDArray[np.float64] | None


@dataclass(slots=True)
class _Ticker:
    started: float
    budget: RunBudget
    planned: int
    progress: ProgressFn | None
    done: int = 0

    def tick(self, label: str) -> None:
        self.done += 1
        elapsed = time.perf_counter() - self.started
        _emit(
            self.progress,
            f"[{self.done}/{self.planned}] {label}  elapsed={elapsed:.1f}s  "
            f"budget={self.budget.seconds}s",
        )


def _run_one_gemm(
    job: _GemmJob, noisefloor_dir: Path, gpu_model: str | None
) -> CaseRecord:
    rows, inner, cols = job.shape
    left_np, right_np = gemm_numpy_pair(rows, inner, cols, case_index=job.case_index)
    device = torch.device("cuda")
    left = torch.from_numpy(np.ascontiguousarray(left_np, dtype=np.float32)).to(
        device=device, dtype=job.dtype
    )
    right = torch.from_numpy(np.ascontiguousarray(right_np, dtype=np.float32)).to(
        device=device, dtype=job.dtype
    )
    with gemm_flags(fp16_reduced=job.fp16_reduced, bf16_reduced=job.bf16_reduced):

        def gemm() -> torch.Tensor:
            return left @ right

        ran = run_cuda_op(
            gemm, workload=job.workload, case=job.case, kernel_kind="gemm"
        )
    check = check_gemm(
        left,
        right,
        ran.result,
        GemmCheckConfig(
            noisefloor_dir=noisefloor_dir,
            workload=job.workload,
            gpu_model=gpu_model,
        ),
    )
    golden_hex = None
    if job.golden is not None:
        err = max_abs_error(gpu_to_fp64(ran.result), job.golden)
        golden_hex = encode_f64(float(err))["hex"]
    return CaseRecord(
        workload=job.workload,
        case=job.case,
        shape=check.shape,
        dtype_name=check.dtype_name,
        status=check.status,
        reason=check.reason,
        residual_hex=check.residual_hex,
        residual_decimal=check.residual_decimal,
        threshold_hex=check.threshold_hex,
        threshold_decimal=check.threshold_decimal,
        sample_max_hex=check.sample_max_hex,
        n_samples=check.n_samples,
        min_samples=check.min_samples,
        noisefloor_spec_version=check.noisefloor_spec_version,
        golden_max_abs_error_hex=golden_hex,
        wall_time_s=ran.wall_time_s,
        skipped=False,
        skip_reason=None,
        counts_toward_verdict=True,
    )


def _observe_only(
    workload: str,
    case: str,
    shape: tuple[int, ...],
    wall: float,
    extra_reason: str,
) -> CaseRecord:
    return CaseRecord(
        workload=workload,
        case=case,
        shape=shape,
        dtype_name=None,
        status=CheckStatus.INCONCLUSIVE,
        reason=extra_reason,
        residual_hex=None,
        residual_decimal=None,
        threshold_hex=None,
        threshold_decimal=None,
        sample_max_hex=None,
        n_samples=None,
        min_samples=None,
        noisefloor_spec_version=None,
        golden_max_abs_error_hex=None,
        wall_time_s=wall,
        skipped=False,
        skip_reason=None,
        counts_toward_verdict=False,
    )


def _lookup_banner(noisefloor_dir: Path, gpu_model: str | None) -> tuple[str, str]:
    found = lookup_abft_tolerance(
        noisefloor_dir,
        workload="W02",
        dtype="bfloat16",
        shape=(
            CHARACTERIZATION_MAX_SIDE,
            CHARACTERIZATION_MAX_SIDE,
            CHARACTERIZATION_MAX_SIDE,
        ),
        gpu_model=gpu_model,
    )
    return found.status.value, found.reason


def _gemm_jobs(budget: RunBudget, reference_dir: Path) -> tuple[_GemmJob, ...]:
    jobs = [
        _GemmJob(
            workload=workload,
            case=f"m{shape[0]}_k{shape[1]}_n{shape[2]}",
            shape=shape,
            case_index=WORKLOAD_GEMM_SHAPES.index(shape),
            dtype=dtype,
            fp16_reduced=fp16_red,
            bf16_reduced=bf16_red,
            golden=None,
        )
        for workload, dtype, fp16_red, bf16_red in _GEMM_DTYPES
        for shape in gemm_shapes_for(budget)
    ]
    jobs.extend(
        _GemmJob(
            workload="W01",
            case=f"golden_m{shape[0]}_k{shape[1]}_n{shape[2]}",
            shape=shape,
            case_index=index,
            dtype=torch.float32,
            fp16_reduced=False,
            bf16_reduced=False,
            golden=_load_gemm_golden(reference_dir, shape),
        )
        for index, shape in enumerate(REFERENCE_GEMM_SHAPES)
    )
    return tuple(jobs)


def _run_gemm_job(
    job: _GemmJob,
    clock: _Ticker,
    noisefloor_dir: Path,
    gpu_model: str,
) -> CaseRecord:
    if not _within_budget(clock.started, clock.budget):
        clock.tick(f"{job.workload} {job.case} SKIP")
        return skipped_case(
            job.workload,
            job.case,
            job.shape,
            _BUDGET_TIMEOUT,
            counts_toward_verdict=True,
        )
    try:
        record = _run_one_gemm(job, noisefloor_dir, gpu_model)
    except RuntimeError as exc:
        clock.tick(f"{job.workload} {job.case} ERROR")
        return skipped_case(
            job.workload,
            job.case,
            job.shape,
            str(exc),
            counts_toward_verdict=True,
        )
    clock.tick(f"{job.workload} {job.case} {record.status.value}")
    return record


def _w07_reason(result: WorkloadResult, reference_dir: Path) -> str:
    token_path = reference_dir / "w07_tokens.npz"
    if not token_path.is_file():
        return _NO_ABFT
    gold = np.ascontiguousarray(np.load(token_path)["greedy"])
    got = result.result.detach().to(device="cpu").numpy()
    if got.shape == gold.shape and np.array_equal(got, gold):
        return _NO_ABFT + "; greedy tokens match committed golden"
    return _NO_ABFT + "; greedy tokens differ from committed golden"


def _const_no_abft(_result: WorkloadResult) -> str:
    return _NO_ABFT


def _observe_group(
    workload: str,
    launch: Callable[[], list[WorkloadResult]],
    reason_for: Callable[[WorkloadResult], str],
    clock: _Ticker,
) -> list[CaseRecord]:
    if not _within_budget(clock.started, clock.budget):
        clock.tick(f"{workload} all SKIP")
        return [
            skipped_case(
                workload,
                "all",
                (),
                _BUDGET_TIMEOUT,
                counts_toward_verdict=False,
            )
        ]
    try:
        results = launch()
    except RuntimeError as exc:
        clock.tick(f"{workload} all ERROR")
        return [
            skipped_case(workload, "all", (), str(exc), counts_toward_verdict=False)
        ]
    records: list[CaseRecord] = []
    for result in results:
        if not _within_budget(clock.started, clock.budget):
            records.append(
                skipped_case(
                    workload,
                    result.case,
                    result.shape,
                    _BUDGET_TIMEOUT,
                    counts_toward_verdict=False,
                )
            )
            clock.tick(f"{workload} {result.case} SKIP")
            continue
        records.append(
            _observe_only(
                workload,
                result.case,
                result.shape,
                result.wall_time_s,
                reason_for(result),
            )
        )
        clock.tick(f"{workload} {result.case} INCONCLUSIVE")
    return records


def execute_assay(
    *,
    budget: RunBudget,
    noisefloor_dir: Path,
    reference_dir: Path,
    device_index: int = 0,
    progress: ProgressFn | None = None,
) -> AssayResult:
    started = time.perf_counter()
    probe = probe_environment()
    if not probe.cuda_available or probe.gpu_count <= 0:
        msg = "no CUDA GPU visible; assay run cannot execute workloads"
        raise AssayOperationalError(msg)
    if device_index < 0 or device_index >= probe.gpu_count:
        msg = f"--device {device_index} is out of range (gpu_count={probe.gpu_count})"
        raise AssayOperationalError(msg)
    torch.cuda.set_device(device_index)
    gpu_model = probe.devices[device_index].model_key
    floor_status, floor_reason = _lookup_banner(noisefloor_dir, gpu_model)
    _emit(
        progress,
        f"probe gpu_count={probe.gpu_count} model={gpu_model} "
        f"noisefloor={floor_status}",
    )
    jobs = _gemm_jobs(budget, reference_dir)
    planned = (
        len(jobs)
        + len(sdpa_shapes_for(budget))
        + 2 * len(reduce_lengths_for(budget))
        + 3
        + 1
    )
    clock = _Ticker(started=started, budget=budget, planned=planned, progress=progress)
    cases: list[CaseRecord] = [
        _run_gemm_job(job, clock, noisefloor_dir, gpu_model) for job in jobs
    ]
    cases.extend(
        _observe_group(
            "W04",
            lambda: run_w04(shapes=sdpa_shapes_for(budget)),
            _const_no_abft,
            clock,
        )
    )
    cases.extend(
        _observe_group(
            "W05",
            lambda: run_w05(lengths=reduce_lengths_for(budget)),
            _const_no_abft,
            clock,
        )
    )
    cases.extend(_observe_group("W06", run_w06, _const_no_abft, clock))
    cases.extend(
        _observe_group(
            "W07",
            run_w07,
            lambda result: _w07_reason(result, reference_dir),
            clock,
        )
    )
    packed = tuple(cases)
    return AssayResult(
        probe=probe,
        gpu_model=gpu_model,
        noisefloor_status=floor_status,
        noisefloor_reason=floor_reason,
        budget=budget,
        cases=packed,
        status=overall_status(packed),
        elapsed_s=time.perf_counter() - started,
    )
