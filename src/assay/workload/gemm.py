"""W01-W03 GEMM workloads. Inputs come from the reference generator."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch

from assay.reference.arrays import generate_array
from assay.reference.spec import (
    DISTRIBUTION_UNIFORM_UNIT,
    WORKLOAD_GEMM_SHAPES,
    sample_factor_seed,
    seed_offset,
)
from assay.workload.context import gemm_flags, run_cuda_op
from assay.workload.report import WorkloadResult


def gemm_numpy_pair(  # noqa: PLR0913
    m_dim: int,
    k_dim: int,
    n_dim: int,
    *,
    case_index: int,
    sample_index: int = 0,
    workload_id: int = 1,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    seed_a = sample_factor_seed(workload_id, case_index, sample_index, 0)
    seed_b = sample_factor_seed(workload_id, case_index, sample_index, 1)
    left = generate_array((m_dim, k_dim), seed_a, DISTRIBUTION_UNIFORM_UNIT)
    right = generate_array((k_dim, n_dim), seed_b, DISTRIBUTION_UNIFORM_UNIT)
    return (
        np.ascontiguousarray(left, dtype=np.dtype("<f8")),
        np.ascontiguousarray(right, dtype=np.dtype("<f8")),
    )


def _matrices(  # noqa: PLR0913
    m_dim: int,
    k_dim: int,
    n_dim: int,
    *,
    case_index: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    seed_a = seed_offset(1, case_index * 2)
    seed_b = seed_offset(1, case_index * 2 + 1)
    left = generate_array((m_dim, k_dim), seed_a, DISTRIBUTION_UNIFORM_UNIT)
    right = generate_array((k_dim, n_dim), seed_b, DISTRIBUTION_UNIFORM_UNIT)
    mat_a = torch.from_numpy(np.ascontiguousarray(left, dtype=np.float32)).to(
        device=device, dtype=dtype
    )
    mat_b = torch.from_numpy(np.ascontiguousarray(right, dtype=np.float32)).to(
        device=device, dtype=dtype
    )
    return mat_a, mat_b


def _run_shapes(
    *,
    workload: str,
    dtype: torch.dtype,
    fp16_reduced: bool,
    bf16_reduced: bool,
) -> list[WorkloadResult]:
    device = torch.device("cuda")
    results: list[WorkloadResult] = []
    with gemm_flags(fp16_reduced=fp16_reduced, bf16_reduced=bf16_reduced):
        for index, (m_dim, k_dim, n_dim) in enumerate(WORKLOAD_GEMM_SHAPES):
            mat_a, mat_b = _matrices(
                m_dim, k_dim, n_dim, case_index=index, dtype=dtype, device=device
            )

            def gemm(
                left: torch.Tensor = mat_a, right: torch.Tensor = mat_b
            ) -> torch.Tensor:
                return torch.matmul(left, right)

            results.append(
                run_cuda_op(
                    gemm,
                    workload=workload,
                    case=f"m{m_dim}_k{k_dim}_n{n_dim}",
                    kernel_kind="gemm",
                )
            )
    return results


def run_w01() -> list[WorkloadResult]:
    """GEMM, fp32."""
    return _run_shapes(
        workload="W01", dtype=torch.float32, fp16_reduced=False, bf16_reduced=False
    )


def run_w02() -> list[WorkloadResult]:
    """GEMM, bf16 inputs with fp32 accumulate (reduced-precision reduction off)."""
    return _run_shapes(
        workload="W02", dtype=torch.bfloat16, fp16_reduced=False, bf16_reduced=False
    )


def run_w03() -> list[WorkloadResult]:
    """GEMM, fp16 with fp16 accumulate requested via reduced-precision reduction."""
    return _run_shapes(
        workload="W03", dtype=torch.float16, fp16_reduced=True, bf16_reduced=False
    )
