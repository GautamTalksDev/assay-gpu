"""W05 large reductions where accumulation order matters."""

from __future__ import annotations

import numpy as np
import torch

from assay.reference.arrays import generate_array
from assay.reference.spec import (
    DISTRIBUTION_UNIFORM_UNIT,
    WORKLOAD_REDUCE_LENGTHS,
    seed_offset,
)
from assay.workload.context import run_cuda_op
from assay.workload.report import WorkloadResult


def run_w05(lengths: tuple[int, ...] | None = None) -> list[WorkloadResult]:
    device = torch.device("cuda")
    results: list[WorkloadResult] = []
    allowed = set(WORKLOAD_REDUCE_LENGTHS if lengths is None else lengths)
    for index, length in enumerate(WORKLOAD_REDUCE_LENGTHS):
        if length not in allowed:
            continue
        array = generate_array(
            (length,), seed_offset(5, index), DISTRIBUTION_UNIFORM_UNIT
        )
        tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).to(
            device=device, dtype=torch.float32
        )

        def do_sum(data: torch.Tensor = tensor) -> torch.Tensor:
            return torch.sum(data)

        def do_mean(data: torch.Tensor = tensor) -> torch.Tensor:
            return torch.mean(data)

        results.append(
            run_cuda_op(
                do_sum,
                workload="W05",
                case=f"sum_n{length}",
                kernel_kind="sum",
            )
        )
        results.append(
            run_cuda_op(
                do_mean,
                workload="W05",
                case=f"mean_n{length}",
                kernel_kind="mean",
            )
        )
    return results
