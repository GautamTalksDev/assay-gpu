"""W06 elementwise transcendentals."""

from __future__ import annotations

import numpy as np
import torch

from assay.reference.arrays import generate_array
from assay.reference.spec import (
    DISTRIBUTION_UNIFORM_POS,
    DISTRIBUTION_UNIFORM_UNIT,
    WORKLOAD_ELEMENTWISE_SHAPE,
    seed_offset,
)
from assay.workload.context import run_cuda_op
from assay.workload.report import WorkloadResult


def run_w06() -> list[WorkloadResult]:
    device = torch.device("cuda")
    unit = generate_array(
        WORKLOAD_ELEMENTWISE_SHAPE, seed_offset(6, 0), DISTRIBUTION_UNIFORM_UNIT
    )
    pos = generate_array(
        WORKLOAD_ELEMENTWISE_SHAPE, seed_offset(6, 1), DISTRIBUTION_UNIFORM_POS
    )
    x_unit = torch.from_numpy(np.ascontiguousarray(unit, dtype=np.float32)).to(
        device=device, dtype=torch.float32
    )
    x_pos = torch.from_numpy(np.ascontiguousarray(pos, dtype=np.float32)).to(
        device=device, dtype=torch.float32
    )

    def do_exp(data: torch.Tensor = x_unit) -> torch.Tensor:
        return torch.exp(data)

    def do_tanh(data: torch.Tensor = x_unit) -> torch.Tensor:
        return torch.tanh(data)

    def do_rsqrt(data: torch.Tensor = x_pos) -> torch.Tensor:
        return torch.rsqrt(data)

    return [
        run_cuda_op(do_exp, workload="W06", case="exp", kernel_kind="exp"),
        run_cuda_op(do_tanh, workload="W06", case="tanh", kernel_kind="tanh"),
        run_cuda_op(do_rsqrt, workload="W06", case="rsqrt", kernel_kind="rsqrt"),
    ]
