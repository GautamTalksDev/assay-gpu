"""W04 scaled dot-product attention."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional

from assay.reference.arrays import generate_array
from assay.reference.spec import (
    DISTRIBUTION_UNIFORM_UNIT,
    WORKLOAD_SDPA_SHAPES,
    seed_offset,
)
from assay.workload.context import run_cuda_op
from assay.workload.report import WorkloadResult


def run_w04(
    shapes: tuple[tuple[int, int, int, int], ...] | None = None,
) -> list[WorkloadResult]:
    device = torch.device("cuda")
    results: list[WorkloadResult] = []
    allowed = set(WORKLOAD_SDPA_SHAPES if shapes is None else shapes)
    for index, (batch, heads, seq, head_dim) in enumerate(WORKLOAD_SDPA_SHAPES):
        shape = (batch, heads, seq, head_dim)
        if shape not in allowed:
            continue
        parts: list[torch.Tensor] = []
        for off in range(3):
            array = generate_array(
                shape, seed_offset(4, index * 3 + off), DISTRIBUTION_UNIFORM_UNIT
            )
            parts.append(
                torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).to(
                    device=device, dtype=torch.float32
                )
            )
        query, key, value = parts

        def sdpa(
            q: torch.Tensor = query,
            k: torch.Tensor = key,
            v: torch.Tensor = value,
        ) -> torch.Tensor:
            return functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=False
            )

        results.append(
            run_cuda_op(
                sdpa,
                workload="W04",
                case=f"b{batch}_h{heads}_s{seq}_d{head_dim}",
                kernel_kind="sdpa",
            )
        )
    return results
