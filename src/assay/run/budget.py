"""Time-budget shape filters. These are wall-clock limits, not detection cutoffs."""

from __future__ import annotations

from dataclasses import dataclass

from assay.reference.spec import (
    CHARACTERIZATION_MAX_SIDE,
    WORKLOAD_ELEMENTWISE_SHAPE,
    WORKLOAD_GEMM_SHAPES,
    WORKLOAD_REDUCE_LENGTHS,
    WORKLOAD_SDPA_SHAPES,
)

QUICK_SECONDS = 120
DEFAULT_SECONDS = 600
THOROUGH_SECONDS = 3600
QUICK_MAX_SIDE = 512


@dataclass(frozen=True, slots=True)
class RunBudget:
    name: str
    seconds: int
    max_gemm_side: int
    max_sdpa_seq: int
    max_reduce_length: int
    max_elementwise_side: int


def budget_from_flags(*, quick: bool, thorough: bool) -> RunBudget:
    if quick and thorough:
        msg = "pass only one of --quick and --thorough"
        raise ValueError(msg)
    if quick:
        return RunBudget(
            name="quick",
            seconds=QUICK_SECONDS,
            max_gemm_side=QUICK_MAX_SIDE,
            max_sdpa_seq=128,
            max_reduce_length=WORKLOAD_REDUCE_LENGTHS[0],
            max_elementwise_side=min(WORKLOAD_ELEMENTWISE_SHAPE),
        )
    if thorough:
        return RunBudget(
            name="thorough",
            seconds=THOROUGH_SECONDS,
            max_gemm_side=max(max(shape) for shape in WORKLOAD_GEMM_SHAPES),
            max_sdpa_seq=max(shape[2] for shape in WORKLOAD_SDPA_SHAPES),
            max_reduce_length=max(WORKLOAD_REDUCE_LENGTHS),
            max_elementwise_side=max(WORKLOAD_ELEMENTWISE_SHAPE),
        )
    return RunBudget(
        name="default",
        seconds=DEFAULT_SECONDS,
        max_gemm_side=CHARACTERIZATION_MAX_SIDE,
        max_sdpa_seq=512,
        max_reduce_length=max(WORKLOAD_REDUCE_LENGTHS),
        max_elementwise_side=max(WORKLOAD_ELEMENTWISE_SHAPE),
    )


def gemm_shapes_for(budget: RunBudget) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        shape for shape in WORKLOAD_GEMM_SHAPES if max(shape) <= budget.max_gemm_side
    )


def sdpa_shapes_for(budget: RunBudget) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(
        shape for shape in WORKLOAD_SDPA_SHAPES if shape[2] <= budget.max_sdpa_seq
    )


def reduce_lengths_for(budget: RunBudget) -> tuple[int, ...]:
    return tuple(
        length
        for length in WORKLOAD_REDUCE_LENGTHS
        if length <= budget.max_reduce_length
    )
