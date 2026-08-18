"""Workload result records. No comparison against references in this checkpoint."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class WorkloadResult:
    workload: str
    case: str
    result: torch.Tensor
    shape: tuple[int, ...]
    wall_time_s: float
    kernel: str | None
    deterministic: bool
    nondeterminism_reason: str | None
    backend: dict[str, str | None]
