"""Live ABFT via nn.Module hooks. A fault is a log line, never an exception."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from assay.abft.check import CheckStatus


@dataclass(frozen=True, slots=True)
class WatchConfig:
    every: int
    noisefloor_dir: Path
    report_path: Path | None
    interval_seconds: float | None
    gpu_model: str | None


@dataclass(frozen=True, slots=True)
class WatchEvent:
    kind: str
    module_type: str
    status: CheckStatus
    reason: str
    residual_hex: str | None
    threshold_hex: str | None
    n_samples: int | None
    min_samples: int | None
    shape: tuple[int, ...] | None
    counts_toward_verdict: bool
    swallowed_exception: str | None
