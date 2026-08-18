"""One-sided GEMM ABFT detector. Thresholds come only from data/noisefloor."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import torch

from assay.abft.reduce import (
    CheckBackend,
    ones_sided_checksums,
    vector_residual_normalized,
)
from assay.noise.floats import encode_f64
from assay.noise.lookup import (
    CharacterizationStatus,
    ToleranceLookup,
    lookup_abft_tolerance,
)
from assay.noise.methodology import load_methodology

_GEMM_NDIM = 2


class CheckStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class GemmCheckConfig:
    noisefloor_dir: Path
    workload: str
    gpu_model: str | None = None
    backend: CheckBackend = CheckBackend.PYTORCH
    checksum_on_cpu: bool = False


@dataclass(frozen=True, slots=True)
class CheckResult:
    status: CheckStatus
    residual: float
    residual_hex: str
    residual_decimal: str
    threshold: float | None
    threshold_hex: str | None
    threshold_decimal: str | None
    sample_max: float | None
    sample_max_hex: str | None
    noisefloor_spec_version: str
    n_samples: int
    min_samples: int
    reason: str
    backend: str
    lookup_status: str
    dtype_name: str
    shape: tuple[int, int, int]


def _dtype_name(tensor: torch.Tensor) -> str:
    mapping = {
        torch.float32: "float32",
        torch.float64: "float64",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }
    name = mapping.get(tensor.dtype)
    if name is None:
        msg = f"unsupported GEMM dtype for ABFT: {tensor.dtype}"
        raise ValueError(msg)
    return name


def _validate_factors(
    left: torch.Tensor, right: torch.Tensor, product: torch.Tensor
) -> tuple[int, int, int]:
    ranks = (left.ndim, right.ndim, product.ndim)
    if ranks != (_GEMM_NDIM, _GEMM_NDIM, _GEMM_NDIM):
        msg = "check_gemm expects 2-D A, B, and C"
        raise ValueError(msg)
    if left.dtype != right.dtype or left.dtype != product.dtype:
        msg = "A, B, and C must share a dtype"
        raise ValueError(msg)
    rows, inner = left.shape
    inner_b, cols = right.shape
    rows_c, cols_c = product.shape
    if inner != inner_b or rows != rows_c or cols != cols_c:
        msg = (
            f"C shape {tuple(product.shape)} is not A {tuple(left.shape)} "
            f"@ B {tuple(right.shape)}"
        )
        raise ValueError(msg)
    return rows, inner, cols


def _prepare_for_checksum(
    left: torch.Tensor,
    right: torch.Tensor,
    product: torch.Tensor,
    config: GemmCheckConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if config.backend is CheckBackend.TRITON and config.checksum_on_cpu:
        msg = "Triton checksum reduction cannot run on CPU"
        raise ValueError(msg)
    if config.checksum_on_cpu:
        return (
            left.detach().to(device="cpu"),
            right.detach().to(device="cpu"),
            product.detach().to(device="cpu"),
        )
    return left, right, product


def decide_from_lookup(
    residual: float, lookup: ToleranceLookup
) -> tuple[CheckStatus, str]:
    """Map a residual onto PASS / FAIL / INCONCLUSIVE using noisefloor bounds.

    Characterized:
      residual <= p-quantile              -> PASS
      p-quantile < residual <= sample_max -> INCONCLUSIVE (ambiguous band)
      residual > sample_max               -> FAIL
    Uncharacterized: INCONCLUSIVE, never FAIL.
    Non-finite residual on a characterized config is FAIL (outside the
    measured finite sample).
    """
    if lookup.status is CharacterizationStatus.UNCHARACTERIZED:
        return CheckStatus.INCONCLUSIVE, lookup.reason
    if lookup.p_quantile_residual_hex is None or lookup.sample_max_residual_hex is None:
        return (
            CheckStatus.INCONCLUSIVE,
            "characterized lookup missing quantile or sample max",
        )
    if not math.isfinite(residual):
        return (
            CheckStatus.FAIL,
            "non-finite checksum residual on a characterized configuration",
        )
    threshold = float.fromhex(lookup.p_quantile_residual_hex)
    sample_max = float.fromhex(lookup.sample_max_residual_hex)
    if residual <= threshold:
        return (
            CheckStatus.PASS,
            "residual at or below the empirical noisefloor quantile",
        )
    if residual > sample_max:
        return (
            CheckStatus.FAIL,
            "residual exceeds every measured noisefloor sample",
        )
    return (
        CheckStatus.INCONCLUSIVE,
        "residual in the ambiguous band (above the quantile, at most sample max)",
    )


def _encoded_or_none(value: float | None) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    payload = encode_f64(value)
    return payload["hex"], payload["decimal"]


def check_gemm(
    left: torch.Tensor,
    right: torch.Tensor,
    product: torch.Tensor,
    config: GemmCheckConfig,
) -> CheckResult:
    """Compare (A @ (B @ e)) against (C @ e). Never FAIL if uncharacterized."""
    rows, inner, cols = _validate_factors(left, right, product)
    dtype_name = _dtype_name(product)
    methodology = load_methodology(config.noisefloor_dir)
    lookup = lookup_abft_tolerance(
        config.noisefloor_dir,
        workload=config.workload,
        dtype=dtype_name,
        shape=(rows, inner, cols),
        gpu_model=config.gpu_model,
    )
    chk_a, chk_b, chk_c = _prepare_for_checksum(left, right, product, config)
    c_e, a_be = ones_sided_checksums(chk_a, chk_b, chk_c, config.backend)
    residual = vector_residual_normalized(c_e, a_be)
    status, reason = decide_from_lookup(residual, lookup)
    residual_enc = encode_f64(residual)
    threshold: float | None = None
    sample_max: float | None = None
    if lookup.p_quantile_residual_hex is not None:
        threshold = float.fromhex(lookup.p_quantile_residual_hex)
    if lookup.sample_max_residual_hex is not None:
        sample_max = float.fromhex(lookup.sample_max_residual_hex)
    _thr_hex, thr_dec = _encoded_or_none(threshold)
    return CheckResult(
        status=status,
        residual=residual,
        residual_hex=residual_enc["hex"],
        residual_decimal=residual_enc["decimal"],
        threshold=threshold,
        threshold_hex=lookup.p_quantile_residual_hex,
        threshold_decimal=thr_dec,
        sample_max=sample_max,
        sample_max_hex=lookup.sample_max_residual_hex,
        noisefloor_spec_version=methodology.spec_id,
        n_samples=lookup.n_samples,
        min_samples=lookup.min_samples,
        reason=reason,
        backend=config.backend.value,
        lookup_status=lookup.status.value,
        dtype_name=dtype_name,
        shape=(rows, inner, cols),
    )
