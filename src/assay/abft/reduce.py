"""Ones-vector matvec: M @ e with e a column of ones.

This is the checksum reduction for one-sided GEMM ABFT (C @ e and B @ e).
The PyTorch path uses a GEMV. The Triton path is a separate kernel; see
assay.abft.triton_reduce and docs/ABFT.md for why that split matters.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import torch

from assay.abft.gemm import normalize_by_scale
from assay.abft.triton_reduce import ones_matvec_triton

_MATRIX_NDIM = 2


class CheckBackend(StrEnum):
    PYTORCH = "pytorch"
    TRITON = "triton"


def ones_matvec_pytorch(matrix: torch.Tensor) -> torch.Tensor:
    """Compute matrix @ e for e = ones(n), same dtype and device as matrix."""
    if matrix.ndim != _MATRIX_NDIM:
        msg = "ones matvec expects a 2-D matrix"
        raise ValueError(msg)
    ones = torch.ones(
        matrix.shape[1],
        dtype=matrix.dtype,
        device=matrix.device,
    )
    return matrix @ ones


def ones_matvec(matrix: torch.Tensor, backend: CheckBackend) -> torch.Tensor:
    if backend is CheckBackend.PYTORCH:
        return ones_matvec_pytorch(matrix)
    return ones_matvec_triton(matrix)


def ones_sided_checksums(
    left: torch.Tensor,
    right: torch.Tensor,
    product: torch.Tensor,
    backend: CheckBackend,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (C @ e, A @ (B @ e)) for e a column of ones."""
    b_e = ones_matvec(right, backend)
    a_be = left @ b_e
    c_e = ones_matvec(product, backend)
    return c_e, a_be


def checksum_abs_residual(c_e: torch.Tensor, a_be: torch.Tensor) -> float:
    """|sum(C@e) - sum(A@(B@e))| after promoting both vectors to float64."""
    c_cpu = c_e.detach().to(dtype=torch.float64, device="cpu").contiguous()
    a_cpu = a_be.detach().to(dtype=torch.float64, device="cpu").contiguous()
    product_sum = np.float64(np.sum(c_cpu.numpy(), dtype=np.float64))
    checksum = np.float64(np.sum(a_cpu.numpy(), dtype=np.float64))
    return float(np.abs(np.subtract(product_sum, checksum)))


def absolute_factor_scale(left: torch.Tensor, right: torch.Tensor) -> float:
    """eᵀ |A| |B| e as (|A| column sums) · (|B| row sums), float64 accum."""
    if left.ndim != _MATRIX_NDIM or right.ndim != _MATRIX_NDIM:
        msg = "absolute factor scale expects 2-D A and B"
        raise ValueError(msg)
    if left.shape[1] != right.shape[0]:
        msg = f"inner dimension mismatch {tuple(left.shape)} vs {tuple(right.shape)}"
        raise ValueError(msg)
    col_sums = left.detach().abs().sum(dim=0, dtype=torch.float64)
    row_sums = right.detach().abs().sum(dim=1, dtype=torch.float64)
    return float(torch.dot(col_sums, row_sums).item())


def vector_residual_parts(
    c_e: torch.Tensor,
    a_be: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[float, float, float]:
    """Return (abs residual, residual-v2 scale, residual-v2).

    Scale is eᵀ |A| |B| e. It does not depend on C.
    """
    abs_residual = np.float64(checksum_abs_residual(c_e, a_be))
    scale = np.float64(absolute_factor_scale(left, right))
    normalized = normalize_by_scale(abs_residual, scale)
    return float(abs_residual), float(scale), float(normalized)


def vector_residual_normalized(
    c_e: torch.Tensor,
    a_be: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> float:
    """residual-v2: abs checksum mismatch over eᵀ |A| |B| e."""
    _abs_residual, _scale, normalized = vector_residual_parts(c_e, a_be, left, right)
    return normalized
