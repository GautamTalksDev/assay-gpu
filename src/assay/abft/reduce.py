"""Ones-vector matvec: M @ e with e a column of ones.

This is the checksum reduction for one-sided GEMM ABFT (C @ e and B @ e).
The PyTorch path uses a GEMV. The Triton path is a separate kernel; see
assay.abft.triton_reduce and docs/ABFT.md for why that split matters.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import torch

from assay.abft.gemm import normalized_checksum_residual
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


def vector_residual_normalized(c_e: torch.Tensor, a_be: torch.Tensor) -> float:
    """Normalized |sum(C@e) - sum(A@(B@e))| after promoting both vectors to fp64.

    Algebraically the same family as noisefloor-v1's scalar checksum residual.
    Reduction order can differ from summing every element of C; that gap is
    why characterized configs still have an ambiguous band.
    """
    c_cpu = c_e.detach().to(dtype=torch.float64, device="cpu").contiguous()
    a_cpu = a_be.detach().to(dtype=torch.float64, device="cpu").contiguous()
    product_sum = np.float64(np.sum(c_cpu.numpy(), dtype=np.float64))
    checksum = np.float64(np.sum(a_cpu.numpy(), dtype=np.float64))
    _abs_residual, normalized = normalized_checksum_residual(product_sum, checksum)
    return float(normalized)
