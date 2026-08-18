"""Ones-vector matvec: M @ e with e a column of ones.

This is the checksum reduction for one-sided GEMM ABFT (C @ e and B @ e).
The PyTorch path uses a GEMV. The Triton path is a separate kernel; see
assay.abft.triton_reduce and docs/ABFT.md for why that split matters.
"""

from __future__ import annotations

from enum import StrEnum

import torch

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
