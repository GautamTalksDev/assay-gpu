"""GEMM algorithm-based fault tolerance checksums. No pass/fail threshold here."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

_GEMM_NDIM = 2


def gemm_checksum_fp64(
    left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]
) -> np.float64:
    """sum_k (sum_i A[i,k]) * (sum_j B[k,j]), K-loop combine after numpy axis sums.

    Algebraically equals sum_{i,j} (A @ B)[i,j] in exact arithmetic.
    """
    left_a = np.ascontiguousarray(left, dtype=np.dtype("<f8"))
    right_a = np.ascontiguousarray(right, dtype=np.dtype("<f8"))
    if left_a.ndim != _GEMM_NDIM or right_a.ndim != _GEMM_NDIM:
        msg = "GEMM checksum expects 2-D factors"
        raise ValueError(msg)
    col_sums = np.atleast_1d(np.sum(left_a, axis=0, dtype=np.float64))
    row_sums = np.atleast_1d(np.sum(right_a, axis=1, dtype=np.float64))
    if col_sums.shape != row_sums.shape:
        msg = f"inner dimension mismatch {left_a.shape} vs {right_a.shape}"
        raise ValueError(msg)
    total = np.float64(0.0)
    for index in range(col_sums.shape[0]):
        total = np.add(total, np.multiply(col_sums[index], row_sums[index]))
    return np.float64(total)


def sum_elements_fp64(matrix: npt.NDArray[np.float64]) -> np.float64:
    data = np.ascontiguousarray(matrix, dtype=np.dtype("<f8"))
    return np.float64(np.sum(data, dtype=np.float64))


def normalized_checksum_residual(
    product_sum: np.float64, checksum: np.float64
) -> tuple[np.float64, np.float64]:
    """Return (abs residual, normalized residual).

    Normalization denominator is max(|checksum|, |product_sum|).
    If both are +0, normalized residual is 0 when abs residual is 0, else inf.
    Zero is IEEE-754, not a detection cutoff.
    """
    abs_residual = np.float64(np.abs(np.subtract(product_sum, checksum)))
    scale = np.float64(max(abs(float(checksum)), abs(float(product_sum))))
    if scale == np.float64(0.0):
        if abs_residual == np.float64(0.0):
            return abs_residual, np.float64(0.0)
        return abs_residual, np.float64(math.inf)
    return abs_residual, np.float64(np.divide(abs_residual, scale))
