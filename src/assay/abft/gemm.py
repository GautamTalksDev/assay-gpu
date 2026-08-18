"""GEMM algorithm-based fault tolerance checksums. No pass/fail threshold here."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

_GEMM_NDIM = 2
RESIDUAL_VERSION = "residual-v2"


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


def normalize_by_scale(abs_residual: np.float64, scale: np.float64) -> np.float64:
    """Divide abs residual by a nonnegative scale. IEEE zero, not a cutoff.

    If scale is +0, return 0 when abs residual is 0, else inf.
    """
    if scale == np.float64(0.0):
        if abs_residual == np.float64(0.0):
            return np.float64(0.0)
        return np.float64(math.inf)
    return np.float64(np.divide(abs_residual, scale))


def normalized_checksum_residual(
    product_sum: np.float64, checksum: np.float64
) -> tuple[np.float64, np.float64]:
    """Return (abs residual, residual-v1 normalized residual).

    residual-v1 denominator is max(|checksum|, |product_sum|). That
    normalizer is void for detection. Kept so the defective formula stays
    testable. Detector and noisefloor use residual-v2.
    """
    abs_residual = np.float64(np.abs(np.subtract(product_sum, checksum)))
    scale = np.float64(max(abs(float(checksum)), abs(float(product_sum))))
    return abs_residual, normalize_by_scale(abs_residual, scale)
