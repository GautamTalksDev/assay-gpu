"""Defined-order float64 CPU compute. No BLAS GEMM — K-loop multiply then add."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from assay.ieee import FP64_EPS


def matmul_fp64(
    left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Matmul on the last two axes. Reduction index increases 0..K-1.

    ``multiply`` and ``add`` are separate ufuncs so the order is explicit.
    """
    left_a = np.ascontiguousarray(left, dtype=np.dtype("<f8"))
    right_a = np.ascontiguousarray(right, dtype=np.dtype("<f8"))
    *batch, m_dim, k_dim = left_a.shape
    *batch_r, k_right, n_dim = right_a.shape
    if k_dim != k_right or tuple(batch) != tuple(batch_r):
        msg = f"incompatible matmul shapes {left_a.shape} @ {right_a.shape}"
        raise ValueError(msg)
    out = np.zeros((*batch, m_dim, n_dim), dtype=np.dtype("<f8"))
    for index in range(k_dim):
        prod = np.multiply(
            left_a[..., :, index : index + 1], right_a[..., index : index + 1, :]
        )
        out = np.add(out, prod)
    return np.ascontiguousarray(out, dtype=np.dtype("<f8"))


def softmax_last_axis(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Softmax on the last axis: subtract max, exp, divide by sum (K-order sum)."""
    data = np.ascontiguousarray(values, dtype=np.dtype("<f8"))
    max_val = np.max(data, axis=-1, keepdims=True)
    shifted = np.subtract(data, max_val)
    exps = np.exp(shifted)
    denom = np.sum(exps, axis=-1, keepdims=True)
    return np.ascontiguousarray(np.divide(exps, denom), dtype=np.dtype("<f8"))


def sdpa_fp64(
    query: npt.NDArray[np.float64],
    key: npt.NDArray[np.float64],
    value: npt.NDArray[np.float64],
    *,
    causal: bool,
) -> npt.NDArray[np.float64]:
    """Scaled dot-product attention. Scale is 1/sqrt(head_dim) by definition."""
    head_dim = query.shape[-1]
    scale = np.float64(1.0) / np.float64(math.sqrt(head_dim))
    scores = matmul_fp64(query, np.swapaxes(key, -1, -2))
    scores = np.multiply(scores, scale)
    if causal:
        seq = query.shape[-2]
        mask = np.triu(np.ones((seq, seq), dtype=np.bool_), k=1)
        scores = np.where(mask, np.float64("-inf"), scores)
    weights = softmax_last_axis(scores)
    return matmul_fp64(weights, value)


def sum_fp64_c_order(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Left-to-right sum of C-order ravel. Returns a 0-d array."""
    total = np.float64(0.0)
    for item in np.ascontiguousarray(values, dtype=np.dtype("<f8")).ravel(order="C"):
        total = np.add(total, np.float64(item))
    return np.asarray(total, dtype=np.dtype("<f8"))


def mean_fp64_c_order(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    data = np.ascontiguousarray(values, dtype=np.dtype("<f8"))
    total = sum_fp64_c_order(data)
    count = np.float64(data.size)
    return np.asarray(np.divide(total, count), dtype=np.dtype("<f8"))


def exp_fp64(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.ascontiguousarray(
        np.exp(np.ascontiguousarray(values, dtype=np.dtype("<f8"))),
        dtype=np.dtype("<f8"),
    )


def tanh_fp64(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.ascontiguousarray(
        np.tanh(np.ascontiguousarray(values, dtype=np.dtype("<f8"))),
        dtype=np.dtype("<f8"),
    )


def rsqrt_fp64(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    data = np.ascontiguousarray(values, dtype=np.dtype("<f8"))
    return np.ascontiguousarray(
        np.divide(np.float64(1.0), np.sqrt(data)), dtype=np.dtype("<f8")
    )


def rmsnorm_fp64(
    values: npt.NDArray[np.float64],
    weight: npt.NDArray[np.float64],
    *,
    eps: float = FP64_EPS,
) -> npt.NDArray[np.float64]:
    data = np.ascontiguousarray(values, dtype=np.dtype("<f8"))
    scale = np.ascontiguousarray(weight, dtype=np.dtype("<f8"))
    mean_square = np.mean(np.square(data), axis=-1, keepdims=True)
    denom = np.sqrt(np.add(mean_square, np.float64(eps)))
    return np.ascontiguousarray(
        np.divide(np.multiply(data, scale), denom), dtype=np.dtype("<f8")
    )


def gelu_erf_fp64(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """GELU with erf (not the tanh approximation)."""
    data = np.ascontiguousarray(values, dtype=np.dtype("<f8"))
    erf_vals = np.empty_like(data)
    sqrt2 = math.sqrt(2.0)
    flat_out = erf_vals.ravel()
    for index, item in enumerate(data.ravel()):
        flat_out[index] = math.erf(float(item) / sqrt2)
    half = np.float64(0.5)
    one = np.float64(1.0)
    return np.ascontiguousarray(
        np.multiply(np.multiply(half, data), np.add(one, erf_vals)),
        dtype=np.dtype("<f8"),
    )
