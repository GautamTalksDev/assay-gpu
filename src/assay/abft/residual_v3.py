"""Residual-v3: elementwise per-row checksum comparison.

max_i |d_i - d'_i| / scale_i  where
  d     = C @ e          (length M, fp64)
  d'    = A @ (B @ e)    (length M, fp64)
  scale = |A| @ (|B| @ e) (length M, fp64, nonneg)

Pure function, no I/O. Does not replace residual-v2.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

_MATRIX_NDIM = 2
_SUBSAMPLE_ROWS = 256
_SUBSAMPLE_SEED = 20260819


def _fixed_row_indices(m: int, n_rows: int = _SUBSAMPLE_ROWS) -> list[int]:
    """Deterministic subsample of row indices, same across all samples."""
    if m <= n_rows:
        return list(range(m))
    rng = np.random.Generator(np.random.PCG64(_SUBSAMPLE_SEED))
    return sorted(rng.choice(m, size=n_rows, replace=False).tolist())


def residual_v3(
    left: torch.Tensor,
    right: torch.Tensor,
    product: torch.Tensor,
) -> dict[str, Any]:
    """Elementwise per-row residual. All arithmetic in fp64.

    Parameters are the original-dtype A, B, C tensors (2-D).
    Returns a dict with r_max, r_median, r_p99, r_rows, n_scale_zero.
    """
    if left.ndim != _MATRIX_NDIM or right.ndim != _MATRIX_NDIM or product.ndim != _MATRIX_NDIM:
        msg = "residual_v3 expects 2-D A, B, C"
        raise ValueError(msg)
    m, k = left.shape
    k2, n = right.shape
    m2, n2 = product.shape
    if k != k2 or m != m2 or n != n2:
        msg = f"shape mismatch: A {tuple(left.shape)}, B {tuple(right.shape)}, C {tuple(product.shape)}"
        raise ValueError(msg)

    a = left.detach().to(dtype=torch.float64, device="cpu")
    b = right.detach().to(dtype=torch.float64, device="cpu")
    c = product.detach().to(dtype=torch.float64, device="cpu")

    e = torch.ones(n, dtype=torch.float64)

    d = c @ e  # (M,)
    be = b @ e  # (K,)
    d_prime = a @ be  # (M,)

    abs_diff = torch.abs(d - d_prime)  # (M,)

    abs_be = b.abs() @ e  # (K,)
    scale = a.abs() @ abs_be  # (M,)

    n_scale_zero = int((scale == 0.0).sum().item())

    r = torch.where(
        scale > 0.0,
        abs_diff / scale,
        torch.where(abs_diff == 0.0, torch.zeros_like(abs_diff), torch.full_like(abs_diff, float("inf"))),
    )

    r_np = r.numpy()
    r_max = float(np.max(r_np))
    r_median = float(np.median(r_np))
    r_p99 = float(np.percentile(r_np, 99))

    row_indices = _fixed_row_indices(m)
    r_rows = [float(r_np[i]) for i in row_indices]

    return {
        "r_max": r_max,
        "r_median": r_median,
        "r_p99": r_p99,
        "r_rows": r_rows,
        "row_indices": row_indices,
        "n_scale_zero": n_scale_zero,
    }
