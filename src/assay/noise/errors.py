"""Compare a GPU tensor to an fp64 reference. No cutoff is applied."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def gpu_to_fp64(tensor: torch.Tensor) -> npt.NDArray[np.float64]:
    """Promote stored bits to float64 (fp16/bf16/fp32 finite values are exact)."""
    array = tensor.detach().to(dtype=torch.float64, device="cpu").contiguous().numpy()
    return np.ascontiguousarray(array, dtype=np.dtype("<f8"))


def max_abs_error(
    gpu: npt.NDArray[np.float64], reference: npt.NDArray[np.float64]
) -> np.float64:
    diff = np.abs(np.subtract(gpu, reference))
    return np.float64(np.max(diff))


def max_rel_error(
    gpu: npt.NDArray[np.float64], reference: npt.NDArray[np.float64]
) -> np.float64:
    """Max |g-r|/|r| where r != 0. All-zero r: 0 iff g==r else inf."""
    ref = np.ascontiguousarray(reference, dtype=np.dtype("<f8"))
    got = np.ascontiguousarray(gpu, dtype=np.dtype("<f8"))
    nonzero = ref != np.float64(0.0)
    if not np.any(nonzero):
        if np.all(got == ref):
            return np.float64(0.0)
        return np.float64(np.inf)
    rel = np.abs(np.divide(np.subtract(got[nonzero], ref[nonzero]), ref[nonzero]))
    return np.float64(np.max(rel))
