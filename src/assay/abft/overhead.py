"""Wall-clock overhead of the checksum path alone. Not a detection threshold."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from assay.abft.reduce import CheckBackend, ones_sided_checksums


@dataclass(frozen=True, slots=True)
class OverheadMeasurement:
    shape: tuple[int, int, int]
    device: str
    backend: str
    repeats: int
    gemm_seconds: float
    checksum_seconds: float
    checksum_over_gemm: float
    dtype_name: str


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_checksum_overhead(
    *,
    shape: tuple[int, int, int],
    repeats: int,
    backend: CheckBackend,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> OverheadMeasurement:
    """Time A@B versus the ones-vector checksum of that product.

    `repeats` has no default. Caller chooses N. This is a performance
    measurement, not a pass/fail cutoff.
    """
    if repeats < 1:
        msg = "repeats must be >= 1"
        raise ValueError(msg)
    if backend is CheckBackend.TRITON and device.type != "cuda":
        msg = "Triton overhead measurement requires a CUDA device"
        raise ValueError(msg)

    rows, inner, cols = shape
    left = torch.ones((rows, inner), dtype=dtype, device=device)
    right = torch.ones((inner, cols), dtype=dtype, device=device)
    _sync(device)
    product = left @ right
    _sync(device)
    ones_sided_checksums(left, right, product, backend)
    _sync(device)

    gemm_elapsed = 0.0
    checksum_elapsed = 0.0
    for _ in range(repeats):
        _sync(device)
        started = time.perf_counter()
        product = left @ right
        _sync(device)
        gemm_elapsed += time.perf_counter() - started

        _sync(device)
        started = time.perf_counter()
        ones_sided_checksums(left, right, product, backend)
        _sync(device)
        checksum_elapsed += time.perf_counter() - started

    gemm_s = gemm_elapsed / repeats
    checksum_s = checksum_elapsed / repeats
    ratio = checksum_s / gemm_s if gemm_s > 0.0 else float("inf")
    dtype_name = {
        torch.float32: "float32",
        torch.float64: "float64",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }.get(dtype, str(dtype))
    return OverheadMeasurement(
        shape=shape,
        device=str(device),
        backend=backend.value,
        repeats=repeats,
        gemm_seconds=gemm_s,
        checksum_seconds=checksum_s,
        checksum_over_gemm=ratio,
        dtype_name=dtype_name,
    )
