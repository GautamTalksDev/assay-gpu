"""Triton row-sum kernel for M @ ones.

Launched as its own kernel, not fused into the GEMM that produced C.
Requires CUDA. Import is lazy so CPU CI does not need a GPU.

The reduction accumulates in float64, then casts back to the input dtype.
Integer-valued inputs whose row sums are exact in the output dtype must
match the PyTorch GEMV path bitwise (see tests).

The kernel is sm75-safe: program_id, masked load, f64 sum, store. No MMA,
TMA, wgmma, fp8, or other post-Turing ops. BLOCK is a power-of-two constexpr.
"""

from __future__ import annotations

from typing import Any

import torch

_BLOCK = 1024
_MATRIX_NDIM = 2
_KERNELS: dict[str, Any] = {}


def triton_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def _kernel() -> Any:
    cached = _KERNELS.get("row_sum")
    if cached is not None:
        return cached

    import triton  # noqa: PLC0415
    import triton.language as tl  # noqa: PLC0415

    @triton.jit  # type: ignore[untyped-decorator]
    def row_sum_kernel(  # type: ignore[no-untyped-def]  # noqa: PLR0913, PLR0917
        x_ptr,
        out_ptr,
        cols,
        stride_row,
        stride_col,
        BLOCK: tl.constexpr,  # noqa: N803
    ) -> None:
        row = tl.program_id(0)
        acc = tl.zeros((), dtype=tl.float64)
        col = 0
        while col < cols:
            offs = col + tl.arange(0, BLOCK)
            mask = offs < cols
            vals = tl.load(
                x_ptr + row * stride_row + offs * stride_col,
                mask=mask,
                other=0.0,
            )
            acc += tl.sum(vals.to(tl.float64), axis=0)
            col += BLOCK
        tl.store(out_ptr + row, acc)

    _KERNELS["row_sum"] = row_sum_kernel
    return row_sum_kernel


def ones_matvec_triton(matrix: torch.Tensor) -> torch.Tensor:
    """Row-sum of a 2-D CUDA tensor. Same contract as ones_matvec_pytorch."""
    if matrix.ndim != _MATRIX_NDIM:
        msg = "ones matvec expects a 2-D matrix"
        raise ValueError(msg)
    if not matrix.is_cuda:
        msg = "Triton checksum reduction requires a CUDA tensor"
        raise RuntimeError(msg)
    if not triton_available():
        msg = "Triton is not importable or CUDA is unavailable"
        raise RuntimeError(msg)

    rows, cols = matrix.shape
    out_f64 = torch.empty(rows, dtype=torch.float64, device=matrix.device)
    _kernel()[(rows,)](
        matrix,
        out_f64,
        cols,
        matrix.stride(0),
        matrix.stride(1),
        BLOCK=_BLOCK,
    )
    return out_f64.to(dtype=matrix.dtype)
