"""Extract GEMM factors from hooked modules. Never raises into the host."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def workload_for_dtype(dtype: torch.dtype) -> str | None:
    if dtype == torch.float32:
        return "W01"
    if dtype == torch.bfloat16:
        return "W02"
    if dtype == torch.float16:
        return "W03"
    return None


def linear_gemm_factors(  # noqa: PLR0911
    module: nn.Linear,
    inputs: tuple[Any, ...],
    output: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Map y = x @ W.T [+ b] onto check_gemm(A, B, C) without touching y."""
    if not inputs:
        return None
    raw = inputs[0]
    if not isinstance(raw, torch.Tensor) or not isinstance(output, torch.Tensor):
        return None
    if raw.ndim < 1 or output.ndim < 1:
        return None
    if raw.shape[-1] != module.in_features:
        return None
    if output.shape[-1] != module.out_features:
        return None
    rows = raw.numel() // module.in_features
    if rows * module.in_features != raw.numel():
        return None
    if output.numel() != rows * module.out_features:
        return None
    left = raw.reshape(rows, module.in_features).detach()
    product = output.reshape(rows, module.out_features).detach()
    if module.bias is not None:
        product = product - module.bias.detach()
    right = module.weight.detach().transpose(0, 1)
    return left, right, product
