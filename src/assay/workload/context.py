"""Determinism flags, wall-clock timing, and whatever kernel names torch exposes."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch

from assay.workload.report import WorkloadResult

_CUBLAS_WORKSPACE = ":4096:8"


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        msg = "CUDA is required for GPU workloads"
        raise RuntimeError(msg)
    return torch.device("cuda")


def backend_snapshot() -> dict[str, str | None]:
    snapshot: dict[str, str | None] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "deterministic_algorithms": str(torch.are_deterministic_algorithms_enabled()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if torch.cuda.is_available():
        snapshot["matmul_allow_tf32"] = str(torch.backends.cuda.matmul.allow_tf32)
        snapshot["cudnn_allow_tf32"] = str(torch.backends.cudnn.allow_tf32)
        cudnn_ver = torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
        snapshot["cudnn"] = str(cudnn_ver) if cudnn_ver is not None else None
        flash = getattr(torch.backends.cuda, "flash_sdp_enabled", None)
        mem = getattr(torch.backends.cuda, "mem_efficient_sdp_enabled", None)
        math = getattr(torch.backends.cuda, "math_sdp_enabled", None)
        snapshot["sdpa_flash"] = str(flash()) if callable(flash) else None
        snapshot["sdpa_mem_efficient"] = str(mem()) if callable(mem) else None
        snapshot["sdpa_math"] = str(math()) if callable(math) else None
        pref = getattr(torch.backends.cuda, "preferred_blas_library", None)
        if callable(pref):
            try:
                snapshot["preferred_blas_library"] = str(pref())
            except RuntimeError:
                snapshot["preferred_blas_library"] = "unavailable"
    return snapshot


def kernel_label(kind: str) -> str | None:
    """Return a kernel/backend name only when torch exposes one. No guessing."""
    if kind == "gemm":
        pref = getattr(torch.backends.cuda, "preferred_blas_library", None)
        if callable(pref):
            try:
                return f"preferred_blas_library={pref()}"
            except RuntimeError:
                return "preferred_blas_library=unavailable"
        return "torch.matmul (BLAS library API not present on this torch)"
    if kind == "sdpa":
        parts: list[str] = []
        for name in (
            "flash_sdp_enabled",
            "mem_efficient_sdp_enabled",
            "math_sdp_enabled",
        ):
            fn = getattr(torch.backends.cuda, name, None)
            if callable(fn):
                parts.append(f"{name}={fn()}")
        return "sdpa[" + ",".join(parts) + "]" if parts else None
    if kind in {"sum", "mean", "exp", "tanh", "rsqrt", "transformer"}:
        return f"torch.{kind}"
    return None


@contextmanager
def gemm_flags(*, fp16_reduced: bool, bf16_reduced: bool) -> Iterator[None]:
    matmul = torch.backends.cuda.matmul
    prev_fp16 = getattr(matmul, "allow_fp16_reduced_precision_reduction", None)
    prev_bf16 = getattr(matmul, "allow_bf16_reduced_precision_reduction", None)
    if prev_fp16 is not None:
        matmul.allow_fp16_reduced_precision_reduction = fp16_reduced
    if prev_bf16 is not None:
        matmul.allow_bf16_reduced_precision_reduction = bf16_reduced
    try:
        yield
    finally:
        if prev_fp16 is not None:
            matmul.allow_fp16_reduced_precision_reduction = prev_fp16
        if prev_bf16 is not None:
            matmul.allow_bf16_reduced_precision_reduction = prev_bf16


def _prepare_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")


def run_cuda_op(
    fn: Callable[[], torch.Tensor],
    *,
    workload: str,
    case: str,
    kernel_kind: str,
) -> WorkloadResult:
    """Run `fn` on CUDA. If deterministic mode rejects the op, record that and rerun."""
    require_cuda()
    _prepare_determinism()
    nondet_reason: str | None = None
    deterministic = True
    torch.use_deterministic_algorithms(True)

    def _timed() -> tuple[torch.Tensor, float]:
        torch.cuda.synchronize()
        start = time.perf_counter()
        tensor = fn()
        torch.cuda.synchronize()
        return tensor, time.perf_counter() - start

    try:
        result, wall = _timed()
    except RuntimeError as exc:
        message = str(exc)
        if "deterministic" not in message.lower():
            raise
        nondet_reason = message
        deterministic = False
        torch.use_deterministic_algorithms(False)
        result, wall = _timed()
    finally:
        torch.use_deterministic_algorithms(False)

    return WorkloadResult(
        workload=workload,
        case=case,
        result=result,
        shape=tuple(int(s) for s in result.shape),
        wall_time_s=wall,
        kernel=kernel_label(kernel_kind),
        deterministic=deterministic,
        nondeterminism_reason=nondet_reason,
        backend=backend_snapshot(),
    )
