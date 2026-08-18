"""Run GEMM characterization samples and persist versioned JSON."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from assay.abft.gemm import (
    gemm_checksum_fp64,
    normalized_checksum_residual,
    sum_elements_fp64,
)
from assay.noise.blas import available_blas_libraries, blas_library
from assay.noise.errors import gpu_to_fp64, max_abs_error, max_rel_error
from assay.noise.floats import decode_f64, encode_f64
from assay.noise.methodology import load_methodology
from assay.probe.identity import model_key, read_identity
from assay.probe.telemetry import read_telemetry, telemetry_dict
from assay.reference.compute import matmul_fp64
from assay.reference.spec import CHARACTERIZATION_MAX_SIDE, WORKLOAD_GEMM_SHAPES
from assay.workload.context import gemm_flags, require_cuda
from assay.workload.gemm import gemm_numpy_pair


def tool_version() -> str:
    try:
        return str(metadata.version("assay-gpu"))
    except metadata.PackageNotFoundError:
        return "unknown"


@dataclass(frozen=True, slots=True)
class GemmTarget:
    workload: str
    dtype_name: str
    torch_dtype: torch.dtype
    fp16_reduced: bool
    bf16_reduced: bool
    shape: tuple[int, int, int]
    case_index: int


DEFAULT_TARGETS: tuple[tuple[str, str, torch.dtype, bool, bool], ...] = (
    ("W01", "float32", torch.float32, False, False),
    ("W02", "bfloat16", torch.bfloat16, False, False),
    ("W03", "float16", torch.float16, True, False),
)


def default_shapes(*, include_large: bool) -> tuple[tuple[int, int, int], ...]:
    selected: list[tuple[int, int, int]] = []
    for shape in WORKLOAD_GEMM_SHAPES:
        if not include_large and max(shape) > CHARACTERIZATION_MAX_SIDE:
            continue
        selected.append(shape)
    return tuple(selected)


def expand_targets(
    *,
    include_large: bool,
    workload: str | None,
    shape_filter: tuple[int, int, int] | None,
) -> list[GemmTarget]:
    shapes = default_shapes(include_large=include_large)
    targets: list[GemmTarget] = []
    for work, dtype_name, torch_dtype, fp16_reduced, bf16_reduced in DEFAULT_TARGETS:
        if workload is not None and work != workload:
            continue
        for case_index, shape in enumerate(WORKLOAD_GEMM_SHAPES):
            if shape not in shapes:
                continue
            if shape_filter is not None and shape != shape_filter:
                continue
            targets.append(
                GemmTarget(
                    workload=work,
                    dtype_name=dtype_name,
                    torch_dtype=torch_dtype,
                    fp16_reduced=fp16_reduced,
                    bf16_reduced=bf16_reduced,
                    shape=shape,
                    case_index=case_index,
                )
            )
    return targets


def _result_sha256(tensor: torch.Tensor) -> str:
    """SHA-256 of dtype, shape, and raw tensor bytes.

    NumPy has no bfloat16 dtype, so this must not call Tensor.numpy() on
    the original dtype. uint8 is representable; the dtype string in the
    header keeps float32 and bfloat16 with identical payloads distinct.
    """
    cpu = tensor.detach().contiguous().cpu()
    header = f"{cpu.dtype}|{tuple(int(dim) for dim in cpu.shape)}".encode("ascii")
    payload = cpu.view(torch.uint8).contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(payload.tobytes())
    return digest.hexdigest()


def _run_one_gemm(left_gpu: torch.Tensor, right_gpu: torch.Tensor) -> torch.Tensor:
    torch.cuda.synchronize()
    product = torch.matmul(left_gpu, right_gpu)
    torch.cuda.synchronize()
    return product


def characterize_gemm(  # noqa: PLR0913, PLR0915
    *,
    noisefloor_dir: Path,
    repeats: int,
    device_index: int,
    include_large: bool,
    workload: str | None,
    shape_filter: tuple[int, int, int] | None,
) -> Path:
    if repeats < 1:
        msg = "repeats must be >= 1"
        raise ValueError(msg)
    require_cuda()
    torch.cuda.set_device(device_index)
    methodology = load_methodology(noisefloor_dir)
    identity = read_identity(device_index)
    gpu_model = model_key(identity)
    started = datetime.now(UTC)
    targets = expand_targets(
        include_large=include_large, workload=workload, shape_filter=shape_filter
    )
    if not targets:
        msg = "no GEMM targets matched the filters"
        raise ValueError(msg)
    blas_names = available_blas_libraries()
    ref_cache: dict[tuple[int, int, int], npt.NDArray[np.float64]] = {}
    checksum_cache: dict[tuple[int, int, int], np.float64] = {}
    factor_cache: dict[
        tuple[int, int, int], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ] = {}
    samples: list[dict[str, Any]] = []
    hashes: dict[tuple[str, str, tuple[int, ...], str], list[str]] = defaultdict(list)
    gpu_copies: dict[tuple[str, str, tuple[int, ...], str], torch.Tensor] = {}
    run_deltas: dict[tuple[str, str, tuple[int, ...], str], list[float]] = defaultdict(
        list
    )

    for target in targets:
        m_dim, k_dim, n_dim = target.shape
        if target.shape not in factor_cache:
            left_np, right_np = gemm_numpy_pair(
                m_dim, k_dim, n_dim, case_index=target.case_index
            )
            factor_cache[target.shape] = (left_np, right_np)
            ref_cache[target.shape] = matmul_fp64(left_np, right_np)
            checksum_cache[target.shape] = gemm_checksum_fp64(left_np, right_np)
        left_np, right_np = factor_cache[target.shape]
        reference = ref_cache[target.shape]
        checksum = checksum_cache[target.shape]
        left_gpu = torch.from_numpy(np.ascontiguousarray(left_np, dtype=np.float32)).to(
            device=f"cuda:{device_index}", dtype=target.torch_dtype
        )
        right_gpu = torch.from_numpy(
            np.ascontiguousarray(right_np, dtype=np.float32)
        ).to(device=f"cuda:{device_index}", dtype=target.torch_dtype)
        with gemm_flags(
            fp16_reduced=target.fp16_reduced, bf16_reduced=target.bf16_reduced
        ):
            for blas_name in blas_names:
                key = (target.workload, target.dtype_name, target.shape, blas_name)
                with blas_library(blas_name):
                    for repeat in range(repeats):
                        before = read_telemetry(device_index)
                        product = _run_one_gemm(left_gpu, right_gpu)
                        after = read_telemetry(device_index)
                        gpu_fp64 = gpu_to_fp64(product)
                        product_sum = sum_elements_fp64(gpu_fp64)
                        abs_abft, norm_abft = normalized_checksum_residual(
                            product_sum, checksum
                        )
                        digest = _result_sha256(product)
                        hashes[key].append(digest)
                        if key in gpu_copies:
                            delta = torch.max(
                                torch.abs(
                                    product.to(torch.float32)
                                    - gpu_copies[key].to(torch.float32)
                                )
                            )
                            run_deltas[key].append(float(delta.item()))
                        else:
                            gpu_copies[key] = product.detach().clone()
                            run_deltas[key].append(0.0)
                        samples.append(
                            {
                                "workload": target.workload,
                                "dtype": target.dtype_name,
                                "shape": list(target.shape),
                                "repeat": repeat,
                                "blas_library": blas_name,
                                "max_abs_error": encode_f64(
                                    float(max_abs_error(gpu_fp64, reference))
                                ),
                                "max_rel_error": encode_f64(
                                    float(max_rel_error(gpu_fp64, reference))
                                ),
                                "abft_residual_abs": encode_f64(float(abs_abft)),
                                "abft_residual_normalized": encode_f64(
                                    float(norm_abft)
                                ),
                                "result_sha256": digest,
                                "telemetry_before": telemetry_dict(before),
                                "telemetry_after": telemetry_dict(after),
                                "kernel": (f"preferred_blas_library={blas_name}"),
                            }
                        )
                        del product, gpu_fp64
        del left_gpu, right_gpu
        torch.cuda.empty_cache()

    aggregates = _aggregates(
        samples, hashes, run_deltas, gpu_model, methodology.min_samples
    )
    finished = datetime.now(UTC)
    payload: dict[str, Any] = {
        "spec_id": methodology.spec_id,
        "tool_version": tool_version(),
        "torch_version": identity.torch_version,
        "cuda_version": identity.cuda_version,
        "driver_version": identity.driver_version,
        "gpu_model": gpu_model,
        "gpu_uuid": identity.uuid,
        "device_index": device_index,
        "repeats": repeats,
        "target_quantile": str(methodology.target_quantile),
        "min_samples_for_quantile": methodology.min_samples,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "blas_libraries": list(blas_names),
        "not_covered": [
            "cuBLAS heuristic algorithm IDs beyond preferred_blas_library",
            "pooling residuals across GPU models",
            "W04-W07 ABFT checksums (GEMM checksum only)",
            "shapes with max dim > 4096 unless --include-large",
        ],
        "samples": samples,
        "aggregates": aggregates,
    }
    cuda = identity.cuda_version or "unknown"
    driver = identity.driver_version or "unknown"
    folder = f"cuda-{cuda}_driver-{driver}_tool-{tool_version()}"
    out_dir = noisefloor_dir / gpu_model / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    out_path = out_dir / f"run-{stamp}-{digest}.json"
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _update_index(noisefloor_dir, gpu_model, out_path)
    return out_path


def _aggregates(
    samples: list[dict[str, Any]],
    hashes: dict[tuple[str, str, tuple[int, ...], str], list[str]],
    run_deltas: dict[tuple[str, str, tuple[int, ...], str], list[float]],
    gpu_model: str,
    min_samples: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, tuple[int, ...], str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for sample in samples:
        key = (
            str(sample["workload"]),
            str(sample["dtype"]),
            tuple(int(x) for x in sample["shape"]),
            str(sample["blas_library"]),
        )
        grouped[key].append(sample)
    rows: list[dict[str, Any]] = []
    for key, group in grouped.items():
        workload, dtype_name, shape, blas_name = key
        n_count = len(group)
        bitwise = len(set(hashes[key])) == 1
        delta_max = max(run_deltas[key]) if run_deltas[key] else 0.0
        abs_vals = [decode_f64(s["max_abs_error"]) for s in group]
        rel_vals = [decode_f64(s["max_rel_error"]) for s in group]
        abft_vals = [decode_f64(s["abft_residual_normalized"]) for s in group]
        status = "uncharacterized"
        reason = f"n={n_count} < min_samples={min_samples}; INCONCLUSIVE"
        if n_count >= min_samples:
            status = "n-sufficient"
            reason = f"n={n_count} meets min_samples; quantile is computed by lookup"
        rows.append(
            {
                "gpu_model": gpu_model,
                "workload": workload,
                "dtype": dtype_name,
                "shape": list(shape),
                "blas_library": blas_name,
                "n": n_count,
                "status": status,
                "reason": reason,
                "bitwise_stable": bitwise,
                "run_to_run_max_abs_delta": encode_f64(delta_max),
                "sample_max_abs_error": encode_f64(max(abs_vals)),
                "sample_max_rel_error": encode_f64(max(rel_vals)),
                "sample_max_abft_normalized": encode_f64(max(abft_vals)),
            }
        )
    return rows


def _update_index(noisefloor_dir: Path, gpu_model: str, run_path: Path) -> None:
    index_path = noisefloor_dir / "index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {
            "spec_id": "noisefloor-v1",
            "measured_gpu_models": [],
            "runs": [],
        }
    models = list(index.get("measured_gpu_models", []))
    if gpu_model not in models:
        models.append(gpu_model)
    runs = list(index.get("runs", []))
    rel_path = str(run_path)
    try:
        rel_path = str(run_path.resolve().relative_to(noisefloor_dir.resolve()))
    except ValueError:
        rel_path = str(run_path)
    runs.append({"gpu_model": gpu_model, "path": rel_path})
    index["measured_gpu_models"] = models
    index["runs"] = runs
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
