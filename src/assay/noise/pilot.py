"""ABFT residual pilot. Not a characterization. Never writes run-*.json."""

from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch

from assay.noise.blas import available_blas_libraries, blas_library
from assay.noise.floats import decode_f64, encode_f64
from assay.noise.methodology import load_methodology
from assay.noise.quantiles import order_statistic
from assay.noise.run import (
    _CHECKSUM_BACKEND,
    _STABILITY_LAUNCHES,
    _WORKLOAD_IDS,
    _factors_to_gpu,
    _result_sha256,
    _run_one_gemm,
    _sample_row,
    expand_targets,
    tool_version,
)
from assay.probe.identity import model_key, read_identity
from assay.reference.spec import WORKLOAD_GEMM_SHAPES
from assay.workload.context import gemm_flags, require_cuda
from assay.workload.gemm import gemm_numpy_pair

# Operator-chosen pilot length from CP-3. Not a detection cutoff.
PILOT_N = 2000
PILOT_WORKLOAD = "W02"
PILOT_DTYPE = "bfloat16"
PILOT_SHAPE = (4096, 4096, 4096)
PILOT_P99_PREFIXES = (100, 250, 500, 1000, 2000)
_P90 = Fraction(9, 10)
_P99 = Fraction(99, 100)
_P999 = Fraction(999, 1000)


def _median(ordered: list[float]) -> float:
    n = len(ordered)
    if n < 1:
        msg = "median requires at least one sample"
        raise ValueError(msg)
    if n % 2 == 1:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return math.inf if numerator != 0.0 else math.nan
    return numerator / denominator


def summarize_residuals(
    residuals: list[float], prefixes: tuple[int, ...] = PILOT_P99_PREFIXES
) -> dict[str, Any]:
    """Percentiles of a measured residual list. Not a detection threshold."""
    if not residuals:
        msg = "summarize_residuals requires at least one sample"
        raise ValueError(msg)
    ordered = sorted(residuals)
    minimum = ordered[0]
    median = _median(ordered)
    p90 = order_statistic(residuals, _P90)
    p99 = order_statistic(residuals, _P99)
    p999 = order_statistic(residuals, _P999)
    maximum = ordered[-1]
    running: dict[str, dict[str, str]] = {}
    for count in prefixes:
        if count > len(residuals):
            continue
        prefix = residuals[:count]
        running[str(count)] = encode_f64(order_statistic(prefix, _P99))
    return {
        "n": len(residuals),
        "min": encode_f64(minimum),
        "median": encode_f64(median),
        "p90": encode_f64(p90),
        "p99": encode_f64(p99),
        "p99_9": encode_f64(p999),
        "max": encode_f64(maximum),
        "max_over_median": encode_f64(_ratio(maximum, median)),
        "p99_9_over_median": encode_f64(_ratio(p999, median)),
        "running_p99": running,
        "sorted_residuals_hex": [float(value).hex() for value in ordered],
    }


def run_abft_pilot(
    *,
    noisefloor_dir: Path,
    device_index: int,
    n_samples: int = PILOT_N,
) -> Path:
    """W02 bf16 4096 cubed independent samples. Writes data/noisefloor/pilot/."""
    if n_samples < 1:
        msg = "n_samples must be >= 1"
        raise ValueError(msg)
    require_cuda()
    torch.cuda.set_device(device_index)
    methodology = load_methodology(noisefloor_dir)
    identity = read_identity(device_index)
    gpu_model = model_key(identity)
    targets = expand_targets(
        include_large=False,
        workload=PILOT_WORKLOAD,
        shape_filter=PILOT_SHAPE,
    )
    if len(targets) != 1:
        msg = f"pilot expects one target, got {len(targets)}"
        raise ValueError(msg)
    target = targets[0]
    if PILOT_SHAPE not in WORKLOAD_GEMM_SHAPES:
        msg = "pilot shape is not a suite GEMM shape"
        raise ValueError(msg)
    blas_names = available_blas_libraries()
    samples: list[dict[str, Any]] = []
    seconds: list[float] = []
    stability_sha: list[str] = []
    started = datetime.now(UTC)
    wall0 = time.perf_counter()
    workload_id = _WORKLOAD_IDS[target.workload]
    with gemm_flags(fp16_reduced=target.fp16_reduced, bf16_reduced=target.bf16_reduced):
        for blas_name in blas_names:
            with blas_library(blas_name):
                for sample_index in range(n_samples):
                    left_np, right_np = gemm_numpy_pair(
                        PILOT_SHAPE[0],
                        PILOT_SHAPE[1],
                        PILOT_SHAPE[2],
                        case_index=target.case_index,
                        sample_index=sample_index,
                        workload_id=workload_id,
                    )
                    left_gpu, right_gpu = _factors_to_gpu(
                        left_np, right_np, target, device_index
                    )
                    launches = _STABILITY_LAUNCHES if sample_index == 0 else 1
                    t0 = time.perf_counter()
                    products = [
                        _run_one_gemm(left_gpu, right_gpu)
                        for _launch in range(launches)
                    ]
                    seconds.append((time.perf_counter() - t0) / launches)
                    if sample_index == 0:
                        stability_sha = [_result_sha256(item) for item in products]
                    samples.append(
                        _sample_row(
                            target=target,
                            sample_index=sample_index,
                            blas_name=blas_name,
                            left_gpu=left_gpu,
                            right_gpu=right_gpu,
                            product=products[0],
                            device_index=device_index,
                        )
                    )
                    del left_gpu, right_gpu, products
    wall = time.perf_counter() - wall0
    residuals = [decode_f64(row["abft_residual_normalized"]) for row in samples]
    finished = datetime.now(UTC)
    mean_s = sum(seconds) / len(seconds) if seconds else math.nan
    payload: dict[str, Any] = {
        "schema": "noisefloor-pilot-v1",
        "not_a_characterization": True,
        "do_not_use_for_lookup": True,
        "min_samples_unchanged": methodology.min_samples,
        "quantile_p99_999": "UNJUSTIFIED until this pilot is read",
        "n": len(samples),
        "workload": PILOT_WORKLOAD,
        "dtype": PILOT_DTYPE,
        "shape": list(PILOT_SHAPE),
        "backend": _CHECKSUM_BACKEND.value,
        "checksum_kind": "ones_sided_C_e_vs_A_Be",
        "gemm": "torch.matmul",
        "gpu_model": gpu_model,
        "gpu_uuid": identity.uuid,
        "driver_version": identity.driver_version,
        "cuda_version": identity.cuda_version,
        "torch_version": identity.torch_version,
        "tool_version": tool_version(),
        "device_index": device_index,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "wall_seconds": encode_f64(wall),
        "seconds_per_sample": encode_f64(mean_s),
        "bitwise_stable_sample0": len(set(stability_sha)) == 1,
        "stability_sha256": stability_sha,
        "distribution": summarize_residuals(residuals),
        "samples": samples,
    }
    out_dir = noisefloor_dir / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"pilot-{stamp}-{gpu_model}.json"
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out_path
