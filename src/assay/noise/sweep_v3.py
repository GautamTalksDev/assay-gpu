"""Clean-only residual-v3 sweep. Writes JSONL, one line per sample, resumable."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from assay.abft.residual_v3 import residual_v3
from assay.noise.blas import available_blas_libraries, blas_library
from assay.noise.floats import encode_f64
from assay.noise.pilot import PILOT_DTYPE, PILOT_N, PILOT_SHAPE, PILOT_WORKLOAD
from assay.noise.run import (
    _CHECKSUM_BACKEND,
    _WORKLOAD_IDS,
    _factors_to_gpu,
    _run_one_gemm,
    expand_targets,
    tool_version,
)
from assay.probe.identity import model_key, read_identity
from assay.reference.spec import WORKLOAD_GEMM_SHAPES
from assay.workload.context import gemm_flags, require_cuda
from assay.workload.gemm import gemm_numpy_pair


def _last_completed_index(path: Path) -> int:
    """Return the highest sample_index in an existing JSONL file, or -1."""
    if not path.is_file():
        return -1
    last = -1
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                idx = obj.get("sample_index", -1)
                if idx > last:
                    last = idx
            except json.JSONDecodeError:
                continue
    return last


def run_v3_sweep(
    *,
    output_path: Path,
    device_index: int = 0,
    n_samples: int = PILOT_N,
) -> Path:
    """W02 bf16 4096³ clean sweep with residual-v3. No fault injection."""
    require_cuda()
    torch.cuda.set_device(device_index)
    identity = read_identity(device_index)
    gpu_model = model_key(identity)

    targets = expand_targets(
        include_large=False,
        workload=PILOT_WORKLOAD,
        shape_filter=PILOT_SHAPE,
    )
    if len(targets) != 1:
        msg = f"sweep expects one target, got {len(targets)}"
        raise ValueError(msg)
    target = targets[0]
    if PILOT_SHAPE not in WORKLOAD_GEMM_SHAPES:
        msg = "sweep shape is not a suite GEMM shape"
        raise ValueError(msg)

    blas_names = available_blas_libraries()
    blas_name = blas_names[0] if blas_names else "unknown"
    workload_id = _WORKLOAD_IDS[target.workload]

    resume_from = _last_completed_index(output_path)
    start_index = resume_from + 1
    if start_index > 0:
        print(f"resuming from sample_index={start_index} (found {resume_from} completed)")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        gemm_flags(fp16_reduced=target.fp16_reduced, bf16_reduced=target.bf16_reduced),
        blas_library(blas_name),
        output_path.open("a", encoding="utf-8") as fh,
    ):
        for sample_index in range(start_index, n_samples):
            t0 = time.perf_counter()
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
            product = _run_one_gemm(left_gpu, right_gpu)

            v3 = residual_v3(left_gpu, right_gpu, product)
            elapsed = time.perf_counter() - t0

            row: dict[str, Any] = {
                "sample_index": sample_index,
                "residual_version": "residual-v3",
                "workload": PILOT_WORKLOAD,
                "dtype": PILOT_DTYPE,
                "shape": list(PILOT_SHAPE),
                "gpu_model": gpu_model,
                "blas_library": blas_name,
                "tool_version": tool_version(),
                "r_max": encode_f64(v3["r_max"]),
                "r_median": encode_f64(v3["r_median"]),
                "r_p99": encode_f64(v3["r_p99"]),
                "r_rows": [float(x).hex() for x in v3["r_rows"]],
                "row_indices": v3["row_indices"],
                "n_scale_zero": v3["n_scale_zero"],
                "seconds": round(elapsed, 3),
            }

            fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()

            del left_gpu, right_gpu, product
            if sample_index % 100 == 0:
                print(f"sample {sample_index}/{n_samples}  r_max={v3['r_max']:.6e}  r_median={v3['r_median']:.6e}  {elapsed:.1f}s")

    print(f"done: {n_samples} samples -> {output_path}")
    return output_path
