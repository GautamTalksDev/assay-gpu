"""Residual-v3 K-scaling sweep. JSONL output, resumable, two phases per K."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from assay.abft.residual_v3 import residual_v3
from assay.inject import BitClass, flip_random
from assay.noise.blas import available_blas_libraries, blas_library
from assay.noise.floats import decode_f64, encode_f64
from assay.noise.pilot import PILOT_DTYPE, PILOT_WORKLOAD
from assay.noise.run import GemmTarget, _WORKLOAD_IDS, _factors_to_gpu, _run_one_gemm, tool_version
from assay.noise.sweep_v3_flips import FLIP_COUNTS, _cell_seed
from assay.probe.identity import model_key, read_identity
from assay.workload.context import gemm_flags, require_cuda
from assay.workload.gemm import gemm_numpy_pair

K_VALUES_DEFAULT: tuple[int, ...] = (512, 1024, 2048, 4096, 8192)
M_DIM = 4096
N_DIM = 4096
CASE_INDEX = 3
CLEAN_N = 500
FLIP_N = 100
REF_K = 4096


def parse_k_values(text: str) -> tuple[int, ...]:
    """Parse comma-separated K inner dimensions."""
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        msg = "k-values must list at least one K"
        raise ValueError(msg)
    values = tuple(int(part) for part in parts)
    for k in values:
        if k < 1:
            msg = f"K must be >= 1, got {k}"
            raise ValueError(msg)
    return values


def sample_resume_key(row: dict[str, Any]) -> tuple[str, int, str, int, int]:
    """(phase, K, bit_class, n_flips, sample_index). Clean uses '', -1."""
    phase = str(row.get("phase", ""))
    k_val = int(row["K"])
    sample_index = int(row["sample_index"])
    if phase == "clean":
        return ("clean", k_val, "", -1, sample_index)
    if phase == "flip":
        return (
            "flip",
            k_val,
            str(row["bit_class"]),
            int(row["n_flips"]),
            sample_index,
        )
    msg = f"unknown phase in row: {phase!r}"
    raise ValueError(msg)


def relative_flops(k: int) -> float:
    """GEMM FLOPs relative to M=N=K=4096 reference."""
    return (M_DIM * k * N_DIM) / float(M_DIM * REF_K * N_DIM)


def estimate_workload(k_values: tuple[int, ...]) -> dict[str, float]:
    """Relative cost estimate before a sweep run."""
    gemms_per_k = CLEAN_N + FLIP_N
    total_gemms = len(k_values) * gemms_per_k
    total_relative = sum(relative_flops(k) for k in k_values) * gemms_per_k
    return {
        "k_count": float(len(k_values)),
        "gemms_per_k": float(gemms_per_k),
        "total_gemm_launches": float(total_gemms),
        "total_relative_flops_vs_4096_cube": total_relative,
    }


def require_clean_max_before_flip(
    *,
    k: int,
    clean_max_by_k: dict[int, float],
    clean_counts: dict[int, int],
    clean_n: int = CLEAN_N,
) -> float:
    """Raise if phase 2 cannot run at this K. Return clean_max[K]."""
    count = clean_counts.get(k, 0)
    if count < clean_n:
        msg = (
            f"phase 2 for K={k} requires clean phase 1 complete "
            f"({count}/{clean_n} samples)"
        )
        raise RuntimeError(msg)
    if k not in clean_max_by_k:
        msg = f"clean phase complete for K={k} but clean_max[{k}] is missing"
        raise RuntimeError(msg)
    return clean_max_by_k[k]


def load_kscale_state(path: Path) -> tuple[
    dict[str, Any] | None,
    dict[int, float],
    set[tuple[str, int, str, int, int]],
    dict[int, int],
    set[int],
]:
    """Return metadata, clean_max, done keys, clean counts, finalized clean_max K."""
    metadata: dict[str, Any] | None = None
    clean_max_by_k: dict[int, float] = {}
    done: set[tuple[str, int, str, int, int]] = set()
    clean_counts: dict[int, int] = {}
    clean_max_finalized: set[int] = set()
    if not path.is_file():
        return metadata, clean_max_by_k, done, clean_counts, clean_max_finalized
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_type = obj.get("record_type")
            if record_type == "metadata":
                metadata = obj
                continue
            if record_type == "clean_max":
                k_val = int(obj["K"])
                clean_max_by_k[k_val] = decode_f64(obj["clean_max"])
                clean_max_finalized.add(k_val)
                continue
            phase = str(obj.get("phase", ""))
            if phase == "clean":
                k_val = int(obj["K"])
                clean_counts[k_val] = clean_counts.get(k_val, 0) + 1
                done.add(sample_resume_key(obj))
                r_max = decode_f64(obj["r_max"])
                prev = clean_max_by_k.get(k_val)
                if prev is None or r_max > prev:
                    clean_max_by_k[k_val] = r_max
            elif phase == "flip":
                done.add(sample_resume_key(obj))
    return metadata, clean_max_by_k, done, clean_counts, clean_max_finalized


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _build_metadata(
    device_index: int,
    k_values: tuple[int, ...],
    clean_max_by_k: dict[int, float],
) -> dict[str, Any]:
    driver = "unknown"
    try:
        driver = torch.cuda.get_driver_version()  # type: ignore[no-untyped-call]
    except Exception:
        pass
    clean_table = {
        str(k): encode_f64(clean_max_by_k[k])
        for k in sorted(clean_max_by_k)
    }
    return {
        "record_type": "metadata",
        "residual_version": "v3",
        "study": "k-scaling",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda or "unknown",
        "gpu_name": torch.cuda.get_device_name(device_index),
        "driver_version": str(driver),
        "numpy_version": np.__version__,
        "python_version": sys.version,
        "git_sha": _git_sha(),
        "workload": PILOT_WORKLOAD,
        "dtype": PILOT_DTYPE,
        "m": M_DIM,
        "n": N_DIM,
        "k_values": list(k_values),
        "clean_n": CLEAN_N,
        "flip_n": FLIP_N,
        "flip_counts": list(FLIP_COUNTS),
        "clean_max_by_k": clean_table,
    }


def _gemm_target(k: int) -> GemmTarget:
    return GemmTarget(
        workload=PILOT_WORKLOAD,
        dtype_name=PILOT_DTYPE,
        torch_dtype=torch.bfloat16,
        fp16_reduced=False,
        bf16_reduced=False,
        shape=(M_DIM, k, N_DIM),
        case_index=CASE_INDEX,
    )


def _write_metadata_snapshot(
    path: Path,
    device_index: int,
    k_values: tuple[int, ...],
    clean_max_by_k: dict[int, float],
) -> None:
    """Rewrite metadata line (line 1) with updated clean_max table."""
    payload = _build_metadata(device_index, k_values, clean_max_by_k)
    line = json.dumps(payload, sort_keys=True) + "\n"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(line, encoding="utf-8")
        return
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if lines and json.loads(lines[0].strip()).get("record_type") == "metadata":
        lines[0] = line
    else:
        lines.insert(0, line)
    path.write_text("".join(lines), encoding="utf-8")


def run_v3_kscale_sweep(
    *,
    output_path: Path,
    device_index: int = 0,
    k_values: tuple[int, ...] = K_VALUES_DEFAULT,
) -> Path:
    """K-scaling study: clean floor then flip matrix at each K."""
    require_cuda()
    torch.cuda.set_device(device_index)
    identity = read_identity(device_index)
    gpu_model = model_key(identity)
    workload_id = _WORKLOAD_IDS[PILOT_WORKLOAD]

    est = estimate_workload(k_values)
    print(
        f"k-scaling sweep: {int(est['total_gemm_launches'])} GEMM launches, "
        f"relative FLOPs {est['total_relative_flops_vs_4096_cube']:.2f}x "
        f"vs 4096³ baseline "
        f"(K=8192 is ~{relative_flops(8192):.1f}x per launch vs K=4096)",
        flush=True,
    )

    blas_names = available_blas_libraries()
    blas_name = blas_names[0] if blas_names else "unknown"

    metadata, clean_max_by_k, done, clean_counts, clean_max_finalized = load_kscale_state(
        output_path
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if metadata is None:
        _write_metadata_snapshot(output_path, device_index, k_values, clean_max_by_k)

    n_flip_cells = len(list(BitClass)) * len(FLIP_COUNTS)

    with gemm_flags(fp16_reduced=False, bf16_reduced=False), blas_library(blas_name):
        for k in k_values:
            if _k_flip_complete(done, k):
                print(f"K={k} already complete, skipping", flush=True)
                continue

            target = _gemm_target(k)

            # Phase 1: clean
            for sample_index in range(CLEAN_N):
                key = ("clean", k, "", -1, sample_index)
                if key in done:
                    continue
                t0 = time.perf_counter()
                left_np, right_np = gemm_numpy_pair(
                    M_DIM,
                    k,
                    N_DIM,
                    case_index=CASE_INDEX,
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
                    "phase": "clean",
                    "K": k,
                    "sample_index": sample_index,
                    "residual_version": "residual-v3",
                    "workload": PILOT_WORKLOAD,
                    "dtype": PILOT_DTYPE,
                    "shape": [M_DIM, k, N_DIM],
                    "gpu_model": gpu_model,
                    "blas_library": blas_name,
                    "tool_version": tool_version(),
                    "r_max": encode_f64(v3["r_max"]),
                    "r_median": encode_f64(v3["r_median"]),
                    "r_p99": encode_f64(v3["r_p99"]),
                    "n_scale_zero": v3["n_scale_zero"],
                    "seconds": round(elapsed, 3),
                }
                with output_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                    fh.flush()
                done.add(key)
                clean_counts[k] = clean_counts.get(k, 0) + 1
                prev = clean_max_by_k.get(k)
                if prev is None or v3["r_max"] > prev:
                    clean_max_by_k[k] = v3["r_max"]
                del left_gpu, right_gpu, product
                if sample_index % 100 == 0:
                    print(
                        f"K={k} clean {sample_index}/{CLEAN_N}  "
                        f"r_max={v3['r_max']:.6e}",
                        flush=True,
                    )

            if (
                clean_counts.get(k, 0) >= CLEAN_N
                and k not in clean_max_finalized
            ):
                clean_max = clean_max_by_k[k]
                record = {
                    "record_type": "clean_max",
                    "K": k,
                    "clean_max": encode_f64(clean_max),
                    "n_clean": CLEAN_N,
                }
                with output_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, sort_keys=True) + "\n")
                    fh.flush()
                clean_max_finalized.add(k)
                _write_metadata_snapshot(
                    output_path, device_index, k_values, clean_max_by_k
                )
                print(f"K={k} clean_max={clean_max:.6e}", flush=True)

            threshold = require_clean_max_before_flip(
                k=k,
                clean_max_by_k=clean_max_by_k,
                clean_counts=clean_counts,
            )

            # Phase 2: flips (one GEMM per sample_index)
            for sample_index in range(FLIP_N):
                t0 = time.perf_counter()
                left_np, right_np = gemm_numpy_pair(
                    M_DIM,
                    k,
                    N_DIM,
                    case_index=CASE_INDEX,
                    sample_index=sample_index,
                    workload_id=workload_id,
                )
                left_gpu, right_gpu = _factors_to_gpu(
                    left_np, right_np, target, device_index
                )
                product = _run_one_gemm(left_gpu, right_gpu)

                for bit_class in BitClass:
                    for n_flips in FLIP_COUNTS:
                        key = ("flip", k, bit_class.value, n_flips, sample_index)
                        if key in done:
                            continue
                        flipped, _locs = flip_random(
                            product,
                            n_flips,
                            bit_class,
                            _cell_seed(bit_class, n_flips, sample_index, workload_id),
                        )
                        v3 = residual_v3(left_gpu, right_gpu, flipped)
                        r_max = v3["r_max"]
                        ratio = (
                            math.inf
                            if not math.isfinite(r_max)
                            else r_max / threshold
                        )
                        flip_row: dict[str, Any] = {
                            "phase": "flip",
                            "K": k,
                            "bit_class": bit_class.value,
                            "n_flips": n_flips,
                            "sample_index": sample_index,
                            "residual_version": "residual-v3",
                            "workload": PILOT_WORKLOAD,
                            "dtype": PILOT_DTYPE,
                            "shape": [M_DIM, k, N_DIM],
                            "gpu_model": gpu_model,
                            "blas_library": blas_name,
                            "tool_version": tool_version(),
                            "clean_max": encode_f64(threshold),
                            "r_max": encode_f64(r_max),
                            "ratio_to_clean_max": encode_f64(ratio),
                            "detected": r_max > threshold,
                            "seconds": round(time.perf_counter() - t0, 3),
                        }
                        with output_path.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(flip_row, sort_keys=True) + "\n")
                            fh.flush()
                        done.add(key)

                del left_gpu, right_gpu, product
                if sample_index % 25 == 0:
                    flip_done = sum(
                        1
                        for bc in BitClass
                        for nf in FLIP_COUNTS
                        if ("flip", k, bc.value, nf, sample_index) in done
                    )
                    print(
                        f"K={k} flip sample {sample_index}/{FLIP_N}  "
                        f"cells={flip_done}/{n_flip_cells}",
                        flush=True,
                    )

    print(f"done -> {output_path}", flush=True)
    return output_path


def _k_clean_complete(
    done: set[tuple[str, int, str, int, int]], k: int
) -> bool:
    return all(("clean", k, "", -1, i) in done for i in range(CLEAN_N))


def _k_flip_complete(
    done: set[tuple[str, int, str, int, int]], k: int
) -> bool:
    return all(
        ("flip", k, bc.value, nf, si) in done
        for bc in BitClass
        for nf in FLIP_COUNTS
        for si in range(FLIP_N)
    )
