"""Residual-v3 flip matrix sweep. JSONL output, resumable, one line per sample."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from assay.abft.residual_v3 import residual_v3
from assay.inject import BitClass, InjectionVerify, flip_random
from assay.noise.blas import available_blas_libraries, blas_library
from assay.noise.floats import decode_f64, encode_f64
from assay.noise.pilot import PILOT_DTYPE, PILOT_SHAPE, PILOT_WORKLOAD
from assay.noise.run import (
    _WORKLOAD_IDS,
    _factors_to_gpu,
    _run_one_gemm,
    expand_targets,
    tool_version,
)
from assay.probe.identity import model_key, read_identity
from assay.reference.spec import BASE_SEED, WORKLOAD_GEMM_SHAPES
from assay.workload.context import gemm_flags, require_cuda
from assay.workload.gemm import gemm_numpy_pair

FLIP_N = 200
FLIP_COUNTS: tuple[int, ...] = (1, 2, 4)
_SAMPLE_STRIDE = 1_000_037

# Observed clean max over n=2000 W02 bf16 4096³ residual-v3 samples.
CLEAN_MAX_V3 = 3.245921e-06
CLEAN_MAX_V3_SOURCE = "data/noisefloor/pilot/sweep-v3.jsonl"
CLEAN_MAX_V3_N = 2000


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


def _cell_seed(bit_class: BitClass, n_flips: int, sample_index: int, workload_id: int) -> int:
    class_id = list(BitClass).index(bit_class) + 1
    return (
        BASE_SEED
        + workload_id * 1_000_003
        + class_id * 1_009
        + n_flips
        + sample_index * _SAMPLE_STRIDE
    )


def _build_metadata(device_index: int) -> dict[str, Any]:
    driver = "unknown"
    try:
        driver = torch.cuda.get_driver_version()  # type: ignore[no-untyped-call]
    except Exception:
        pass
    return {
        "record_type": "metadata",
        "residual_version": "v3",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda or "unknown",
        "gpu_name": torch.cuda.get_device_name(device_index),
        "driver_version": str(driver),
        "numpy_version": np.__version__,
        "python_version": sys.version,
        "git_sha": _git_sha(),
        "clean_max_v3": encode_f64(CLEAN_MAX_V3),
        "clean_max_v3_source": CLEAN_MAX_V3_SOURCE,
        "clean_max_v3_n": CLEAN_MAX_V3_N,
        "flip_n_per_cell": FLIP_N,
        "flip_counts": list(FLIP_COUNTS),
    }


def _has_metadata_header(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8") as fh:
        first = fh.readline().strip()
        if not first:
            return False
        try:
            obj = json.loads(first)
            return obj.get("record_type") == "metadata"
        except json.JSONDecodeError:
            return False


def _completed_triples(path: Path) -> set[tuple[str, int, int]]:
    """Return (bit_class, n_flips, sample_index) already written."""
    if not path.is_file():
        return set()
    done: set[tuple[str, int, int]] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("record_type") == "metadata":
                continue
            bit_class = obj.get("bit_class")
            n_flips = obj.get("n_flips")
            sample_index = obj.get("sample_index")
            if bit_class is None or n_flips is None or sample_index is None:
                continue
            done.add((str(bit_class), int(n_flips), int(sample_index)))
    return done


@dataclass(frozen=True, slots=True)
class FlipCellSummary:
    bit_class: str
    n_flips: int
    n: int
    min_ratio: float
    median_ratio: float
    max_ratio: float
    n_exceeding_one: int


def _median(ordered: list[float]) -> float:
    count = len(ordered)
    if count < 1:
        msg = "median requires at least one sample"
        raise ValueError(msg)
    if count % 2 == 1:
        return ordered[count // 2]
    return 0.5 * (ordered[count // 2 - 1] + ordered[count // 2])


def summarize_flip_ratios(
    ratios_by_cell: dict[tuple[str, int], list[float]],
) -> list[FlipCellSummary]:
    """min / median / max of r_max/clean_max and count with ratio > 1."""
    rows: list[FlipCellSummary] = []
    for bit_class in BitClass:
        for n_flips in FLIP_COUNTS:
            key = (bit_class.value, n_flips)
            ratios = ratios_by_cell.get(key, [])
            if not ratios:
                continue
            ordered = sorted(ratios)
            n_over = sum(
                1
                for value in ratios
                if (not math.isfinite(value)) or value > 1.0
            )
            rows.append(
                FlipCellSummary(
                    bit_class=bit_class.value,
                    n_flips=n_flips,
                    n=len(ratios),
                    min_ratio=ordered[0],
                    median_ratio=_median(ordered),
                    max_ratio=ordered[-1],
                    n_exceeding_one=n_over,
                )
            )
    return rows


def _verify_fields(stats: InjectionVerify) -> dict[str, Any]:
    """JSONL fields for --verify-injection. Additive only."""
    return {
        "n_elements_flipped": stats.n_elements_flipped,
        "n_elements_bitwise_equal": stats.n_elements_bitwise_equal,
        "achieved_rel_delta_max": encode_f64(stats.achieved_rel_delta_max),
        "achieved_rel_delta_median": encode_f64(stats.achieved_rel_delta_median),
    }


def run_v3_flip_sweep(
    *,
    output_path: Path,
    device_index: int = 0,
    n_samples: int = FLIP_N,
    verify_injection: bool = False,
) -> Path:
    """W02 bf16 4096³ flip matrix with residual-v3. One GEMM per sample_index."""
    if n_samples < 1:
        msg = "n_samples must be >= 1"
        raise ValueError(msg)
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
        msg = f"flip sweep expects one target, got {len(targets)}"
        raise ValueError(msg)
    target = targets[0]
    if PILOT_SHAPE not in WORKLOAD_GEMM_SHAPES:
        msg = "flip sweep shape is not a suite GEMM shape"
        raise ValueError(msg)

    blas_names = available_blas_libraries()
    blas_name = blas_names[0] if blas_names else "unknown"
    workload_id = _WORKLOAD_IDS[target.workload]

    done = _completed_triples(output_path)
    n_cells = len(list(BitClass)) * len(FLIP_COUNTS)
    n_expected = n_cells * n_samples
    if len(done) >= n_expected:
        print(f"already complete: {len(done)}/{n_expected} samples in {output_path}")
        return output_path
    if done:
        print(f"resuming: {len(done)}/{n_expected} samples already in {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not _has_metadata_header(output_path):
        with output_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(_build_metadata(device_index), sort_keys=True) + "\n")
            fh.flush()

    with (
        gemm_flags(fp16_reduced=target.fp16_reduced, bf16_reduced=target.bf16_reduced),
        blas_library(blas_name),
        output_path.open("a", encoding="utf-8") as fh,
    ):
        for sample_index in range(n_samples):
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

            for bit_class in BitClass:
                for n_flips in FLIP_COUNTS:
                    triple = (bit_class.value, n_flips, sample_index)
                    if triple in done:
                        continue
                    seed = _cell_seed(bit_class, n_flips, sample_index, workload_id)
                    if verify_injection:
                        flipped, _locs, inj_stats = flip_random(
                            product,
                            n_flips,
                            bit_class,
                            seed,
                            verify=True,
                        )
                    else:
                        flipped, _locs = flip_random(
                            product,
                            n_flips,
                            bit_class,
                            seed,
                        )
                        inj_stats = None
                    v3 = residual_v3(left_gpu, right_gpu, flipped)
                    elapsed = time.perf_counter() - t0
                    r_max = v3["r_max"]
                    ratio = (
                        math.inf
                        if not math.isfinite(r_max)
                        else r_max / CLEAN_MAX_V3
                    )

                    row: dict[str, Any] = {
                        "bit_class": bit_class.value,
                        "n_flips": n_flips,
                        "sample_index": sample_index,
                        "residual_version": "residual-v3",
                        "workload": PILOT_WORKLOAD,
                        "dtype": PILOT_DTYPE,
                        "shape": list(PILOT_SHAPE),
                        "gpu_model": gpu_model,
                        "blas_library": blas_name,
                        "tool_version": tool_version(),
                        "r_max": encode_f64(r_max),
                        "ratio_to_clean_max": encode_f64(ratio),
                        "detected": r_max > CLEAN_MAX_V3,
                        "seconds": round(elapsed, 3),
                    }
                    if inj_stats is not None:
                        row.update(_verify_fields(inj_stats))
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                    fh.flush()
                    done.add(triple)

            del left_gpu, right_gpu, product
            if sample_index % 20 == 0:
                print(
                    f"sample {sample_index}/{n_samples}  "
                    f"completed={len(done)}/{n_expected}",
                    flush=True,
                )

    print(f"done: {len(done)}/{n_expected} samples -> {output_path}")
    return output_path


def load_flip_samples(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    metadata = None
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("record_type") == "metadata":
                metadata = obj
            else:
                samples.append(obj)
    return metadata, samples


def summarize_from_jsonl(path: Path) -> list[FlipCellSummary]:
    metadata, samples = load_flip_samples(path)
    clean_max = CLEAN_MAX_V3
    if metadata is not None and "clean_max_v3" in metadata:
        clean_max = decode_f64(metadata["clean_max_v3"])
    ratios: dict[tuple[str, int], list[float]] = {}
    for row in samples:
        key = (str(row["bit_class"]), int(row["n_flips"]))
        if "ratio_to_clean_max" in row:
            ratio = decode_f64(row["ratio_to_clean_max"])
        else:
            r_max = decode_f64(row["r_max"])
            ratio = math.inf if not math.isfinite(r_max) else r_max / clean_max
        ratios.setdefault(key, []).append(float(ratio))
    return summarize_flip_ratios(ratios)
