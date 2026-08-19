"""Residual-v3 FPR tail-data sweep. JSONL output, resumable, two passes."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from assay.abft.residual_v3 import residual_v3_rows
from assay.noise.blas import available_blas_libraries, blas_library
from assay.noise.floats import decode_f64, encode_f64
from assay.noise.pilot import PILOT_DTYPE, PILOT_WORKLOAD
from assay.noise.run import GemmTarget, _WORKLOAD_IDS, _factors_to_gpu, _run_one_gemm, tool_version
from assay.probe.identity import model_key, read_identity
from assay.workload.context import gemm_flags, require_cuda
from assay.workload.gemm import gemm_numpy_pair

K_DIM = 4096
M_DIM = 4096
N_DIM = 4096
CASE_INDEX = 3
PASS1_N = 500
DEFAULT_N = 20_000
TOP_TAIL_K = 64
HIST_BIN_COUNT = 200
HIST_LO = 1e-9
HIST_HI = 1e-3
HIST_BIN_EDGES: np.ndarray = np.logspace(
    np.log10(HIST_LO),
    np.log10(HIST_HI),
    HIST_BIN_COUNT + 1,
    dtype=np.float64,
)


def sample_resume_key(row: dict[str, Any]) -> tuple[int, int]:
    """(pass, sample_index)."""
    return (int(row["pass"]), int(row["sample_index"]))


def histogram_bin_edges_list() -> list[float]:
    """Fixed log-spaced bin edges for JSON metadata (reproducible)."""
    return [float(x) for x in HIST_BIN_EDGES]


def pass1_memmap_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".pass1.f64")


def require_frozen_pot_thresholds(
    *,
    pass_num: int,
    frozen: dict[str, float] | None,
) -> dict[str, float]:
    """Raise if pass 2 starts without frozen POT thresholds from pass 1."""
    if pass_num != 2:
        msg = f"require_frozen_pot_thresholds called for pass {pass_num}, expected 2"
        raise ValueError(msg)
    if frozen is None or not frozen:
        msg = (
            "pass 2 requires frozen POT thresholds from pass 1; "
            "run pass 1 (500 samples) to completion first"
        )
        raise RuntimeError(msg)
    for key in ("p95", "p99", "p999"):
        if key not in frozen:
            msg = f"frozen POT thresholds missing key {key!r}"
            raise RuntimeError(msg)
    return frozen


def summarize_r_rows(r_np: np.ndarray) -> dict[str, Any]:
    """Scalar stats, top-64 tail, and fixed-bin histogram for one sample."""
    r_max = float(np.max(r_np))
    r_median = float(np.median(r_np))
    r_p99 = float(np.percentile(r_np, 99))
    r_p999 = float(np.percentile(r_np, 99.9))
    top64 = np.sort(r_np)[-TOP_TAIL_K:][::-1]
    hist_counts, _ = np.histogram(r_np, bins=HIST_BIN_EDGES)
    if hist_counts.shape[0] != HIST_BIN_COUNT:
        msg = f"expected {HIST_BIN_COUNT} histogram bins, got {hist_counts.shape[0]}"
        raise RuntimeError(msg)
    return {
        "r_max": r_max,
        "r_median": r_median,
        "r_p99": r_p99,
        "r_p999": r_p999,
        "r_top64": top64,
        "hist_counts": hist_counts.astype(np.int64),
    }


def count_above_thresholds(
    r_np: np.ndarray,
    *,
    p95: float,
    p99: float,
    p999: float,
) -> dict[str, int]:
    return {
        "n_above_p95": int(np.sum(r_np > p95)),
        "n_above_p99": int(np.sum(r_np > p99)),
        "n_above_p999": int(np.sum(r_np > p999)),
    }


def compute_frozen_pot_thresholds(memmap: np.ndarray, n_pass1: int) -> dict[str, float]:
    """Global POT thresholds from all row observations in pass 1."""
    if n_pass1 < PASS1_N:
        msg = f"pass 1 incomplete: {n_pass1}/{PASS1_N} samples"
        raise RuntimeError(msg)
    flat = memmap[:PASS1_N].reshape(-1)
    return {
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "p999": float(np.percentile(flat, 99.9)),
    }


def load_fpr_state(path: Path) -> tuple[
    dict[str, Any] | None,
    set[tuple[int, int]],
    dict[str, float] | None,
    bool,
    int,
]:
    """Return metadata, done keys, frozen thresholds, pass1 complete, pass1 count."""
    metadata: dict[str, Any] | None = None
    done: set[tuple[int, int]] = set()
    frozen: dict[str, float] | None = None
    pass1_complete = False
    pass1_count = 0
    if not path.is_file():
        return metadata, done, frozen, pass1_complete, pass1_count
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
                if "pot_threshold_p95" in obj:
                    frozen = {
                        "p95": decode_f64(obj["pot_threshold_p95"]),
                        "p99": decode_f64(obj["pot_threshold_p99"]),
                        "p999": decode_f64(obj["pot_threshold_p999"]),
                    }
                    pass1_complete = True
                continue
            if record_type == "pot_thresholds":
                frozen = {
                    "p95": decode_f64(obj["p95"]),
                    "p99": decode_f64(obj["p99"]),
                    "p999": decode_f64(obj["p999"]),
                }
                pass1_complete = True
                continue
            if obj.get("phase") == "fpr_clean":
                key = sample_resume_key(obj)
                done.add(key)
                if int(obj["pass"]) == 1:
                    pass1_count += 1
    return metadata, done, frozen, pass1_complete, pass1_count


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
    *,
    n_samples: int,
    frozen: dict[str, float] | None,
) -> dict[str, Any]:
    driver = "unknown"
    try:
        driver = torch.cuda.get_driver_version()  # type: ignore[no-untyped-call]
    except Exception:
        pass
    meta: dict[str, Any] = {
        "record_type": "metadata",
        "residual_version": "v3",
        "study": "fpr-tail",
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
        "k": K_DIM,
        "n": N_DIM,
        "pass1_n": PASS1_N,
        "total_n": n_samples,
        "pass2_n": n_samples - PASS1_N,
        "top_tail_k": TOP_TAIL_K,
        "hist_bin_count": HIST_BIN_COUNT,
        "hist_lo": HIST_LO,
        "hist_hi": HIST_HI,
        "hist_bin_edges": histogram_bin_edges_list(),
    }
    if frozen is not None:
        meta["pot_threshold_p95"] = encode_f64(frozen["p95"])
        meta["pot_threshold_p99"] = encode_f64(frozen["p99"])
        meta["pot_threshold_p999"] = encode_f64(frozen["p999"])
    return meta


def _write_metadata_snapshot(
    path: Path,
    device_index: int,
    *,
    n_samples: int,
    frozen: dict[str, float] | None,
) -> None:
    payload = _build_metadata(device_index, n_samples=n_samples, frozen=frozen)
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


def _sample_row(
    *,
    pass_num: int,
    sample_index: int,
    summary: dict[str, Any],
    n_scale_zero: int,
    gpu_model: str,
    blas_name: str,
    frozen: dict[str, float] | None,
    r_np: np.ndarray | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "phase": "fpr_clean",
        "pass": pass_num,
        "sample_index": sample_index,
        "residual_version": "residual-v3",
        "workload": PILOT_WORKLOAD,
        "dtype": PILOT_DTYPE,
        "shape": [M_DIM, K_DIM, N_DIM],
        "gpu_model": gpu_model,
        "blas_library": blas_name,
        "tool_version": tool_version(),
        "r_max": encode_f64(summary["r_max"]),
        "r_median": encode_f64(summary["r_median"]),
        "r_p99": encode_f64(summary["r_p99"]),
        "r_p999": encode_f64(summary["r_p999"]),
        "r_top64": [encode_f64(float(v)) for v in summary["r_top64"]],
        "hist_counts": [int(x) for x in summary["hist_counts"]],
        "n_scale_zero": n_scale_zero,
    }
    if frozen is not None and r_np is not None:
        row.update(count_above_thresholds(r_np, **frozen))
    return row


def _backfill_pass1_pot_counts(
    path: Path,
    memmap: np.memmap,
    frozen: dict[str, float],
) -> None:
    """Add frozen-threshold exceedance counts to pass-1 rows missing them."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    updated: list[str] = []
    changed = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            updated.append(line)
            continue
        obj = json.loads(stripped)
        if obj.get("phase") != "fpr_clean" or int(obj.get("pass", 0)) != 1:
            updated.append(line)
            continue
        if "n_above_p95" in obj:
            updated.append(line)
            continue
        idx = int(obj["sample_index"])
        r_np = np.array(memmap[idx], dtype=np.float64)
        obj.update(count_above_thresholds(r_np, **frozen))
        updated.append(json.dumps(obj, sort_keys=True) + "\n")
        changed = True
    if changed:
        path.write_text("".join(updated), encoding="utf-8")


def _finalize_pass1(
    output_path: Path,
    device_index: int,
    n_samples: int,
    memmap: np.memmap,
) -> dict[str, float]:
    frozen = compute_frozen_pot_thresholds(memmap, PASS1_N)
    record = {
        "record_type": "pot_thresholds",
        "pass1_n": PASS1_N,
        "p95": encode_f64(frozen["p95"]),
        "p99": encode_f64(frozen["p99"]),
        "p999": encode_f64(frozen["p999"]),
    }
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
        fh.flush()
    _backfill_pass1_pot_counts(output_path, memmap, frozen)
    _write_metadata_snapshot(
        output_path,
        device_index,
        n_samples=n_samples,
        frozen=frozen,
    )
    print(
        f"pass 1 complete: POT thresholds frozen "
        f"p95={frozen['p95']:.6e} p99={frozen['p99']:.6e} "
        f"p999={frozen['p999']:.6e}",
        flush=True,
    )
    return frozen


def estimate_runtime(n_samples: int) -> dict[str, float]:
    """Rough workload estimate before a sweep run."""
    total_gemms = float(n_samples)
    row_observations = float(n_samples * M_DIM)
    return {
        "total_gemm_launches": total_gemms,
        "total_row_observations": row_observations,
        "pass1_gemms": float(min(PASS1_N, n_samples)),
        "pass2_gemms": float(max(0, n_samples - PASS1_N)),
    }


def _gemm_target() -> GemmTarget:
    return GemmTarget(
        workload=PILOT_WORKLOAD,
        dtype_name=PILOT_DTYPE,
        torch_dtype=torch.bfloat16,
        fp16_reduced=False,
        bf16_reduced=False,
        shape=(M_DIM, K_DIM, N_DIM),
        case_index=CASE_INDEX,
    )


def _run_one_sample(
    *,
    pass_num: int,
    sample_index: int,
    global_sample_index: int,
    target: GemmTarget,
    workload_id: int,
    device_index: int,
    gpu_model: str,
    blas_name: str,
    output_path: Path,
    frozen: dict[str, float] | None,
    memmap: np.memmap | None,
) -> None:
    t0 = time.perf_counter()
    left_np, right_np = gemm_numpy_pair(
        M_DIM,
        K_DIM,
        N_DIM,
        case_index=CASE_INDEX,
        sample_index=global_sample_index,
        workload_id=workload_id,
    )
    left_gpu, right_gpu = _factors_to_gpu(left_np, right_np, target, device_index)
    product = _run_one_gemm(left_gpu, right_gpu)
    r_np, n_scale_zero = residual_v3_rows(
        left_gpu, right_gpu, product, return_n_scale_zero=True
    )
    summary = summarize_r_rows(r_np)
    if pass_num == 1 and memmap is not None:
        memmap[sample_index] = r_np
        memmap.flush()
    row = _sample_row(
        pass_num=pass_num,
        sample_index=sample_index,
        summary=summary,
        n_scale_zero=n_scale_zero,
        gpu_model=gpu_model,
        blas_name=blas_name,
        frozen=frozen if pass_num == 2 else None,
        r_np=r_np if pass_num == 2 else None,
    )
    row["seconds"] = round(time.perf_counter() - t0, 3)
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()
    del left_gpu, right_gpu, product


def run_v3_fpr_sweep(
    *,
    output_path: Path,
    device_index: int = 0,
    n_samples: int = DEFAULT_N,
) -> Path:
    """FPR tail-data collection: full 4096-row residuals, no fitting."""
    if n_samples < PASS1_N:
        msg = f"n_samples must be >= pass1_n ({PASS1_N}), got {n_samples}"
        raise ValueError(msg)
    require_cuda()
    torch.cuda.set_device(device_index)
    identity = read_identity(device_index)
    gpu_model = model_key(identity)
    workload_id = _WORKLOAD_IDS[PILOT_WORKLOAD]
    pass2_n = n_samples - PASS1_N

    est = estimate_runtime(n_samples)
    print(
        f"fpr sweep: {int(est['total_gemm_launches'])} GEMM launches, "
        f"~{est['total_row_observations']:.2e} row observations "
        f"(pass 1: {int(est['pass1_gemms'])}, pass 2: {int(est['pass2_gemms'])})",
        flush=True,
    )

    blas_names = available_blas_libraries()
    blas_name = blas_names[0] if blas_names else "unknown"
    target = _gemm_target()

    metadata, done, frozen, pass1_complete, _pass1_count = load_fpr_state(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if metadata is None:
        _write_metadata_snapshot(
            output_path, device_index, n_samples=n_samples, frozen=None
        )

    mm_path = pass1_memmap_path(output_path)
    memmap: np.memmap | None = None
    if not pass1_complete:
        memmap = np.memmap(
            mm_path,
            dtype=np.float64,
            mode="r+" if mm_path.is_file() else "w+",
            shape=(PASS1_N, M_DIM),
        )

    with gemm_flags(fp16_reduced=False, bf16_reduced=False), blas_library(blas_name):
        # Pass 1: establish frozen POT thresholds
        if not pass1_complete:
            for sample_index in range(PASS1_N):
                key = (1, sample_index)
                if key in done:
                    continue
                _run_one_sample(
                    pass_num=1,
                    sample_index=sample_index,
                    global_sample_index=sample_index,
                    target=target,
                    workload_id=workload_id,
                    device_index=device_index,
                    gpu_model=gpu_model,
                    blas_name=blas_name,
                    output_path=output_path,
                    frozen=None,
                    memmap=memmap,
                )
                done.add(key)
                if sample_index % 50 == 0:
                    print(f"pass 1 sample {sample_index}/{PASS1_N}", flush=True)
            assert memmap is not None
            frozen = _finalize_pass1(output_path, device_index, n_samples, memmap)
            pass1_complete = True
            memmap = None

        frozen = require_frozen_pot_thresholds(pass_num=2, frozen=frozen)

        # Pass 2: remaining samples with frozen thresholds
        for sample_index in range(pass2_n):
            key = (2, sample_index)
            if key in done:
                continue
            _run_one_sample(
                pass_num=2,
                sample_index=sample_index,
                global_sample_index=PASS1_N + sample_index,
                target=target,
                workload_id=workload_id,
                device_index=device_index,
                gpu_model=gpu_model,
                blas_name=blas_name,
                output_path=output_path,
                frozen=frozen,
                memmap=None,
            )
            done.add(key)
            if sample_index % 500 == 0:
                print(f"pass 2 sample {sample_index}/{pass2_n}", flush=True)

    print(f"done -> {output_path}", flush=True)
    return output_path
