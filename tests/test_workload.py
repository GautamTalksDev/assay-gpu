"""GPU tests: each workload runs and returns the documented shapes."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from assay.reference.spec import (
    W07_DECODE_STEPS,
    W07_PROMPT_LEN,
    WORKLOAD_ELEMENTWISE_SHAPE,
    WORKLOAD_GEMM_SHAPES,
    WORKLOAD_REDUCE_LENGTHS,
    WORKLOAD_SDPA_SHAPES,
)
from assay.workload.elementwise import run_w06
from assay.workload.gemm import run_w01, run_w02, run_w03
from assay.workload.reductions import run_w05
from assay.workload.sdpa import run_w04
from assay.workload.suite import double_run, write_double_run_report
from assay.workload.transformer import run_w07

pytestmark = pytest.mark.gpu

RECORD_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "workload" / "double_run.json"
)


def test_w01_gemm_fp32_shapes() -> None:
    results = run_w01()
    assert len(results) == len(WORKLOAD_GEMM_SHAPES)
    for result, (m_dim, _k, n_dim) in zip(results, WORKLOAD_GEMM_SHAPES, strict=True):
        assert result.shape == (m_dim, n_dim)
        assert result.result.dtype == torch.float32


def test_w02_gemm_bf16_shapes() -> None:
    results = run_w02()
    assert len(results) == len(WORKLOAD_GEMM_SHAPES)
    for result, (m_dim, _k, n_dim) in zip(results, WORKLOAD_GEMM_SHAPES, strict=True):
        assert result.shape == (m_dim, n_dim)
        assert result.result.dtype == torch.bfloat16


def test_w03_gemm_fp16_shapes() -> None:
    results = run_w03()
    assert len(results) == len(WORKLOAD_GEMM_SHAPES)
    for result, (m_dim, _k, n_dim) in zip(results, WORKLOAD_GEMM_SHAPES, strict=True):
        assert result.shape == (m_dim, n_dim)
        assert result.result.dtype == torch.float16


def test_w04_sdpa_shapes() -> None:
    results = run_w04()
    assert len(results) == len(WORKLOAD_SDPA_SHAPES)
    for result, shape in zip(results, WORKLOAD_SDPA_SHAPES, strict=True):
        assert result.shape == shape


def test_w05_reduction_shapes() -> None:
    results = run_w05()
    assert len(results) == 2 * len(WORKLOAD_REDUCE_LENGTHS)
    for result in results:
        assert result.shape == ()


def test_w06_elementwise_shapes() -> None:
    results = run_w06()
    assert [r.case for r in results] == ["exp", "tanh", "rsqrt"]
    for result in results:
        assert result.shape == WORKLOAD_ELEMENTWISE_SHAPE


def test_w07_greedy_decode_shape() -> None:
    results = run_w07()
    assert len(results) == 1
    assert results[0].shape == (W07_PROMPT_LEN + W07_DECODE_STEPS,)
    assert results[0].result.dtype == torch.int64


def test_double_run_records_bitwise_identity() -> None:
    """Records identity. Does not fail if the two runs differ — that is CP-3 input."""
    report = double_run()
    write_double_run_report(report, RECORD_PATH)
    assert len(report.cases) > 0
    names = {case.workload for case in report.cases}
    assert names == {"W01", "W02", "W03", "W04", "W05", "W06", "W07"}
