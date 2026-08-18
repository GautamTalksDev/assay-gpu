"""CPU tests for noisefloor lookup and quantile rules."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest
import torch

from assay.noise.floats import encode_f64
from assay.noise.lookup import (
    CharacterizationStatus,
    assay_verdict,
    lookup_abft_tolerance,
)
from assay.noise.methodology import load_methodology
from assay.noise.quantiles import empirical_quantile
from assay.noise.run import _result_sha256

pytestmark = pytest.mark.cpu

REPO_NOISE = Path(__file__).resolve().parents[1] / "data" / "noisefloor"


def test_methodology_min_samples_is_ceil_inv_one_minus_p() -> None:
    method = load_methodology(REPO_NOISE)
    assert method.spec_id == "noisefloor-v1"
    assert method.target_quantile == Fraction(99999, 100000)
    missing = 1 - method.target_quantile
    expected = int((missing.denominator + missing.numerator - 1) // missing.numerator)
    assert method.min_samples == expected


def test_empirical_quantile_none_when_n_too_small() -> None:
    p = Fraction(99999, 100000)
    assert empirical_quantile([0.0, 1.0, 2.0], p) is None


def test_empirical_quantile_order_statistic_when_n_meets_min() -> None:
    p = Fraction(99999, 100000)
    missing = 1 - p
    min_n = int((missing.denominator + missing.numerator - 1) // missing.numerator)
    samples = [float(index) for index in range(min_n)]
    value = empirical_quantile(samples, p)
    assert value is not None
    assert value == float(min_n - 2)


def test_lookup_without_runs_is_inconclusive() -> None:
    found = lookup_abft_tolerance(
        REPO_NOISE,
        workload="W02",
        dtype="bfloat16",
        shape=(4096, 4096, 4096),
    )
    assert found.status is CharacterizationStatus.UNCHARACTERIZED
    assert assay_verdict(found) == "INCONCLUSIVE"
    assert found.n_samples == 0
    assert found.p_quantile_residual_hex is None


def test_lookup_short_run_stays_uncharacterized(tmp_path: Path) -> None:
    method_src = REPO_NOISE / "methodology-v1.json"
    dest = tmp_path / "noisefloor"
    dest.mkdir()
    dest.joinpath("methodology-v1.json").write_text(
        method_src.read_text(encoding="utf-8"), encoding="utf-8"
    )
    sample = {
        "workload": "W02",
        "dtype": "bfloat16",
        "shape": [4096, 4096, 4096],
        "repeat": 0,
        "blas_library": "cublas",
        "residual_version": "residual-v2",
        "abft_residual_normalized": encode_f64(0.25),
    }
    payload = {
        "gpu_model": "Tesla_T4",
        "samples": [sample, {**sample, "repeat": 1}],
        "aggregates": [],
    }
    (dest / "run-test.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    found = lookup_abft_tolerance(
        dest,
        workload="W02",
        dtype="bfloat16",
        shape=(4096, 4096, 4096),
        gpu_model="Tesla_T4",
    )
    assert found.status is CharacterizationStatus.UNCHARACTERIZED
    assert assay_verdict(found) == "INCONCLUSIVE"
    assert found.n_samples == 2
    assert found.p_quantile_residual_hex is None
    assert found.sample_max_residual_hex == (0.25).hex()


def test_lookup_ignores_residual_v1_samples(tmp_path: Path) -> None:
    method_src = REPO_NOISE / "methodology-v1.json"
    dest = tmp_path / "noisefloor"
    dest.mkdir()
    dest.joinpath("methodology-v1.json").write_text(
        method_src.read_text(encoding="utf-8"), encoding="utf-8"
    )
    sample = {
        "workload": "W02",
        "dtype": "bfloat16",
        "shape": [4096, 4096, 4096],
        "repeat": 0,
        "blas_library": "cublas",
        "residual_version": "residual-v1",
        "abft_residual_normalized": encode_f64(0.25),
    }
    payload = {
        "gpu_model": "Tesla_T4",
        "residual_version": "residual-v1",
        "samples": [sample],
        "aggregates": [],
    }
    (dest / "run-v1.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    found = lookup_abft_tolerance(
        dest,
        workload="W02",
        dtype="bfloat16",
        shape=(4096, 4096, 4096),
        gpu_model="Tesla_T4",
    )
    assert found.n_samples == 0
    assert found.p_quantile_residual_hex is None


def test_result_sha256_covers_dtypes_and_separates_float32_from_bfloat16() -> None:
    dtypes = (torch.float32, torch.bfloat16, torch.float16, torch.int8)
    for dtype in dtypes:
        digest = _result_sha256(torch.zeros(3, 5, dtype=dtype))
        assert len(digest) == 64
        int(digest, 16)

    raw = torch.arange(8, dtype=torch.uint8)
    as_f32 = raw.view(torch.float32)
    as_bf16 = raw.view(torch.bfloat16)
    f32_bytes = as_f32.contiguous().view(torch.uint8)
    bf16_bytes = as_bf16.contiguous().view(torch.uint8)
    assert torch.equal(f32_bytes, bf16_bytes)
    assert _result_sha256(as_f32) != _result_sha256(as_bf16)
