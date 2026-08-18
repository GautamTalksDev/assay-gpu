"""CPU tests for GEMM checksum algebra and the check_gemm detector."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from assay.abft.check import (
    CheckStatus,
    GemmCheckConfig,
    check_gemm,
    decide_from_lookup,
)
from assay.abft.gemm import (
    gemm_checksum_fp64,
    normalized_checksum_residual,
    sum_elements_fp64,
)
from assay.abft.overhead import measure_checksum_overhead
from assay.abft.reduce import CheckBackend, ones_matvec_pytorch
from assay.noise.floats import encode_f64
from assay.noise.lookup import (
    CharacterizationStatus,
    ToleranceLookup,
    lookup_abft_tolerance,
)
from assay.noise.methodology import load_methodology
from assay.reference.compute import matmul_fp64
from assay.reference.spec import REFERENCE_GEMM_SHAPES

pytestmark = pytest.mark.cpu

REPO_NOISE = Path(__file__).resolve().parents[1] / "data" / "noisefloor"


def test_checksum_matches_sum_of_integer_product() -> None:
    left = np.ascontiguousarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.dtype("<f8"))
    right = np.ascontiguousarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.dtype("<f8"))
    product = matmul_fp64(left, right)
    checksum = gemm_checksum_fp64(left, right)
    total = sum_elements_fp64(product)
    assert checksum == np.float64(134.0)
    assert total == np.float64(134.0)
    abs_res, norm = normalized_checksum_residual(total, checksum)
    assert abs_res == np.float64(0.0)
    assert norm == np.float64(0.0)


def test_ones_matvec_integer_row_sums() -> None:
    matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    summed = ones_matvec_pytorch(matrix)
    assert torch.equal(summed, torch.tensor([6.0, 15.0], dtype=torch.float32))


def test_check_gemm_repo_noisefloor_is_inconclusive_even_when_corrupt() -> None:
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
    product = left @ right
    config = GemmCheckConfig(noisefloor_dir=REPO_NOISE, workload="W01")
    clean = check_gemm(left, right, product, config)
    assert clean.status is CheckStatus.INCONCLUSIVE
    assert clean.residual == 0.0
    assert clean.threshold is None
    method = load_methodology(REPO_NOISE)
    assert clean.noisefloor_spec_version == method.spec_id
    assert clean.n_samples == 0
    assert clean.min_samples == method.min_samples
    broken = product.clone()
    broken[0, 0] = 10_000.0
    dirty = check_gemm(left, right, broken, config)
    assert dirty.status is CheckStatus.INCONCLUSIVE
    assert dirty.residual > 0.0
    assert dirty.threshold is None


def test_check_gemm_uncharacterized_never_fail() -> None:
    left = torch.ones((4, 3), dtype=torch.float32)
    right = torch.ones((3, 5), dtype=torch.float32)
    product = torch.zeros((4, 5), dtype=torch.float32)
    result = check_gemm(
        left,
        right,
        product,
        GemmCheckConfig(noisefloor_dir=REPO_NOISE, workload="W01"),
    )
    assert result.status is CheckStatus.INCONCLUSIVE
    assert result.status is not CheckStatus.FAIL


def _write_tiny_methodology(dest: Path, spec_id: str) -> None:
    payload = {
        "spec_id": spec_id,
        "target_quantile": "1/2",
        "target_quantile_note": "unit-test methodology; not noisefloor-v1",
        "abft_note": "fixture",
        "never_extrapolate": True,
    }
    dest.joinpath("methodology-v1.json").write_text(
        json.dumps(payload) + "\n", encoding="utf-8"
    )


def _write_run(
    dest: Path,
    *,
    residuals: list[float],
    gpu_model: str = "TestGPU",
    shape: tuple[int, int, int] = (2, 2, 2),
) -> None:
    samples = [
        {
            "workload": "W01",
            "dtype": "float32",
            "shape": list(shape),
            "repeat": index,
            "abft_residual_normalized": encode_f64(value),
        }
        for index, value in enumerate(residuals)
    ]
    payload = {"gpu_model": gpu_model, "samples": samples, "aggregates": []}
    dest.joinpath("run-test.json").write_text(
        json.dumps(payload) + "\n", encoding="utf-8"
    )


def test_decide_pass_fail_and_ambiguous_band() -> None:
    lookup = ToleranceLookup(
        status=CharacterizationStatus.CHARACTERIZED,
        reason="fixture",
        workload="W01",
        dtype="float32",
        shape=(2, 2, 2),
        n_samples=2,
        min_samples=2,
        target_quantile="1/2",
        p_quantile_residual_hex=(0.01).hex(),
        p_quantile_residual_decimal="0.01",
        sample_max_residual_hex=(0.05).hex(),
        source_files=("run-test.json",),
        gpu_models=("TestGPU",),
    )
    status, _reason = decide_from_lookup(0.01, lookup)
    assert status is CheckStatus.PASS
    status, _reason = decide_from_lookup(0.03, lookup)
    assert status is CheckStatus.INCONCLUSIVE
    status, _reason = decide_from_lookup(0.06, lookup)
    assert status is CheckStatus.FAIL
    empty = ToleranceLookup(
        status=CharacterizationStatus.UNCHARACTERIZED,
        reason="no data",
        workload="W01",
        dtype="float32",
        shape=(2, 2, 2),
        n_samples=0,
        min_samples=2,
        target_quantile="1/2",
        p_quantile_residual_hex=None,
        p_quantile_residual_decimal=None,
        sample_max_residual_hex=None,
        source_files=(),
        gpu_models=(),
    )
    status, _reason = decide_from_lookup(1.0, empty)
    assert status is CheckStatus.INCONCLUSIVE


def test_check_gemm_characterized_pass_fail_ambiguous(tmp_path: Path) -> None:
    dest = tmp_path / "noisefloor"
    dest.mkdir()
    _write_tiny_methodology(dest, "noisefloor-test-half")
    _write_run(dest, residuals=[0.01, 0.05])
    found = lookup_abft_tolerance(
        dest,
        workload="W01",
        dtype="float32",
        shape=(2, 2, 2),
        gpu_model="TestGPU",
    )
    assert found.status is CharacterizationStatus.CHARACTERIZED
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
    product = left @ right
    config = GemmCheckConfig(
        noisefloor_dir=dest,
        workload="W01",
        gpu_model="TestGPU",
    )
    clean = check_gemm(left, right, product, config)
    assert clean.status is CheckStatus.PASS
    assert clean.threshold is not None
    assert clean.noisefloor_spec_version == "noisefloor-test-half"
    assert clean.n_samples == 2
    assert clean.residual == 0.0

    assert found.p_quantile_residual_hex is not None
    assert found.sample_max_residual_hex is not None
    lo = float.fromhex(found.p_quantile_residual_hex)
    hi = float.fromhex(found.sample_max_residual_hex)
    total = float(product.sum().item())
    band = (lo + hi) / 2
    delta_band = band * total / (1 - band)
    ambiguous_c = product.clone()
    ambiguous_c[0, 0] = ambiguous_c[0, 0] + delta_band
    mid = check_gemm(left, right, ambiguous_c, config)
    assert mid.status is CheckStatus.INCONCLUSIVE
    assert mid.threshold == lo
    assert mid.sample_max == hi

    delta_fail = 2 * hi * total / (1 - hi)
    fail_c = product.clone()
    fail_c[0, 0] = fail_c[0, 0] + delta_fail
    dirty = check_gemm(left, right, fail_c, config)
    assert dirty.status is CheckStatus.FAIL
    assert dirty.threshold == lo
    assert dirty.n_samples == 2
    assert dirty.noisefloor_spec_version == "noisefloor-test-half"


def test_check_gemm_carries_threshold_and_spec_on_every_result() -> None:
    left = torch.ones((2, 2), dtype=torch.float32)
    right = torch.ones((2, 2), dtype=torch.float32)
    product = left @ right
    result = check_gemm(
        left,
        right,
        product,
        GemmCheckConfig(noisefloor_dir=REPO_NOISE, workload="W01"),
    )
    method = load_methodology(REPO_NOISE)
    assert result.noisefloor_spec_version == method.spec_id
    assert result.n_samples == 0
    assert result.min_samples == method.min_samples
    assert result.threshold is None
    assert result.backend == CheckBackend.PYTORCH.value


def test_ones_matvec_reference_shapes_match_row_sum() -> None:
    for rows, _inner, cols in REFERENCE_GEMM_SHAPES:
        matrix = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
        summed = ones_matvec_pytorch(matrix)
        expected = matrix.sum(dim=1)
        assert torch.equal(summed, expected)


def test_checksum_on_cpu_matches_same_device_path() -> None:
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
    product = left @ right
    same = check_gemm(
        left,
        right,
        product,
        GemmCheckConfig(noisefloor_dir=REPO_NOISE, workload="W01"),
    )
    cpu = check_gemm(
        left,
        right,
        product,
        GemmCheckConfig(
            noisefloor_dir=REPO_NOISE,
            workload="W01",
            checksum_on_cpu=True,
        ),
    )
    assert same.residual_hex == cpu.residual_hex
    assert cpu.status is CheckStatus.INCONCLUSIVE


def test_measure_checksum_overhead_cpu() -> None:
    measured = measure_checksum_overhead(
        shape=(32, 32, 32),
        repeats=3,
        backend=CheckBackend.PYTORCH,
        device=torch.device("cpu"),
    )
    assert measured.repeats == 3
    assert measured.gemm_seconds > 0.0
    assert measured.checksum_seconds > 0.0
    assert measured.backend == "pytorch"
