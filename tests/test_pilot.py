"""CPU tests for independent GEMM sampling and the ABFT residual pilot summary."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

from assay.abft.reduce import (
    absolute_factor_scale,
    vector_residual_normalized,
    vector_residual_parts,
)
from assay.noise.floats import decode_f64
from assay.noise.lookup import lookup_abft_tolerance
from assay.noise.pilot import (
    pearson_correlation,
    selected_pilot_blas,
    study_normalizer_link,
    summarize_residuals,
)
from assay.noise.quantiles import order_statistic
from assay.reference.arrays import generate_array
from assay.reference.spec import (
    DISTRIBUTION_UNIFORM_UNIT,
    sample_factor_seed,
    seed_offset,
)
from assay.workload.gemm import gemm_numpy_pair

pytestmark = pytest.mark.cpu

REPO_NOISE = Path(__file__).resolve().parents[1] / "data" / "noisefloor"


def test_sample_index_zero_matches_legacy_seed_offset() -> None:
    assert sample_factor_seed(1, 3, 0, 0) == seed_offset(1, 6)
    assert sample_factor_seed(1, 3, 0, 1) == seed_offset(1, 7)
    left, right = gemm_numpy_pair(8, 8, 8, case_index=3)
    expect_a = generate_array((8, 8), seed_offset(1, 6), DISTRIBUTION_UNIFORM_UNIT)
    expect_b = generate_array((8, 8), seed_offset(1, 7), DISTRIBUTION_UNIFORM_UNIT)
    assert np.array_equal(left, expect_a)
    assert np.array_equal(right, expect_b)


def test_independent_sample_index_changes_factors() -> None:
    left0, right0 = gemm_numpy_pair(
        8, 8, 8, case_index=3, sample_index=0, workload_id=2
    )
    left1, right1 = gemm_numpy_pair(
        8, 8, 8, case_index=3, sample_index=1, workload_id=2
    )
    assert not np.array_equal(left0, left1)
    assert not np.array_equal(right0, right1)


def test_order_statistic_p99_on_known_list() -> None:
    samples = [float(index) for index in range(100)]
    assert order_statistic(samples, Fraction(99, 100)) == 98.0


def test_summarize_residuals_reports_ratios_and_running_p99() -> None:
    samples = [float(index + 1) for index in range(200)]
    summary = summarize_residuals(samples, prefixes=(100, 200))
    assert summary["n"] == 200
    assert summary["min"]["hex"] == (1.0).hex()
    assert summary["max"]["hex"] == (200.0).hex()
    running = summary["running_p99"]
    assert "100" in running
    assert "200" in running
    assert len(summary["sorted_residuals_hex"]) == 200


def test_lookup_ignores_pilot_directory(tmp_path: Path) -> None:
    dest = tmp_path / "noisefloor"
    dest.mkdir()
    dest.joinpath("methodology-v1.json").write_text(
        (REPO_NOISE / "methodology-v1.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    pilot = dest / "pilot"
    pilot.mkdir()
    payload = {
        "gpu_model": "Tesla_T4",
        "samples": [
            {
                "workload": "W02",
                "dtype": "bfloat16",
                "shape": [4096, 4096, 4096],
                "abft_residual_normalized": {"hex": (0.25).hex(), "decimal": "0.25"},
            }
        ],
    }
    (pilot / "run-should-not-count.json").write_text(
        json.dumps(payload) + "\n", encoding="utf-8"
    )
    found = lookup_abft_tolerance(
        dest,
        workload="W02",
        dtype="bfloat16",
        shape=(4096, 4096, 4096),
        gpu_model="Tesla_T4",
    )
    assert found.n_samples == 0
    assert found.p_quantile_residual_hex is None


def test_pilot_records_one_draw_per_sample_index() -> None:
    assert selected_pilot_blas(("cublas", "cublaslt")) == "cublas"
    assert selected_pilot_blas(("unavailable",)) == "unavailable"
    n_samples = 8
    blas_name = selected_pilot_blas(("cublas", "cublaslt"))
    jobs = [(index, blas_name) for index in range(n_samples)]
    assert len(jobs) == n_samples
    assert [job[0] for job in jobs] == list(range(n_samples))
    assert {job[1] for job in jobs} == {blas_name}


def test_sample_index_seeds_are_unique_across_pilot_n() -> None:
    seeds = [sample_factor_seed(2, 3, sample_index, 0) for sample_index in range(2000)]
    assert len(set(seeds)) == 2000
    assert sample_factor_seed(2, 3, 0, 0) != sample_factor_seed(2, 3, 1, 0)
    assert sample_factor_seed(2, 3, 0, 1) != sample_factor_seed(2, 3, 1, 1)


def test_vector_residual_parts_uses_absolute_factor_scale() -> None:
    left = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float64)
    right = torch.tensor([[5.0, -6.0], [7.0, -8.0]], dtype=torch.float64)
    product = left @ right
    c_e = product.sum(dim=1)
    a_be = left @ right.sum(dim=1)
    abs_residual, scale, normalized = vector_residual_parts(c_e, a_be, left, right)
    # |A| col sums [4, 6], |B| row sums [11, 15], dot = 134.
    assert scale == pytest.approx(134.0)
    assert abs_residual == pytest.approx(0.0)
    assert normalized == pytest.approx(0.0)
    assert absolute_factor_scale(left, right) == pytest.approx(134.0)
    assert vector_residual_normalized(c_e, a_be, left, right) == normalized


def test_grand_sum_cancellation_does_not_inflate_residual_v2() -> None:
    left = torch.ones((2, 2), dtype=torch.float64)
    right = torch.ones((2, 2), dtype=torch.float64)
    c_e = torch.tensor([1.0e-12, -1.0e-12], dtype=torch.float64)
    a_be = torch.tensor([2.0e-12, -1.0e-12], dtype=torch.float64)
    abs_residual, scale, normalized = vector_residual_parts(c_e, a_be, left, right)
    assert abs_residual == pytest.approx(1.0e-12)
    assert scale == pytest.approx(8.0)
    assert normalized == pytest.approx(1.0e-12 / 8.0)
    assert normalized < 1.0e-12


def test_pearson_and_normalizer_link_on_inverse_scale() -> None:
    scales = [0.01, 0.1, 1.0, 10.0, 100.0]
    abs_residuals = [1.0, 1.0, 1.0, 1.0, 1.0]
    normalized = [1.0 / scale for scale in scales]
    assert pearson_correlation(normalized, scales) < 0.0
    inv = [1.0 / scale for scale in scales]
    assert pearson_correlation(normalized, inv) == pytest.approx(1.0)
    link = study_normalizer_link(abs_residuals, scales, normalized)
    assert link["n"] == 5
    assert link["n_normalizer_zero"] == 0
    assert decode_f64(link["pearson_normalized_vs_inv_normalizer"]) == pytest.approx(
        1.0
    )
    assert decode_f64(link["pearson_normalized_vs_normalizer"]) < 0.0
    high_median = decode_f64(link["median_normalizer_at_or_above_p99_normalized"])
    overall_median = decode_f64(link["median_normalizer"])
    assert high_median < overall_median
