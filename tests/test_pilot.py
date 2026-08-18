"""CPU tests for independent GEMM sampling and the ABFT residual pilot summary."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from assay.noise.lookup import lookup_abft_tolerance
from assay.noise.pilot import summarize_residuals
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
