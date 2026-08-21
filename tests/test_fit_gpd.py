"""CPU tests for GPD FPR fit helpers (synthetic exceedances)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from assay.noise.floats import encode_f64

# Import from scripts/ via path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from fit_gpd import (  # noqa: E402
    M_DIM,
    collect_exceedances,
    fit_gpd_mle,
    gpd_return_level,
    load_pass2,
)


pytestmark = pytest.mark.cpu


def _gpd_sample(rng: np.random.Generator, xi: float, sigma: float, n: int) -> np.ndarray:
    # Inverse CDF: U ~ Unif, Y = sigma/xi * ((1-U)^(-xi) - 1) for xi != 0
    u = rng.random(n)
    if abs(xi) < 1e-12:
        return -sigma * np.log(1.0 - u)
    return (sigma / xi) * ((1.0 - u) ** (-xi) - 1.0)


def test_gpd_mle_recovers_negative_xi() -> None:
    rng = np.random.Generator(np.random.PCG64(20260821))
    xi_true = -0.2
    sigma_true = 1.5e-7
    y = _gpd_sample(rng, xi_true, sigma_true, 5000)
    fit = fit_gpd_mle(y, n_total=1_000_000, n_rate=5000, u=1e-6)
    assert fit.xi == pytest.approx(xi_true, abs=0.08)
    assert fit.sigma == pytest.approx(sigma_true, rel=0.15)
    assert fit.xi_ci[0] < fit.xi < fit.xi_ci[1]
    assert fit.sigma_ci[0] < fit.sigma < fit.sigma_ci[1]


def test_return_level_monotone_in_p() -> None:
    rng = np.random.Generator(np.random.PCG64(7))
    y = _gpd_sample(rng, -0.15, 2e-7, 3000)
    fit = fit_gpd_mle(y, n_total=500_000, n_rate=3000, u=1e-6)
    x1 = gpd_return_level(fit, 1e-8)
    x2 = gpd_return_level(fit, 1e-10)
    assert x2 > x1 > fit.u


def test_collect_exceedances_from_top64(tmp_path: Path) -> None:
    u = 1.0e-6
    top = [u + 1e-7, u + 2e-7, u - 1e-8] + [u * 0.5] * 61
    row = {
        "pass": 2,
        "phase": "fpr_clean",
        "sample_index": 0,
        "shape": [M_DIM, M_DIM, M_DIM],
        "n_above_p99": 2,
        "r_top64": [encode_f64(float(v)) for v in top],
    }
    path = tmp_path / "fpr.jsonl"
    path.write_text(
        json.dumps({"record_type": "metadata"})
        + "\n"
        + json.dumps({"record_type": "pot_thresholds", "p95": encode_f64(0.8e-6),
                      "p99": encode_f64(u), "p999": encode_f64(1.2e-6)})
        + "\n"
        + json.dumps(row)
        + "\n",
        encoding="utf-8",
    )
    samples, frozen = load_pass2(path)
    assert frozen is not None
    assert len(samples) == 1
    y, n_rate, counts = collect_exceedances(samples, u=u, count_key="n_above_p99")
    assert n_rate == 2
    assert counts == [2]
    assert y.shape == (2,)
    assert np.allclose(sorted(y), [1e-7, 2e-7])
