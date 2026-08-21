"""CPU tests for GPD FPR fit helpers (synthetic exceedances)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from assay.noise.floats import encode_f64

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from fit_gpd import (  # noqa: E402
    M_DIM,
    TOP_TAIL_K,
    collect_exceedances,
    fit_gpd_mle,
    gpd_return_level,
    load_pass2,
)

pytestmark = pytest.mark.cpu


def _gpd_sample(rng: np.random.Generator, xi: float, sigma: float, n: int) -> np.ndarray:
    u = rng.random(n)
    if abs(xi) < 1e-12:
        return -sigma * np.log(1.0 - u)
    return (sigma / xi) * ((1.0 - u) ** (-xi) - 1.0)


def test_gpd_mle_recovers_negative_xi() -> None:
    rng = np.random.Generator(np.random.PCG64(20260821))
    xi_true = -0.2
    sigma_true = 1.5e-7
    y = _gpd_sample(rng, xi_true, sigma_true, 5000)
    fit = fit_gpd_mle(y, n_total=1_000_000, n_rate=5000, u=1e-6, label="test")
    assert fit.xi == pytest.approx(xi_true, abs=0.08)
    assert fit.sigma == pytest.approx(sigma_true, rel=0.15)
    assert fit.xi_ci[0] < fit.xi < fit.xi_ci[1]
    assert fit.sigma_ci[0] < fit.sigma < fit.sigma_ci[1]


def test_return_level_monotone_in_p() -> None:
    rng = np.random.Generator(np.random.PCG64(7))
    y = _gpd_sample(rng, -0.15, 2e-7, 3000)
    fit = fit_gpd_mle(y, n_total=500_000, n_rate=3000, u=1e-6, label="test")
    x1 = gpd_return_level(fit, 1e-8)
    x2 = gpd_return_level(fit, 1e-10)
    assert x2 > x1 > fit.u


def test_collect_exceedances_no_raise_on_truncation(tmp_path: Path) -> None:
    u = 1.0e-6
    # 64 recovered values above u; n_above=70 => 6 censored on (0, c]
    recovered = [u + 1e-6 - i * 1e-9 for i in range(TOP_TAIL_K)]
    top = recovered  # all > u
    row_ok = {
        "pass": 2,
        "phase": "fpr_clean",
        "sample_index": 0,
        "shape": [M_DIM, M_DIM, M_DIM],
        "n_above_p99": 2,
        "r_top64": [
            encode_f64(u + 2e-7),
            encode_f64(u + 1e-7),
        ]
        + [encode_f64(u * 0.5) for _ in range(62)],
    }
    row_trunc = {
        "pass": 2,
        "phase": "fpr_clean",
        "sample_index": 1,
        "shape": [M_DIM, M_DIM, M_DIM],
        "n_above_p99": 70,
        "r_top64": [encode_f64(float(v)) for v in top],
    }
    path = tmp_path / "fpr.jsonl"
    path.write_text(
        json.dumps({"record_type": "metadata"})
        + "\n"
        + json.dumps(row_ok)
        + "\n"
        + json.dumps(row_trunc)
        + "\n",
        encoding="utf-8",
    )
    samples, _ = load_pass2(path)
    col = collect_exceedances(
        samples, u=u, count_key="n_above_p99", threshold_name="p99"
    )
    assert col.summary.n_over_64 == 1
    assert col.summary.max_above == 70
    assert len(col.censor_n) == 1
    assert col.censor_n[0] == 6
    assert col.y_exclude.shape == (2,)
    assert col.y_with_truncated_recovered.shape == (2 + TOP_TAIL_K,)
    # censored fit must run
    fit = fit_gpd_mle(
        col.y_with_truncated_recovered,
        n_total=col.n_total_all,
        n_rate=col.n_rate_all,
        u=u,
        label="cens",
        censor_c=col.censor_c,
        censor_n=col.censor_n,
    )
    assert math.isfinite(fit.xi)  # noqa: need import
