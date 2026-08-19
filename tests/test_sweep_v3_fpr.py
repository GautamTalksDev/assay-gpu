"""CPU tests for FPR tail-data sweep helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from assay.noise.floats import encode_f64
from assay.noise.sweep_v3_fpr import (
    HIST_BIN_COUNT,
    HIST_BIN_EDGES,
    HIST_HI,
    HIST_LO,
    PASS1_N,
    compute_frozen_pot_thresholds,
    histogram_bin_edges_list,
    load_fpr_state,
    pass1_memmap_path,
    require_frozen_pot_thresholds,
    sample_resume_key,
    summarize_r_rows,
)


@pytest.mark.cpu
def test_histogram_bin_edges_stable_module_constant() -> None:
    edges = histogram_bin_edges_list()
    assert len(edges) == HIST_BIN_COUNT + 1
    assert edges[0] == pytest.approx(HIST_LO)
    assert edges[-1] == pytest.approx(HIST_HI)
    assert np.allclose(edges, HIST_BIN_EDGES)
    # Repeated calls return identical edges.
    assert histogram_bin_edges_list() == edges


@pytest.mark.cpu
def test_sample_resume_key_pass_and_index() -> None:
    row = {"pass": 2, "sample_index": 42, "phase": "fpr_clean"}
    assert sample_resume_key(row) == (2, 42)


@pytest.mark.cpu
def test_require_frozen_pot_thresholds_raises_without_pass1() -> None:
    with pytest.raises(RuntimeError, match="pass 2 requires frozen POT thresholds"):
        require_frozen_pot_thresholds(pass_num=2, frozen=None)
    with pytest.raises(RuntimeError, match="missing key 'p99'"):
        require_frozen_pot_thresholds(pass_num=2, frozen={"p95": 1e-6})


@pytest.mark.cpu
def test_require_frozen_pot_thresholds_accepts_complete_dict() -> None:
    frozen = {"p95": 1e-7, "p99": 2e-7, "p999": 3e-7}
    assert require_frozen_pot_thresholds(pass_num=2, frozen=frozen) == frozen


@pytest.mark.cpu
def test_load_fpr_state_resume_and_frozen_thresholds(tmp_path: Path) -> None:
    path = tmp_path / "fpr.jsonl"
    lines = [
        {
            "record_type": "metadata",
            "study": "fpr-tail",
            "hist_bin_edges": histogram_bin_edges_list(),
        },
        {
            "phase": "fpr_clean",
            "pass": 1,
            "sample_index": 0,
            "r_max": encode_f64(1e-6),
        },
        {
            "record_type": "pot_thresholds",
            "p95": encode_f64(1.1e-7),
            "p99": encode_f64(2.2e-7),
            "p999": encode_f64(3.3e-7),
        },
        {
            "phase": "fpr_clean",
            "pass": 2,
            "sample_index": 0,
            "r_max": encode_f64(1e-6),
            "n_above_p95": 10,
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in lines) + "\n",
        encoding="utf-8",
    )
    metadata, done, frozen, pass1_complete, pass1_count = load_fpr_state(path)
    assert metadata is not None
    assert (1, 0) in done
    assert (2, 0) in done
    assert pass1_count == 1
    assert pass1_complete is True
    assert frozen is not None
    assert frozen["p99"] == pytest.approx(2.2e-7)


@pytest.mark.cpu
def test_compute_frozen_pot_thresholds_from_memmap(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    mm = np.memmap(
        tmp_path / "p1.f64",
        dtype=np.float64,
        mode="w+",
        shape=(PASS1_N, 16),
    )
    mm[:] = rng.lognormal(mean=-14.0, sigma=0.5, size=mm.shape)
    frozen = compute_frozen_pot_thresholds(mm, PASS1_N)
    flat = mm.reshape(-1)
    assert frozen["p95"] == pytest.approx(float(np.percentile(flat, 95)))
    assert frozen["p99"] == pytest.approx(float(np.percentile(flat, 99)))
    assert frozen["p999"] == pytest.approx(float(np.percentile(flat, 99.9)))


@pytest.mark.cpu
def test_summarize_r_rows_top64_and_histogram() -> None:
    r_np = np.logspace(-8, -4, 4096)
    summary = summarize_r_rows(r_np)
    assert len(summary["r_top64"]) == 64
    assert summary["r_top64"][0] >= summary["r_top64"][-1]
    assert len(summary["hist_counts"]) == HIST_BIN_COUNT
    assert int(summary["hist_counts"].sum()) == 4096


@pytest.mark.cpu
def test_pass1_memmap_path_suffix() -> None:
    assert pass1_memmap_path(Path("data/out.jsonl")) == Path(
        "data/out.jsonl.pass1.f64"
    )


@pytest.mark.cpu
def test_sweep_v3_fpr_importable_without_cycle() -> None:
    from assay.noise.sweep_v3_fpr import run_v3_fpr_sweep  # noqa: F401
