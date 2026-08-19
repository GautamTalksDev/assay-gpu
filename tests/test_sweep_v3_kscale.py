"""CPU tests for K-scaling sweep helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from assay.noise.floats import encode_f64
from assay.noise.sweep_v3_kscale import (
    CLEAN_N,
    load_kscale_state,
    parse_k_values,
    require_clean_max_before_flip,
    sample_resume_key,
)


@pytest.mark.cpu
def test_parse_k_values() -> None:
    assert parse_k_values("512,1024,4096") == (512, 1024, 4096)


@pytest.mark.cpu
def test_sample_resume_key_clean_and_flip() -> None:
    clean = {"phase": "clean", "K": 512, "sample_index": 3}
    flip = {
        "phase": "flip",
        "K": 1024,
        "bit_class": "SIGN",
        "n_flips": 1,
        "sample_index": 7,
    }
    assert sample_resume_key(clean) == ("clean", 512, "", -1, 3)
    assert sample_resume_key(flip) == ("flip", 1024, "SIGN", 1, 7)


@pytest.mark.cpu
def test_require_clean_max_before_flip_raises_if_clean_incomplete() -> None:
    with pytest.raises(RuntimeError, match="phase 2 for K=512 requires clean phase 1"):
        require_clean_max_before_flip(
            k=512,
            clean_max_by_k={},
            clean_counts={512: 100},
        )


@pytest.mark.cpu
def test_require_clean_max_before_flip_raises_if_max_missing() -> None:
    with pytest.raises(RuntimeError, match="clean_max\\[512\\] is missing"):
        require_clean_max_before_flip(
            k=512,
            clean_max_by_k={},
            clean_counts={512: CLEAN_N},
        )


@pytest.mark.cpu
def test_load_kscale_state_resume_and_phase_ordering(tmp_path: Path) -> None:
    path = tmp_path / "kscale.jsonl"
    lines = [
        {"record_type": "metadata", "study": "k-scaling"},
        {
            "phase": "clean",
            "K": 512,
            "sample_index": 0,
            "r_max": encode_f64(1e-6),
        },
        {
            "record_type": "clean_max",
            "K": 512,
            "clean_max": encode_f64(2e-6),
            "n_clean": CLEAN_N,
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in lines) + "\n",
        encoding="utf-8",
    )
    metadata, clean_max, done, counts, finalized = load_kscale_state(path)
    assert metadata is not None
    assert clean_max[512] == pytest.approx(2e-6)
    assert ("clean", 512, "", -1, 0) in done
    assert counts[512] == 1
    assert finalized == {512}
    assert require_clean_max_before_flip(
        k=512,
        clean_max_by_k=clean_max,
        clean_counts={512: CLEAN_N},
    ) == pytest.approx(2e-6)


@pytest.mark.cpu
def test_sweep_v3_kscale_importable_without_cycle() -> None:
    from assay.noise.sweep_v3_kscale import run_v3_kscale_sweep  # noqa: F401


@pytest.mark.cpu
def test_analyze_v3_flips_importable_without_cycle() -> None:
    from assay.noise.sweep_v3_flips import load_flip_samples  # noqa: F401
