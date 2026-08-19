"""CPU tests for residual-v3 flip matrix sweep helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from assay.inject import BitClass
from assay.noise.floats import encode_f64
from assay.noise.sweep_v3_flips import (
    CLEAN_MAX_V3,
    FLIP_COUNTS,
    _completed_triples,
    _has_metadata_header,
    load_flip_samples,
    summarize_flip_ratios,
    summarize_from_jsonl,
)


@pytest.mark.cpu
def test_summarize_flip_ratios_min_median_max_and_count() -> None:
    collected: dict[tuple[str, int], list[float]] = {
        (bit_class.value, n_flips): [0.5, 0.5, 0.5]
        for bit_class in BitClass
        for n_flips in FLIP_COUNTS
    }
    collected[("EXPONENT_LOW", 1)] = [0.9, 1.068, 2.0]
    collected[("EXPONENT_HIGH", 1)] = [1.1, 10.0, math.inf]
    rows = summarize_flip_ratios(collected)
    by_key = {(row.bit_class, row.n_flips): row for row in rows}
    low = by_key[("EXPONENT_LOW", 1)]
    assert low.n == 3
    assert low.min_ratio == 0.9
    assert low.median_ratio == 1.068
    assert low.max_ratio == 2.0
    assert low.n_exceeding_one == 2
    high = by_key[("EXPONENT_HIGH", 1)]
    assert high.n_exceeding_one == 3
    sign = by_key[("SIGN", 1)]
    assert sign.n_exceeding_one == 0


@pytest.mark.cpu
def test_completed_triples_skips_metadata(tmp_path: Path) -> None:
    path = tmp_path / "flips.jsonl"
    path.write_text(
        json.dumps({"record_type": "metadata", "clean_max_v3": encode_f64(CLEAN_MAX_V3)})
        + "\n"
        + json.dumps({"bit_class": "SIGN", "n_flips": 1, "sample_index": 0})
        + "\n"
        + json.dumps({"bit_class": "SIGN", "n_flips": 1, "sample_index": 3})
        + "\n",
        encoding="utf-8",
    )
    assert _has_metadata_header(path)
    assert _completed_triples(path) == {("SIGN", 1, 0), ("SIGN", 1, 3)}


@pytest.mark.cpu
def test_summarize_from_jsonl_uses_metadata_clean_max(tmp_path: Path) -> None:
    path = tmp_path / "flips.jsonl"
    clean = 2.0e-6
    rows = [
        {"record_type": "metadata", "clean_max_v3": encode_f64(clean)},
        {
            "bit_class": "SIGN",
            "n_flips": 1,
            "sample_index": 0,
            "r_max": encode_f64(4.0e-6),
        },
        {
            "bit_class": "SIGN",
            "n_flips": 1,
            "sample_index": 1,
            "r_max": encode_f64(1.0e-6),
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    metadata, samples = load_flip_samples(path)
    assert metadata is not None
    summary = summarize_from_jsonl(path)
    sign = next(row for row in summary if row.bit_class == "SIGN" and row.n_flips == 1)
    assert sign.n == 2
    assert sign.min_ratio == pytest.approx(0.5)
    assert sign.max_ratio == pytest.approx(2.0)
    assert sign.n_exceeding_one == 1
    assert len(samples) == 2
