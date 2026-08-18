"""GPU characterization smoke: writes a real run file, does not claim a quantile."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from assay.noise.lookup import (
    CharacterizationStatus,
    assay_verdict,
    lookup_abft_tolerance,
)
from assay.noise.pilot import run_abft_pilot
from assay.noise.run import characterize_gemm

pytestmark = pytest.mark.gpu

REPO_METHOD = (
    Path(__file__).resolve().parents[1] / "data" / "noisefloor" / "methodology-v1.json"
)


def test_characterize_gemm_writes_uncharacterized_short_run(tmp_path: Path) -> None:
    assert torch.cuda.is_available()
    dest = tmp_path / "noisefloor"
    dest.mkdir()
    dest.joinpath("methodology-v1.json").write_text(
        REPO_METHOD.read_text(encoding="utf-8"), encoding="utf-8"
    )
    path = characterize_gemm(
        noisefloor_dir=dest,
        repeats=2,
        device_index=0,
        include_large=False,
        workload="W02",
        shape_filter=(512, 512, 512),
    )
    assert path.is_file()
    found = lookup_abft_tolerance(
        dest,
        workload="W02",
        dtype="bfloat16",
        shape=(512, 512, 512),
    )
    assert found.n_samples >= 2
    assert found.status is CharacterizationStatus.UNCHARACTERIZED
    assert assay_verdict(found) == "INCONCLUSIVE"
    assert found.p_quantile_residual_hex is None


def test_pilot_writes_outside_run_json_and_records_backend(tmp_path: Path) -> None:
    assert torch.cuda.is_available()
    dest = tmp_path / "noisefloor"
    dest.mkdir()
    dest.joinpath("methodology-v1.json").write_text(
        REPO_METHOD.read_text(encoding="utf-8"), encoding="utf-8"
    )
    path = run_abft_pilot(noisefloor_dir=dest, device_index=0, n_samples=2)
    assert path.is_file()
    assert path.parent.name == "pilot"
    assert not path.name.startswith("run-")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["not_a_characterization"] is True
    assert payload["backend"] == "pytorch"
    assert payload["n"] == 2
    found = lookup_abft_tolerance(
        dest,
        workload="W02",
        dtype="bfloat16",
        shape=(4096, 4096, 4096),
    )
    assert found.n_samples == 0
