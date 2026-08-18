"""CPU tests: reference generation is bit-reproducible."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from assay.reference.catalog import generate_catalog, manifest_dict
from assay.reference.hashing import sha256_array
from assay.reference.serialize import write_catalog

pytestmark = pytest.mark.cpu

REPO_REFERENCE = Path(__file__).resolve().parents[1] / "data" / "reference"


def test_generate_catalog_twice_identical_hashes() -> None:
    first, _ = generate_catalog()
    second, _ = generate_catalog()
    assert [a.sha256 for a in first] == [a.sha256 for a in second]
    assert [a.name for a in first] == [a.name for a in second]
    assert first[0].numpy_version == np.__version__


def test_write_and_reload_hashes_match(tmp_path: Path) -> None:
    artifacts = write_catalog(tmp_path)
    artifacts_again = write_catalog(tmp_path / "second")
    assert [a.sha256 for a in artifacts] == [a.sha256 for a in artifacts_again]
    for artifact in artifacts:
        loaded = np.load(tmp_path / artifact.file)
        array = loaded[artifact.key]
        assert sha256_array(array) == artifact.sha256


def test_committed_manifest_matches_regeneration() -> None:
    manifest_path = REPO_REFERENCE / "manifest.json"
    assert manifest_path.is_file(), "run: uv run assay reference generate"
    committed = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts, _ = generate_catalog()
    generated = manifest_dict(artifacts)
    committed_hashes = {item["name"]: item["sha256"] for item in committed["artifacts"]}
    generated_hashes = {item["name"]: item["sha256"] for item in generated["artifacts"]}
    assert committed_hashes == generated_hashes
    assert committed["numpy_version"] == np.__version__
    assert committed["numpy_version"] == generated["numpy_version"]
