"""Write .npz files and manifest.json. Hashes are of arrays, not ZIP metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from assay.reference.catalog import Artifact, generate_catalog, manifest_dict


def write_catalog(directory: Path) -> list[Artifact]:
    directory.mkdir(parents=True, exist_ok=True)
    artifacts, files = generate_catalog()
    for filename, arrays in files.items():
        packed: dict[str, npt.NDArray[np.generic]] = {
            key: np.ascontiguousarray(value) for key, value in arrays.items()
        }
        np.savez(directory / filename, **cast(dict[str, Any], packed))
    manifest_path = directory / "manifest.json"
    payload = manifest_dict(artifacts)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artifacts
