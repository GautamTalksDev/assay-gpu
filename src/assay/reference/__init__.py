"""Golden reference vectors and fp64 CPU compute."""

from assay.reference.catalog import Artifact, generate_catalog, manifest_dict
from assay.reference.hashing import sha256_array
from assay.reference.serialize import write_catalog

__all__ = [
    "Artifact",
    "generate_catalog",
    "manifest_dict",
    "sha256_array",
    "write_catalog",
]
