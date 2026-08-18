"""Canonical SHA-256 of an ndarray (contents, not the .npz wrapper)."""

from __future__ import annotations

import hashlib

import numpy as np
import numpy.typing as npt


def sha256_array(array: npt.NDArray[np.generic]) -> str:
    """Hash C-contiguous payload plus dtype.str and shape.

    .npz ZIP timestamps are not part of the hash. Regenerating on another
    machine must match this digest when the bytes match.
    """
    contiguous = np.ascontiguousarray(array)
    header = f"{contiguous.dtype.str}|{contiguous.shape}".encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()
