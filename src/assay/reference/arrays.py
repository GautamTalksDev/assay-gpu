"""Seeded ndarray generation. PCG64, little-endian IEEE payloads."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from assay.reference.spec import (
    DISTRIBUTION_NORMAL,
    DISTRIBUTION_UNIFORM_POS,
    DISTRIBUTION_UNIFORM_UNIT,
)


def make_generator(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def generate_array(
    shape: tuple[int, ...],
    seed: int,
    distribution: str,
    *,
    dtype: np.dtype[np.generic] | None = None,
) -> npt.NDArray[np.generic]:
    """Draw an array. dtype defaults to little-endian float64."""
    resolved = np.dtype("<f8") if dtype is None else np.dtype(dtype)
    rng = make_generator(seed)
    if distribution == DISTRIBUTION_UNIFORM_UNIT:
        values = rng.uniform(-1.0, 1.0, size=shape)
    elif distribution == DISTRIBUTION_NORMAL:
        values = rng.standard_normal(size=shape)
    elif distribution == DISTRIBUTION_UNIFORM_POS:
        values = rng.uniform(1.0, 4.0, size=shape)
    else:
        msg = f"unknown distribution: {distribution}"
        raise ValueError(msg)
    return np.ascontiguousarray(values, dtype=resolved)


def generate_int64(
    shape: tuple[int, ...],
    seed: int,
    low: int,
    high: int,
) -> npt.NDArray[np.int64]:
    rng = make_generator(seed)
    values = rng.integers(low, high, size=shape, dtype=np.int64)
    return np.ascontiguousarray(values, dtype=np.dtype("<i8"))
