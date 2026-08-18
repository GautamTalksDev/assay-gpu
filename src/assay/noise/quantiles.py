"""Empirical quantiles from a finite sample. No interpolation past observed ranks."""

from __future__ import annotations

from fractions import Fraction


def empirical_quantile(samples: list[float], p: Fraction) -> float | None:
    """Inverse empirical CDF: x at index ceil(p*n)-1.

    Returns None when n is too small for p (caller must treat as UNCHARACTERIZED).
    """
    n = len(samples)
    missing = 1 - p
    if missing <= 0:
        return None
    min_n = int((missing.denominator + missing.numerator - 1) // missing.numerator)
    if n < min_n:
        return None
    ordered = sorted(samples)
    rank = int(p * n)
    if p * n > rank:
        rank += 1
    index = rank - 1
    index = max(index, 0)
    if index >= n:
        index = n - 1
    return ordered[index]
