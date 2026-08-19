#!/usr/bin/env python3
"""Read a residual-v3 sweep JSONL and print summary statistics."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


def _load(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _decode(payload: dict) -> float:
    h = payload["hex"]
    if h == "nan":
        return math.nan
    if h in ("inf", "-inf"):
        return float(h)
    return float.fromhex(h)


def main() -> None:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <sweep-v3.jsonl>", file=sys.stderr)
        sys.exit(1)
    path = Path(sys.argv[1])
    rows = _load(path)
    if not rows:
        print("no samples", file=sys.stderr)
        sys.exit(1)

    r_medians = np.array([_decode(r["r_median"]) for r in rows])
    r_maxes = np.array([_decode(r["r_max"]) for r in rows])

    print(f"n_samples = {len(rows)}")
    print()
    print("r_median across samples:")
    for label, pct in [("median", 50), ("p90", 90), ("p99", 99), ("p99.9", 99.9), ("max", 100)]:
        if label == "max":
            val = np.max(r_medians)
        else:
            val = np.percentile(r_medians, pct)
        print(f"  {label:8s} = {val:.6e}")

    print()
    print("r_max across samples:")
    print(f"  median   = {np.median(r_maxes):.6e}")
    print(f"  max      = {np.max(r_maxes):.6e}")

    print()
    ratio = np.max(r_maxes) / np.median(r_medians) if np.median(r_medians) > 0 else math.inf
    print(f"max(r_max) / median(r_median) = {ratio:.2f}")

    # Row correlation
    all_r_rows = []
    for r in rows:
        decoded = [float.fromhex(h) for h in r["r_rows"]]
        all_r_rows.append(decoded)
    mat = np.array(all_r_rows)  # (n_samples, n_cols)
    n_cols = mat.shape[1]
    if n_cols < 2 or mat.shape[0] < 2:
        print("\ntoo few rows/samples for correlation")
        return

    corr = np.corrcoef(mat.T)  # (n_cols, n_cols)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    pairwise = corr[mask]
    finite = pairwise[np.isfinite(pairwise)]
    if len(finite) == 0:
        print("\nno finite pairwise correlations")
        return

    print()
    print(f"row-to-row correlation (2000 x {n_cols} matrix):")
    print(f"  mean pairwise Pearson = {np.mean(finite):.6f}")
    print(f"  std                   = {np.std(finite):.6f}")
    print(f"  min                   = {np.min(finite):.6f}")
    print(f"  max                   = {np.max(finite):.6f}")


if __name__ == "__main__":
    main()
