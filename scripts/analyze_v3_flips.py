#!/usr/bin/env python3
"""Read a residual-v3 flip-matrix JSONL and print per-cell detection stats."""

from __future__ import annotations

import math
import sys
from pathlib import Path

from assay.noise.floats import decode_f64
from assay.noise.sweep_v3_flips import (
    CLEAN_MAX_V3,
    summarize_flip_ratios,
    load_flip_samples,
)


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return "inf"
    return format(value, ".6g")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <sweep-v3-flips.jsonl>", file=sys.stderr)
        sys.exit(1)
    path = Path(sys.argv[1])
    metadata, samples = load_flip_samples(path)
    if metadata:
        print("=== environment ===")
        for key in (
            "torch_version",
            "torch_cuda_version",
            "gpu_name",
            "driver_version",
            "numpy_version",
            "python_version",
            "git_sha",
            "residual_version",
        ):
            print(f"  {key}: {metadata.get(key, 'n/a')}")
        if "clean_max_v3" in metadata:
            print(f"  clean_max_v3: {decode_f64(metadata['clean_max_v3']):.6e}")
            print(f"  clean_max_v3_source: {metadata.get('clean_max_v3_source', 'n/a')}")
            print(f"  clean_max_v3_n: {metadata.get('clean_max_v3_n', 'n/a')}")
        print()
    if not samples:
        print("no samples", file=sys.stderr)
        sys.exit(1)

    clean_max = CLEAN_MAX_V3
    if metadata is not None and "clean_max_v3" in metadata:
        clean_max = decode_f64(metadata["clean_max_v3"])

    ratios: dict[tuple[str, int], list[float]] = {}
    for row in samples:
        key = (str(row["bit_class"]), int(row["n_flips"]))
        if "ratio_to_clean_max" in row:
            ratio = decode_f64(row["ratio_to_clean_max"])
        else:
            r_max = decode_f64(row["r_max"])
            ratio = math.inf if not math.isfinite(r_max) else r_max / clean_max
        ratios.setdefault(key, []).append(float(ratio))

    rows = summarize_flip_ratios(ratios)
    print(f"n_sample_lines = {len(samples)}")
    print(f"clean_max_v3 = {clean_max:.6e}")
    print()
    print(
        "| bit_class | n_flips | n | min ratio | median ratio | max ratio | n(ratio>1) |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        frac = f"{row.n_exceeding_one}/{row.n}"
        print(
            f"| {row.bit_class} | {row.n_flips} | {row.n} | "
            f"{_fmt(row.min_ratio)} | {_fmt(row.median_ratio)} | "
            f"{_fmt(row.max_ratio)} | {frac} |"
        )


if __name__ == "__main__":
    main()
