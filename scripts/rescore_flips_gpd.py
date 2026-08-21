#!/usr/bin/env python3
"""Re-score the K=4096 flip matrix at a fixed GPD-extrapolated threshold.

Does not re-inject, does not re-run GEMMs, does not edit RESIDUAL.md,
does not write a results document.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from assay.noise.floats import decode_f64
from assay.noise.sweep_v3_flips import FLIP_COUNTS, load_flip_samples

# Locked outputs from CP-FPR-CENSOR (p99 exclude-truncated fit). Not averaged.
THRESHOLD_GPD = 3.144535556e-06
THRESHOLD_GPD_PROVENANCE = (
    "GPD POT extrapolate 1-2.44e-10; u=p99=1.392735e-06; "
    "fit=p99_exclude_truncated (CP-FPR-CENSOR); xi=-0.07806036162, "
    "sigma=1.83504137e-07"
)
# Comparison baseline used in F4 / F2 (K=4096 clean max from K-scale study).
CLEAN_MAX_K4096 = 2.684525e-06

# Exact row order of the flip matrix in docs/RESULTS-KT1-v3.md.
CELL_ORDER: tuple[tuple[str, int], ...] = tuple(
    (bit_class, n_flips)
    for bit_class in (
        "EXPONENT_HIGH",
        "EXPONENT_LOW",
        "MANTISSA_HIGH",
        "MANTISSA_LOW",
        "SIGN",
    )
    for n_flips in FLIP_COUNTS
)


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return "inf"
    return format(value, ".6g")


def _median(ordered: list[float]) -> float:
    count = len(ordered)
    if count < 1:
        msg = "median requires at least one sample"
        raise ValueError(msg)
    if count % 2 == 1:
        return ordered[count // 2]
    return 0.5 * (ordered[count // 2 - 1] + ordered[count // 2])


def _load_rmax_by_cell(
    path: Path,
) -> dict[tuple[str, int], list[float]]:
    _metadata, samples = load_flip_samples(path)
    by_cell: dict[tuple[str, int], list[float]] = {}
    for row in samples:
        key = (str(row["bit_class"]), int(row["n_flips"]))
        r_max = decode_f64(row["r_max"])
        by_cell.setdefault(key, []).append(float(r_max))
    missing = [cell for cell in CELL_ORDER if cell not in by_cell]
    if missing:
        msg = f"flip JSONL missing cells: {missing}"
        raise RuntimeError(msg)
    return by_cell


def _cell_stats(
    r_maxes: list[float], *, threshold: float
) -> tuple[int, float, float, float, int]:
    ratios = [
        math.inf if not math.isfinite(r) else r / threshold for r in r_maxes
    ]
    ordered = sorted(ratios)
    n_over = sum(1 for value in ratios if (not math.isfinite(value)) or value > 1.0)
    return (
        len(ratios),
        ordered[0],
        _median(ordered),
        ordered[-1],
        n_over,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flips",
        type=Path,
        default=Path("data/sweep-v3-flips.jsonl"),
        help="Existing flip-matrix JSONL (default: data/sweep-v3-flips.jsonl)",
    )
    parser.add_argument(
        "--threshold-gpd",
        type=float,
        default=THRESHOLD_GPD,
        help="GPD-extrapolated threshold (default: p99 exclude-truncated)",
    )
    parser.add_argument(
        "--clean-max",
        type=float,
        default=CLEAN_MAX_K4096,
        help="Comparison clean-max threshold (default: K=4096 clean max)",
    )
    args = parser.parse_args(argv)

    if not args.flips.is_file():
        print(f"ERROR: flip JSONL not found: {args.flips}", file=sys.stderr)
        return 2

    by_cell = _load_rmax_by_cell(args.flips)

    print("=== flip matrix re-score (no re-injection) ===")
    print(f"flips_path = {args.flips}")
    print(f"threshold_gpd = {args.threshold_gpd:.10g}")
    print(f"threshold_gpd_provenance = {THRESHOLD_GPD_PROVENANCE}")
    print(f"clean_max_comparison = {args.clean_max:.10g}")
    print()

    print(
        "| bit_class | n_flips | n | min ratio | median ratio | max ratio | n(ratio>1) |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- |")
    gpd_detected: dict[tuple[str, int], tuple[int, int]] = {}
    for bit_class, n_flips in CELL_ORDER:
        key = (bit_class, n_flips)
        n, mn, med, mx, n_over = _cell_stats(
            by_cell[key], threshold=args.threshold_gpd
        )
        gpd_detected[key] = (n_over, n)
        print(
            f"| {bit_class} | {n_flips} | {n} | "
            f"{_fmt(mn)} | {_fmt(med)} | {_fmt(mx)} | {n_over}/{n} |"
        )

    print()
    print(
        "| bit_class | n_flips | detected @ clean_max "
        f"({args.clean_max:.6e}) | detected @ threshold_gpd | delta_pp |"
    )
    print("| --- | --- | --- | --- | --- |")
    n_increased = 0
    for bit_class, n_flips in CELL_ORDER:
        key = (bit_class, n_flips)
        n = len(by_cell[key])
        n_clean = sum(
            1
            for r in by_cell[key]
            if (not math.isfinite(r)) or r > args.clean_max
        )
        n_gpd, n_g = gpd_detected[key]
        if n_gpd > n_clean:
            n_increased += 1
        delta_pp = 100.0 * (n_gpd / n_g - n_clean / n)
        print(
            f"| {bit_class} | {n_flips} | {n_clean}/{n} | {n_gpd}/{n_g} | "
            f"{delta_pp:+.2f} |"
        )

    print()
    eh = gpd_detected[("EXPONENT_HIGH", 1)]
    el = gpd_detected[("EXPONENT_LOW", 1)]
    eh_c = sum(
        1
        for r in by_cell[("EXPONENT_HIGH", 1)]
        if (not math.isfinite(r)) or r > args.clean_max
    )
    el_c = sum(
        1
        for r in by_cell[("EXPONENT_LOW", 1)]
        if (not math.isfinite(r)) or r > args.clean_max
    )
    pooled_gpd = eh[0] + el[0]
    pooled_clean = eh_c + el_c
    pooled_n = eh[1] + el[1]
    print(
        f"pooled_exponent_n_flips_1 @ clean_max = "
        f"{pooled_clean}/{pooled_n} "
        f"({100.0 * pooled_clean / pooled_n:.4g}%)"
    )
    print(
        f"pooled_exponent_n_flips_1 @ threshold_gpd = "
        f"{pooled_gpd}/{pooled_n} "
        f"({100.0 * pooled_gpd / pooled_n:.4g}%)"
    )

    mant_low_total = sum(
        gpd_detected[("MANTISSA_LOW", nf)][0] for nf in FLIP_COUNTS
    )
    mant_low_n = sum(gpd_detected[("MANTISSA_LOW", nf)][1] for nf in FLIP_COUNTS)
    print(
        f"MANTISSA_LOW_total @ threshold_gpd = "
        f"{mant_low_total}/{mant_low_n}"
    )

    sign_gpd = gpd_detected[("SIGN", 1)]
    print(f"SIGN_n_flips_1 @ threshold_gpd = {sign_gpd[0]}/{sign_gpd[1]}")
    print(f"cells_where_detection_increased = {n_increased}")

    if n_increased != 0:
        print(
            "ERROR: detection increased at higher threshold — scoring bug",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
