#!/usr/bin/env python3
"""Analyze residual-v3 K-scaling sweep JSONL."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from assay.noise.floats import decode_f64
from assay.noise.sweep_v3_kscale import load_kscale_state


def _decode_row(payload: dict) -> float:
    return decode_f64(payload)


def _load_rows(path: Path) -> tuple[dict | None, list[dict], dict[int, float]]:
    metadata, clean_max_by_k, _done, _counts, _finalized = load_kscale_state(path)
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("record_type") in ("metadata", "clean_max"):
                continue
            if obj.get("phase") in ("clean", "flip"):
                rows.append(obj)
    return metadata, rows, clean_max_by_k


def _log_log_exponent(k_vals: list[float], medians: list[float]) -> float:
    if len(k_vals) < 2:
        return math.nan
    x = np.log(np.array(k_vals, dtype=np.float64))
    y = np.log(np.array(medians, dtype=np.float64))
    slope, _intercept = np.polyfit(x, y, 1)
    return float(slope)


def main() -> None:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <sweep-v3-kscale.jsonl>", file=sys.stderr)
        sys.exit(1)
    path = Path(sys.argv[1])
    metadata, rows, clean_max_by_k = _load_rows(path)
    if metadata:
        print("=== environment ===")
        for key in (
            "torch_version",
            "torch_cuda_version",
            "gpu_name",
            "git_sha",
            "k_values",
            "clean_n",
            "flip_n",
        ):
            print(f"  {key}: {metadata.get(key, 'n/a')}")
        print()

    clean_by_k: dict[int, list[dict]] = defaultdict(list)
    flip_by_k: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        k_val = int(row["K"])
        if row.get("phase") == "clean":
            clean_by_k[k_val].append(row)
        elif row.get("phase") == "flip":
            flip_by_k[k_val].append(row)

    if not clean_by_k:
        print("no clean samples", file=sys.stderr)
        sys.exit(1)

    print("=== clean floor by K ===")
    print("| K | n | med r_median | p90 r_median | p99 r_median | max r_median | med r_max | max r_max | clean_max |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    fit_k: list[float] = []
    fit_med: list[float] = []
    for k in sorted(clean_by_k):
        group = clean_by_k[k]
        r_med = np.array([_decode_row(r["r_median"]) for r in group])
        r_max = np.array([_decode_row(r["r_max"]) for r in group])
        cmax = clean_max_by_k.get(k, float(np.max(r_max)))
        fit_k.append(float(k))
        fit_med.append(float(np.median(r_med)))
        print(
            f"| {k} | {len(group)} | {np.median(r_med):.6e} | "
            f"{np.percentile(r_med, 90):.6e} | {np.percentile(r_med, 99):.6e} | "
            f"{np.max(r_med):.6e} | {np.median(r_max):.6e} | {np.max(r_max):.6e} | "
            f"{cmax:.6e} |"
        )

    exponent = _log_log_exponent(fit_k, fit_med)
    print()
    print(f"log-log fit exponent a (median r_median vs K): {exponent:.4f}")

    print()
    print("=== detection rate by K and cell ===")
    print("| K | bit_class | n_flips | n | detected | rate |")
    print("| --- | --- | --- | --- | --- | --- |")
    for k in sorted(flip_by_k):
        cells: dict[tuple[str, int], list[bool]] = defaultdict(list)
        for row in flip_by_k[k]:
            key = (str(row["bit_class"]), int(row["n_flips"]))
            cells[key].append(bool(row.get("detected", False)))
        for (bit_class, n_flips), flags in sorted(cells.items()):
            n_det = sum(1 for flag in flags if flag)
            rate = n_det / len(flags) if flags else math.nan
            print(
                f"| {k} | {bit_class} | {n_flips} | {len(flags)} | "
                f"{n_det}/{len(flags)} | {rate:.2%} |"
            )

    print()
    print("=== SIGN 1-flip detection vs K ===")
    print("| K | n | detected | rate |")
    print("| --- | --- | --- | --- |")
    for k in sorted(flip_by_k):
        sign1 = [
            row
            for row in flip_by_k[k]
            if row.get("bit_class") == "SIGN" and int(row.get("n_flips", 0)) == 1
        ]
        if not sign1:
            continue
        n_det = sum(1 for row in sign1 if row.get("detected"))
        rate = n_det / len(sign1)
        print(f"| {k} | {len(sign1)} | {n_det}/{len(sign1)} | {rate:.2%} |")


if __name__ == "__main__":
    main()
