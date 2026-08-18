"""Lookup a tolerance from measured noisefloor files. Never invent one."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from assay.noise.floats import decode_f64
from assay.noise.methodology import load_methodology
from assay.noise.quantiles import empirical_quantile


class CharacterizationStatus(StrEnum):
    CHARACTERIZED = "characterized"
    UNCHARACTERIZED = "uncharacterized"


@dataclass(frozen=True, slots=True)
class ToleranceLookup:
    status: CharacterizationStatus
    reason: str
    workload: str
    dtype: str
    shape: tuple[int, ...]
    n_samples: int
    min_samples: int
    target_quantile: str
    p_quantile_residual_hex: str | None
    p_quantile_residual_decimal: str | None
    sample_max_residual_hex: str | None
    source_files: tuple[str, ...]
    gpu_models: tuple[str, ...]


def _measurement_files(noisefloor_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(noisefloor_dir.glob("**/*.json")):
        if path.name in {"methodology-v1.json", "index.json"}:
            continue
        if path.name.startswith("run-") and path.suffix == ".json":
            files.append(path)
    return files


def _sample_match(
    sample: dict[str, Any],
    *,
    workload: str,
    dtype: str,
    shape: tuple[int, ...],
) -> bool:
    if sample.get("workload") != workload:
        return False
    if sample.get("dtype") != dtype:
        return False
    return tuple(sample.get("shape", ())) == shape


def lookup_abft_tolerance(
    noisefloor_dir: Path,
    *,
    workload: str,
    dtype: str,
    shape: tuple[int, ...],
    gpu_model: str | None = None,
) -> ToleranceLookup:
    """Tolerance is the empirical target quantile of ABFT normalized residual.

    Distinct GPU models are not pooled. If gpu_model is omitted and more than
    one model has data, the result is UNCHARACTERIZED (no extrapolation).
    """
    methodology = load_methodology(noisefloor_dir)
    residuals: list[float] = []
    models: set[str] = set()
    used: list[str] = []
    for path in _measurement_files(noisefloor_dir):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = str(payload.get("gpu_model", ""))
        if gpu_model is not None and model != gpu_model:
            continue
        matched = False
        for sample in payload.get("samples", []):
            if not _sample_match(sample, workload=workload, dtype=dtype, shape=shape):
                continue
            encoded = sample.get("abft_residual_normalized")
            if not isinstance(encoded, dict):
                continue
            residuals.append(decode_f64(encoded))
            matched = True
        if matched:
            models.add(model)
            used.append(str(path))
    n_total = len(residuals)
    min_n = methodology.min_samples
    target = str(methodology.target_quantile)
    empty = ToleranceLookup(
        status=CharacterizationStatus.UNCHARACTERIZED,
        reason="no noisefloor measurements for this configuration",
        workload=workload,
        dtype=dtype,
        shape=shape,
        n_samples=n_total,
        min_samples=min_n,
        target_quantile=target,
        p_quantile_residual_hex=None,
        p_quantile_residual_decimal=None,
        sample_max_residual_hex=None,
        source_files=tuple(used),
        gpu_models=tuple(sorted(models)),
    )
    if not residuals:
        return empty
    if gpu_model is None and len(models) > 1:
        return ToleranceLookup(
            status=CharacterizationStatus.UNCHARACTERIZED,
            reason=(
                "multiple GPU models present; pass gpu_model to avoid pooling "
                f"({sorted(models)})"
            ),
            workload=workload,
            dtype=dtype,
            shape=shape,
            n_samples=n_total,
            min_samples=min_n,
            target_quantile=target,
            p_quantile_residual_hex=None,
            p_quantile_residual_decimal=None,
            sample_max_residual_hex=None,
            source_files=tuple(used),
            gpu_models=tuple(sorted(models)),
        )
    max_hex = max(residuals).hex()
    if n_total < min_n:
        return ToleranceLookup(
            status=CharacterizationStatus.UNCHARACTERIZED,
            reason=(
                f"n={n_total} < min_samples={min_n} for quantile {target}; "
                "INCONCLUSIVE, do not extrapolate"
            ),
            workload=workload,
            dtype=dtype,
            shape=shape,
            n_samples=n_total,
            min_samples=min_n,
            target_quantile=target,
            p_quantile_residual_hex=None,
            p_quantile_residual_decimal=None,
            sample_max_residual_hex=max_hex,
            source_files=tuple(used),
            gpu_models=tuple(sorted(models)),
        )
    value = empirical_quantile(residuals, methodology.target_quantile)
    if value is None:
        return ToleranceLookup(
            status=CharacterizationStatus.UNCHARACTERIZED,
            reason="quantile undefined",
            workload=workload,
            dtype=dtype,
            shape=shape,
            n_samples=n_total,
            min_samples=min_n,
            target_quantile=target,
            p_quantile_residual_hex=None,
            p_quantile_residual_decimal=None,
            sample_max_residual_hex=max_hex,
            source_files=tuple(used),
            gpu_models=tuple(sorted(models)),
        )
    return ToleranceLookup(
        status=CharacterizationStatus.CHARACTERIZED,
        reason="empirical quantile of measured ABFT normalized residuals",
        workload=workload,
        dtype=dtype,
        shape=shape,
        n_samples=n_total,
        min_samples=min_n,
        target_quantile=target,
        p_quantile_residual_hex=float(value).hex(),
        p_quantile_residual_decimal=format(value, ".17g"),
        sample_max_residual_hex=max_hex,
        source_files=tuple(used),
        gpu_models=tuple(sorted(models)),
    )


def assay_verdict(lookup: ToleranceLookup) -> str:
    """Unmeasured or under-sampled configs are INCONCLUSIVE. No PASS/FAIL here."""
    if lookup.status is CharacterizationStatus.UNCHARACTERIZED:
        return "INCONCLUSIVE"
    return "CHARACTERIZED"
