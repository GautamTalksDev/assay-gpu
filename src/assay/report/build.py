"""Build an attestation-v1 body from a run. Timestamp is metadata, not a verdict."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from assay.noise.floats import encode_f64
from assay.noise.methodology import load_methodology
from assay.reference.spec import WORKLOAD_SUITE_VERSION
from assay.report.canonical import canonical_dumps, to_jsonable
from assay.report.constants import (
    ASSAY_SIGNED_NOTE,
    MODE_ASSAY_SIGNED,
    MODE_SELF_SIGNED,
    SELF_SIGNED_DISCLAIMER,
    SPEC_ID,
)
from assay.run.guarantee import NETWORK_GUARANTEE
from assay.run.render import case_payload, verdict_reason
from assay.run.types import AssayResult, exit_code_for


def tool_version() -> str:
    try:
        return version("assay-gpu")
    except PackageNotFoundError:
        return "0.0.0"


def format_issued_at_utc(moment: datetime) -> str:
    utc = moment.astimezone(UTC)
    return utc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{utc.microsecond:06d}Z"


def reference_catalog_sha256(reference_dir: Path) -> str:
    """SHA-256 of canonical JSON of sorted {file, key, sha256} rows.

    Array hashes themselves are defined in docs/WORKLOADS.md and
    assay.reference.hashing.sha256_array. This catalog digest is those
    hashes, not a re-hash of .npz ZIP bytes.
    """
    manifest = reference_dir / "manifest.json"
    rows: list[dict[str, str]] = []
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        rows.extend(
            {
                "file": str(artifact["file"]),
                "key": str(artifact["key"]),
                "sha256": str(artifact["sha256"]),
            }
            for artifact in payload.get("artifacts", [])
        )
    rows.sort(key=lambda row: (row["file"], row["key"]))
    digest = hashlib.sha256()
    digest.update(canonical_dumps(rows).encode("ascii"))
    return digest.hexdigest()


def compute_run_id(body_without_run_id: dict[str, Any]) -> str:
    placeholder = dict(body_without_run_id)
    placeholder["run_id"] = ""
    digest = hashlib.sha256()
    digest.update(canonical_dumps(placeholder).encode("ascii"))
    return digest.hexdigest()


def _independence(mode: str) -> dict[str, str]:
    if mode == MODE_SELF_SIGNED:
        return {"mode": MODE_SELF_SIGNED, "disclaimer": SELF_SIGNED_DISCLAIMER}
    if mode == MODE_ASSAY_SIGNED:
        return {"mode": MODE_ASSAY_SIGNED, "disclaimer": ASSAY_SIGNED_NOTE}
    msg = f"unknown independence mode: {mode}"
    raise ValueError(msg)


def _case_row(payload: dict[str, Any]) -> dict[str, Any]:
    row = dict(payload)
    wall = float(row.pop("wall_time_s"))
    row["wall_time_s_hex"] = encode_f64(wall)["hex"]
    return row


def build_body(
    result: AssayResult,
    *,
    reference_dir: Path,
    noisefloor_dir: Path,
    issued_at_utc: str,
    mode: str = MODE_SELF_SIGNED,
) -> dict[str, Any]:
    if mode == MODE_ASSAY_SIGNED:
        msg = "v1 cannot issue assay-signed reports (no issuance service)"
        raise ValueError(msg)
    methodology = load_methodology(noisefloor_dir)
    body: dict[str, Any] = {
        "budget": to_jsonable(asdict(result.budget)),
        "cases": [_case_row(case_payload(case)) for case in result.cases],
        "elapsed_s_hex": encode_f64(result.elapsed_s)["hex"],
        "exit_code": int(exit_code_for(result.status)),
        "gpu_model": result.gpu_model,
        "independence": _independence(mode),
        "issued_at_utc": issued_at_utc,
        "network_guarantee": NETWORK_GUARANTEE,
        "network_requests": 0,
        "noisefloor": {
            "reason": result.noisefloor_reason,
            "status": result.noisefloor_status,
        },
        "noisefloor_spec_version": methodology.spec_id,
        "probe": to_jsonable(asdict(result.probe)),
        "reference_catalog_sha256": reference_catalog_sha256(reference_dir),
        "run_id": "",
        "status": result.status.value,
        "tool_version": tool_version(),
        "verdict_reason": verdict_reason(result),
        "workload_suite_version": WORKLOAD_SUITE_VERSION,
    }
    body["run_id"] = compute_run_id(body)
    return body


def wrap_document(body: dict[str, Any], signature: dict[str, str]) -> dict[str, Any]:
    return {"body": body, "signature": signature, "spec": SPEC_ID}


def utc_now() -> str:
    return format_issued_at_utc(datetime.now(UTC))
