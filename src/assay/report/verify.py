"""Offline verification of attestation-v1. Never opens a network socket."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from assay.abft.check import CheckStatus
from assay.report.build import compute_run_id, reference_catalog_sha256
from assay.report.canonical import canonical_dumps
from assay.report.constants import (
    ASSAY_SIGNED_NOTE,
    MODE_ASSAY_SIGNED,
    MODE_SELF_SIGNED,
    SELF_SIGNED_DISCLAIMER,
    SIGNATURE_ALG,
    SPEC_ID,
)
from assay.report.keys import load_public_key_hex, verify_bytes
from assay.run.types import CaseRecord, ExitCode, exit_code_for, overall_status

_UTC = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX128 = re.compile(r"^[0-9a-f]{128}$")


class VerifyOutcome(StrEnum):
    VALID = "VALID"
    INVALID = "INVALID"
    OPERATIONAL = "OPERATIONAL"


@dataclass(frozen=True, slots=True)
class VerifyResult:
    outcome: VerifyOutcome
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    mode: str | None
    status: str | None


def _case_record(row: dict[str, Any]) -> CaseRecord:
    raw_dtype = row.get("dtype")
    dtype_name = str(raw_dtype) if isinstance(raw_dtype, str) else None
    raw_n = row.get("n_samples")
    n_samples = int(raw_n) if isinstance(raw_n, int) else None
    raw_min = row.get("min_samples")
    min_samples = int(raw_min) if isinstance(raw_min, int) else None
    return CaseRecord(
        workload=str(row["workload"]),
        case=str(row["case"]),
        shape=tuple(int(dim) for dim in row["shape"]),
        dtype_name=dtype_name,
        status=CheckStatus(str(row["status"])),
        reason=str(row.get("reason", "")),
        residual_hex=_opt_str(row.get("residual_hex")),
        residual_decimal=_opt_str(row.get("residual_decimal")),
        threshold_hex=_opt_str(row.get("threshold_hex")),
        threshold_decimal=_opt_str(row.get("threshold_decimal")),
        sample_max_hex=_opt_str(row.get("sample_max_hex")),
        n_samples=n_samples,
        min_samples=min_samples,
        noisefloor_spec_version=_opt_str(row.get("noisefloor_spec_version")),
        golden_max_abs_error_hex=_opt_str(row.get("golden_max_abs_error_hex")),
        wall_time_s=0.0,
        skipped=bool(row["skipped"]),
        skip_reason=_opt_str(row.get("skip_reason")),
        counts_toward_verdict=bool(row["counts_toward_verdict"]),
    )


def _opt_str(value: object) -> str | None:
    return str(value) if isinstance(value, str) else None


def _require_str(mapping: dict[str, Any], key: str, failures: list[str]) -> str | None:
    if key not in mapping:
        failures.append(f"missing field {key}")
        return None
    value = mapping[key]
    if not isinstance(value, str):
        failures.append(f"{key} must be a string")
        return None
    return value


def verify_document(  # noqa: PLR0912, PLR0915
    document: dict[str, Any],
    *,
    pinned_public_key_hex: str | None = None,
    expected_catalog_sha256: str | None = None,
) -> VerifyResult:
    failures: list[str] = []
    warnings: list[str] = []
    if document.get("spec") != SPEC_ID:
        failures.append(f"spec must be {SPEC_ID}")
    body = document.get("body")
    signature = document.get("signature")
    if not isinstance(body, dict):
        failures.append("body must be an object")
        return VerifyResult(VerifyOutcome.INVALID, tuple(failures), (), None, None)
    if not isinstance(signature, dict):
        failures.append("signature must be an object")
        return VerifyResult(VerifyOutcome.INVALID, tuple(failures), (), None, None)

    mode = signature.get("mode")
    status = body.get("status") if isinstance(body.get("status"), str) else None
    independence = body.get("independence")
    if not isinstance(independence, dict):
        failures.append("body.independence must be an object")
        independence = {}

    if signature.get("alg") != SIGNATURE_ALG:
        failures.append(f"signature.alg must be {SIGNATURE_ALG}")
    if mode not in {MODE_SELF_SIGNED, MODE_ASSAY_SIGNED}:
        failures.append("signature.mode must be self-signed or assay-signed")
    if independence.get("mode") != mode:
        failures.append("signature.mode must equal body.independence.mode")

    if mode == MODE_SELF_SIGNED:
        if independence.get("disclaimer") != SELF_SIGNED_DISCLAIMER:
            failures.append("self-signed disclaimer text does not match the spec")
        warnings.append(SELF_SIGNED_DISCLAIMER)
    elif mode == MODE_ASSAY_SIGNED:
        if independence.get("disclaimer") != ASSAY_SIGNED_NOTE:
            failures.append("assay-signed disclaimer text does not match the spec")
        warnings.append(ASSAY_SIGNED_NOTE)
        if pinned_public_key_hex is None:
            warnings.append(
                "assay-signed public key is not pinned; pass --pubkey to "
                "bind this report to a known issuer"
            )

    pub = signature.get("public_key_hex")
    sig = signature.get("signature_hex")
    if not isinstance(pub, str) or not _HEX64.match(pub):
        failures.append("signature.public_key_hex must be 64 lowercase hex chars")
        pub = None
    if not isinstance(sig, str) or not _HEX128.match(sig):
        failures.append("signature.signature_hex must be 128 lowercase hex chars")
        sig = None
    if pub is not None and sig is not None:
        message = canonical_dumps(body).encode("ascii")
        if not verify_bytes(message, sig, pub):
            failures.append("Ed25519 signature does not match the canonical body")
    if (
        pinned_public_key_hex is not None
        and pub is not None
        and pinned_public_key_hex.lower() != pub
    ):
        failures.append("embedded public key does not match --pubkey")

    issued = body.get("issued_at_utc")
    if not isinstance(issued, str) or not _UTC.match(issued):
        failures.append("issued_at_utc must be YYYY-MM-DDTHH:MM:SS.ffffffZ")

    run_id = body.get("run_id")
    if not isinstance(run_id, str) or not _HEX64.match(run_id):
        failures.append("run_id must be 64 lowercase hex chars")
    elif compute_run_id(body) != run_id:
        failures.append("run_id does not match SHA-256 of the canonical body")

    for key in (
        "tool_version",
        "noisefloor_spec_version",
        "workload_suite_version",
        "reference_catalog_sha256",
        "verdict_reason",
    ):
        _require_str(body, key, failures)

    catalog = body.get("reference_catalog_sha256")
    if isinstance(catalog, str) and not _HEX64.match(catalog):
        failures.append("reference_catalog_sha256 must be 64 lowercase hex chars")
    if (
        expected_catalog_sha256 is not None
        and isinstance(catalog, str)
        and catalog != expected_catalog_sha256
    ):
        failures.append("reference_catalog_sha256 does not match --reference-dir")

    if body.get("network_requests") != 0:
        failures.append("network_requests must be 0")

    cases = body.get("cases")
    if not isinstance(cases, list):
        failures.append("body.cases must be an array")
    else:
        try:
            records = tuple(_case_record(row) for row in cases if isinstance(row, dict))
            if len(records) != len(cases):
                failures.append("every case must be an object")
            else:
                computed = overall_status(records)
                if status != computed.value:
                    failures.append(
                        f"status {status!r} does not match cases ({computed.value})"
                    )
                if body.get("exit_code") != int(exit_code_for(computed)):
                    failures.append("exit_code does not match status")
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"cases are not internally consistent: {exc}")

    outcome = VerifyOutcome.INVALID if failures else VerifyOutcome.VALID
    return VerifyResult(
        outcome=outcome,
        reasons=tuple(failures),
        warnings=tuple(warnings),
        mode=mode if isinstance(mode, str) else None,
        status=status if isinstance(status, str) else None,
    )


def load_report(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_path(
    path: Path,
    *,
    pubkey_path: Path | None = None,
    reference_dir: Path | None = None,
) -> VerifyResult:
    try:
        document = load_report(path)
    except (OSError, json.JSONDecodeError) as exc:
        return VerifyResult(
            VerifyOutcome.OPERATIONAL,
            (f"cannot read report: {exc}",),
            (),
            None,
            None,
        )
    if not isinstance(document, dict):
        return VerifyResult(
            VerifyOutcome.OPERATIONAL,
            ("report JSON must be an object",),
            (),
            None,
            None,
        )
    pinned = None
    expected_hash = None
    try:
        if pubkey_path is not None:
            pinned = load_public_key_hex(pubkey_path)
        if reference_dir is not None:
            expected_hash = reference_catalog_sha256(reference_dir)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return VerifyResult(
            VerifyOutcome.OPERATIONAL,
            (str(exc),),
            (),
            None,
            None,
        )
    return verify_document(
        document,
        pinned_public_key_hex=pinned,
        expected_catalog_sha256=expected_hash,
    )


def render_verify(result: VerifyResult) -> str:
    lines = [f"VERIFY  {result.outcome.value}"]
    if result.mode is not None:
        lines.append(f"  mode    {result.mode}")
    if result.status is not None:
        lines.append(f"  status  {result.status}")
    lines.extend(f"  error   {reason}" for reason in result.reasons)
    lines.extend(f"WARNING  {warning}" for warning in result.warnings)
    return "\n".join(lines) + "\n"


def exit_code_for_verify(result: VerifyResult) -> int:
    if result.outcome is VerifyOutcome.VALID:
        return 0
    if result.outcome is VerifyOutcome.INVALID:
        return 1
    return int(ExitCode.OPERATIONAL)
