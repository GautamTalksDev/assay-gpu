"""Write a signed attestation-v1 document. Never uploaded."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from assay.report.build import build_body, utc_now, wrap_document
from assay.report.canonical import canonical_dumps
from assay.report.keys import KeyPair, signature_envelope
from assay.run.types import AssayResult


def sign_document(body: dict[str, Any], pair: KeyPair) -> dict[str, Any]:
    independence = body.get("independence")
    if not isinstance(independence, dict):
        msg = "body.independence must be an object"
        raise ValueError(msg)
    mode = str(independence["mode"])
    signature = signature_envelope(body, pair, mode=mode)
    return wrap_document(body, signature)


def write_attestation(  # noqa: PLR0913
    path: Path,
    result: AssayResult,
    *,
    signing_key: KeyPair,
    reference_dir: Path,
    noisefloor_dir: Path,
    issued_at_utc: str | None = None,
) -> dict[str, Any]:
    body = build_body(
        result,
        reference_dir=reference_dir,
        noisefloor_dir=noisefloor_dir,
        issued_at_utc=issued_at_utc or utc_now(),
    )
    document = sign_document(body, signing_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document


def signed_canonical_body(document: dict[str, Any]) -> str:
    return canonical_dumps(document["body"])
