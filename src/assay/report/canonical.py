"""Canonical JSON for attestation-v1. Independent verifiers must match this."""

from __future__ import annotations

import json
from typing import Any


def canonical_dumps(value: object) -> str:
    """UTF-8 JSON, sorted keys, no extra whitespace, ASCII-only.

    This is the byte string that Ed25519 signs. Do not pretty-print it.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def to_jsonable(value: object) -> Any:
    """Round-trip through canonical JSON so tuples become lists."""
    return json.loads(canonical_dumps(value))
