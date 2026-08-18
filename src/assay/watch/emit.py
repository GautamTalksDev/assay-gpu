"""Serialize a watch log. Not attestation-v1 (that document is signed)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_watch_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
