"""Fail if src/ grows a hardcoded detection tolerance."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu

ROOT = Path(__file__).resolve().parents[1] / "src" / "assay"
ALLOWED = {ROOT / "ieee.py"}
SUSPECT = re.compile(
    r"\b(atol|rtol)\b"
    r"|tolerance\s*=\s*[-+]?\d"
    r"|[<>]=?\s*\d+\.\d*[eE][+-]?\d+"
    r"|\d+\.\d*[eE][+-]?\d+\s*[<>]"
)


def test_no_hardcoded_numerical_tolerances_in_src() -> None:
    hits: list[str] = []
    for path in ROOT.rglob("*.py"):
        if path in ALLOWED:
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.split("#", 1)[0]
            if SUSPECT.search(stripped):
                rel = path.relative_to(ROOT.parent.parent)
                hits.append(f"{rel}:{lineno}:{line.strip()}")
    assert hits == []
