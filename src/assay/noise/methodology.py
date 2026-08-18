"""Load noisefloor-v1 methodology from data/noisefloor. Not a residual magnitude."""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Methodology:
    spec_id: str
    target_quantile: Fraction
    abft_note: str
    source_path: Path

    @property
    def min_samples(self) -> int:
        """ceil(1/(1-p)); empirical p-quantile is an observed order statistic."""
        missing = 1 - self.target_quantile
        if missing <= 0:
            msg = "target quantile must be in (0, 1)"
            raise ValueError(msg)
        return int((missing.denominator + missing.numerator - 1) // missing.numerator)


def methodology_path(noisefloor_dir: Path) -> Path:
    return noisefloor_dir / "methodology-v1.json"


def load_methodology(noisefloor_dir: Path) -> Methodology:
    path = methodology_path(noisefloor_dir)
    if not path.is_file():
        msg = f"missing methodology file: {path}"
        raise FileNotFoundError(msg)
    payload = json.loads(path.read_text(encoding="utf-8"))
    spec_id = str(payload["spec_id"])
    frac_text = str(payload["target_quantile"])
    quantile = Fraction(frac_text)
    return Methodology(
        spec_id=spec_id,
        target_quantile=quantile,
        abft_note=str(payload["abft_note"]),
        source_path=path,
    )
