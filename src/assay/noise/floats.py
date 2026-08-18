"""IEEE float encoding for JSON. Hex is the exact value; decimal is for reading."""

from __future__ import annotations

import math
from typing import Any


def encode_f64(value: float) -> dict[str, str]:
    if math.isnan(value):
        return {"decimal": "nan", "hex": "nan"}
    if math.isinf(value):
        sign = "-" if value < 0 else ""
        return {"decimal": f"{sign}inf", "hex": f"{sign}inf"}
    return {"decimal": format(value, ".17g"), "hex": float(value).hex()}


def decode_f64(payload: dict[str, Any]) -> float:
    hex_text = str(payload["hex"])
    if hex_text == "nan":
        return math.nan
    if hex_text == "inf":
        return math.inf
    if hex_text == "-inf":
        return -math.inf
    return float.fromhex(hex_text)
