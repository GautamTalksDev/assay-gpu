"""IEEE-754 and int8 bit-field layouts for the fault injector.

Bit 0 is the least-significant bit of the in-register pattern (IEEE logical
layout), not a memory-endian byte index.

int8 has no exponent or mantissa. Those class names still select bits, by
significance, so the API is the same for every dtype. See INT8_NOTE.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch


class BitClass(StrEnum):
    SIGN = "SIGN"
    EXPONENT_HIGH = "EXPONENT_HIGH"
    EXPONENT_LOW = "EXPONENT_LOW"
    MANTISSA_HIGH = "MANTISSA_HIGH"
    MANTISSA_LOW = "MANTISSA_LOW"


INT8_NOTE = (
    "int8 is two's-complement. SIGN is bit 7. EXPONENT_* and MANTISSA_* "
    "are positional analogues on bits 6-0, not IEEE fields."
)


@dataclass(frozen=True, slots=True)
class DTypeLayout:
    name: str
    torch_dtype: torch.dtype
    width: int
    sign: tuple[int, ...]
    exponent_high: tuple[int, ...]
    exponent_low: tuple[int, ...]
    mantissa_high: tuple[int, ...]
    mantissa_low: tuple[int, ...]

    def bits_for(self, bit_class: BitClass) -> tuple[int, ...]:
        mapping = {
            BitClass.SIGN: self.sign,
            BitClass.EXPONENT_HIGH: self.exponent_high,
            BitClass.EXPONENT_LOW: self.exponent_low,
            BitClass.MANTISSA_HIGH: self.mantissa_high,
            BitClass.MANTISSA_LOW: self.mantissa_low,
        }
        return mapping[bit_class]

    def all_bits(self) -> tuple[int, ...]:
        return (
            self.sign
            + self.exponent_high
            + self.exponent_low
            + self.mantissa_high
            + self.mantissa_low
        )


def _validate_layout(layout: DTypeLayout) -> DTypeLayout:
    bits = layout.all_bits()
    if len(bits) != layout.width or len(set(bits)) != layout.width:
        msg = f"layout {layout.name} does not partition 0..{layout.width - 1}"
        raise ValueError(msg)
    if tuple(sorted(bits)) != tuple(range(layout.width)):
        msg = f"layout {layout.name} bit ids are not 0..{layout.width - 1}"
        raise ValueError(msg)
    return layout


# Split each IEEE field at ceil(width/2) on the high (more significant) side.
# fp32: sign 31 | exp 30-23 | mantissa 22-0
# fp16: sign 15 | exp 14-10 | mantissa 9-0
# bf16: sign 15 | exp 14-7  | mantissa 6-0
LAYOUT_FP32 = _validate_layout(
    DTypeLayout(
        name="float32",
        torch_dtype=torch.float32,
        width=32,
        sign=(31,),
        exponent_high=(27, 28, 29, 30),
        exponent_low=(23, 24, 25, 26),
        mantissa_high=(11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22),
        mantissa_low=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    )
)
LAYOUT_FP16 = _validate_layout(
    DTypeLayout(
        name="float16",
        torch_dtype=torch.float16,
        width=16,
        sign=(15,),
        exponent_high=(12, 13, 14),
        exponent_low=(10, 11),
        mantissa_high=(5, 6, 7, 8, 9),
        mantissa_low=(0, 1, 2, 3, 4),
    )
)
LAYOUT_BF16 = _validate_layout(
    DTypeLayout(
        name="bfloat16",
        torch_dtype=torch.bfloat16,
        width=16,
        sign=(15,),
        exponent_high=(11, 12, 13, 14),
        exponent_low=(7, 8, 9, 10),
        mantissa_high=(3, 4, 5, 6),
        mantissa_low=(0, 1, 2),
    )
)
LAYOUT_INT8 = _validate_layout(
    DTypeLayout(
        name="int8",
        torch_dtype=torch.int8,
        width=8,
        sign=(7,),
        exponent_high=(5, 6),
        exponent_low=(3, 4),
        mantissa_high=(1, 2),
        mantissa_low=(0,),
    )
)

_BY_DTYPE: dict[torch.dtype, DTypeLayout] = {
    torch.float32: LAYOUT_FP32,
    torch.float16: LAYOUT_FP16,
    torch.bfloat16: LAYOUT_BF16,
    torch.int8: LAYOUT_INT8,
}


def layout_for(dtype: torch.dtype) -> DTypeLayout:
    layout = _BY_DTYPE.get(dtype)
    if layout is None:
        msg = f"unsupported injector dtype: {dtype}"
        raise ValueError(msg)
    return layout
