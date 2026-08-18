"""CPU tests for the bit-flip injector. Hand-computed IEEE patterns."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from assay.inject import (
    LAYOUT_BF16,
    LAYOUT_FP16,
    LAYOUT_FP32,
    LAYOUT_INT8,
    BitClass,
    flip,
    flip_random,
    layout_for,
)

pytestmark = pytest.mark.cpu

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "assay"
SEED = 20260818


def _u32(value: int) -> int:
    return int(np.uint32(value))


def _u16(value: int) -> int:
    return int(np.uint16(value))


def _pattern(tensor: torch.Tensor) -> int:
    item = tensor.detach().contiguous().cpu().reshape(-1)[0]
    if tensor.dtype == torch.float32:
        return _u32(item.view(torch.int32).item() & 0xFFFFFFFF)
    if tensor.dtype in {torch.float16, torch.bfloat16}:
        return _u16(item.view(torch.int16).item() & 0xFFFF)
    if tensor.dtype == torch.int8:
        return int(item.view(torch.uint8).item())
    msg = f"unhandled dtype {tensor.dtype}"
    raise AssertionError(msg)


def test_layouts_partition_every_bit() -> None:
    for layout in (LAYOUT_FP32, LAYOUT_FP16, LAYOUT_BF16, LAYOUT_INT8):
        bits = layout.all_bits()
        assert len(bits) == layout.width
        assert sorted(bits) == list(range(layout.width))
        seen: set[int] = set()
        for bit_class in BitClass:
            group = layout.bits_for(bit_class)
            assert group
            overlap = seen.intersection(group)
            assert overlap == set()
            seen.update(group)
        assert seen == set(range(layout.width))


def test_fp32_one_hand_computed_flips() -> None:
    one = torch.tensor([1.0], dtype=torch.float32)
    assert _pattern(one) == 0x3F800000
    sign = flip(one, 31, 0, SEED)
    assert _pattern(sign) == 0xBF800000
    assert sign.item() == -1.0
    exp_low = flip(one, 23, 0, SEED)
    assert _pattern(exp_low) == 0x3F000000
    assert exp_low.item() == 0.5
    exp_high = flip(one, 30, 0, SEED)
    assert _pattern(exp_high) == 0x7F800000
    assert torch.isinf(exp_high).all()
    mant_high = flip(one, 22, 0, SEED)
    assert _pattern(mant_high) == 0x3FC00000
    assert mant_high.item() == 1.5
    mant_low = flip(one, 0, 0, SEED)
    assert _pattern(mant_low) == 0x3F800001
    assert _pattern(one) == 0x3F800000


def test_fp16_one_hand_computed_flips() -> None:
    one = torch.tensor([1.0], dtype=torch.float16)
    assert _pattern(one) == 0x3C00
    sign = flip(one, 15, 0, SEED)
    assert _pattern(sign) == 0xBC00
    assert sign.item() == -1.0
    exp_low = flip(one, 10, 0, SEED)
    assert _pattern(exp_low) == 0x3800
    assert exp_low.item() == 0.5
    exp_high = flip(one, 14, 0, SEED)
    assert _pattern(exp_high) == 0x7C00
    assert torch.isinf(exp_high.to(torch.float32)).all()
    mant_high = flip(one, 9, 0, SEED)
    assert _pattern(mant_high) == 0x3E00
    assert mant_high.item() == 1.5
    mant_low = flip(one, 0, 0, SEED)
    assert _pattern(mant_low) == 0x3C01


def test_bf16_one_hand_computed_flips() -> None:
    one = torch.tensor([1.0], dtype=torch.bfloat16)
    assert _pattern(one) == 0x3F80
    sign = flip(one, 15, 0, SEED)
    assert _pattern(sign) == 0xBF80
    assert sign.item() == -1.0
    exp_low = flip(one, 7, 0, SEED)
    assert _pattern(exp_low) == 0x3F00
    assert exp_low.item() == 0.5
    exp_high = flip(one, 14, 0, SEED)
    assert _pattern(exp_high) == 0x7F80
    assert torch.isinf(exp_high.to(torch.float32)).all()
    mant_high = flip(one, 6, 0, SEED)
    assert _pattern(mant_high) == 0x3FC0
    assert mant_high.item() == 1.5
    mant_low = flip(one, 0, 0, SEED)
    assert _pattern(mant_low) == 0x3F81


def test_int8_one_hand_computed_flips() -> None:
    one = torch.tensor([1], dtype=torch.int8)
    assert _pattern(one) == 0x01
    sign = flip(one, 7, 0, SEED)
    assert _pattern(sign) == 0x81
    assert int(sign.item()) == -127
    exp_high = flip(one, 6, 0, SEED)
    assert _pattern(exp_high) == 0x41
    assert int(exp_high.item()) == 65
    exp_low = flip(one, 3, 0, SEED)
    assert _pattern(exp_low) == 0x09
    assert int(exp_low.item()) == 9
    mant_high = flip(one, 1, 0, SEED)
    assert _pattern(mant_high) == 0x03
    assert int(mant_high.item()) == 3
    mant_low = flip(one, 0, 0, SEED)
    assert _pattern(mant_low) == 0x00
    assert int(mant_low.item()) == 0


def test_flip_seed_does_not_change_specified_bit() -> None:
    one = torch.tensor([1.0], dtype=torch.float32)
    first = flip(one, 0, 0, 1)
    second = flip(one, 0, 0, 99)
    assert torch.equal(first, second)


def test_flip_random_deterministic_and_in_class() -> None:
    matrix = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    first, loc_a = flip_random(matrix, 3, BitClass.MANTISSA_LOW, SEED)
    second, loc_b = flip_random(matrix, 3, BitClass.MANTISSA_LOW, SEED)
    assert torch.equal(first, second)
    assert loc_a == loc_b
    allowed = set(layout_for(torch.float32).bits_for(BitClass.MANTISSA_LOW))
    assert len(loc_a) == 3
    assert len({(item.element_index, item.bit_index) for item in loc_a}) == 3
    for item in loc_a:
        assert item.bit_index in allowed
        assert 0 <= item.element_index < matrix.numel()
    other, loc_c = flip_random(matrix, 3, BitClass.SIGN, SEED)
    assert {item.bit_index for item in loc_c} <= {31}
    assert not torch.equal(first, other)


def test_src_outside_inject_does_not_mention_assay_inject() -> None:
    hits: list[str] = []
    for path in SRC.rglob("*.py"):
        if "inject" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "assay.inject" in text:
            hits.append(str(path.relative_to(REPO)))
    assert hits == []


def test_run_path_does_not_load_inject_module() -> None:
    code = (
        "import sys\n"
        "import assay.cli\n"
        "import assay.workload.suite\n"
        "import assay.run.session\n"
        "import assay.watch.session\n"
        "names = [name for name in sys.modules if name.startswith('assay.inject')]\n"
        "assert names == [], names\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
