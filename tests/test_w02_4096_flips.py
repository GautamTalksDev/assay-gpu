"""W02 bf16 4096³ residual-v2 flip residuals vs the T4 pilot floor.

Does not evaluate KT-1 and does not write a threshold.
Imports assay.inject on purpose; the run path must not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from assay.abft.gemm import normalize_by_scale
from assay.abft.reduce import (
    CheckBackend,
    absolute_factor_scale,
    checksum_abs_residual,
    ones_matvec,
)
from assay.inject import LAYOUT_BF16, BitClass, flip_random
from assay.reference.spec import BASE_SEED
from assay.workload.context import gemm_flags
from assay.workload.gemm import gemm_numpy_pair

REPO = Path(__file__).resolve().parents[1]
_SHAPE = (4096, 4096, 4096)
_CASE_INDEX = 3
_WORKLOAD_ID = 2
_FLIP_COUNTS = (1, 2, 4)
_BACKEND = CheckBackend.PYTORCH
# Same prime as sample_factor_seed in assay.reference.spec.
_SAMPLE_STRIDE = 1_000_037
FLIP_N = 200
# Fraction of samples with residual / clean_max above this. Not a cutoff.
CLEAN_MAX_RATIO = 1.0

# Tesla T4 residual-v2 pilot, n=2000. Not a detection threshold.
PILOT_N = 2000
PILOT_MEDIAN = 1.07e-8
PILOT_P99_9 = 5.07e-8
PILOT_MAX = 5.57e-8


def _cell_seed(bit_class: BitClass, n_flips: int, sample_index: int) -> int:
    class_id = list(BitClass).index(bit_class) + 1
    return (
        BASE_SEED
        + _WORKLOAD_ID * 1_000_003
        + class_id * 1_009
        + n_flips
        + sample_index * _SAMPLE_STRIDE
    )


@dataclass(frozen=True, slots=True)
class FlipCellSummary:
    bit_class: str
    n_flips: int
    n: int
    min_ratio: float
    median_ratio: float
    max_ratio: float
    n_exceeding_one: int
    fraction_exceeding_one: float
    n_nonfinite: int


def _median(ordered: list[float]) -> float:
    count = len(ordered)
    if count < 1:
        msg = "median requires at least one sample"
        raise ValueError(msg)
    if count % 2 == 1:
        return ordered[count // 2]
    return 0.5 * (ordered[count // 2 - 1] + ordered[count // 2])


def summarize_ratio_lists(
    collected: dict[tuple[str, int], list[float]],
) -> list[FlipCellSummary]:
    """min / median / max of residual/clean_max, and fraction > 1.

    Non-finite ratios count as exceeding 1. Not a detection threshold.
    """
    rows: list[FlipCellSummary] = []
    for bit_class in BitClass:
        for n_flips in _FLIP_COUNTS:
            ratios = collected[(bit_class.value, n_flips)]
            if not ratios:
                msg = f"no ratios for {bit_class.value} n_flips={n_flips}"
                raise ValueError(msg)
            ordered = sorted(ratios)
            n_nonfinite = sum(1 for value in ratios if not math.isfinite(value))
            n_over = sum(
                1
                for value in ratios
                if (not math.isfinite(value)) or value > CLEAN_MAX_RATIO
            )
            rows.append(
                FlipCellSummary(
                    bit_class=bit_class.value,
                    n_flips=n_flips,
                    n=len(ratios),
                    min_ratio=ordered[0],
                    median_ratio=_median(ordered),
                    max_ratio=ordered[-1],
                    n_exceeding_one=n_over,
                    fraction_exceeding_one=n_over / len(ratios),
                    n_nonfinite=n_nonfinite,
                )
            )
    return rows


def _factors_at(
    device: torch.device, sample_index: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, inner, cols = _SHAPE
    left_np, right_np = gemm_numpy_pair(
        rows,
        inner,
        cols,
        case_index=_CASE_INDEX,
        sample_index=sample_index,
        workload_id=_WORKLOAD_ID,
    )
    left = torch.from_numpy(np.ascontiguousarray(left_np, dtype=np.float32)).to(
        device=device, dtype=torch.bfloat16
    )
    right = torch.from_numpy(np.ascontiguousarray(right_np, dtype=np.float32)).to(
        device=device, dtype=torch.bfloat16
    )
    product = torch.matmul(left, right)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return left, right, product


def measure_w02_4096_flip_residuals(
    device: torch.device, *, n_samples: int = FLIP_N
) -> list[FlipCellSummary]:
    """n_samples independent (A, B), then every (bit_class, n_flips) on that C.

    One GEMM per sample_index. a_be and scale_v2 are computed once; each cell
    only re-reduces C@e after flipping C. Flips are C-after-GEMM, not
    intermediate SDC.
    """
    if n_samples < 1:
        msg = "n_samples must be >= 1"
        raise ValueError(msg)
    collected: dict[tuple[str, int], list[float]] = {
        (bit_class.value, n_flips): []
        for bit_class in BitClass
        for n_flips in _FLIP_COUNTS
    }
    with gemm_flags(fp16_reduced=False, bf16_reduced=False):
        for sample_index in range(n_samples):
            print(  # noqa: T201
                f"w02-4096-flips sample {sample_index + 1}/{n_samples}",
                flush=True,
            )
            left, right, product = _factors_at(device, sample_index)
            b_e = ones_matvec(right, _BACKEND)
            a_be = left @ b_e
            scale = np.float64(absolute_factor_scale(left, right))
            for bit_class in BitClass:
                for n_flips in _FLIP_COUNTS:
                    flipped, _locs = flip_random(
                        product,
                        n_flips,
                        bit_class,
                        _cell_seed(bit_class, n_flips, sample_index),
                    )
                    c_e = ones_matvec(flipped, _BACKEND)
                    abs_r = np.float64(checksum_abs_residual(c_e, a_be))
                    residual = float(normalize_by_scale(abs_r, scale))
                    if not math.isfinite(residual):
                        ratio = math.inf
                    else:
                        ratio = residual / PILOT_MAX
                    collected[(bit_class.value, n_flips)].append(float(ratio))
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return summarize_ratio_lists(collected)


def _fmt_ratio(value: float) -> str:
    if not math.isfinite(value):
        return "inf"
    return format(value, ".6g")


def render_bf16_bit_class_table() -> str:
    """Exact injector bits. bf16: sign 15 | exponent 14-7 | mantissa 6-0."""
    lines = [
        "bfloat16 bit_class → bit indices (bit 0 is IEEE LSB of the 16-bit",
        "pattern). Exponent field is bits 14-7 (8 bits). Mantissa is 6-0",
        "(7 bits). Split is ceil(field_width/2) on the high side",
        "(`LAYOUT_BF16` in `src/assay/inject/bits.py`).",
        "",
        "| bit_class | bits | field role |",
        "| --- | --- | --- |",
        "| SIGN | 15 | sign |",
        "| EXPONENT_HIGH | 11, 12, 13, 14 | exponent bits [4:7], includes MSB |",
        "| EXPONENT_LOW | 7, 8, 9, 10 | exponent bits [0:3], includes LSB |",
        "| MANTISSA_HIGH | 3, 4, 5, 6 | mantissa bits [3:6] |",
        "| MANTISSA_LOW | 0, 1, 2 | mantissa bits [0:2] |",
        "",
        "`flip_random` draws uniformly among the bits in the class.",
        "**EXPONENT_LOW includes the exponent LSB (bit 7).** Flipping only",
        "that bit scales one element by 2x or 1/2. That is a small change to",
        "a 4096-term ones-vector row sum: a structural property of this",
        "checksum, not a detector bug. The same class also contains bits",
        "8, 9, and 10 (element scale 4x, 16x, 256x), so a 1-flip cell is a",
        "mixture of those four magnitudes.",
        "",
        "SIGN is bit 15 only: `C_ij → -C_ij`, which adds `-2 C_ij` to the",
        "row sum. The ones-vector can cancel that the same way it can",
        "cancel a mantissa ULP.",
        "",
        "Flips are injected into **C after the GEMM**. That is not an",
        "intermediate accumulator SDC.",
        "",
    ]
    assert LAYOUT_BF16.exponent_low == (7, 8, 9, 10)
    assert LAYOUT_BF16.exponent_high == (11, 12, 13, 14)
    return "\n".join(lines)


def render_w02_4096_section(rows: list[FlipCellSummary] | None) -> str:
    lines = [
        "## W02 bfloat16 4096³ flip residuals vs T4 residual-v2 pilot floor",
        "",
        "Not a KT-1 evaluation. No threshold is set. residual-v1 flip",
        "residuals are void. A single `sample_index=0` draw per cell is",
        "not evidence; cells below are n = 200 independent `(A, B)`.",
        "",
        "Clean floor: Tesla T4 residual-v2 pilot, n = 2000, W02 bf16 4096³.",
        "",
        "| statistic | value |",
        "| --- | --- |",
        f"| n | {PILOT_N} |",
        f"| median | {PILOT_MEDIAN:.2e} |",
        f"| p99.9 | {PILOT_P99_9:.2e} |",
        f"| max | {PILOT_MAX:.2e} |",
        "",
        "Ratio is residual-v2 / clean max "
        f"({PILOT_MAX:.2e}). `n(ratio>1)` is the count of samples with",
        "ratio > 1 or non-finite, out of 200. Not a pass/fail rule.",
        "",
        render_bf16_bit_class_table().rstrip(),
        "",
        "Factors: `gemm_numpy_pair` case_index=3, workload_id=2,",
        "`sample_index` in 0..199 (same mixer as the pilot). One GEMM per",
        "sample_index; every (bit_class, n_flips) reuses that C.",
        "",
    ]
    if rows is None:
        lines.extend(
            [
                "200-sample flip residuals: **UNMEASURED** on the writer",
                "(no NVIDIA GPU). Reproduce on CUDA (~200 GEMMs at 4096³):",
                "",
                "```bash",
                "uv run pytest -m gpu tests/test_w02_4096_flips.py -s",
                "```",
                "",
            ]
        )
        return "\n".join(lines)
    lines.extend(
        [
            "| bit_class | n_flips | n | min ratio | median ratio | max ratio |"
            " n(ratio>1) | n_nonfinite |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        frac = f"{row.n_exceeding_one}/{row.n}"
        lines.append(
            f"| {row.bit_class} | {row.n_flips} | {row.n} | "
            f"{_fmt_ratio(row.min_ratio)} | {_fmt_ratio(row.median_ratio)} | "
            f"{_fmt_ratio(row.max_ratio)} | {frac} | {row.n_nonfinite} |"
        )
    lines.extend(
        [
            "",
            "```bash",
            "uv run pytest -m gpu tests/test_w02_4096_flips.py -s",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


@pytest.mark.cpu
def test_summarize_ratio_lists_min_median_max_and_fraction() -> None:
    collected: dict[tuple[str, int], list[float]] = {
        (bit_class.value, n_flips): [0.5, 0.5, 0.5]
        for bit_class in BitClass
        for n_flips in _FLIP_COUNTS
    }
    collected[("EXPONENT_LOW", 1)] = [0.9, 1.068, 2.0]
    collected[("EXPONENT_HIGH", 1)] = [1.1, 10.0, math.inf]
    rows = summarize_ratio_lists(collected)
    by_key = {(row.bit_class, row.n_flips): row for row in rows}
    low = by_key[("EXPONENT_LOW", 1)]
    assert low.n == 3
    assert low.min_ratio == 0.9
    assert low.median_ratio == 1.068
    assert low.max_ratio == 2.0
    assert low.n_exceeding_one == 2
    assert low.fraction_exceeding_one == 2 / 3
    assert low.n_nonfinite == 0
    high = by_key[("EXPONENT_HIGH", 1)]
    assert high.min_ratio == 1.1
    assert high.max_ratio == math.inf
    assert high.n_exceeding_one == 3
    assert high.n_nonfinite == 1
    sign = by_key[("SIGN", 1)]
    assert sign.n_exceeding_one == 0
    assert sign.max_ratio == 0.5


@pytest.mark.cpu
def test_w02_4096_unmeasured_section_in_doc() -> None:
    doc = (REPO / "docs" / "DETECTION_MATRIX.md").read_text(encoding="utf-8")
    section = render_w02_4096_section(None)
    assert section in doc
    assert "exponent LSB (bit 7)" in doc
    assert "C after the GEMM" in doc
    assert "Not a KT-1 evaluation" in doc


@pytest.mark.gpu
def test_w02_4096_flip_residuals_print_table() -> None:
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    rows = measure_w02_4096_flip_residuals(device, n_samples=FLIP_N)
    expected = len(list(BitClass)) * len(_FLIP_COUNTS)
    assert len(rows) == expected
    assert all(row.n == FLIP_N for row in rows)
    section = render_w02_4096_section(rows)
    print("\n" + section)  # noqa: T201
    doc = (REPO / "docs" / "DETECTION_MATRIX.md").read_text(encoding="utf-8")
    assert "## W02 bfloat16 4096" in doc
    assert "exponent LSB (bit 7)" in doc
