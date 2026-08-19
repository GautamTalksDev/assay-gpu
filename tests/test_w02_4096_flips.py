"""W02 bf16 4096³ residual-v2 flip residuals vs the T4 pilot floor.

GPU only. Does not evaluate KT-1 and does not write a threshold.
Imports assay.inject on purpose; the run path must not.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from assay.abft.reduce import (
    CheckBackend,
    ones_matvec,
    ones_sided_checksums,
    vector_residual_parts,
)
from assay.inject import BitClass, flip_random
from assay.reference.spec import BASE_SEED
from assay.workload.context import gemm_flags
from assay.workload.gemm import gemm_numpy_pair

pytestmark = pytest.mark.gpu

REPO = Path(__file__).resolve().parents[1]
_SHAPE = (4096, 4096, 4096)
_CASE_INDEX = 3
_WORKLOAD_ID = 2
_FLIP_COUNTS = (1, 2, 4)
_BACKEND = CheckBackend.PYTORCH

# Tesla T4 residual-v2 pilot, n=2000. Not a detection threshold.
PILOT_N = 2000
PILOT_MEDIAN = 1.07e-8
PILOT_P99_9 = 5.07e-8
PILOT_MAX = 5.57e-8


def _cell_seed(bit_class: BitClass, n_flips: int) -> int:
    class_id = list(BitClass).index(bit_class) + 1
    return BASE_SEED + 2 * 1_000_003 + class_id * 1_009 + n_flips


@dataclass(frozen=True, slots=True)
class FlipResidualRow:
    bit_class: str
    n_flips: int
    residual: float
    residual_hex: str
    ratio_to_clean_max: float
    nonfinite: bool


def _factors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, inner, cols = _SHAPE
    left_np, right_np = gemm_numpy_pair(
        rows,
        inner,
        cols,
        case_index=_CASE_INDEX,
        sample_index=0,
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


def measure_w02_4096_flip_residuals(device: torch.device) -> list[FlipResidualRow]:
    """One W02 4096³ GEMM, then residual-v2 after each seeded C flip."""
    with gemm_flags(fp16_reduced=False, bf16_reduced=False):
        left, right, product = _factors(device)
        _c_e, a_be = ones_sided_checksums(left, right, product, _BACKEND)
        rows: list[FlipResidualRow] = []
        for bit_class in BitClass:
            for n_flips in _FLIP_COUNTS:
                flipped, _locs = flip_random(
                    product,
                    n_flips,
                    bit_class,
                    _cell_seed(bit_class, n_flips),
                )
                c_e = ones_matvec(flipped, _BACKEND)
                _abs_r, _scale, residual = vector_residual_parts(c_e, a_be, left, right)
                nonfinite = not np.isfinite(residual)
                ratio = (
                    float("inf")
                    if nonfinite or PILOT_MAX == 0.0
                    else residual / PILOT_MAX
                )
                rows.append(
                    FlipResidualRow(
                        bit_class=bit_class.value,
                        n_flips=n_flips,
                        residual=float(residual),
                        residual_hex=float(residual).hex(),
                        ratio_to_clean_max=float(ratio),
                        nonfinite=nonfinite,
                    )
                )
        return rows


def render_w02_4096_section(rows: list[FlipResidualRow] | None) -> str:
    lines = [
        "## W02 bfloat16 4096³ flip residuals vs T4 residual-v2 pilot floor",
        "",
        "Not a KT-1 evaluation. No threshold is set. residual-v1 flip",
        "residuals (near 1.0 on the 16-cubed v1 matrix) are void.",
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
        "Flips: `flip_random` on C after one GEMM, same bit_class grid as",
        "the 16-cubed matrix. Factors: `gemm_numpy_pair` case_index=3,",
        "sample_index=0, workload_id=2. Ratio is flip residual / clean max.",
        "",
    ]
    if rows is None:
        lines.extend(
            [
                "Flip residuals: **UNMEASURED** on the writer (no NVIDIA GPU).",
                "Reproduce on CUDA:",
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
            "| bit_class | n_flips | residual | residual_hex | residual / clean max |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        resid_txt = "inf" if row.nonfinite else format(row.residual, ".6g")
        if np.isfinite(row.ratio_to_clean_max):
            ratio_txt = format(row.ratio_to_clean_max, ".6g")
        else:
            ratio_txt = "inf"
        lines.append(
            f"| {row.bit_class} | {row.n_flips} | {resid_txt} | "
            f"`{row.residual_hex}` | {ratio_txt} |"
        )
    lines.append("")
    by_class: dict[str, list[float]] = {}
    for row in rows:
        by_class.setdefault(row.bit_class, []).append(row.ratio_to_clean_max)
    lines.extend(
        [
            "Per bit_class, residual / clean max over n_flips in {1, 2, 4}:",
            "",
            "| bit_class | ratios |",
            "| --- | --- |",
        ]
    )
    for name, ratios in by_class.items():
        rendered = ", ".join(
            "inf" if not np.isfinite(value) else format(value, ".6g")
            for value in ratios
        )
        lines.append(f"| {name} | {rendered} |")
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


def test_w02_4096_flip_residuals_print_table() -> None:
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    rows = measure_w02_4096_flip_residuals(device)
    assert len(rows) == len(list(BitClass)) * len(_FLIP_COUNTS)
    section = render_w02_4096_section(rows)
    print("\n" + section)  # noqa: T201
    doc = (REPO / "docs" / "DETECTION_MATRIX.md").read_text(encoding="utf-8")
    assert "## W02 bfloat16 4096" in doc
