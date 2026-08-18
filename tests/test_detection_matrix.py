"""Cross product: does check_gemm FAIL a seeded flip?

This is a test harness. It imports assay.inject on purpose. The run path
must not. See tests/test_inject.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from assay.abft.check import CheckStatus, GemmCheckConfig, check_gemm
from assay.inject import BitClass, flip_random
from assay.reference.spec import BASE_SEED
from assay.workload.gemm import gemm_numpy_pair

pytestmark = pytest.mark.cpu

REPO = Path(__file__).resolve().parents[1]
NOISEFLOOR = REPO / "data" / "noisefloor"
DOC_PATH = REPO / "docs" / "DETECTION_MATRIX.md"

_SHAPE = (16, 16, 16)
_FLIP_COUNTS = (1, 2, 4)
_GEMM_CASES: tuple[tuple[str, torch.dtype], ...] = (
    ("W01", torch.float32),
    ("W02", torch.bfloat16),
    ("W03", torch.float16),
)
_SUITE_DTYPE = {
    "W01": "float32",
    "W02": "bfloat16",
    "W03": "float16",
    "W04": "float32",
    "W05": "float32",
    "W06": "float32",
    "W07": "float32",
}
_ALL_WORKLOADS = ("W01", "W02", "W03", "W04", "W05", "W06", "W07")
_ALL_DTYPES = ("float32", "bfloat16", "float16", "int8")


@dataclass(frozen=True, slots=True)
class MatrixCell:
    workload: str
    dtype_name: str
    bit_class: str
    n_flips: int
    detector_status: str
    checksum_moved: bool
    caught: bool
    residual_clean_hex: str
    residual_flipped_hex: str
    n_samples: int
    spec_version: str
    reason: str


def _cell_seed(workload: str, bit_class: BitClass, n_flips: int) -> int:
    work = int(workload[1:])
    class_id = list(BitClass).index(bit_class) + 1
    return BASE_SEED + work * 1_000_003 + class_id * 1_009 + n_flips


def _factors(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, inner, cols = _SHAPE
    left_np, right_np = gemm_numpy_pair(rows, inner, cols, case_index=0)
    left = torch.from_numpy(np.ascontiguousarray(left_np, dtype=np.float32)).to(
        dtype=dtype
    )
    right = torch.from_numpy(np.ascontiguousarray(right_np, dtype=np.float32)).to(
        dtype=dtype
    )
    return left, right, left @ right


def measure_gemm_cell(
    workload: str,
    dtype: torch.dtype,
    bit_class: BitClass,
    n_flips: int,
) -> MatrixCell:
    left, right, product = _factors(dtype)
    config = GemmCheckConfig(noisefloor_dir=NOISEFLOOR, workload=workload)
    clean = check_gemm(left, right, product, config)
    flipped, _locs = flip_random(
        product, n_flips, bit_class, _cell_seed(workload, bit_class, n_flips)
    )
    dirty = check_gemm(left, right, flipped, config)
    moved = clean.residual_hex != dirty.residual_hex
    return MatrixCell(
        workload=workload,
        dtype_name=clean.dtype_name,
        bit_class=bit_class.value,
        n_flips=n_flips,
        detector_status=dirty.status.value,
        checksum_moved=moved,
        caught=dirty.status is CheckStatus.FAIL,
        residual_clean_hex=clean.residual_hex,
        residual_flipped_hex=dirty.residual_hex,
        n_samples=dirty.n_samples,
        spec_version=dirty.noisefloor_spec_version,
        reason=dirty.reason,
    )


def measure_gemm_product() -> list[MatrixCell]:
    return [
        measure_gemm_cell(workload, dtype, bit_class, n_flips)
        for workload, dtype in _GEMM_CASES
        for bit_class in BitClass
        for n_flips in _FLIP_COUNTS
    ]


def applicability_reason(workload: str, dtype_name: str) -> str:
    if dtype_name == "int8":
        return "check_gemm rejects int8; injector-only"
    if workload in {"W04", "W05", "W06", "W07"}:
        return "no GEMM ABFT in CP-4"
    native = _SUITE_DTYPE[workload]
    if dtype_name != native:
        return f"not a suite dtype (suite uses {native})"
    return "measured"


def render_detection_matrix(cells: list[MatrixCell]) -> str:
    caught_n = sum(1 for cell in cells if cell.caught)
    moved_n = sum(1 for cell in cells if cell.checksum_moved)
    class_lines: list[str] = []
    for item in BitClass:
        group = [cell for cell in cells if cell.bit_class == item.value]
        moved = sum(1 for cell in group if cell.checksum_moved)
        caught = sum(1 for cell in group if cell.caught)
        class_lines.append(f"| {item.value} | {len(group)} | {moved} | {caught} |")

    table_rows = [
        "| workload | dtype | bit_class | n_flips | detector | checksum_moved | "
        "caught | residual_clean_hex | residual_flipped_hex | n_samples | spec |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    table_rows.extend(
        (
            f"| {cell.workload} | {cell.dtype_name} | {cell.bit_class} | "
            f"{cell.n_flips} | {cell.detector_status} | {cell.checksum_moved} | "
            f"{cell.caught} | `{cell.residual_clean_hex}` | "
            f"`{cell.residual_flipped_hex}` | {cell.n_samples} | "
            f"{cell.spec_version} |"
        )
        for cell in cells
    )

    app_rows = [
        "| workload | float32 | bfloat16 | float16 | int8 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for workload in _ALL_WORKLOADS:
        reasons = [applicability_reason(workload, name) for name in _ALL_DTYPES]
        app_rows.append("| " + " | ".join((workload, *reasons)) + " |")

    spec = cells[0].spec_version
    n_samples = cells[0].n_samples
    reason = cells[0].reason
    invisible = [cell for cell in cells if not cell.checksum_moved]
    if invisible:
        invisible_block = [
            "2. Checksum-invisible flips. The residual hex was unchanged in",
            f"   {len(invisible)} of {len(cells)} cells. Classes: "
            + ", ".join(sorted({cell.bit_class for cell in invisible}))
            + ".",
            "",
            *[
                f"- {cell.workload} {cell.dtype_name} {cell.bit_class} "
                f"n_flips={cell.n_flips} residual `{cell.residual_flipped_hex}`"
                for cell in invisible
            ],
            "",
            "A ones-vector checksum sums each row of C. A MANTISSA_LOW flip",
            "is a few ULPs in one element. Native-dtype `C @ e` can round",
            "that delta away, so A@(B@e) vs C@e does not move.",
            "Characterizing the GPU will not make those cells FAIL. This is",
            "the asymmetry the bit_class API exists to record.",
        ]
    else:
        invisible_block = [
            "2. Checksum-invisible flips: none in this grid. Every class",
            "   moved the residual hex.",
        ]
    lines = [
        "# Detection matrix (CP-5)",
        "",
        "Test fixture, not a product feature. Each measured cell corrupts",
        "`C` after `C = A @ B` with `flip_random`, then calls `check_gemm`",
        "against the committed `data/noisefloor/` (noisefloor-v1).",
        "",
        "Shape: 16 x 16 x 16 GEMM, factors from `gemm_numpy_pair` case 0.",
        "Device: CPU. Backend: PyTorch. Flip counts: 1, 2, 4.",
        f"Noisefloor: `{spec}`, n_samples={n_samples}.",
        "",
        "## What 'caught' means",
        "",
        "`caught` is `CheckResult.status == FAIL`. INCONCLUSIVE is not a",
        "catch. PASS is not a catch. This table does not invent a threshold",
        "to pretend a FAIL happened.",
        "",
        f"Lookup reason on every measured cell: `{reason}`",
        "",
        "## Summary of measured GEMM cells",
        "",
        f"Measured cells: {len(cells)}.",
        f"Detector FAIL (caught): {caught_n}.",
        f"Checksum residual hex changed after the flip: {moved_n}.",
        "",
        "| bit_class | n_cells | checksum_moved | caught |",
        "| --- | --- | --- | --- |",
        *class_lines,
        "",
        "## Which classes the detector does not catch, and why",
        "",
        f"Caught (FAIL): {caught_n} of {len(cells)}. The detector did not",
        "catch any flip. Two different reasons, do not mix them:",
        "",
        "1. Uncharacterized noisefloor. Every cell is INCONCLUSIVE because",
        f"   n_samples={n_samples}. `check_gemm` is not allowed to FAIL.",
        "   Residual-v2 scale does not depend on C. checksum_moved is the",
        "   numerator. See the summary table for per-class counts.",
        "   EXPONENT_HIGH includes Inf-class cells. Those still do not FAIL",
        "   today. After a real floor exists, EXPONENT_HIGH is the class",
        "   most likely to exceed sample max.",
        "",
        *invisible_block,
        "",
        "W04-W07 have no GEMM checksum. int8 has an injector layout and no",
        "`check_gemm` path. Those are out of scope for CP-4, not a mantissa",
        "finding.",
        "",
        "## Applicability of workload x dtype",
        "",
        *app_rows,
        "",
        "## Measured product: GEMM workload x native dtype x bit_class x n_flips",
        "",
        *table_rows,
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run pytest -m cpu tests/test_detection_matrix.py",
        "```",
        "",
        "The test re-measures and asserts this file matches. Do not edit",
        "numbers by hand.",
        "",
    ]
    return "\n".join(lines)


def test_detection_matrix_matches_committed_doc() -> None:
    cells = measure_gemm_product()
    expected_n = len(_GEMM_CASES) * len(list(BitClass)) * len(_FLIP_COUNTS)
    assert len(cells) == expected_n
    assert all(cell.detector_status == CheckStatus.INCONCLUSIVE.value for cell in cells)
    assert all(cell.n_samples == 0 for cell in cells)
    assert not any(cell.caught for cell in cells)
    rendered = render_detection_matrix(cells)
    assert DOC_PATH.is_file()
    assert DOC_PATH.read_text(encoding="utf-8") == rendered
