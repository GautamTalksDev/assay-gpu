"""GPU tests: Triton checksum reduction agrees with PyTorch across shapes."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from assay.abft.check import CheckStatus, GemmCheckConfig, check_gemm
from assay.abft.reduce import CheckBackend, ones_matvec
from assay.abft.triton_reduce import triton_available
from assay.reference.spec import WORKLOAD_GEMM_SHAPES

pytestmark = pytest.mark.gpu

REPO_NOISE = Path(__file__).resolve().parents[1] / "data" / "noisefloor"


def test_triton_available_on_gpu_runner() -> None:
    assert torch.cuda.is_available()
    assert triton_available()


def test_triton_ones_matvec_agrees_with_pytorch_shape_sweep() -> None:
    assert torch.cuda.is_available()
    assert triton_available()
    device = torch.device("cuda")
    for rows, _inner, cols in WORKLOAD_GEMM_SHAPES:
        matrix = torch.ones((rows, cols), dtype=torch.float32, device=device)
        pytorch = ones_matvec(matrix, CheckBackend.PYTORCH)
        triton = ones_matvec(matrix, CheckBackend.TRITON)
        assert torch.equal(pytorch, triton), (rows, cols)
        integers = torch.arange(
            rows * cols, dtype=torch.float32, device=device
        ).reshape(rows, cols)
        # Values exceed the 24-bit mantissa for large shapes, so only the
        # ones-matrix case is required to be bitwise across the full sweep.
        if rows * cols < (1 << 24):
            pytorch_i = ones_matvec(integers, CheckBackend.PYTORCH)
            triton_i = ones_matvec(integers, CheckBackend.TRITON)
            assert torch.equal(pytorch_i, triton_i), (rows, cols)


def test_check_gemm_pytorch_and_triton_agree_on_exact_gemm() -> None:
    assert torch.cuda.is_available()
    assert triton_available()
    device = torch.device("cuda")
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=device)
    right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32, device=device)
    product = left @ right
    config_pt = GemmCheckConfig(
        noisefloor_dir=REPO_NOISE,
        workload="W01",
        backend=CheckBackend.PYTORCH,
    )
    config_tr = GemmCheckConfig(
        noisefloor_dir=REPO_NOISE,
        workload="W01",
        backend=CheckBackend.TRITON,
    )
    pytorch = check_gemm(left, right, product, config_pt)
    triton = check_gemm(left, right, product, config_tr)
    assert pytorch.status is CheckStatus.INCONCLUSIVE
    assert triton.status is CheckStatus.INCONCLUSIVE
    assert pytorch.residual_hex == triton.residual_hex
    assert pytorch.noisefloor_spec_version == triton.noisefloor_spec_version
    assert pytorch.n_samples == triton.n_samples


def test_check_gemm_backends_agree_ones_shape_sweep() -> None:
    assert torch.cuda.is_available()
    assert triton_available()
    device = torch.device("cuda")
    for rows, inner, cols in WORKLOAD_GEMM_SHAPES:
        left = torch.ones((rows, inner), dtype=torch.float32, device=device)
        right = torch.ones((inner, cols), dtype=torch.float32, device=device)
        product = left @ right
        pytorch = check_gemm(
            left,
            right,
            product,
            GemmCheckConfig(
                noisefloor_dir=REPO_NOISE,
                workload="W01",
                backend=CheckBackend.PYTORCH,
            ),
        )
        triton = check_gemm(
            left,
            right,
            product,
            GemmCheckConfig(
                noisefloor_dir=REPO_NOISE,
                workload="W01",
                backend=CheckBackend.TRITON,
            ),
        )
        assert pytorch.residual_hex == triton.residual_hex, (rows, inner, cols)
        assert pytorch.status is CheckStatus.INCONCLUSIVE
        assert triton.status is CheckStatus.INCONCLUSIVE
