"""Unit tests for residual-v3 elementwise per-row residual."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from assay.abft.residual_v3 import _fixed_row_indices, residual_v3


@pytest.mark.cpu
class TestResidualV3:
    def test_exact_product_gives_zero_residual(self) -> None:
        """If C == A @ B exactly (fp64), all row residuals are zero."""
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float64)
        c = a @ b
        result = residual_v3(a, b, c)
        assert result["r_max"] == 0.0
        assert result["r_median"] == 0.0
        assert result["r_p99"] == 0.0
        assert result["n_scale_zero"] == 0

    def test_known_perturbation(self) -> None:
        """Hand-computable case: flip one element of C by a known amount."""
        a = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        b = torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float64)
        c = a @ b  # [[2, 3], [4, 5]]
        # Perturb C[0,0] by +10
        c_bad = c.clone()
        c_bad[0, 0] += 10.0
        result = residual_v3(a, b, c_bad)
        # d = C_bad @ e = [2+10+3, 4+5] = [15, 9]
        # d' = A @ (B @ e) = A @ [5, 9] = [5, 9]
        # abs_diff = [10, 0]
        # scale = |A| @ (|B| @ e) = I @ [5, 9] = [5, 9]
        # r = [10/5, 0/9] = [2.0, 0.0]
        assert result["r_max"] == pytest.approx(2.0)
        assert result["r_median"] == pytest.approx(1.0)  # median of [0, 2] = 1
        assert result["n_scale_zero"] == 0

    def test_scale_nonnegative(self) -> None:
        """Scale is always nonneg for any input."""
        rng = np.random.default_rng(42)
        a = torch.from_numpy(rng.standard_normal((16, 16)))
        b = torch.from_numpy(rng.standard_normal((16, 16)))
        c = torch.from_numpy(rng.standard_normal((16, 16)))
        result = residual_v3(a, b, c)
        assert result["r_max"] >= 0.0
        assert result["r_median"] >= 0.0
        assert result["n_scale_zero"] >= 0

    def test_invariant_to_row_permutation_of_a(self) -> None:
        """r_max is invariant to permuting rows of A (with matching C rows)."""
        rng = np.random.default_rng(123)
        a = torch.from_numpy(rng.standard_normal((8, 8)))
        b = torch.from_numpy(rng.standard_normal((8, 8)))
        c = a @ b
        # Add a perturbation to C
        c[3, 2] += 100.0

        result_orig = residual_v3(a, b, c)

        # Permute rows of A and C together
        perm = [7, 6, 5, 4, 3, 2, 1, 0]
        a_perm = a[perm]
        c_perm = c[perm]
        result_perm = residual_v3(a_perm, b, c_perm)

        assert result_orig["r_max"] == pytest.approx(result_perm["r_max"])

    def test_rejects_non_2d(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            residual_v3(torch.zeros(4), torch.zeros(4), torch.zeros(4))

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            residual_v3(
                torch.zeros(2, 3),
                torch.zeros(4, 2),
                torch.zeros(2, 2),
            )

    def test_r_rows_subset(self) -> None:
        """r_rows contains values for the fixed subsample indices."""
        a = torch.eye(4, dtype=torch.float64)
        b = torch.ones(4, 4, dtype=torch.float64)
        c = a @ b
        c[1, 0] += 1.0
        result = residual_v3(a, b, c)
        # M=4 <= 256, so all rows returned
        assert len(result["r_rows"]) == 4
        assert result["row_indices"] == [0, 1, 2, 3]


@pytest.mark.cpu
class TestFixedRowIndices:
    def test_small_m_returns_all(self) -> None:
        assert _fixed_row_indices(100) == list(range(100))

    def test_large_m_returns_256(self) -> None:
        indices = _fixed_row_indices(4096)
        assert len(indices) == 256
        assert indices == sorted(indices)
        assert all(0 <= i < 4096 for i in indices)

    def test_deterministic(self) -> None:
        a = _fixed_row_indices(4096)
        b = _fixed_row_indices(4096)
        assert a == b
