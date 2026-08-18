"""Checksum detector (algorithm-based fault tolerance)."""

from assay.abft.check import (
    CheckResult,
    CheckStatus,
    GemmCheckConfig,
    check_gemm,
    decide_from_lookup,
)
from assay.abft.gemm import (
    RESIDUAL_VERSION,
    gemm_checksum_fp64,
    normalize_by_scale,
    normalized_checksum_residual,
    sum_elements_fp64,
)
from assay.abft.reduce import (
    CheckBackend,
    absolute_factor_scale,
    ones_matvec,
    ones_matvec_pytorch,
    ones_sided_checksums,
    vector_residual_normalized,
)

__all__ = [
    "RESIDUAL_VERSION",
    "CheckBackend",
    "CheckResult",
    "CheckStatus",
    "GemmCheckConfig",
    "absolute_factor_scale",
    "check_gemm",
    "decide_from_lookup",
    "gemm_checksum_fp64",
    "normalize_by_scale",
    "normalized_checksum_residual",
    "ones_matvec",
    "ones_matvec_pytorch",
    "ones_sided_checksums",
    "sum_elements_fp64",
    "vector_residual_normalized",
]
