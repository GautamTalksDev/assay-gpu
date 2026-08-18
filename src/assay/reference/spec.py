"""Fixed seeds, shapes, and distributions for references and workloads.

These values specify the suite. They are not detection thresholds.
"""

from __future__ import annotations

# Identifier seed for this project (ISO date of the workload checkpoint).
BASE_SEED = 20260818

# Suite identity recorded in attestation-v1. Not a detection threshold.
WORKLOAD_SUITE_VERSION = "workload-suite-v1"

DISTRIBUTION_UNIFORM_UNIT = "uniform_-1_1"
DISTRIBUTION_NORMAL = "normal_0_1"
DISTRIBUTION_UNIFORM_POS = "uniform_1_4"

# Small GEMM shapes stored as fp64 goldens (CPU, version-controlled).
REFERENCE_GEMM_SHAPES: tuple[tuple[int, int, int], ...] = (
    (32, 32, 32),
    (64, 48, 32),
    (128, 64, 96),
)

# W01-W03: square 512..8192 and rectangular (M, K, N).
WORKLOAD_GEMM_SHAPES: tuple[tuple[int, int, int], ...] = (
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (512, 1024, 2048),
    (1024, 512, 2048),
    (2048, 1024, 512),
    (4096, 8192, 1024),
    (8192, 1024, 4096),
)

# Default characterize coverage: skip suite shapes larger than this side.
# 8192 cases require --include-large. This is a shape filter, not a tolerance.
CHARACTERIZATION_MAX_SIDE = 4096

# W04: (batch, heads, seq, head_dim)
WORKLOAD_SDPA_SHAPES: tuple[tuple[int, int, int, int], ...] = (
    (1, 8, 128, 64),
    (1, 8, 512, 64),
    (1, 8, 1024, 64),
    (4, 8, 128, 64),
    (4, 16, 128, 64),
    (8, 16, 512, 64),
)

REFERENCE_SDPA_SHAPES: tuple[tuple[int, int, int, int], ...] = (
    (1, 4, 16, 8),
    (2, 8, 32, 16),
)

REFERENCE_REDUCE_LENGTH = 8192
WORKLOAD_REDUCE_LENGTHS: tuple[int, ...] = (1 << 20, 1 << 24)

REFERENCE_ELEMENTWISE_SHAPE: tuple[int, int] = (64, 64)
WORKLOAD_ELEMENTWISE_SHAPE: tuple[int, int] = (4096, 4096)

W07_VOCAB = 128
W07_D_MODEL = 64
W07_HEADS = 4
W07_LAYERS = 2
W07_FF = 128
W07_PROMPT_LEN = 8
W07_DECODE_STEPS = 8


def seed_offset(workload_id: int, case_index: int) -> int:
    """Deterministic seed mix. Does not use Python's salted hash()."""
    return BASE_SEED + workload_id * 1_000_003 + case_index


# Prime stride so sample_index cannot collide with case_index in seed_offset.
# This is a seed mixer, not a detection threshold.
_SAMPLE_STRIDE = 1_000_037


def sample_factor_seed(
    workload_id: int, shape_index: int, sample_index: int, factor: int
) -> int:
    """Seed for one GEMM factor. factor 0 is A, factor 1 is B.

    Mix is BASE_SEED + workload id + shape index + sample index, integer
    arithmetic only. sample_index=0 reproduces seed_offset(workload_id,
    shape_index*2+factor).
    """
    if factor not in (0, 1):
        msg = "factor must be 0 (A) or 1 (B)"
        raise ValueError(msg)
    if sample_index < 0:
        msg = "sample_index must be >= 0"
        raise ValueError(msg)
    return (
        seed_offset(workload_id, shape_index * 2 + factor)
        + sample_index * _SAMPLE_STRIDE
    )
