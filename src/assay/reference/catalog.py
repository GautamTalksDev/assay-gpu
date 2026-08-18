"""Build the golden catalog: arrays, fp64 results, hashes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from assay.reference.arrays import generate_array
from assay.reference.compute import (
    exp_fp64,
    matmul_fp64,
    mean_fp64_c_order,
    rsqrt_fp64,
    sdpa_fp64,
    sum_fp64_c_order,
    tanh_fp64,
)
from assay.reference.hashing import sha256_array
from assay.reference.spec import (
    BASE_SEED,
    DISTRIBUTION_UNIFORM_POS,
    DISTRIBUTION_UNIFORM_UNIT,
    REFERENCE_ELEMENTWISE_SHAPE,
    REFERENCE_GEMM_SHAPES,
    REFERENCE_REDUCE_LENGTH,
    REFERENCE_SDPA_SHAPES,
    seed_offset,
)
from assay.reference.transformer import generate_prompt, generate_weights, greedy_decode


@dataclass(frozen=True, slots=True)
class Artifact:
    name: str
    file: str
    key: str
    shape: tuple[int, ...]
    dtype: str
    seed: int
    distribution: str
    numpy_version: str
    sha256: str
    role: str
    note: str


def _artifact(  # noqa: PLR0913
    *,
    name: str,
    file: str,
    key: str,
    array: npt.NDArray[np.generic],
    seed: int,
    distribution: str,
    role: str,
    note: str,
) -> tuple[Artifact, npt.NDArray[np.generic]]:
    item = Artifact(
        name=name,
        file=file,
        key=key,
        shape=tuple(int(s) for s in array.shape),
        dtype=str(array.dtype),
        seed=seed,
        distribution=distribution,
        numpy_version=str(np.__version__),
        sha256=sha256_array(array),
        role=role,
        note=note,
    )
    return item, array


def generate_catalog() -> tuple[  # noqa: PLR0915
    list[Artifact], dict[str, dict[str, npt.NDArray[np.generic]]]
]:
    """Return metadata and npz-filename -> key -> array."""
    artifacts: list[Artifact] = []
    files: dict[str, dict[str, npt.NDArray[np.generic]]] = {}

    def add(art: Artifact, array: npt.NDArray[np.generic]) -> None:
        artifacts.append(art)
        files.setdefault(art.file, {})[art.key] = array

    for index, (m_dim, k_dim, n_dim) in enumerate(REFERENCE_GEMM_SHAPES):
        seed_a = seed_offset(1, index * 2)
        seed_b = seed_offset(1, index * 2 + 1)
        mat_a = generate_array((m_dim, k_dim), seed_a, DISTRIBUTION_UNIFORM_UNIT)
        mat_b = generate_array((k_dim, n_dim), seed_b, DISTRIBUTION_UNIFORM_UNIT)
        product = matmul_fp64(
            np.ascontiguousarray(mat_a, dtype=np.dtype("<f8")),
            np.ascontiguousarray(mat_b, dtype=np.dtype("<f8")),
        )
        stem = f"gemm_m{m_dim}_k{k_dim}_n{n_dim}"
        fname = f"{stem}.npz"
        art, arr = _artifact(
            name=f"{stem}_a",
            file=fname,
            key="a",
            array=mat_a,
            seed=seed_a,
            distribution=DISTRIBUTION_UNIFORM_UNIT,
            role="input",
            note="fp64 GEMM left factor",
        )
        add(art, arr)
        art, arr = _artifact(
            name=f"{stem}_b",
            file=fname,
            key="b",
            array=mat_b,
            seed=seed_b,
            distribution=DISTRIBUTION_UNIFORM_UNIT,
            role="input",
            note="fp64 GEMM right factor",
        )
        add(art, arr)
        art, arr = _artifact(
            name=f"{stem}_c",
            file=fname,
            key="c",
            array=product,
            seed=seed_a,
            distribution="fp64_kloop_matmul",
            role="fp64_result",
            note="defined-order K-loop multiply-then-add; not BLAS",
        )
        add(art, arr)

    for index, (batch, heads, seq, head_dim) in enumerate(REFERENCE_SDPA_SHAPES):
        shape = (batch, heads, seq, head_dim)
        fname = f"sdpa_b{batch}_h{heads}_s{seq}_d{head_dim}.npz"
        stem = fname.removesuffix(".npz")
        parts: dict[str, npt.NDArray[np.generic]] = {}
        for key_name, off in (("q", 0), ("k", 1), ("v", 2)):
            seed = seed_offset(4, index * 3 + off)
            parts[key_name] = generate_array(shape, seed, DISTRIBUTION_UNIFORM_UNIT)
            art, arr = _artifact(
                name=f"{stem}_{key_name}",
                file=fname,
                key=key_name,
                array=parts[key_name],
                seed=seed,
                distribution=DISTRIBUTION_UNIFORM_UNIT,
                role="input",
                note="fp64 SDPA operand",
            )
            add(art, arr)
        out = sdpa_fp64(
            np.ascontiguousarray(parts["q"], dtype=np.dtype("<f8")),
            np.ascontiguousarray(parts["k"], dtype=np.dtype("<f8")),
            np.ascontiguousarray(parts["v"], dtype=np.dtype("<f8")),
            causal=False,
        )
        art, arr = _artifact(
            name=f"{stem}_out",
            file=fname,
            key="out",
            array=out,
            seed=seed_offset(4, index * 3),
            distribution="fp64_sdpa",
            role="fp64_result",
            note="scale=1/sqrt(head_dim); defined-order matmul and softmax",
        )
        add(art, arr)

    reduce_seed = seed_offset(5, 0)
    reduce_in = generate_array(
        (REFERENCE_REDUCE_LENGTH,), reduce_seed, DISTRIBUTION_UNIFORM_UNIT
    )
    fname = "reduce_8192.npz"
    art, arr = _artifact(
        name="reduce_in",
        file=fname,
        key="x",
        array=reduce_in,
        seed=reduce_seed,
        distribution=DISTRIBUTION_UNIFORM_UNIT,
        role="input",
        note="fp64 reduction input, C-order",
    )
    add(art, arr)
    x64 = np.ascontiguousarray(reduce_in, dtype=np.dtype("<f8"))
    art, arr = _artifact(
        name="reduce_sum",
        file=fname,
        key="sum",
        array=sum_fp64_c_order(x64),
        seed=reduce_seed,
        distribution="fp64_c_order_sum",
        role="fp64_result",
        note="left-to-right sum of C-order ravel",
    )
    add(art, arr)
    art, arr = _artifact(
        name="reduce_mean",
        file=fname,
        key="mean",
        array=mean_fp64_c_order(x64),
        seed=reduce_seed,
        distribution="fp64_c_order_mean",
        role="fp64_result",
        note="c-order sum divided by length",
    )
    add(art, arr)

    ew_seed = seed_offset(6, 0)
    ew_unit = generate_array(
        REFERENCE_ELEMENTWISE_SHAPE, ew_seed, DISTRIBUTION_UNIFORM_UNIT
    )
    ew_pos = generate_array(
        REFERENCE_ELEMENTWISE_SHAPE, seed_offset(6, 1), DISTRIBUTION_UNIFORM_POS
    )
    fname = "elementwise.npz"
    art, arr = _artifact(
        name="elementwise_unit",
        file=fname,
        key="x_unit",
        array=ew_unit,
        seed=ew_seed,
        distribution=DISTRIBUTION_UNIFORM_UNIT,
        role="input",
        note="domain for exp and tanh",
    )
    add(art, arr)
    art, arr = _artifact(
        name="elementwise_pos",
        file=fname,
        key="x_pos",
        array=ew_pos,
        seed=seed_offset(6, 1),
        distribution=DISTRIBUTION_UNIFORM_POS,
        role="input",
        note="domain for rsqrt (strictly positive)",
    )
    add(art, arr)
    u64 = np.ascontiguousarray(ew_unit, dtype=np.dtype("<f8"))
    p64 = np.ascontiguousarray(ew_pos, dtype=np.dtype("<f8"))
    for key_name, produced, dist, note in (
        ("exp", exp_fp64(u64), "fp64_exp", "numpy.exp"),
        ("tanh", tanh_fp64(u64), "fp64_tanh", "numpy.tanh"),
        ("rsqrt", rsqrt_fp64(p64), "fp64_rsqrt", "1/sqrt, defined-order"),
    ):
        art, arr = _artifact(
            name=f"elementwise_{key_name}",
            file=fname,
            key=key_name,
            array=produced,
            seed=ew_seed,
            distribution=dist,
            role="fp64_result",
            note=note,
        )
        add(art, arr)

    weights = generate_weights()
    fname = "w07_weights.npz"
    for key_name, array in weights.items():
        art, arr = _artifact(
            name=f"w07_{key_name}",
            file=fname,
            key=key_name,
            array=array,
            seed=BASE_SEED + 7000,
            distribution="w07_seeded_open_weights",
            role="weights",
            note="in-repo decoder weights; no network fetch",
        )
        add(art, arr)
    prompt = generate_prompt()
    art, arr = _artifact(
        name="w07_prompt",
        file="w07_tokens.npz",
        key="prompt",
        array=prompt,
        seed=BASE_SEED + 7099,
        distribution="integers_0_vocab",
        role="input",
        note="fixed greedy-decode prompt",
    )
    add(art, arr)
    tokens = greedy_decode(weights)
    art, arr = _artifact(
        name="w07_greedy",
        file="w07_tokens.npz",
        key="greedy",
        array=tokens,
        seed=BASE_SEED + 7000,
        distribution="fp64_greedy_argmax",
        role="fp64_result",
        note="argmax decode; fp64 CPU forward",
    )
    add(art, arr)

    return artifacts, files


def manifest_dict(artifacts: list[Artifact]) -> dict[str, Any]:
    return {
        "numpy_version": str(np.__version__),
        "hash_algorithm": "sha256(dtype.str|shape + NUL + C-contiguous tobytes)",
        "byteorder": "little-endian payloads (<f8 / <i8)",
        "artifacts": [asdict(item) for item in artifacts],
    }
