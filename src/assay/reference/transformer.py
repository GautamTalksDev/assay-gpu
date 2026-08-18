"""In-repo decoder (open weights generated from a seed — no download)."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from assay.ieee import FP64_EPS
from assay.reference.arrays import generate_array, generate_int64
from assay.reference.compute import gelu_erf_fp64, matmul_fp64, rmsnorm_fp64, sdpa_fp64
from assay.reference.spec import (
    BASE_SEED,
    DISTRIBUTION_NORMAL,
    DISTRIBUTION_UNIFORM_POS,
    DISTRIBUTION_UNIFORM_UNIT,
    W07_D_MODEL,
    W07_DECODE_STEPS,
    W07_FF,
    W07_HEADS,
    W07_LAYERS,
    W07_PROMPT_LEN,
    W07_VOCAB,
)


def _init_scale(fan_in: int) -> np.float64:
    return np.float64(1.0) / np.float64(math.sqrt(fan_in))


def generate_weights() -> dict[str, npt.NDArray[np.float64]]:
    """Named weight tensors. Each draw uses a dedicated seed offset."""
    seq = {"n": 0}

    def next_seed() -> int:
        seq["n"] += 1
        return BASE_SEED + 7000 + seq["n"]

    weights: dict[str, npt.NDArray[np.float64]] = {}
    tok = generate_array((W07_VOCAB, W07_D_MODEL), next_seed(), DISTRIBUTION_NORMAL)
    weights["tok_emb"] = np.ascontiguousarray(
        np.multiply(tok, _init_scale(W07_D_MODEL)), dtype=np.dtype("<f8")
    )
    for layer in range(W07_LAYERS):
        prefix = f"layer{layer}_"
        weights[prefix + "attn_scale"] = np.ascontiguousarray(
            generate_array((W07_D_MODEL,), next_seed(), DISTRIBUTION_UNIFORM_POS),
            dtype=np.dtype("<f8"),
        )
        for name, shape, fan in (
            ("wq", (W07_D_MODEL, W07_D_MODEL), W07_D_MODEL),
            ("wk", (W07_D_MODEL, W07_D_MODEL), W07_D_MODEL),
            ("wv", (W07_D_MODEL, W07_D_MODEL), W07_D_MODEL),
            ("wo", (W07_D_MODEL, W07_D_MODEL), W07_D_MODEL),
            ("w1", (W07_D_MODEL, W07_FF), W07_D_MODEL),
            ("w2", (W07_FF, W07_D_MODEL), W07_FF),
        ):
            raw = generate_array(shape, next_seed(), DISTRIBUTION_UNIFORM_UNIT)
            weights[prefix + name] = np.ascontiguousarray(
                np.multiply(raw, _init_scale(fan)), dtype=np.dtype("<f8")
            )
        weights[prefix + "ffn_scale"] = np.ascontiguousarray(
            generate_array((W07_D_MODEL,), next_seed(), DISTRIBUTION_UNIFORM_POS),
            dtype=np.dtype("<f8"),
        )
    weights["final_scale"] = np.ascontiguousarray(
        generate_array((W07_D_MODEL,), next_seed(), DISTRIBUTION_UNIFORM_POS),
        dtype=np.dtype("<f8"),
    )
    raw_head = generate_array(
        (W07_D_MODEL, W07_VOCAB), next_seed(), DISTRIBUTION_UNIFORM_UNIT
    )
    weights["lm_head"] = np.ascontiguousarray(
        np.multiply(raw_head, _init_scale(W07_D_MODEL)), dtype=np.dtype("<f8")
    )
    return weights


def generate_prompt() -> npt.NDArray[np.int64]:
    return generate_int64((W07_PROMPT_LEN,), BASE_SEED + 7099, 0, W07_VOCAB)


def _split_heads(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    seq_len, dim = x.shape
    head_dim = dim // W07_HEADS
    return np.ascontiguousarray(
        np.transpose(x.reshape(seq_len, W07_HEADS, head_dim), (1, 0, 2)),
        dtype=np.dtype("<f8"),
    )


def _merge_heads(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    heads, seq_len, head_dim = x.shape
    return np.ascontiguousarray(
        np.transpose(x, (1, 0, 2)).reshape(seq_len, heads * head_dim),
        dtype=np.dtype("<f8"),
    )


def forward_tokens(
    token_ids: npt.NDArray[np.int64],
    weights: dict[str, npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Logits of shape (seq, vocab) in float64."""
    x = weights["tok_emb"][np.ascontiguousarray(token_ids)]
    for layer in range(W07_LAYERS):
        prefix = f"layer{layer}_"
        n_x = rmsnorm_fp64(x, weights[prefix + "attn_scale"], eps=FP64_EPS)
        query = matmul_fp64(n_x, weights[prefix + "wq"])
        key = matmul_fp64(n_x, weights[prefix + "wk"])
        value = matmul_fp64(n_x, weights[prefix + "wv"])
        attn = sdpa_fp64(
            _split_heads(query),
            _split_heads(key),
            _split_heads(value),
            causal=True,
        )
        x = np.add(x, matmul_fp64(_merge_heads(attn), weights[prefix + "wo"]))
        n2 = rmsnorm_fp64(x, weights[prefix + "ffn_scale"], eps=FP64_EPS)
        hidden = gelu_erf_fp64(matmul_fp64(n2, weights[prefix + "w1"]))
        x = np.add(x, matmul_fp64(hidden, weights[prefix + "w2"]))
    x = rmsnorm_fp64(x, weights["final_scale"], eps=FP64_EPS)
    return matmul_fp64(x, weights["lm_head"])


def greedy_decode(
    weights: dict[str, npt.NDArray[np.float64]] | None = None,
) -> npt.NDArray[np.int64]:
    resolved = generate_weights() if weights is None else weights
    tokens = [int(t) for t in generate_prompt().tolist()]
    for _ in range(W07_DECODE_STEPS):
        ids = np.ascontiguousarray(np.array(tokens, dtype=np.dtype("<i8")))
        logits = forward_tokens(ids, resolved)
        tokens.append(int(np.argmax(logits[-1])))
    return np.ascontiguousarray(np.array(tokens, dtype=np.dtype("<i8")))
