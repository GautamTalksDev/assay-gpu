"""W07 short decoder forward + greedy decode. Weights are in-repo, not downloaded."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch.nn import functional

from assay.ieee import FP32_EPS
from assay.reference.spec import W07_D_MODEL, W07_DECODE_STEPS, W07_HEADS, W07_LAYERS
from assay.reference.transformer import generate_prompt, generate_weights
from assay.workload.context import run_cuda_op
from assay.workload.report import WorkloadResult


def _to_gpu(array: npt.NDArray[np.generic]) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).to(
        device="cuda", dtype=torch.float32
    )


def _split_heads(x: torch.Tensor) -> torch.Tensor:
    seq_len, dim = x.shape
    head_dim = dim // W07_HEADS
    return x.view(seq_len, W07_HEADS, head_dim).transpose(0, 1).contiguous()


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    heads, seq_len, head_dim = x.shape
    return x.transpose(0, 1).contiguous().view(seq_len, heads * head_dim)


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    mean_square = x.pow(2).mean(dim=-1, keepdim=True)
    return x * weight / torch.sqrt(mean_square + FP32_EPS)


def _forward(token_ids: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    x = weights["tok_emb"][token_ids]
    for layer in range(W07_LAYERS):
        prefix = f"layer{layer}_"
        n_x = _rmsnorm(x, weights[prefix + "attn_scale"])
        query = n_x @ weights[prefix + "wq"]
        key = n_x @ weights[prefix + "wk"]
        value = n_x @ weights[prefix + "wv"]
        q_h = _split_heads(query).unsqueeze(0)
        k_h = _split_heads(key).unsqueeze(0)
        v_h = _split_heads(value).unsqueeze(0)
        attn = functional.scaled_dot_product_attention(
            q_h, k_h, v_h, dropout_p=0.0, is_causal=True
        )
        x = x + _merge_heads(attn.squeeze(0)) @ weights[prefix + "wo"]
        n2 = _rmsnorm(x, weights[prefix + "ffn_scale"])
        hidden = functional.gelu(n2 @ weights[prefix + "w1"], approximate="none")
        x = x + hidden @ weights[prefix + "w2"]
    x = _rmsnorm(x, weights["final_scale"])
    return x @ weights["lm_head"]


def run_w07() -> list[WorkloadResult]:
    np_weights = generate_weights()
    weights = {name: _to_gpu(array) for name, array in np_weights.items()}
    prompt = generate_prompt()

    def decode() -> torch.Tensor:
        tokens = [int(t) for t in prompt.tolist()]
        for _ in range(W07_DECODE_STEPS):
            ids = torch.tensor(tokens, device="cuda", dtype=torch.int64)
            logits = _forward(ids, weights)
            tokens.append(int(torch.argmax(logits[-1]).item()))
        return torch.tensor(tokens, device="cuda", dtype=torch.int64)

    return [
        run_cuda_op(
            decode,
            workload="W07",
            case=f"greedy_d{W07_D_MODEL}_layers{W07_LAYERS}",
            kernel_kind="transformer",
        )
    ]
