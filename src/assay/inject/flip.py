"""Seeded bit flips. TEST FIXTURE ONLY. Not imported by assay run."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from assay.inject.bits import BitClass, layout_for
from assay.reference.arrays import make_generator

_U8 = np.dtype(np.uint8)
_U16 = np.dtype(np.uint16)
_U32 = np.dtype(np.uint32)


@dataclass(frozen=True, slots=True)
class FlipLocation:
    element_index: int
    bit_index: int


@dataclass(frozen=True, slots=True)
class InjectionVerify:
    """Per-injection bit-survival audit. Emitted only when verify=True."""

    n_elements_flipped: int
    n_elements_bitwise_equal: int
    achieved_rel_delta_max: float
    achieved_rel_delta_median: float
    pre_bits: tuple[int, ...]
    post_bits: tuple[int, ...]
    element_indices: tuple[int, ...]


def _as_uint(tensor: torch.Tensor) -> npt.NDArray[np.unsignedinteger]:
    contiguous = tensor.detach().contiguous().cpu()
    if contiguous.dtype == torch.float32:
        return contiguous.view(torch.int32).numpy().view(_U32).copy()
    if contiguous.dtype == torch.float16:
        return contiguous.view(torch.int16).numpy().view(_U16).copy()
    if contiguous.dtype == torch.bfloat16:
        return contiguous.view(torch.int16).numpy().view(_U16).copy()
    if contiguous.dtype == torch.int8:
        return contiguous.numpy().view(_U8).copy()
    msg = f"unsupported injector dtype: {contiguous.dtype}"
    raise ValueError(msg)


def _from_uint(
    bits: npt.NDArray[np.unsignedinteger],
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    flat_shape = shape
    if dtype == torch.float32:
        array = np.ascontiguousarray(bits.reshape(flat_shape).view(np.float32))
        return torch.from_numpy(array).to(device=device)
    if dtype == torch.float16:
        array = np.ascontiguousarray(bits.reshape(flat_shape).view(np.float16))
        return torch.from_numpy(array).to(device=device)
    if dtype == torch.bfloat16:
        as_i16 = np.ascontiguousarray(bits.reshape(flat_shape).view(np.uint16)).view(
            np.int16
        )
        return (
            torch.from_numpy(np.ascontiguousarray(as_i16.copy()))
            .view(torch.bfloat16)
            .to(device=device)
        )
    if dtype == torch.int8:
        array = np.ascontiguousarray(bits.reshape(flat_shape).view(np.int8))
        return torch.from_numpy(array).to(device=device)
    msg = f"unsupported injector dtype: {dtype}"
    raise ValueError(msg)


def _xor_one(
    bits: npt.NDArray[np.unsignedinteger],
    element_index: int,
    bit_index: int,
    width: int,
) -> None:
    n_elem = int(np.prod(bits.shape, dtype=np.int64))
    if element_index < 0 or element_index >= n_elem:
        msg = f"element_index {element_index} out of range for {n_elem} elements"
        raise ValueError(msg)
    if bit_index < 0 or bit_index >= width:
        msg = f"bit_index {bit_index} out of range for width {width}"
        raise ValueError(msg)
    flat = bits.reshape(-1)
    mask = bits.dtype.type(1) << bits.dtype.type(bit_index)
    flat[element_index] = np.bitwise_xor(flat[element_index], mask)


def _uint_scalar_to_float(bit_pattern: int, dtype: torch.dtype) -> float:
    """Interpret a raw IEEE bit pattern as a Python float (via fp32)."""
    if dtype == torch.float32:
        return float(np.array([np.uint32(bit_pattern)], dtype=np.uint32).view(np.float32)[0])
    if dtype == torch.float16:
        return float(np.array([np.uint16(bit_pattern)], dtype=np.uint16).view(np.float16)[0])
    if dtype == torch.bfloat16:
        as_i16 = np.array([np.int16(np.uint16(bit_pattern))], dtype=np.int16)
        return float(torch.from_numpy(as_i16).view(torch.bfloat16).to(torch.float32).item())
    if dtype == torch.int8:
        return float(np.int8(np.uint8(bit_pattern)))
    msg = f"unsupported injector dtype: {dtype}"
    raise ValueError(msg)


def _rel_deltas(
    pre_bits: list[int],
    post_bits: list[int],
    dtype: torch.dtype,
) -> list[float]:
    deltas: list[float] = []
    for old_bits, new_bits in zip(pre_bits, post_bits, strict=True):
        old = _uint_scalar_to_float(old_bits, dtype)
        new = _uint_scalar_to_float(new_bits, dtype)
        if old == 0.0:
            deltas.append(0.0)
        else:
            deltas.append(abs(new - old) / abs(old))
    return deltas


def _median(ordered: list[float]) -> float:
    count = len(ordered)
    if count < 1:
        return 0.0
    if count % 2 == 1:
        return ordered[count // 2]
    return 0.5 * (ordered[count // 2 - 1] + ordered[count // 2])


def _build_injection_verify(
    *,
    pre_flat: npt.NDArray[np.unsignedinteger],
    post_tensor: torch.Tensor,
    element_indices: list[int],
    dtype: torch.dtype,
) -> InjectionVerify:
    """Compare pre-flip vs post-cast bit patterns for each targeted element."""
    unique = sorted(set(element_indices))
    post_flat = _as_uint(post_tensor).reshape(-1)
    pre_bits = [int(pre_flat.reshape(-1)[i]) for i in unique]
    post_bits = [int(post_flat[i]) for i in unique]
    n_equal = sum(1 for a, b in zip(pre_bits, post_bits, strict=True) if a == b)
    deltas = _rel_deltas(pre_bits, post_bits, dtype)
    ordered = sorted(deltas)
    return InjectionVerify(
        n_elements_flipped=len(unique),
        n_elements_bitwise_equal=n_equal,
        achieved_rel_delta_max=ordered[-1] if ordered else 0.0,
        achieved_rel_delta_median=_median(ordered),
        pre_bits=tuple(pre_bits),
        post_bits=tuple(post_bits),
        element_indices=tuple(unique),
    )


def flip(
    tensor: torch.Tensor,
    bit_index: int,
    element_index: int,
    seed: int,
) -> torch.Tensor:
    """XOR one bit of one element. Does not mutate `tensor`.

    `seed` binds a PCG64 stream (same identity as flip_random) but does not
    choose the bit: `bit_index` and `element_index` fully specify the flip.
    """
    rng = make_generator(seed)
    _ = rng.bit_generator.state
    layout = layout_for(tensor.dtype)
    bits = _as_uint(tensor)
    _xor_one(bits, element_index, bit_index, layout.width)
    return _from_uint(bits, tensor.dtype, tuple(tensor.shape), tensor.device)


def flip_random(
    tensor: torch.Tensor,
    n_flips: int,
    bit_class: BitClass,
    seed: int,
    *,
    verify: bool = False,
) -> (
    tuple[torch.Tensor, list[FlipLocation]]
    | tuple[torch.Tensor, list[FlipLocation], InjectionVerify]
):
    """XOR `n_flips` distinct (element, bit) pairs inside `bit_class`.

    Same seed, tensor shape/dtype, n_flips, and class -> same locations
    and same payload, always. Sampling is without replacement.

    When `verify` is True, also return InjectionVerify comparing pre-flip
    and post-cast raw bit patterns of every perturbed element. Default off
    keeps the return value and numerics identical to the pre-verify path.
    """
    if n_flips < 0:
        msg = "n_flips must be >= 0"
        raise ValueError(msg)
    layout = layout_for(tensor.dtype)
    class_bits = layout.bits_for(bit_class)
    n_elem = int(tensor.numel())
    n_slots = n_elem * len(class_bits)
    if n_flips > n_slots:
        msg = f"n_flips={n_flips} exceeds {n_slots} slots in {bit_class.value}"
        raise ValueError(msg)
    rng = make_generator(seed)
    bits = _as_uint(tensor)
    pre_flat = bits.reshape(-1).copy() if verify else None
    locations: list[FlipLocation] = []
    if n_flips == 0:
        out = _from_uint(bits, tensor.dtype, tuple(tensor.shape), tensor.device)
        if not verify:
            return out, []
        assert pre_flat is not None
        stats = _build_injection_verify(
            pre_flat=pre_flat,
            post_tensor=out,
            element_indices=[],
            dtype=tensor.dtype,
        )
        return out, [], stats
    picks = rng.choice(n_slots, size=n_flips, replace=False)
    n_class = len(class_bits)
    for slot in np.atleast_1d(picks):
        slot_i = int(slot)
        element_index = slot_i // n_class
        bit_index = class_bits[slot_i % n_class]
        _xor_one(bits, element_index, bit_index, layout.width)
        locations.append(FlipLocation(element_index, bit_index))
    out = _from_uint(bits, tensor.dtype, tuple(tensor.shape), tensor.device)
    if not verify:
        return out, locations
    assert pre_flat is not None
    stats = _build_injection_verify(
        pre_flat=pre_flat,
        post_tensor=out,
        element_indices=[loc.element_index for loc in locations],
        dtype=tensor.dtype,
    )
    return out, locations, stats
