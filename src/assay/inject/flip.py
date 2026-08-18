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
) -> tuple[torch.Tensor, list[FlipLocation]]:
    """XOR `n_flips` distinct (element, bit) pairs inside `bit_class`.

    Same seed, tensor shape/dtype, n_flips, and class -> same locations
    and same payload, always. Sampling is without replacement.
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
    locations: list[FlipLocation] = []
    if n_flips == 0:
        return _from_uint(bits, tensor.dtype, tuple(tensor.shape), tensor.device), []
    picks = rng.choice(n_slots, size=n_flips, replace=False)
    n_class = len(class_bits)
    for slot in np.atleast_1d(picks):
        slot_i = int(slot)
        element_index = slot_i // n_class
        bit_index = class_bits[slot_i % n_class]
        _xor_one(bits, element_index, bit_index, layout.width)
        locations.append(FlipLocation(element_index, bit_index))
    return _from_uint(bits, tensor.dtype, tuple(tensor.shape), tensor.device), locations
