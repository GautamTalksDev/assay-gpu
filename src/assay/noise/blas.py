"""Force cuBLAS vs cuBLASLt when torch exposes preferred_blas_library."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress

import torch

_CANDIDATES: tuple[str, ...] = ("cublas", "cublaslt")


def available_blas_libraries() -> tuple[str, ...]:
    setter = getattr(torch.backends.cuda, "preferred_blas_library", None)
    if not callable(setter):
        return ("unavailable",)
    found: list[str] = []
    previous = None
    getter_ok = True
    try:
        previous = setter()
    except TypeError:
        getter_ok = False
    except RuntimeError:
        previous = None
    for name in _CANDIDATES:
        try:
            setter(name)
        except (RuntimeError, TypeError, ValueError):
            continue
        found.append(name)
    if getter_ok and previous is not None:
        with suppress(RuntimeError, TypeError, ValueError):
            setter(previous)
    return tuple(found) if found else ("default",)


@contextmanager
def blas_library(name: str) -> Iterator[None]:
    setter = getattr(torch.backends.cuda, "preferred_blas_library", None)
    if not callable(setter) or name in {"unavailable", "default"}:
        yield
        return
    previous = None
    try:
        previous = setter()
    except TypeError:
        previous = None
    except RuntimeError:
        previous = None
    setter(name)
    try:
        yield
    finally:
        if previous is not None:
            with suppress(RuntimeError, TypeError, ValueError):
                setter(previous)
