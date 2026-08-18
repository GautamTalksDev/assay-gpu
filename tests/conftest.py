"""Require every test to be marked cpu or gpu."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    unmarked = [
        item.nodeid
        for item in items
        if "cpu" not in item.keywords and "gpu" not in item.keywords
    ]
    if unmarked:
        joined = ", ".join(unmarked)
        msg = f"every test must be marked cpu or gpu: {joined}"
        raise pytest.UsageError(msg)
