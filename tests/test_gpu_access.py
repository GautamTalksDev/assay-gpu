"""GPU-access tests. Run with: pytest -m gpu."""

from __future__ import annotations

import subprocess

import pytest

pytestmark = pytest.mark.gpu


def test_nvidia_smi_lists_a_gpu() -> None:
    completed = subprocess.run(
        ["nvidia-smi", "-L"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "GPU 0:" in completed.stdout
