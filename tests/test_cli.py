"""CPU tests for the CLI skeleton."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from assay.cli import app

pytestmark = pytest.mark.cpu

runner = CliRunner()


def test_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_run_subcommand_exists() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--quick" in result.output
    assert "--json" in result.output


def test_characterize_lookup_help() -> None:
    result = runner.invoke(app, ["characterize", "--help"])
    assert result.exit_code == 0
    assert "repeats" in result.output


def test_characterize_lookup_without_gpu_is_inconclusive() -> None:
    result = runner.invoke(
        app,
        [
            "characterize",
            "--lookup",
            "--workload",
            "W02",
            "--dtype",
            "bfloat16",
            "--m",
            "4096",
            "--k",
            "4096",
            "--n",
            "4096",
        ],
    )
    assert result.exit_code == 0
    assert "INCONCLUSIVE" in result.output
    assert "n_samples=0" in result.output


def test_abft_overhead_help() -> None:
    result = runner.invoke(app, ["abft", "overhead", "--help"])
    assert result.exit_code == 0
    assert "repeats" in result.output
