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
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 0
