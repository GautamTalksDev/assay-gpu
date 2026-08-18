"""GPU test for assay run. Uncharacterized noisefloor must be INCONCLUSIVE."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from assay.cli import app
from assay.run.types import ExitCode

pytestmark = pytest.mark.gpu

REPO = Path(__file__).resolve().parents[1]
runner = CliRunner()


def test_run_quick_json_uncharacterized_is_exit_2() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--quick",
            "--json",
            "--noisefloor-dir",
            str(REPO / "data" / "noisefloor"),
            "--reference-dir",
            str(REPO / "data" / "reference"),
        ],
    )
    assert result.exit_code == int(ExitCode.INCONCLUSIVE), result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "INCONCLUSIVE"
    assert payload["noisefloor"]["status"] == "uncharacterized"
    assert payload["network_requests"] == 0
    assert payload["cases"]
