"""CPU tests for assay run: verdicts, rendering, CLI, isolation."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from assay.abft.check import CheckStatus
from assay.cli import app
from assay.probe.environment import probe_environment
from assay.reference.spec import CHARACTERIZATION_MAX_SIDE, WORKLOAD_GEMM_SHAPES
from assay.run.budget import (
    DEFAULT_SECONDS,
    QUICK_MAX_SIDE,
    QUICK_SECONDS,
    THOROUGH_SECONDS,
    budget_from_flags,
    gemm_shapes_for,
)
from assay.run.guarantee import NETWORK_GUARANTEE
from assay.run.render import fail_detail_lines, render_human, render_json
from assay.run.types import AssayResult, CaseRecord, ExitCode, overall_status

pytestmark = pytest.mark.cpu

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "assay"
runner = CliRunner()


def _case(  # noqa: PLR0913
    *,
    workload: str = "W01",
    case: str = "m512_k512_n512",
    shape: tuple[int, ...] = (512, 512, 512),
    status: CheckStatus = CheckStatus.INCONCLUSIVE,
    reason: str = "fixture",
    residual_hex: str | None = None,
    residual_decimal: str | None = None,
    threshold_hex: str | None = None,
    threshold_decimal: str | None = None,
    sample_max_hex: str | None = None,
    n_samples: int | None = None,
    min_samples: int | None = None,
    skipped: bool = False,
    counts_toward_verdict: bool = True,
) -> CaseRecord:
    return CaseRecord(
        workload=workload,
        case=case,
        shape=shape,
        dtype_name="float32",
        status=status,
        reason=reason,
        residual_hex=residual_hex,
        residual_decimal=residual_decimal,
        threshold_hex=threshold_hex,
        threshold_decimal=threshold_decimal,
        sample_max_hex=sample_max_hex,
        n_samples=n_samples,
        min_samples=min_samples,
        noisefloor_spec_version="noisefloor-v1",
        golden_max_abs_error_hex=None,
        wall_time_s=0.1,
        skipped=skipped,
        skip_reason="time budget" if skipped else None,
        counts_toward_verdict=counts_toward_verdict,
    )


def _result(
    cases: tuple[CaseRecord, ...],
    *,
    noisefloor_status: str = "uncharacterized",
    noisefloor_reason: str = "n_samples=0 < min_samples",
) -> AssayResult:
    return AssayResult(
        probe=probe_environment(),
        gpu_model="TestGPU",
        noisefloor_status=noisefloor_status,
        noisefloor_reason=noisefloor_reason,
        budget=budget_from_flags(quick=True, thorough=False),
        cases=cases,
        status=overall_status(cases),
        elapsed_s=1.25,
    )


def test_help_prints_zero_network_guarantee() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ZERO network" in result.output
    assert "telemetry" in result.output
    assert "phone-home" in result.output


def test_run_help_documents_flags_and_exit_codes() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--quick" in result.output
    assert "--thorough" in result.output
    assert "--json" in result.output
    assert "--report" in result.output
    assert "--signing-key" in result.output
    assert "0 PASS" in result.output
    assert "1 FAIL" in result.output
    assert "2 INCONCLUSIVE" in result.output
    assert "3 operational" in result.output
    assert "--double" not in result.output
    assert "ZERO network" in result.output


def test_run_without_cuda_is_operational_exit_3() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is visible; operational no-GPU path cannot be tested")
    result = runner.invoke(app, ["run", "--json"])
    assert result.exit_code == int(ExitCode.OPERATIONAL)
    payload = json.loads(result.stdout)
    assert payload["status"] == "OPERATIONAL"
    assert payload["exit_code"] == 3
    assert payload["network_requests"] == 0
    assert "CUDA" in payload["reason"] or "cuda" in payload["reason"].lower()


def test_quick_and_thorough_together_is_operational() -> None:
    result = runner.invoke(app, ["run", "--quick", "--thorough", "--json"])
    assert result.exit_code == int(ExitCode.OPERATIONAL)
    payload = json.loads(result.stdout)
    assert payload["status"] == "OPERATIONAL"


def test_budget_shape_filters() -> None:
    quick = budget_from_flags(quick=True, thorough=False)
    default = budget_from_flags(quick=False, thorough=False)
    thorough = budget_from_flags(quick=False, thorough=True)
    assert quick.seconds == QUICK_SECONDS
    assert default.seconds == DEFAULT_SECONDS
    assert thorough.seconds == THOROUGH_SECONDS
    assert max(max(shape) for shape in gemm_shapes_for(quick)) <= QUICK_MAX_SIDE
    assert all(
        max(shape) <= CHARACTERIZATION_MAX_SIDE for shape in gemm_shapes_for(default)
    )
    assert gemm_shapes_for(thorough) == WORKLOAD_GEMM_SHAPES


def test_overall_status_fail_wins_and_observe_only_does_not_block_pass() -> None:
    observe = _case(
        workload="W04",
        case="sdpa",
        status=CheckStatus.INCONCLUSIVE,
        counts_toward_verdict=False,
    )
    passed = _case(status=CheckStatus.PASS, reason="at or below quantile")
    assert overall_status((passed, observe)) is CheckStatus.PASS
    failed = _case(
        status=CheckStatus.FAIL,
        reason="residual exceeds every measured noisefloor sample",
        residual_hex="0x1p-1",
        threshold_hex="0x1p-4",
        n_samples=100000,
    )
    assert overall_status((failed, observe)) is CheckStatus.FAIL
    skipped = _case(status=CheckStatus.INCONCLUSIVE, skipped=True)
    assert overall_status((passed, skipped)) is CheckStatus.INCONCLUSIVE
    assert overall_status((observe,)) is CheckStatus.INCONCLUSIVE


def test_fail_output_names_workload_shape_residual_threshold_samples() -> None:
    case = _case(
        workload="W02",
        case="m4096_k4096_n4096",
        shape=(4096, 4096, 4096),
        status=CheckStatus.FAIL,
        reason="residual exceeds every measured noisefloor sample",
        residual_hex="0x1.0000000000000p-1",
        residual_decimal="0.5",
        threshold_hex="0x1.0000000000000p-10",
        threshold_decimal="0.0009765625",
        sample_max_hex="0x1.0000000000000p-8",
        n_samples=100000,
        min_samples=100000,
    )
    text = render_human(
        _result((case,), noisefloor_status="characterized", noisefloor_reason="ok"),
        color=False,
    )
    assert "VERDICT  FAIL" in text
    assert "FAIL DETAIL" in text
    detail = "\n".join(fail_detail_lines(case))
    assert "W02" in detail
    assert "4096x4096x4096" in detail
    assert "0x1.0000000000000p-1" in detail
    assert "0x1.0000000000000p-10" in detail
    assert "100000" in detail
    assert text.strip() != "FAIL"
    payload = json.loads(
        render_json(
            _result((case,), noisefloor_status="characterized", noisefloor_reason="ok")
        )
    )
    assert payload["exit_code"] == int(ExitCode.FAIL)
    assert payload["cases"][0]["n_samples"] == 100000


def test_uncharacterized_human_output_is_inconclusive_not_a_guess() -> None:
    case = _case(
        reason="uncharacterized: n_samples=0",
        n_samples=0,
        min_samples=100000,
        residual_hex="0x0.0p+0",
        residual_decimal="0",
    )
    text = render_human(_result((case,)), color=False)
    assert "UNCHARACTERIZED" in text
    assert "will not guess PASS or FAIL" in text
    assert "VERDICT  INCONCLUSIVE" in text
    assert "exit 2" in text
    assert NETWORK_GUARANTEE in text


def test_probe_environment_does_not_require_cuda() -> None:
    probe = probe_environment()
    assert probe.gpu_count >= 0
    assert isinstance(probe.appears_virtualized, bool)
    assert isinstance(probe.appears_shared, bool)


def test_src_does_not_import_http_clients() -> None:
    roots = {"urllib", "requests", "httpx", "aiohttp"}
    hits: list[str] = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.append(node.module)
        hits.extend(
            f"{path.name}:{name}"
            for name in imported
            if name.split(".")[0] in roots or name == "http.client"
        )
    assert hits == []
