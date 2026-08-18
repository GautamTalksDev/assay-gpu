"""CPU tests for assay watch: hooks, sampling, faults never reach the host."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from typer.testing import CliRunner

from assay.abft.check import CheckResult, CheckStatus
from assay.cli import app
from assay.watch.session import WatchSession
from assay.watch.types import WatchConfig

pytestmark = pytest.mark.cpu

REPO = Path(__file__).resolve().parents[1]
NOISE = REPO / "data" / "noisefloor"
runner = CliRunner()


def _config(every: int, report: Path | None = None) -> WatchConfig:
    return WatchConfig(
        every=every,
        noisefloor_dir=NOISE,
        report_path=report,
        interval_seconds=None,
        gpu_model=None,
    )


def _fail_result() -> CheckResult:
    return CheckResult(
        status=CheckStatus.FAIL,
        residual=1.0,
        residual_hex="0x1.0p+0",
        residual_decimal="1",
        threshold=0.01,
        threshold_hex="0x1.47ae147ae147bp-7",
        threshold_decimal="0.01",
        sample_max=0.02,
        sample_max_hex="0x1.47ae147ae147bp-6",
        noisefloor_spec_version="noisefloor-v1",
        n_samples=100000,
        min_samples=100000,
        reason="residual exceeds every measured noisefloor sample",
        backend="pytorch",
        lookup_status="characterized",
        dtype_name="float32",
        shape=(4, 4, 4),
    )


def test_linear_forward_is_unchecked_host_output() -> None:
    layer = nn.Linear(4, 4)
    data = torch.ones(3, 4)
    with torch.no_grad():
        expected = layer(data)
    with WatchSession(_config(every=1)), torch.no_grad():
        got = layer(data)
    assert torch.equal(got, expected)


def test_uncharacterized_linear_is_inconclusive_not_fail() -> None:
    layer = nn.Linear(8, 8)
    data = torch.randn(2, 8)
    with WatchSession(_config(every=1)) as session:
        layer(data)
    assert session.gemm_seen == 1
    assert session.gemm_checked == 1
    assert session.events
    assert session.events[0].status is CheckStatus.INCONCLUSIVE
    assert session.overall_status() is CheckStatus.INCONCLUSIVE


def test_sampling_checks_one_in_n() -> None:
    layer = nn.Linear(4, 4)
    data = torch.ones(1, 4)
    with WatchSession(_config(every=3)) as session:
        for _ in range(6):
            layer(data)
    assert session.gemm_seen == 6
    assert session.gemm_checked == 2
    assert len(session.events) == 2


def test_failed_check_does_not_raise_into_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "assay.watch.session.check_gemm",
        lambda *_args, **_kwargs: _fail_result(),
    )
    layer = nn.Linear(4, 4)
    data = torch.ones(2, 4)
    with WatchSession(_config(every=1)) as session:
        out = layer(data)
    assert out.shape == (2, 4)
    assert session.events[0].status is CheckStatus.FAIL
    assert session.overall_status() is CheckStatus.FAIL


def test_check_exception_is_recorded_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_args: object, **_kwargs: object) -> CheckResult:
        msg = "checksum exploded"
        raise RuntimeError(msg)

    monkeypatch.setattr("assay.watch.session.check_gemm", boom)
    layer = nn.Linear(4, 4)
    data = torch.ones(2, 4)
    with WatchSession(_config(every=1)) as session:
        out = layer(data)
    assert out.shape == (2, 4)
    assert session.events[0].kind == "error"
    assert session.events[0].swallowed_exception is not None
    assert "checksum exploded" in session.events[0].swallowed_exception
    assert session.overall_status() is CheckStatus.INCONCLUSIVE


def test_attention_hook_is_inconclusive_and_silent() -> None:
    attention = nn.MultiheadAttention(8, 2, batch_first=True)
    data = torch.randn(2, 5, 8)
    with WatchSession(_config(every=1)) as session:
        out, _weights = attention(data, data, data)
    assert out.shape == (2, 5, 8)
    kinds = {event.kind for event in session.events}
    assert "attention" in kinds
    assert all(
        event.status is CheckStatus.INCONCLUSIVE
        for event in session.events
        if event.kind == "attention"
    )


def test_watch_cli_runs_script_without_rewriting_it(tmp_path: Path) -> None:
    script = tmp_path / "infer.py"
    script.write_text(
        "import torch\n"
        "from torch import nn\n"
        "layer = nn.Linear(4, 4)\n"
        "print(float(layer(torch.ones(2, 4)).detach().sum()))\n",
        encoding="utf-8",
    )
    log = tmp_path / "watch.json"
    result = runner.invoke(
        app,
        [
            "watch",
            "--every",
            "1",
            "--report",
            str(log),
            "--noisefloor-dir",
            str(NOISE),
            "--",
            str(script),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "KT-2 FAIL" in result.output
    assert log.is_file()
    text = log.read_text(encoding="utf-8")
    assert "assay-watch-v1" in text
    assert '"shipped": false' in text


def test_watch_help_requires_every_and_names_kt2() -> None:
    result = runner.invoke(app, ["watch", "--help"])
    assert result.exit_code == 0
    assert "--every" in result.output
    assert "KT-2" in result.output
    assert "ZERO network" in result.output
