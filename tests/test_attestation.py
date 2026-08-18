"""CPU tests for attestation-v1 signing and offline verify."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
from typer.testing import CliRunner

from assay.abft.check import CheckStatus
from assay.cli import app
from assay.probe.environment import probe_environment
from assay.report.attestation import sign_document, write_attestation
from assay.report.build import build_body, compute_run_id, reference_catalog_sha256
from assay.report.constants import (
    ASSAY_SIGNED_NOTE,
    MODE_ASSAY_SIGNED,
    MODE_SELF_SIGNED,
    SELF_SIGNED_DISCLAIMER,
    SPEC_ID,
)
from assay.report.keys import generate_keypair, write_keypair
from assay.report.verify import VerifyOutcome, verify_document, verify_path
from assay.run.budget import budget_from_flags
from assay.run.types import AssayResult, CaseRecord, overall_status

pytestmark = pytest.mark.cpu

REPO = Path(__file__).resolve().parents[1]
NOISE = REPO / "data" / "noisefloor"
REFERENCE = REPO / "data" / "reference"
ISSUED = "2026-08-18T19:00:00.000000Z"
runner = CliRunner()


def _inconclusive_result() -> AssayResult:
    case = CaseRecord(
        workload="W01",
        case="m512_k512_n512",
        shape=(512, 512, 512),
        dtype_name="float32",
        status=CheckStatus.INCONCLUSIVE,
        reason="uncharacterized: n_samples=0",
        residual_hex="0x0.0p+0",
        residual_decimal="0",
        threshold_hex=None,
        threshold_decimal=None,
        sample_max_hex=None,
        n_samples=0,
        min_samples=100000,
        noisefloor_spec_version="noisefloor-v1",
        golden_max_abs_error_hex=None,
        wall_time_s=0.25,
        skipped=False,
        skip_reason=None,
        counts_toward_verdict=True,
    )
    cases = (case,)
    return AssayResult(
        probe=probe_environment(),
        gpu_model="TestGPU",
        noisefloor_status="uncharacterized",
        noisefloor_reason="n_samples=0",
        budget=budget_from_flags(quick=True, thorough=False),
        cases=cases,
        status=overall_status(cases),
        elapsed_s=1.25,
    )


def _write_signed(tmp_path: Path) -> tuple[Path, Path]:
    pair = generate_keypair()
    key_path = tmp_path / "operator.json"
    write_keypair(key_path, pair)
    report_path = tmp_path / "report.json"
    write_attestation(
        report_path,
        _inconclusive_result(),
        signing_key=pair,
        reference_dir=REFERENCE,
        noisefloor_dir=NOISE,
        issued_at_utc=ISSUED,
    )
    return report_path, key_path


def test_signed_round_trip_and_disclaimer(tmp_path: Path) -> None:
    report_path, key_path = _write_signed(tmp_path)
    document = json.loads(report_path.read_text(encoding="utf-8"))
    assert document["spec"] == SPEC_ID
    assert document["body"]["independence"]["mode"] == MODE_SELF_SIGNED
    assert document["body"]["independence"]["disclaimer"] == SELF_SIGNED_DISCLAIMER
    assert document["body"]["issued_at_utc"] == ISSUED
    assert document["body"]["network_requests"] == 0
    catalog = reference_catalog_sha256(REFERENCE)
    assert document["body"]["reference_catalog_sha256"] == catalog
    checked = verify_path(report_path, pubkey_path=key_path, reference_dir=REFERENCE)
    assert checked.outcome is VerifyOutcome.VALID
    assert SELF_SIGNED_DISCLAIMER in checked.warnings
    cli = runner.invoke(app, ["verify", str(report_path), "--pubkey", str(key_path)])
    assert cli.exit_code == 0
    assert "SELF-SIGNED" in cli.output
    assert "not independent evidence" in cli.output


def test_verify_detects_hand_edited_field(tmp_path: Path) -> None:
    report_path, _key_path = _write_signed(tmp_path)
    document = json.loads(report_path.read_text(encoding="utf-8"))
    document["body"]["status"] = "PASS"
    report_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    cli = runner.invoke(app, ["verify", str(report_path)])
    assert cli.exit_code == 1
    checked = verify_path(report_path)
    assert checked.outcome is VerifyOutcome.INVALID
    assert any("signature" in reason.lower() for reason in checked.reasons)


def test_verify_detects_inconsistent_verdict_even_if_re_signed(
    tmp_path: Path,
) -> None:
    pair = generate_keypair()
    body = build_body(
        _inconclusive_result(),
        reference_dir=REFERENCE,
        noisefloor_dir=NOISE,
        issued_at_utc=ISSUED,
    )
    body["status"] = "PASS"
    body["exit_code"] = 0
    body["run_id"] = compute_run_id(body)
    document = sign_document(body, pair)
    path = tmp_path / "lie.json"
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    checked = verify_path(path)
    assert checked.outcome is VerifyOutcome.INVALID
    assert any("does not match cases" in reason for reason in checked.reasons)


def test_verify_works_when_sockets_are_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path, _key = _write_signed(tmp_path)

    def blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    checked = verify_path(report_path)
    assert checked.outcome is VerifyOutcome.VALID


def test_keygen_and_wrong_pubkey(tmp_path: Path) -> None:
    out = tmp_path / "k.json"
    gen = runner.invoke(app, ["keygen", "--out", str(out)])
    assert gen.exit_code == 0
    assert "not independent evidence" in gen.output
    report_path, _orig = _write_signed(tmp_path)
    other = tmp_path / "other.json"
    runner.invoke(app, ["keygen", "--out", str(other)])
    cli = runner.invoke(app, ["verify", str(report_path), "--pubkey", str(other)])
    assert cli.exit_code == 1


def test_assay_signed_format_verifies_but_warns_unpinned() -> None:
    pair = generate_keypair()
    body = build_body(
        _inconclusive_result(),
        reference_dir=REFERENCE,
        noisefloor_dir=NOISE,
        issued_at_utc=ISSUED,
    )
    body["independence"] = {
        "mode": MODE_ASSAY_SIGNED,
        "disclaimer": ASSAY_SIGNED_NOTE,
    }
    body["run_id"] = compute_run_id(body)
    document = sign_document(body, pair)
    checked = verify_document(document)
    assert checked.outcome is VerifyOutcome.VALID
    assert checked.mode == MODE_ASSAY_SIGNED
    assert any("not pinned" in warning for warning in checked.warnings)


def test_v1_build_refuses_assay_signed_issuance() -> None:
    with pytest.raises(ValueError, match="cannot issue assay-signed"):
        build_body(
            _inconclusive_result(),
            reference_dir=REFERENCE,
            noisefloor_dir=NOISE,
            issued_at_utc=ISSUED,
            mode=MODE_ASSAY_SIGNED,
        )


def test_verify_help_is_offline() -> None:
    result = runner.invoke(app, ["verify", "--help"])
    assert result.exit_code == 0
    assert "ZERO network" in result.output
    assert "--pubkey" in result.output
