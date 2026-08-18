"""CPU tests for survey preflight (HARD_CEILING) and anonymized collect."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from assay.abft.check import CheckStatus
from assay.probe.environment import probe_environment
from assay.report.attestation import write_attestation
from assay.report.keys import generate_keypair
from assay.run.budget import budget_from_flags
from assay.run.types import AssayResult, CaseRecord, overall_status

pytestmark = pytest.mark.cpu

REPO = Path(__file__).resolve().parents[1]
PREFLIGHT = REPO / "scripts" / "survey" / "preflight.sh"
COLLECT = REPO / "scripts" / "survey" / "collect.py"
NOISE = REPO / "data" / "noisefloor"
REFERENCE = REPO / "data" / "reference"
LEDGER_HEADER = (
    "| date | provider | instance type | GPU model | credit or cash | "
    "amount spent | cumulative total | verdict | attempt | report path |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
)


def _ledger(tmp_path: Path, rows: str) -> Path:
    path = tmp_path / "BUDGET.md"
    path.write_text("# ledger\n\n" + LEDGER_HEADER + rows, encoding="utf-8")
    return path


def _preflight(ledger: Path, ceiling: str = "75") -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["BUDGET_MD"] = str(ledger)
    env["HARD_CEILING"] = ceiling
    return subprocess.run(
        ["bash", str(PREFLIGHT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _collect(
    reports_dir: Path,
    out: Path,
    attempts: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    argv = [
        sys.executable,
        str(COLLECT),
        "--reports-dir",
        str(reports_dir),
        "--out",
        str(out),
    ]
    if attempts is not None:
        argv.extend(["--attempts", str(attempts)])
    return subprocess.run(argv, check=False, capture_output=True, text=True)


def _case(*, status: CheckStatus, gpu_note: str) -> CaseRecord:
    return CaseRecord(
        workload="W01",
        case=gpu_note,
        shape=(512, 512, 512),
        dtype_name="float32",
        status=status,
        reason="fixture",
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


def _result(*, status: CheckStatus, gpu_model: str, case_name: str) -> AssayResult:
    cases = (_case(status=status, gpu_note=case_name),)
    return AssayResult(
        probe=probe_environment(),
        gpu_model=gpu_model,
        noisefloor_status="uncharacterized",
        noisefloor_reason="n_samples=0",
        budget=budget_from_flags(quick=False, thorough=True),
        cases=cases,
        status=overall_status(cases),
        elapsed_s=1.25,
    )


def _write_report(
    path: Path, *, status: CheckStatus, gpu_model: str, issued: str, case_name: str
) -> Path:
    write_attestation(
        path,
        _result(status=status, gpu_model=gpu_model, case_name=case_name),
        signing_key=generate_keypair(),
        reference_dir=REFERENCE,
        noisefloor_dir=NOISE,
        issued_at_utc=issued,
    )
    return path


def test_preflight_ok_on_empty_repo_ledger() -> None:
    result = _preflight(REPO / "docs" / "BUDGET.md")
    assert result.returncode == 0
    assert "cumulative_spend_usd=0.00" in result.stdout
    assert "hard_ceiling_usd=75.00" in result.stdout
    assert "remaining_usd=75.00" in result.stdout
    assert "preflight: OK" in result.stdout


def test_preflight_refuses_at_ceiling(tmp_path: Path) -> None:
    ledger = _ledger(
        tmp_path,
        "| 2026-08-18 | RunPod | cheap | T4 | credit | $75.00 | $75.00 | "
        "INCONCLUSIVE | 1 | data/survey/reports/a.json |\n",
    )
    result = _preflight(ledger, ceiling="75")
    assert result.returncode == 1
    assert "REFUSE" in result.stderr
    assert "survey is over" in result.stderr
    assert "Do not rent" in result.stderr
    assert "cumulative_spend_usd=75.00" in result.stdout


def test_preflight_allows_spend_just_under_ceiling(tmp_path: Path) -> None:
    ledger = _ledger(
        tmp_path,
        "| 2026-08-18 | Vast.ai | spot | RTX_3090 | cash | $74.99 | $74.99 | "
        "INCONCLUSIVE | 1 | data/survey/reports/b.json |\n",
    )
    result = _preflight(ledger, ceiling="75")
    assert result.returncode == 0
    assert "remaining_usd=0.01" in result.stdout
    assert "preflight: OK" in result.stdout


def test_preflight_refuses_unreconciled_cumulative(tmp_path: Path) -> None:
    ledger = _ledger(
        tmp_path,
        "| 2026-08-18 | Paperspace | gpu | A10 | cash | $3.00 | $9.00 | "
        "INCONCLUSIVE | 1 | data/survey/reports/c.json |\n",
    )
    result = _preflight(ledger, ceiling="75")
    assert result.returncode == 1
    assert "not reconciled" in result.stderr


def test_preflight_sums_two_rows(tmp_path: Path) -> None:
    ledger = _ledger(
        tmp_path,
        "| 2026-08-18 | TensorDock | spot | T4 | cash | $2.00 | $2.00 | "
        "INCONCLUSIVE | 1 | a.json |\n"
        "| 2026-08-19 | TensorDock | spot | T4 | cash | $3.00 | $5.00 | "
        "INCONCLUSIVE | 2 | b.json |\n",
    )
    result = _preflight(ledger, ceiling="75")
    assert result.returncode == 0
    assert "rows=2" in result.stdout
    assert "cumulative_spend_usd=5.00" in result.stdout
    assert "remaining_usd=70.00" in result.stdout


def test_preflight_override_ceiling_zero_refuses_empty(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, "")
    result = _preflight(ledger, ceiling="0")
    assert result.returncode == 1
    assert "at or past HARD_CEILING" in result.stderr


def test_collect_empty_reports_writes_header_only(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    out = tmp_path / "summary.csv"
    result = _collect(reports, out)
    assert result.returncode == 0
    assert "n_reports=0" in result.stderr
    text = out.read_text(encoding="utf-8")
    assert "provider" not in text
    rows = list(csv.DictReader(text.splitlines()))
    assert rows == []
    header = text.splitlines()[0]
    assert "gpu_model" in header
    assert "named_in_publication" in header


def test_collect_anonymizes_and_never_names_provider(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    fail_path = reports / "RunPod-secret-host.json"
    _write_report(
        fail_path,
        status=CheckStatus.FAIL,
        gpu_model="NVIDIA_A100-SXM4-40GB",
        issued="2026-08-18T12:00:00.000000Z",
        case_name="fail_case",
    )
    _write_report(
        reports / "second.json",
        status=CheckStatus.INCONCLUSIVE,
        gpu_model="Tesla_T4",
        issued="2026-08-19T12:00:00.000000Z",
        case_name="inc_case",
    )
    sha = hashlib.sha256(fail_path.read_bytes()).hexdigest()
    attempts = tmp_path / "attempts.json"
    attempts.write_text(
        json.dumps(
            {"attempts": [{"report_sha256": sha, "attempt": 2, "reproduced": False}]}
        ),
        encoding="utf-8",
    )
    out = tmp_path / "summary.csv"
    result = _collect(reports, out, attempts=attempts)
    assert result.returncode == 0
    text = out.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "runpod" not in lowered
    assert "provider" not in lowered
    assert "public_key" not in lowered
    assert "uuid" not in lowered
    rows = list(csv.DictReader(text.splitlines()))
    assert len(rows) == 2
    assert rows[0]["gpu_model"] == "NVIDIA_A100-SXM4-40GB"
    assert rows[0]["status"] == "FAIL"
    assert rows[0]["attempt"] == "2"
    assert rows[0]["unreproduced"] == "true"
    assert rows[0]["named_in_publication"] == "false"
    assert rows[1]["gpu_model"] == "Tesla_T4"
    assert rows[1]["named_in_publication"] == "false"
    assert rows[1]["budget_name"] == "thorough"
    for row in rows:
        assert row["named_in_publication"] == "false"


def test_collect_help_is_offline() -> None:
    result = subprocess.run(
        [sys.executable, str(COLLECT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "ZERO network" in result.stdout
    assert "Never names a provider" in result.stdout


def test_ideas_has_at_least_ten_unbuilt_entries() -> None:
    text = (REPO / "IDEAS.md").read_text(encoding="utf-8")
    entries = [line for line in text.splitlines() if line.split(".", 1)[0].isdigit()]
    assert len(entries) >= 10


def _h2(text: str) -> list[str]:
    return [line[3:].strip() for line in text.splitlines() if line.startswith("## ")]


def _section(text: str, title: str) -> str:
    headings = _h2(text)
    start = text.find(f"## {title}")
    assert start != -1, title
    idx = headings.index(title)
    if idx + 1 < len(headings):
        end = text.find(f"## {headings[idx + 1]}")
        return text[start:end]
    return text[start:]


def test_survey_2026_names_policy_and_limitations_before_findings() -> None:
    text = (REPO / "docs" / "SURVEY-2026.md").read_text(encoding="utf-8")
    headings = _h2(text)
    assert headings[0] == "Naming policy"
    assert headings.index("Limitations") < headings.index("Aggregate findings")
    assert headings.index("Naming policy") < headings.index("Aggregate findings")
    assert headings.index("Limitations") < headings.index("Anomalies")
    assert "Methodology critique" in headings


def test_survey_2026_limitations_are_the_stated_ones() -> None:
    text = (REPO / "docs" / "SURVEY-2026.md").read_text(encoding="utf-8")
    limits = _section(text, "Limitations")
    findings = _section(text, "Aggregate findings")
    assert "n=30 supports descriptive statistics, not inferential ones." in limits
    assert "provider A is less reliable than provider B" in limits
    assert "Short spot runs miss time-dependent faults." in limits
    assert "Cheapest-tier bias." in limits
    assert "Signup-credit instances may not be representative." in limits
    assert "provider A is less reliable than provider B" not in findings
    assert "We assayed 0 rented GPU instances" in findings
    assert "We assayed 30 rented GPU instances" not in findings


def test_survey_2026_anomalies_never_name_a_provider() -> None:
    anomalies = _section(
        (REPO / "docs" / "SURVEY-2026.md").read_text(encoding="utf-8"),
        "Anomalies",
    )
    assert "named_provider" in anomalies
    assert "never" in anomalies
    lowered = anomalies.lower()
    for brand in ("runpod", "vast.ai", "lambda", "coreweave", "paperspace"):
        assert brand not in lowered


def test_dataset_release_is_cc_by_4_and_anonymized() -> None:
    license_text = (REPO / "data" / "survey" / "LICENSE").read_text(encoding="utf-8")
    needle = "Creative Commons Attribution 4.0 International Public License"
    assert needle in license_text
    csv_path = REPO / "data" / "survey" / "summary.csv"
    text = csv_path.read_text(encoding="utf-8")
    rows = list(csv.DictReader(text.splitlines()))
    assert rows == []
    header = text.splitlines()[0]
    assert "provider" not in header
    assert "named_in_publication" in header
    readme = (REPO / "data" / "survey" / "README.md").read_text(encoding="utf-8")
    assert "CC-BY-4.0" in readme or "CC BY 4.0" in readme
    assert "zero" in readme.lower()
