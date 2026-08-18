#!/usr/bin/env python3
"""Aggregate archived attestation-v1 reports into an anonymized CSV.

Never emits a provider name, hostname, GPU UUID, DMI string, IP, or
public key. Naming a provider for an individual failure is forbidden:
the CSV column named_in_publication is always false.

Makes ZERO network requests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

_ISSUED_DAY = re.compile(r"^(\d{4}-\d{2}-\d{2})")
SPEC_ID = "attestation-v1"
FORBIDDEN_FIELDS = frozenset(
    {
        "provider",
        "hostname",
        "host",
        "uuid",
        "serial",
        "public_key",
        "public_key_hex",
        "dmi",
        "dmi_sys_vendor",
        "dmi_product_name",
        "ip",
        "instance_type",
        "instance",
        "region",
        "account",
    }
)
CSV_FIELDS = (
    "instance_index",
    "gpu_model",
    "status",
    "noisefloor_status",
    "budget_name",
    "n_cases",
    "n_pass",
    "n_fail",
    "n_inconclusive",
    "n_skipped",
    "n_fail_voting",
    "attempt",
    "reproduced",
    "unreproduced",
    "named_in_publication",
    "report_sha256",
    "issued_on_utc",
    "spec",
    "tool_version",
)


def _stderr(message: str) -> None:
    sys.stderr.write(message + "\n")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _count_status(cases: list[dict[str, Any]], wanted: str) -> int:
    return sum(1 for case in cases if str(case.get("status", "")) == wanted)


def _issued_on_utc(issued_at_utc: object) -> str:
    if not isinstance(issued_at_utc, str):
        return ""
    match = _ISSUED_DAY.match(issued_at_utc)
    return match.group(1) if match else ""


def row_from_report(path: Path) -> dict[str, str]:
    document = _load_json(path)
    if not isinstance(document, dict):
        msg = f"{path} is not a JSON object"
        raise ValueError(msg)
    spec = document.get("spec")
    if spec != SPEC_ID:
        msg = f"{path} spec is {spec!r}, want {SPEC_ID}"
        raise ValueError(msg)
    body = document.get("body")
    if not isinstance(body, dict):
        msg = f"{path} body is not an object"
        raise ValueError(msg)
    raw_cases = body.get("cases", [])
    if not isinstance(raw_cases, list):
        msg = f"{path} body.cases is not an array"
        raise ValueError(msg)
    cases = [case for case in raw_cases if isinstance(case, dict)]
    noisefloor = body.get("noisefloor")
    noisefloor_status = ""
    if isinstance(noisefloor, dict):
        noisefloor_status = str(noisefloor.get("status") or "")
    budget = body.get("budget")
    budget_name = ""
    if isinstance(budget, dict) and budget.get("name") is not None:
        budget_name = str(budget["name"])
    gpu_model = body.get("gpu_model")
    gpu_text = "" if gpu_model is None else str(gpu_model)
    n_fail_voting = sum(
        1
        for case in cases
        if str(case.get("status", "")) == "FAIL"
        and case.get("counts_toward_verdict") is True
        and case.get("skipped") is not True
    )
    return {
        "gpu_model": gpu_text,
        "status": str(body.get("status") or ""),
        "noisefloor_status": noisefloor_status,
        "budget_name": budget_name,
        "n_cases": str(len(cases)),
        "n_pass": str(_count_status(cases, "PASS")),
        "n_fail": str(_count_status(cases, "FAIL")),
        "n_inconclusive": str(_count_status(cases, "INCONCLUSIVE")),
        "n_skipped": str(sum(1 for case in cases if case.get("skipped") is True)),
        "n_fail_voting": str(n_fail_voting),
        "named_in_publication": "false",
        "report_sha256": _sha256_bytes(path),
        "issued_on_utc": _issued_on_utc(body.get("issued_at_utc")),
        "spec": str(spec),
        "tool_version": str(body.get("tool_version") or ""),
    }


def load_attempts(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    payload = _load_json(path)
    raw_attempts = payload.get("attempts", []) if isinstance(payload, dict) else payload
    if not isinstance(raw_attempts, list):
        msg = "attempts file must be a list or an object with key attempts"
        raise ValueError(msg)
    by_sha: dict[str, dict[str, str]] = {}
    for item in raw_attempts:
        if not isinstance(item, dict):
            continue
        sha = item.get("report_sha256")
        if not isinstance(sha, str) or not sha:
            continue
        attempt = item.get("attempt")
        reproduced = item.get("reproduced")
        unreproduced = ""
        if reproduced is False:
            unreproduced = "true"
        elif reproduced is True:
            unreproduced = "false"
        by_sha[sha] = {
            "attempt": "" if attempt is None else str(attempt),
            "reproduced": "" if reproduced is None else str(reproduced).lower(),
            "unreproduced": unreproduced,
        }
    return by_sha


def collect_rows(
    reports_dir: Path, attempts: dict[str, dict[str, str]]
) -> list[dict[str, str]]:
    paths = sorted(path for path in reports_dir.glob("*.json") if path.is_file())
    rows: list[dict[str, str]] = []
    for path in paths:
        row = row_from_report(path)
        extra = attempts.get(
            row["report_sha256"],
            {"attempt": "", "reproduced": "", "unreproduced": ""},
        )
        row.update(extra)
        rows.append(row)
    rows.sort(key=lambda row: (row["issued_on_utc"], row["report_sha256"]))
    numbered: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        out = {"instance_index": str(index), **row}
        numbered.append(out)
    return numbered


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    leaked = [name for name in CSV_FIELDS if name in FORBIDDEN_FIELDS]
    leaked.extend(name for row in rows for name in row if name in FORBIDDEN_FIELDS)
    if leaked:
        msg = f"anonymized CSV must not contain {sorted(set(leaked))}"
        raise ValueError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=CSV_FIELDS, extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in CSV_FIELDS})


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate archived attestation-v1 JSON into an anonymized CSV. "
            "Makes ZERO network requests."
        ),
        epilog=(
            "Never names a provider. named_in_publication is always false. "
            "ZERO network requests."
        ),
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        required=True,
        help="Directory of archived attestation-v1 JSON files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--attempts",
        type=Path,
        default=None,
        help="Optional JSON mapping report_sha256 to reproduction metadata.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reports_dir = args.reports_dir
    if not reports_dir.is_dir():
        _stderr(f"collect: reports dir not found: {reports_dir}")
        return 1
    try:
        attempts = load_attempts(args.attempts)
        rows = collect_rows(reports_dir, attempts)
        write_csv(args.out, rows)
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        _stderr(f"collect: {exc}")
        return 1
    _stderr(f"collect: n_reports={len(rows)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
