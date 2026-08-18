#!/usr/bin/env bash
# Real budget gate. Run before every rental. Exit 1 refuses; do not rent.
#
#   ./scripts/survey/preflight.sh
#   HARD_CEILING=75 BUDGET_MD=docs/BUDGET.md ./scripts/survey/preflight.sh
#
# Default HARD_CEILING=75 (USD, credit or cash, all providers combined).
# This is the check, not a comment: a spend at or past the ceiling exits 1.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUDGET_MD="${BUDGET_MD:-${ROOT}/docs/BUDGET.md}"
HARD_CEILING="${HARD_CEILING:-75}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "preflight: REFUSE python3 is required to parse the ledger" >&2
  exit 1
fi

export BUDGET_MD
export HARD_CEILING

python3 - "$BUDGET_MD" "$HARD_CEILING" <<'PY'
from __future__ import annotations

import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

LEDGER_PATH = Path(sys.argv[1])
CEILING_RAW = sys.argv[2]
MONEY = re.compile(r"^\$?(0|[1-9][0-9]*)(\.[0-9]{1,2})?$")


def fail(message: str) -> None:
    sys.stderr.write(f"preflight: REFUSE {message}\n")
    sys.stderr.write(
        "preflight: survey is over. Publish what you have. Do not rent.\n"
    )
    raise SystemExit(1)


def parse_usd(raw: str, *, field: str) -> Decimal:
    text = raw.strip()
    if not MONEY.fullmatch(text):
        fail(f"{field} is not a USD amount to the cent: {raw!r}")
    try:
        return Decimal(text.removeprefix("$"))
    except InvalidOperation:
        fail(f"{field} is not a USD amount: {raw!r}")
        raise AssertionError("unreachable") from None


def cells_of(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return None
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def is_separator(cells: list[str]) -> bool:
    if not cells:
        return False
    return all(cell != "" and set(cell) <= set("-:") for cell in cells)


def load_rows(text: str) -> tuple[list[str], list[dict[str, str]]]:
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        cells = cells_of(line)
        if cells is None:
            continue
        if header is None:
            header = [cell.lower() for cell in cells]
            continue
        if is_separator(cells):
            continue
        if len(cells) != len(header):
            fail(
                f"ledger row has {len(cells)} cells, header has {len(header)}"
            )
        rows.append(dict(zip(header, cells, strict=True)))
    if header is None:
        fail("docs/BUDGET.md has no markdown table")
    return header, rows


def required(row: dict[str, str], key: str) -> str:
    if key not in row:
        fail(f"ledger table is missing column {key!r}")
    return row[key]


def main() -> None:
    try:
        ceiling = parse_usd(CEILING_RAW, field="HARD_CEILING")
    except SystemExit:
        raise
    except Exception:
        fail(f"HARD_CEILING is not a USD amount: {CEILING_RAW!r}")

    if not LEDGER_PATH.is_file():
        fail(f"ledger not found: {LEDGER_PATH}")

    header, rows = load_rows(LEDGER_PATH.read_text(encoding="utf-8"))
    needed = ("amount spent", "cumulative total")
    missing = [name for name in needed if name not in header]
    if missing:
        fail(f"ledger table missing columns: {', '.join(missing)}")

    spent = Decimal("0.00")
    last_cumulative: Decimal | None = None
    for index, row in enumerate(rows, start=1):
        amount = parse_usd(required(row, "amount spent"), field=f"amount spent row {index}")
        cumulative = parse_usd(
            required(row, "cumulative total"),
            field=f"cumulative total row {index}",
        )
        spent += amount
        if cumulative != spent:
            fail(
                f"row {index} cumulative total ${cumulative} "
                f"!= sum of amount spent ${spent} (ledger not reconciled)"
            )
        last_cumulative = cumulative

    if last_cumulative is None:
        last_cumulative = Decimal("0.00")
    if last_cumulative != spent:
        fail("last cumulative total does not match sum of amount spent")

    remaining = ceiling - spent
    sys.stdout.write(f"preflight: ledger={LEDGER_PATH}\n")
    sys.stdout.write(f"preflight: rows={len(rows)}\n")
    sys.stdout.write(f"preflight: cumulative_spend_usd={spent:.2f}\n")
    sys.stdout.write(f"preflight: hard_ceiling_usd={ceiling:.2f}\n")
    if spent >= ceiling:
        over = spent - ceiling
        sys.stdout.write("preflight: remaining_usd=0.00\n")
        sys.stdout.write(f"preflight: over_by_usd={over:.2f}\n")
        fail(
            f"cumulative spend ${spent:.2f} is at or past HARD_CEILING "
            f"${ceiling:.2f}"
        )
    sys.stdout.write(f"preflight: remaining_usd={remaining:.2f}\n")
    sys.stdout.write("preflight: OK\n")


if __name__ == "__main__":
    main()
PY
