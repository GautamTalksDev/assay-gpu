"""nvidia-smi queries. Local process only. Missing fields stay null."""

from __future__ import annotations

import shutil
import subprocess


def smi_bin() -> str | None:
    return shutil.which("nvidia-smi")


def smi_query(device_index: int, fields: str) -> tuple[str, ...] | None:
    """CSV query for one GPU. Returns None if the binary or field set fails."""
    binary = smi_bin()
    if binary is None:
        return None
    completed = subprocess.run(
        [
            binary,
            f"--id={device_index}",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    line = completed.stdout.strip().splitlines()
    if not line:
        return None
    parts = [part.strip() for part in line[0].split(",")]
    return tuple(parts)


def smi_clean(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = raw.strip()
    if text == "" or text.upper() == "[N/A]":
        return None
    return text


def smi_query_apps(device_index: int) -> tuple[tuple[str, str], ...]:
    """Running compute PIDs on this GPU. Empty if the query is unsupported."""
    binary = smi_bin()
    if binary is None:
        return ()
    completed = subprocess.run(
        [
            binary,
            f"--id={device_index}",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return ()
    rows: list[tuple[str, str]] = []
    for line in completed.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not parts or parts[0] == "":
            continue
        pid = parts[0]
        name = parts[1] if len(parts) > 1 else ""
        rows.append((pid, name))
    return tuple(rows)
