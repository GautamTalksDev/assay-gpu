"""Clocks, power, temperature via nvidia-smi. Null if the query fails."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GpuTelemetry:
    sm_clock_mhz: str | None
    mem_clock_mhz: str | None
    power_w: str | None
    temperature_c: str | None
    query_ok: bool
    query_error: str | None


def _clean(raw: str) -> str | None:
    text = raw.strip()
    if text == "" or text.upper() == "[N/A]":
        return None
    return text


def read_telemetry(device_index: int) -> GpuTelemetry:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return GpuTelemetry(None, None, None, None, False, "nvidia-smi not on PATH")
    completed = subprocess.run(
        [
            binary,
            f"--id={device_index}",
            "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        err = completed.stderr.strip() or f"exit {completed.returncode}"
        return GpuTelemetry(None, None, None, None, False, err)
    line = completed.stdout.strip().splitlines()
    if not line:
        return GpuTelemetry(None, None, None, None, False, "empty nvidia-smi stdout")
    parts = [part.strip() for part in line[0].split(",")]
    wanted = 4
    while len(parts) < wanted:
        parts.append("")
    return GpuTelemetry(
        sm_clock_mhz=_clean(parts[0]),
        mem_clock_mhz=_clean(parts[1]),
        power_w=_clean(parts[2]),
        temperature_c=_clean(parts[3]),
        query_ok=True,
        query_error=None,
    )


def telemetry_dict(sample: GpuTelemetry) -> dict[str, Any]:
    return asdict(sample)
