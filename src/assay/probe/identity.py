"""GPU identity from torch and nvidia-smi. Missing fields stay null."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class GpuIdentity:
    device_index: int
    torch_name: str | None
    smi_name: str | None
    uuid: str | None
    driver_version: str | None
    cuda_version: str | None
    torch_version: str


def _smi_fields(device_index: int) -> tuple[str | None, str | None, str | None]:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None, None, None
    query = "name,uuid,driver_version"
    completed = subprocess.run(
        [
            binary,
            f"--id={device_index}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None, None, None
    line = completed.stdout.strip().splitlines()
    if not line:
        return None, None, None
    parts = [part.strip() for part in line[0].split(",")]
    wanted = 3
    while len(parts) < wanted:
        parts.append("")

    def _clean(raw: str) -> str | None:
        if raw == "" or raw.upper() == "[N/A]":
            return None
        return raw

    return _clean(parts[0]), _clean(parts[1]), _clean(parts[2])


def read_identity(device_index: int) -> GpuIdentity:
    torch_name = None
    if torch.cuda.is_available() and device_index < torch.cuda.device_count():
        torch_name = torch.cuda.get_device_name(device_index)
    smi_name, uuid, driver = _smi_fields(device_index)
    return GpuIdentity(
        device_index=device_index,
        torch_name=torch_name,
        smi_name=smi_name,
        uuid=uuid,
        driver_version=driver,
        cuda_version=torch.version.cuda,
        torch_version=torch.__version__,
    )


def model_key(identity: GpuIdentity) -> str:
    name = identity.smi_name or identity.torch_name or "unknown-gpu"
    return name.replace(" ", "_").replace("/", "_")
