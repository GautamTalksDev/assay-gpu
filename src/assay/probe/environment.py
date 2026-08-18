"""Local environment probe. Never opens a network socket."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from assay.probe.identity import GpuIdentity, model_key, read_identity
from assay.probe.smi import smi_clean, smi_query, smi_query_apps
from assay.probe.telemetry import GpuTelemetry, read_telemetry

_DMI_VENDOR = Path("/sys/class/dmi/id/sys_vendor")
_DMI_PRODUCT = Path("/sys/class/dmi/id/product_name")
_CPUINFO = Path("/proc/cpuinfo")


@dataclass(frozen=True, slots=True)
class DeviceProbe:
    identity: GpuIdentity
    model_key: str
    vbios: str | None
    ecc_mode: str | None
    ecc_uncorrected_aggregate: str | None
    mig_mode: str | None
    sm_clock_mhz: str | None
    mem_clock_mhz: str | None
    sm_clock_max_mhz: str | None
    mem_clock_max_mhz: str | None
    power_w: str | None
    power_limit_w: str | None
    temperature_c: str | None
    compute_apps: tuple[tuple[str, str], ...]
    telemetry: GpuTelemetry


@dataclass(frozen=True, slots=True)
class EnvironmentProbe:
    cuda_available: bool
    gpu_count: int
    torch_version: str
    cuda_version: str | None
    hip_version: str | None
    driver_version: str | None
    dmi_sys_vendor: str | None
    dmi_product_name: str | None
    cpu_hypervisor_flag: bool
    appears_virtualized: bool
    appears_shared: bool
    shared_reasons: tuple[str, ...]
    devices: tuple[DeviceProbe, ...]


def _read_text(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def _cpu_hypervisor_flag() -> bool:
    text = _read_text(_CPUINFO)
    if text is None:
        return False
    for line in text.splitlines():
        if line.startswith("flags") and "hypervisor" in line.split():
            return True
    return False


def _one_device(index: int) -> DeviceProbe:
    identity = read_identity(index)
    telemetry = read_telemetry(index)
    extra = {
        "vbios": smi_query(index, "vbios_version"),
        "ecc": smi_query(index, "ecc.mode.current"),
        "ecc_err": smi_query(index, "ecc.errors.uncorrected.aggregate.total"),
        "mig": smi_query(index, "mig.mode.current"),
        "max_sm": smi_query(index, "clocks.max.sm"),
        "max_mem": smi_query(index, "clocks.max.mem"),
        "pwr_lim": smi_query(index, "power.limit"),
    }

    def first(key: str) -> str | None:
        row = extra[key]
        if row is None or not row:
            return None
        return smi_clean(row[0])

    return DeviceProbe(
        identity=identity,
        model_key=model_key(identity),
        vbios=first("vbios"),
        ecc_mode=first("ecc"),
        ecc_uncorrected_aggregate=first("ecc_err"),
        mig_mode=first("mig"),
        sm_clock_mhz=telemetry.sm_clock_mhz,
        mem_clock_mhz=telemetry.mem_clock_mhz,
        sm_clock_max_mhz=first("max_sm"),
        mem_clock_max_mhz=first("max_mem"),
        power_w=telemetry.power_w,
        power_limit_w=first("pwr_lim"),
        temperature_c=telemetry.temperature_c,
        compute_apps=smi_query_apps(index),
        telemetry=telemetry,
    )


def _shared_reasons(devices: tuple[DeviceProbe, ...], our_pid: str) -> tuple[str, ...]:
    reasons: list[str] = []
    for device in devices:
        mig = (device.mig_mode or "").lower()
        if mig in {"enabled", "1", "on"}:
            reasons.append(f"MIG enabled on GPU {device.identity.device_index}")
        others = [
            f"{pid}:{name}"
            for pid, name in device.compute_apps
            if pid not in {our_pid, "N/A"}
        ]
        if others:
            reasons.append(
                f"other compute PIDs on GPU {device.identity.device_index}: "
                + ", ".join(others)
            )
    return tuple(reasons)


def probe_environment() -> EnvironmentProbe:
    """Inspect this machine. Does not contact the network."""
    cuda_ok = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if cuda_ok else 0
    devices = tuple(_one_device(index) for index in range(max(count, 0)))
    if not devices and smi_query(0, "name") is not None:
        devices = (_one_device(0),)
    driver = None
    if devices:
        driver = devices[0].identity.driver_version
    vendor = _read_text(_DMI_VENDOR)
    product = _read_text(_DMI_PRODUCT)
    hyper = _cpu_hypervisor_flag()
    virt = hyper or (
        vendor is not None
        and any(
            token in vendor.lower()
            for token in ("google", "amazon", "microsoft", "qemu", "vmware", "xen")
        )
    )
    reasons = _shared_reasons(devices, str(os.getpid()))
    hip = getattr(torch.version, "hip", None)
    hip_text = str(hip) if hip else None
    return EnvironmentProbe(
        cuda_available=cuda_ok,
        gpu_count=count,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        hip_version=hip_text,
        driver_version=driver,
        dmi_sys_vendor=vendor,
        dmi_product_name=product,
        cpu_hypervisor_flag=hyper,
        appears_virtualized=virt,
        appears_shared=bool(reasons),
        shared_reasons=reasons,
        devices=devices,
    )


def probe_dict(probe: EnvironmentProbe) -> dict[str, Any]:
    payload = asdict(probe)
    return payload
