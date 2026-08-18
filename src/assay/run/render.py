"""Human and JSON output for assay run. Color only when asked (TTY)."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from assay.abft.check import CheckStatus
from assay.probe.environment import EnvironmentProbe
from assay.run.guarantee import NETWORK_GUARANTEE
from assay.run.types import AssayResult, CaseRecord, ExitCode, exit_code_for

_RESET = "\033[0m"
_BOLD = "\033[1m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"


def _paint(text: str, code: str, *, color: bool) -> str:
    if not color:
        return text
    return f"{code}{text}{_RESET}"


def _status_paint(status: CheckStatus, *, color: bool) -> str:
    word = status.value
    if status is CheckStatus.PASS:
        return _paint(word, _GREEN, color=color)
    if status is CheckStatus.FAIL:
        return _paint(word, _RED, color=color)
    return _paint(word, _YELLOW, color=color)


def format_shape(shape: tuple[int, ...]) -> str:
    if not shape:
        return "(empty)"
    return "x".join(str(dim) for dim in shape)


def fail_detail_lines(case: CaseRecord) -> list[str]:
    """Every FAIL names workload, shape, residual, threshold, and n_samples."""
    return [
        f"  workload     {case.workload}  {case.case}",
        f"  shape        {format_shape(case.shape)}",
        f"  dtype        {case.dtype_name or '(none)'}",
        f"  residual     {case.residual_hex}  ({case.residual_decimal})",
        f"  threshold    {case.threshold_hex}  ({case.threshold_decimal})"
        f"  [empirical noisefloor quantile]",
        f"  sample_max   {case.sample_max_hex}",
        f"  n_samples    {case.n_samples}  "
        f"(min_samples={case.min_samples}, spec={case.noisefloor_spec_version})",
        f"  reason       {case.reason}",
    ]


def render_probe(probe: EnvironmentProbe) -> list[str]:
    lines = [
        "ENVIRONMENT",
        f"  cuda_available     {probe.cuda_available}",
        f"  gpu_count          {probe.gpu_count}",
        f"  torch              {probe.torch_version}",
        f"  CUDA               {probe.cuda_version or '(none)'}",
        f"  ROCm/HIP           {probe.hip_version or '(none)'}",
        f"  driver             {probe.driver_version or '(none)'}",
        f"  DMI vendor/product {probe.dmi_sys_vendor or '(none)'} / "
        f"{probe.dmi_product_name or '(none)'}",
        f"  virtualized        {probe.appears_virtualized}"
        f"  (cpu hypervisor flag={probe.cpu_hypervisor_flag})",
        f"  shared instance    {probe.appears_shared}",
    ]
    lines.extend(f"    - {reason}" for reason in probe.shared_reasons)
    if not probe.devices:
        lines.append("  devices            (none)")
        return lines
    for device in probe.devices:
        name = device.identity.smi_name or device.identity.torch_name or "(unknown)"
        lines.append(f"  GPU {device.identity.device_index}  {name}")
        lines.append(f"    model_key   {device.model_key}")
        lines.append(f"    uuid        {device.identity.uuid or '(none)'}")
        lines.append(f"    VBIOS       {device.vbios or '(none)'}")
        lines.append(
            f"    ECC         mode={device.ecc_mode or '(none)'}  "
            f"uncorrected_aggregate={device.ecc_uncorrected_aggregate or '(none)'}"
        )
        lines.append(f"    MIG         {device.mig_mode or '(none)'}")
        lines.append(
            f"    clocks      sm={device.sm_clock_mhz or '(none)'} MHz  "
            f"mem={device.mem_clock_mhz or '(none)'} MHz  "
            f"sm_max={device.sm_clock_max_mhz or '(none)'}  "
            f"mem_max={device.mem_clock_max_mhz or '(none)'}"
        )
        lines.append(
            f"    power/temp  {device.power_w or '(none)'} W / "
            f"limit {device.power_limit_w or '(none)'} W / "
            f"{device.temperature_c or '(none)'} C"
        )
        if device.compute_apps:
            apps = ", ".join(f"{pid}:{name}" for pid, name in device.compute_apps)
            lines.append(f"    compute_apps {apps}")
        else:
            lines.append("    compute_apps (none reported)")
    return lines


def _uncharacterized_banner(result: AssayResult, *, color: bool) -> list[str]:
    if result.noisefloor_status != "uncharacterized":
        return [
            "NOISEFLOOR",
            f"  status  {result.noisefloor_status}",
            f"  {result.noisefloor_reason}",
        ]
    title = _paint("UNCHARACTERIZED", _YELLOW + _BOLD, color=color)
    return [
        title,
        "  This configuration has no noisefloor-v1 samples in data/noisefloor/.",
        "  assay will print measurements and will not guess PASS or FAIL.",
        f"  {result.noisefloor_reason}",
    ]


def verdict_reason(result: AssayResult) -> str:
    if result.status is CheckStatus.FAIL:
        fails = [
            case
            for case in result.cases
            if case.status is CheckStatus.FAIL and not case.skipped
        ]
        return f"{len(fails)} GEMM case(s) exceeded every measured noisefloor sample"
    if result.noisefloor_status == "uncharacterized":
        return result.noisefloor_reason
    voting = [case for case in result.cases if case.counts_toward_verdict]
    if any(case.skipped for case in voting):
        return "a voting GEMM case was skipped (time budget or error); not a FAIL"
    if result.status is CheckStatus.PASS:
        return "every voting GEMM residual was at or below the empirical quantile"
    return "GEMM cases did not all PASS against a characterized noisefloor"


def render_human(result: AssayResult, *, color: bool) -> str:
    painted = _status_paint(result.status, color=color)
    exit_code = int(exit_code_for(result.status))
    lines: list[str] = [
        f"assay run  budget={result.budget.name} ({result.budget.seconds}s)",
        NETWORK_GUARANTEE,
        "",
    ]
    lines.extend(render_probe(result.probe))
    lines.append("")
    lines.extend(_uncharacterized_banner(result, color=color))
    lines.append("")
    lines.append("CASES")
    for case in result.cases:
        tag = _status_paint(case.status, color=color)
        skip = " SKIP" if case.skipped else ""
        extra = ""
        if case.residual_hex is not None:
            extra = f"  residual={case.residual_hex}"
            if case.n_samples is not None:
                extra += f"  n_samples={case.n_samples}"
        if case.golden_max_abs_error_hex is not None:
            extra += f"  golden_max_abs={case.golden_max_abs_error_hex}"
        lines.append(
            f"  {case.workload} {case.case}  {format_shape(case.shape)}  "
            f"{tag}{skip}{extra}"
        )
        lines.append(f"    {case.reason}")
    fails = [
        case
        for case in result.cases
        if case.status is CheckStatus.FAIL and not case.skipped
    ]
    if fails:
        lines.append("")
        lines.append(_paint("FAIL DETAIL", _RED + _BOLD, color=color))
        lines.append(
            "Each FAIL lists workload, shape, residual, threshold, and n_samples."
        )
        for case in fails:
            lines.extend(fail_detail_lines(case))
    lines.append("")
    lines.append(f"VERDICT  {painted}  (exit {exit_code})")
    lines.append(f"  {verdict_reason(result)}")
    lines.append(f"  elapsed {result.elapsed_s:.1f}s")
    if result.status is CheckStatus.FAIL and not fails:
        lines.append("  internal error: FAIL status without a failing case")
    return "\n".join(lines) + "\n"


def case_payload(case: CaseRecord) -> dict[str, Any]:
    return {
        "workload": case.workload,
        "case": case.case,
        "shape": list(case.shape),
        "dtype": case.dtype_name,
        "status": case.status.value,
        "reason": case.reason,
        "residual_hex": case.residual_hex,
        "residual_decimal": case.residual_decimal,
        "threshold_hex": case.threshold_hex,
        "threshold_decimal": case.threshold_decimal,
        "sample_max_hex": case.sample_max_hex,
        "n_samples": case.n_samples,
        "min_samples": case.min_samples,
        "noisefloor_spec_version": case.noisefloor_spec_version,
        "golden_max_abs_error_hex": case.golden_max_abs_error_hex,
        "wall_time_s": case.wall_time_s,
        "skipped": case.skipped,
        "skip_reason": case.skip_reason,
        "counts_toward_verdict": case.counts_toward_verdict,
    }


def result_payload(result: AssayResult) -> dict[str, Any]:
    exit_code = int(exit_code_for(result.status))
    return {
        "schema": "assay-attestation-v1",
        "network_requests": 0,
        "network_guarantee": NETWORK_GUARANTEE,
        "status": result.status.value,
        "exit_code": exit_code,
        "elapsed_s": result.elapsed_s,
        "gpu_model": result.gpu_model,
        "noisefloor": {
            "status": result.noisefloor_status,
            "reason": result.noisefloor_reason,
        },
        "budget": asdict(result.budget),
        "probe": asdict(result.probe),
        "cases": [case_payload(case) for case in result.cases],
        "verdict_reason": verdict_reason(result),
    }


def render_json(result: AssayResult) -> str:
    return json.dumps(result_payload(result), indent=2) + "\n"


def operational_payload(reason: str, probe: EnvironmentProbe | None) -> dict[str, Any]:
    return {
        "schema": "assay-attestation-v1",
        "network_requests": 0,
        "network_guarantee": NETWORK_GUARANTEE,
        "status": "OPERATIONAL",
        "exit_code": int(ExitCode.OPERATIONAL),
        "reason": reason,
        "probe": asdict(probe) if probe is not None else None,
    }
