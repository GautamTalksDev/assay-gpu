"""Command-line entrypoint for assay."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Annotated

import torch
import typer

from assay.abft.overhead import measure_checksum_overhead
from assay.abft.reduce import CheckBackend
from assay.noise.lookup import assay_verdict, lookup_abft_tolerance
from assay.noise.pilot import PILOT_N, run_abft_pilot
from assay.noise.run import characterize_gemm
from assay.noise.sweep_v3 import run_v3_sweep
from assay.probe.environment import probe_environment
from assay.reference.serialize import write_catalog
from assay.reference.spec import CHARACTERIZATION_MAX_SIDE
from assay.report.attestation import write_attestation
from assay.report.keys import generate_keypair, load_keypair, write_keypair
from assay.report.verify import exit_code_for_verify, render_verify, verify_path
from assay.run.budget import budget_from_flags
from assay.run.guarantee import NETWORK_GUARANTEE
from assay.run.render import operational_payload, render_human, render_json
from assay.run.session import execute_assay
from assay.run.types import AssayOperationalError, ExitCode, exit_code_for
from assay.watch.launch import run_user_script
from assay.watch.session import WatchSession
from assay.watch.types import WatchConfig
from assay.workload.suite import double_run, run_all, write_double_run_report

app = typer.Typer(
    no_args_is_help=True,
    help=("Deterministic GPU correctness assay. " + NETWORK_GUARANTEE),
)
reference_app = typer.Typer(help="Generate fp64 golden references.")
workload_app = typer.Typer(help="Run seeded GPU workloads.")
abft_app = typer.Typer(help="GEMM checksum detector.")
app.add_typer(reference_app, name="reference")
app.add_typer(workload_app, name="workload")
app.add_typer(abft_app, name="abft")

DEFAULT_REFERENCE_DIR = Path("data/reference")
DEFAULT_DOUBLE_RUN_RECORD = Path("data/workload/double_run.json")
DEFAULT_NOISEFLOOR_DIR = Path("data/noisefloor")


@app.callback()
def main() -> None:
    """Deterministic GPU correctness assay.

    assay run makes ZERO network requests. No telemetry, no version check,
    no analytics, no phone-home. It runs inside customers' private
    infrastructure.
    """


@reference_app.command("generate")
def reference_generate(
    output: Annotated[Path, typer.Option("--output", "-o")] = DEFAULT_REFERENCE_DIR,
) -> None:
    """Write .npz goldens and manifest.json. Array hashes, not ZIP bytes."""
    artifacts = write_catalog(output)
    typer.echo(f"wrote {len(artifacts)} artifacts to {output}")
    for artifact in artifacts:
        typer.echo(f"{artifact.sha256}  {artifact.name}")


def _run_workloads(*, double: bool, record: Path) -> None:
    if not torch.cuda.is_available():
        typer.echo("CUDA is required for workloads.", err=True)
        raise typer.Exit(code=1)
    if double:
        report = double_run()
        write_double_run_report(report, record)
        typer.echo(f"wrote {record}")
        typer.echo(f"all_bitwise_identical={report.all_bitwise_identical}")
        for case in report.cases:
            typer.echo(
                f"{case.workload} {case.case} identical={case.bitwise_identical} "
                f"det={case.deterministic_run1} kernel={case.kernel}"
            )
        return
    for result in run_all():
        typer.echo(
            f"{result.workload} {result.case} shape={result.shape} "
            f"wall_s={result.wall_time_s:.6f} det={result.deterministic} "
            f"kernel={result.kernel}"
        )
        if result.nondeterminism_reason is not None:
            typer.echo(f"  nondeterminism: {result.nondeterminism_reason}")


@workload_app.command("run")
def workload_run(
    double: Annotated[
        bool,
        typer.Option("--double", help="Run twice and record bitwise identity."),
    ] = False,
    record: Annotated[Path, typer.Option("--record")] = DEFAULT_DOUBLE_RUN_RECORD,
) -> None:
    """Execute W01-W07. Does not compare against references."""
    _run_workloads(double=double, record=record)


def _color_enabled(*, as_json: bool) -> bool:
    if as_json:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return bool(sys.stdout.isatty())


def _emit_operational(*, reason: str, as_json: bool) -> None:
    probe = probe_environment()
    if as_json:
        typer.echo(json.dumps(operational_payload(reason, probe), indent=2))
    else:
        typer.echo(reason, err=True)
        typer.echo(
            f"cuda_available={probe.cuda_available} gpu_count={probe.gpu_count}",
            err=True,
        )
        typer.echo("exit 3 (operational error)", err=True)


@app.command("run")
def run(  # noqa: PLR0913, PLR0917
    quick: Annotated[
        bool,
        typer.Option("--quick", help="2-minute budget. Smaller GEMM/SDPA shapes."),
    ] = False,
    thorough: Annotated[
        bool,
        typer.Option("--thorough", help="60-minute budget. Full suite including 8192."),
    ] = False,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Machine-readable attestation on stdout."),
    ] = False,
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Write a signed attestation-v1 JSON to PATH."),
    ] = None,
    signing_key: Annotated[
        Path | None,
        typer.Option(
            "--signing-key",
            help="Ed25519 key from `assay keygen`. Required with --report.",
        ),
    ] = None,
    noisefloor_dir: Annotated[
        Path, typer.Option("--noisefloor-dir")
    ] = DEFAULT_NOISEFLOOR_DIR,
    reference_dir: Annotated[
        Path, typer.Option("--reference-dir")
    ] = DEFAULT_REFERENCE_DIR,
    device: Annotated[int, typer.Option("--device")] = 0,
) -> None:
    """Probe this machine, run the suite, compare goldens, and ABFT-check GEMMs.

    Exit codes: 0 PASS, 1 FAIL, 2 INCONCLUSIVE, 3 operational error.

    Uncharacterized GPUs never receive a PASS/FAIL verdict. FAIL always
    names the workload, shape, residual, threshold, and n_samples.

    Makes ZERO network requests. No telemetry, no version check, no
    analytics, no phone-home.
    """
    try:
        budget = budget_from_flags(quick=quick, thorough=thorough)
    except ValueError as exc:
        _emit_operational(reason=str(exc), as_json=as_json)
        raise typer.Exit(code=int(ExitCode.OPERATIONAL)) from None

    if not torch.cuda.is_available():
        _emit_operational(
            reason="no CUDA GPU visible; assay run cannot execute workloads",
            as_json=as_json,
        )
        raise typer.Exit(code=int(ExitCode.OPERATIONAL))

    def progress(message: str) -> None:
        typer.echo(message, err=True)

    try:
        result = execute_assay(
            budget=budget,
            noisefloor_dir=noisefloor_dir,
            reference_dir=reference_dir,
            device_index=device,
            progress=progress,
        )
    except (AssayOperationalError, FileNotFoundError) as exc:
        _emit_operational(reason=str(exc), as_json=as_json)
        raise typer.Exit(code=int(ExitCode.OPERATIONAL)) from None

    if report is not None:
        if signing_key is None:
            _emit_operational(
                reason="--report requires --signing-key (assay keygen --out PATH)",
                as_json=as_json,
            )
            raise typer.Exit(code=int(ExitCode.OPERATIONAL))
        try:
            pair = load_keypair(signing_key)
            write_attestation(
                report,
                result,
                signing_key=pair,
                reference_dir=reference_dir,
                noisefloor_dir=noisefloor_dir,
            )
        except (OSError, ValueError, KeyError) as exc:
            _emit_operational(reason=str(exc), as_json=as_json)
            raise typer.Exit(code=int(ExitCode.OPERATIONAL)) from None
        typer.echo(f"wrote signed attestation {report}", err=True)
        typer.echo(
            "SELF-SIGNED. A provider grading its own hardware is not "
            "independent evidence.",
            err=True,
        )

    if as_json:
        typer.echo(render_json(result), nl=False)
    else:
        typer.echo(
            render_human(result, color=_color_enabled(as_json=False)),
            nl=False,
        )
    raise typer.Exit(code=int(exit_code_for(result.status)))


@app.command("keygen")
def keygen_cmd(
    out: Annotated[Path, typer.Option("--out", help="Write Ed25519 keypair JSON.")],
) -> None:
    """Generate an Ed25519 keypair for self-signed reports. Local file only."""
    pair = generate_keypair()
    write_keypair(out, pair)
    typer.echo(f"wrote {out}")
    typer.echo(f"public_key_hex={pair.public_key_hex}")
    typer.echo(
        "SELF-SIGNED reports prove integrity, not independence. "
        "A provider grading its own hardware is not independent evidence."
    )


@app.command("verify")
def verify_cmd(
    report: Annotated[Path, typer.Argument(help="Signed attestation-v1 JSON.")],
    pubkey: Annotated[
        Path | None,
        typer.Option("--pubkey", help="Pin a public key (hex or key JSON)."),
    ] = None,
    reference_dir: Annotated[
        Path | None,
        typer.Option("--reference-dir", help="Recompute reference catalog hash."),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Machine-readable verify result."),
    ] = False,
) -> None:
    """Validate signature and internal consistency. Fully offline.

    Exit 0 valid, 1 invalid (tamper or inconsistency), 3 operational.
    Self-signed reports always print the independence disclaimer.
    Makes ZERO network requests.
    """
    checked = verify_path(report, pubkey_path=pubkey, reference_dir=reference_dir)
    if as_json:
        payload = {
            "outcome": checked.outcome.value,
            "mode": checked.mode,
            "status": checked.status,
            "reasons": list(checked.reasons),
            "warnings": list(checked.warnings),
            "exit_code": exit_code_for_verify(checked),
        }
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo(render_verify(checked), nl=False)
    raise typer.Exit(code=exit_code_for_verify(checked))


@app.command(
    "watch",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def watch_cmd(
    ctx: typer.Context,
    every: Annotated[
        int,
        typer.Option(
            "--every",
            help="Check 1 in N Linear GEMMs. Required; no 5% default exists.",
        ),
    ],
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Write a rolling assay-watch-v1 JSON log."),
    ] = None,
    interval_seconds: Annotated[
        float | None,
        typer.Option(
            "--interval-seconds",
            help="Rewrite --report this often. Omit to write on exit only.",
        ),
    ] = None,
    noisefloor_dir: Annotated[
        Path, typer.Option("--noisefloor-dir")
    ] = DEFAULT_NOISEFLOOR_DIR,
) -> None:
    """Hook Linear and MultiheadAttention on a live script. Not a shipped product.

    KT-2 is FAIL (docs/RESULTS-KT2.md). This command is a measurement harness.
    A failed ABFT check is recorded; it is never raised into the host workload.

    Pass the user script after `--`: assay watch --every 8 -- infer.py

    Makes ZERO network requests. Does not download weights.
    """
    typer.echo(
        "KT-2 FAIL: assay watch is not a shipped product. See docs/RESULTS-KT2.md.",
        err=True,
    )
    if every < 1:
        _emit_operational(reason="--every must be >= 1", as_json=False)
        raise typer.Exit(code=int(ExitCode.OPERATIONAL))
    argv = list(ctx.args)
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv:
        _emit_operational(
            reason="pass a script after options, e.g. --every 8 -- script.py",
            as_json=False,
        )
        raise typer.Exit(code=int(ExitCode.OPERATIONAL))
    gpu_model = None
    probe = probe_environment()
    if probe.devices:
        gpu_model = probe.devices[0].model_key
    session = WatchSession(
        WatchConfig(
            every=every,
            noisefloor_dir=noisefloor_dir,
            report_path=report,
            interval_seconds=interval_seconds,
            gpu_model=gpu_model,
        )
    )
    session.install()
    try:
        run_user_script(argv)
    except (OSError, ValueError) as exc:
        session.flush()
        session.uninstall()
        _emit_operational(reason=str(exc), as_json=False)
        raise typer.Exit(code=int(ExitCode.OPERATIONAL)) from None
    finally:
        session.flush()
        session.uninstall()
    typer.echo(
        f"watch status={session.overall_status().value} "
        f"gemm_seen={session.gemm_seen} checked={session.gemm_checked}",
        err=True,
    )


def _print_lookup(
    *,
    noisefloor_dir: Path,
    workload: str,
    dtype_name: str,
    shape: tuple[int, int, int],
    gpu_model: str | None,
) -> None:
    found = lookup_abft_tolerance(
        noisefloor_dir,
        workload=workload,
        dtype=dtype_name,
        shape=shape,
        gpu_model=gpu_model,
    )
    typer.echo(f"verdict={assay_verdict(found)}")
    typer.echo(f"status={found.status.value}")
    typer.echo(f"reason={found.reason}")
    typer.echo(
        f"workload={found.workload} dtype={found.dtype} shape={list(found.shape)}"
    )
    typer.echo(f"n_samples={found.n_samples} min_samples={found.min_samples}")
    typer.echo(f"target_quantile={found.target_quantile}")
    typer.echo(f"p_quantile_residual_hex={found.p_quantile_residual_hex}")
    typer.echo(f"p_quantile_residual_decimal={found.p_quantile_residual_decimal}")
    typer.echo(f"sample_max_residual_hex={found.sample_max_residual_hex}")
    typer.echo(f"gpu_models={list(found.gpu_models)}")
    typer.echo(f"source_files={list(found.source_files)}")


@app.command("characterize")
def characterize_cmd(  # noqa: PLR0913, PLR0917
    repeats: Annotated[
        int | None,
        typer.Option("--repeats", help="GPU repeats. Required unless --lookup."),
    ] = None,
    lookup: Annotated[
        bool,
        typer.Option("--lookup", help="Print tolerance lookup; no GPU."),
    ] = False,
    noisefloor_dir: Annotated[
        Path, typer.Option("--noisefloor-dir")
    ] = DEFAULT_NOISEFLOOR_DIR,
    device: Annotated[int, typer.Option("--device")] = 0,
    include_large: Annotated[bool, typer.Option("--include-large")] = False,
    workload: Annotated[str | None, typer.Option("--workload")] = None,
    m_dim: Annotated[int | None, typer.Option("--m")] = None,
    k_dim: Annotated[int | None, typer.Option("--k")] = None,
    n_dim: Annotated[int | None, typer.Option("--n")] = None,
    dtype_name: Annotated[str, typer.Option("--dtype")] = "bfloat16",
    gpu_model: Annotated[str | None, typer.Option("--gpu-model")] = None,
    pilot: Annotated[
        bool,
        typer.Option(
            "--pilot",
            help="W02 bf16 4096 cubed, 2000 samples. Not a characterization.",
        ),
    ] = False,
    sweep_v3: Annotated[
        bool,
        typer.Option(
            "--sweep-v3",
            help="Clean-only residual-v3 sweep. W02 bf16 4096 cubed, JSONL output.",
        ),
    ] = False,
    sweep_v3_output: Annotated[
        Path,
        typer.Option("--sweep-v3-output", help="JSONL output path for --sweep-v3."),
    ] = Path("data/noisefloor/pilot/sweep-v3.jsonl"),
    sweep_v3_flips: Annotated[
        bool,
        typer.Option(
            "--sweep-v3-flips",
            help="Residual-v3 flip matrix. W02 bf16 4096 cubed, JSONL output.",
        ),
    ] = False,
    sweep_v3_flips_output: Annotated[
        Path,
        typer.Option(
            "--sweep-v3-flips-output",
            help="JSONL output path for --sweep-v3-flips.",
        ),
    ] = Path("data/noisefloor/pilot/sweep-v3-flips.jsonl"),
    sweep_v3_kscale: Annotated[
        bool,
        typer.Option(
            "--sweep-v3-kscale",
            help="Residual-v3 K-scaling study. W02 bf16, JSONL output.",
        ),
    ] = False,
    sweep_v3_kscale_output: Annotated[
        Path,
        typer.Option(
            "--sweep-v3-kscale-output",
            help="JSONL output path for --sweep-v3-kscale.",
        ),
    ] = Path("data/noisefloor/pilot/sweep-v3-kscale.jsonl"),
    k_values: Annotated[
        str | None,
        typer.Option(
            "--k-values",
            help="Comma-separated inner K dimensions for --sweep-v3-kscale.",
        ),
    ] = None,
) -> None:
    """Measure GPU vs fp64 noise floor, or look up a stored tolerance."""
    shape: tuple[int, int, int] | None
    if m_dim is None and k_dim is None and n_dim is None:
        shape = None
    elif m_dim is not None and k_dim is not None and n_dim is not None:
        shape = (m_dim, k_dim, n_dim)
    else:
        typer.echo("pass all of --m --k --n or none of them", err=True)
        raise typer.Exit(code=1)
    exclusive = sum([lookup, pilot, sweep_v3, sweep_v3_flips, sweep_v3_kscale])
    if exclusive > 1:
        typer.echo(
            "pass only one of --lookup, --pilot, --sweep-v3, "
            "--sweep-v3-flips, and --sweep-v3-kscale",
            err=True,
        )
        raise typer.Exit(code=1)
    if lookup:
        lookup_shape = shape or (
            CHARACTERIZATION_MAX_SIDE,
            CHARACTERIZATION_MAX_SIDE,
            CHARACTERIZATION_MAX_SIDE,
        )
        lookup_workload = workload or "W02"
        _print_lookup(
            noisefloor_dir=noisefloor_dir,
            workload=lookup_workload,
            dtype_name=dtype_name,
            shape=lookup_shape,
            gpu_model=gpu_model,
        )
        return
    if sweep_v3_kscale:
        if not torch.cuda.is_available():
            typer.echo(
                "CUDA is required for assay characterize --sweep-v3-kscale",
                err=True,
            )
            raise typer.Exit(code=1)
        from assay.noise.sweep_v3_kscale import (
            K_VALUES_DEFAULT,
            parse_k_values,
            run_v3_kscale_sweep,
        )

        selected = (
            parse_k_values(k_values) if k_values is not None else K_VALUES_DEFAULT
        )
        path = run_v3_kscale_sweep(
            output_path=sweep_v3_kscale_output,
            device_index=device,
            k_values=selected,
        )
        typer.echo(f"wrote {path}")
        return
    if sweep_v3_flips:
        if not torch.cuda.is_available():
            typer.echo(
                "CUDA is required for assay characterize --sweep-v3-flips",
                err=True,
            )
            raise typer.Exit(code=1)
        from assay.noise.sweep_v3_flips import FLIP_N, run_v3_flip_sweep

        path = run_v3_flip_sweep(
            output_path=sweep_v3_flips_output,
            device_index=device,
            n_samples=repeats if repeats is not None else FLIP_N,
        )
        typer.echo(f"wrote {path}")
        return
    if sweep_v3:
        if not torch.cuda.is_available():
            typer.echo("CUDA is required for assay characterize --sweep-v3", err=True)
            raise typer.Exit(code=1)
        path = run_v3_sweep(
            output_path=sweep_v3_output,
            device_index=device,
            n_samples=repeats if repeats is not None else PILOT_N,
        )
        typer.echo(f"wrote {path}")
        return
    if pilot:
        if not torch.cuda.is_available():
            typer.echo("CUDA is required for assay characterize --pilot", err=True)
            raise typer.Exit(code=1)
        path = run_abft_pilot(
            noisefloor_dir=noisefloor_dir,
            device_index=device,
            n_samples=repeats if repeats is not None else PILOT_N,
        )
        typer.echo(f"wrote {path}")
        return
    if repeats is None:
        typer.echo("--repeats is required to measure (no silent default)", err=True)
        raise typer.Exit(code=1)
    if not torch.cuda.is_available():
        typer.echo("CUDA is required for assay characterize", err=True)
        raise typer.Exit(code=1)

    path = characterize_gemm(
        noisefloor_dir=noisefloor_dir,
        repeats=repeats,
        device_index=device,
        include_large=include_large,
        workload=workload,
        shape_filter=shape,
    )
    typer.echo(f"wrote {path}")
    report_shape = shape or (
        CHARACTERIZATION_MAX_SIDE,
        CHARACTERIZATION_MAX_SIDE,
        CHARACTERIZATION_MAX_SIDE,
    )
    report_workload = workload or "W02"
    _print_lookup(
        noisefloor_dir=noisefloor_dir,
        workload=report_workload,
        dtype_name=dtype_name,
        shape=report_shape,
        gpu_model=gpu_model,
    )


@abft_app.command("overhead")
def abft_overhead_cmd(  # noqa: PLR0913, PLR0917
    repeats: Annotated[
        int,
        typer.Option("--repeats", help="Timed repeats. Required; no silent default."),
    ],
    rows: Annotated[int, typer.Option("--m")] = 512,
    inner: Annotated[int, typer.Option("--k")] = 512,
    cols: Annotated[int, typer.Option("--n")] = 512,
    backend_name: Annotated[str, typer.Option("--backend")] = "pytorch",
    device_name: Annotated[str, typer.Option("--device")] = "cpu",
    dtype_name: Annotated[str, typer.Option("--dtype")] = "float32",
) -> None:
    """Measure checksum and residual-v2 scale wall time versus GEMM."""
    try:
        backend = CheckBackend(backend_name)
    except ValueError:
        typer.echo("backend must be pytorch or triton", err=True)
        raise typer.Exit(code=1) from None
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_name)
    if dtype is None:
        typer.echo("dtype must be float32, float16, or bfloat16", err=True)
        raise typer.Exit(code=1)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        typer.echo("CUDA is required for --device cuda", err=True)
        raise typer.Exit(code=1)
    measured = measure_checksum_overhead(
        shape=(rows, inner, cols),
        repeats=repeats,
        backend=backend,
        device=device,
        dtype=dtype,
    )
    typer.echo(f"shape={list(measured.shape)}")
    typer.echo(f"device={measured.device}")
    typer.echo(f"backend={measured.backend}")
    typer.echo(f"dtype={measured.dtype_name}")
    typer.echo(f"repeats={measured.repeats}")
    typer.echo(f"gemm_seconds={measured.gemm_seconds:.8f}")
    typer.echo(f"checksum_seconds={measured.checksum_seconds:.8f}")
    typer.echo(f"checksum_over_gemm={measured.checksum_over_gemm:.8f}")
    typer.echo(f"normalizer_seconds={measured.normalizer_seconds:.8f}")
    typer.echo(f"normalizer_over_gemm={measured.normalizer_over_gemm:.8f}")
