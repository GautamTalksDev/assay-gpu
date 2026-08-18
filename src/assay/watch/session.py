"""WatchSession: module hooks, 1-in-N sampling, never raises into the host."""

from __future__ import annotations

import time
from types import TracebackType
from typing import Any

from torch import nn
from torch.utils.hooks import RemovableHandle

from assay.abft.check import CheckStatus, GemmCheckConfig, check_gemm
from assay.run.types import CaseRecord, overall_status
from assay.watch.emit import write_watch_log
from assay.watch.factors import linear_gemm_factors, workload_for_dtype
from assay.watch.types import WatchConfig, WatchEvent

_LINEAR = nn.Linear
_ATTENTION = nn.MultiheadAttention


class WatchSession:
    """Process-wide forward hooks. Host forward must succeed even if ABFT dies."""

    def __init__(self, config: WatchConfig) -> None:
        if config.every < 1:
            msg = "--every must be >= 1"
            raise ValueError(msg)
        self.config = config
        self.events: list[WatchEvent] = []
        self.gemm_seen = 0
        self.gemm_checked = 0
        self.attention_seen = 0
        self._handle: RemovableHandle | None = None
        self._in_check = False
        self._last_flush = time.monotonic()

    def install(self) -> None:
        if self._handle is not None:
            return
        self._handle = nn.modules.module.register_module_forward_hook(self._hook)

    def uninstall(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> WatchSession:
        self.install()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            self.flush()
        finally:
            self.uninstall()

    def _hook(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: object,
    ) -> None:
        if self._in_check:
            return
        try:
            self._dispatch(module, inputs, output)
        except Exception as exc:  # noqa: BLE001
            self._swallowed(module, exc)
        self._maybe_roll()

    def _dispatch(
        self, module: nn.Module, inputs: tuple[Any, ...], output: object
    ) -> None:
        if isinstance(module, _LINEAR):
            self._linear(module, inputs, output)
            return
        if isinstance(module, _ATTENTION):
            self._attention(module)

    def _should_check(self, seen: int) -> bool:
        return (seen - 1) % self.config.every == 0

    def _linear(
        self, module: nn.Linear, inputs: tuple[Any, ...], output: object
    ) -> None:
        self.gemm_seen += 1
        if not self._should_check(self.gemm_seen):
            return
        self.gemm_checked += 1
        factors = linear_gemm_factors(module, inputs, output)
        if factors is None:
            self.events.append(
                WatchEvent(
                    kind="linear_gemm",
                    module_type=type(module).__name__,
                    status=CheckStatus.INCONCLUSIVE,
                    reason="could not map this Linear forward onto a GEMM",
                    residual_hex=None,
                    threshold_hex=None,
                    n_samples=None,
                    min_samples=None,
                    shape=None,
                    counts_toward_verdict=False,
                    swallowed_exception=None,
                )
            )
            return
        left, right, product = factors
        workload = workload_for_dtype(product.dtype)
        if workload is None:
            self.events.append(
                WatchEvent(
                    kind="linear_gemm",
                    module_type=type(module).__name__,
                    status=CheckStatus.INCONCLUSIVE,
                    reason=f"no noisefloor workload for dtype {product.dtype}",
                    residual_hex=None,
                    threshold_hex=None,
                    n_samples=None,
                    min_samples=None,
                    shape=tuple(int(dim) for dim in product.shape),
                    counts_toward_verdict=False,
                    swallowed_exception=None,
                )
            )
            return
        self._in_check = True
        try:
            checked = check_gemm(
                left,
                right,
                product,
                GemmCheckConfig(
                    noisefloor_dir=self.config.noisefloor_dir,
                    workload=workload,
                    gpu_model=self.config.gpu_model,
                ),
            )
        finally:
            self._in_check = False
        self.events.append(
            WatchEvent(
                kind="linear_gemm",
                module_type=type(module).__name__,
                status=checked.status,
                reason=checked.reason,
                residual_hex=checked.residual_hex,
                threshold_hex=checked.threshold_hex,
                n_samples=checked.n_samples,
                min_samples=checked.min_samples,
                shape=checked.shape,
                counts_toward_verdict=True,
                swallowed_exception=None,
            )
        )

    def _attention(self, module: nn.Module) -> None:
        self.attention_seen += 1
        if not self._should_check(self.attention_seen):
            return
        self.events.append(
            WatchEvent(
                kind="attention",
                module_type=type(module).__name__,
                status=CheckStatus.INCONCLUSIVE,
                reason="no GEMM ABFT in noisefloor-v1 for attention modules",
                residual_hex=None,
                threshold_hex=None,
                n_samples=None,
                min_samples=None,
                shape=None,
                counts_toward_verdict=False,
                swallowed_exception=None,
            )
        )

    def _swallowed(self, module: nn.Module, exc: Exception) -> None:
        self.events.append(
            WatchEvent(
                kind="error",
                module_type=type(module).__name__,
                status=CheckStatus.INCONCLUSIVE,
                reason="ABFT check raised; recorded and suppressed",
                residual_hex=None,
                threshold_hex=None,
                n_samples=None,
                min_samples=None,
                shape=None,
                counts_toward_verdict=False,
                swallowed_exception=f"{type(exc).__name__}: {exc}",
            )
        )

    def _maybe_roll(self) -> None:
        interval = self.config.interval_seconds
        if interval is None or self.config.report_path is None:
            return
        now = time.monotonic()
        if now - self._last_flush < interval:
            return
        self.flush()
        self._last_flush = now

    def overall_status(self) -> CheckStatus:
        records = tuple(
            _event_as_case(event, index) for index, event in enumerate(self.events)
        )
        return overall_status(records)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "assay-watch-v1",
            "kt2": "FAIL",
            "shipped": False,
            "every": self.config.every,
            "gpu_model": self.config.gpu_model,
            "gemm_seen": self.gemm_seen,
            "gemm_checked": self.gemm_checked,
            "attention_seen": self.attention_seen,
            "status": self.overall_status().value,
            "events": [_event_payload(event) for event in self.events],
            "note": (
                "assay watch is not a shipped product (docs/RESULTS-KT2.md). "
                "This file is a local event log, not a signed attestation-v1."
            ),
        }

    def flush(self) -> None:
        if self.config.report_path is None:
            return
        write_watch_log(self.config.report_path, self.to_payload())


def _event_payload(event: WatchEvent) -> dict[str, Any]:
    return {
        "kind": event.kind,
        "module_type": event.module_type,
        "status": event.status.value,
        "reason": event.reason,
        "residual_hex": event.residual_hex,
        "threshold_hex": event.threshold_hex,
        "n_samples": event.n_samples,
        "min_samples": event.min_samples,
        "shape": list(event.shape) if event.shape is not None else None,
        "counts_toward_verdict": event.counts_toward_verdict,
        "swallowed_exception": event.swallowed_exception,
    }


def _event_as_case(event: WatchEvent, index: int) -> CaseRecord:
    shape = event.shape if event.shape is not None else ()
    return CaseRecord(
        workload="watch",
        case=f"{event.kind}_{index}",
        shape=shape,
        dtype_name=None,
        status=event.status,
        reason=event.reason,
        residual_hex=event.residual_hex,
        residual_decimal=None,
        threshold_hex=event.threshold_hex,
        threshold_decimal=None,
        sample_max_hex=None,
        n_samples=event.n_samples,
        min_samples=event.min_samples,
        noisefloor_spec_version=None,
        golden_max_abs_error_hex=None,
        wall_time_s=0.0,
        skipped=False,
        skip_reason=None,
        counts_toward_verdict=event.counts_toward_verdict,
    )
