#!/usr/bin/env python3
"""GPD peaks-over-threshold fit for residual-v3 FPR (method locked in RESIDUAL.md).

CP-FPR-CENSOR: top-64 capture may truncate p99 (rarely) and p95 (always).
Does not edit docs/RESIDUAL.md. Uses only numpy (no scipy).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from assay.noise.floats import decode_f64

M_DIM = 4096
P95_FROZEN = 1.057096e-06
P99_FROZEN = 1.392735e-06
P999_FROZEN = 1.776693e-06
CLEAN_MAX_K4096 = 2.684525e-06
TARGET_TAIL_PROB = 2.44e-10
CHI2_1_95 = 3.841458820694124
TOP_TAIL_K = 64


@dataclass(frozen=True, slots=True)
class TruncationSummary:
    threshold_name: str
    u: float
    mean_above: float
    max_above: int
    n_over_64: int
    n_samples: int


@dataclass(frozen=True, slots=True)
class ExceedanceCollection:
    """Recovered exceedances plus optional right-censor records for truncated GEMMs."""

    y_exclude: np.ndarray
    y_with_truncated_recovered: np.ndarray
    n_rate_exclude: int
    n_rate_all: int
    n_total_exclude: int
    n_total_all: int
    per_gemm_counts: list[int]
    censor_c: tuple[float, ...]
    censor_n: tuple[int, ...]
    summary: TruncationSummary


@dataclass(frozen=True, slots=True)
class GpdFit:
    xi: float
    sigma: float
    nll: float
    xi_ci: tuple[float, float]
    sigma_ci: tuple[float, float]
    n_exceedances_fit: int
    n_exceedances_rate: int
    n_total: int
    u: float
    label: str


def _decode_top64(row: dict[str, Any]) -> np.ndarray:
    raw = row["r_top64"]
    out = np.empty(len(raw), dtype=np.float64)
    for i, item in enumerate(raw):
        if isinstance(item, dict):
            out[i] = decode_f64(item)
        else:
            out[i] = float(item)
    return out


def load_pass2(path: Path) -> tuple[list[dict[str, Any]], dict[str, float] | None]:
    frozen: dict[str, float] | None = None
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rtype = obj.get("record_type")
            if rtype == "metadata":
                if "pot_threshold_p99" in obj:
                    frozen = {
                        "p95": decode_f64(obj["pot_threshold_p95"]),
                        "p99": decode_f64(obj["pot_threshold_p99"]),
                        "p999": decode_f64(obj["pot_threshold_p999"]),
                    }
                continue
            if rtype == "pot_thresholds":
                frozen = {
                    "p95": decode_f64(obj["p95"]),
                    "p99": decode_f64(obj["p99"]),
                    "p999": decode_f64(obj["p999"]),
                }
                continue
            if obj.get("phase") != "fpr_clean":
                continue
            if int(obj.get("pass", 0)) != 2:
                continue
            samples.append(obj)
    return samples, frozen


def collect_exceedances(
    samples: list[dict[str, Any]],
    *,
    u: float,
    count_key: str,
    threshold_name: str,
) -> ExceedanceCollection:
    """Collect exceedances from r_top64 without raising on n_above > 64.

    When n_above > 64, the 64 recovered values are kept and
    (n_above - 64) unrecovered exceedances are recorded as right-censored
    on (0, c] with c = min(recovered) - u (all recovered lie above u).
    """
    y_exclude: list[float] = []
    y_all_rec: list[float] = []
    n_rate_exclude = 0
    n_rate_all = 0
    n_total_exclude = 0
    n_total_all = 0
    per_gemm_counts: list[int] = []
    censor_c: list[float] = []
    censor_n: list[int] = []
    n_over_64 = 0
    max_above = 0

    for row in samples:
        if count_key not in row:
            msg = f"pass-2 row missing {count_key!r} (sample_index={row.get('sample_index')})"
            raise RuntimeError(msg)
        n_above = int(row[count_key])
        per_gemm_counts.append(n_above)
        max_above = max(max_above, n_above)
        n_total_all += M_DIM
        n_rate_all += n_above

        top = _decode_top64(row)
        above = top[top > u]
        truncated = n_above > TOP_TAIL_K
        if truncated:
            n_over_64 += 1
            if above.size != TOP_TAIL_K:
                msg = (
                    f"truncated sample expected {TOP_TAIL_K} recovered values > u, "
                    f"got {above.size} (sample_index={row.get('sample_index')})"
                )
                raise RuntimeError(msg)
            c = float(np.min(above) - u)
            if c <= 0.0:
                msg = (
                    f"censorship bound c={c} not positive "
                    f"(sample_index={row.get('sample_index')})"
                )
                raise RuntimeError(msg)
            n_c = n_above - TOP_TAIL_K
            censor_c.append(c)
            censor_n.append(n_c)
            for value in above:
                y_all_rec.append(float(value - u))
            # excluded from y_exclude / n_rate_exclude / n_total_exclude
            continue

        if above.size != n_above:
            msg = (
                f"top64 above u mismatch: found {above.size} values > {u}, "
                f"n_above={n_above} (sample_index={row.get('sample_index')})"
            )
            raise RuntimeError(msg)
        n_total_exclude += M_DIM
        n_rate_exclude += n_above
        for value in above:
            y = float(value - u)
            y_exclude.append(y)
            y_all_rec.append(y)

    n_samples = len(samples)
    mean_above = (n_rate_all / n_samples) if n_samples else 0.0
    summary = TruncationSummary(
        threshold_name=threshold_name,
        u=u,
        mean_above=mean_above,
        max_above=max_above,
        n_over_64=n_over_64,
        n_samples=n_samples,
    )
    return ExceedanceCollection(
        y_exclude=np.asarray(y_exclude, dtype=np.float64),
        y_with_truncated_recovered=np.asarray(y_all_rec, dtype=np.float64),
        n_rate_exclude=n_rate_exclude,
        n_rate_all=n_rate_all,
        n_total_exclude=n_total_exclude,
        n_total_all=n_total_all,
        per_gemm_counts=per_gemm_counts,
        censor_c=tuple(censor_c),
        censor_n=tuple(censor_n),
        summary=summary,
    )


def gpd_cdf(xi: float, sigma: float, y: float) -> float:
    if y < 0.0 or sigma <= 0.0:
        return 0.0
    if abs(xi) < 1e-12:
        return float(1.0 - math.exp(-y / sigma))
    z = 1.0 + xi * y / sigma
    if z <= 0.0:
        return math.nan
    return float(1.0 - z ** (-1.0 / xi))


def gpd_nll(xi: float, sigma: float, y: np.ndarray) -> float:
    """Negative log-likelihood of exact GPD exceedances y > 0."""
    if sigma <= 0.0 or y.size < 1:
        return math.inf
    if abs(xi) < 1e-12:
        return float(y.size * math.log(sigma) + np.sum(y) / sigma)
    z = 1.0 + xi * y / sigma
    if np.any(z <= 0.0):
        return math.inf
    return float(y.size * math.log(sigma) + (1.0 + 1.0 / xi) * np.sum(np.log(z)))


def gpd_nll_censored(
    xi: float,
    sigma: float,
    y_exact: np.ndarray,
    censor_c: tuple[float, ...],
    censor_n: tuple[int, ...],
) -> float:
    """Exact exceedances plus right-censored counts on (0, c]."""
    if y_exact.size < 1 and not censor_n:
        return math.inf
    nll = 0.0
    if y_exact.size:
        nll = gpd_nll(xi, sigma, y_exact)
        if not math.isfinite(nll):
            return math.inf
    for c, n_c in zip(censor_c, censor_n, strict=True):
        if n_c < 1:
            continue
        fc = gpd_cdf(xi, sigma, c)
        if not math.isfinite(fc) or fc <= 0.0:
            return math.inf
        nll -= n_c * math.log(fc)
    return float(nll)


def _y_max_for_support(
    y: np.ndarray, censor_c: tuple[float, ...],
) -> float:
    parts = [float(np.max(y))] if y.size else []
    parts.extend(censor_c)
    if not parts:
        return 0.0
    return max(parts)


def _optimize_sigma(
    xi: float,
    y: np.ndarray,
    *,
    censor_c: tuple[float, ...] = (),
    censor_n: tuple[int, ...] = (),
) -> float:
    y_max = _y_max_for_support(y, censor_c)
    lo = max(1e-30, (-xi * y_max) * (1.0 + 1e-9) if xi < 0.0 else 1e-30)
    mean = float(np.mean(y)) if y.size else (censor_c[0] if censor_c else 1e-7)
    # Coarse log-grid then golden-section refine (keeps large-n fits tractable).
    candidates = np.geomspace(lo, max(lo * 1e8, mean * 1e4), 48)
    best_s = float(candidates[0])
    best_nll = gpd_nll_censored(xi, best_s, y, censor_c, censor_n)
    for sigma in candidates:
        nll = gpd_nll_censored(xi, float(sigma), y, censor_c, censor_n)
        if nll < best_nll:
            best_nll = nll
            best_s = float(sigma)
    left = max(best_s / 3.0, lo * (1.0 + 1e-12))
    right = best_s * 3.0
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    for _ in range(40):
        m1 = right - (right - left) / phi
        m2 = left + (right - left) / phi
        n1 = gpd_nll_censored(xi, m1, y, censor_c, censor_n)
        n2 = gpd_nll_censored(xi, m2, y, censor_c, censor_n)
        if n1 < n2:
            right = m2
        else:
            left = m1
    return float(0.5 * (left + right))


def fit_gpd_mle(
    y: np.ndarray,
    *,
    n_total: int,
    n_rate: int,
    u: float,
    label: str,
    censor_c: tuple[float, ...] = (),
    censor_n: tuple[int, ...] = (),
) -> GpdFit:
    n_fit = int(y.size) + int(sum(censor_n))
    if n_fit < 10:
        msg = f"too few exceedances for GPD MLE: {n_fit}"
        raise RuntimeError(msg)
    print(f"fitting {label}: n_exact={y.size} n_censored={sum(censor_n)} ...", flush=True)
    # Coarse xi grid, then local refine.
    xi_grid = np.linspace(-0.6, 0.4, 101)
    best_xi = 0.0
    best_sigma = float(np.mean(y)) if y.size else 1e-7
    best_nll = math.inf
    for xi in xi_grid:
        sigma = _optimize_sigma(float(xi), y, censor_c=censor_c, censor_n=censor_n)
        nll = gpd_nll_censored(float(xi), sigma, y, censor_c, censor_n)
        if nll < best_nll:
            best_nll = nll
            best_xi = float(xi)
            best_sigma = sigma
    lo_xi = best_xi - 0.04
    hi_xi = best_xi + 0.04
    for _ in range(35):
        m1 = lo_xi + (hi_xi - lo_xi) / 3.0
        m2 = hi_xi - (hi_xi - lo_xi) / 3.0
        s1 = _optimize_sigma(m1, y, censor_c=censor_c, censor_n=censor_n)
        s2 = _optimize_sigma(m2, y, censor_c=censor_c, censor_n=censor_n)
        n1 = gpd_nll_censored(m1, s1, y, censor_c, censor_n)
        n2 = gpd_nll_censored(m2, s2, y, censor_c, censor_n)
        if n1 < n2:
            hi_xi = m2
            if n1 < best_nll:
                best_nll, best_xi, best_sigma = n1, m1, s1
        else:
            lo_xi = m1
            if n2 < best_nll:
                best_nll, best_xi, best_sigma = n2, m2, s2
    best_sigma = _optimize_sigma(
        best_xi, y, censor_c=censor_c, censor_n=censor_n
    )
    best_nll = gpd_nll_censored(best_xi, best_sigma, y, censor_c, censor_n)

    xi_ci = _profile_ci_xi(
        y, best_xi, best_nll, censor_c=censor_c, censor_n=censor_n
    )
    sigma_ci = _profile_ci_sigma(
        y, best_sigma, best_nll, censor_c=censor_c, censor_n=censor_n
    )
    return GpdFit(
        xi=best_xi,
        sigma=best_sigma,
        nll=best_nll,
        xi_ci=xi_ci,
        sigma_ci=sigma_ci,
        n_exceedances_fit=n_fit,
        n_exceedances_rate=int(n_rate),
        n_total=int(n_total),
        u=float(u),
        label=label,
    )


def _profile_ci_xi(
    y: np.ndarray,
    xi_hat: float,
    nll_hat: float,
    *,
    censor_c: tuple[float, ...] = (),
    censor_n: tuple[int, ...] = (),
) -> tuple[float, float]:
    def profile(xi: float) -> float:
        sigma = _optimize_sigma(xi, y, censor_c=censor_c, censor_n=censor_n)
        return gpd_nll_censored(xi, sigma, y, censor_c, censor_n)

    def walk(direction: float) -> float:
        step = 0.01
        xi = xi_hat
        last_ok = xi_hat
        for _ in range(500):
            xi = xi + direction * step
            nll = profile(xi)
            if not math.isfinite(nll):
                step *= 0.5
                xi = last_ok
                if step < 1e-6:
                    break
                continue
            if 2.0 * (nll - nll_hat) <= CHI2_1_95:
                last_ok = xi
            else:
                lo, hi = (last_ok, xi) if direction > 0 else (xi, last_ok)
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    if 2.0 * (profile(mid) - nll_hat) <= CHI2_1_95:
                        if direction > 0:
                            lo = mid
                        else:
                            hi = mid
                    elif direction > 0:
                        hi = mid
                    else:
                        lo = mid
                return float(lo if direction > 0 else hi)
        return float(last_ok)

    return (walk(-1.0), walk(1.0))


def _profile_ci_sigma(
    y: np.ndarray,
    sigma_hat: float,
    nll_hat: float,
    *,
    censor_c: tuple[float, ...] = (),
    censor_n: tuple[int, ...] = (),
) -> tuple[float, float]:
    def profile(sigma: float) -> float:
        # Optimize xi for fixed sigma via coarse grid + local refine.
        best = math.inf
        best_xi = 0.0
        for xi in np.linspace(-0.6, 0.4, 41):
            nll = gpd_nll_censored(float(xi), sigma, y, censor_c, censor_n)
            if nll < best:
                best = nll
                best_xi = float(xi)
        lo, hi = best_xi - 0.03, best_xi + 0.03
        for _ in range(20):
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            n1 = gpd_nll_censored(m1, sigma, y, censor_c, censor_n)
            n2 = gpd_nll_censored(m2, sigma, y, censor_c, censor_n)
            if n1 < n2:
                hi = m2
                best = min(best, n1)
            else:
                lo = m1
                best = min(best, n2)
        return best

    def walk(direction: float) -> float:
        factor = 1.05 if direction > 0 else 1.0 / 1.05
        sigma = sigma_hat
        last_ok = sigma_hat
        for _ in range(200):
            sigma = sigma * factor
            if sigma <= 0.0:
                break
            nll = profile(sigma)
            if not math.isfinite(nll):
                break
            if 2.0 * (nll - nll_hat) <= CHI2_1_95:
                last_ok = sigma
            else:
                lo, hi = (last_ok, sigma) if direction > 0 else (sigma, last_ok)
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    if 2.0 * (profile(mid) - nll_hat) <= CHI2_1_95:
                        if direction > 0:
                            lo = mid
                        else:
                            hi = mid
                    elif direction > 0:
                        hi = mid
                    else:
                        lo = mid
                return float(lo if direction > 0 else hi)
        return float(last_ok)

    return (walk(-1.0), walk(1.0))


def gpd_return_level(fit: GpdFit, p: float) -> float:
    zeta = fit.n_exceedances_rate / fit.n_total
    if zeta <= 0.0 or p <= 0.0 or p >= 1.0:
        msg = f"invalid zeta={zeta} or p={p}"
        raise RuntimeError(msg)
    if p >= zeta:
        msg = f"target p={p} is not below empirical zeta_u={zeta}"
        raise RuntimeError(msg)
    xi = fit.xi
    sigma = fit.sigma
    u = fit.u
    if abs(xi) < 1e-12:
        return float(u - sigma * math.log(p / zeta))
    return float(u + (sigma / xi) * ((p / zeta) ** (-xi) - 1.0))


def score_flips_at_threshold(
    flips_path: Path,
    *,
    threshold: float,
    k: int = 4096,
) -> dict[tuple[str, int], tuple[int, int]]:
    counts: dict[tuple[str, int], list[bool]] = {}
    with flips_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("record_type") in {"metadata", "clean_max", "pot_thresholds"}:
                continue
            if obj.get("phase") == "clean":
                continue
            if "K" in obj and int(obj["K"]) != k:
                continue
            if "bit_class" not in obj or "n_flips" not in obj:
                continue
            if "r_max" not in obj:
                continue
            r_max = decode_f64(obj["r_max"])
            key = (str(obj["bit_class"]), int(obj["n_flips"]))
            counts.setdefault(key, []).append(bool(r_max > threshold))
    return {key: (sum(flags), len(flags)) for key, flags in counts.items()}


def _print_truncation_table(summaries: list[TruncationSummary]) -> None:
    print("--- truncation summary ---")
    print("| threshold | u | mean exceedances/GEMM | max | GEMMs over 64 |")
    print("| --- | --- | --- | --- | --- |")
    for s in summaries:
        print(
            f"| {s.threshold_name} | {s.u:.6e} | {s.mean_above:.4g} | "
            f"{s.max_above} | {s.n_over_64}/{s.n_samples} |"
        )


def _print_fit(fit: GpdFit) -> None:
    print(f"--- GPD MLE ({fit.label}) ---")
    print(f"xi = {fit.xi:.10g}")
    print(f"sigma = {fit.sigma:.10g}")
    print(f"xi_95ci = [{fit.xi_ci[0]:.10g}, {fit.xi_ci[1]:.10g}]")
    print(f"sigma_95ci = [{fit.sigma_ci[0]:.10g}, {fit.sigma_ci[1]:.10g}]")
    print(f"n_exceedances_fit = {fit.n_exceedances_fit}")
    print(f"n_exceedances_rate = {fit.n_exceedances_rate}")
    print(f"zeta_u = {fit.n_exceedances_rate / fit.n_total:.10g}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fpr_jsonl",
        nargs="?",
        default="data/fpr.jsonl",
        type=Path,
        help="FPR sweep JSONL (default: data/fpr.jsonl)",
    )
    parser.add_argument(
        "--flips",
        type=Path,
        default=None,
        help="Flip-matrix JSONL for step 6 (K=4096). Required for F4.",
    )
    parser.add_argument(
        "--clean-max-baseline",
        type=float,
        default=CLEAN_MAX_K4096,
        help="Observed clean max used for F2 ratio and F4 baseline scoring",
    )
    args = parser.parse_args(argv)

    path: Path = args.fpr_jsonl
    if not path.is_file():
        print(f"ERROR: FPR JSONL not found: {path}", file=sys.stderr)
        return 2

    samples, frozen = load_pass2(path)
    n_samples = len(samples)
    if n_samples < 1:
        print("ERROR: no pass-2 samples in JSONL", file=sys.stderr)
        return 2
    row_obs = 0
    for row in samples:
        shape = row.get("shape")
        if not isinstance(shape, list) or len(shape) != 3:
            print(
                f"ERROR: pass-2 row missing shape "
                f"(sample_index={row.get('sample_index')})",
                file=sys.stderr,
            )
            return 2
        if int(shape[0]) != M_DIM:
            print(
                f"ERROR: expected M={M_DIM}, got shape={shape} "
                f"(sample_index={row.get('sample_index')})",
                file=sys.stderr,
            )
            return 2
        row_obs += int(shape[0])
    n_row_observations = row_obs
    print(f"n_samples = {n_samples}")
    print(f"n_row_observations = {n_row_observations}")
    if n_row_observations != n_samples * M_DIM:
        print(
            f"ERROR: n_row_observations mismatch: {n_row_observations} != "
            f"{n_samples} * {M_DIM}",
            file=sys.stderr,
        )
        return 2
    if n_samples != 19_500:
        print(
            f"WARNING: expected 19500 pass-2 samples, got {n_samples}",
            file=sys.stderr,
        )

    p95 = frozen["p95"] if frozen else P95_FROZEN
    p99 = frozen["p99"] if frozen else P99_FROZEN
    p999 = frozen["p999"] if frozen else P999_FROZEN
    print(f"frozen_p95 = {p95:.6e}")
    print(f"frozen_p99 = {p99:.6e}")
    print(f"frozen_p999 = {p999:.6e}")

    col95 = collect_exceedances(
        samples, u=p95, count_key="n_above_p95", threshold_name="p95"
    )
    col99 = collect_exceedances(
        samples, u=p99, count_key="n_above_p99", threshold_name="p99"
    )
    col999 = collect_exceedances(
        samples, u=p999, count_key="n_above_p999", threshold_name="p999"
    )
    _print_truncation_table([col95.summary, col99.summary, col999.summary])

    # --- p999: no truncation expected ---
    if col999.summary.n_over_64 != 0:
        print(
            f"ERROR: p999 unexpected truncation: "
            f"{col999.summary.n_over_64} GEMMs over 64",
            file=sys.stderr,
        )
        return 2
    fit999 = fit_gpd_mle(
        col999.y_exclude,
        n_total=col999.n_total_all,
        n_rate=col999.n_rate_all,
        u=p999,
        label="p999",
    )
    _print_fit(fit999)
    x999 = gpd_return_level(fit999, TARGET_TAIL_PROB)

    # --- p99: exclude vs right-censored ---
    fit99_excl = fit_gpd_mle(
        col99.y_exclude,
        n_total=col99.n_total_exclude,
        n_rate=col99.n_rate_exclude,
        u=p99,
        label="p99_exclude_truncated",
    )
    fit99_cens = fit_gpd_mle(
        col99.y_with_truncated_recovered,
        n_total=col99.n_total_all,
        n_rate=col99.n_rate_all,
        u=p99,
        label="p99_right_censored",
        censor_c=col99.censor_c,
        censor_n=col99.censor_n,
    )
    _print_fit(fit99_excl)
    _print_fit(fit99_cens)
    xi_delta = abs(fit99_excl.xi - fit99_cens.xi)
    ci_width_excl = fit99_excl.xi_ci[1] - fit99_excl.xi_ci[0]
    ci_width_cens = fit99_cens.xi_ci[1] - fit99_cens.xi_ci[0]
    print("--- p99 exclude vs censored ---")
    print(f"xi_exclude = {fit99_excl.xi:.10g}")
    print(f"xi_censored = {fit99_cens.xi:.10g}")
    print(f"xi_abs_delta = {xi_delta:.10g}")
    print(f"xi_ci_width_exclude = {ci_width_excl:.10g}")
    print(f"xi_ci_width_censored = {ci_width_cens:.10g}")
    print(
        f"delta_over_ci_width_exclude = "
        f"{(xi_delta / ci_width_excl) if ci_width_excl > 0 else math.inf:.10g}"
    )
    print(
        f"delta_over_ci_width_censored = "
        f"{(xi_delta / ci_width_cens) if ci_width_cens > 0 else math.inf:.10g}"
    )
    print(f"n_truncated_gemms = {col99.summary.n_over_64}")
    print(f"n_unrecovered_exceedances = {sum(col99.censor_n)}")

    x99_excl = gpd_return_level(fit99_excl, TARGET_TAIL_PROB)
    x99_cens = gpd_return_level(fit99_cens, TARGET_TAIL_PROB)
    print("--- extrapolation (both p99 fits; not averaged) ---")
    print(f"target_tail_prob = {TARGET_TAIL_PROB}")
    print(f"extrapolated_threshold_p99_exclude = {x99_excl:.10g}")
    print(f"ratio_to_clean_max_p99_exclude = {x99_excl / args.clean_max_baseline:.10g}")
    print(f"extrapolated_threshold_p99_censored = {x99_cens:.10g}")
    print(
        f"ratio_to_clean_max_p99_censored = {x99_cens / args.clean_max_baseline:.10g}"
    )
    print(f"extrapolated_threshold_p999 = {x999:.10g}")
    print(f"ratio_to_clean_max_p999 = {x999 / args.clean_max_baseline:.10g}")
    print(f"clean_max = {args.clean_max_baseline:.10g}")

    # --- p95: NOT EVALUABLE ---
    print("--- sensitivity ---")
    print(
        "threshold_at_p95 = NOT_EVALUABLE  "
        f"(reason: top-64 capture; mean={col95.summary.mean_above:.4g}/GEMM, "
        f"n_over_64={col95.summary.n_over_64}/{col95.summary.n_samples})"
    )
    print(
        f"threshold_at_p99_exclude = {x99_excl:.10g}  "
        f"(xi={fit99_excl.xi:.10g}, sigma={fit99_excl.sigma:.10g})"
    )
    print(
        f"threshold_at_p99_censored = {x99_cens:.10g}  "
        f"(xi={fit99_cens.xi:.10g}, sigma={fit99_cens.sigma:.10g})"
    )
    print(
        f"threshold_at_p999 = {x999:.10g}  "
        f"(xi={fit999.xi:.10g}, sigma={fit999.sigma:.10g})"
    )
    # F3 partial: max/min among evaluable thresholds only (both p99 + p999)
    evaluable = [x99_excl, x99_cens, x999]
    sens_ratio = max(evaluable) / min(evaluable)
    print(f"sensitivity_max_min_ratio_evaluable = {sens_ratio:.10g}")
    print(
        "F3_status = PARTIALLY_EVALUABLE  "
        "(p99 and p999 compared; p95 not evaluable due to top-64 capture)"
    )

    # --- independence ---
    counts_arr = np.asarray(col99.per_gemm_counts, dtype=np.float64)
    obs_mean = float(np.mean(counts_arr))
    obs_var = float(np.var(counts_arr, ddof=1)) if counts_arr.size > 1 else 0.0
    binom_p = 0.01
    binom_mean = M_DIM * binom_p
    binom_var = M_DIM * binom_p * (1.0 - binom_p)
    var_ratio = obs_var / binom_var if binom_var > 0 else math.inf
    print("--- independence (n_above_p99 vs Binomial(4096, 0.01)) ---")
    print(f"observed_mean = {obs_mean:.10g}")
    print(f"observed_variance = {obs_var:.10g}")
    print(f"binomial_mean = {binom_mean:.10g}")
    print(f"binomial_variance = {binom_var:.10g}")
    print(f"variance_ratio = {var_ratio:.10g}")

    # --- flip re-score at both extrapolated p99 thresholds ---
    print("--- flip re-score ---")
    flip_excl = None
    flip_cens = None
    flip_base = None
    if args.flips is None or not args.flips.is_file():
        print("ERROR: --flips JSONL required for step 6 / F4", file=sys.stderr)
    else:
        flip_excl = score_flips_at_threshold(args.flips, threshold=x99_excl, k=4096)
        flip_cens = score_flips_at_threshold(args.flips, threshold=x99_cens, k=4096)
        flip_base = score_flips_at_threshold(
            args.flips, threshold=args.clean_max_baseline, k=4096
        )
        keys = sorted(set(flip_excl) | set(flip_base))
        for key in keys:
            d_b, n_b = flip_base.get(key, (0, 0))
            d_e, n_e = flip_excl.get(key, (0, 0))
            d_c, n_c = flip_cens.get(key, (0, 0))
            print(
                f"  {key[0]} n_flips={key[1]}: "
                f"clean_max {d_b}/{n_b} "
                f"({100.0 * d_b / n_b if n_b else 0:.2f}%), "
                f"p99_excl {d_e}/{n_e} "
                f"({100.0 * d_e / n_e if n_e else 0:.2f}%), "
                f"p99_cens {d_c}/{n_c} "
                f"({100.0 * d_c / n_c if n_c else 0:.2f}%)"
            )

    print("--- PASTE_BLOCK_BEGIN ---")
    print(f"truncation_p95_mean = {col95.summary.mean_above:.10g}")
    print(f"truncation_p95_max = {col95.summary.max_above}")
    print(f"truncation_p95_n_over_64 = {col95.summary.n_over_64}")
    print(f"truncation_p99_mean = {col99.summary.mean_above:.10g}")
    print(f"truncation_p99_max = {col99.summary.max_above}")
    print(f"truncation_p99_n_over_64 = {col99.summary.n_over_64}")
    print(f"truncation_p999_mean = {col999.summary.mean_above:.10g}")
    print(f"truncation_p999_max = {col999.summary.max_above}")
    print(f"truncation_p999_n_over_64 = {col999.summary.n_over_64}")
    print(f"xi_p99_exclude = {fit99_excl.xi:.10g}")
    print(f"sigma_p99_exclude = {fit99_excl.sigma:.10g}")
    print(
        f"xi_95ci_p99_exclude = "
        f"[{fit99_excl.xi_ci[0]:.10g}, {fit99_excl.xi_ci[1]:.10g}]"
    )
    print(
        f"sigma_95ci_p99_exclude = "
        f"[{fit99_excl.sigma_ci[0]:.10g}, {fit99_excl.sigma_ci[1]:.10g}]"
    )
    print(f"xi_p99_censored = {fit99_cens.xi:.10g}")
    print(f"sigma_p99_censored = {fit99_cens.sigma:.10g}")
    print(
        f"xi_95ci_p99_censored = "
        f"[{fit99_cens.xi_ci[0]:.10g}, {fit99_cens.xi_ci[1]:.10g}]"
    )
    print(
        f"sigma_95ci_p99_censored = "
        f"[{fit99_cens.sigma_ci[0]:.10g}, {fit99_cens.sigma_ci[1]:.10g}]"
    )
    print(f"xi_abs_delta = {xi_delta:.10g}")
    print(f"xi_ci_width_exclude = {ci_width_excl:.10g}")
    print(f"xi_ci_width_censored = {ci_width_cens:.10g}")
    print(f"xi_p999 = {fit999.xi:.10g}")
    print(f"sigma_p999 = {fit999.sigma:.10g}")
    print(f"extrapolated_threshold_p99_exclude = {x99_excl:.10g}")
    print(f"extrapolated_threshold_p99_censored = {x99_cens:.10g}")
    print(f"extrapolated_threshold_p999 = {x999:.10g}")
    print(f"ratio_to_clean_max_p99_exclude = {x99_excl / args.clean_max_baseline:.10g}")
    print(
        f"ratio_to_clean_max_p99_censored = {x99_cens / args.clean_max_baseline:.10g}"
    )
    print(f"ratio_to_clean_max_p999 = {x999 / args.clean_max_baseline:.10g}")
    print("threshold_at_p95 = NOT_EVALUABLE")
    print(f"sensitivity_max_min_ratio_evaluable = {sens_ratio:.10g}")
    print(
        "F3_status = PARTIALLY_EVALUABLE  "
        "(p99 and p999 compared; p95 not evaluable due to top-64 capture)"
    )
    print(f"variance_ratio = {var_ratio:.10g}")
    print(f"observed_mean = {obs_mean:.10g}")
    print(f"observed_variance = {obs_var:.10g}")
    print(f"binomial_mean = {binom_mean:.10g}")
    print(f"binomial_variance = {binom_var:.10g}")
    if flip_excl is not None and flip_cens is not None and flip_base is not None:
        for tag, scored in (
            ("clean_max", flip_base),
            ("p99_exclude", flip_excl),
            ("p99_censored", flip_cens),
        ):
            sign = scored.get(("SIGN", 1))
            if sign and sign[1]:
                print(
                    f"SIGN_1_{tag} = {sign[0]}/{sign[1]} "
                    f"({100.0 * sign[0] / sign[1]:.4g}%)"
                )
    print("--- PASTE_BLOCK_END ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
