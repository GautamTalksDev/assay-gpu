#!/usr/bin/env python3
"""GPD peaks-over-threshold fit for residual-v3 FPR (method locked in RESIDUAL.md).

Does not edit docs/RESIDUAL.md. Uses only numpy + torch (no scipy).
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
# Frozen pass-1 POT thresholds (prompt / pass-1 finalize). Verified against JSONL.
P95_FROZEN = 1.057096e-06
P99_FROZEN = 1.392735e-06
P999_FROZEN = 1.776693e-06
CLEAN_MAX_K4096 = 2.684525e-06
# Per-row FPR for per-GEMM 1e-6 under independence: 1 - (1 - 1e-6)^(1/4096)
TARGET_TAIL_PROB = 2.44e-10
CHI2_1_95 = 3.841458820694124  # chi-square 95% critical value, 1 df
TOP_TAIL_K = 64


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
    """Load pass-2 fpr_clean rows and optional frozen thresholds from metadata."""
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
) -> tuple[np.ndarray, int, list[int]]:
    """Exceedance magnitudes (r - u) from r_top64; rate count from n_above_*."""
    ys: list[float] = []
    n_rate = 0
    per_gemm_counts: list[int] = []
    for row in samples:
        if count_key not in row:
            msg = f"pass-2 row missing {count_key!r} (sample_index={row.get('sample_index')})"
            raise RuntimeError(msg)
        n_above = int(row[count_key])
        per_gemm_counts.append(n_above)
        n_rate += n_above
        if n_above > TOP_TAIL_K:
            msg = (
                f"n_above={n_above} > top-{TOP_TAIL_K} at sample_index="
                f"{row.get('sample_index')}; cannot recover all exceedances"
            )
            raise RuntimeError(msg)
        top = _decode_top64(row)
        above = top[top > u]
        if above.size != n_above:
            msg = (
                f"top64 above u mismatch: found {above.size} values > {u}, "
                f"n_above={n_above} (sample_index={row.get('sample_index')})"
            )
            raise RuntimeError(msg)
        for value in above:
            ys.append(float(value - u))
    return np.asarray(ys, dtype=np.float64), n_rate, per_gemm_counts


def gpd_nll(xi: float, sigma: float, y: np.ndarray) -> float:
    """Negative log-likelihood of GPD exceedances y > 0."""
    if sigma <= 0.0 or y.size < 1:
        return math.inf
    if abs(xi) < 1e-12:
        return float(y.size * math.log(sigma) + np.sum(y) / sigma)
    z = 1.0 + xi * y / sigma
    if np.any(z <= 0.0):
        return math.inf
    return float(y.size * math.log(sigma) + (1.0 + 1.0 / xi) * np.sum(np.log(z)))


def _optimize_sigma(xi: float, y: np.ndarray) -> float:
    """1-D minimize nll over sigma > max(0, -xi * y_max) for fixed xi."""
    y_max = float(np.max(y))
    lo = max(1e-30, (-xi * y_max) * (1.0 + 1e-9) if xi < 0.0 else 1e-30)
    # Bracket: start near mean of exceedances
    mean = float(np.mean(y))
    candidates = np.unique(
        np.concatenate(
            [
                np.geomspace(lo, max(lo * 1e6, mean * 100.0), 80),
                np.array([mean, mean * 0.5, mean * 2.0], dtype=np.float64),
            ]
        )
    )
    candidates = candidates[candidates > lo]
    best_s = float(candidates[0])
    best_nll = gpd_nll(xi, best_s, y)
    for sigma in candidates:
        nll = gpd_nll(xi, float(sigma), y)
        if nll < best_nll:
            best_nll = nll
            best_s = float(sigma)
    # Local refine by ternary-ish shrinking
    left = best_s / 2.0
    right = best_s * 2.0
    left = max(left, lo * (1.0 + 1e-12))
    for _ in range(60):
        m1 = left + (right - left) / 3.0
        m2 = right - (right - left) / 3.0
        n1 = gpd_nll(xi, m1, y)
        n2 = gpd_nll(xi, m2, y)
        if n1 < n2:
            right = m2
        else:
            left = m1
    sigma = 0.5 * (left + right)
    return float(sigma)


def fit_gpd_mle(y: np.ndarray, *, n_total: int, n_rate: int, u: float) -> GpdFit:
    if y.size < 10:
        msg = f"too few exceedances for GPD MLE: {y.size}"
        raise RuntimeError(msg)
    # xi grid: allow mild positive (falsifier region) and negative
    xi_grid = np.linspace(-0.8, 0.8, 321)
    best_xi = 0.0
    best_sigma = float(np.mean(y))
    best_nll = math.inf
    for xi in xi_grid:
        sigma = _optimize_sigma(float(xi), y)
        nll = gpd_nll(float(xi), sigma, y)
        if nll < best_nll:
            best_nll = nll
            best_xi = float(xi)
            best_sigma = sigma
    # Refine xi locally
    lo_xi = best_xi - 0.05
    hi_xi = best_xi + 0.05
    for _ in range(40):
        m1 = lo_xi + (hi_xi - lo_xi) / 3.0
        m2 = hi_xi - (hi_xi - lo_xi) / 3.0
        s1 = _optimize_sigma(m1, y)
        s2 = _optimize_sigma(m2, y)
        n1 = gpd_nll(m1, s1, y)
        n2 = gpd_nll(m2, s2, y)
        if n1 < n2:
            hi_xi = m2
            if n1 < best_nll:
                best_nll = n1
                best_xi = m1
                best_sigma = s1
        else:
            lo_xi = m1
            if n2 < best_nll:
                best_nll = n2
                best_xi = m2
                best_sigma = s2
    best_sigma = _optimize_sigma(best_xi, y)
    best_nll = gpd_nll(best_xi, best_sigma, y)

    xi_ci = _profile_ci_xi(y, best_xi, best_nll)
    sigma_ci = _profile_ci_sigma(y, best_sigma, best_nll)

    return GpdFit(
        xi=best_xi,
        sigma=best_sigma,
        nll=best_nll,
        xi_ci=xi_ci,
        sigma_ci=sigma_ci,
        n_exceedances_fit=int(y.size),
        n_exceedances_rate=int(n_rate),
        n_total=int(n_total),
        u=float(u),
    )


def _profile_ci_xi(y: np.ndarray, xi_hat: float, nll_hat: float) -> tuple[float, float]:
    def profile(xi: float) -> float:
        sigma = _optimize_sigma(xi, y)
        return gpd_nll(xi, sigma, y)

    # Walk left / right until 2*delta_nll exceeds chi2
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
                # binary search between last_ok and xi
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
    y: np.ndarray, sigma_hat: float, nll_hat: float
) -> tuple[float, float]:
    def profile(sigma: float) -> float:
        # optimize xi on a local grid for fixed sigma
        best = math.inf
        for xi in np.linspace(-0.8, 0.8, 161):
            nll = gpd_nll(float(xi), sigma, y)
            if nll < best:
                best = nll
        return best

    def walk(direction: float) -> float:
        # multiplicative steps
        factor = 1.02 if direction > 0 else 1.0 / 1.02
        sigma = sigma_hat
        last_ok = sigma_hat
        for _ in range(500):
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


def gpd_return_level(fit: GpdFit, p: float) -> float:
    """Threshold x such that P(R > x) = p, via POT formula."""
    zeta = fit.n_exceedances_rate / fit.n_total
    if zeta <= 0.0 or p <= 0.0 or p >= 1.0:
        msg = f"invalid zeta={zeta} or p={p}"
        raise RuntimeError(msg)
    if p >= zeta:
        # requested probability not in the tail above u
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
    """Detection counts (detected, n) per (bit_class, n_flips) at absolute threshold.

    Accepts sweep-v3-flips.jsonl or sweep-v3-kscale.jsonl (phase=flip, K filter).
    """
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fpr_jsonl",
        nargs="?",
        default="/kaggle/working/fpr.jsonl",
        type=Path,
        help="FPR sweep JSONL (default: /kaggle/working/fpr.jsonl)",
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
            msg = f"pass-2 row missing shape (sample_index={row.get('sample_index')})"
            print(f"ERROR: {msg}", file=sys.stderr)
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

    # Prefer frozen thresholds from file; fall back to prompt constants.
    p95 = frozen["p95"] if frozen else P95_FROZEN
    p99 = frozen["p99"] if frozen else P99_FROZEN
    p999 = frozen["p999"] if frozen else P999_FROZEN
    print(f"frozen_p95 = {p95:.6e}")
    print(f"frozen_p99 = {p99:.6e}")
    print(f"frozen_p999 = {p999:.6e}")
    for name, got, expect in (
        ("p95", p95, P95_FROZEN),
        ("p99", p99, P99_FROZEN),
        ("p999", p999, P999_FROZEN),
    ):
        if abs(got - expect) / expect > 1e-5:
            print(
                f"WARNING: frozen {name}={got:.6e} differs from locked "
                f"{expect:.6e}",
                file=sys.stderr,
            )

    # --- Step 2: primary fit at p99 ---
    y99, n_rate99, counts99 = collect_exceedances(
        samples, u=p99, count_key="n_above_p99"
    )
    fit99 = fit_gpd_mle(y99, n_total=n_row_observations, n_rate=n_rate99, u=p99)
    print("--- GPD MLE at p99 ---")
    print(f"xi = {fit99.xi:.10g}")
    print(f"sigma = {fit99.sigma:.10g}")
    print(f"xi_95ci = [{fit99.xi_ci[0]:.10g}, {fit99.xi_ci[1]:.10g}]")
    print(f"sigma_95ci = [{fit99.sigma_ci[0]:.10g}, {fit99.sigma_ci[1]:.10g}]")
    print(f"n_exceedances_fit = {fit99.n_exceedances_fit}")
    print(f"n_exceedances_rate = {fit99.n_exceedances_rate}")
    print(f"zeta_u = {fit99.n_exceedances_rate / fit99.n_total:.10g}")

    # --- Step 3: extrapolate ---
    x_ext = gpd_return_level(fit99, TARGET_TAIL_PROB)
    ratio_clean = x_ext / args.clean_max_baseline
    print("--- extrapolation ---")
    print(f"target_tail_prob = {TARGET_TAIL_PROB}")
    print(f"extrapolated_threshold = {x_ext:.10g}")
    print(f"ratio_to_clean_max = {ratio_clean:.10g}")
    print(f"clean_max = {args.clean_max_baseline:.10g}")

    # --- Step 4: sensitivity ---
    y95, n_rate95, _ = collect_exceedances(samples, u=p95, count_key="n_above_p95")
    y999, n_rate999, _ = collect_exceedances(
        samples, u=p999, count_key="n_above_p999"
    )
    fit95 = fit_gpd_mle(y95, n_total=n_row_observations, n_rate=n_rate95, u=p95)
    fit999 = fit_gpd_mle(y999, n_total=n_row_observations, n_rate=n_rate999, u=p999)
    x95 = gpd_return_level(fit95, TARGET_TAIL_PROB)
    x999 = gpd_return_level(fit999, TARGET_TAIL_PROB)
    sens = [x95, x_ext, x999]
    sens_ratio = max(sens) / min(sens)
    print("--- sensitivity ---")
    print(f"threshold_at_p95 = {x95:.10g}  (xi={fit95.xi:.10g}, sigma={fit95.sigma:.10g})")
    print(f"threshold_at_p99 = {x_ext:.10g}  (xi={fit99.xi:.10g}, sigma={fit99.sigma:.10g})")
    print(
        f"threshold_at_p999 = {x999:.10g}  (xi={fit999.xi:.10g}, sigma={fit999.sigma:.10g})"
    )
    print(f"sensitivity_max_min_ratio = {sens_ratio:.10g}")

    # --- Step 5: independence via n_above_p99 counts ---
    counts_arr = np.asarray(counts99, dtype=np.float64)
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

    # --- Step 6: re-score flips ---
    print("--- flip re-score at extrapolated threshold ---")
    if args.flips is None or not args.flips.is_file():
        print("ERROR: --flips JSONL required for step 6 / F4", file=sys.stderr)
        flip_gpd: dict[tuple[str, int], tuple[int, int]] | None = None
        flip_base: dict[tuple[str, int], tuple[int, int]] | None = None
    else:
        flip_gpd = score_flips_at_threshold(args.flips, threshold=x_ext, k=4096)
        flip_base = score_flips_at_threshold(
            args.flips, threshold=args.clean_max_baseline, k=4096
        )
        for key in sorted(flip_gpd):
            d_g, n_g = flip_gpd[key]
            d_b, n_b = flip_base.get(key, (0, 0))
            print(
                f"  {key[0]} n_flips={key[1]}: "
                f"clean_max {d_b}/{n_b} ({100.0 * d_b / n_b if n_b else 0:.2f}%), "
                f"gpd {d_g}/{n_g} ({100.0 * d_g / n_g if n_g else 0:.2f}%)"
            )

    # Machine-readable block for RESULTS-FPR.md paste
    print("--- PASTE_BLOCK_BEGIN ---")
    print(f"xi = {fit99.xi:.10g}")
    print(f"sigma = {fit99.sigma:.10g}")
    print(f"xi_95ci = [{fit99.xi_ci[0]:.10g}, {fit99.xi_ci[1]:.10g}]")
    print(f"sigma_95ci = [{fit99.sigma_ci[0]:.10g}, {fit99.sigma_ci[1]:.10g}]")
    print(f"extrapolated_threshold = {x_ext:.10g}")
    print(f"ratio_to_clean_max = {ratio_clean:.10g}")
    print(f"threshold_at_p95 = {x95:.10g}")
    print(f"threshold_at_p99 = {x_ext:.10g}")
    print(f"threshold_at_p999 = {x999:.10g}")
    print(f"sensitivity_max_min_ratio = {sens_ratio:.10g}")
    print(f"variance_ratio = {var_ratio:.10g}")
    print(f"observed_mean = {obs_mean:.10g}")
    print(f"observed_variance = {obs_var:.10g}")
    print(f"binomial_mean = {binom_mean:.10g}")
    print(f"binomial_variance = {binom_var:.10g}")
    if flip_gpd is not None and flip_base is not None:
        sign_g = flip_gpd.get(("SIGN", 1))
        sign_b = flip_base.get(("SIGN", 1))
        if sign_g and sign_b and sign_g[1] and sign_b[1]:
            print(
                f"SIGN_1_clean_max = {sign_b[0]}/{sign_b[1]} "
                f"({100.0 * sign_b[0] / sign_b[1]:.4g}%)"
            )
            print(
                f"SIGN_1_gpd = {sign_g[0]}/{sign_g[1]} "
                f"({100.0 * sign_g[0] / sign_g[1]:.4g}%)"
            )
    print("--- PASTE_BLOCK_END ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
