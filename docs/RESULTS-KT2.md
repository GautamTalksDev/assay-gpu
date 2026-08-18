# RESULTS-KT2

**Verdict: FAIL.** `assay watch` has not shown end-to-end
assay-protected inference overhead at or below 10% on a representative
transformer workload, on any GPU.

The bar is the same sentence as `README.md` at CP-0. It was not moved
to 25% or any other number. No sampling rate was invented to claim a
5% overhead.

## 1. Method

KT-2 as written at CP-0:

> If end-to-end assay-protected inference overhead exceeds 10% on
> representative transformer workloads, nobody will run it continuously
> and the paid tier has no product. This project stops.

The measurement that would satisfy the bar:

1. A real open-weights transformer, inference, tokens/sec.
2. Baseline: the same script, no hooks.
3. `assay watch --every N` at several N (including 1, and sparser).
4. Overhead = `(baseline_tok_s - watch_tok_s) / baseline_tok_s`.
5. Repeat on at least two GPU models.
6. PASS only if some useful N has overhead `<= 10%` on those models.

`--every` has no default. A default "chosen so overhead stays under 5%"
would be a guessed cutoff. This repository has no measured tokens/sec
trace to derive N from (`data/` contains no watch overhead files).

## 2. GPUs used

None. This workstation had no NVIDIA GPU. No open-weights model was
downloaded (watch makes zero network requests; `transformers` is not a
dependency). Budget spent: `$0` (`docs/BUDGET.md`).

| GPU model | n tokens/sec pairs | source |
| --- | --- | --- |
| (none) | 0 | — |

## 3. What was measured on CPU

`tests/test_watch.py` (not KT-2):

- `nn.Linear` forward output is unchanged when hooks run.
- Uncharacterized noisefloor → INCONCLUSIVE, not FAIL.
- `--every N` checks 1 in N GEMMs (deterministic counter, not RNG).
- A mocked ABFT FAIL and a raising `check_gemm` are recorded events.
  The host forward still returns a tensor. No exception enters the
  user module.
- `nn.MultiheadAttention` is hooked as measurement-only
  (no noisefloor-v1 attention ABFT).

Those tests do not produce a tokens/sec delta.

## 4. Verdict against the bar

There is no interval for e2e overhead. Zero GPU samples means the
claim "overhead <= 10%" has not been shown at any sampling rate,
including the most expensive (`--every 1`) and any sparser rate.

**FAIL.**

`assay watch` is therefore not a shipped product. `assay run` remains
the acceptance-test CLI. The watch command exists only as a
measurement harness so a later GPU run can fill this file without
changing the 10% bar.

## 5. What this is not

- Not a 5% sampling default.
- Not a 25% bar.
- Not a PASS because CPU hooks "seem cheap."
- Not an open-weights model result. That DoD item is unmet here.
