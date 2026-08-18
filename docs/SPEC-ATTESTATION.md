# attestation-v1

Version: **attestation-v1**. This document is enough for someone else to
implement a verifier. Implementation: `src/assay/report/`. Command:
`assay verify`.

A report is evidence a third party can check **without a network**,
without trusting the GPU provider, and without calling Assay's servers.
The signature proves the JSON was not altered after signing. It does not
by itself prove the measurements are true.

v1 does **not** include an Assay issuance service. `assay run --report`
only emits **self-signed** reports.

## Document shape

A report is a UTF-8 JSON object with exactly these top-level keys:

```json
{
  "spec": "attestation-v1",
  "body": { },
  "signature": { }
}
```

`spec` MUST be the string `attestation-v1`. Unknown specs MUST be
rejected.

### body

Required keys (no extras are required of a verifier; extra keys MUST be
preserved when canonicalizing because they are covered by the signature):

| Key | Type | Meaning |
| --- | --- | --- |
| `tool_version` | string | `assay-gpu` package version (e.g. `0.0.0`) |
| `noisefloor_spec_version` | string | From `data/noisefloor/methodology-v1.json` `spec_id` |
| `workload_suite_version` | string | `workload-suite-v1` (see `docs/WORKLOADS.md`) |
| `issued_at_utc` | string | RFC 3339 UTC with microseconds: `YYYY-MM-DDTHH:MM:SS.ffffffZ` |
| `run_id` | string | 64 lowercase hex chars; see run_id below |
| `network_requests` | integer | MUST be `0` |
| `network_guarantee` | string | The `assay run` zero-network sentence |
| `probe` | object | Full environment probe (GPU, driver, CUDA/ROCm, VBIOS, ECC, MIG, clocks, power, thermal, virtualization, shared-instance reasons) |
| `gpu_model` | string or null | `model_key` of the selected GPU |
| `budget` | object | Time-budget name and shape filters |
| `noisefloor` | object | `{ "status": "characterized" \| "uncharacterized", "reason": string }` |
| `cases` | array | Per-workload rows (below) |
| `status` | string | `PASS`, `FAIL`, or `INCONCLUSIVE` |
| `exit_code` | integer | `0` PASS, `1` FAIL, `2` INCONCLUSIVE |
| `verdict_reason` | string | Human reason for `status` |
| `elapsed_s_hex` | string | Wall time of the run as a Python `float.hex()` string |
| `reference_catalog_sha256` | string | 64 lowercase hex chars; see catalog hash below |
| `independence` | object | `{ "mode": "...", "disclaimer": "..." }` |

`issued_at_utc` and `run_id` are metadata. They MUST NOT be used as a
detection threshold or to invent a PASS/FAIL.

Each `cases[]` row includes at least:

| Key | Type |
| --- | --- |
| `workload` | string (`W01`…`W07`) |
| `case` | string |
| `shape` | array of integers |
| `dtype` | string or null |
| `status` | `PASS` \| `FAIL` \| `INCONCLUSIVE` |
| `reason` | string |
| `residual_hex` | string or null |
| `residual_decimal` | string or null |
| `threshold_hex` | string or null |
| `threshold_decimal` | string or null |
| `sample_max_hex` | string or null |
| `n_samples` | integer or null |
| `min_samples` | integer or null |
| `noisefloor_spec_version` | string or null |
| `golden_max_abs_error_hex` | string or null |
| `wall_time_s_hex` | string |
| `skipped` | boolean |
| `skip_reason` | string or null |
| `counts_toward_verdict` | boolean |

Hex floats are IEEE-754 binary64 via Python `float.hex()` (`0x1.0p+0`
form), or `nan` / `inf` / `-inf`.

### signature

| Key | Type | Meaning |
| --- | --- | --- |
| `alg` | string | MUST be `Ed25519` |
| `mode` | string | `self-signed` or `assay-signed` |
| `public_key_hex` | string | 32-byte public key, 64 lowercase hex chars |
| `signature_hex` | string | 64-byte signature, 128 lowercase hex chars |

`signature.mode` MUST equal `body.independence.mode`.

## Canonical JSON (what is signed)

The signed message is the UTF-8 (ASCII-only) encoding of **canonical
JSON of `body`**. Produce it as:

1. JSON object / array / string / integer / boolean / null only.
   No IEEE floats in the signed object (times and residuals are hex
   strings).
2. Object keys sorted lexicographically by UTF-16 code units, which for
   this document is ASCII sort.
3. No insignificant whitespace: separators are `,` and `:`.
4. Strings escaped as RFC 8259 with `ensure_ascii=True` (non-ASCII as
   `\uXXXX`).
5. Integers in base 10, no leading zeros (except `0`).

Python 3.11+ equivalent:

```python
json.dumps(
    body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
).encode("ascii")
```

Ed25519 signs those bytes (RFC 8032, pure Ed25519, no prehash).

## run_id

1. Copy `body`.
2. Set `run_id` to the empty string `""`.
3. `run_id = SHA-256(canonical_json(copy)).hexdigest()` (64 lowercase hex).

A verifier MUST recompute this and reject a mismatch.

## Reference catalog hash

Let `rows` be the list of objects `{ "file", "key", "sha256" }` taken
from `data/reference/manifest.json` `artifacts`, sorted by
`(file, key)` ascending.

`reference_catalog_sha256 = SHA-256(canonical_json(rows)).hexdigest()`.

The per-array `sha256` values are **not** ZIP bytes. They are
`SHA-256(ascii(dtype.str + "|" + shape) + NUL + C-contiguous tobytes())`
as in `docs/WORKLOADS.md`. If `--reference-dir` is passed, recompute and
compare. Signature verification itself does not need the `.npz` files.

## Verdict aggregation (internal consistency)

Recompute overall `status` from `cases`:

1. If any non-skipped case has `status == FAIL` → `FAIL`.
2. Restrict to cases with `counts_toward_verdict == true`.
3. If any of those is `skipped` → `INCONCLUSIVE`.
4. If every remaining voting case is `PASS` and at least one exists → `PASS`.
5. Otherwise → `INCONCLUSIVE`.

`exit_code` MUST be `0`/`1`/`2` for PASS/FAIL/INCONCLUSIVE.
A mismatch is a failed verification even if the signature is valid
(the signer attested an internally contradictory document).

W04–W07 in this repository set `counts_toward_verdict` false (no GEMM
ABFT in noisefloor-v1).

## Independence modes

### self-signed

`independence.disclaimer` MUST be exactly this UTF-8 string (one line):

```
SELF-SIGNED. A provider grading its own hardware is not independent evidence. This signature proves the report bytes were not altered after signing. It does not prove an independent party observed the run.
```

A verifier MUST print that text on the face of the result. A provider
grading its own hardware is not independent evidence.

v1 `assay run --report` only produces this mode.

### assay-signed

Reserved for a report issued by an Assay **project-observed** run.
**Do not implement an issuance service in v1.** The JSON shape exists so
Stage 3 can fill it.

`independence.disclaimer` MUST be exactly:

```
ASSAY-SIGNED. This mode is reserved for reports issued by an Assay project-observed run. v1 does not include an issuance service. A verifier must pin the Assay project public key with --pubkey; the embedded key alone does not establish independence.
```

Anyone can put `mode: assay-signed` on a document they signed with their
own key. That is why `assay verify --pubkey <file>` exists. Without a
pinned key, a valid signature only means "this public key signed this
body." This repository does not ship an Assay project secret or a
production project public key.

## Key file (operator)

JSON object, mode `0600`:

```json
{
  "crv": "Ed25519",
  "private_key_hex": "<64 lowercase hex chars, 32-byte seed>",
  "public_key_hex": "<64 lowercase hex chars, 32-byte point>"
}
```

`assay keygen --out PATH` writes one. Public-only pin file: the same
object without `private_key_hex`, or a file containing only the 64 hex
chars.

## `assay verify`

```bash
assay verify REPORT.json
assay verify REPORT.json --pubkey operator.pub
assay verify REPORT.json --reference-dir data/reference
```

MUST work with no network. Suggested checks, all offline:

1. Parse JSON; `spec == attestation-v1`.
2. Canonicalize `body`; Ed25519-verify with `signature.public_key_hex`.
3. If `--pubkey` is set, it MUST equal `signature.public_key_hex`.
4. Recompute `run_id`; match `issued_at_utc` grammar; `network_requests == 0`.
5. Recompute overall status from `cases`; match `status` and `exit_code`.
6. Check the independence disclaimer string for the declared mode.
7. Optionally recompute `reference_catalog_sha256` from `--reference-dir`.

Exit codes: **0** valid (warnings may still print), **1** invalid
(tamper, bad signature, inconsistency), **3** operational (unreadable
file, bad key file).

Hand-edit any signed field in `body` and verification MUST fail at step 2.

## What this is not

- Not a hardware PASS just because the file is signed.
- Not independent evidence when self-signed.
- Not a phone-home license check. There is no issuance URL in v1.
