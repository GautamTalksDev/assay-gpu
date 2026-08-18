# Developer environment and GPU access

This checkpoint is about running code on a real GPU for $0. Notebook free
tiers cannot validate Docker. Record every signup credit in `docs/BUDGET.md`.
Signup credits are survey budget: a provider you have an account with is a
provider you can survey.

## CPU (this repo, no GPU)

```bash
uv sync
uv run ruff check .
uv run mypy --strict src/
uv run pytest -m cpu
```

CI on GitHub Actions runs exactly those steps on `ubuntu-latest` with Python
3.11 and 3.12. GPU-marked tests are not executed in CI.

## Tests

| Marker | Where it runs | What it covers |
| --- | --- | --- |
| `cpu` | GitHub Actions, any laptop | Imports, CLI skeleton, anything that must not need CUDA |
| `gpu` | Manual: Kaggle, Colab, Lightning, HPC, or `scripts/gpu_smoke.sh` | `nvidia-smi` sees a device; later, real workloads |

Every test must carry exactly one of those markers (`tests/conftest.py`
rejects unmarked tests).

```bash
uv run pytest -m cpu
uv run pytest -m gpu    # requires a GPU and nvidia-smi
```

## GPU access — spend $0

Do this in order. Do not burn Thunder Compute, Hatch, or paid time until
noise-floor characterization (CP-3).

### 1. University HPC cluster (check first)

What you get: possibly A100/H100, free, if you have an allocation.

Use it for: the only realistic path to datacenter-class GPUs before the
survey. File the access form before using any cloud credit.

This implementation session had no cluster credentials. If you have a
university allocation, load CUDA, clone this repo, `uv sync`, then
`uv run pytest -m gpu`.

### 2. Kaggle (primary workhorse)

What you get: ~30 GPU-hours/week, P100 or 2×T4, no credit card.

Use it for: day-to-day `pytest -m gpu`. The dual-T4 config is the free
cross-GPU agreement check.

Kaggle has no root and no Docker. Do not try `scripts/gpu_smoke.sh` here.

Notebook steps (GPU session enabled):

```python
# Settings → Accelerator → GPU
!pip install -q uv
!git clone https://github.com/<owner>/<repo>.git
%cd <repo>
!uv sync
!uv run pytest -m gpu
```

On a 2×T4 kernel, `nvidia-smi -L` must list two GPUs. That is the
cross-device check until workload code exists.

### 3. Colab free (secondary)

What you get: T4s, occasionally other SKUs depending on availability.

Use it for: hardware variability. Different silicon is a feature for later
noise-floor work, not a nuisance.

Runtime → Change runtime type → GPU. Same clone/`uv sync`/`pytest -m gpu`
sequence as Kaggle. No Docker.

### 4. Lightning AI free

What you get: 15 credits/month ≈ 22 T4-hours, persistent storage.

Use it for: long runs that would time out in a notebook (CP-3).

Studio with a T4: clone, `uv sync`, `uv run pytest -m gpu`. No Docker on
the free notebook-style path; a persistent Studio VM that offers Docker is
still not a substitute for the bare-metal `--gpus all` check below.

### 5. Hold — do not burn on CP-1

| Source | What you get | Hold for |
| --- | --- | --- |
| Thunder Compute student credit | ~$20 on A100s | CP-3 A100 characterization |
| DigitalOcean Hatch | H100 test time | CP-3 |

## Docker path (cannot be done on Kaggle or Colab)

`docker run --gpus all <image> run` must be proven on a **bare-metal**
provider using a **signup credit**, not paid time.

Typical new-account credits: RunPod or Vast, $5–10. At ~$0.20/hr spot that
is 25–50 hours. Use the cheapest GPU that the NVIDIA Container Toolkit
supports (RTX 3090 or A4000 is enough). This test proves Docker GPU
passthrough, not numerical correctness.

On the instance:

```bash
# NVIDIA driver + nvidia-container-toolkit already present on GPU templates
export ASSAY_IMAGE=ghcr.io/<owner>/<repo>:v0.0.0
curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/scripts/gpu_smoke.sh | bash
```

Or from a clone:

```bash
ASSAY_IMAGE=ghcr.io/<owner>/<repo>:v0.0.0 ./scripts/gpu_smoke.sh
```

Local image (daemon running, NVIDIA Container Toolkit installed):

```bash
docker build -t assay-gpu:local .
ASSAY_SKIP_PULL=1 ASSAY_IMAGE=assay-gpu:local ./scripts/gpu_smoke.sh
```

Image contract:

- Base: `nvidia/cuda:12.4.1-runtime-ubuntu22.04`
- Entrypoint dispatches to `assay`, so `docker run --gpus all <image> run`
  runs the `run` subcommand
- Must stay under 3 GB (`docker image ls` SIZE column)

## Release (signed container + PyPI wheel)

A tag matching `v*` runs `.github/workflows/release.yml`:

1. `uv build` → sdist + wheel
2. `SHA256SUMS` over those files
3. Cosign keyless (Sigstore) signatures on the artifacts
4. PyPI upload via trusted publishing (OIDC). No API token in the repo.
5. Build/push `ghcr.io/<owner>/<repo>` and cosign the digest

Configure once before the first tag:

- GitHub → Settings → Actions → general: allow GitHub Actions to create
  and approve pull requests is unrelated; **Packages** write is granted by
  the workflow `permissions`.
- PyPI → Publishing → pending publisher: GitHub, this repository, workflow
  filename `release.yml`, environment empty, tag is the version source.
- Package settings on GHCR: after the first push, set visibility as needed.

Create a GitHub Release by pushing `v0.0.0` (or later) only after the
publisher is configured. Do not retag.

## What this workstation actually had (2026-08-18)

| Check | Result |
| --- | --- |
| `uv run pytest -m cpu` | Must pass locally before push |
| `nvidia-smi` | Not present (WSL2, no NVIDIA driver in the guest) |
| Docker daemon | CLI installed; daemon not running (`dockerDesktopLinuxEngine` missing) |
| GitHub | `gh` not authenticated; no remote; Actions not yet green |
| Cloud GPU | No provider account was opened; $0 signup credit consumed |

Until `scripts/gpu_smoke.sh` has been run on a real GPU instance, the
distribution path is unproven. Do not start CP-2 on that basis.

## Spending rules

- Free-tier notebook hours (Kaggle, Colab, Lightning, university HPC) are
  not counted against the $75 ceiling.
- RunPod/Vast (or similar) signup credits **are** survey budget. Log them
  in `docs/BUDGET.md` the day they are claimed, including remaining credit.
- HARD CEILING remains $75 cash-or-credit across all providers.
