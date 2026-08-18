#!/usr/bin/env bash
# Pull the published assay container on a GPU instance and run GPU-marked tests.
#
# Required: docker with NVIDIA Container Toolkit (`docker run --gpus all`).
# This script cannot run on Kaggle or Colab (no root, no Docker).
#
# Usage on a freshly rented instance (repo checkout is not required for pull+run):
#   export ASSAY_IMAGE=ghcr.io/<owner>/<repo>:v0.0.0
#   curl -fsSL <raw-url-to-this-script> | bash
# Or from a clone:
#   ASSAY_IMAGE=ghcr.io/<owner>/<repo>:v0.0.0 ./scripts/gpu_smoke.sh
#
# Local image without pulling:
#   ASSAY_SKIP_PULL=1 ASSAY_IMAGE=assay-gpu:local ./scripts/gpu_smoke.sh

set -euo pipefail

REPO_LOWER="$(echo "${GITHUB_REPOSITORY:-assay-gpu/assay-gpu}" | tr '[:upper:]' '[:lower:]')"
IMAGE="${ASSAY_IMAGE:-ghcr.io/${REPO_LOWER}:latest}"

if ! command -v docker >/dev/null 2>&1; then
  echo "gpu_smoke: docker is required" >&2
  exit 1
fi

if [[ "${ASSAY_SKIP_PULL:-0}" != "1" ]]; then
  docker pull "${IMAGE}"
fi

# Contract: docker run --gpus all <image> run
docker run --gpus all --rm "${IMAGE}" run

# GPU-marked tests shipped in the image (pytest is the test extra).
docker run --gpus all --rm \
  --entrypoint /opt/nvidia/nvidia_entrypoint.sh \
  "${IMAGE}" \
  pytest -m gpu
