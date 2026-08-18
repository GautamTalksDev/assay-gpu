# NVIDIA CUDA runtime. Must be invoked as:
#   docker run --gpus all <image> run
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=3.12 \
    UV_PYTHON_DOWNLOADS=automatic \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
COPY src src
COPY tests tests
COPY data data

RUN uv sync --frozen --no-dev --group test --no-editable \
    && rm -rf /root/.cache/uv

# Keep the NVIDIA entrypoint so CUDA env is set, then dispatch to assay.
# `docker run --gpus all <image> run` therefore becomes `assay run`.
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "assay"]
