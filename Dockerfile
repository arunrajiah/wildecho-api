# syntax=docker/dockerfile:1
#
# Multi-stage build. The builder compiles wheels; the final image carries only the
# installed virtualenv, ffmpeg, and the application. Model weights are never baked
# in: they are ~390 MiB, Apache-2.0 licensed by Google, and mounted as a volume at
# runtime (see docker-compose.yml).

# ---------------------------------------------------------------------------
# Stage 1: build the virtualenv
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Dependencies change far less often than source, so install them from the
# metadata alone first and let Docker cache that layer.
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
COPY data ./data

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install .

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="wildecho-api" \
      org.opencontainers.image.description="Self-hostable species ID from audio, powered by Google's open Perch 2.0 model" \
      org.opencontainers.image.source="https://github.com/arunrajiah/wildecho-api" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.authors="Arun Rajiah"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    WILDECHO_MODEL_PATH=/app/models/perch_v2.onnx \
    WILDECHO_TAXONOMY_PATH=/app/data/taxonomy.csv

# ffmpeg does all container parsing and resampling. curl is only here so the
# container healthcheck has something to call.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY data ./data
COPY scripts ./scripts

# Run unprivileged. models/ is created here so a read-only bind mount can land on
# an existing directory owned by the app user.
RUN useradd --create-home --uid 10001 wildecho \
    && mkdir -p /app/models \
    && chown -R wildecho:wildecho /app
USER wildecho

EXPOSE 8000

# Reports "degraded" rather than failing when weights are absent, so a fresh
# install is diagnosable instead of crash-looping.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/v1/health || exit 1

CMD ["uvicorn", "wildecho_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
