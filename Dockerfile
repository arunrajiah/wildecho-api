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
    WILDECHO_TAXONOMY_PATH=/app/data/taxonomy.csv \
    WILDECHO_CALIBRATION_PATH=/app/data/calibration.yaml \
    WILDECHO_FEEDBACK_DB_PATH=/app/data/feedback.sqlite3 \
    WILDECHO_FEEDBACK_CLIPS_DIR=/app/data/feedback_clips

# ffmpeg does all container parsing and resampling. curl is only here so the
# container healthcheck has something to call. gosu lets entrypoint.sh drop root
# privileges after fixing ownership of a freshly mounted volume - see its
# comments for why that's needed on hosted platforms.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl gosu \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY data ./data
COPY scripts ./scripts
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# models/ is created here so a read-only bind mount (docker-compose) can land on
# an existing directory owned by the app user. The image is left running as root
# by design - entrypoint.sh always drops to this unprivileged user via gosu
# before the application itself ever runs, after first fixing permissions on any
# freshly mounted volume that root alone could otherwise write to.
RUN useradd --create-home --uid 10001 wildecho \
    && mkdir -p /app/models \
    && chown -R wildecho:wildecho /app

EXPOSE 8000

# Reports "degraded" rather than failing when weights are absent, so a fresh
# install is diagnosable instead of crash-looping.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/v1/health || exit 1

# entrypoint.sh drops root privileges, optionally fetches the model weights (see
# its comments), then hands off to whatever CMD says.
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "wildecho_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
