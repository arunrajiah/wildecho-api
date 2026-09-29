# Hugging Face Spaces image (Docker SDK). Unlike the main Dockerfile, this bakes the
# model weights in at build time because free Spaces have no persistent volume, and
# it runs directly as uid 1000 because Spaces never grants root at runtime (so the
# gosu-based entrypoint can't be used). Staged and pushed by scripts/deploy_hf_space.sh.
FROM python:3.12-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    WILDECHO_MODEL_PATH=/app/models/perch_v2.onnx \
    WILDECHO_TAXONOMY_PATH=/app/data/taxonomy.csv \
    WILDECHO_CALIBRATION_PATH=/app/data/calibration.yaml \
    WILDECHO_FEEDBACK_DB_PATH=/tmp/wildecho/feedback.sqlite3 \
    WILDECHO_FEEDBACK_CLIPS_DIR=/tmp/wildecho/feedback_clips \
    WILDECHO_FEEDBACK_STORE_AUDIO=false \
    FORWARDED_ALLOW_IPS="*"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 user \
    && python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel

WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
COPY data ./data
COPY scripts ./scripts

RUN /opt/venv/bin/pip install . \
    && python scripts/download_model.py --output-dir /app/models \
    && chown -R user:user /app

USER user
EXPOSE 7860

# FORWARDED_ALLOW_IPS=* lets uvicorn trust X-Forwarded-For from the Spaces proxy, so
# per-IP rate limits key on real clients rather than one shared proxy address.
CMD ["uvicorn", "wildecho_api.main:app", "--host", "0.0.0.0", "--port", "7860"]
