#!/usr/bin/env bash
# (Re)start wildecho-api on a shared Docker host behind nginx. Build first with
# `docker build -t wildecho-api:latest .` from a context staged like
# scripts/deploy_hf_space.sh does (deploy/huggingface/Dockerfile bakes in the model).
# Tuned for a host shared with other apps: capped memory, one inference at a time
# (others queue), 30s clips, and no ONNX arena so peak memory is released.
set -euo pipefail
docker rm -f wildecho-api >/dev/null 2>&1 || true
docker run -d --name wildecho-api --restart unless-stopped \
  -p 127.0.0.1:7860:7860 -m 1400m --memory-swap 2g \
  --log-opt max-size=10m --log-opt max-file=3 \
  -e WILDECHO_MAX_DURATION_SECONDS=30 \
  -e WILDECHO_MAX_CONCURRENT_INFERENCES=1 \
  -e WILDECHO_ONNX_MEMORY_ARENA=false \
  -e WILDECHO_LOG_FORMAT=json \
  wildecho-api:latest
