#!/usr/bin/env bash
# Deploy wildecho-api to a Hugging Face Space (free CPU tier: 2 vCPU, 16 GB RAM).
# Prereq: `hf auth login` with a write token.
# Usage: scripts/deploy_hf_space.sh [owner/space-name]   (default: <you>/wildecho-api)
set -euo pipefail
cd "$(dirname "$0")/.."

SPACE="${1:-$(hf auth whoami | head -1 | awk '{print $NF}')/wildecho-api}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/data"
cp -R src scripts pyproject.toml LICENSE NOTICE "$STAGE/"
cp data/taxonomy.csv data/calibration.yaml "$STAGE/data/"
cp deploy/huggingface/Dockerfile "$STAGE/Dockerfile"
cp deploy/huggingface/SPACE_README.md "$STAGE/README.md"
find "$STAGE" -name __pycache__ -type d -prune -exec rm -rf {} +

hf repos create "$SPACE" --repo-type space --space-sdk docker --exist-ok
hf upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy wildecho-api"

HOST="$(echo "$SPACE" | tr '/' '-' | tr '[:upper:]_.' '[:lower:]--')"
echo "Deployed $SPACE. Once the build finishes: https://$HOST.hf.space/v1/health"
