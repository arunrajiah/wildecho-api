#!/bin/sh
# Runs before uvicorn on every container start.
#
# The container image starts as root (see the Dockerfile - USER is not set at
# build time). That is deliberate and confined to this script: a freshly created
# volume on a hosted platform (a Fly volume, a Render disk) is typically
# root-owned, and the unprivileged app user cannot write to it or create
# subdirectories under it. Root fixes that ownership once here, then this script
# re-executes itself as the unprivileged `wildecho` user via gosu before doing
# anything else - the actual server process never runs as root. This is the same
# pattern used by images like postgres and grafana for exactly this reason.
#
# Local docker-compose users are unaffected either way: their bind-mounted
# ./models is read-only, so the chown below silently no-ops on it (redirected to
# /dev/null), and the rest of this script behaves identically.
set -e

MODEL_PATH="${WILDECHO_MODEL_PATH:-/app/models/perch_v2.onnx}"
FEEDBACK_DB_PATH="${WILDECHO_FEEDBACK_DB_PATH:-/app/data/feedback.sqlite3}"
FEEDBACK_CLIPS_DIR="${WILDECHO_FEEDBACK_CLIPS_DIR:-/app/data/feedback_clips}"

if [ "$(id -u)" = "0" ]; then
    mkdir -p "$(dirname "$MODEL_PATH")" "$(dirname "$FEEDBACK_DB_PATH")" "$FEEDBACK_CLIPS_DIR"
    chown -R wildecho:wildecho \
        "$(dirname "$MODEL_PATH")" "$(dirname "$FEEDBACK_DB_PATH")" "$FEEDBACK_CLIPS_DIR" \
        2>/dev/null || true
    exec gosu wildecho "$0" "$@"
fi

# From here on we are always the unprivileged `wildecho` user, whether the
# container started as root and dropped down above, or started as wildecho
# directly (e.g. `docker run --user`).

if [ "${WILDECHO_AUTO_DOWNLOAD_MODEL:-false}" = "true" ] && [ ! -f "$MODEL_PATH" ]; then
    echo "wildecho-api: WILDECHO_AUTO_DOWNLOAD_MODEL=true and no weights at $MODEL_PATH; downloading (~390 MiB, Apache-2.0, Google LLC)..." >&2
    python scripts/download_model.py --output-dir "$(dirname "$MODEL_PATH")"
    if [ ! -f "$MODEL_PATH" ]; then
        echo "wildecho-api: download finished but $MODEL_PATH is still missing. WILDECHO_MODEL_PATH must end in the exported filename (perch_v2.onnx by default)." >&2
        exit 1
    fi
fi

exec "$@"
