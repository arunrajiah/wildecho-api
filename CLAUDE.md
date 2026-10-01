# wildecho-api

Self-hostable FastAPI service that identifies species from audio clips using Google's
Perch 2.0 (ONNX). Backend for the WildEcho Android app (`../wildecho`).

## Stack
- Python 3.12, FastAPI, ONNX Runtime (CPU), ffmpeg for decoding, slowapi rate limits
- Docker; model weights (~390 MiB) fetched by `scripts/download_model.py`, never committed

## Layout
- `src/wildecho_api/` main.py (routes), inference.py, audio.py, config.py (all `WILDECHO_*` env vars), feedback.py
- `data/` taxonomy.csv, calibration.yaml; `tests/` with `tests/fixtures/` audio
- `deploy/huggingface/Dockerfile` model-baked image, runs as uid 1000, port 7860 (used for the droplet too)
- `deploy/droplet/` run.sh, nginx-wildecho.conf, watchdog.sh for the public instance
- `fly.toml`, `render.yaml`, `scripts/deploy_hf_space.sh` other targets (HF Docker Spaces need PRO)

## Public instance
- https://wildecho.arunrajiah.com on the user's shared droplet `lvl1-server-v3` (root@157.245.114.249)
- Container `wildecho-api` behind existing nginx; own certbot cert; files in /opt/wildecho-api
- Shared host with other production apps: never touch other nginx sites or containers; `nginx -t` before reload
- Redeploy: rsync a staged context (see deploy_hf_space.sh staging) to /opt/wildecho-api, `docker build -t wildecho-api:latest .`, `./run.sh`
- Monitoring: `.github/workflows/uptime.yml` (15 min health, daily identify) + cron watchdog every 2 min

## Commands
- `.venv/bin/python -m pytest -q` (190+ tests); `.venv/bin/ruff check src tests`; `.venv/bin/ruff format src tests`; `.venv/bin/mypy src`
- `docker compose up --build` local server on :8000 (needs models/ downloaded)

## Conventions
- Conventional Commits; never add Claude/AI attribution
- No em dashes in user-facing copy
- New behavior goes behind a `WILDECHO_*` setting with the old behavior as default

## Token efficiency
- Grep/Glob to the target file; read only the relevant section
- Don't re-read files after editing; verify once per batch of edits
- Keep progress narration and summaries to 2-3 sentences
