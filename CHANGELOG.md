# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

* Per-taxonomic-group confidence calibration (`data/calibration.yaml`): discounts
  displayed confidence and raises the low-confidence bar for groups Perch is
  documented as weaker on (frogs, insects, mammals) relative to birds, without a
  code change. Raw model ranking and `raw_confidence` are unaffected.
* `POST /v1/feedback`: opt-in, locally stored user corrections (SQLite by
  default), for improving accuracy over time. Nothing is collected unless a
  client calls it, and nothing is sent anywhere else. See README "Feedback" for
  the privacy note and how to swap in Postgres.
* Structured logging with a per-request ID: every request gets an ID (reused
  from an incoming `X-Request-ID` header, or generated), attached to every log
  line during that request, returned as a response header and in
  `POST /v1/identify`'s body as `request_id` - hand it to `/v1/feedback` to
  correlate a correction with the original request's logs.
* `WILDECHO_LOG_FORMAT`: `text` (default) or `json`, for hosted log aggregators.
* `fly.toml` and `render.yaml` deployment configs, with an entrypoint that
  auto-downloads model weights on first boot (`WILDECHO_AUTO_DOWNLOAD_MODEL`,
  opt-in) and fixes ownership of a freshly mounted hosted volume before dropping
  to the unprivileged runtime user.
* README "Capacity planning": measured per-window inference cost and the math
  for how many concurrent `/v1/identify` requests a small instance can sustain.

### Changed

* `decode_file` and `model.identify` now run in FastAPI's threadpool instead of
  directly on the event loop, so concurrent requests no longer serialize behind
  one another.
* `Prediction.confidence` is now the per-group-calibrated value;
  `Prediction.raw_confidence` carries the original, uncalibrated softmax score.
  `predictions[]` order is decided by `raw_confidence` and is never reshuffled
  by calibration.

## [0.1.0] - 2026-07-30

Initial release. Self-hostable species identification from audio, wrapping
Google Research's Perch 2.0 model.

### Added

* `POST /v1/identify` endpoint accepting a multipart audio upload (wav, mp3, m4a,
  webm, ogg, flac) and returning ranked species predictions with clip metadata.
* `GET /v1/health` reporting model load status, and `GET /v1/about` reporting
  model provenance, taxa coverage and the accuracy caveats.
* ONNX inference pipeline (`wildecho_api.inference`): resampling to 32 kHz mono,
  5-second windowing with a 2.5-second stride, logit averaging across windows,
  softmax, and top-N ranking.
* Filtering of the 198 FSD50K general sound event classes (wind, rain, speech,
  engines) out of results, with a log line recording when a non-animal class was
  the top prediction.
* `low_confidence` flag on responses, tripped when top confidence is below
  `WILDECHO_LOW_CONFIDENCE_THRESHOLD` (default 0.3).
* `data/taxonomy.csv`: index to species mapping for all 14,795 Perch classes,
  with scientific name, English common name and coarse taxonomic group.
  Rebuildable with `scripts/build_taxonomy.py`.
* `scripts/download_model.py` to fetch the ~400 MB ONNX export into the
  gitignored `models/` directory, with checksum verification.
* Per-IP rate limiting via slowapi, configurable with `WILDECHO_RATE_LIMIT`.
* Configurable CORS, permissive by default for mobile clients.
* Multi-stage `Dockerfile` and `docker-compose.yml` mounting model weights as a
  volume.
* Test suite covering the inference pipeline against a bundled CC0 European
  Nightjar recording, plus API tests against a mocked model.

### Notes

* This wrapper is MIT licensed. Perch 2.0 remains Apache-2.0, copyright Google
  LLC, and its weights are not redistributed here. See [NOTICE](NOTICE).
* Authentication, usage analytics and hosted deployment are out of scope.

[Unreleased]: https://github.com/arunrajiah/wildecho-api/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/arunrajiah/wildecho-api/releases/tag/v0.1.0
