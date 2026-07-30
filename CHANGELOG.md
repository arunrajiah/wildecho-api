# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
