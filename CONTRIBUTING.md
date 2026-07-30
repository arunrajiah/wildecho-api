# Contributing to wildecho-api

Thanks for considering a contribution. This is a small, focused project: a
self-hostable HTTP wrapper around Google Research's open Perch 2.0 bioacoustics
model. Bug reports, accuracy findings, and format-support fixes are all welcome.

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ground rules

* **Model weights never enter git.** They are ~400 MB and Apache-2.0 licensed by
  Google. `models/` is gitignored. If a PR adds a `.onnx`, `.tflite`, or
  `.safetensors` file, it will be closed.
* **This wrapper is MIT. The model is not.** Do not add code that implies the
  Perch weights are MIT licensed. See [NOTICE](NOTICE).
* **Be honest about accuracy.** Perch 2.0 is bird-heavy and has no bat coverage.
  Do not remove or soften the limitations text in the README or the `/v1/about`
  response to make the project look better than it is.

## Development setup

You need Python 3.11+ and `ffmpeg` on your `PATH`.

```bash
git clone https://github.com/arunrajiah/wildecho-api.git
cd wildecho-api
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/download_model.py     # ~400 MB, one time
pytest
```

`ffmpeg` install: `brew install ffmpeg` (macOS) or
`sudo apt-get install ffmpeg` (Debian/Ubuntu).

## Running the service locally

```bash
uvicorn wildecho_api.main:app --reload --port 8000
```

Then open http://localhost:8000/docs for the interactive API reference.

## Before you open a PR

Run the same three checks CI runs. All must pass.

```bash
ruff check . && ruff format --check . && mypy src scripts && pytest
```

Tests do **not** require the model weights. Anything touching real inference is
marked `@pytest.mark.model` and is skipped automatically when `models/` is empty,
which is why CI stays fast and green without a 400 MB download. Run the full set
locally with:

```bash
pytest -m model
```

## Commit and PR conventions

We use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
Keep the subject line under 72 characters.

```
feat(inference): add per-window score output
fix(audio): reject clips shorter than 0.5s with 422 instead of 500
docs(readme): clarify bat coverage limitation
chore(deps): bump onnxruntime to 1.24
```

Common types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `ci`.

Please keep pull requests small and single-purpose, and update
[CHANGELOG.md](CHANGELOG.md) under `## [Unreleased]` for anything user-facing.

## Adding an audio format

Decoding goes through `ffmpeg` in
[`src/wildecho_api/audio.py`](src/wildecho_api/audio.py), so most formats work
already. To advertise a new one, add its MIME type and extension to
`ALLOWED_CONTENT_TYPES` there and add a round-trip test.

## Reporting a misidentification

Accuracy reports are genuinely useful, but please include enough to reproduce:

1. The audio clip, or a link to it, plus its license.
2. What the service returned (the full `/v1/identify` JSON).
3. What you believe the correct species is, and how you know.

Note that a wrong answer is not necessarily a bug in this wrapper. It is often
the model's limitation, in which case the useful outcome is a documented caveat
rather than a code change.

## Security

Please do not open public issues for vulnerabilities. See
[SECURITY.md](SECURITY.md).
