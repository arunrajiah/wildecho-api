# wildecho-api

**Self-hostable species ID from audio, powered by Google's open Perch 2.0 model.**

[![CI](https://github.com/arunrajiah/wildecho-api/actions/workflows/ci.yml/badge.svg)](https://github.com/arunrajiah/wildecho-api/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Model: Apache 2.0](https://img.shields.io/badge/Model-Apache%202.0-green.svg)](https://github.com/google-research/perch)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

Upload a short recording of an animal, get back ranked species candidates. It runs
entirely on your own hardware, on CPU, with no API keys and no data leaving the box.

> Part of an open wildlife toolkit. Not sure this is the project you need? See [which project to use](#part-of-an-open-wildlife-toolkit).

This is a thin, honest HTTP wrapper. All of the intelligence belongs to
[Perch 2.0](https://github.com/google-research/perch), released openly by Google
Research under Apache 2.0. This repository adds inference plumbing, audio handling,
a species metadata table, and a documented API.

```bash
curl -X POST http://localhost:8000/v1/identify \
  -F "file=@tests/fixtures/european_nightjar_xc1008591.mp3"
```

```json
{
  "predictions": [
    {
      "common_name": "European Nightjar",
      "scientific_name": "Caprimulgus europaeus",
      "taxonomic_group": "bird",
      "confidence": 0.909,
      "raw_confidence": 0.909,
      "low_confidence": false,
      "class_index": 2211
    }
  ],
  "low_confidence": false,
  "non_animal_top_class": null,
  "model_version": "perch_v2",
  "request_id": "9cbd9cca508b4fb1937b5da48d3850dc",
  "metadata": {
    "duration_seconds": 10.0,
    "windows_processed": 3,
    "window_seconds": 5.0,
    "window_stride_seconds": 2.5,
    "source_sample_rate": 32000,
    "source_channels": 1,
    "processed_sample_rate": 32000,
    "inference_ms": 214.6
  }
}
```

## Contents

- [Read this first: accuracy and limitations](#read-this-first-accuracy-and-limitations)
- [Quickstart](#quickstart)
- [API reference](#api-reference)
- [Configuration](#configuration)
- [Confidence calibration](#confidence-calibration)
- [Feedback](#feedback)
- [Logging and request IDs](#logging-and-request-ids)
- [Self-hosting notes](#self-hosting-notes)
- [Deployment](#deployment)
- [Capacity planning](#capacity-planning)
- [How it works](#how-it-works)
- [Development](#development)
- [Part of an open wildlife toolkit](#part-of-an-open-wildlife-toolkit)
- [Credits](#credits)
- [License](#license)

## Read this first: accuracy and limitations

This section is at the top on purpose. If you are building something people rely
on, these limits matter more than the API surface.

### It is a bird model that also knows some other things

Perch 2.0 covers about 14,600 taxa, but the training data is heavily bird-weighted.
Of the species classes in the label set:

| Group | Classes | Share |
| --- | --- | --- |
| Birds | 10,256 | 70.3% |
| Insects | 2,133 | 14.6% |
| Frogs and toads | 1,344 | 9.2% |
| Mammals | 677 | 4.6% |
| Other (reptiles, fish, salamanders, and so on) | 187 | 1.3% |

Class count is not the same as data volume, and the imbalance in recordings per
class is steeper still. Perch 2.0's lineage is a decade of bird vocalization work;
non-bird taxa were added in this release and are much more thinly and unevenly
sampled. In practice: a clear recording of a common bird is often right, and an
insect or mammal result is a lead to verify, not an answer.

### There is no bat coverage. None.

Bat echolocation is mostly ultrasonic, from roughly 20 kHz to over 100 kHz. This
model consumes 32 kHz audio, so by the Nyquist limit it cannot represent anything
above 16 kHz. Bat calls are not merely underrepresented in the training data, they
are physically outside what the model can see. If you record a bat, you will get a
confident-looking list of birds and insects that is entirely wrong. Bat detection
needs dedicated ultrasonic hardware and a different model.

### Other things worth knowing

- **Confidence is a ranking score, not a probability of presence.** It is a softmax
  over all 14,795 classes. Perch is natively a multi-label classifier, so if two
  species genuinely overlap in a clip, softmax splits the score between them and
  both look less certain than they are. Compare candidates against each other;
  do not read 0.62 as "62% likely to be correct".
- **Geography is not used.** The model has no idea where you are. It will happily
  suggest a bird from another continent. Filtering candidates by your region using
  the `ebird_code` column in [`data/taxonomy.csv`](data/taxonomy.csv) is the single
  highest-value improvement most callers can make.
- **Non-animal sounds are recognised but hidden.** The label set includes 198
  general sound event classes from FSD50K (wind, rain, speech, engines, music).
  These are filtered out of results, but when one of them was the top-scoring class
  overall, the response sets `non_animal_top_class`. When you see that, the species
  list below it is noise. Show the user "no animal detected" instead.
- **Common names cover about 79% of species.** They come from Wikidata. When a taxon
  has no English common name, `common_name` is `null` and you should display
  `scientific_name`. The service does not invent a name.
- **Short clips are weaker.** Anything under 5 seconds is zero-padded to a single
  window. Aim for 5 to 15 seconds of the animal actually vocalising.

**Do not use this service alone for conservation, regulatory, legal, or safety
decisions.** Verify with a recordist, a regional expert, or a curated reference.

## Quickstart

### Docker (recommended)

Requires Docker and about 1 GB of free disk.

```bash
git clone https://github.com/arunrajiah/wildecho-api.git
cd wildecho-api
```

Fetch the model weights. They are ~390 MiB, Apache-2.0 licensed by Google, and
never committed to this repository:

```bash
python3 scripts/download_model.py
```

Start the service:

```bash
docker compose up --build
```

Confirm it loaded, then identify the bundled test clip:

```bash
curl -s http://localhost:8000/v1/health
```

```bash
curl -s -X POST http://localhost:8000/v1/identify -F "file=@tests/fixtures/european_nightjar_xc1008591.mp3"
```

Interactive API docs are at http://localhost:8000/docs.

`docker compose up` works before you download the weights, but `/v1/identify` will
return `503 model_unavailable` and `/v1/health` will report `degraded`. That is
deliberate: a missing model is a state to report, not a reason to crash-loop.

### Local Python

Requires Python 3.11 or newer and `ffmpeg` on your `PATH`
(`brew install ffmpeg`, or `sudo apt-get install ffmpeg`).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python scripts/download_model.py
uvicorn wildecho_api.main:app --port 8000
```

### As a library

```python
import numpy as np
from wildecho_api.inference import identify, load_model

load_model()
audio = np.load("clip.npy")  # mono or multi-channel, any sample rate
for prediction in identify(audio, sample_rate=44_100):
    print(prediction.common_name, prediction.scientific_name, round(prediction.confidence, 3))
```

## API reference

### `POST /v1/identify`

Identify species in an audio clip.

**Request:** `multipart/form-data` with a single `file` field.

Accepted formats: `wav`, `mp3`, `m4a`, `webm`, `ogg`, `opus`, `flac`, and anything
else your `ffmpeg` build can decode. `application/octet-stream` is accepted, since
many mobile HTTP clients send it.

**Response `200`:** see the example at the top of this README.

| Field | Meaning |
| --- | --- |
| `predictions[]` | Top candidates, ordered by the model's own `raw_confidence` (best first). Non-animal classes excluded. |
| `predictions[].common_name` | English common name, or `null`. Fall back to `scientific_name`. |
| `predictions[].scientific_name` | Latin binomial from the Perch label set. |
| `predictions[].taxonomic_group` | One of `bird`, `frog`, `insect`, `mammal`, `other`. |
| `predictions[].confidence` | Softmax score after this taxon's group calibration is applied (see [Confidence calibration](#confidence-calibration)). The number to show a user. |
| `predictions[].raw_confidence` | The uncalibrated softmax score. Equal to `confidence` when no calibration is configured. Order in the list follows this value, not the calibrated one. |
| `predictions[].low_confidence` | This candidate's calibrated confidence is below its group's effective threshold. |
| `predictions[].class_index` | Raw Perch output index, for joining against `data/taxonomy.csv`. |
| `low_confidence` | The top prediction is below its group's threshold. Treat the whole result as weak. |
| `non_animal_top_class` | Set when the highest-scoring class of all was a sound event. Distrust the list. |
| `request_id` | Correlates this response with server logs. Hand it back in `POST /v1/feedback` when correcting this result. Also returned as the `X-Request-ID` response header. |
| `metadata.windows_processed` | Number of 5-second windows analysed. |
| `metadata.duration_seconds` | Duration of the decoded clip. |

**Errors.** Every error uses the same shape, with a stable `error` code:

```json
{ "error": "audio_too_short", "detail": "Clip is 0.21s; the minimum is 0.5s. Record a longer sample." }
```

| Status | `error` | Cause |
| --- | --- | --- |
| 413 | `audio_too_long` | Clip exceeds `WILDECHO_MAX_DURATION_SECONDS`. |
| 415 | `unsupported_format` | Not a decodable audio file. |
| 422 | `audio_too_short` | Shorter than `WILDECHO_MIN_DURATION_SECONDS`. |
| 422 | `audio_silent` | RMS below the silence threshold. Usually a microphone permission problem. |
| 422 | `audio_empty` | Zero-length upload or no decodable samples. |
| 422 | `audio_decode_failed` | Truncated or corrupt file. |
| 429 | `rate_limited` | Per-IP rate limit exceeded. |
| 503 | `model_unavailable` | Weights not downloaded. Run `scripts/download_model.py`. |
| 503 | `ffmpeg_unavailable` | `ffmpeg` is not installed on the server. |

The `detail` strings are written to be safe to show to an end user.

### `POST /v1/feedback`

Submit a correction for a previous identification. See [Feedback](#feedback) for
the full picture (what this collects, why, and how it's stored); this is the wire
format.

**Request:** `multipart/form-data`. Only `corrected_text` is required.

| Field | Required | Meaning |
| --- | --- | --- |
| `corrected_text` | yes | The species you believe this clip actually is: a scientific or common name (matched case-insensitively against the taxonomy) or free text. Always stored verbatim. |
| `clip_id` | no | An identifier you choose, for correlating with your own records. Not validated. |
| `request_id` | no | The `request_id` from the original `/v1/identify` response, for correlation. |
| `original_scientific_name` | no | What the model originally predicted as top-1, if you have it. |
| `original_confidence` | no | The model's original top-1 confidence, if you have it. |
| `notes` | no | Free-text context, e.g. how you know the correct identification. |
| `file` | no | The audio clip this correction is about. Saved to disk unless `WILDECHO_FEEDBACK_STORE_AUDIO=false`. |

```bash
curl -X POST http://localhost:8000/v1/feedback \
  -F "corrected_text=Caprimulgus europaeus" \
  -F "request_id=9cbd9cca508b4fb1937b5da48d3850dc" \
  -F "notes=Heard the distinctive churring song at dusk"
```

```json
{
  "id": 1,
  "received_at": "2026-07-30T06:31:32.649595+00:00",
  "matched_scientific_name": "Caprimulgus europaeus",
  "matched_common_name": "European Nightjar",
  "matched_taxonomic_group": "bird",
  "stored_audio": false
}
```

`matched_*` fields are `null` when `corrected_text` didn't exactly match a known
scientific or common name (case-insensitively) - the raw text is still stored
either way, nothing is lost, it just wasn't auto-resolved to a taxonomy entry.

A `404 feedback_disabled` response means the operator turned this off
(`WILDECHO_FEEDBACK_ENABLED=false`); a `503 feedback_unavailable` response means it's
on but the store failed to initialize (check server logs). Both cases return the
usual `{error, detail}` shape.

### `GET /v1/health`

Returns `200` whether or not the model loaded. Check `model_loaded`. A load failure
is reported, not thrown, so a container does not restart forever over a missing file.

```json
{
  "status": "ok",
  "model_status": "loaded",
  "model_path": "/app/models/perch_v2.onnx",
  "model_loaded": true,
  "detail": null,
  "num_classes": 14795,
  "taxonomy_loaded": true,
  "feedback_enabled": true,
  "feedback_store_ready": true,
  "version": "0.1.0"
}
```

### `GET /v1/about`

Model provenance, taxa coverage counts, configured limits, and the full accuracy
disclaimer as a string. If you build a client, surface
`coverage.disclaimer` somewhere a user can read it.

## Configuration

Every setting is an environment variable prefixed `WILDECHO_`. See
[`.env.example`](.env.example) for the annotated full list.

| Variable | Default | Notes |
| --- | --- | --- |
| `WILDECHO_MODEL_PATH` | `models/perch_v2.onnx` | Where the ONNX weights are. |
| `WILDECHO_TAXONOMY_PATH` | `data/taxonomy.csv` | Committed; you should not need to change this. |
| `WILDECHO_CALIBRATION_PATH` | `data/calibration.yaml` | Per-group confidence adjustment. See [Confidence calibration](#confidence-calibration). Missing file = no-op. |
| `WILDECHO_TOP_K` | `10` | Candidates returned. |
| `WILDECHO_LOW_CONFIDENCE_THRESHOLD` | `0.3` | See "Why the threshold is 0.3" below. |
| `WILDECHO_RATE_LIMIT` | `20/hour` | Per client IP, applies to `/v1/identify` and `/v1/feedback`. **Tune this.** |
| `WILDECHO_RATE_LIMIT_ENABLED` | `true` | Set `false` for private use behind a firewall. |
| `WILDECHO_CORS_ORIGINS` | `*` | Comma-separated. **Narrow this in production.** |
| `WILDECHO_MAX_UPLOAD_BYTES` | `26214400` (25 MB) | |
| `WILDECHO_MAX_DURATION_SECONDS` | `300` | Inference cost scales with duration. |
| `WILDECHO_MIN_DURATION_SECONDS` | `0.5` | |
| `WILDECHO_ONNX_INTRA_OP_THREADS` | `0` (auto) | Set to 1 or 2 when serving concurrent requests. See [Capacity planning](#capacity-planning). |
| `WILDECHO_LOG_LEVEL` | `INFO` | Root log level. |
| `WILDECHO_LOG_FORMAT` | `text` | `text` for local development, `json` for hosted log aggregators. See [Logging and request IDs](#logging-and-request-ids). |
| `WILDECHO_REQUEST_ID_HEADER` | `X-Request-ID` | Response header carrying the request ID; reused if the client sends it on the way in. |
| `WILDECHO_FEEDBACK_ENABLED` | `true` | Enables `POST /v1/feedback`. Set `false` to remove the endpoint (404). |
| `WILDECHO_FEEDBACK_DB_PATH` | `data/feedback.sqlite3` | SQLite file, created on first use. See [Feedback](#feedback). |
| `WILDECHO_FEEDBACK_STORE_AUDIO` | `true` | Save uploaded clips from feedback submissions to disk. Set `false` to keep only the text correction. |
| `WILDECHO_FEEDBACK_CLIPS_DIR` | `data/feedback_clips` | Where those clips are saved, if enabled. |
| `WILDECHO_FEEDBACK_MAX_UPLOAD_BYTES` | `26214400` (25 MB) | Same idea as `WILDECHO_MAX_UPLOAD_BYTES`, for feedback's optional file. |

The two `data/feedback*` defaults resolve relative to the working directory the
process starts in (there's no packaged copy to fall back to, unlike the taxonomy
and calibration files) - every deployment config in this repo (Docker, Fly, Render)
sets them explicitly, so this only matters if you run `uvicorn` directly yourself.

### Why the threshold is 0.3

It was measured, not guessed. Against the real model on real input:

| Input | Top softmax score |
| --- | --- |
| Clean European Nightjar recording (bundled fixture) | 0.909 |
| Longer, noisier nightjar recording | 0.634 |
| 1 kHz pure tone | 0.202 (and the top class is `Alarm`, a sound event) |
| White noise | 0.050 |
| Digital silence | 0.010 |

0.3 sits in the gap between real detections and junk. Raise it if you would rather
show nothing than show a bad guess; lower it if you would rather show a weak lead.
This is the *base* threshold; each taxonomic group can shift it further - see below.

## Confidence calibration

Perch is documented (see [above](#it-is-a-bird-model-that-also-knows-some-other-things))
as far stronger on birds than on the other taxa it covers. Its raw softmax score
doesn't know that: a mammal prediction at 0.6 confidence is not the same reliability
as a bird prediction at 0.6, but the number alone can't tell you which you're
looking at.

[`data/calibration.yaml`](data/calibration.yaml) applies a per-taxonomic-group
adjustment to correct for this, without a code change:

```yaml
groups:
  mammal:
    confidence_scale: 0.90    # displayed confidence is multiplied by this
    threshold_offset: 0.05    # added to WILDECHO_LOW_CONFIDENCE_THRESHOLD for this group
```

* **`confidence_scale`** discounts a group's displayed `confidence` (clamped to
  `[0, 1]`). Birds ship at `1.0` (no discount); frogs and insects at `0.85`;
  mammals at `0.90`; everything else at `0.75`.
* **`threshold_offset`** is added to `WILDECHO_LOW_CONFIDENCE_THRESHOLD` before
  deciding `low_confidence` *for that group*, so a discounted group also needs to
  clear a higher bar before it's shown as trustworthy.

**Ranking is never touched.** Which candidate is `predictions[0]` is decided purely
by the model's own `raw_confidence`; calibration only changes the displayed
`confidence` and `low_confidence` flag for each already-ranked candidate. That
means the displayed `confidence` values in a response are usually, but not always,
in descending order - if you need a monotonic order regardless of taxonomic group,
sort by `raw_confidence` instead.

The shipped values are **placeholders reflecting the documented coverage gap, not a
fitted correction against real accuracy data.** There's no accuracy benchmark run
per group yet - only the qualitative fact that Perch's training data is bird-heavy.
Tune `data/calibration.yaml` once you have real signal, ideally sourced from
[`/v1/feedback`](#feedback) corrections grouped by taxonomic group: if mammal
corrections are disproportionately common relative to how often mammals are
predicted, that group's `confidence_scale` should probably drop further.

Format (YAML or JSON, sniffed by extension) and full field docs are in the comments
at the top of `data/calibration.yaml`. A missing file is not an error - the service
falls back to no calibration at all, identical to before this feature existed.

## Feedback

`POST /v1/feedback` lets a client tell you when the model got a clip wrong and what
the correct species actually was. This is the mechanism for improving accuracy over
time (calibration tuning, spotting systematic confusions, eventually retraining
priorities) - see [Confidence calibration](#confidence-calibration) for how the data
feeds back in.

**This is opt-in and entirely local by default.** Concretely:

* Nothing is collected unless a client calls the endpoint. Identifying a clip via
  `/v1/identify` never writes anything to the feedback store on its own.
* Storage is a single SQLite file on this machine (`WILDECHO_FEEDBACK_DB_PATH`,
  default `data/feedback.sqlite3`), created on first use.
* **Nothing submitted here is sent anywhere else.** There is no telemetry, no
  phone-home, no remote sync of any kind in this codebase. If you want to
  aggregate feedback across multiple self-hosted instances, that's a sync you
  build yourself against the SQLite file (or the Postgres store, below) - this
  project deliberately does not make that choice for you.
* Set `WILDECHO_FEEDBACK_ENABLED=false` to remove the endpoint entirely (`404`) if
  you don't want to collect corrections at all.
* Set `WILDECHO_FEEDBACK_STORE_AUDIO=false` to keep only the text correction and
  discard any uploaded clip immediately after decoding, if you want the accuracy
  signal without retaining audio.

### Swapping in Postgres

`SQLiteFeedbackStore` in [`src/wildecho_api/feedback.py`](src/wildecho_api/feedback.py)
implements a two-method interface (`FeedbackStore.init` / `.insert`) against plain
SQL with no SQLite-specific syntax beyond `AUTOINCREMENT`. To move to Postgres:

1. Implement the same interface with `psycopg` or SQLAlchemy, swapping
   `INTEGER PRIMARY KEY AUTOINCREMENT` for `SERIAL PRIMARY KEY` (or
   `GENERATED ALWAYS AS IDENTITY`) in the schema.
2. Point the store construction in `src/wildecho_api/main.py`'s
   `_init_feedback_store` at your implementation instead of `SQLiteFeedbackStore`.
3. Everything else - the endpoint, the matching logic, the response shape - is
   unaffected, since they only depend on the abstract interface.

This is a genuine seam, not a promise: no Postgres backend ships in this repo today.
Contributions implementing one are welcome via PR.

## Logging and request IDs

Every request gets an ID: reused from the client's `X-Request-ID` header if it sent
one, otherwise generated fresh. It shows up in three places:

1. Every log line emitted anywhere during that request (structured logging, see
   below).
2. The `X-Request-ID` response header.
3. The `request_id` field in `POST /v1/identify`'s response body.

That last one is the point: a mobile client can hand the ID from an identification
back to `POST /v1/feedback`'s `request_id` field when correcting it, and you can
then `grep` the JSON logs for that exact ID to see the whole request's server-side
trail - what was decoded, how long inference took, what was predicted - when
debugging a misidentification a user reported.

Set `WILDECHO_LOG_FORMAT=json` for one JSON object per line (what Fly, Render, and
most log aggregators want); the default `text` is friendlier for a local terminal.
Both include the request ID on every line logged during a request:

```
# text (default)
2026-07-30 06:31:32 INFO     [9cbd9cca508b4fb1937b5da48d3850dc] wildecho_api.main: POST /v1/identify -> 200 (455.0ms)

# json
{"timestamp": "2026-07-30T06:31:32", "level": "INFO", "logger": "wildecho_api.main", "message": "POST /v1/identify -> 200 (455.0ms)", "request_id": "9cbd9cca508b4fb1937b5da48d3850dc"}
```

## Self-hosting notes

**There is no authentication.** That is out of scope by design. Anything you expose
publicly is an open, CPU-heavy endpoint. Read [SECURITY.md](SECURITY.md) before
putting this on the internet; it has a threat model table and what to do about each
row. The short version:

1. **Tune the rate limit.** `20/hour` per IP is a cautious public default and a bad
   one for your own phone. Raise it for private use.
2. **Put a reverse proxy in front** (Caddy, nginx, Traefik) for TLS, and add
   authentication there if you need it.
3. **Narrow CORS.** A native mobile app does not use CORS at all, so if you are not
   also serving a browser client, set `WILDECHO_CORS_ORIGINS` to your origins or
   nothing.
4. **Cap CPU.** Uncomment the `deploy.resources` block in
   [`docker-compose.yml`](docker-compose.yml). One 300-second clip can saturate
   every core. See [Capacity planning](#capacity-planning) for the math.
5. **Keep `ffmpeg` patched.** It parses your untrusted input. It is the largest
   attack surface in the stack, which is exactly why it runs in a subprocess with a
   timeout rather than in-process.
6. **Decide about feedback.** `POST /v1/feedback` is on by default and stores
   locally. Read [Feedback](#feedback) and decide whether that's what you want for
   a public instance - the data never leaves the box on its own, but it does
   accumulate on it.

Rate limiting keys on the socket peer address, so behind a proxy every request looks
like it comes from the proxy. Configure real client IP forwarding at the proxy and
run uvicorn with `--proxy-headers --forwarded-allow-ips=<proxy-ip>`.

## Deployment

Two platforms are set up here because they're what this project actually deploys
to: **Fly.io** ([`fly.toml`](fly.toml)) and **Render** ([`render.yaml`](render.yaml)).
Both mount a persistent volume/disk for the model weights and the feedback
database, and both set `WILDECHO_AUTO_DOWNLOAD_MODEL=true` so the ~390 MiB weights
are fetched automatically on first boot rather than needing a manual step - see
"Auto-download on hosted platforms" below for how that works.

Other platforms (Railway, Google Cloud Run, a bare VPS with `docker compose`, etc.)
are not configured here, but the image is a standard multi-stage Dockerfile with no
platform-specific assumptions beyond a writable volume and one environment
variable, so porting this to another platform is mechanical. **Contributions adding
a config for another platform are welcome via PR** - see
[CONTRIBUTING.md](CONTRIBUTING.md).

### Fly.io

```bash
fly launch --no-deploy                              # creates the app; pick a unique name
fly volumes create wildecho_data --region iad --size 3
fly deploy
```

`fly.toml` sets `WILDECHO_MODEL_PATH` and the feedback paths onto the `/data`
volume, so they persist across deploys and restarts. Check status with
`fly status` and `fly logs`; the health check is `/v1/health`.

### Render

Push this repo to GitHub, then in the Render dashboard: **New +** -> **Blueprint**,
and point it at the repo. Render reads [`render.yaml`](render.yaml) and provisions
the service and its disk. The blueprint requests the `standard` plan (2 GB) - see
[Capacity planning](#capacity-planning) for why the model needs that much memory;
Render's smallest `starter` plan (512 MB) will not fit it.

### Hugging Face Spaces (needs HF PRO)

```bash
hf auth login                      # write token from huggingface.co/settings/tokens
scripts/deploy_hf_space.sh         # creates/updates <you>/wildecho-api
```

Docker Spaces on the CPU tier (2 vCPU, 16 GB RAM) require a Hugging Face PRO subscription; the model fits comfortably. Spaces has no
persistent volume on that tier and never runs containers as root, so
[`deploy/huggingface/Dockerfile`](deploy/huggingface/Dockerfile) bakes the weights
in at build time, runs as uid 1000, and disables feedback audio storage. Free
Spaces sleep after 48 hours without traffic; the first request afterwards wakes
them, which takes a minute or two.

### A VPS shared with other apps

[`deploy/droplet/`](deploy/droplet/) runs the Spaces image behind an existing nginx
on a small shared server (this is how the public instance runs). `run.sh` caps the
container at 1.4 GB, runs one inference at a time (others queue), limits clips to
30 s, and turns off the ONNX memory arena so peak memory is released between
requests. `nginx-wildecho.conf` is the matching site; add TLS with
`certbot --nginx -d <your-domain>`.

All configs set `FORWARDED_ALLOW_IPS=*` so uvicorn trusts the platform
proxy's `X-Forwarded-For` header. Without it, every client appears to share the
proxy's IP and the per-IP rate limit becomes one global limit.

### Auto-download on hosted platforms

Locally, you run `python scripts/download_model.py` yourself and mount the result
into the container (`docker-compose.yml`'s default). There's no equivalent host to
run that on for a hosted platform, so instead [`docker/entrypoint.sh`](docker/entrypoint.sh)
does it automatically when `WILDECHO_AUTO_DOWNLOAD_MODEL=true` and the weights
aren't already present in the mounted volume - once, on first boot; every restart
after that finds them already there.

This is opt-in (default `false`) on purpose: it keeps CI's "start the container with
no weights and expect a degraded health check" smoke test fast and unchanged, and it
means a plain `docker run` never silently starts a few-hundred-megabyte download
unless you asked for it.

The same entrypoint also handles a permissions wrinkle specific to hosted volumes: a
freshly created Fly volume or Render disk is typically root-owned, but the
container runs its actual server process as an unprivileged user (see
[SECURITY.md](SECURITY.md)). The entrypoint starts as root just long enough to fix
ownership of the mounted paths, then drops to the unprivileged user via `gosu`
before running anything else - including the download - the same pattern used by
images like `postgres` and `grafana` for this exact reason. Local `docker compose`
users are unaffected: your bind-mounted `./models` is read-only, so the ownership
fix silently no-ops on it.

## Capacity planning

Rough figures on an Apple Silicon laptop, CPU only, `onnxruntime` 1.24:

| | |
| --- | --- |
| Model load (once, at startup) | ~0.9 s |
| 10-second clip (3 windows), inference only | ~0.2 s (~66 ms/window) |
| 15-second clip (5 windows), inference only | ~0.33 s (~66 ms/window) |
| Resident memory | ~1.2 GB |

Memory is dominated by the 390 MiB model plus ONNX Runtime's arenas. Budget 2 GB per
container - this is why the Fly and Render configs both request that much.

### How many concurrent requests can a small instance handle?

`decode_file` (an `ffmpeg` subprocess) and `model.identify` (ONNX Runtime) both run
inside `run_in_threadpool` rather than directly on the event loop, specifically so
concurrent requests don't serialize behind one another - and because both release
the GIL while their C code runs, requests genuinely executing at the same time can
use separate cores.

Work it through for a typical 10-second clip:

* **Inference:** 3 windows x ~66 ms = **~200 ms** of CPU-bound work.
* **Decode:** ffmpeg subprocess overhead adds roughly another **~100 ms** on top
  (varies by format and container, mp3/wav are cheap; measure your own with the
  `inference_ms` field in the response plus your own request timing, since that
  field deliberately excludes decode).
* **Total: ~300 ms of CPU-bound work per typical request.**

On a **2 vCPU** instance (the Fly/Render default here): roughly
`2 cores / 0.3 s per request` &approx; **~6-7 requests/second sustained**, or about
**400/minute**. On **1 vCPU**, roughly half that: **~3 requests/second**, **~200/minute**.

These are theoretical ceilings, not a promise - they assume nothing else on the box
is competing for CPU, that clips are short (a 300-second clip, the configured
maximum, is roughly 30x this cost), and that request parsing/serialization
overhead is negligible (it mostly is, relative to inference). Treat them as the
right order of magnitude for planning, not an SLA. **The per-IP rate limit
(`20/hour` default) is not the relevant constraint here** - it exists to stop one
client from hammering the service, not to shape aggregate capacity across many
users; the CPU ceiling above is what you should actually plan around.

Two concrete levers if you need more throughput:

1. **Tune `WILDECHO_ONNX_INTRA_OP_THREADS`.** By default (`0`) ONNX Runtime lets a
   *single* inference call use every visible core. That's great for latency on one
   request at a time, but under concurrent load it means N simultaneous requests
   are each trying to grab every core for themselves, fighting each other instead
   of the OS scheduler cleanly interleaving them. Setting this to `1` or `2` on a
   small, concurrently-loaded instance usually improves aggregate throughput at a
   small cost to any single request's latency.
2. **Scale horizontally, not with more workers per box.** Each worker process loads
   its own full copy of the model (~1.2 GB resident). A 2 GB instance comfortably
   fits one worker, not several - so prefer running one process per container
   (what `fly.toml` and `render.yaml` both do) and adding more machines under load
   (Fly Machines autoscaling, Render autoscaling) over passing `--workers N` to
   uvicorn on a single small instance.

## How it works

```
upload -> ffmpeg (decode, downmix, resample) -> 5s windows @ 2.5s stride
       -> Perch 2.0 ONNX -> mean logits -> softmax
       -> drop 198 sound event classes -> top 10
```

1. **Decode.** `ffmpeg` runs in a subprocess and produces 32 kHz mono float32. All
   container parsing happens outside the API process. `ffprobe` reads the duration
   first so an over-long clip is rejected before any PCM is allocated.
2. **Window.** Audio is cut into 5-second windows (160,000 samples, the model's
   fixed input) with a 2.5-second stride, so a call landing on a boundary still
   appears whole in a neighbouring window. Clips shorter than 5 seconds are
   zero-padded to one window. A final end-anchored window is added when striding
   would leave a tail uncovered, so the last seconds are never dropped.
3. **Score.** Windows go through the classifier in batches of 16. Only the `label`
   output is requested, so the runtime skips materialising the embeddings.
4. **Combine.** Logits are averaged across windows *before* the nonlinearity.
   Averaging probabilities instead would let one loud window dominate a long quiet
   recording. Then softmax over all 14,795 classes.
5. **Filter and rank.** The 198 general sound event classes are removed, and the top
   10 species are returned. If a sound event was the overall winner, that fact is
   logged and returned in `non_animal_top_class`.

### Where the model and labels come from

| Artifact | Source | License |
| --- | --- | --- |
| `perch_v2.onnx` (390 MiB) | [justinchuby/Perch-onnx](https://huggingface.co/justinchuby/Perch-onnx) | Apache-2.0 (Google LLC) |
| `labels.csv`, `perch_v2_ebird_classes.csv` | [cgeorgiaw/Perch](https://huggingface.co/cgeorgiaw/Perch) | Apache-2.0 (Google LLC) |
| English common names | [Wikidata](https://www.wikidata.org) property P1843 | CC0 |
| Taxonomic class | [GBIF Backbone Taxonomy](https://www.gbif.org/dataset/d7dddbf4-2cf0-4f39-9b2a-bb099caae36c) | CC-BY 4.0 |

The ONNX graph takes `inputs` of shape `[batch, 160000]` and emits `label` of shape
`[batch, 14795]`, alongside `embedding`, `spatial_embedding` and `spectrogram`. We
use only `label`.

[`data/taxonomy.csv`](data/taxonomy.csv) is committed (1.2 MB). It is derived
metadata, not weights, and shipping it means `docker compose up` needs no network
beyond the one-time weight download. Regenerate it with:

```bash
python scripts/build_taxonomy.py
```

Perch's own asset files give a raw label per index and, for birds, an eBird code.
They do not give common names or a taxonomic group, which is what a consumer app
needs, so the build script joins those in. The `index` column must line up exactly
with the model's output order; the loader validates this at startup because a
one-row shift would make every prediction confidently and silently wrong.

## Development

```bash
git clone https://github.com/arunrajiah/wildecho-api.git
cd wildecho-api
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Tests pass without the model weights. Anything needing real inference is marked
`model` and skipped automatically, which is how CI stays fast and green. Run the
full set after downloading the weights:

```bash
pytest -m model
```

The same three checks CI runs:

```bash
ruff check . && ruff format --check . && mypy src scripts && pytest
```

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions
(Conventional Commits, no model weights in git) and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Part of an open wildlife toolkit

Eight open source projects for listening to, identifying and mapping wildlife. They work together, but you rarely need more than one or two. Start from what you want to do:

| I want to | Use | What it is |
|---|---|---|
| See where birds and wildlife are moving, or download the data | [WildNetwork](https://github.com/arunrajiah/wildnetwork) | The live map and open data: [wildnetwork.arunrajiah.com](https://wildnetwork.arunrajiah.com) |
| Build a monitoring station from open hardware | [WildNetwork Base](https://github.com/arunrajiah/wildnetwork-base) | Software and a ready-to-flash SD card image (beta) for the open WildNetwork station (Raspberry Pi, microphone, solar) |
| Share detections from a BirdNET-Pi, BirdNET-Go, camera trap or bat detector you already have | [wdx-agent](https://github.com/arunrajiah/wdx-agent) | One small program that sends your station's detections. BirdWeather stations are already included and need nothing |
| Follow your own station on your phone | [BirdEcho](https://github.com/arunrajiah/birdecho) | Android app for BirdNET-Pi, BirdNET-Go and BirdWeather stations |
| Identify a sound you just heard | [WildEcho](https://github.com/arunrajiah/wildecho) | Phone app: record a clip, get ranked species |
| Run your own sound identification server | [wildecho-api](https://github.com/arunrajiah/wildecho-api) (this project) | Self-hosted species identification from audio, on CPU, no API keys |
| Check camera trap predictions before you use them | [SpeciesNet Studio](https://github.com/arunrajiah/speciesnet-studio) | Self-hosted review of SpeciesNet results |
| Make your own software or device produce or read detections in a common format | [WDX](https://github.com/arunrajiah/wildlife-detection-exchange) | The open format for one AI wildlife detection; maps to Darwin Core |

**How this one fits.** wildecho-api is the identification server behind WildEcho.

**How they connect:** stations (a WildNetwork Base, BirdNET-Pi, BirdNET-Go, camera traps) produce detections; wdx-agent sends them in the WDX format; WildNetwork maps them. Connected today: wdx-agent and the WildNetwork Base send to WildNetwork, and WildEcho uses wildecho-api. Planned: Base setup in BirdEcho, and WDX export from wildecho-api and SpeciesNet Studio.

## Credits

**Perch 2.0 is the work of Google Research's Perch team and its collaborators.**
This project is a wrapper around their model and takes no credit for its
capabilities. If you use wildecho-api in research, cite Perch, not this repository:

```bibtex
@article{perch2,
  title  = {Perch 2.0: The Bittern Lesson for Bioacoustics},
  author = {Hamer, Jenny and Denton, Tom and others},
  journal= {arXiv preprint arXiv:2508.04665},
  year   = {2025}
}
```

- [google-research/perch](https://github.com/google-research/perch), the research codebase
- [perch-hoplite](https://github.com/google-research/perch-hoplite), the official inference and tooling repo
- [Perch 2.0 paper](https://arxiv.org/abs/2508.04665)
- [justinchuby/Perch-onnx](https://huggingface.co/justinchuby/Perch-onnx) for the ONNX export
- [cgeorgiaw/Perch](https://huggingface.co/cgeorgiaw/Perch) for the SavedModel and label assets
- Perch's training data draws on [Xeno-canto](https://xeno-canto.org),
  [iNaturalist](https://www.inaturalist.org), the
  [Tierstimmenarchiv](https://www.museumfuernaturkunde.berlin/en/science/animal-sound-archive)
  and [FSD50K](https://zenodo.org/records/4060432), all built by people who recorded
  and labelled animals so that models like this could exist
- The bundled test clip is a CC0 European Nightjar recording by ChristianSW via
  [Xeno-canto XC1008591](https://www.xeno-canto.org/1008591)

Full attribution and license terms for every third-party component are in
[NOTICE](NOTICE).

## License

The wildecho-api wrapper code is [MIT](LICENSE), copyright 2026 Arun Rajiah.

**The Perch 2.0 model is not MIT.** It is Apache-2.0, copyright Google LLC, and is
not redistributed in this repository. Downloading it with
`scripts/download_model.py` means accepting Apache-2.0 for those weights. Your
obligations for the model are Google's terms, not this project's.

Derived data in `data/taxonomy.csv` carries Apache-2.0 (Perch labels), CC0
(Wikidata) and CC-BY-4.0 (GBIF) terms. See [NOTICE](NOTICE).

---

If this is useful to you, [sponsoring the project](https://github.com/sponsors/arunrajiah)
helps keep it maintained.
