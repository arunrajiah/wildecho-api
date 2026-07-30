# wildecho-api

**Self-hostable species ID from audio, powered by Google's open Perch 2.0 model.**

[![CI](https://github.com/arunrajiah/wildecho-api/actions/workflows/ci.yml/badge.svg)](https://github.com/arunrajiah/wildecho-api/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Model: Apache 2.0](https://img.shields.io/badge/Model-Apache%202.0-green.svg)](https://github.com/google-research/perch)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

Upload a short recording of an animal, get back ranked species candidates. It runs
entirely on your own hardware, on CPU, with no API keys and no data leaving the box.

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
      "low_confidence": false,
      "class_index": 2211
    }
  ],
  "low_confidence": false,
  "non_animal_top_class": null,
  "model_version": "perch_v2",
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
- [Self-hosting notes](#self-hosting-notes)
- [How it works](#how-it-works)
- [Development](#development)
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
| `predictions[]` | Top candidates, best first. Non-animal classes excluded. |
| `predictions[].common_name` | English common name, or `null`. Fall back to `scientific_name`. |
| `predictions[].scientific_name` | Latin binomial from the Perch label set. |
| `predictions[].taxonomic_group` | One of `bird`, `frog`, `insect`, `mammal`, `other`. |
| `predictions[].confidence` | Softmax score in `[0, 1]`. Ranking signal, not a calibrated probability. |
| `predictions[].low_confidence` | This candidate's own score is below the threshold. |
| `predictions[].class_index` | Raw Perch output index, for joining against `data/taxonomy.csv`. |
| `low_confidence` | The top prediction is below the threshold. Treat the whole result as weak. |
| `non_animal_top_class` | Set when the highest-scoring class of all was a sound event. Distrust the list. |
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
| `WILDECHO_TOP_K` | `10` | Candidates returned. |
| `WILDECHO_LOW_CONFIDENCE_THRESHOLD` | `0.3` | See the calibration note below. |
| `WILDECHO_RATE_LIMIT` | `20/hour` | Per client IP. **Tune this.** |
| `WILDECHO_RATE_LIMIT_ENABLED` | `true` | Set `false` for private use behind a firewall. |
| `WILDECHO_CORS_ORIGINS` | `*` | Comma-separated. **Narrow this in production.** |
| `WILDECHO_MAX_UPLOAD_BYTES` | `26214400` (25 MB) | |
| `WILDECHO_MAX_DURATION_SECONDS` | `300` | Inference cost scales with duration. |
| `WILDECHO_MIN_DURATION_SECONDS` | `0.5` | |
| `WILDECHO_ONNX_INTRA_OP_THREADS` | `0` (auto) | Set to 1 or 2 on a shared box. |

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
   every core.
5. **Keep `ffmpeg` patched.** It parses your untrusted input. It is the largest
   attack surface in the stack, which is exactly why it runs in a subprocess with a
   timeout rather than in-process.

Rate limiting keys on the socket peer address, so behind a proxy every request looks
like it comes from the proxy. Configure real client IP forwarding at the proxy and
run uvicorn with `--proxy-headers --forwarded-allow-ips=<proxy-ip>`.

### Performance

Rough figures on an Apple Silicon laptop, CPU only, `onnxruntime` 1.24:

| | |
| --- | --- |
| Model load (once, at startup) | ~0.9 s |
| 10-second clip (3 windows) | ~0.2 s |
| 15-second clip (5 windows) | ~0.33 s |
| Resident memory | ~1.2 GB |

Memory is dominated by the 390 MiB model plus ONNX Runtime's arenas. Budget 2 GB
per container.

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
