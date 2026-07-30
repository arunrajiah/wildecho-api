---
name: Bug report
about: Something is broken, crashes, or returns the wrong kind of response
title: "fix: "
labels: bug
assignees: ''
---

## What happened

<!-- A clear description of the actual behaviour. -->

## What you expected

<!-- What should have happened instead. -->

## Reproduction

Steps to reproduce, ideally including the exact request:

```bash
curl -X POST http://localhost:8000/v1/identify -F "file=@clip.wav"
```

## Response or error you got

<!-- Paste the full JSON response, traceback, or container logs. -->

```
paste here
```

## Audio clip (if relevant)

<!--
If the bug depends on a specific clip, please attach it or link to it, and state
its license. Do not upload audio you do not have the right to share.
-->

* Format and duration:
* Sample rate and channels (`ffprobe clip.wav` output is ideal):
* License / source:

## Environment

* wildecho-api version or commit:
* How you are running it: `docker compose up` / local uvicorn / other
* OS and architecture:
* Python version (if not using Docker):
* `ffmpeg -version` first line:
* Model file present in `models/`? Output of `ls -l models/`:

## Checklist

- [ ] I ran `python scripts/download_model.py` and `models/` contains the ONNX file
- [ ] `ffmpeg` is installed and on my `PATH`
- [ ] `GET /v1/health` shows the model as loaded
- [ ] I searched existing issues for this problem

## Note on misidentifications

If the service ran fine but named the wrong species, that is usually a model
accuracy limitation rather than a bug in this wrapper. Please read the
"Accuracy and limitations" section of the README first, then open the issue with
the correct species and how you know it. Those reports are welcome, they just
tend to become documentation rather than code changes.
