---
title: WildEcho API
emoji: 🐦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Species ID from audio clips, powered by Perch 2.0
---

# wildecho-api

Public instance of [wildecho-api](https://github.com/arunrajiah/wildecho-api), the backend
for the WildEcho Android app. Send a short audio clip to `POST /v1/identify` and get ranked
species candidates. Interactive docs at `/docs`.

Model: Google's Perch 2.0 (Apache-2.0). Not affiliated with or endorsed by Google.
Rate limited per client IP. Clips are not stored.
