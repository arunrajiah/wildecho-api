# Test fixtures

## `european_nightjar_xc1008591.mp3`

A 10-second excerpt of a European Nightjar (*Caprimulgus europaeus*) song.

| | |
| --- | --- |
| Species | European Nightjar (*Caprimulgus europaeus*) |
| Recordist | ChristianSW |
| Original source | [Xeno-canto XC1008591](https://www.xeno-canto.org/1008591) |
| Obtained from | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Caprimulgus_europaeus_-_European_Nightjar_XC1008591.mp3) |
| License | [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) (public domain dedication) |
| Recorded | 2025-06-22, Schneverdingen, Germany |
| Modification | Trimmed to 10s from 2.0s, downmixed to mono, resampled to 32 kHz, re-encoded as 64 kbps MP3 |

CC0 places no restriction on use and imposes no attribution requirement. Credit is
given here and in [`NOTICE`](../../NOTICE) voluntarily, because the recordist did
the work.

### Why this clip

It is a clean, single-species recording of a distinctive vocalisation, which makes
it a stable regression target. Perch 2.0 scores it at roughly 0.91 softmax
confidence for the correct species, well clear of the 0.30 low-confidence
threshold, so `tests/test_inference.py` can assert on the top-1 result without
being flaky.

The nightjar's churring song is also a useful case because it is a sustained trill
rather than a short call, so it exercises multi-window averaging: 10 seconds
produces 3 overlapping windows.

### Adding another fixture

Anything committed here must be public domain, CC0, or CC-BY, and the license has
to be verifiable from a stable URL. Record the same table of provenance above, add
an entry to `NOTICE`, and keep the file small (under ~200 KB) by trimming and
re-encoding. Do not add recordings under CC-BY-NC or with unclear provenance.
