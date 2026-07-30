#!/usr/bin/env python3
"""Build ``data/taxonomy.csv``: the index -> species metadata table for Perch 2.0.

Perch 2.0 emits 14,795 logits. The upstream asset files tell us the raw label for
each index and, for birds, an eBird species code. They do NOT give us English
common names or a taxonomic group, which is what a consumer app actually wants to
display. This script joins the upstream labels against two open datasets to fill
those in, and writes a single flat CSV that the service loads at startup.

Sources and licenses (see NOTICE for the full statement):

  * ``labels.csv`` / ``perch_v2_ebird_classes.csv`` -- Perch 2.0 assets,
    Apache-2.0, Google LLC. Retrieved from huggingface.co/cgeorgiaw/Perch.
  * Common names -- Wikidata (property P1843), CC0. One batched SPARQL POST per
    2,000 names, so the whole pass takes seconds.
  * Taxonomic class -- GBIF Backbone Taxonomy via the species match API, CC-BY.
    One request per non-bird species (~4,900), so this pass takes ~10 minutes.

The output CSV is committed to the repository. It is small (~1 MB), derived from
openly licensed data, and committing it means ``docker compose up`` needs no
network access beyond the one-time model weight download. Re-run this script only
when upstream Perch labels change.

Usage:
    python scripts/build_taxonomy.py                # full build
    python scripts/build_taxonomy.py --offline      # labels + eBird only
    python scripts/build_taxonomy.py --limit-gbif 200   # quick smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(__file__).resolve().parent / ".taxonomy_cache"
OUTPUT_CSV = REPO_ROOT / "data" / "taxonomy.csv"

LABELS_URL = "https://huggingface.co/cgeorgiaw/Perch/resolve/main/assets/labels.csv"
EBIRD_URL = "https://huggingface.co/cgeorgiaw/Perch/resolve/main/assets/perch_v2_ebird_classes.csv"

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
GBIF_MATCH_URL = "https://api.gbif.org/v1/species/match"

USER_AGENT = "wildecho-api/0.1.0 taxonomy-builder (+https://github.com/arunrajiah/wildecho-api)"

#: Perch's 14,795 classes are a mix of iNaturalist species and FSD50K general
#: sound events. Every species label is a Latin binomial containing a space
#: ("Turdus migratorius"); every FSD50K label uses underscores instead
#: ("Human_voice", "Wind", "Chirp_and_tweet"). That single rule separates them
#: exactly: 14,597 species and 198 sound events.
SOUND_EVENT_HAS_NO_SPACE = True

NO_EBIRD = "no_ebird_code"

WIKIDATA_BATCH_SIZE = 2000
GBIF_WORKERS = 8

CSV_COLUMNS = [
    "index",
    "label",
    "kind",
    "scientific_name",
    "common_name",
    "group",
    "taxon_class",
    "ebird_code",
]


def log(message: str) -> None:
    print(f"[build_taxonomy] {message}", file=sys.stderr, flush=True)


def _http(
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
) -> bytes:
    request = urllib.request.Request(
        url,
        data=data,
        headers={"User-Agent": USER_AGENT, **(headers or {})},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload: bytes = response.read()
    return payload


def fetch_upstream_assets(refresh: bool = False) -> tuple[list[str], list[str]]:
    """Download and cache the two Perch asset files, returning them as lists.

    The first line of each file is a header naming the taxonomy version
    ("inat2024_fsd50k", "ebird2021"), not a class, so it is stripped.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out: list[list[str]] = []
    for url, name in ((LABELS_URL, "labels.csv"), (EBIRD_URL, "ebird_classes.csv")):
        path = CACHE_DIR / name
        if refresh or not path.exists():
            log(f"downloading {name}")
            path.write_bytes(_http(url))
        lines = path.read_text(encoding="utf-8").splitlines()
        log(f"{name}: header={lines[0]!r} rows={len(lines) - 1}")
        out.append(lines[1:])

    labels, ebird = out
    if len(labels) != len(ebird):
        raise SystemExit(f"asset length mismatch: {len(labels)} labels vs {len(ebird)} eBird codes")
    return labels, ebird


def is_sound_event(label: str) -> bool:
    """True for FSD50K general sound event classes (wind, rain, speech, ...)."""
    return " " not in label


def fetch_common_names(names: list[str]) -> dict[str, str]:
    """Map scientific name -> English common name using Wikidata P1843.

    Batched via SPARQL POST (GET blows the URI length limit). Coverage is around
    80%; uncovered species simply keep a blank common name and the API falls back
    to displaying the scientific name.
    """
    cache_path = CACHE_DIR / "common_names.json"
    if cache_path.exists():
        cached: dict[str, str] = json.loads(cache_path.read_text(encoding="utf-8"))
        log(f"common names: reusing cache ({len(cached)} entries)")
        return cached

    candidates: dict[str, list[str]] = {}
    batches = [
        names[i : i + WIKIDATA_BATCH_SIZE] for i in range(0, len(names), WIKIDATA_BATCH_SIZE)
    ]
    for number, batch in enumerate(batches, start=1):
        values = " ".join('"{}"'.format(n.replace("\\", "").replace('"', "")) for n in batch)
        query = (
            f"SELECT ?n ?c WHERE {{ VALUES ?n {{ {values} }} "
            "?i wdt:P225 ?n . ?i wdt:P1843 ?c . FILTER(LANG(?c) = 'en') }"
        )
        try:
            payload = _http(
                WIKIDATA_ENDPOINT,
                data=urllib.parse.urlencode({"query": query}).encode(),
                headers={
                    "Accept": "application/sparql-results+json",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                timeout=180,
            )
            bindings = json.loads(payload)["results"]["bindings"]
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
            log(f"  wikidata batch {number}/{len(batches)} failed ({exc}); skipping")
            continue

        for row in bindings:
            candidates.setdefault(row["n"]["value"], []).append(row["c"]["value"])
        log(f"  wikidata batch {number}/{len(batches)}: {len(candidates)} names so far")
        time.sleep(1.0)  # be a polite Wikidata client

    # Wikidata often lists several vernacular names. Pick deterministically:
    # shortest first, then alphabetical, so rebuilds produce identical output.
    resolved = {
        name: sorted(values, key=lambda v: (len(v), v))[0] for name, values in candidates.items()
    }
    cache_path.write_text(json.dumps(resolved, sort_keys=True), encoding="utf-8")
    log(f"common names: resolved {len(resolved)}/{len(names)}")
    return resolved


def _gbif_one(name: str) -> tuple[str, dict[str, str]]:
    url = f"{GBIF_MATCH_URL}?strict=false&name={urllib.parse.quote(name)}"
    for attempt in range(3):
        try:
            data: dict[str, Any] = json.loads(_http(url, timeout=30))
            return name, {
                "class": str(data.get("class") or ""),
                "order": str(data.get("order") or ""),
            }
        except (urllib.error.URLError, TimeoutError, ValueError):
            if attempt == 2:
                return name, {"class": "", "order": ""}
            time.sleep(1.0 + attempt)
    return name, {"class": "", "order": ""}


def fetch_taxon_classes(names: list[str], limit: int | None = None) -> dict[str, dict[str, str]]:
    """Map scientific name -> {class, order} using the GBIF Backbone.

    Only called for species without an eBird code; birds are already known.
    Results are cached because this is the slow pass (~9 minutes for ~4,900).
    """
    cache_path = CACHE_DIR / "gbif_classes.json"
    resolved: dict[str, dict[str, str]] = {}
    if cache_path.exists():
        resolved = json.loads(cache_path.read_text(encoding="utf-8"))
        log(f"gbif: reusing cache ({len(resolved)} entries)")

    pending = [n for n in names if n not in resolved]
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        return resolved

    log(f"gbif: looking up {len(pending)} species with {GBIF_WORKERS} workers")
    started = time.time()
    with ThreadPoolExecutor(GBIF_WORKERS) as pool:
        for done, (name, info) in enumerate(pool.map(_gbif_one, pending), start=1):
            resolved[name] = info
            if done % 500 == 0:
                rate = done / (time.time() - started)
                log(f"  gbif {done}/{len(pending)} ({rate:.1f}/s)")
    cache_path.write_text(json.dumps(resolved, sort_keys=True), encoding="utf-8")
    log(f"gbif: done in {time.time() - started:.0f}s")
    return resolved


def to_group(taxon_class: str, order: str) -> str:
    """Collapse a taxonomic class into the coarse group the API exposes."""
    if taxon_class == "Aves":
        return "bird"
    if taxon_class == "Insecta":
        return "insect"
    if taxon_class == "Mammalia":
        return "mammal"
    # Only frogs and toads (Anura) are reported as "frog"; salamanders and
    # caecilians fall through to "other" rather than being mislabelled.
    if taxon_class == "Amphibia" and order == "Anura":
        return "frog"
    return "other"


def build(offline: bool = False, limit_gbif: int | None = None, refresh: bool = False) -> None:
    labels, ebird_codes = fetch_upstream_assets(refresh=refresh)

    species_names = [label for label in labels if not is_sound_event(label)]
    sound_event_count = len(labels) - len(species_names)
    log(
        f"classes: {len(labels)} total, {len(species_names)} species, "
        f"{sound_event_count} sound events"
    )

    # Birds are settled by the upstream eBird mapping, no lookup required.
    bird_labels = {
        label
        for label, code in zip(labels, ebird_codes, strict=True)
        if code != NO_EBIRD and not is_sound_event(label)
    }
    log(f"birds resolved from eBird codes: {len(bird_labels)}")

    needs_class = sorted(set(species_names) - bird_labels)
    log(f"species needing a GBIF class lookup: {len(needs_class)}")

    common_names: dict[str, str] = {}
    taxon_info: dict[str, dict[str, str]] = {}
    if offline:
        log("offline mode: skipping Wikidata and GBIF enrichment")
    else:
        common_names = fetch_common_names(species_names)
        taxon_info = fetch_taxon_classes(needs_class, limit=limit_gbif)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    group_counts: dict[str, int] = {}
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        for index, (label, ebird_code) in enumerate(zip(labels, ebird_codes, strict=True)):
            if is_sound_event(label):
                writer.writerow([index, label, "sound_event", "", "", "", "", ""])
                group_counts["sound_event"] = group_counts.get("sound_event", 0) + 1
                continue

            info = taxon_info.get(label, {})
            taxon_class = "Aves" if label in bird_labels else info.get("class", "")
            group = to_group(taxon_class, info.get("order", ""))
            writer.writerow(
                [
                    index,
                    label,
                    "species",
                    label,
                    common_names.get(label, ""),
                    group,
                    taxon_class,
                    "" if ebird_code == NO_EBIRD else ebird_code,
                ]
            )
            group_counts[group] = group_counts.get(group, 0) + 1

    size_kb = OUTPUT_CSV.stat().st_size / 1024
    log(f"wrote {OUTPUT_CSV.relative_to(REPO_ROOT)} ({size_kb:.0f} KB)")
    for group, count in sorted(group_counts.items(), key=lambda kv: -kv[1]):
        log(f"  {group:12} {count}")
    with_common = sum(1 for name in species_names if common_names.get(name))
    if species_names:
        share = with_common / len(species_names)
        log(f"  common names: {with_common}/{len(species_names)} ({share:.0%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="skip Wikidata/GBIF enrichment; emit labels, eBird codes and bird groups only",
    )
    parser.add_argument(
        "--limit-gbif",
        type=int,
        default=None,
        metavar="N",
        help="only resolve N uncached species via GBIF (for smoke tests)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-download the upstream Perch asset files",
    )
    args = parser.parse_args()
    build(offline=args.offline, limit_gbif=args.limit_gbif, refresh=args.refresh)


if __name__ == "__main__":
    main()
