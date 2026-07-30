#!/usr/bin/env python3
"""Download the Perch 2.0 ONNX weights into ``models/``.

The weights are ~390 MiB and are deliberately not committed to this repository.
They are published by Google Research under the Apache License 2.0; this script
just fetches them. See NOTICE for full attribution.

Which file and why:

    perch_v2.onnx           <- what we use. Input `inputs` [batch, 160000], output
                               `label` [batch, 14795] plus embeddings.
    perch_v2_no_dft.onnx    <- same model with the DFT op rewritten as a matmul, for
                               runtimes lacking a DFT kernel. Slightly larger and
                               slower; use `--variant no_dft` only if the default
                               fails to load on your platform.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --force               # re-download
    python scripts/download_model.py --variant no_dft      # DFT-free export
    python scripts/download_model.py --skip-checksum       # accept an upstream update
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

USER_AGENT = "wildecho-api/0.1.0 model-downloader (+https://github.com/arunrajiah/wildecho-api)"

BASE_URL = "https://huggingface.co/justinchuby/Perch-onnx/resolve/main"


@dataclass(frozen=True)
class Variant:
    """One downloadable ONNX export, with the digest observed at pinning time."""

    filename: str
    size_bytes: int
    sha256: str
    note: str


VARIANTS: dict[str, Variant] = {
    "default": Variant(
        filename="perch_v2.onnx",
        size_bytes=409_148_616,
        sha256="bf0c8467a924cb074663970ca4a0ab1e143602121930209657d0dff5d5cefa1f",
        note="Standard export. Verified to load under onnxruntime CPU.",
    ),
    "no_dft": Variant(
        filename="perch_v2_no_dft.onnx",
        size_bytes=413_350_933,
        sha256="",  # not pinned; only verified by size
        note="DFT rewritten as matmul, for runtimes without a DFT kernel.",
    ),
}


def human(num_bytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TiB"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_size: int) -> None:
    """Stream a URL to ``destination`` via a temp file, with a progress line."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            total = int(response.headers.get("Content-Length") or expected_size or 0)
            written = 0
            with temporary.open("wb") as handle:
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
                    written += len(chunk)
                    if total:
                        percent = written * 100 / total
                        print(
                            f"\r  {human(written)} / {human(total)} ({percent:5.1f}%)",
                            end="",
                            file=sys.stderr,
                            flush=True,
                        )
                    else:
                        print(f"\r  {human(written)}", end="", file=sys.stderr, flush=True)
            print(file=sys.stderr)
    except urllib.error.HTTPError as exc:
        temporary.unlink(missing_ok=True)
        raise SystemExit(
            f"Download failed with HTTP {exc.code} for {url}\n"
            "The upstream repository may have moved. Check "
            "https://huggingface.co/justinchuby/Perch-onnx for the current file list."
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"Download failed: {exc}. Check your network and retry.") from exc

    temporary.replace(destination)


def verify(path: Path, variant: Variant, skip_checksum: bool) -> None:
    actual_size = path.stat().st_size
    if actual_size != variant.size_bytes:
        print(
            f"warning: {path.name} is {human(actual_size)}, expected "
            f"{human(variant.size_bytes)}. Upstream may have republished the file.",
            file=sys.stderr,
        )

    if skip_checksum or not variant.sha256:
        return

    print("  verifying checksum...", file=sys.stderr)
    actual = sha256_of(path)
    if actual != variant.sha256:
        raise SystemExit(
            f"Checksum mismatch for {path.name}.\n"
            f"  expected sha256: {variant.sha256}\n"
            f"  actual   sha256: {actual}\n\n"
            "The file downloaded is not the one this release was tested against. That is\n"
            "either an upstream republish or a corrupt transfer. Re-run to retry, or pass\n"
            "--skip-checksum if you have independently confirmed the new file is genuine."
        )
    print("  checksum OK", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        default="default",
        help="Which ONNX export to fetch (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_DIR,
        help="Where to put the weights (default: %(default)s).",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Do not verify the sha256. Use only if upstream has legitimately republished.",
    )
    args = parser.parse_args()

    variant = VARIANTS[args.variant]
    destination = args.output_dir / variant.filename

    print(f"Perch 2.0 ONNX weights ({args.variant}): {variant.note}", file=sys.stderr)
    print(
        "License: Apache-2.0, copyright Google LLC. Not covered by this repo's MIT license.",
        file=sys.stderr,
    )

    if destination.exists() and not args.force:
        print(
            f"\n{destination} already exists ({human(destination.stat().st_size)}). "
            "Nothing to do; pass --force to re-download.",
            file=sys.stderr,
        )
        verify(destination, variant, args.skip_checksum)
        return

    free = shutil.disk_usage(args.output_dir.parent).free
    needed = variant.size_bytes * 2  # the temp file plus the final file
    if free < needed:
        raise SystemExit(
            f"Not enough disk space: {human(free)} free, about {human(needed)} needed "
            "during download."
        )

    url = f"{BASE_URL}/{variant.filename}"
    print(f"\nDownloading {url}", file=sys.stderr)
    download(url, destination, variant.size_bytes)
    verify(destination, variant, args.skip_checksum)

    print(
        f"\nDone. Weights at {destination}\n"
        "This path is gitignored; do not commit it.\n\n"
        "Next: uvicorn wildecho_api.main:app --port 8000",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
