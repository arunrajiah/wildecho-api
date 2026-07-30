"""Console entry points.

``pip install wildecho-api`` exposes ``wildecho-download-model``, which is the same
thing as running ``python scripts/download_model.py`` from a clone. The logic lives
in the script so a clone needs nothing installed to fetch weights.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _script_path(name: str) -> Path:
    return Path(__file__).resolve().parent.parent.parent / "scripts" / name


def download_model_main() -> None:
    """Run ``scripts/download_model.py`` as if invoked directly."""
    script = _script_path("download_model.py")
    if not script.exists():
        print(
            "scripts/download_model.py is not available in this installation. "
            "Clone https://github.com/arunrajiah/wildecho-api and run it from there, "
            "or download perch_v2.onnx manually from "
            "https://huggingface.co/justinchuby/Perch-onnx into your models/ directory.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    runpy.run_path(str(script), run_name="__main__")
