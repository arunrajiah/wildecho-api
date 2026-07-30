"""Shared fixtures.

Environment variables are set at import time, before ``wildecho_api.main`` is ever
imported, because that module reads settings once at module scope to build the CORS
middleware and the rate limiter.

Tests here never need the real 390 MiB weights. A fake ONNX session stands in and
returns deterministic logits, so the whole pipeline (windowing, batching, averaging,
softmax, sound-event filtering, ranking) is exercised on every CI run. Tests that do
need real weights are marked ``model`` and deselected by default.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"

NIGHTJAR_CLIP = FIXTURES / "european_nightjar_xc1008591.mp3"

os.environ.setdefault("WILDECHO_TAXONOMY_PATH", str(REPO_ROOT / "data" / "taxonomy.csv"))
os.environ.setdefault("WILDECHO_MODEL_PATH", str(REPO_ROOT / "models" / "perch_v2.onnx"))
# Off by default so ordinary tests are not throttled. The rate limiting test turns
# the limiter back on for itself.
os.environ.setdefault("WILDECHO_RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("WILDECHO_RATE_LIMIT", "1000/minute")
os.environ.setdefault("WILDECHO_LOG_LEVEL", "WARNING")

from wildecho_api import inference, main  # noqa: E402
from wildecho_api.config import (  # noqa: E402
    NUM_CLASSES,
    TARGET_SAMPLE_RATE,
    WINDOW_SAMPLES,
    Settings,
    get_settings,
)
from wildecho_api.taxonomy import Taxonomy, load_taxonomy  # noqa: E402

TAXONOMY_PATH = REPO_ROOT / "data" / "taxonomy.csv"


# ---------------------------------------------------------------------------
# Fake ONNX session
# ---------------------------------------------------------------------------
class _IO:
    """Duck-types ``onnxruntime.NodeArg``."""

    def __init__(self, name: str, shape: list[Any]) -> None:
        self.name = name
        self.shape = shape
        self.type = "tensor(float)"


class FakeSession:
    """Minimal stand-in for ``onnxruntime.InferenceSession``.

    Returns a fixed logit vector for every window, optionally scaled per window, so
    assertions about ranking and averaging are exact rather than approximate.
    """

    def __init__(self, logits: np.ndarray, input_samples: int = WINDOW_SAMPLES) -> None:
        self._logits = np.asarray(logits, dtype=np.float32)
        self._input_samples = input_samples
        self.calls: list[tuple[int, int]] = []  # (batch_size, samples) per run()

    def get_inputs(self) -> list[_IO]:
        return [_IO("inputs", ["batch", self._input_samples])]

    def get_outputs(self) -> list[_IO]:
        return [
            _IO("embedding", ["batch", 1536]),
            _IO("spatial_embedding", ["batch", 16, 4, 1536]),
            _IO("spectrogram", ["batch", 500, 128]),
            _IO("label", ["batch", self._logits.size]),
        ]

    def run(self, output_names: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        batch = feeds["inputs"]
        self.calls.append((batch.shape[0], batch.shape[1]))
        assert output_names == ["label"], "inference should request only the label output"
        return [np.tile(self._logits, (batch.shape[0], 1))]


def logits_favouring(index: int, *, peak: float = 12.0, floor: float = -2.0) -> np.ndarray:
    """A logit vector where ``index`` clearly wins."""
    logits = np.full(NUM_CLASSES, floor, dtype=np.float32)
    logits[index] = peak
    return logits


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def taxonomy() -> Taxonomy:
    """The real committed taxonomy. Loading it also asserts the file is well-formed."""
    return load_taxonomy(TAXONOMY_PATH, expected_rows=NUM_CLASSES)


@pytest.fixture
def settings() -> Settings:
    return get_settings()


@pytest.fixture
def nightjar_index(taxonomy: Taxonomy) -> int:
    return next(e.index for e in taxonomy.entries if e.label == "Caprimulgus europaeus")


@pytest.fixture
def wind_index(taxonomy: Taxonomy) -> int:
    """A general sound event class, used to test non-animal filtering."""
    return next(e.index for e in taxonomy.entries if e.label == "Wind")


@pytest.fixture
def make_model(taxonomy: Taxonomy, settings: Settings):
    """Build a :class:`PerchModel` backed by a fake session."""

    def _make(logits: np.ndarray) -> inference.PerchModel:
        return inference.PerchModel(
            session=FakeSession(logits),  # type: ignore[arg-type]
            taxonomy=taxonomy,
            settings=settings,
        )

    return _make


@pytest.fixture
def mocked_model(make_model, nightjar_index: int) -> Iterator[inference.PerchModel]:
    """Install a fake model into the process-wide slot for the duration of a test."""
    model = make_model(logits_favouring(nightjar_index))
    inference.set_model(model)
    try:
        yield model
    finally:
        inference.set_model(None)


@pytest.fixture
def no_model() -> Iterator[None]:
    inference.set_model(None, error="Model file not found (test).")
    try:
        yield
    finally:
        inference.set_model(None)


@pytest.fixture
def client(make_model, nightjar_index: int) -> Iterator[Any]:
    """A TestClient whose app has the fake model loaded and the taxonomy available.

    The model is installed *inside* the TestClient context on purpose. Entering that
    context runs the app's lifespan, which tries to load the real weights and would
    otherwise clear whatever we injected beforehand.
    """
    from fastapi.testclient import TestClient

    with TestClient(main.app) as test_client:
        inference.set_model(make_model(logits_favouring(nightjar_index)))
        main.state.taxonomy = load_taxonomy(TAXONOMY_PATH, expected_rows=NUM_CLASSES)
        main.state.taxonomy_error = None
        _reset_rate_limiter()
        try:
            yield test_client
        finally:
            inference.set_model(None)


@pytest.fixture
def bare_client() -> Iterator[Any]:
    """A TestClient with no model loaded, for degraded-mode assertions.

    Clears the model after the lifespan has run, so this holds even on a machine
    where the real weights are present.
    """
    from fastapi.testclient import TestClient

    with TestClient(main.app) as test_client:
        inference.set_model(None, error="Model file not found (test).")
        main.state.taxonomy = load_taxonomy(TAXONOMY_PATH, expected_rows=NUM_CLASSES)
        main.state.taxonomy_error = None
        _reset_rate_limiter()
        try:
            yield test_client
        finally:
            inference.set_model(None)


def _reset_rate_limiter() -> None:
    """Clear slowapi's in-memory counters so tests do not leak limits into each other."""
    storage = getattr(main.limiter, "_storage", None)
    reset = getattr(storage, "reset", None)
    if callable(reset):
        reset()


@pytest.fixture
def reset_rate_limiter() -> Iterator[None]:
    _reset_rate_limiter()
    yield
    _reset_rate_limiter()


# ---------------------------------------------------------------------------
# Synthetic audio helpers
# ---------------------------------------------------------------------------
def tone(
    seconds: float,
    frequency: float = 440.0,
    sample_rate: int = TARGET_SAMPLE_RATE,
    amplitude: float = 0.3,
) -> np.ndarray:
    """A sine wave. Loud enough to pass the silence check."""
    t = np.arange(int(seconds * sample_rate), dtype=np.float64) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def noise(seconds: float, sample_rate: int = TARGET_SAMPLE_RATE, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * sample_rate)) * 0.1).astype(np.float32)


@pytest.fixture
def wav_bytes():
    """Encode a numpy array as a WAV file in memory, without a soundfile dependency."""
    import io
    import wave

    def _encode(samples: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> bytes:
        clipped = np.clip(samples, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype("<i2")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())
        return buffer.getvalue()

    return _encode


def has_real_model() -> bool:
    return get_settings().model_path.exists()


requires_model = pytest.mark.skipif(
    not has_real_model(),
    reason="real Perch weights not present; run `python scripts/download_model.py`",
)
