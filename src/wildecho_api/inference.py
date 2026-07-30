"""Perch 2.0 ONNX inference: windowing, batching, scoring and ranking.

The model is loaded once per process at startup and reused. It is a pure function
of its input, so a single :class:`onnxruntime.InferenceSession` is shared across
requests without locking; ONNX Runtime's ``Run`` is thread-safe.

Pipeline for one request:

1. Resample to 32 kHz mono (a no-op for HTTP requests, ffmpeg already did it).
2. Cut into 5-second windows with a 2.5-second stride, so a call landing on a
   window boundary still appears whole in a neighbouring window.
3. Run the windows through the classifier head in batches.
4. Average logits across windows, then softmax.
5. Drop the 198 general sound event classes and return the top N species.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .audio import resample, to_mono, validate_samples
from .config import (
    NUM_CLASSES,
    TARGET_SAMPLE_RATE,
    WINDOW_SAMPLES,
    WINDOW_SECONDS,
    Settings,
    get_settings,
)
from .schemas import Prediction
from .taxonomy import Taxonomy, TaxonomyError, load_taxonomy

logger = logging.getLogger(__name__)

#: Hop between windows. Half a window, so every instant is covered twice except at
#: the clip's edges.
WINDOW_STRIDE_SAMPLES = WINDOW_SAMPLES // 2
WINDOW_STRIDE_SECONDS = WINDOW_SECONDS / 2

#: The graph exposes embeddings and the spectrogram too. We ask only for the
#: classifier head so the runtime can skip materialising the rest, which for a
#: 120-window clip saves ~50 MB of copies.
OUTPUT_LOGITS = "label"
INPUT_NAME = "inputs"

#: Windows per ONNX call. Bounds peak memory on long clips; 16 windows is ~10 MB
#: of input and ~1 MB of logits.
MAX_BATCH_WINDOWS = 16


class ModelLoadError(RuntimeError):
    """The ONNX file or the taxonomy could not be loaded."""


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Everything one identification produced, including what the API reports as metadata."""

    predictions: list[Prediction]
    low_confidence: bool
    non_animal_top_class: str | None
    windows_processed: int
    duration_seconds: float
    inference_ms: float


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exponentiated = np.exp(shifted, dtype=np.float64)
    normalised: np.ndarray = exponentiated / exponentiated.sum()
    return normalised


def make_windows(samples: np.ndarray) -> np.ndarray:
    """Cut mono audio into ``(n_windows, 160000)`` float32 windows.

    Audio shorter than one window is zero-padded to exactly one window. For longer
    audio, a final window anchored to the end of the clip is added when the striding
    would otherwise leave a tail uncovered, so the last few seconds are never
    silently dropped.
    """
    if samples.ndim != 1:
        samples = to_mono(samples)
    total = samples.size

    if total <= WINDOW_SAMPLES:
        window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        window[:total] = samples
        return window[np.newaxis, :]

    starts = list(range(0, total - WINDOW_SAMPLES + 1, WINDOW_STRIDE_SAMPLES))
    tail_start = total - WINDOW_SAMPLES
    if starts[-1] != tail_start:
        starts.append(tail_start)

    windows = np.empty((len(starts), WINDOW_SAMPLES), dtype=np.float32)
    for row, start in enumerate(starts):
        windows[row] = samples[start : start + WINDOW_SAMPLES]
    return windows


class PerchModel:
    """A loaded Perch 2.0 classifier plus the label table its outputs index into."""

    def __init__(
        self,
        session: ort.InferenceSession,
        taxonomy: Taxonomy,
        settings: Settings,
        model_path: Path | None = None,
    ) -> None:
        self.session = session
        self.taxonomy = taxonomy
        self.settings = settings
        self.model_path = model_path
        self.num_classes = len(taxonomy)

    # -- Loading -------------------------------------------------------------
    @classmethod
    def load(cls, settings: Settings | None = None) -> PerchModel:
        """Load the ONNX session and the taxonomy, validating that they agree."""
        settings = settings or get_settings()
        model_path = settings.model_path

        if not model_path.exists():
            raise ModelLoadError(
                f"Model file not found at {model_path}. Run "
                "`python scripts/download_model.py` to fetch it (~400 MB), or set "
                "WILDECHO_MODEL_PATH to an existing perch_v2.onnx."
            )

        try:
            taxonomy = load_taxonomy(settings.taxonomy_path, expected_rows=NUM_CLASSES)
        except TaxonomyError as exc:
            raise ModelLoadError(str(exc)) from exc

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if settings.onnx_intra_op_threads > 0:
            options.intra_op_num_threads = settings.onnx_intra_op_threads

        started = time.perf_counter()
        try:
            session = ort.InferenceSession(
                str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # onnxruntime raises bare Exception subclasses
            raise ModelLoadError(f"Could not load ONNX model at {model_path}: {exc}") from exc

        cls._validate_signature(session, model_path)
        logger.info(
            "loaded Perch model from %s in %.2fs (%d classes)",
            model_path,
            time.perf_counter() - started,
            len(taxonomy),
        )
        return cls(session, taxonomy, settings, model_path=model_path)

    @staticmethod
    def _validate_signature(session: ort.InferenceSession, model_path: Path) -> None:
        """Fail loudly at startup if the graph is not the Perch 2.0 export we expect."""
        input_names = [i.name for i in session.get_inputs()]
        if INPUT_NAME not in input_names:
            raise ModelLoadError(
                f"{model_path} does not look like the Perch 2.0 export: expected an input "
                f"named {INPUT_NAME!r}, found {input_names}."
            )
        output_names = [o.name for o in session.get_outputs()]
        if OUTPUT_LOGITS not in output_names:
            raise ModelLoadError(
                f"{model_path} has no {OUTPUT_LOGITS!r} output (found {output_names}). "
                "This build cannot classify species."
            )

        shape = next(i.shape for i in session.get_inputs() if i.name == INPUT_NAME)
        if len(shape) != 2 or (isinstance(shape[1], int) and shape[1] != WINDOW_SAMPLES):
            raise ModelLoadError(
                f"{model_path} expects input shape {shape}, but this service feeds "
                f"[batch, {WINDOW_SAMPLES}] (5s at {TARGET_SAMPLE_RATE} Hz)."
            )

    # -- Inference -----------------------------------------------------------
    def _run_logits(self, windows: np.ndarray) -> np.ndarray:
        """Run windows through the classifier head, in batches, returning raw logits."""
        chunks: list[np.ndarray] = []
        for start in range(0, windows.shape[0], MAX_BATCH_WINDOWS):
            batch = np.ascontiguousarray(
                windows[start : start + MAX_BATCH_WINDOWS], dtype=np.float32
            )
            outputs = self.session.run([OUTPUT_LOGITS], {INPUT_NAME: batch})
            chunks.append(np.asarray(outputs[0], dtype=np.float32))
        logits = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]

        if logits.shape[1] != self.num_classes:
            raise ModelLoadError(
                f"Model emitted {logits.shape[1]} logits but the taxonomy has "
                f"{self.num_classes} rows. Predictions would be misattributed."
            )
        return logits

    def identify(self, audio: np.ndarray, sample_rate: int) -> InferenceResult:
        """Identify species in ``audio``, which may be any sample rate or channel count."""
        samples = resample(audio, sample_rate, TARGET_SAMPLE_RATE)
        validate_samples(samples, TARGET_SAMPLE_RATE, self.settings)

        windows = make_windows(samples)
        started = time.perf_counter()
        logits = self._run_logits(windows)
        inference_ms = (time.perf_counter() - started) * 1000.0

        # Average in logit space, before the nonlinearity. Averaging probabilities
        # instead would let one loud window dominate a long quiet recording.
        mean_logits = logits.mean(axis=0)
        probabilities = _softmax(mean_logits)

        non_animal_top_class = self._detect_non_animal_top(probabilities)
        predictions = self._rank_species(probabilities)
        low_confidence = (
            not predictions or predictions[0].confidence < self.settings.low_confidence_threshold
        )

        return InferenceResult(
            predictions=predictions,
            low_confidence=low_confidence,
            non_animal_top_class=non_animal_top_class,
            windows_processed=int(windows.shape[0]),
            duration_seconds=samples.size / TARGET_SAMPLE_RATE,
            inference_ms=inference_ms,
        )

    def _detect_non_animal_top(self, probabilities: np.ndarray) -> str | None:
        """Report, and log, when the highest-scoring class of all was a sound event.

        Perch's label set includes the 198 FSD50K general sound event classes, so a
        recording of wind or a person talking scores highest on "Wind" or "Speech".
        Those never reach the caller as predictions, but knowing it happened is the
        difference between "the model is wrong" and "the recording had no animal in
        it", so it is both logged and returned.
        """
        top_index = int(np.argmax(probabilities))
        if not self.taxonomy.is_sound_event(top_index):
            return None
        entry = self.taxonomy[top_index]
        logger.info(
            "top class was the non-animal sound event %r (p=%.3f); "
            "returning species candidates only",
            entry.label,
            float(probabilities[top_index]),
        )
        return entry.label

    def _rank_species(self, probabilities: np.ndarray) -> list[Prediction]:
        """Take the top-k species, excluding general sound event classes."""
        species_indices = self.taxonomy.species_indices
        species_probabilities = probabilities[species_indices]

        wanted = min(self.settings.top_k, species_indices.size)
        if wanted <= 0:
            return []

        # argpartition finds the top-k in O(n) rather than sorting all 14,597.
        partitioned = np.argpartition(-species_probabilities, wanted - 1)[:wanted]
        ordered = partitioned[np.argsort(-species_probabilities[partitioned])]

        threshold = self.settings.low_confidence_threshold
        predictions: list[Prediction] = []
        for offset in ordered:
            class_index = int(species_indices[offset])
            entry = self.taxonomy[class_index]
            confidence = float(species_probabilities[offset])
            predictions.append(
                Prediction(
                    common_name=entry.common_name,
                    scientific_name=entry.scientific_name or entry.label,
                    taxonomic_group=entry.group,
                    confidence=round(confidence, 6),
                    low_confidence=confidence < threshold,
                    class_index=class_index,
                )
            )
        return predictions


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------
_model: PerchModel | None = None
_load_error: str | None = None


def load_model(settings: Settings | None = None) -> PerchModel:
    """Load the model into the process-wide slot. Called once from the app lifespan."""
    global _model, _load_error
    try:
        _model = PerchModel.load(settings)
        _load_error = None
    except ModelLoadError as exc:
        _model = None
        _load_error = str(exc)
        raise
    return _model


def set_model(model: PerchModel | None, error: str | None = None) -> None:
    """Install a model directly. Used by tests to inject a mocked session."""
    global _model, _load_error
    _model = model
    _load_error = error


def get_model() -> PerchModel | None:
    return _model


def get_load_error() -> str | None:
    return _load_error


def is_loaded() -> bool:
    return _model is not None


def identify(audio: np.ndarray, sample_rate: int) -> list[Prediction]:
    """Identify species in ``audio``, returning the top candidates best-first.

    This is the library entry point. ``audio`` may be mono or multi-channel and at
    any sample rate; it is downmixed and resampled to 32 kHz internally. The model
    must already be loaded via :func:`load_model`.

    Raises:
        ModelLoadError: if no model has been loaded.
        AudioError: if the clip is empty, too short, too long, or silent.
    """
    model = get_model()
    if model is None:
        raise ModelLoadError(
            _load_error
            or "No model is loaded. Call load_model() before identify(), or check /v1/health."
        )
    return model.identify(audio, sample_rate).predictions
