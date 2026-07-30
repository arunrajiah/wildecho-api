"""Windowing, scoring, ranking and sound-event filtering.

Everything here except the ``model``-marked tests runs against a fake ONNX session,
so CI exercises the full pipeline without the 390 MiB download.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FakeSession, logits_favouring, noise, tone

from wildecho_api import inference
from wildecho_api.audio import AudioSilentError, AudioTooShortError
from wildecho_api.config import NUM_CLASSES, TARGET_SAMPLE_RATE, WINDOW_SAMPLES, Settings
from wildecho_api.inference import (
    MAX_BATCH_WINDOWS,
    WINDOW_STRIDE_SAMPLES,
    ModelLoadError,
    PerchModel,
    make_windows,
)
from wildecho_api.schemas import TaxonomicGroup
from wildecho_api.taxonomy import Taxonomy


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def test_short_audio_becomes_one_padded_window() -> None:
    windows = make_windows(tone(2.0))
    assert windows.shape == (1, WINDOW_SAMPLES)
    # The tail past 2s must be silence, not garbage.
    assert float(np.abs(windows[0, 2 * TARGET_SAMPLE_RATE :]).max()) == 0.0


def test_exactly_one_window_is_not_padded() -> None:
    windows = make_windows(tone(5.0))
    assert windows.shape == (1, WINDOW_SAMPLES)
    assert float(np.abs(windows[0, -100:]).max()) > 0.0


@pytest.mark.parametrize(
    ("seconds", "expected_windows"),
    [
        (5.0, 1),
        (7.5, 2),  # starts at 0 and 2.5
        (10.0, 3),  # 0, 2.5, 5.0
        (15.0, 5),  # 0, 2.5, 5, 7.5, 10
        (20.0, 7),
    ],
)
def test_window_count_for_stride(seconds: float, expected_windows: int) -> None:
    assert make_windows(tone(seconds)).shape[0] == expected_windows


def test_stride_is_half_a_window() -> None:
    assert WINDOW_STRIDE_SAMPLES == WINDOW_SAMPLES // 2
    assert int(2.5 * TARGET_SAMPLE_RATE) == WINDOW_STRIDE_SAMPLES


def test_consecutive_windows_overlap_by_half() -> None:
    samples = tone(10.0)
    windows = make_windows(samples)
    np.testing.assert_array_equal(
        windows[0, WINDOW_STRIDE_SAMPLES:], windows[1, :WINDOW_STRIDE_SAMPLES]
    )


def test_ragged_tail_is_covered_by_an_end_anchored_window() -> None:
    """A 9-second clip must not silently drop its last 1.5 seconds."""
    samples = tone(9.0)
    windows = make_windows(samples)
    # Strided starts alone reach 0, 2.5; an end-anchored window at 4.0s is added.
    assert windows.shape[0] == 3
    np.testing.assert_array_equal(windows[-1], samples[-WINDOW_SAMPLES:])


def test_windows_are_float32_and_contiguous() -> None:
    windows = make_windows(tone(12.0))
    assert windows.dtype == np.float32


def test_make_windows_downmixes_multichannel() -> None:
    stereo = np.stack([tone(8.0), tone(8.0)], axis=1)
    assert make_windows(stereo).shape == (3, WINDOW_SAMPLES)


# ---------------------------------------------------------------------------
# Scoring and ranking
# ---------------------------------------------------------------------------
def test_identify_returns_top_k_species(
    make_model, nightjar_index: int, settings: Settings
) -> None:
    model = make_model(logits_favouring(nightjar_index))
    result = model.identify(tone(6.0), TARGET_SAMPLE_RATE)
    assert len(result.predictions) == settings.top_k


def test_identify_ranks_the_expected_species_first(
    make_model, nightjar_index: int, taxonomy: Taxonomy
) -> None:
    model = make_model(logits_favouring(nightjar_index))
    result = model.identify(tone(6.0), TARGET_SAMPLE_RATE)

    top = result.predictions[0]
    assert top.class_index == nightjar_index
    assert top.scientific_name == "Caprimulgus europaeus"
    assert top.common_name == "European Nightjar"
    assert top.taxonomic_group is TaxonomicGroup.BIRD


def test_raw_confidences_are_descending_and_in_range(make_model, nightjar_index: int) -> None:
    """Order is decided by raw_confidence; calibration must never reshuffle it."""
    model = make_model(logits_favouring(nightjar_index))
    predictions = model.identify(tone(6.0), TARGET_SAMPLE_RATE).predictions
    raw = [p.raw_confidence for p in predictions]
    assert raw == sorted(raw, reverse=True)
    assert all(0.0 <= p.confidence <= 1.0 for p in predictions)
    assert all(0.0 <= p.raw_confidence <= 1.0 for p in predictions)


def test_calibration_noop_leaves_confidence_equal_to_raw(make_model, nightjar_index: int) -> None:
    """With no calibration file loaded, confidence and raw_confidence must be identical."""
    model = make_model(logits_favouring(nightjar_index))
    for prediction in model.identify(tone(6.0), TARGET_SAMPLE_RATE).predictions:
        assert prediction.confidence == prediction.raw_confidence


def test_high_logit_produces_high_confidence(make_model, nightjar_index: int) -> None:
    model = make_model(logits_favouring(nightjar_index, peak=15.0, floor=-5.0))
    result = model.identify(tone(6.0), TARGET_SAMPLE_RATE)
    assert result.predictions[0].confidence > 0.9
    assert result.low_confidence is False
    assert result.predictions[0].low_confidence is False


def test_flat_logits_produce_low_confidence(make_model) -> None:
    """With every class equally likely, confidence is ~1/14795 and the flag must trip."""
    model = make_model(np.zeros(NUM_CLASSES, dtype=np.float32))
    result = model.identify(tone(6.0), TARGET_SAMPLE_RATE)
    assert result.low_confidence is True
    assert all(p.low_confidence for p in result.predictions)
    assert result.predictions[0].confidence < 0.01


def test_low_confidence_threshold_is_configurable(
    taxonomy: Taxonomy, nightjar_index: int, settings: Settings
) -> None:
    strict = settings.model_copy(update={"low_confidence_threshold": 0.999})
    model = PerchModel(
        session=FakeSession(logits_favouring(nightjar_index, peak=8.0, floor=-2.0)),  # type: ignore[arg-type]
        taxonomy=taxonomy,
        settings=strict,
    )
    result = model.identify(tone(6.0), TARGET_SAMPLE_RATE)
    assert result.low_confidence is True


# ---------------------------------------------------------------------------
# General sound event handling
# ---------------------------------------------------------------------------
def test_sound_event_classes_are_never_returned(
    make_model, wind_index: int, taxonomy: Taxonomy
) -> None:
    model = make_model(logits_favouring(wind_index, peak=20.0))
    result = model.identify(noise(6.0), TARGET_SAMPLE_RATE)

    returned = {p.class_index for p in result.predictions}
    assert wind_index not in returned
    assert all(taxonomy[index].is_species for index in returned)


def test_non_animal_top_class_is_reported(make_model, wind_index: int) -> None:
    model = make_model(logits_favouring(wind_index, peak=20.0))
    result = model.identify(noise(6.0), TARGET_SAMPLE_RATE)
    assert result.non_animal_top_class == "Wind"


def test_non_animal_top_class_is_logged(make_model, wind_index: int, caplog) -> None:
    model = make_model(logits_favouring(wind_index, peak=20.0))
    with caplog.at_level("INFO", logger="wildecho_api.inference"):
        model.identify(noise(6.0), TARGET_SAMPLE_RATE)
    assert any("Wind" in record.getMessage() for record in caplog.records)


def test_non_animal_top_class_is_none_for_a_species(make_model, nightjar_index: int) -> None:
    model = make_model(logits_favouring(nightjar_index))
    assert model.identify(tone(6.0), TARGET_SAMPLE_RATE).non_animal_top_class is None


def test_species_ranking_still_populated_when_top_is_a_sound_event(
    make_model, wind_index: int
) -> None:
    """The caller still gets candidates; `non_animal_top_class` is the signal to distrust them."""
    model = make_model(logits_favouring(wind_index, peak=20.0))
    result = model.identify(noise(6.0), TARGET_SAMPLE_RATE)
    assert len(result.predictions) > 0
    assert result.low_confidence is True


# ---------------------------------------------------------------------------
# Metadata and batching
# ---------------------------------------------------------------------------
def test_metadata_reports_windows_and_duration(make_model, nightjar_index: int) -> None:
    model = make_model(logits_favouring(nightjar_index))
    result = model.identify(tone(10.0), TARGET_SAMPLE_RATE)
    assert result.windows_processed == 3
    assert 9.9 < result.duration_seconds < 10.1
    assert result.inference_ms >= 0.0


def test_long_clip_is_batched(make_model, nightjar_index: int) -> None:
    model = make_model(logits_favouring(nightjar_index))
    session: FakeSession = model.session  # type: ignore[assignment]
    result = model.identify(tone(120.0), TARGET_SAMPLE_RATE)

    assert result.windows_processed == 47
    assert len(session.calls) == 3  # 47 windows at 16 per call
    assert all(batch <= MAX_BATCH_WINDOWS for batch, _ in session.calls)
    assert all(samples == WINDOW_SAMPLES for _, samples in session.calls)


def test_logits_are_averaged_across_windows(
    taxonomy: Taxonomy, settings: Settings, nightjar_index: int
) -> None:
    """Averaging happens in logit space, before softmax."""

    class PerWindowSession(FakeSession):
        def run(self, output_names, feeds):  # type: ignore[no-untyped-def]
            batch = feeds["inputs"]
            self.calls.append((batch.shape[0], batch.shape[1]))
            out = np.zeros((batch.shape[0], NUM_CLASSES), dtype=np.float32)
            # First window votes for the nightjar, the rest are indifferent.
            out[0, nightjar_index] = 30.0
            return [out]

    model = PerchModel(
        session=PerWindowSession(np.zeros(NUM_CLASSES, dtype=np.float32)),  # type: ignore[arg-type]
        taxonomy=taxonomy,
        settings=settings,
    )
    result = model.identify(tone(10.0), TARGET_SAMPLE_RATE)
    # 30.0 seen in one of 3 windows averages to a logit of 10.0. Softmax against
    # 14,794 classes still at 0.0 gives e^10 / (e^10 + 14794) = 0.598, so the win
    # survives dilution. Averaging probabilities instead would have given ~0.33.
    assert result.predictions[0].class_index == nightjar_index
    assert 0.55 < result.predictions[0].confidence < 0.65


def test_resampling_happens_before_inference(make_model, nightjar_index: int) -> None:
    model = make_model(logits_favouring(nightjar_index))
    session: FakeSession = model.session  # type: ignore[assignment]
    model.identify(tone(6.0, sample_rate=44_100), 44_100)
    assert all(samples == WINDOW_SAMPLES for _, samples in session.calls)


# ---------------------------------------------------------------------------
# Bad input
# ---------------------------------------------------------------------------
def test_identify_rejects_silence(make_model, nightjar_index: int) -> None:
    model = make_model(logits_favouring(nightjar_index))
    with pytest.raises(AudioSilentError):
        model.identify(np.zeros(TARGET_SAMPLE_RATE * 3, dtype=np.float32), TARGET_SAMPLE_RATE)


def test_identify_rejects_too_short(make_model, nightjar_index: int) -> None:
    model = make_model(logits_favouring(nightjar_index))
    with pytest.raises(AudioTooShortError):
        model.identify(tone(0.1), TARGET_SAMPLE_RATE)


def test_module_identify_requires_a_loaded_model() -> None:
    inference.set_model(None)
    with pytest.raises(ModelLoadError):
        inference.identify(tone(6.0), TARGET_SAMPLE_RATE)


def test_module_identify_uses_the_loaded_model(mocked_model, nightjar_index: int) -> None:
    predictions = inference.identify(tone(6.0), TARGET_SAMPLE_RATE)
    assert predictions[0].class_index == nightjar_index


def test_signature_validation_rejects_a_wrong_graph(taxonomy: Taxonomy) -> None:
    class WrongSession(FakeSession):
        def get_outputs(self):  # type: ignore[no-untyped-def]
            return [type("IO", (), {"name": "embedding", "shape": ["batch", 1536]})()]

    with pytest.raises(ModelLoadError, match="no 'label' output"):
        PerchModel._validate_signature(
            WrongSession(np.zeros(NUM_CLASSES, dtype=np.float32)),  # type: ignore[arg-type]
            __import__("pathlib").Path("fake.onnx"),
        )


def test_taxonomy_length_mismatch_is_fatal(
    taxonomy: Taxonomy, settings: Settings, nightjar_index: int
) -> None:
    """A misaligned taxonomy would make every prediction confidently wrong."""
    model = PerchModel(
        session=FakeSession(np.zeros(NUM_CLASSES - 5, dtype=np.float32)),  # type: ignore[arg-type]
        taxonomy=taxonomy,
        settings=settings,
    )
    with pytest.raises(ModelLoadError, match="misattributed"):
        model.identify(tone(6.0), TARGET_SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Real weights. Skipped unless models/perch_v2.onnx exists.
# ---------------------------------------------------------------------------
@pytest.mark.model
def test_real_model_identifies_the_nightjar_fixture(settings: Settings) -> None:
    """The end-to-end regression test: real weights, real recording, known answer."""
    from conftest import NIGHTJAR_CLIP

    from wildecho_api.audio import decode_file

    model = PerchModel.load(settings)
    decoded = decode_file(NIGHTJAR_CLIP, settings)
    result = model.identify(decoded.samples, decoded.sample_rate)

    top = result.predictions[0]
    assert top.scientific_name == "Caprimulgus europaeus"
    assert top.common_name == "European Nightjar"
    assert top.taxonomic_group is TaxonomicGroup.BIRD
    # Measured at 0.909 on this clip; the margin allows for onnxruntime version drift.
    assert top.confidence > 0.6
    assert result.low_confidence is False
    assert result.non_animal_top_class is None
    assert result.windows_processed == 3


@pytest.mark.model
def test_real_model_flags_white_noise_as_low_confidence(settings: Settings) -> None:
    """Noise scores ~0.05, comfortably under the 0.3 threshold."""
    model = PerchModel.load(settings)
    result = model.identify(noise(10.0, seed=7), TARGET_SAMPLE_RATE)
    assert result.low_confidence is True
    assert result.predictions[0].confidence < 0.3


@pytest.mark.model
def test_real_model_reports_a_pure_tone_as_non_animal(settings: Settings) -> None:
    """A 1 kHz sine reads as an alarm or a telephone, not an animal."""
    model = PerchModel.load(settings)
    result = model.identify(tone(6.0, frequency=1_000.0), TARGET_SAMPLE_RATE)
    assert result.non_animal_top_class is not None
    assert result.low_confidence is True
