"""Per-taxonomic-group confidence calibration."""

from __future__ import annotations

import pytest
from conftest import logits_favouring, tone

from wildecho_api.calibration import (
    Calibration,
    CalibrationError,
    GroupCalibration,
    load_calibration,
)
from wildecho_api.config import TARGET_SAMPLE_RATE, Settings, get_settings
from wildecho_api.inference import PerchModel
from wildecho_api.schemas import TaxonomicGroup
from wildecho_api.taxonomy import Taxonomy


# ---------------------------------------------------------------------------
# Calibration.apply / for_group
# ---------------------------------------------------------------------------
def test_noop_calibration_leaves_confidence_and_threshold_unchanged() -> None:
    calibration = Calibration.noop()
    adjusted, low = calibration.apply(0.42, TaxonomicGroup.MAMMAL, base_threshold=0.3)
    assert adjusted == 0.42
    assert low is False


def test_apply_scales_confidence_down() -> None:
    calibration = Calibration(
        groups={TaxonomicGroup.MAMMAL: GroupCalibration(confidence_scale=0.5, threshold_offset=0.0)}
    )
    adjusted, _ = calibration.apply(0.8, TaxonomicGroup.MAMMAL, base_threshold=0.3)
    assert adjusted == pytest.approx(0.4)


def test_apply_threshold_offset_raises_the_bar() -> None:
    calibration = Calibration(
        groups={TaxonomicGroup.INSECT: GroupCalibration(confidence_scale=1.0, threshold_offset=0.2)}
    )
    # 0.45 clears the base 0.3 threshold but not 0.3 + 0.2 = 0.5.
    adjusted, low = calibration.apply(0.45, TaxonomicGroup.INSECT, base_threshold=0.3)
    assert adjusted == 0.45
    assert low is True


def test_apply_clamps_confidence_to_unit_range() -> None:
    over_one = Calibration(
        groups={TaxonomicGroup.BIRD: GroupCalibration(confidence_scale=2.0, threshold_offset=0.0)}
    )
    adjusted, _ = over_one.apply(0.9, TaxonomicGroup.BIRD, base_threshold=0.3)
    assert adjusted == 1.0


def test_apply_clamps_threshold_to_unit_range() -> None:
    calibration = Calibration(
        groups={TaxonomicGroup.OTHER: GroupCalibration(confidence_scale=1.0, threshold_offset=5.0)}
    )
    # Effective threshold would be 5.3, clamped to 1.0; nothing can be >= 1.0 and
    # not low_confidence except a perfect score, so 0.99 still trips it.
    adjusted, low = calibration.apply(0.99, TaxonomicGroup.OTHER, base_threshold=0.3)
    assert adjusted == 0.99
    assert low is True


def test_for_group_falls_back_to_default_for_unlisted_group() -> None:
    default = GroupCalibration(confidence_scale=0.7, threshold_offset=0.1)
    calibration = Calibration(groups={}, default=default)
    assert calibration.for_group(TaxonomicGroup.FROG) == default
    assert calibration.for_group(TaxonomicGroup.BIRD) == default


def test_for_group_prefers_listed_group_over_default() -> None:
    listed = GroupCalibration(confidence_scale=0.9, threshold_offset=0.0)
    calibration = Calibration(
        groups={TaxonomicGroup.MAMMAL: listed},
        default=GroupCalibration(confidence_scale=0.5, threshold_offset=0.0),
    )
    assert calibration.for_group(TaxonomicGroup.MAMMAL) == listed


# ---------------------------------------------------------------------------
# load_calibration
# ---------------------------------------------------------------------------
def test_missing_file_is_a_noop(tmp_path) -> None:
    calibration = load_calibration(tmp_path / "absent.yaml")
    adjusted, low = calibration.apply(0.5, TaxonomicGroup.MAMMAL, base_threshold=0.3)
    assert adjusted == 0.5
    assert low is False
    assert calibration.source_path is None


def test_loads_yaml(tmp_path) -> None:
    path = tmp_path / "calibration.yaml"
    path.write_text(
        "default:\n"
        "  confidence_scale: 1.0\n"
        "  threshold_offset: 0.0\n"
        "groups:\n"
        "  mammal:\n"
        "    confidence_scale: 0.8\n"
        "    threshold_offset: 0.05\n",
        encoding="utf-8",
    )
    calibration = load_calibration(path)
    adjustment = calibration.for_group(TaxonomicGroup.MAMMAL)
    assert adjustment.confidence_scale == 0.8
    assert adjustment.threshold_offset == 0.05
    assert calibration.source_path == path


def test_loads_json(tmp_path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        '{"default": {"confidence_scale": 1.0, "threshold_offset": 0.0}, '
        '"groups": {"insect": {"confidence_scale": 0.7, "threshold_offset": 0.1}}}',
        encoding="utf-8",
    )
    calibration = load_calibration(path)
    adjustment = calibration.for_group(TaxonomicGroup.INSECT)
    assert adjustment.confidence_scale == 0.7
    assert adjustment.threshold_offset == 0.1


def test_empty_file_is_a_noop(tmp_path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    calibration = load_calibration(path)
    adjusted, _ = calibration.apply(0.5, TaxonomicGroup.BIRD, base_threshold=0.3)
    assert adjusted == 0.5


def test_missing_default_section_falls_back_to_noop_default(tmp_path) -> None:
    path = tmp_path / "calibration.yaml"
    path.write_text("groups:\n  mammal:\n    confidence_scale: 0.5\n", encoding="utf-8")
    calibration = load_calibration(path)
    assert calibration.for_group(TaxonomicGroup.BIRD) == GroupCalibration(1.0, 0.0)
    assert calibration.for_group(TaxonomicGroup.MAMMAL).confidence_scale == 0.5


def test_unknown_group_raises(tmp_path) -> None:
    path = tmp_path / "calibration.yaml"
    path.write_text("groups:\n  reptile:\n    confidence_scale: 0.5\n", encoding="utf-8")
    with pytest.raises(CalibrationError, match="unknown taxonomic group"):
        load_calibration(path)


def test_non_mapping_top_level_raises(tmp_path) -> None:
    path = tmp_path / "calibration.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(CalibrationError, match="mapping at the top level"):
        load_calibration(path)


def test_non_mapping_group_entry_raises(tmp_path) -> None:
    path = tmp_path / "calibration.yaml"
    path.write_text("groups:\n  mammal: not_a_mapping\n", encoding="utf-8")
    with pytest.raises(CalibrationError, match="must be a mapping"):
        load_calibration(path)


def test_non_numeric_scale_raises(tmp_path) -> None:
    path = tmp_path / "calibration.yaml"
    path.write_text("groups:\n  mammal:\n    confidence_scale: not_a_number\n", encoding="utf-8")
    with pytest.raises(CalibrationError, match="must be numbers"):
        load_calibration(path)


def test_negative_scale_raises(tmp_path) -> None:
    path = tmp_path / "calibration.yaml"
    path.write_text("groups:\n  mammal:\n    confidence_scale: -0.1\n", encoding="utf-8")
    with pytest.raises(CalibrationError, match=">= 0"):
        load_calibration(path)


def test_malformed_yaml_raises_calibration_error(tmp_path) -> None:
    path = tmp_path / "calibration.yaml"
    path.write_text("groups: [unclosed\n", encoding="utf-8")
    with pytest.raises(CalibrationError, match="Could not parse"):
        load_calibration(path)


# ---------------------------------------------------------------------------
# The committed data/calibration.yaml
# ---------------------------------------------------------------------------
def test_committed_calibration_file_loads() -> None:
    calibration = load_calibration(get_settings().calibration_path)
    bird = calibration.for_group(TaxonomicGroup.BIRD)
    mammal = calibration.for_group(TaxonomicGroup.MAMMAL)
    assert bird.confidence_scale == 1.0
    assert bird.threshold_offset == 0.0
    assert mammal.confidence_scale < 1.0
    assert mammal.threshold_offset > 0.0


# ---------------------------------------------------------------------------
# Integration: calibration wired through PerchModel
# ---------------------------------------------------------------------------
def test_perch_model_applies_calibration_to_confidence_and_flag(
    taxonomy: Taxonomy, settings: Settings, nightjar_index: int
) -> None:
    """Confirms the full path: PerchModel._rank_species -> Calibration.apply -> Prediction."""
    from conftest import FakeSession

    calibration = Calibration(
        groups={TaxonomicGroup.BIRD: GroupCalibration(confidence_scale=0.5, threshold_offset=0.2)}
    )
    model = PerchModel(
        session=FakeSession(logits_favouring(nightjar_index, peak=10.0, floor=-3.0)),  # type: ignore[arg-type]
        taxonomy=taxonomy,
        settings=settings,
        calibration=calibration,
    )
    result = model.identify(tone(6.0), TARGET_SAMPLE_RATE)
    top = result.predictions[0]

    assert top.class_index == nightjar_index
    # Both values are independently rounded to 6dp in inference.py, so the exact
    # relationship only holds before rounding - allow for that quantization.
    assert top.confidence == pytest.approx(top.raw_confidence * 0.5, abs=1e-6)
    assert top.raw_confidence > top.confidence  # calibration discounted it
    # Effective threshold for bird is settings.low_confidence_threshold + 0.2; the
    # calibrated confidence must be compared against that, not the base threshold.
    effective_threshold = settings.low_confidence_threshold + 0.2
    assert top.low_confidence == (top.confidence < effective_threshold)


def test_perch_model_calibration_never_reorders_predictions(
    taxonomy: Taxonomy, settings: Settings
) -> None:
    """Even a calibration that inverts relative confidence must not reorder the list."""
    import numpy as np
    from conftest import FakeSession

    from wildecho_api.config import NUM_CLASSES

    bird_index = next(e.index for e in taxonomy.entries if e.label == "Turdus migratorius")
    mammal_index = next(e.index for e in taxonomy.entries if e.label == "Panthera leo")

    logits = np.full(NUM_CLASSES, -5.0, dtype=np.float32)
    logits[mammal_index] = 10.0  # ranked #1 by the raw model
    logits[bird_index] = 9.5  # ranked #2 by the raw model

    # Discount mammals heavily enough that, by displayed confidence alone, the
    # bird would look more trustworthy than the mammal - ranking must not follow.
    calibration = Calibration(
        groups={TaxonomicGroup.MAMMAL: GroupCalibration(confidence_scale=0.1, threshold_offset=0.0)}
    )
    model = PerchModel(
        session=FakeSession(logits),  # type: ignore[arg-type]
        taxonomy=taxonomy,
        settings=settings,
        calibration=calibration,
    )
    result = model.identify(tone(6.0), TARGET_SAMPLE_RATE)

    assert result.predictions[0].class_index == mammal_index
    assert result.predictions[1].class_index == bird_index
    # The inversion actually happened in the *displayed* numbers, proving this
    # test would catch a reordering bug rather than passing vacuously.
    assert result.predictions[0].confidence < result.predictions[1].confidence
