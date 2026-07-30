"""The committed taxonomy table.

These tests guard the property that matters most: that row N of the CSV describes
output N of the model. A one-row shift would make every prediction confidently and
silently wrong.
"""

from __future__ import annotations

import csv

import pytest
from conftest import TAXONOMY_PATH

from wildecho_api.config import NUM_CLASSES
from wildecho_api.schemas import TaxonomicGroup
from wildecho_api.taxonomy import Taxonomy, TaxonomyError, load_taxonomy


def test_row_count_matches_the_models_output_size(taxonomy: Taxonomy) -> None:
    assert len(taxonomy) == NUM_CLASSES


def test_indices_are_dense_and_in_order(taxonomy: Taxonomy) -> None:
    for position, entry in enumerate(taxonomy.entries):
        assert entry.index == position


def test_species_and_sound_event_counts(taxonomy: Taxonomy) -> None:
    assert taxonomy.count_sound_events() == 198
    assert taxonomy.count_species() == 14_597
    assert taxonomy.count_species() + taxonomy.count_sound_events() == NUM_CLASSES


def test_group_counts_are_plausible(taxonomy: Taxonomy) -> None:
    """Perch is bird-heavy. If this stops being true, the README caveat needs rewriting."""
    birds = taxonomy.count_group(TaxonomicGroup.BIRD)
    assert birds > 10_000
    assert birds / taxonomy.count_species() > 0.6
    assert taxonomy.count_group(TaxonomicGroup.INSECT) > 1_000
    assert taxonomy.count_group(TaxonomicGroup.FROG) > 1_000
    assert taxonomy.count_group(TaxonomicGroup.MAMMAL) > 500


@pytest.mark.parametrize(
    ("label", "common_name", "group"),
    [
        ("Caprimulgus europaeus", "European Nightjar", TaxonomicGroup.BIRD),
        ("Turdus migratorius", "American Robin", TaxonomicGroup.BIRD),
        ("Apis mellifera", "Honey bee", TaxonomicGroup.INSECT),
        ("Rana temporaria", "Common frog", TaxonomicGroup.FROG),
        ("Panthera leo", "Lion", TaxonomicGroup.MAMMAL),
    ],
)
def test_known_species_are_classified_correctly(
    taxonomy: Taxonomy, label: str, common_name: str, group: TaxonomicGroup
) -> None:
    entry = next(e for e in taxonomy.entries if e.label == label)
    assert entry.is_species
    assert entry.scientific_name == label
    assert entry.common_name == common_name
    assert entry.group is group


@pytest.mark.parametrize("label", ["Wind", "Rain", "Speech", "Human_voice", "Engine", "Music"])
def test_known_sound_events_are_flagged(taxonomy: Taxonomy, label: str) -> None:
    entry = next(e for e in taxonomy.entries if e.label == label)
    assert not entry.is_species
    assert taxonomy.is_sound_event(entry.index)


def test_sound_event_labels_never_contain_a_space(taxonomy: Taxonomy) -> None:
    """The rule that separates FSD50K classes from Latin binomials."""
    for entry in taxonomy.entries:
        if entry.is_species:
            assert " " in entry.label, entry.label
        else:
            assert " " not in entry.label, entry.label


def test_species_mask_matches_species_indices(taxonomy: Taxonomy) -> None:
    assert taxonomy.species_mask.sum() == taxonomy.species_indices.size
    assert taxonomy.species_indices.size == taxonomy.count_species()
    assert all(taxonomy[int(i)].is_species for i in taxonomy.species_indices[:100])


def test_every_species_has_a_scientific_name(taxonomy: Taxonomy) -> None:
    assert all(e.scientific_name for e in taxonomy.entries if e.is_species)


def test_common_name_coverage_is_reported_honestly(taxonomy: Taxonomy) -> None:
    """Roughly 79% at build time. The API returns null for the rest, it does not invent one."""
    species = [e for e in taxonomy.entries if e.is_species]
    with_common = sum(1 for e in species if e.common_name)
    assert 0.70 < with_common / len(species) < 0.95


def test_display_name_falls_back_to_scientific_name(taxonomy: Taxonomy) -> None:
    without_common = next(e for e in taxonomy.entries if e.is_species and not e.common_name)
    assert without_common.display_name == without_common.scientific_name


def test_birds_carry_an_ebird_code(taxonomy: Taxonomy) -> None:
    """9,706 of the 10,256 birds come straight from Perch's own eBird mapping."""
    birds = [e for e in taxonomy.entries if e.group is TaxonomicGroup.BIRD]
    with_code = sum(1 for e in birds if e.ebird_code)
    assert with_code == 9_706


def test_sound_events_have_no_species_metadata(taxonomy: Taxonomy) -> None:
    for entry in taxonomy.entries:
        if not entry.is_species:
            assert entry.common_name is None
            assert entry.ebird_code is None


# ---------------------------------------------------------------------------
# Loader error handling
# ---------------------------------------------------------------------------
def test_missing_file_raises(tmp_path) -> None:
    with pytest.raises(TaxonomyError, match="not found"):
        load_taxonomy(tmp_path / "absent.csv")


def test_wrong_row_count_raises(tmp_path) -> None:
    path = tmp_path / "short.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["index", "label", "kind", "scientific_name", "common_name", "group"])
        writer.writerow(
            [0, "Turdus migratorius", "species", "Turdus migratorius", "American Robin", "bird"]
        )
    with pytest.raises(TaxonomyError, match="logits"):
        load_taxonomy(path, expected_rows=NUM_CLASSES)


def test_out_of_order_indices_raise(tmp_path) -> None:
    """The failure mode this check exists for: silently misattributed predictions."""
    path = tmp_path / "shuffled.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["index", "label", "kind", "scientific_name", "common_name", "group"])
        writer.writerow([0, "A a", "species", "A a", "", "bird"])
        writer.writerow([5, "B b", "species", "B b", "", "bird"])
    with pytest.raises(TaxonomyError, match="not in output order"):
        load_taxonomy(path)


def test_missing_columns_raise(tmp_path) -> None:
    path = tmp_path / "thin.csv"
    path.write_text("index,label\n0,Turdus migratorius\n", encoding="utf-8")
    with pytest.raises(TaxonomyError, match="missing columns"):
        load_taxonomy(path)


def test_empty_file_raises(tmp_path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("index,label,kind,scientific_name,common_name,group\n", encoding="utf-8")
    with pytest.raises(TaxonomyError, match="no rows"):
        load_taxonomy(path)


def test_unknown_group_falls_back_to_other(tmp_path) -> None:
    path = tmp_path / "odd.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["index", "label", "kind", "scientific_name", "common_name", "group"])
        writer.writerow([0, "X y", "species", "X y", "", "cephalopod"])
    assert load_taxonomy(path)[0].group is TaxonomicGroup.OTHER


def test_committed_csv_has_the_expected_header() -> None:
    with TAXONOMY_PATH.open(encoding="utf-8") as handle:
        header = handle.readline().strip()
    assert header == "index,label,kind,scientific_name,common_name,group,taxon_class,ebird_code"
