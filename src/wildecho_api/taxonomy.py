"""Loads ``data/taxonomy.csv``: the index -> species mapping for Perch's 14,795 logits.

Perch's own label file gives a raw label per index and, for birds, an eBird code.
It does not give English common names or a taxonomic group. ``scripts/build_taxonomy.py``
joins those in from Wikidata (CC0) and the GBIF backbone (CC-BY) and writes the CSV
this module reads. See NOTICE for attribution.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .schemas import TaxonomicGroup

#: Labels for the 198 FSD50K general sound event classes are marked this way in
#: the CSV. Everything else is a species.
KIND_SOUND_EVENT = "sound_event"
KIND_SPECIES = "species"


class TaxonomyError(RuntimeError):
    """Raised when the taxonomy file is missing, malformed, or the wrong length."""


@dataclass(frozen=True, slots=True)
class TaxonEntry:
    """One row of the taxonomy table, i.e. one output index of the classifier."""

    index: int
    label: str
    is_species: bool
    scientific_name: str
    common_name: str | None
    group: TaxonomicGroup
    taxon_class: str
    ebird_code: str | None

    @property
    def display_name(self) -> str:
        """Best available human-readable name."""
        return self.common_name or self.scientific_name or self.label


class Taxonomy:
    """Immutable lookup table from classifier output index to taxon metadata."""

    __slots__ = ("_entries", "_species_index_array", "_species_mask")

    def __init__(self, entries: list[TaxonEntry]) -> None:
        self._entries = entries
        # Precompute the species mask once. Filtering the 198 general sound event
        # classes out of every request is otherwise the hot path's only Python loop.
        self._species_mask = np.array([e.is_species for e in entries], dtype=bool)
        self._species_index_array = np.flatnonzero(self._species_mask)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> TaxonEntry:
        return self._entries[index]

    @property
    def entries(self) -> list[TaxonEntry]:
        return self._entries

    @property
    def species_mask(self) -> np.ndarray:
        """Boolean array, True where the class is a species rather than a sound event."""
        return self._species_mask

    @property
    def species_indices(self) -> np.ndarray:
        """Integer array of the output indices that correspond to species."""
        return self._species_index_array

    def is_sound_event(self, index: int) -> bool:
        return not self._entries[index].is_species

    # -- Coverage summary, surfaced by /v1/about -----------------------------
    def count_species(self) -> int:
        return int(self._species_mask.sum())

    def count_sound_events(self) -> int:
        return len(self._entries) - self.count_species()

    def count_group(self, group: TaxonomicGroup) -> int:
        return sum(1 for e in self._entries if e.is_species and e.group is group)


def _parse_group(raw: str) -> TaxonomicGroup:
    try:
        return TaxonomicGroup(raw)
    except ValueError:
        return TaxonomicGroup.OTHER


def load_taxonomy(path: Path, expected_rows: int | None = None) -> Taxonomy:
    """Read the taxonomy CSV, validating that it lines up with the model's output size.

    A silently misaligned taxonomy is the worst possible failure here: every
    prediction would be confidently wrong by a constant offset. So the row count,
    the ``index`` column ordering, and the header are all checked.
    """
    if not path.exists():
        raise TaxonomyError(
            f"Taxonomy file not found at {path}. It ships in data/taxonomy.csv; "
            "regenerate it with `python scripts/build_taxonomy.py` if it is missing."
        )

    entries: list[TaxonEntry] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"index", "label", "kind", "scientific_name", "common_name", "group"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise TaxonomyError(f"Taxonomy file {path} is missing columns: {sorted(missing)}")

        for row_number, row in enumerate(reader):
            try:
                index = int(row["index"])
            except (TypeError, ValueError) as exc:
                raise TaxonomyError(
                    f"Taxonomy file {path} row {row_number}: non-integer index {row['index']!r}"
                ) from exc
            if index != row_number:
                raise TaxonomyError(
                    f"Taxonomy file {path} is not in output order: row {row_number} "
                    f"declares index {index}. Predictions would be misattributed."
                )
            common = (row.get("common_name") or "").strip()
            ebird = (row.get("ebird_code") or "").strip()
            entries.append(
                TaxonEntry(
                    index=index,
                    label=row["label"],
                    is_species=row["kind"] == KIND_SPECIES,
                    scientific_name=(row.get("scientific_name") or "").strip(),
                    common_name=common or None,
                    group=_parse_group((row.get("group") or "").strip()),
                    taxon_class=(row.get("taxon_class") or "").strip(),
                    ebird_code=ebird or None,
                )
            )

    if not entries:
        raise TaxonomyError(f"Taxonomy file {path} contains no rows.")
    if expected_rows is not None and len(entries) != expected_rows:
        raise TaxonomyError(
            f"Taxonomy file {path} has {len(entries)} rows but the model emits "
            f"{expected_rows} logits. These must match exactly."
        )
    return Taxonomy(entries)
