"""Feedback storage: user-submitted corrections to a prediction.

This is the data that improves the product over time: when a client believes the
model got a clip wrong, it can submit what the correct species actually was via
``POST /v1/feedback``. Storage is entirely local by default and nothing here talks
to a remote service - see README "Feedback" for the privacy posture and how to
point this at Postgres if you outgrow SQLite.

The storage layer is a small abstract interface (:class:`FeedbackStore`) with one
built-in implementation (:class:`SQLiteFeedbackStore`). Swapping in Postgres means
implementing the same two methods against psycopg or SQLAlchemy; the schema below
is deliberately plain SQL with no SQLite-specific syntax beyond ``AUTOINCREMENT``,
so the translation is mechanical.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .schemas import TaxonomicGroup
from .taxonomy import Taxonomy

#: Deliberately plain SQL, no SQLite-only types beyond AUTOINCREMENT, so a Postgres
#: port only needs INTEGER PRIMARY KEY AUTOINCREMENT -> SERIAL PRIMARY KEY.
SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    request_id TEXT,
    clip_id TEXT,
    clip_path TEXT,
    original_scientific_name TEXT,
    original_confidence REAL,
    corrected_text TEXT NOT NULL,
    matched_scientific_name TEXT,
    matched_common_name TEXT,
    matched_class_index INTEGER,
    matched_taxonomic_group TEXT,
    notes TEXT,
    client_ip TEXT
);
"""


class FeedbackDisabledError(RuntimeError):
    """Feedback collection is turned off (``WILDECHO_FEEDBACK_ENABLED=false``)."""


class FeedbackStoreUnavailableError(RuntimeError):
    """The feedback store failed to initialize; see server logs for the cause."""


@dataclass(frozen=True, slots=True)
class MatchedSpecies:
    """A correction's free text, resolved to a known taxonomy entry."""

    scientific_name: str
    common_name: str | None
    class_index: int
    taxonomic_group: TaxonomicGroup


@dataclass(frozen=True, slots=True)
class FeedbackRecord:
    """One correction submission, ready to persist."""

    created_at: str
    request_id: str | None
    clip_id: str | None
    clip_path: str | None
    original_scientific_name: str | None
    original_confidence: float | None
    corrected_text: str
    matched: MatchedSpecies | None
    notes: str | None
    client_ip: str | None


def match_correction(text: str, taxonomy: Taxonomy) -> MatchedSpecies | None:
    """Best-effort case-insensitive match of free text against known species.

    Tries an exact match against scientific_name first, then common_name. Returns
    None rather than guessing when nothing matches - the raw corrected_text is
    always stored either way, so no information is lost when this misses.
    """
    needle = text.strip().casefold()
    if not needle:
        return None
    for entry in taxonomy.entries:
        if entry.is_species and entry.scientific_name.casefold() == needle:
            return MatchedSpecies(
                entry.scientific_name, entry.common_name, entry.index, entry.group
            )
    for entry in taxonomy.entries:
        if entry.is_species and entry.common_name and entry.common_name.casefold() == needle:
            return MatchedSpecies(
                entry.scientific_name, entry.common_name, entry.index, entry.group
            )
    return None


class FeedbackStore(ABC):
    """Storage backend for feedback corrections.

    Both methods are synchronous by design: the API layer calls them via
    ``run_in_threadpool``, the same pattern used for the ffmpeg and ONNX calls in
    ``/v1/identify``, rather than requiring every backend implementation to be
    async. Feedback is a low-volume, human-driven path, so the extra thread hop
    costs nothing that matters.
    """

    @abstractmethod
    def init(self) -> None:
        """Create the schema if it doesn't exist. Called once at startup."""

    @abstractmethod
    def insert(self, record: FeedbackRecord) -> int:
        """Persist one correction, returning its storage-assigned ID."""


class SQLiteFeedbackStore(FeedbackStore):
    """The self-hosted default: a single SQLite file, created on first use.

    Each call opens and closes its own connection rather than holding one open for
    the process lifetime. Feedback is a human submitting a correction, not a hot
    request path, so connection setup cost is irrelevant - and this sidesteps any
    question of sharing a ``sqlite3.Connection`` across the threadpool's worker
    threads, which is not safe without ``check_same_thread`` gymnastics.

    To swap in Postgres: implement this same two-method interface against
    ``psycopg`` or SQLAlchemy, point ``main.py``'s store construction at it, and
    reuse (or port) the schema above. See README "Feedback" for the full note.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def init(self) -> None:
        connection = self._connect()
        try:
            connection.execute(SCHEMA)
            connection.commit()
        finally:
            connection.close()

    def insert(self, record: FeedbackRecord) -> int:
        connection = self._connect()
        try:
            cursor = connection.execute(
                """
                INSERT INTO feedback (
                    created_at, request_id, clip_id, clip_path,
                    original_scientific_name, original_confidence, corrected_text,
                    matched_scientific_name, matched_common_name, matched_class_index,
                    matched_taxonomic_group, notes, client_ip
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.created_at,
                    record.request_id,
                    record.clip_id,
                    record.clip_path,
                    record.original_scientific_name,
                    record.original_confidence,
                    record.corrected_text,
                    record.matched.scientific_name if record.matched else None,
                    record.matched.common_name if record.matched else None,
                    record.matched.class_index if record.matched else None,
                    record.matched.taxonomic_group.value if record.matched else None,
                    record.notes,
                    record.client_ip,
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)  # type: ignore[arg-type]
        finally:
            connection.close()
