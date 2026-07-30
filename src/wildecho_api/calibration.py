"""Per-taxonomic-group confidence calibration.

Perch is documented (README "Accuracy and limitations") as far stronger on birds
than on the other taxa it covers: birds are 70% of its species classes and
overwhelmingly dominate its training data, while frogs, insects and mammals were
added in the 2.0 release and are more thinly and unevenly sampled. Its raw softmax
confidence does not know this - a mammal prediction at 0.6 is not evidence of the
same reliability as a bird prediction at 0.6.

This module applies a per-group adjustment loaded from ``data/calibration.yaml``
(or a JSON file with the same shape) so an operator can correct for that without a
code change: a ``confidence_scale`` that discounts a group's displayed confidence,
and a ``threshold_offset`` that raises or lowers how much confidence that group
needs before ``low_confidence`` trips. Both default to a no-op for every group
until the file says otherwise, and a missing file is itself a no-op, not an error.

The raw model ranking is never touched here: which candidate is #1 is exactly what
the model computed, and its unadjusted score survives as ``raw_confidence`` on
every prediction. Calibration only changes the displayed ``confidence`` and the
``low_confidence`` flag.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .schemas import TaxonomicGroup


class CalibrationError(RuntimeError):
    """The calibration config file exists but is malformed."""


@dataclass(frozen=True, slots=True)
class GroupCalibration:
    """The adjustment applied to one taxonomic group."""

    confidence_scale: float
    threshold_offset: float


_NOOP_ENTRY = GroupCalibration(confidence_scale=1.0, threshold_offset=0.0)


class Calibration:
    """Loaded per-group adjustments, with a safe no-op fallback for unlisted groups."""

    __slots__ = ("_default", "_groups", "source_path")

    def __init__(
        self,
        groups: dict[TaxonomicGroup, GroupCalibration],
        default: GroupCalibration = _NOOP_ENTRY,
        source_path: Path | None = None,
    ) -> None:
        self._groups = groups
        self._default = default
        self.source_path = source_path

    def for_group(self, group: TaxonomicGroup) -> GroupCalibration:
        """The adjustment for ``group``, or the file's ``default`` if unlisted."""
        return self._groups.get(group, self._default)

    def apply(
        self, raw_confidence: float, group: TaxonomicGroup, base_threshold: float
    ) -> tuple[float, bool]:
        """Return ``(adjusted_confidence, low_confidence)`` for one prediction.

        ``base_threshold`` is ``WILDECHO_LOW_CONFIDENCE_THRESHOLD``; this group's
        ``threshold_offset`` is added to it before comparing against the *adjusted*
        confidence, so a discounted group also needs to clear a higher bar.
        """
        adjustment = self.for_group(group)
        adjusted = max(0.0, min(1.0, raw_confidence * adjustment.confidence_scale))
        effective_threshold = max(0.0, min(1.0, base_threshold + adjustment.threshold_offset))
        return adjusted, adjusted < effective_threshold

    @classmethod
    def noop(cls) -> Calibration:
        """A calibration that changes nothing. Used when no config file is present."""
        return cls(groups={}, default=_NOOP_ENTRY)


def _parse_entry(raw: dict[str, Any], where: str) -> GroupCalibration:
    try:
        scale = float(raw.get("confidence_scale", 1.0))
        offset = float(raw.get("threshold_offset", 0.0))
    except (TypeError, ValueError) as exc:
        raise CalibrationError(
            f"{where}: confidence_scale and threshold_offset must be numbers"
        ) from exc
    if scale < 0:
        raise CalibrationError(f"{where}: confidence_scale must be >= 0, got {scale}")
    return GroupCalibration(confidence_scale=scale, threshold_offset=offset)


def load_calibration(path: Path) -> Calibration:
    """Load a YAML or JSON calibration file, sniffed by extension.

    A missing file is the expected state until an operator decides to tune
    anything, and yields the same no-op behaviour the service had before this
    feature existed - it is deliberately not an error.
    """
    if not path.exists():
        return Calibration.noop()

    try:
        raw_text = path.read_text(encoding="utf-8")
        data = json.loads(raw_text) if path.suffix.lower() == ".json" else yaml.safe_load(raw_text)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise CalibrationError(f"Could not parse calibration file {path}: {exc}") from exc

    if data is None:
        return Calibration.noop()
    if not isinstance(data, dict):
        raise CalibrationError(f"Calibration file {path} must be a mapping at the top level.")

    default = _parse_entry(data.get("default") or {}, f"{path}: default")

    groups: dict[TaxonomicGroup, GroupCalibration] = {}
    for key, entry in (data.get("groups") or {}).items():
        try:
            group = TaxonomicGroup(key)
        except ValueError as exc:
            valid = [g.value for g in TaxonomicGroup]
            raise CalibrationError(
                f"{path}: unknown taxonomic group {key!r} under groups. Valid groups: {valid}"
            ) from exc
        if not isinstance(entry, dict):
            raise CalibrationError(f"{path}: groups.{key} must be a mapping")
        groups[group] = _parse_entry(entry, f"{path}: groups.{key}")

    return Calibration(groups=groups, default=default, source_path=path)
