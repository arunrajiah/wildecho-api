"""Pydantic v2 request and response models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class TaxonomicGroup(StrEnum):
    """Coarse group a prediction belongs to.

    Deliberately blunt. Perch covers ~14,600 taxa across many classes; collapsing
    them into five buckets is what a consumer app can actually display. The
    precise taxonomic class is in ``data/taxonomy.csv`` if you need it.
    """

    BIRD = "bird"
    FROG = "frog"
    INSECT = "insect"
    MAMMAL = "mammal"
    OTHER = "other"


class Prediction(BaseModel):
    """A single candidate species."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "common_name": "European Nightjar",
                "scientific_name": "Caprimulgus europaeus",
                "taxonomic_group": "bird",
                "confidence": 0.7307,
                "raw_confidence": 0.7307,
                "low_confidence": False,
                "class_index": 2211,
            }
        }
    )

    common_name: str | None = Field(
        default=None,
        description=(
            "English common name, or null when no common name is known for this taxon. "
            "Roughly 80% of species have one; fall back to scientific_name for display."
        ),
    )
    scientific_name: str = Field(description="Latin binomial as published in the Perch label set.")
    taxonomic_group: TaxonomicGroup = Field(description="Coarse group for display and filtering.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Softmax probability, averaged across windows, after this taxon's group "
            "calibration (see data/calibration.yaml) has been applied. This is the number "
            "to show a user. Ranking is unaffected by calibration; only this value and "
            "low_confidence change."
        ),
    )
    raw_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "The uncalibrated softmax probability over all 14,795 classes, before any "
            "per-group adjustment. Equal to confidence when no calibration is configured."
        ),
    )
    low_confidence: bool = Field(
        description=(
            "True when this candidate's calibrated confidence is below the threshold "
            "effective for its taxonomic group (the configured base threshold plus that "
            "group's threshold_offset)."
        )
    )
    class_index: int = Field(
        ge=0, description="Raw Perch output index. Useful for debugging and for joining taxonomy."
    )


class ClipMetadata(BaseModel):
    """What the service actually processed, after decoding and windowing."""

    duration_seconds: float = Field(description="Duration of the decoded clip, in seconds.")
    windows_processed: int = Field(
        ge=1, description="Number of 5-second windows run through the model."
    )
    window_seconds: float = Field(description="Analysis window length, in seconds.")
    window_stride_seconds: float = Field(description="Hop between consecutive windows, in seconds.")
    source_sample_rate: int = Field(description="Sample rate of the uploaded file, in Hz.")
    source_channels: int = Field(description="Channel count of the uploaded file.")
    processed_sample_rate: int = Field(
        description="Sample rate used for inference. Always 32000 for Perch 2.0."
    )
    inference_ms: float = Field(description="Model execution time, excluding decode.")


class IdentifyResponse(BaseModel):
    """Response body for ``POST /v1/identify``."""

    predictions: list[Prediction] = Field(
        description=(
            "Top candidates, ordered by the model's own raw_confidence (highest first). "
            "Non-animal classes are excluded. Because confidence is separately discounted "
            "per taxonomic group (see data/calibration.yaml), it is usually but not always "
            "monotonic across this list: order reflects the model's actual belief and is "
            "never reshuffled by calibration."
        )
    )
    low_confidence: bool = Field(
        description=(
            "True when the top prediction's confidence is below the configured threshold. "
            "Treat the whole result as a weak guess and consider prompting for a better clip."
        )
    )
    non_animal_top_class: str | None = Field(
        default=None,
        description=(
            "Set when the single highest-scoring class was a general sound event (wind, rain, "
            "speech, an engine) rather than an animal. The species list is then almost "
            "certainly noise. Useful for telling a user their recording was unusable."
        ),
    )
    model_version: str = Field(description="Identifier of the model that produced this result.")
    metadata: ClipMetadata
    request_id: str = Field(
        description=(
            "Correlates this response with the server's logs. Also returned as the "
            "X-Request-ID response header. Hand this back in POST /v1/feedback's "
            "request_id field when submitting a correction for this result."
        )
    )


class ModelStatus(StrEnum):
    LOADED = "loaded"
    NOT_LOADED = "not_loaded"
    ERROR = "error"


class HealthResponse(BaseModel):
    """Response body for ``GET /v1/health``."""

    status: str = Field(description="'ok' when the model is loaded and ready, else 'degraded'.")
    model_status: ModelStatus
    model_path: str = Field(description="Where the service looked for the ONNX file.")
    model_loaded: bool
    detail: str | None = Field(
        default=None, description="Why the model is unavailable, when it is not loaded."
    )
    num_classes: int | None = Field(default=None, description="Classes in the loaded label set.")
    taxonomy_loaded: bool
    feedback_enabled: bool = Field(description="Whether POST /v1/feedback is enabled by config.")
    feedback_store_ready: bool = Field(
        description="Whether the feedback store initialized successfully, when enabled."
    )
    version: str = Field(description="wildecho-api version.")


class Attribution(BaseModel):
    """Credit and licensing for the model and data this service depends on."""

    model_name: str
    model_authors: str
    model_license: str
    citation: str
    links: dict[str, str]


class TaxaCoverage(BaseModel):
    """Honest accounting of what the classifier can and cannot recognise."""

    total_classes: int
    species_classes: int
    general_sound_event_classes: int
    birds: int
    non_bird_species: int
    disclaimer: str


class AboutResponse(BaseModel):
    """Response body for ``GET /v1/about``."""

    name: str
    version: str
    description: str
    model_version: str
    license: str = Field(description="License of this wrapper service, not of the model.")
    coverage: TaxaCoverage
    attribution: Attribution
    limits: dict[str, float | int | str]


class FeedbackResponse(BaseModel):
    """Response body for ``POST /v1/feedback``."""

    id: int = Field(description="Storage-assigned ID for this correction.")
    received_at: str = Field(description="UTC timestamp this correction was recorded, ISO 8601.")
    matched_scientific_name: str | None = Field(
        default=None,
        description=(
            "Set when corrected_text matched a known species in the taxonomy by exact "
            "scientific or common name (case-insensitive). Null means the free text was "
            "stored as-is with no automatic match; it can still be reviewed manually."
        ),
    )
    matched_common_name: str | None = Field(default=None)
    matched_taxonomic_group: TaxonomicGroup | None = Field(default=None)
    stored_audio: bool = Field(
        description="Whether an uploaded clip was saved to feedback_clips_dir for this submission."
    )


class ErrorResponse(BaseModel):
    """Uniform error body. Every 4xx and 5xx from this service uses this shape."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "audio_too_short",
                "detail": "Clip is 0.21s; the minimum is 0.5s. Record a longer sample.",
            }
        }
    )

    error: str = Field(description="Stable machine-readable error code.")
    detail: str = Field(description="Human-readable explanation, safe to show to an end user.")
