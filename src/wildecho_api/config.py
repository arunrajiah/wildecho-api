"""Runtime configuration, all overridable with ``WILDECHO_*`` environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Perch 2.0 is trained on fixed 5-second windows of 32 kHz mono audio. These are
#: properties of the exported graph (input shape ``[batch, 160000]``), not
#: preferences, so they are module constants rather than settings.
TARGET_SAMPLE_RATE = 32_000
WINDOW_SECONDS = 5.0
WINDOW_SAMPLES = int(TARGET_SAMPLE_RATE * WINDOW_SECONDS)  # 160_000

#: Number of logits the classifier head emits, and therefore the number of rows
#: data/taxonomy.csv must contain. Checked at startup.
NUM_CLASSES = 14_795


def _package_root() -> Path:
    return Path(__file__).resolve().parent


def _repo_root() -> Path:
    return _package_root().parent.parent


def default_model_path() -> Path:
    return _repo_root() / "models" / "perch_v2.onnx"


def default_taxonomy_path() -> Path:
    """Locate ``taxonomy.csv``, whether running from a clone or an installed wheel."""
    packaged = _package_root() / "data" / "taxonomy.csv"
    if packaged.exists():
        return packaged
    return _repo_root() / "data" / "taxonomy.csv"


class Settings(BaseSettings):
    """Service settings. Every field maps to ``WILDECHO_<FIELD_NAME>``."""

    model_config = SettingsConfigDict(
        env_prefix="WILDECHO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `model_path` would otherwise collide with pydantic's protected
        # `model_` namespace and emit a warning.
        protected_namespaces=(),
    )

    # -- Model ---------------------------------------------------------------
    model_path: Path = Field(
        default_factory=default_model_path,
        description="Path to the Perch 2.0 ONNX file. Downloaded by scripts/download_model.py.",
    )
    taxonomy_path: Path = Field(
        default_factory=default_taxonomy_path,
        description="Path to the index -> species CSV shipped in data/.",
    )
    model_version: str = Field(
        default="perch_v2",
        description="Reported by /v1/about. Informational only.",
    )
    onnx_intra_op_threads: int = Field(
        default=0,
        ge=0,
        description="ONNX Runtime intra-op thread count. 0 lets the runtime decide.",
    )

    # -- Prediction behaviour ------------------------------------------------
    top_k: int = Field(default=10, ge=1, le=100, description="Candidates returned per request.")
    low_confidence_threshold: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description=(
            "Top confidence below this marks the result low_confidence. The 0.30 default "
            "was calibrated against real clips: a clean single-species recording scores "
            "~0.63-0.73, white noise ~0.05, and digital silence ~0.01."
        ),
    )

    # -- Upload limits (these are the DoS controls, see SECURITY.md) ---------
    max_upload_bytes: int = Field(
        default=25 * 1024 * 1024,
        gt=0,
        description="Reject uploads larger than this many bytes.",
    )
    max_duration_seconds: float = Field(
        default=300.0,
        gt=0,
        description="Reject clips longer than this. Inference cost scales with duration.",
    )
    min_duration_seconds: float = Field(
        default=0.5,
        gt=0,
        description="Reject clips shorter than this as unusable.",
    )
    silence_rms_threshold: float = Field(
        default=1e-4,
        ge=0.0,
        description="Reject clips whose RMS amplitude is below this as silent.",
    )

    # -- ffmpeg --------------------------------------------------------------
    ffmpeg_path: str = Field(default="ffmpeg", description="ffmpeg binary name or path.")
    ffprobe_path: str = Field(default="ffprobe", description="ffprobe binary name or path.")
    ffmpeg_timeout_seconds: int = Field(
        default=60, gt=0, description="Wall-clock timeout for each ffmpeg/ffprobe call."
    )

    # -- HTTP ----------------------------------------------------------------
    # Kept as a plain string rather than list[str] on purpose. pydantic-settings
    # JSON-decodes complex-typed fields inside the env source, before any validator
    # runs, so `WILDECHO_CORS_ORIGINS=*` would fail to parse. Use cors_origin_list.
    cors_origins: str = Field(
        default="*",
        description=(
            "Comma-separated allowed origins, or '*' for any. Defaults to '*' so a mobile "
            "app works out of the box; lock this down for a public deployment."
        ),
    )
    rate_limit: str = Field(
        default="20/hour",
        description="slowapi limit string applied per client IP, e.g. '20/hour' or '5/minute'.",
    )
    rate_limit_enabled: bool = Field(default=True, description="Set false to disable limiting.")
    log_level: str = Field(default="INFO", description="Root log level for the service.")

    @property
    def cors_origin_list(self) -> list[str]:
        """``cors_origins`` split into the list Starlette's CORS middleware wants.

        Accepts ``*``, a single origin, or a comma-separated list. An empty or
        whitespace-only value disables cross-origin access entirely, which is the
        right setting for a service only ever called by a native mobile app.
        """
        raw = self.cors_origins.strip()
        if raw == "*":
            return ["*"]
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        return value.upper()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
