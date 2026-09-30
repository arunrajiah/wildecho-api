"""FastAPI application: the HTTP surface over Perch 2.0."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
import tempfile
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.concurrency import run_in_threadpool

from . import __version__
from .audio import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_EXTENSIONS,
    AudioEmptyError,
    AudioError,
    FfmpegUnavailableError,
    UnsupportedFormatError,
    decode_file,
)
from .config import (
    NUM_CLASSES,
    TARGET_SAMPLE_RATE,
    WINDOW_SECONDS,
    Settings,
    get_settings,
)
from .feedback import (
    FeedbackDisabledError,
    FeedbackRecord,
    FeedbackStore,
    FeedbackStoreUnavailableError,
    SQLiteFeedbackStore,
    match_correction,
)
from .inference import WINDOW_STRIDE_SECONDS, ModelLoadError, get_load_error, get_model, load_model
from .logging_utils import configure_logging, get_request_id, new_request_id, set_request_id
from .schemas import (
    AboutResponse,
    Attribution,
    ClipMetadata,
    ErrorResponse,
    FeedbackResponse,
    HealthResponse,
    IdentifyResponse,
    ModelStatus,
    TaxaCoverage,
    TaxonomicGroup,
)
from .taxonomy import Taxonomy, TaxonomyError, load_taxonomy

logger = logging.getLogger(__name__)

UPLOAD_CHUNK_BYTES = 1 << 20  # 1 MiB

DISCLAIMER = (
    "Perch 2.0 recognises about 14,600 taxa, but its training data is heavily "
    "bird-weighted: about 70% of its species classes are birds (10,256 of 14,597), and its "
    "non-bird coverage (insects, frogs, mammals) is much thinner and less evenly "
    "sampled. There is no bat coverage at all, because bat echolocation is largely "
    "ultrasonic and this model only sees audio up to 16 kHz. Results for anything "
    "other than a clearly recorded bird should be treated as a ranked set of "
    "suggestions to verify, not an identification. Never rely on this service alone "
    "for conservation, regulatory, or safety decisions."
)

ATTRIBUTION = Attribution(
    model_name="Perch 2.0",
    model_authors="Google Research (Perch team) and contributors",
    model_license="Apache-2.0",
    citation=(
        "Hamer, J., Denton, T., et al. 'Perch 2.0: The Bittern Lesson for "
        "Bioacoustics.' arXiv:2508.04665 (2025)."
    ),
    links={
        "perch_github": "https://github.com/google-research/perch",
        "perch_hoplite": "https://github.com/google-research/perch-hoplite",
        "paper": "https://arxiv.org/abs/2508.04665",
        "model_card": "https://www.kaggle.com/models/google/perch",
        "onnx_export": "https://huggingface.co/justinchuby/Perch-onnx",
        "labels_and_weights": "https://huggingface.co/cgeorgiaw/Perch",
        "wildecho_api": "https://github.com/arunrajiah/wildecho-api",
    },
)


class AppState:
    """Holds what was loaded at startup.

    The taxonomy is loaded independently of the weights so that ``/v1/health`` and
    ``/v1/about`` still answer usefully on a fresh install where nobody has run the
    400 MB download yet. Starting up degraded and saying so beats crash-looping.
    """

    def __init__(self) -> None:
        self.taxonomy: Taxonomy | None = None
        self.taxonomy_error: str | None = None
        self.feedback_store: FeedbackStore | None = None
        self.feedback_error: str | None = None


state = AppState()


async def _init_feedback_store(settings: Settings) -> None:
    if not settings.feedback_enabled:
        state.feedback_store = None
        state.feedback_error = None
        return
    store = SQLiteFeedbackStore(settings.feedback_db_path)
    try:
        await run_in_threadpool(store.init)
    except (OSError, sqlite3.Error) as exc:
        # OSError covers mkdir/permission failures on the containing directory;
        # sqlite3.Error covers the database itself. Either way this must degrade
        # rather than crash startup - feedback is optional, the service is not.
        state.feedback_store = None
        state.feedback_error = str(exc)
        logger.error("feedback store unavailable: %s", exc)
    else:
        state.feedback_store = store
        state.feedback_error = None
        logger.info("feedback store ready at %s", settings.feedback_db_path)


def _load_taxonomy_into_state(settings: Settings) -> None:
    try:
        state.taxonomy = load_taxonomy(settings.taxonomy_path, expected_rows=NUM_CLASSES)
        state.taxonomy_error = None
        logger.info("loaded taxonomy: %d classes", len(state.taxonomy))
    except TaxonomyError as exc:
        state.taxonomy = None
        state.taxonomy_error = str(exc)
        logger.error("taxonomy unavailable: %s", exc)


_inference_semaphores: dict[int, asyncio.Semaphore] = {}


def _inference_slot(limit: int) -> contextlib.AbstractAsyncContextManager[object]:
    """Async context that holds one of ``limit`` inference slots (no-op when 0)."""
    if limit <= 0:
        return contextlib.nullcontext()
    if limit not in _inference_semaphores:
        _inference_semaphores[limit] = asyncio.Semaphore(limit)
    return _inference_semaphores[limit]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    _load_taxonomy_into_state(settings)
    await _init_feedback_store(settings)
    try:
        load_model(settings)
    except ModelLoadError as exc:
        logger.warning("starting without a model: %s", exc)
    yield


settings = get_settings()

limiter = Limiter(
    key_func=get_remote_address,
    enabled=settings.rate_limit_enabled,
    default_limits=[],
)

app = FastAPI(
    title="wildecho-api",
    version=__version__,
    summary="Self-hostable species ID from audio, powered by Google's open Perch 2.0 model.",
    description=(
        "Upload a short recording, get back ranked species candidates.\n\n"
        "This wrapper is MIT licensed. The Perch 2.0 model it runs is Apache-2.0, "
        "copyright Google LLC, and is not redistributed with this code. "
        "See `GET /v1/about` for attribution and the accuracy caveats, which you "
        "should read before showing results to anyone."
    ),
    lifespan=lifespan,
    openapi_tags=[
        {"name": "identify", "description": "Species identification from audio."},
        {"name": "feedback", "description": "Opt-in, locally stored accuracy corrections."},
        {"name": "meta", "description": "Health, provenance and coverage."},
    ],
)

app.state.limiter = limiter

# Permissive by default so a mobile app can call the service without a proxy in the
# middle. Self-hosters exposing this publicly should set WILDECHO_CORS_ORIGINS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,
)


def _error(status_code: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=code, detail=detail).model_dump(),
    )


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> Response:
    return _error(
        status.HTTP_429_TOO_MANY_REQUESTS,
        "rate_limited",
        f"Rate limit exceeded ({exc.detail}). This instance allows "
        f"{get_settings().rate_limit} per client IP. If you self-host, raise "
        "WILDECHO_RATE_LIMIT.",
    )


@app.exception_handler(AudioError)
async def audio_error_handler(request: Request, exc: AudioError) -> Response:
    return _error(exc.http_status, exc.code, exc.detail)


@app.exception_handler(ModelLoadError)
async def model_error_handler(request: Request, exc: ModelLoadError) -> Response:
    logger.error("model unavailable: %s", exc)
    return _error(status.HTTP_503_SERVICE_UNAVAILABLE, "model_unavailable", str(exc))


@app.exception_handler(FfmpegUnavailableError)
async def ffmpeg_error_handler(request: Request, exc: FfmpegUnavailableError) -> Response:
    logger.error("ffmpeg unavailable: %s", exc)
    return _error(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "ffmpeg_unavailable",
        "Audio decoding is unavailable on this server because ffmpeg is not installed.",
    )


@app.exception_handler(FeedbackDisabledError)
async def feedback_disabled_handler(request: Request, exc: FeedbackDisabledError) -> Response:
    return _error(status.HTTP_404_NOT_FOUND, "feedback_disabled", str(exc))


@app.exception_handler(FeedbackStoreUnavailableError)
async def feedback_unavailable_handler(
    request: Request, exc: FeedbackStoreUnavailableError
) -> Response:
    logger.error("feedback store unavailable: %s", exc)
    return _error(status.HTTP_503_SERVICE_UNAVAILABLE, "feedback_unavailable", str(exc))


def _rate_limit() -> str:
    """Read the limit at call time so tests and env changes take effect."""
    return get_settings().rate_limit


async def _spool_upload(upload: UploadFile, destination: Path, max_bytes: int) -> int:
    """Stream an upload to disk, aborting as soon as it exceeds ``max_bytes``.

    Streaming rather than ``await upload.read()`` means a hostile 2 GB body never
    becomes a 2 GB Python object.
    """
    written = 0
    with destination.open("wb") as handle:
        while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
            written += len(chunk)
            if written > max_bytes:
                raise AudioError(
                    f"Upload exceeds the {max_bytes // (1024 * 1024)} MB limit. "
                    "Send a shorter or more compressed clip."
                )
            handle.write(chunk)
    return written


def _check_declared_format(upload: UploadFile) -> None:
    """Cheap pre-flight rejection on content type and extension.

    Advisory only. ffmpeg is the real authority on whether bytes are decodable; this
    just avoids spawning a subprocess for an obvious image or PDF upload.
    """
    content_type = (upload.content_type or "").split(";")[0].strip().lower()
    suffix = Path(upload.filename or "").suffix.lower()

    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        if suffix in ALLOWED_EXTENSIONS:
            return
        raise UnsupportedFormatError(
            f"Content type {content_type!r} is not a supported audio type. "
            "Send wav, mp3, m4a, webm, ogg or flac."
        )
    if not content_type and suffix and suffix not in ALLOWED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"File extension {suffix!r} is not a supported audio format. "
            "Send wav, mp3, m4a, webm, ogg or flac."
        )


@app.post(
    "/v1/identify",
    response_model=IdentifyResponse,
    tags=["identify"],
    summary="Identify species in an audio clip",
    responses={
        413: {"model": ErrorResponse, "description": "Upload or clip too large"},
        415: {"model": ErrorResponse, "description": "Unsupported audio format"},
        422: {"model": ErrorResponse, "description": "Clip empty, too short, or silent"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        503: {"model": ErrorResponse, "description": "Model or ffmpeg unavailable"},
    },
)
@limiter.limit(_rate_limit)
async def identify_endpoint(
    request: Request,
    file: UploadFile = File(..., description="Audio clip: wav, mp3, m4a, webm, ogg or flac."),
) -> IdentifyResponse:
    """Identify the species in an uploaded recording.

    Audio is decoded to 32 kHz mono, split into overlapping 5-second windows, scored
    by Perch 2.0, and averaged. General sound event classes such as wind, rain and
    speech are excluded from the results; when one of them was the top-scoring class
    overall, `non_animal_top_class` says so and the species list should be distrusted.
    """
    active_settings = get_settings()
    model = get_model()
    if model is None:
        raise ModelLoadError(
            get_load_error() or "The model is not loaded. See GET /v1/health for details."
        )

    _check_declared_format(file)

    # A temp file rather than a pipe: ffprobe needs to seek to read container
    # metadata, and it lets us bound duration before decoding.
    with tempfile.TemporaryDirectory(prefix="wildecho-") as directory:
        # The client's filename is never used as a path component.
        destination = Path(directory) / "upload"
        size = await _spool_upload(file, destination, active_settings.max_upload_bytes)
        if size == 0:
            raise AudioEmptyError("The uploaded file is empty.")

        # decode_file (an ffmpeg subprocess) and model.identify (ONNX Runtime) are
        # both synchronous, CPU/IO-bound calls. Awaiting them directly would block
        # this single process's event loop for their whole duration, serialising
        # every concurrent request behind one at a time. Running them in the
        # threadpool keeps the loop free to accept and log other requests, and -
        # since both release the GIL while their C code runs - lets genuinely
        # concurrent requests use multiple cores instead of only one. See the
        # README "Capacity planning" section for the throughput math this enables.
        decoded = await run_in_threadpool(decode_file, destination, active_settings)
        async with _inference_slot(active_settings.max_concurrent_inferences):
            result = await run_in_threadpool(model.identify, decoded.samples, decoded.sample_rate)

    return IdentifyResponse(
        predictions=result.predictions,
        low_confidence=result.low_confidence,
        non_animal_top_class=result.non_animal_top_class,
        model_version=active_settings.model_version,
        request_id=get_request_id() or new_request_id(),
        metadata=ClipMetadata(
            duration_seconds=round(result.duration_seconds, 3),
            windows_processed=result.windows_processed,
            window_seconds=WINDOW_SECONDS,
            window_stride_seconds=WINDOW_STRIDE_SECONDS,
            source_sample_rate=decoded.source.sample_rate,
            source_channels=decoded.source.channels,
            processed_sample_rate=decoded.sample_rate,
            inference_ms=round(result.inference_ms, 2),
        ),
    )


@app.post(
    "/v1/feedback",
    response_model=FeedbackResponse,
    tags=["feedback"],
    summary="Submit a correction for a previous identification",
    responses={
        404: {"model": ErrorResponse, "description": "Feedback is disabled on this server"},
        413: {"model": ErrorResponse, "description": "Optional clip upload too large"},
        422: {"model": ErrorResponse, "description": "Missing corrected_text"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        503: {"model": ErrorResponse, "description": "Feedback store unavailable"},
    },
)
@limiter.limit(_rate_limit)
async def feedback_endpoint(
    request: Request,
    corrected_text: str = Form(
        ...,
        min_length=1,
        description=(
            "The species you believe this clip actually is: a scientific or common name "
            "(matched case-insensitively against the taxonomy) or free text if neither is "
            "known. Always stored verbatim regardless of whether it matches."
        ),
    ),
    clip_id: str | None = Form(
        default=None,
        description=(
            "An identifier you choose, for correlating with your own records. Not validated."
        ),
    ),
    request_id: str | None = Form(
        default=None,
        description="The request_id from the original /v1/identify response, for correlation.",
    ),
    original_scientific_name: str | None = Form(
        default=None,
        description="What the model originally predicted as top-1, if you have it.",
    ),
    original_confidence: float | None = Form(
        default=None,
        description="The model's original top-1 confidence, if you have it.",
    ),
    notes: str | None = Form(
        default=None,
        description="Any free-text context, e.g. how you know the correct identification.",
    ),
    file: UploadFile | None = File(
        default=None,
        description="Optional: the audio clip this correction is about.",
    ),
) -> FeedbackResponse:
    """Record a user-submitted correction for later review and model tuning.

    Entirely local and opt-in: nothing is collected unless a client calls this
    endpoint, storage is a SQLite file on this machine by default, and nothing
    submitted here is sent anywhere else unless you build a remote sync yourself.
    See README "Feedback" for the full privacy note and how to disable this
    endpoint or swap in Postgres.
    """
    active_settings = get_settings()
    if not active_settings.feedback_enabled:
        raise FeedbackDisabledError(
            "Feedback is disabled on this server. Enable it with WILDECHO_FEEDBACK_ENABLED=true."
        )
    store = state.feedback_store
    if store is None:
        raise FeedbackStoreUnavailableError(
            state.feedback_error or "The feedback store failed to initialize. Check server logs."
        )

    stored_audio = False
    clip_path: str | None = None
    if file is not None and file.filename:
        if not active_settings.feedback_store_audio:
            await file.read()  # drain the multipart body; nothing is written to disk
        else:
            clips_dir = active_settings.feedback_clips_dir
            clips_dir.mkdir(parents=True, exist_ok=True)
            suffix = Path(file.filename).suffix or ".bin"
            # A generated name, never the client's filename, avoids path traversal
            # and collisions between submissions.
            destination = clips_dir / f"{new_request_id()}{suffix}"
            size = await _spool_upload(file, destination, active_settings.feedback_max_upload_bytes)
            if size == 0:
                destination.unlink(missing_ok=True)
            else:
                clip_path = str(destination)
                stored_audio = True

    taxonomy = state.taxonomy
    matched = match_correction(corrected_text, taxonomy) if taxonomy is not None else None

    record = FeedbackRecord(
        created_at=datetime.now(UTC).isoformat(),
        request_id=request_id,
        clip_id=clip_id,
        clip_path=clip_path,
        original_scientific_name=original_scientific_name,
        original_confidence=original_confidence,
        corrected_text=corrected_text,
        matched=matched,
        notes=notes,
        client_ip=get_remote_address(request),
    )
    feedback_id = await run_in_threadpool(store.insert, record)

    logger.info(
        "feedback %d recorded: corrected_text=%r matched=%s request_id=%s clip_id=%s",
        feedback_id,
        corrected_text,
        matched.scientific_name if matched else None,
        request_id,
        clip_id,
    )

    return FeedbackResponse(
        id=feedback_id,
        received_at=record.created_at,
        matched_scientific_name=matched.scientific_name if matched else None,
        matched_common_name=matched.common_name if matched else None,
        matched_taxonomic_group=matched.taxonomic_group if matched else None,
        stored_audio=stored_audio,
    )


@app.get(
    "/v1/health",
    response_model=HealthResponse,
    tags=["meta"],
    summary="Model load status",
)
async def health_endpoint() -> HealthResponse:
    """Report whether the service can actually serve identifications.

    Returns 200 either way. A load failure is a configuration state to report, not a
    reason to fail the probe and get restarted forever. Check `model_loaded`.
    """
    active_settings = get_settings()
    model = get_model()
    error = get_load_error() or state.taxonomy_error

    if model is not None:
        model_status = ModelStatus.LOADED
    elif get_load_error():
        model_status = ModelStatus.ERROR
    else:
        model_status = ModelStatus.NOT_LOADED

    return HealthResponse(
        status="ok" if model is not None else "degraded",
        model_status=model_status,
        model_path=str(active_settings.model_path),
        model_loaded=model is not None,
        detail=error,
        num_classes=model.num_classes if model is not None else None,
        taxonomy_loaded=state.taxonomy is not None,
        feedback_enabled=active_settings.feedback_enabled,
        feedback_store_ready=state.feedback_store is not None,
        version=__version__,
    )


@app.get(
    "/v1/about",
    response_model=AboutResponse,
    tags=["meta"],
    summary="Model provenance, taxa coverage and accuracy caveats",
)
async def about_endpoint() -> AboutResponse:
    """Describe the service, credit the model's authors, and state its limits plainly."""
    active_settings = get_settings()
    taxonomy = state.taxonomy

    if taxonomy is not None:
        species = taxonomy.count_species()
        coverage = TaxaCoverage(
            total_classes=len(taxonomy),
            species_classes=species,
            general_sound_event_classes=taxonomy.count_sound_events(),
            birds=taxonomy.count_group(TaxonomicGroup.BIRD),
            non_bird_species=species - taxonomy.count_group(TaxonomicGroup.BIRD),
            disclaimer=DISCLAIMER,
        )
    else:
        coverage = TaxaCoverage(
            total_classes=NUM_CLASSES,
            species_classes=0,
            general_sound_event_classes=0,
            birds=0,
            non_bird_species=0,
            disclaimer=(
                "Taxonomy table unavailable, so coverage counts cannot be reported. " + DISCLAIMER
            ),
        )

    return AboutResponse(
        name="wildecho-api",
        version=__version__,
        description=(
            "Self-hostable species ID from audio, powered by Google's open Perch 2.0 model."
        ),
        model_version=active_settings.model_version,
        license="MIT (this wrapper only; Perch 2.0 is Apache-2.0, copyright Google LLC)",
        coverage=coverage,
        attribution=ATTRIBUTION,
        limits={
            "max_upload_bytes": active_settings.max_upload_bytes,
            "max_duration_seconds": active_settings.max_duration_seconds,
            "min_duration_seconds": active_settings.min_duration_seconds,
            "window_seconds": WINDOW_SECONDS,
            "window_stride_seconds": WINDOW_STRIDE_SECONDS,
            "processed_sample_rate": TARGET_SAMPLE_RATE,
            "top_k": active_settings.top_k,
            "low_confidence_threshold": active_settings.low_confidence_threshold,
            "rate_limit": (
                active_settings.rate_limit if active_settings.rate_limit_enabled else "disabled"
            ),
        },
    )


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {
        "name": "wildecho-api",
        "version": __version__,
        "docs": "/docs",
        "health": "/v1/health",
        "about": "/v1/about",
    }


# Registered last so it wraps every route above, including error responses that
# our exception handlers turn into ordinary Responses.
@app.middleware("http")
async def request_context_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Binds a request ID for this request's lifetime and logs its outcome.

    The ID is reused from the client's request_id_header if it sent one, so a
    mobile client can mint its own and see it echoed back verbatim in these logs
    and in a later /v1/feedback submission; otherwise a fresh one is generated.
    Every log line anywhere during this request carries it automatically, via
    logging_utils' contextvar-backed filter.
    """
    active_settings = get_settings()
    incoming = request.headers.get(active_settings.request_id_header)
    request_id = incoming.strip() if incoming and incoming.strip() else new_request_id()
    set_request_id(request_id)

    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - started) * 1000.0

    logger.info(
        "%s %s -> %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    response.headers[active_settings.request_id_header] = request_id
    response.headers["X-Wildecho-Version"] = __version__
    return response
