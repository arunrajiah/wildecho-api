"""HTTP endpoint behaviour, against a mocked model."""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from conftest import noise, tone

from wildecho_api import __version__, main
from wildecho_api.config import TARGET_SAMPLE_RATE, Settings, get_settings

ffmpeg_available = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
requires_ffmpeg = pytest.mark.skipif(not ffmpeg_available, reason="ffmpeg/ffprobe not installed")


# ---------------------------------------------------------------------------
# /v1/health
# ---------------------------------------------------------------------------
def test_health_reports_ok_when_loaded(client) -> None:
    response = client.get("/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_status"] == "loaded"
    assert body["taxonomy_loaded"] is True
    assert body["num_classes"] == 14_795
    assert body["version"] == __version__


def test_health_reports_degraded_without_a_model(bare_client) -> None:
    """A missing model must be a reportable state, not a 500 or a crash loop."""
    response = bare_client.get("/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert body["model_status"] == "error"
    assert body["detail"]


# ---------------------------------------------------------------------------
# /v1/about
# ---------------------------------------------------------------------------
def test_about_reports_coverage(client) -> None:
    body = client.get("/v1/about").json()
    assert body["name"] == "wildecho-api"
    assert body["coverage"]["total_classes"] == 14_795
    assert body["coverage"]["species_classes"] == 14_597
    assert body["coverage"]["general_sound_event_classes"] == 198
    assert body["coverage"]["birds"] > 10_000
    assert body["coverage"]["non_bird_species"] > 4_000


def test_about_states_the_limitations_plainly(client) -> None:
    """These caveats are load-bearing. If someone deletes them, this test fails."""
    disclaimer = client.get("/v1/about").json()["coverage"]["disclaimer"].lower()
    assert "bird" in disclaimer
    assert "bat" in disclaimer
    assert "no bat coverage" in disclaimer


def test_about_credits_google_research(client) -> None:
    attribution = client.get("/v1/about").json()["attribution"]
    assert attribution["model_name"] == "Perch 2.0"
    assert "Google Research" in attribution["model_authors"]
    assert attribution["model_license"] == "Apache-2.0"
    assert "arxiv" in attribution["citation"].lower()
    assert "github.com/google-research/perch" in attribution["links"]["perch_github"]


def test_about_separates_wrapper_and_model_licenses(client) -> None:
    body = client.get("/v1/about").json()
    assert "MIT" in body["license"]
    assert "Apache-2.0" in body["license"]


def test_about_works_without_a_model(bare_client) -> None:
    body = bare_client.get("/v1/about").json()
    assert body["name"] == "wildecho-api"
    assert "bat" in body["coverage"]["disclaimer"].lower()


def test_about_reports_limits(client, settings: Settings) -> None:
    limits = client.get("/v1/about").json()["limits"]
    assert limits["processed_sample_rate"] == TARGET_SAMPLE_RATE
    assert limits["window_seconds"] == 5.0
    assert limits["window_stride_seconds"] == 2.5
    assert limits["top_k"] == settings.top_k


# ---------------------------------------------------------------------------
# /v1/identify
# ---------------------------------------------------------------------------
@requires_ffmpeg
def test_identify_returns_predictions(client, wav_bytes, nightjar_index: int) -> None:
    payload = wav_bytes(tone(6.0))
    response = client.post("/v1/identify", files={"file": ("clip.wav", payload, "audio/wav")})
    assert response.status_code == 200, response.text

    body = response.json()
    assert len(body["predictions"]) == 10
    top = body["predictions"][0]
    assert top["class_index"] == nightjar_index
    assert top["scientific_name"] == "Caprimulgus europaeus"
    assert top["common_name"] == "European Nightjar"
    assert top["taxonomic_group"] == "bird"
    assert 0.0 <= top["confidence"] <= 1.0
    assert top["low_confidence"] is False
    assert body["low_confidence"] is False
    assert body["non_animal_top_class"] is None


@requires_ffmpeg
def test_identify_reports_clip_metadata(client, wav_bytes) -> None:
    payload = wav_bytes(tone(10.0, sample_rate=44_100), sample_rate=44_100)
    body = client.post("/v1/identify", files={"file": ("clip.wav", payload, "audio/wav")}).json()

    metadata = body["metadata"]
    assert 9.9 < metadata["duration_seconds"] < 10.1
    assert metadata["windows_processed"] == 3
    assert metadata["window_seconds"] == 5.0
    assert metadata["window_stride_seconds"] == 2.5
    assert metadata["source_sample_rate"] == 44_100
    assert metadata["source_channels"] == 1
    assert metadata["processed_sample_rate"] == TARGET_SAMPLE_RATE
    assert metadata["inference_ms"] >= 0.0


@requires_ffmpeg
def test_identify_accepts_the_mp3_fixture(client) -> None:
    from conftest import NIGHTJAR_CLIP

    with NIGHTJAR_CLIP.open("rb") as handle:
        response = client.post(
            "/v1/identify",
            files={"file": ("nightjar.mp3", handle.read(), "audio/mpeg")},
        )
    assert response.status_code == 200, response.text
    assert response.json()["metadata"]["windows_processed"] == 3


@requires_ffmpeg
def test_identify_excludes_sound_event_classes(
    client, make_model, wind_index: int, wav_bytes
) -> None:
    from conftest import logits_favouring

    from wildecho_api import inference

    inference.set_model(make_model(logits_favouring(wind_index, peak=20.0)))
    body = client.post(
        "/v1/identify", files={"file": ("wind.wav", wav_bytes(noise(6.0)), "audio/wav")}
    ).json()

    assert body["non_animal_top_class"] == "Wind"
    assert wind_index not in {p["class_index"] for p in body["predictions"]}
    assert body["low_confidence"] is True


@requires_ffmpeg
def test_identify_rejects_silence(client, wav_bytes) -> None:
    payload = wav_bytes(np.zeros(TARGET_SAMPLE_RATE * 3, dtype=np.float32))
    response = client.post("/v1/identify", files={"file": ("silence.wav", payload, "audio/wav")})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "audio_silent"
    assert "silent" in body["detail"].lower()


@requires_ffmpeg
def test_identify_rejects_too_short(client, wav_bytes) -> None:
    response = client.post(
        "/v1/identify", files={"file": ("blip.wav", wav_bytes(tone(0.1)), "audio/wav")}
    )
    assert response.status_code == 422
    assert response.json()["error"] == "audio_too_short"


@requires_ffmpeg
def test_identify_rejects_garbage_bytes(client) -> None:
    response = client.post(
        "/v1/identify", files={"file": ("clip.wav", b"not audio at all" * 200, "audio/wav")}
    )
    assert response.status_code in (415, 422)
    assert response.json()["error"] in ("unsupported_format", "audio_decode_failed")


def test_identify_rejects_empty_file(client) -> None:
    response = client.post("/v1/identify", files={"file": ("empty.wav", b"", "audio/wav")})
    assert response.status_code == 422
    assert response.json()["error"] in ("audio_empty", "unsupported_format", "audio_decode_failed")


def test_identify_rejects_wrong_content_type(client) -> None:
    response = client.post(
        "/v1/identify", files={"file": ("photo.jpg", b"\xff\xd8\xff\xe0jpegdata", "image/jpeg")}
    )
    assert response.status_code == 415
    assert response.json()["error"] == "unsupported_format"


def test_identify_accepts_octet_stream_from_mobile_clients(client, wav_bytes) -> None:
    """Mobile HTTP libraries frequently send application/octet-stream."""
    response = client.post(
        "/v1/identify",
        files={"file": ("clip.wav", wav_bytes(tone(6.0)), "application/octet-stream")},
    )
    assert response.status_code in (200, 503)  # 503 only if ffmpeg is missing
    if response.status_code == 200:
        assert response.json()["predictions"]


def test_identify_requires_a_file(client) -> None:
    assert client.post("/v1/identify").status_code == 422


def test_identify_returns_503_without_a_model(bare_client, wav_bytes) -> None:
    response = bare_client.post(
        "/v1/identify", files={"file": ("clip.wav", wav_bytes(tone(6.0)), "audio/wav")}
    )
    assert response.status_code == 503
    assert response.json()["error"] == "model_unavailable"


def test_identify_rejects_oversized_upload(client, monkeypatch, settings: Settings) -> None:
    small = settings.model_copy(update={"max_upload_bytes": 1024})
    monkeypatch.setattr(main, "get_settings", lambda: small)
    response = client.post("/v1/identify", files={"file": ("big.wav", b"\x00" * 8192, "audio/wav")})
    assert response.status_code == 422
    assert "limit" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
def test_rate_limit_returns_429_with_a_useful_body(
    client, monkeypatch, settings: Settings, reset_rate_limiter
) -> None:
    throttled = settings.model_copy(update={"rate_limit": "2/minute", "rate_limit_enabled": True})
    monkeypatch.setattr(main, "get_settings", lambda: throttled)
    monkeypatch.setattr(main.limiter, "enabled", True)

    files = {"file": ("photo.jpg", b"nope", "image/jpeg")}
    # The first two requests fail format validation (415) but still consume quota.
    assert client.post("/v1/identify", files=files).status_code == 415
    assert client.post("/v1/identify", files=files).status_code == 415

    response = client.post("/v1/identify", files=files)
    assert response.status_code == 429
    body = response.json()
    assert body["error"] == "rate_limited"
    assert "WILDECHO_RATE_LIMIT" in body["detail"]


def test_rate_limit_is_disabled_by_default_in_tests() -> None:
    assert get_settings().rate_limit_enabled is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("*", ["*"]),
        ("https://a.app", ["https://a.app"]),
        ("https://a.app,https://b.app", ["https://a.app", "https://b.app"]),
        (" https://a.app , https://b.app ", ["https://a.app", "https://b.app"]),
        ("", []),
        ("   ", []),
    ],
)
def test_cors_origins_parses_from_a_plain_string(
    monkeypatch, raw: str, expected: list[str]
) -> None:
    """Regression: `WILDECHO_CORS_ORIGINS=*` used to crash at import.

    pydantic-settings JSON-decodes complex-typed fields inside the env source, so a
    `list[str]` field could never accept a bare `*`. The container failed to start.
    """
    monkeypatch.setenv("WILDECHO_CORS_ORIGINS", raw)
    get_settings.cache_clear()
    try:
        assert get_settings().cors_origin_list == expected
    finally:
        monkeypatch.delenv("WILDECHO_CORS_ORIGINS", raising=False)
        get_settings.cache_clear()


def test_rate_limit_string_comes_from_settings(monkeypatch, settings: Settings) -> None:
    custom = settings.model_copy(update={"rate_limit": "7/hour"})
    monkeypatch.setattr(main, "get_settings", lambda: custom)
    assert main._rate_limit() == "7/hour"


# ---------------------------------------------------------------------------
# Cross-cutting
# ---------------------------------------------------------------------------
def test_cors_headers_are_present(client) -> None:
    response = client.options(
        "/v1/identify",
        headers={
            "Origin": "https://example.app",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code in (200, 204)
    assert "access-control-allow-origin" in {k.lower() for k in response.headers}


def test_version_header_is_added(client) -> None:
    assert client.get("/v1/health").headers["X-Wildecho-Version"] == __version__


def test_root_lists_entry_points(client) -> None:
    body = client.get("/").json()
    assert body["health"] == "/v1/health"
    assert body["about"] == "/v1/about"


def test_openapi_schema_is_valid(client) -> None:
    schema = client.get("/openapi.json").json()
    assert "/v1/identify" in schema["paths"]
    assert "/v1/health" in schema["paths"]
    assert "/v1/about" in schema["paths"]


def test_unknown_route_is_404(client) -> None:
    assert client.get("/v1/nope").status_code == 404
