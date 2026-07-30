"""POST /v1/feedback: the opt-in, locally stored correction endpoint."""

from __future__ import annotations

import sqlite3

from conftest import NIGHTJAR_CLIP


def _rows(db_path: str) -> list[sqlite3.Row]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute("SELECT * FROM feedback ORDER BY id").fetchall()
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def test_matches_by_exact_scientific_name(isolated_feedback_client) -> None:
    response = isolated_feedback_client.post(
        "/v1/feedback", data={"corrected_text": "Turdus migratorius"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["matched_scientific_name"] == "Turdus migratorius"
    assert body["matched_common_name"] == "American Robin"
    assert body["matched_taxonomic_group"] == "bird"


def test_matches_by_exact_common_name_case_insensitively(isolated_feedback_client) -> None:
    response = isolated_feedback_client.post(
        "/v1/feedback", data={"corrected_text": "american robin"}
    )
    body = response.json()
    assert body["matched_scientific_name"] == "Turdus migratorius"
    assert body["matched_common_name"] == "American Robin"


def test_unmatched_free_text_is_stored_without_a_match(isolated_feedback_client) -> None:
    response = isolated_feedback_client.post(
        "/v1/feedback", data={"corrected_text": "some bird I couldn't identify"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["matched_scientific_name"] is None
    assert body["matched_common_name"] is None
    assert body["matched_taxonomic_group"] is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_requires_corrected_text(isolated_feedback_client) -> None:
    response = isolated_feedback_client.post("/v1/feedback", data={})
    assert response.status_code == 422


def test_rejects_blank_corrected_text(isolated_feedback_client) -> None:
    response = isolated_feedback_client.post("/v1/feedback", data={"corrected_text": ""})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def test_first_submission_gets_id_one(isolated_feedback_client) -> None:
    response = isolated_feedback_client.post(
        "/v1/feedback", data={"corrected_text": "Panthera leo"}
    )
    assert response.json()["id"] == 1


def test_ids_increment_across_submissions(isolated_feedback_client) -> None:
    first = isolated_feedback_client.post("/v1/feedback", data={"corrected_text": "Panthera leo"})
    second = isolated_feedback_client.post(
        "/v1/feedback", data={"corrected_text": "Apis mellifera"}
    )
    assert first.json()["id"] == 1
    assert second.json()["id"] == 2


def test_optional_fields_are_persisted(isolated_feedback_client, tmp_path) -> None:
    response = isolated_feedback_client.post(
        "/v1/feedback",
        data={
            "corrected_text": "Caprimulgus europaeus",
            "clip_id": "my-clip-42",
            "request_id": "abc123",
            "original_scientific_name": "Anthus trivialis",
            "original_confidence": "0.28",
            "notes": "heard the churring song at dusk",
        },
    )
    assert response.status_code == 200

    rows = _rows(str(tmp_path / "feedback.sqlite3"))
    assert len(rows) == 1
    row = rows[0]
    assert row["clip_id"] == "my-clip-42"
    assert row["request_id"] == "abc123"
    assert row["original_scientific_name"] == "Anthus trivialis"
    assert row["original_confidence"] == 0.28
    assert row["notes"] == "heard the churring song at dusk"
    assert row["matched_scientific_name"] == "Caprimulgus europaeus"
    assert row["matched_taxonomic_group"] == "bird"
    assert row["client_ip"] is not None


def test_response_received_at_matches_stored_created_at(isolated_feedback_client, tmp_path) -> None:
    response = isolated_feedback_client.post(
        "/v1/feedback", data={"corrected_text": "Rana temporaria"}
    )
    received_at = response.json()["received_at"]
    rows = _rows(str(tmp_path / "feedback.sqlite3"))
    assert rows[0]["created_at"] == received_at


# ---------------------------------------------------------------------------
# Audio uploads
# ---------------------------------------------------------------------------
def test_uploaded_clip_is_stored_by_default(isolated_feedback_client, tmp_path) -> None:
    with NIGHTJAR_CLIP.open("rb") as handle:
        response = isolated_feedback_client.post(
            "/v1/feedback",
            data={"corrected_text": "Caprimulgus europaeus"},
            files={"file": ("clip.mp3", handle.read(), "audio/mpeg")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["stored_audio"] is True

    clips_dir = tmp_path / "feedback_clips"
    saved = list(clips_dir.glob("*.mp3"))
    assert len(saved) == 1
    assert saved[0].stat().st_size == NIGHTJAR_CLIP.stat().st_size


def test_without_a_file_stored_audio_is_false(isolated_feedback_client) -> None:
    response = isolated_feedback_client.post(
        "/v1/feedback", data={"corrected_text": "Rana temporaria"}
    )
    assert response.json()["stored_audio"] is False


def test_feedback_store_audio_false_discards_the_upload(
    isolated_feedback_client, monkeypatch, settings, tmp_path
) -> None:
    from wildecho_api import main

    no_audio = settings.model_copy(
        update={
            "feedback_store_audio": False,
            "feedback_db_path": tmp_path / "feedback.sqlite3",
            "feedback_clips_dir": tmp_path / "feedback_clips",
        }
    )
    monkeypatch.setattr(main, "get_settings", lambda: no_audio)

    with NIGHTJAR_CLIP.open("rb") as handle:
        response = isolated_feedback_client.post(
            "/v1/feedback",
            data={"corrected_text": "Caprimulgus europaeus"},
            files={"file": ("clip.mp3", handle.read(), "audio/mpeg")},
        )
    assert response.status_code == 200
    assert response.json()["stored_audio"] is False
    assert not (tmp_path / "feedback_clips").exists() or not list(
        (tmp_path / "feedback_clips").glob("*")
    )


# ---------------------------------------------------------------------------
# Disabled / unavailable
# ---------------------------------------------------------------------------
def test_disabled_returns_404(isolated_feedback_client, monkeypatch, settings) -> None:
    from wildecho_api import main

    disabled = settings.model_copy(update={"feedback_enabled": False})
    monkeypatch.setattr(main, "get_settings", lambda: disabled)

    response = isolated_feedback_client.post("/v1/feedback", data={"corrected_text": "x"})
    assert response.status_code == 404
    assert response.json()["error"] == "feedback_disabled"


def test_store_unavailable_returns_503(isolated_feedback_client, monkeypatch) -> None:
    from wildecho_api import main

    monkeypatch.setattr(main.state, "feedback_store", None)
    monkeypatch.setattr(main.state, "feedback_error", "disk full (test)")

    response = isolated_feedback_client.post("/v1/feedback", data={"corrected_text": "x"})
    assert response.status_code == 503
    assert response.json()["error"] == "feedback_unavailable"
    assert "disk full" in response.json()["detail"]


# ---------------------------------------------------------------------------
# /v1/health reflects feedback state
# ---------------------------------------------------------------------------
def test_health_reports_feedback_enabled_and_ready(isolated_feedback_client) -> None:
    body = isolated_feedback_client.get("/v1/health").json()
    assert body["feedback_enabled"] is True
    assert body["feedback_store_ready"] is True


def test_health_reports_feedback_disabled(client, monkeypatch, settings) -> None:
    from wildecho_api import main

    disabled = settings.model_copy(update={"feedback_enabled": False})
    monkeypatch.setattr(main, "get_settings", lambda: disabled)

    body = client.get("/v1/health").json()
    assert body["feedback_enabled"] is False


# ---------------------------------------------------------------------------
# Matching helper, exercised directly
# ---------------------------------------------------------------------------
def test_match_correction_returns_none_for_blank_text(taxonomy) -> None:
    from wildecho_api.feedback import match_correction

    assert match_correction("   ", taxonomy) is None


def test_match_correction_prefers_scientific_over_common_name(taxonomy) -> None:
    """If free text happens to collide with both, scientific name wins (checked first)."""
    from wildecho_api.feedback import match_correction

    result = match_correction("Turdus migratorius", taxonomy)
    assert result is not None
    assert result.scientific_name == "Turdus migratorius"
    assert result.class_index >= 0
