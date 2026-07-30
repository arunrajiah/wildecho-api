"""Settings loading from environment variables."""

from __future__ import annotations

from pathlib import Path

import pytest

from wildecho_api.config import (
    Settings,
    default_feedback_clips_dir,
    default_feedback_db_path,
    get_settings,
)


def _env_settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    for key, value in env.items():
        monkeypatch.setenv(f"WILDECHO_{key.upper()}", value)
    get_settings.cache_clear()
    try:
        return get_settings()
    finally:
        get_settings.cache_clear()


def test_defaults_require_no_environment_variables() -> None:
    settings = Settings()
    assert settings.top_k == 10
    assert settings.low_confidence_threshold == 0.3
    assert settings.feedback_enabled is True
    assert settings.log_format == "text"


def test_calibration_path_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    custom = tmp_path / "custom_calibration.yaml"
    settings = _env_settings(monkeypatch, calibration_path=str(custom))
    assert settings.calibration_path == custom


def test_feedback_enabled_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _env_settings(monkeypatch, feedback_enabled="false").feedback_enabled is False
    assert _env_settings(monkeypatch, feedback_enabled="true").feedback_enabled is True


def test_feedback_db_path_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    custom = tmp_path / "somewhere" / "feedback.sqlite3"
    settings = _env_settings(monkeypatch, feedback_db_path=str(custom))
    assert settings.feedback_db_path == custom


def test_feedback_store_audio_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _env_settings(monkeypatch, feedback_store_audio="false").feedback_store_audio is False


def test_feedback_clips_dir_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    custom = tmp_path / "clips"
    settings = _env_settings(monkeypatch, feedback_clips_dir=str(custom))
    assert settings.feedback_clips_dir == custom


def test_feedback_max_upload_bytes_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _env_settings(monkeypatch, feedback_max_upload_bytes="1024")
    assert settings.feedback_max_upload_bytes == 1024


def test_log_format_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _env_settings(monkeypatch, log_format="json").log_format == "json"
    assert _env_settings(monkeypatch, log_format="JSON").log_format == "json"


def test_log_format_rejects_invalid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WILDECHO_LOG_FORMAT", "xml")
    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="log_format"):
            get_settings()
    finally:
        monkeypatch.delenv("WILDECHO_LOG_FORMAT", raising=False)
        get_settings.cache_clear()


def test_request_id_header_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _env_settings(monkeypatch, request_id_header="X-Correlation-Id")
    assert settings.request_id_header == "X-Correlation-Id"


def test_settings_are_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Regression: default feedback paths must never depend on where the package
# happens to be installed (e.g. inside a virtualenv's site-packages). A prior
# version resolved them relative to the installed package location, which is
# unwritable in a wheel install and crashed the whole app at startup rather than
# degrading gracefully - see CHANGELOG "Fixed" / commit history.
# ---------------------------------------------------------------------------
def test_default_feedback_db_path_is_cwd_relative_not_package_relative() -> None:
    path = default_feedback_db_path()
    assert not path.is_absolute()
    assert "site-packages" not in str(path)
    assert path == Path("data") / "feedback.sqlite3"


def test_default_feedback_clips_dir_is_cwd_relative_not_package_relative() -> None:
    path = default_feedback_clips_dir()
    assert not path.is_absolute()
    assert "site-packages" not in str(path)
    assert path == Path("data") / "feedback_clips"
