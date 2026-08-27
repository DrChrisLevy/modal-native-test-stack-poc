from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from modal_native_test_stack_poc.application.settings import ApplicationSettings, Settings


@pytest.fixture(autouse=True)
def _isolate_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in tuple(os.environ):
        if key.upper().startswith("MULTIMODAL_"):
            monkeypatch.delenv(key, raising=False)


def test_settings_alias_is_public() -> None:
    assert Settings is ApplicationSettings


def test_settings_have_remote_workspace_defaults() -> None:
    settings = ApplicationSettings()
    assert settings.models_lock_path == Path("/workspace/models.lock.json")
    assert settings.models_root == Path("/models")


def test_settings_default_to_cpu() -> None:
    assert ApplicationSettings().model_device == "cpu"


def test_settings_require_immutable_model_pins_by_default() -> None:
    assert ApplicationSettings().require_commit_pins is True


def test_settings_default_embedding_dimensions_match_models() -> None:
    settings = ApplicationSettings()
    assert settings.text_embedding_dimensions == 384
    assert settings.image_embedding_dimensions == 512


@pytest.mark.parametrize(
    ("variable", "attribute", "value", "expected"),
    [
        ("MULTIMODAL_APP_NAME", "app_name", "Remote Lab", "Remote Lab"),
        ("MULTIMODAL_MODEL_DEVICE", "model_device", "cuda", "cuda"),
        ("MULTIMODAL_CACHE_NAMESPACE", "cache_namespace", "test-worker", "test-worker"),
        ("MULTIMODAL_CACHE_TTL_SECONDS", "cache_ttl_seconds", "90", 90),
        ("MULTIMODAL_TEXT_EMBEDDING_DIMENSIONS", "text_embedding_dimensions", "16", 16),
        ("MULTIMODAL_MAXIMUM_UPLOAD_BYTES", "maximum_upload_bytes", "1024", 1024),
    ],
)
def test_settings_read_prefixed_environment(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    attribute: str,
    value: str,
    expected: object,
) -> None:
    monkeypatch.setenv(variable, value)
    assert getattr(ApplicationSettings(), attribute) == expected


def test_settings_environment_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("multimodal_app_name", "lowercase")
    assert ApplicationSettings().app_name == "lowercase"


def test_unrelated_environment_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOMETHING_UNRELATED", "value")
    assert ApplicationSettings().app_name == "Multimodal Asset API"


@pytest.mark.parametrize("cache_ttl", [0, -1, 86_401])
def test_cache_ttl_is_bounded(cache_ttl: int) -> None:
    with pytest.raises(ValidationError):
        ApplicationSettings(cache_ttl_seconds=cache_ttl)


@pytest.mark.parametrize(
    "field",
    [
        "text_embedding_dimensions",
        "image_embedding_dimensions",
        "maximum_text_characters",
        "maximum_upload_bytes",
    ],
)
def test_positive_numeric_settings_reject_zero(field: str) -> None:
    with pytest.raises(ValidationError):
        ApplicationSettings(**{field: 0})


def test_service_urls_can_be_replaced_as_a_group() -> None:
    settings = ApplicationSettings(
        postgres_url="postgresql://db/app",
        redis_url="redis://cache:6379/1",
        opensearch_url="http://search:9200",
    )
    assert settings.postgres_url.endswith("db/app")
    assert settings.redis_url.endswith("6379/1")
    assert settings.opensearch_url.endswith("9200")
