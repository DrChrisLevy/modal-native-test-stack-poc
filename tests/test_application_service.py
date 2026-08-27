from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from modal_native_test_stack_poc.application import (
    ApplicationSettings,
    MultimodalService,
    OpenSearchAssetIndex,
    PostgresAssetRepository,
    RedisJsonCache,
    build_service,
)
from modal_native_test_stack_poc.application.service import (
    _clean_title,
    _sha256,
    project_default_lockfile,
)
from modal_native_test_stack_poc.inference import ModelRegistry


@pytest.fixture
def unstarted_service(models_lock_path: Path, models_root: Path) -> MultimodalService:
    settings = ApplicationSettings(
        models_lock_path=models_lock_path,
        models_root=models_root,
        require_commit_pins=True,
        maximum_text_characters=20,
        maximum_upload_bytes=10,
    )
    return MultimodalService(
        registry=ModelRegistry.from_lockfile(
            models_lock_path,
            models_root=models_root,
            require_commit_pins=True,
        ),
        repository=PostgresAssetRepository(settings.postgres_url),
        cache=RedisJsonCache(settings.redis_url),
        search_index=OpenSearchAssetIndex(settings.opensearch_url, settings.opensearch_index),
        settings=settings,
    )


def test_build_service_does_not_connect_or_load_models(
    models_lock_path: Path, models_root: Path
) -> None:
    service = build_service(
        ApplicationSettings(
            models_lock_path=models_lock_path,
            models_root=models_root,
            require_commit_pins=True,
        )
    )
    assert service.registry.loaded_keys == frozenset()
    assert service.repository._pool is None  # type: ignore[attr-defined]
    assert service.cache._client is None  # type: ignore[attr-defined]
    assert service.search_index._client is None  # type: ignore[attr-defined]


def test_text_validation_strips_outer_whitespace(unstarted_service: MultimodalService) -> None:
    assert unstarted_service._validate_text("  useful text \n") == "useful text"


@pytest.mark.parametrize("text", ["", " ", "\n\t"])
def test_text_validation_rejects_blank_input(
    unstarted_service: MultimodalService, text: str
) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        unstarted_service._validate_text(text)


def test_text_validation_rejects_oversized_input(
    unstarted_service: MultimodalService,
) -> None:
    with pytest.raises(ValueError, match="exceeds 20"):
        unstarted_service._validate_text("x" * 21)


def test_text_validation_accepts_exact_limit(unstarted_service: MultimodalService) -> None:
    assert unstarted_service._validate_text("x" * 20) == "x" * 20


def test_upload_validation_rejects_empty_content(unstarted_service: MultimodalService) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        unstarted_service._validate_upload(b"")


def test_upload_validation_rejects_oversized_content(
    unstarted_service: MultimodalService,
) -> None:
    with pytest.raises(ValueError, match="exceeds 10"):
        unstarted_service._validate_upload(b"x" * 11)


def test_upload_validation_accepts_exact_limit(unstarted_service: MultimodalService) -> None:
    unstarted_service._validate_upload(b"x" * 10)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, None), ("", None), ("   ", None), (" title ", "title")],
)
def test_title_cleaning(raw: str | None, expected: str | None) -> None:
    assert _clean_title(raw) == expected


def test_content_digest_is_stable_sha256() -> None:
    assert _sha256(b"modal") == hashlib.sha256(b"modal").hexdigest()


def test_cache_key_includes_every_model_revision(
    unstarted_service: MultimodalService,
) -> None:
    key = unstarted_service._cache_key("text", "digest", ("text_embedding", "sentiment"))
    assert key.startswith("modal-native-test-stack-poc:v1:text:digest:")
    assert "text_embedding@" in key
    assert "sentiment@" in key


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, 51])
async def test_search_limit_is_checked_before_model_inference(
    unstarted_service: MultimodalService, limit: int
) -> None:
    with pytest.raises(ValueError, match="limit must be between"):
        await unstarted_service.search("valid", limit=limit)
    assert unstarted_service.registry.loaded_keys == frozenset()


def test_default_lockfile_path_points_at_project_file(project_root: Path) -> None:
    assert project_default_lockfile() == project_root / "models.lock.json"
