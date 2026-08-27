from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from modal_native_test_stack_poc.application.adapters import (
    OpenSearchAssetIndex,
    PostgresAssetRepository,
    RedisJsonCache,
    _json_object,
)
from modal_native_test_stack_poc.application.ports import AssetKind, IndexedAsset, StoredAsset


def _stored_asset() -> StoredAsset:
    return StoredAsset(
        id=uuid4(),
        kind=AssetKind.TEXT,
        title="Contract",
        content_text="content",
        content_sha256="a" * 64,
        metadata={},
        analysis={},
        created_at=datetime.now(UTC),
    )


def test_postgres_repository_retains_url() -> None:
    repository = PostgresAssetRepository("postgresql://db/app")
    assert repository.url == "postgresql://db/app"


@pytest.mark.asyncio
async def test_postgres_ping_before_initialize_is_false() -> None:
    assert await PostgresAssetRepository("postgresql://db/app").ping() is False


@pytest.mark.asyncio
async def test_postgres_save_requires_initialize() -> None:
    repository = PostgresAssetRepository("postgresql://db/app")
    with pytest.raises(RuntimeError, match="initialize"):
        await repository.save(_stored_asset())


@pytest.mark.asyncio
async def test_postgres_get_requires_initialize() -> None:
    repository = PostgresAssetRepository("postgresql://db/app")
    with pytest.raises(RuntimeError, match="initialize"):
        await repository.get(uuid4())


@pytest.mark.asyncio
async def test_postgres_close_before_initialize_is_safe() -> None:
    await PostgresAssetRepository("postgresql://db/app").close()


def test_redis_cache_retains_url() -> None:
    cache = RedisJsonCache("redis://cache:6379/0")
    assert cache.url == "redis://cache:6379/0"


@pytest.mark.asyncio
async def test_redis_ping_before_initialize_is_false() -> None:
    assert await RedisJsonCache("redis://cache:6379/0").ping() is False


@pytest.mark.asyncio
async def test_redis_read_requires_initialize() -> None:
    cache = RedisJsonCache("redis://cache:6379/0")
    with pytest.raises(RuntimeError, match="initialize"):
        await cache.get_json("key")


@pytest.mark.asyncio
async def test_redis_write_requires_initialize() -> None:
    cache = RedisJsonCache("redis://cache:6379/0")
    with pytest.raises(RuntimeError, match="initialize"):
        await cache.set_json("key", {}, ttl_seconds=10)


@pytest.mark.asyncio
async def test_redis_close_before_initialize_is_safe() -> None:
    await RedisJsonCache("redis://cache:6379/0").close()


def test_opensearch_adapter_retains_connection_contract() -> None:
    index = OpenSearchAssetIndex("https://user:password@search:9200", "assets")
    assert index.url.startswith("https://")
    assert index.index_name == "assets"
    assert index.verify_certs is False


@pytest.mark.asyncio
async def test_opensearch_ping_before_initialize_is_false() -> None:
    index = OpenSearchAssetIndex("http://search:9200", "assets")
    assert await index.ping() is False


@pytest.mark.asyncio
async def test_opensearch_index_requires_initialize() -> None:
    index = OpenSearchAssetIndex("http://search:9200", "assets")
    with pytest.raises(RuntimeError, match="initialize"):
        await index.index(
            IndexedAsset(
                id=uuid4(),
                kind=AssetKind.TEXT,
                title=None,
                content_text="content",
            )
        )


@pytest.mark.asyncio
async def test_opensearch_search_requires_initialize() -> None:
    index = OpenSearchAssetIndex("http://search:9200", "assets")
    with pytest.raises(RuntimeError, match="initialize"):
        await index.search("query", text_embedding=(1.0,), limit=1)


@pytest.mark.asyncio
async def test_opensearch_close_before_initialize_is_safe() -> None:
    await OpenSearchAssetIndex("http://search:9200", "assets").close()


def test_json_object_accepts_mapping() -> None:
    assert _json_object({"key": "value"}) == {"key": "value"}


def test_json_object_decodes_database_json_string() -> None:
    assert _json_object('{"key":"value"}') == {"key": "value"}


@pytest.mark.parametrize("value", [[], "[]", 1, "null"])
def test_json_object_rejects_non_objects(value) -> None:
    with pytest.raises(TypeError, match="not an object"):
        _json_object(value)
