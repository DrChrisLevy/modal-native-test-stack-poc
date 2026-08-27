from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest


def _redis_url() -> str:
    url = os.getenv("MODAL_ML_REDIS_URL") or os.getenv("REDIS_URL")
    if not url:
        pytest.skip("Redis Sidecar URL is not configured")
    return url


@pytest.fixture
async def redis_client() -> AsyncIterator[Any]:
    redis = pytest.importorskip("redis.asyncio")
    client = redis.from_url(_redis_url(), decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def redis_key() -> str:
    return f"modal-native-test-stack-poc:test:{uuid.uuid4().hex}"


@pytest.mark.services
async def test_redis_sidecar_responds_to_ping(redis_client) -> None:
    assert await redis_client.ping() is True


@pytest.mark.services
async def test_redis_round_trips_unicode(redis_client, redis_key: str) -> None:
    try:
        await redis_client.set(redis_key, "Modal says hello ☁")
        assert await redis_client.get(redis_key) == "Modal says hello ☁"
    finally:
        await redis_client.delete(redis_key)


@pytest.mark.services
async def test_redis_nx_enforces_idempotency(redis_client, redis_key: str) -> None:
    try:
        assert await redis_client.set(redis_key, "first", nx=True) is True
        assert await redis_client.set(redis_key, "second", nx=True) is None
        assert await redis_client.get(redis_key) == "first"
    finally:
        await redis_client.delete(redis_key)


@pytest.mark.services
async def test_redis_expiry_is_attached_atomically(redis_client, redis_key: str) -> None:
    try:
        await redis_client.set(redis_key, "cached", ex=30)
        ttl = await redis_client.ttl(redis_key)
        assert 0 < ttl <= 30
    finally:
        await redis_client.delete(redis_key)


@pytest.mark.services
async def test_redis_hash_round_trip(redis_client, redis_key: str) -> None:
    expected = {"model": "clip", "revision": "abc123", "dimensions": "512"}
    try:
        await redis_client.hset(redis_key, mapping=expected)
        assert await redis_client.hgetall(redis_key) == expected
    finally:
        await redis_client.delete(redis_key)


@pytest.mark.services
async def test_redis_list_preserves_queue_order(redis_client, redis_key: str) -> None:
    try:
        await redis_client.rpush(redis_key, "first", "second", "third")
        assert await redis_client.lrange(redis_key, 0, -1) == ["first", "second", "third"]
    finally:
        await redis_client.delete(redis_key)


@pytest.mark.services
async def test_redis_increment_is_atomic(redis_client, redis_key: str) -> None:
    try:
        values = [await redis_client.incr(redis_key) for _ in range(5)]
        assert values == [1, 2, 3, 4, 5]
    finally:
        await redis_client.delete(redis_key)
