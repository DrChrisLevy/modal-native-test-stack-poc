from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest


def _postgres_url() -> str:
    url = (
        os.getenv("MODAL_ML_POSTGRES_URL") or os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
    )
    if not url:
        pytest.skip("PostgreSQL Sidecar URL is not configured")
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


@pytest.fixture
async def pg_connection() -> AsyncIterator[Any]:
    asyncpg = pytest.importorskip("asyncpg")
    connection = await asyncpg.connect(_postgres_url())
    try:
        yield connection
    finally:
        await connection.close()


@pytest.fixture
async def pg_table(pg_connection) -> AsyncIterator[str]:
    table = f"contract_{uuid.uuid4().hex}"
    await pg_connection.execute(
        f"""
        CREATE TABLE {table} (
            id BIGSERIAL PRIMARY KEY,
            external_id TEXT NOT NULL UNIQUE,
            payload JSONB NOT NULL,
            counter INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    try:
        yield table
    finally:
        await pg_connection.execute(f"DROP TABLE IF EXISTS {table}")


@pytest.mark.services
async def test_postgres_accepts_a_real_query(pg_connection) -> None:
    assert await pg_connection.fetchval("SELECT 40 + 2") == 42


@pytest.mark.services
async def test_postgres_reports_its_server_version(pg_connection) -> None:
    version = await pg_connection.fetchval("SHOW server_version")
    assert version and version[0].isdigit()


@pytest.mark.services
async def test_postgres_round_trips_jsonb(pg_connection, pg_table: str) -> None:
    payload = {"model": "all-MiniLM-L6-v2", "dimensions": 384, "ready": True}
    encoded = json.dumps(payload)
    await pg_connection.execute(
        f"INSERT INTO {pg_table} (external_id, payload) VALUES ($1, $2::jsonb)",
        "asset-1",
        encoded,
    )
    stored = await pg_connection.fetchval(
        f"SELECT payload::text FROM {pg_table} WHERE external_id = $1", "asset-1"
    )
    assert json.loads(stored) == payload


@pytest.mark.services
async def test_postgres_enforces_unique_idempotency_key(pg_connection, pg_table: str) -> None:
    asyncpg = pytest.importorskip("asyncpg")
    await pg_connection.execute(
        f"INSERT INTO {pg_table} (external_id, payload) VALUES ($1, '{{}}'::jsonb)",
        "same-key",
    )
    with pytest.raises(asyncpg.UniqueViolationError):
        await pg_connection.execute(
            f"INSERT INTO {pg_table} (external_id, payload) VALUES ($1, '{{}}'::jsonb)",
            "same-key",
        )


@pytest.mark.services
async def test_postgres_transaction_rolls_back(pg_connection, pg_table: str) -> None:
    transaction = pg_connection.transaction()
    await transaction.start()
    await pg_connection.execute(
        f"INSERT INTO {pg_table} (external_id, payload) VALUES ('temporary', '{{}}'::jsonb)"
    )
    await transaction.rollback()
    assert await pg_connection.fetchval(f"SELECT count(*) FROM {pg_table}") == 0


@pytest.mark.services
async def test_postgres_atomic_update_returns_new_value(pg_connection, pg_table: str) -> None:
    row_id = await pg_connection.fetchval(
        f"INSERT INTO {pg_table} (external_id, payload) VALUES ('counter', '{{}}'::jsonb) "
        "RETURNING id"
    )
    counter = await pg_connection.fetchval(
        f"UPDATE {pg_table} SET counter = counter + 1 WHERE id = $1 RETURNING counter", row_id
    )
    assert counter == 1


@pytest.mark.services
async def test_postgres_timestamp_is_timezone_aware(pg_connection, pg_table: str) -> None:
    created_at = await pg_connection.fetchval(
        f"INSERT INTO {pg_table} (external_id, payload) VALUES ('time', '{{}}'::jsonb) "
        "RETURNING created_at"
    )
    assert created_at.tzinfo is not None
