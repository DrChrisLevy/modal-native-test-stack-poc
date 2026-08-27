"""PostgreSQL, Redis, and OpenSearch adapters."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import UUID

from .ports import AssetKind, IndexedAsset, SearchHit, StoredAsset


class PostgresAssetRepository:
    """Small asyncpg repository; PostgreSQL remains the source of truth."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._pool: Any | None = None

    async def initialize(self) -> None:
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                dsn=self.url,
                min_size=1,
                max_size=4,
                command_timeout=30,
            )
        await self._pool.execute(
            """
            CREATE TABLE IF NOT EXISTS assets (
                id UUID PRIMARY KEY,
                kind TEXT NOT NULL CHECK (kind IN ('text', 'image', 'audio')),
                title TEXT,
                content_text TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                metadata JSONB NOT NULL,
                analysis JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL
            )
            """
        )
        await self._pool.execute(
            "CREATE INDEX IF NOT EXISTS assets_created_at_idx ON assets (created_at DESC)"
        )
        await self._pool.execute(
            "CREATE INDEX IF NOT EXISTS assets_content_sha_idx ON assets (content_sha256)"
        )

    async def ping(self) -> bool:
        try:
            pool = self._require_pool()
            return await pool.fetchval("SELECT 1") == 1
        except Exception:
            return False

    async def save(self, asset: StoredAsset) -> None:
        pool = self._require_pool()
        await pool.execute(
            """
            INSERT INTO assets (
                id, kind, title, content_text, content_sha256,
                metadata, analysis, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8)
            ON CONFLICT (id) DO UPDATE SET
                kind = EXCLUDED.kind,
                title = EXCLUDED.title,
                content_text = EXCLUDED.content_text,
                content_sha256 = EXCLUDED.content_sha256,
                metadata = EXCLUDED.metadata,
                analysis = EXCLUDED.analysis
            """,
            asset.id,
            asset.kind.value,
            asset.title,
            asset.content_text,
            asset.content_sha256,
            json.dumps(asset.metadata, separators=(",", ":"), sort_keys=True),
            json.dumps(asset.analysis, separators=(",", ":"), sort_keys=True),
            asset.created_at,
        )

    async def get(self, asset_id: UUID) -> StoredAsset | None:
        pool = self._require_pool()
        row = await pool.fetchrow(
            """
            SELECT id, kind, title, content_text, content_sha256,
                   metadata, analysis, created_at
            FROM assets
            WHERE id = $1
            """,
            asset_id,
        )
        if row is None:
            return None
        return StoredAsset(
            id=row["id"],
            kind=AssetKind(row["kind"]),
            title=row["title"],
            content_text=row["content_text"],
            content_sha256=row["content_sha256"],
            metadata=_json_object(row["metadata"]),
            analysis=_json_object(row["analysis"]),
            created_at=row["created_at"],
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _require_pool(self) -> Any:
        if self._pool is None:
            raise RuntimeError("PostgresAssetRepository.initialize() has not been called")
        return self._pool


class RedisJsonCache:
    """JSON cache with explicit model-versioned keys supplied by the service."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._client: Any | None = None

    async def initialize(self) -> None:
        if self._client is None:
            from redis.asyncio import Redis

            self._client = Redis.from_url(self.url, decode_responses=True)

    async def ping(self) -> bool:
        try:
            return bool(await self._require_client().ping())
        except Exception:
            return False

    async def get_json(self, key: str) -> dict[str, Any] | None:
        value = await self._require_client().get(key)
        if value is None:
            return None
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise TypeError(f"cached value for {key!r} is not a JSON object")
        return parsed

    async def set_json(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        encoded = json.dumps(value, separators=(",", ":"), sort_keys=True)
        await self._require_client().set(key, encoded, ex=ttl_seconds)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _require_client(self) -> Any:
        if self._client is None:
            raise RuntimeError("RedisJsonCache.initialize() has not been called")
        return self._client


class OpenSearchAssetIndex:
    """OpenSearch-backed lexical and vector index."""

    def __init__(self, url: str, index_name: str, *, verify_certs: bool = False) -> None:
        self.url = url
        self.index_name = index_name
        self.verify_certs = verify_certs
        self._client: Any | None = None

    async def initialize(self, *, text_dimensions: int, image_dimensions: int) -> None:
        if self._client is None:
            from opensearchpy import AsyncOpenSearch

            parsed = urlparse(self.url)
            host: dict[str, Any] = {
                "host": parsed.hostname or "opensearch",
                "port": parsed.port or (443 if parsed.scheme == "https" else 9200),
                "scheme": parsed.scheme or "http",
            }
            kwargs: dict[str, Any] = {
                "hosts": [host],
                "verify_certs": self.verify_certs,
                "ssl_show_warn": False,
            }
            if parsed.username is not None:
                kwargs["http_auth"] = (
                    unquote(parsed.username),
                    unquote(parsed.password or ""),
                )
            self._client = AsyncOpenSearch(**kwargs)

        client = self._require_client()
        if not await client.indices.exists(index=self.index_name):
            await client.indices.create(
                index=self.index_name,
                body={
                    "settings": {"index": {"knn": True}},
                    "mappings": {
                        "dynamic": "strict",
                        "properties": {
                            "id": {"type": "keyword"},
                            "kind": {"type": "keyword"},
                            "title": {"type": "text"},
                            "content_text": {"type": "text"},
                            "text_embedding": {
                                "type": "knn_vector",
                                "dimension": text_dimensions,
                                "method": {
                                    "name": "hnsw",
                                    "space_type": "cosinesimil",
                                    "engine": "lucene",
                                },
                            },
                            "image_embedding": {
                                "type": "knn_vector",
                                "dimension": image_dimensions,
                                "method": {
                                    "name": "hnsw",
                                    "space_type": "cosinesimil",
                                    "engine": "lucene",
                                },
                            },
                        },
                    },
                },
            )

    async def ping(self) -> bool:
        try:
            return bool(await self._require_client().ping())
        except Exception:
            return False

    async def index(self, asset: IndexedAsset) -> None:
        document: dict[str, Any] = {
            "id": str(asset.id),
            "kind": asset.kind.value,
            "title": asset.title,
            "content_text": asset.content_text,
        }
        if asset.text_embedding is not None:
            document["text_embedding"] = list(asset.text_embedding)
        if asset.image_embedding is not None:
            document["image_embedding"] = list(asset.image_embedding)
        await self._require_client().index(
            index=self.index_name,
            id=str(asset.id),
            body=document,
            refresh="wait_for",
        )

    async def search(
        self,
        query: str,
        *,
        text_embedding: tuple[float, ...],
        limit: int,
    ) -> tuple[SearchHit, ...]:
        body = {
            "size": limit,
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^2", "content_text"],
                            }
                        },
                        {
                            "knn": {
                                "text_embedding": {
                                    "vector": list(text_embedding),
                                    "k": max(limit, 10),
                                }
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            },
        }
        response = await self._require_client().search(index=self.index_name, body=body)
        hits: list[SearchHit] = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            hits.append(
                SearchHit(
                    id=UUID(source["id"]),
                    kind=AssetKind(source["kind"]),
                    title=source.get("title"),
                    content_text=source["content_text"],
                    score=float(hit.get("_score") or 0.0),
                )
            )
        return tuple(hits)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _require_client(self) -> Any:
        if self._client is None:
            raise RuntimeError("OpenSearchAssetIndex.initialize() has not been called")
        return self._client


def _json_object(value: Any) -> dict[str, Any]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise TypeError("database JSON value is not an object")
    return parsed
