"""Application records and infrastructure boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol
from uuid import UUID


class AssetKind(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"


@dataclass(frozen=True, slots=True)
class StoredAsset:
    id: UUID
    kind: AssetKind
    title: str | None
    content_text: str
    content_sha256: str
    metadata: dict[str, Any]
    analysis: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class IndexedAsset:
    id: UUID
    kind: AssetKind
    title: str | None
    content_text: str
    text_embedding: tuple[float, ...] | None = None
    image_embedding: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class SearchHit:
    id: UUID
    kind: AssetKind
    title: str | None
    content_text: str
    score: float


class AssetRepository(Protocol):
    async def initialize(self) -> None: ...

    async def ping(self) -> bool: ...

    async def save(self, asset: StoredAsset) -> None: ...

    async def get(self, asset_id: UUID) -> StoredAsset | None: ...

    async def close(self) -> None: ...


class JsonCache(Protocol):
    async def initialize(self) -> None: ...

    async def ping(self) -> bool: ...

    async def get_json(self, key: str) -> dict[str, Any] | None: ...

    async def set_json(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None: ...

    async def close(self) -> None: ...


class SearchIndex(Protocol):
    async def initialize(self, *, text_dimensions: int, image_dimensions: int) -> None: ...

    async def ping(self) -> bool: ...

    async def index(self, asset: IndexedAsset) -> None: ...

    async def search(
        self,
        query: str,
        *,
        text_embedding: tuple[float, ...],
        limit: int,
    ) -> tuple[SearchHit, ...]: ...

    async def close(self) -> None: ...
