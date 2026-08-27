"""Application orchestration over real models and real Sidecar services."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from modal_native_test_stack_poc.inference import (
    EntityCollection,
    ImageAnalysis,
    ImageClassification,
    ImageEmbedding,
    ModelRegistry,
    ModelStatus,
    Sentiment,
    Summary,
    TextAnalysis,
    TextEmbedding,
    Transcription,
)

from .ports import (
    AssetKind,
    AssetRepository,
    IndexedAsset,
    JsonCache,
    SearchHit,
    SearchIndex,
    StoredAsset,
)
from .settings import ApplicationSettings


class MultimodalService:
    """Use case layer shared by HTTP, tests, shells, and coding agents."""

    def __init__(
        self,
        *,
        registry: ModelRegistry,
        repository: AssetRepository,
        cache: JsonCache,
        search_index: SearchIndex,
        settings: ApplicationSettings,
    ) -> None:
        self.registry = registry
        self.repository = repository
        self.cache = cache
        self.search_index = search_index
        self.settings = settings

    async def startup(self) -> None:
        await asyncio.gather(
            self.repository.initialize(),
            self.cache.initialize(),
            self.search_index.initialize(
                text_dimensions=self.settings.text_embedding_dimensions,
                image_dimensions=self.settings.image_embedding_dimensions,
            ),
        )

    async def close(self) -> None:
        await asyncio.gather(
            self.search_index.close(),
            self.cache.close(),
            self.repository.close(),
            return_exceptions=True,
        )

    async def readiness(self) -> dict[str, bool]:
        postgres, redis, opensearch = await asyncio.gather(
            self.repository.ping(),
            self.cache.ping(),
            self.search_index.ping(),
        )
        return {
            "model_snapshots": self.registry.all_snapshots_available(),
            "model_revisions_pinned": self.registry.manifest.all_commit_pinned,
            "postgres": postgres,
            "redis": redis,
            "opensearch": opensearch,
        }

    def model_status(self) -> tuple[ModelStatus, ...]:
        return self.registry.status()

    async def embed_text(self, text: str) -> TextEmbedding:
        return await asyncio.to_thread(self.registry.embed_text, self._validate_text(text))

    async def predict_sentiment(self, text: str) -> Sentiment:
        return await asyncio.to_thread(self.registry.sentiment, self._validate_text(text))

    async def extract_entities(self, text: str) -> EntityCollection:
        return await asyncio.to_thread(self.registry.named_entities, self._validate_text(text))

    async def summarize(
        self, text: str, *, max_new_tokens: int = 64, num_beams: int = 2
    ) -> Summary:
        cleaned = self._validate_text(text)
        return await asyncio.to_thread(
            self.registry.summarize,
            cleaned,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

    async def classify_image(self, image: bytes, *, top_k: int = 5) -> ImageClassification:
        self._validate_upload(image)
        return await asyncio.to_thread(self.registry.classify_image, image, top_k=top_k)

    async def embed_image(self, image: bytes) -> ImageEmbedding:
        self._validate_upload(image)
        return await asyncio.to_thread(self.registry.embed_image, image)

    async def transcribe(self, audio: bytes) -> Transcription:
        self._validate_upload(audio)
        return await asyncio.to_thread(self.registry.transcribe, audio)

    async def analyze_text(self, text: str) -> TextAnalysis:
        cleaned = self._validate_text(text)
        digest = _sha256(cleaned.encode())
        cache_key = self._cache_key(
            "text",
            digest,
            ("text_embedding", "sentiment", "named_entities", "summary"),
        )
        cached = await self.cache.get_json(cache_key)
        if cached is not None:
            return TextAnalysis.from_dict(cached)

        # Loading sequentially keeps peak RAM predictable. Modal parallelizes model
        # lanes across separate Sandboxes rather than forcing every model into one.
        analysis = TextAnalysis(
            embedding=await self.embed_text(cleaned),
            sentiment=await self.predict_sentiment(cleaned),
            entities=await self.extract_entities(cleaned),
            summary=await self.summarize(cleaned),
        )
        await self.cache.set_json(
            cache_key,
            analysis.to_dict(),
            ttl_seconds=self.settings.cache_ttl_seconds,
        )
        return analysis

    async def analyze_image(self, image: bytes, *, top_k: int = 5) -> ImageAnalysis:
        self._validate_upload(image)
        digest = _sha256(image)
        cache_key = self._cache_key(
            f"image:top_k={top_k}",
            digest,
            ("image_classification", "image_embedding"),
        )
        cached = await self.cache.get_json(cache_key)
        if cached is not None:
            return ImageAnalysis.from_dict(cached)
        analysis = ImageAnalysis(
            classification=await self.classify_image(image, top_k=top_k),
            embedding=await self.embed_image(image),
        )
        await self.cache.set_json(
            cache_key,
            analysis.to_dict(),
            ttl_seconds=self.settings.cache_ttl_seconds,
        )
        return analysis

    async def cached_transcription(self, audio: bytes) -> Transcription:
        self._validate_upload(audio)
        digest = _sha256(audio)
        cache_key = self._cache_key("audio", digest, ("speech_to_text",))
        cached = await self.cache.get_json(cache_key)
        if cached is not None:
            return Transcription.from_dict(cached)
        transcription = await self.transcribe(audio)
        await self.cache.set_json(
            cache_key,
            transcription.to_dict(),
            ttl_seconds=self.settings.cache_ttl_seconds,
        )
        return transcription

    async def create_text_asset(self, text: str, *, title: str | None = None) -> StoredAsset:
        cleaned = self._validate_text(text)
        analysis = await self.analyze_text(cleaned)
        asset = StoredAsset(
            id=uuid4(),
            kind=AssetKind.TEXT,
            title=_clean_title(title),
            content_text=cleaned,
            content_sha256=_sha256(cleaned.encode()),
            metadata={"characters": len(cleaned)},
            analysis=analysis.to_dict(),
            created_at=datetime.now(UTC),
        )
        await self.repository.save(asset)
        await self.search_index.index(
            IndexedAsset(
                id=asset.id,
                kind=asset.kind,
                title=asset.title,
                content_text=asset.content_text,
                text_embedding=analysis.embedding.vector,
            )
        )
        return asset

    async def create_image_asset(
        self,
        image: bytes,
        *,
        title: str | None = None,
        content_type: str | None = None,
        top_k: int = 5,
    ) -> StoredAsset:
        analysis = await self.analyze_image(image, top_k=top_k)
        labels = ", ".join(analysis.classification.labels)
        asset = StoredAsset(
            id=uuid4(),
            kind=AssetKind.IMAGE,
            title=_clean_title(title),
            content_text=labels,
            content_sha256=_sha256(image),
            metadata={"bytes": len(image), "content_type": content_type},
            analysis=analysis.to_dict(),
            created_at=datetime.now(UTC),
        )
        await self.repository.save(asset)
        await self.search_index.index(
            IndexedAsset(
                id=asset.id,
                kind=asset.kind,
                title=asset.title,
                content_text=asset.content_text,
                image_embedding=analysis.embedding.vector,
            )
        )
        return asset

    async def create_audio_asset(
        self,
        audio: bytes,
        *,
        title: str | None = None,
        content_type: str | None = None,
    ) -> StoredAsset:
        transcription = await self.cached_transcription(audio)
        analysis: dict[str, Any] = {"transcription": transcription.to_dict()}
        text_embedding: tuple[float, ...] | None = None
        if transcription.text:
            text_analysis = await self.analyze_text(transcription.text)
            analysis["text"] = text_analysis.to_dict()
            text_embedding = text_analysis.embedding.vector
        asset = StoredAsset(
            id=uuid4(),
            kind=AssetKind.AUDIO,
            title=_clean_title(title),
            content_text=transcription.text,
            content_sha256=_sha256(audio),
            metadata={
                "bytes": len(audio),
                "content_type": content_type,
                "duration_seconds": transcription.duration_seconds,
            },
            analysis=analysis,
            created_at=datetime.now(UTC),
        )
        await self.repository.save(asset)
        await self.search_index.index(
            IndexedAsset(
                id=asset.id,
                kind=asset.kind,
                title=asset.title,
                content_text=asset.content_text,
                text_embedding=text_embedding,
            )
        )
        return asset

    async def get_asset(self, asset_id: UUID) -> StoredAsset | None:
        return await self.repository.get(asset_id)

    async def search(self, query: str, *, limit: int = 10) -> tuple[SearchHit, ...]:
        cleaned = self._validate_text(query)
        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50")
        embedding = await self.embed_text(cleaned)
        return await self.search_index.search(
            cleaned,
            text_embedding=embedding.vector,
            limit=limit,
        )

    def _validate_text(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("text must not be empty")
        if len(cleaned) > self.settings.maximum_text_characters:
            raise ValueError(f"text exceeds {self.settings.maximum_text_characters} characters")
        return cleaned

    def _validate_upload(self, content: bytes) -> None:
        if not content:
            raise ValueError("upload must not be empty")
        if len(content) > self.settings.maximum_upload_bytes:
            raise ValueError(f"upload exceeds {self.settings.maximum_upload_bytes} bytes")

    def _cache_key(self, kind: str, digest: str, model_keys: tuple[str, ...]) -> str:
        revisions = ":".join(f"{key}@{self.registry.get_spec(key).revision}" for key in model_keys)
        return f"modal-native-test-stack-poc:v1:{kind}:{digest}:{revisions}"


def build_service(settings: ApplicationSettings | None = None) -> MultimodalService:
    """Construct adapters without connecting or loading models."""

    from .adapters import OpenSearchAssetIndex, PostgresAssetRepository, RedisJsonCache

    resolved = settings or ApplicationSettings()
    registry = ModelRegistry.from_lockfile(
        resolved.models_lock_path,
        models_root=resolved.models_root,
        device=resolved.model_device,
        require_commit_pins=resolved.require_commit_pins,
    )
    return MultimodalService(
        registry=registry,
        repository=PostgresAssetRepository(resolved.postgres_url),
        cache=RedisJsonCache(resolved.redis_url),
        search_index=OpenSearchAssetIndex(
            resolved.opensearch_url,
            resolved.opensearch_index,
            verify_certs=resolved.opensearch_verify_certs,
        ),
        settings=resolved,
    )


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _clean_title(title: str | None) -> str | None:
    if title is None:
        return None
    cleaned = title.strip()
    return cleaned or None


def project_default_lockfile() -> Path:
    """Convenience path for non-Modal diagnostics from the repository root."""

    return Path(__file__).resolve().parents[3] / "models.lock.json"
