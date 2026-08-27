"""Typed HTTP request and response contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from modal_native_test_stack_poc.inference import (
    EntityCollection,
    ImageClassification,
    ImageEmbedding,
    ModelStatus,
    Sentiment,
    Summary,
    TextEmbedding,
    Transcription,
)

from .ports import SearchHit, StoredAsset


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextRequest(StrictModel):
    text: str = Field(min_length=1)


class SummaryRequest(TextRequest):
    max_new_tokens: int = Field(default=64, ge=1, le=256)
    num_beams: int = Field(default=2, ge=1, le=8)


class TextAssetRequest(TextRequest):
    title: str | None = Field(default=None, max_length=200)


class SearchRequest(TextRequest):
    limit: int = Field(default=10, ge=1, le=50)


class EmbeddingResponse(StrictModel):
    vector: list[float]
    dimensions: int
    normalized: bool
    model_key: str

    @classmethod
    def from_text(cls, result: TextEmbedding) -> EmbeddingResponse:
        return cls(**result.to_dict())

    @classmethod
    def from_image(cls, result: ImageEmbedding) -> EmbeddingResponse:
        return cls(**result.to_dict())


class SentimentResponse(StrictModel):
    label: str
    score: float
    model_key: str

    @classmethod
    def from_result(cls, result: Sentiment) -> SentimentResponse:
        return cls(**result.to_dict())


class EntityResponse(StrictModel):
    text: str
    label: str
    score: float
    start: int | None = None
    end: int | None = None


class EntitiesResponse(StrictModel):
    entities: list[EntityResponse]
    model_key: str

    @classmethod
    def from_result(cls, result: EntityCollection) -> EntitiesResponse:
        return cls(**result.to_dict())


class SummaryResponse(StrictModel):
    text: str
    model_key: str

    @classmethod
    def from_result(cls, result: Summary) -> SummaryResponse:
        return cls(**result.to_dict())


class LabelPredictionResponse(StrictModel):
    label: str
    score: float


class ImageClassificationResponse(StrictModel):
    predictions: list[LabelPredictionResponse]
    labels: list[str]
    model_key: str

    @classmethod
    def from_result(cls, result: ImageClassification) -> ImageClassificationResponse:
        return cls(**result.to_dict())


class TranscriptionResponse(StrictModel):
    text: str
    duration_seconds: float
    sample_rate: int
    model_key: str

    @classmethod
    def from_result(cls, result: Transcription) -> TranscriptionResponse:
        return cls(**result.to_dict())


class ModelStatusResponse(StrictModel):
    key: str
    repo_id: str
    revision: str
    task: str
    commit_pinned: bool
    available: bool
    loaded: bool

    @classmethod
    def from_status(cls, status: ModelStatus) -> ModelStatusResponse:
        return cls(
            key=status.key,
            repo_id=status.repo_id,
            revision=status.revision,
            task=status.task,
            commit_pinned=status.commit_pinned,
            available=status.available,
            loaded=status.loaded,
        )


class HealthResponse(StrictModel):
    status: str
    checks: dict[str, bool] = Field(default_factory=dict)


class AssetResponse(StrictModel):
    id: UUID
    kind: str
    title: str | None
    content_text: str
    content_sha256: str
    metadata: dict[str, Any]
    analysis: dict[str, Any]
    created_at: datetime

    @classmethod
    def from_asset(cls, asset: StoredAsset) -> AssetResponse:
        return cls(
            id=asset.id,
            kind=asset.kind.value,
            title=asset.title,
            content_text=asset.content_text,
            content_sha256=asset.content_sha256,
            metadata=asset.metadata,
            analysis=asset.analysis,
            created_at=asset.created_at,
        )


class SearchHitResponse(StrictModel):
    id: UUID
    kind: str
    title: str | None
    content_text: str
    score: float

    @classmethod
    def from_hit(cls, hit: SearchHit) -> SearchHitResponse:
        return cls(
            id=hit.id,
            kind=hit.kind.value,
            title=hit.title,
            content_text=hit.content_text,
            score=hit.score,
        )


class SearchResponse(StrictModel):
    hits: list[SearchHitResponse]
