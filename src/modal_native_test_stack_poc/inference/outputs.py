"""Typed, serialization-friendly inference results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Self


@dataclass(frozen=True, slots=True)
class TextEmbedding:
    """A normalized sentence embedding."""

    vector: tuple[float, ...]
    model_key: str = "text_embedding"
    normalized: bool = True

    @property
    def dimensions(self) -> int:
        return len(self.vector)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector": list(self.vector),
            "model_key": self.model_key,
            "normalized": self.normalized,
            "dimensions": self.dimensions,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        return cls(
            vector=tuple(float(item) for item in value["vector"]),
            model_key=str(value.get("model_key", "text_embedding")),
            normalized=bool(value.get("normalized", True)),
        )


@dataclass(frozen=True, slots=True)
class Sentiment:
    label: str
    score: float
    model_key: str = "sentiment"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        return cls(
            label=str(value["label"]),
            score=float(value["score"]),
            model_key=str(value.get("model_key", "sentiment")),
        )


@dataclass(frozen=True, slots=True)
class NamedEntity:
    text: str
    label: str
    score: float
    start: int | None = None
    end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        return cls(
            text=str(value["text"]),
            label=str(value["label"]),
            score=float(value["score"]),
            start=int(value["start"]) if value.get("start") is not None else None,
            end=int(value["end"]) if value.get("end") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class EntityCollection:
    entities: tuple[NamedEntity, ...]
    model_key: str = "named_entities"

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.entities)

    def __len__(self) -> int:
        return len(self.entities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": [entity.to_dict() for entity in self.entities],
            "model_key": self.model_key,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        return cls(
            entities=tuple(NamedEntity.from_dict(item) for item in value["entities"]),
            model_key=str(value.get("model_key", "named_entities")),
        )


@dataclass(frozen=True, slots=True)
class Summary:
    text: str
    model_key: str = "summary"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        return cls(text=str(value["text"]), model_key=str(value.get("model_key", "summary")))


@dataclass(frozen=True, slots=True)
class LabelPrediction:
    label: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        return cls(label=str(value["label"]), score=float(value["score"]))


@dataclass(frozen=True, slots=True)
class ImageClassification:
    predictions: tuple[LabelPrediction, ...]
    model_key: str = "image_classification"

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(prediction.label for prediction in self.predictions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "labels": list(self.labels),
            "model_key": self.model_key,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        return cls(
            predictions=tuple(LabelPrediction.from_dict(item) for item in value["predictions"]),
            model_key=str(value.get("model_key", "image_classification")),
        )


@dataclass(frozen=True, slots=True)
class ImageEmbedding:
    vector: tuple[float, ...]
    model_key: str = "image_embedding"
    normalized: bool = True

    @property
    def dimensions(self) -> int:
        return len(self.vector)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector": list(self.vector),
            "model_key": self.model_key,
            "normalized": self.normalized,
            "dimensions": self.dimensions,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        return cls(
            vector=tuple(float(item) for item in value["vector"]),
            model_key=str(value.get("model_key", "image_embedding")),
            normalized=bool(value.get("normalized", True)),
        )


@dataclass(frozen=True, slots=True)
class Transcription:
    text: str
    duration_seconds: float
    sample_rate: int = 16_000
    model_key: str = "speech_to_text"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        return cls(
            text=str(value["text"]),
            duration_seconds=float(value["duration_seconds"]),
            sample_rate=int(value.get("sample_rate", 16_000)),
            model_key=str(value.get("model_key", "speech_to_text")),
        )


@dataclass(frozen=True, slots=True)
class TextAnalysis:
    embedding: TextEmbedding
    sentiment: Sentiment
    entities: EntityCollection
    summary: Summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "embedding": self.embedding.to_dict(),
            "sentiment": self.sentiment.to_dict(),
            "entities": self.entities.to_dict(),
            "summary": self.summary.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        return cls(
            embedding=TextEmbedding.from_dict(value["embedding"]),
            sentiment=Sentiment.from_dict(value["sentiment"]),
            entities=EntityCollection.from_dict(value["entities"]),
            summary=Summary.from_dict(value["summary"]),
        )


@dataclass(frozen=True, slots=True)
class ImageAnalysis:
    classification: ImageClassification
    embedding: ImageEmbedding

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification.to_dict(),
            "embedding": self.embedding.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        return cls(
            classification=ImageClassification.from_dict(value["classification"]),
            embedding=ImageEmbedding.from_dict(value["embedding"]),
        )
