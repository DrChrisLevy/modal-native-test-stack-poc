from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from modal_native_test_stack_poc.application.ports import AssetKind, SearchHit, StoredAsset
from modal_native_test_stack_poc.application.schemas import (
    AssetResponse,
    EmbeddingResponse,
    EntitiesResponse,
    ImageClassificationResponse,
    SearchHitResponse,
    SearchRequest,
    SentimentResponse,
    SummaryRequest,
    SummaryResponse,
    TextAssetRequest,
    TextRequest,
    TranscriptionResponse,
)
from modal_native_test_stack_poc.inference import (
    EntityCollection,
    ImageClassification,
    ImageEmbedding,
    LabelPrediction,
    NamedEntity,
    Sentiment,
    Summary,
    TextEmbedding,
    Transcription,
)


def test_text_request_accepts_nonempty_text() -> None:
    assert TextRequest(text="hello").text == "hello"


def test_text_request_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        TextRequest(text="")


def test_request_models_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        TextRequest(text="valid", unexpected=True)  # type: ignore[call-arg]


@pytest.mark.parametrize("max_new_tokens", [0, 257])
def test_summary_request_bounds_tokens(max_new_tokens: int) -> None:
    with pytest.raises(ValidationError):
        SummaryRequest(text="valid", max_new_tokens=max_new_tokens)


@pytest.mark.parametrize("num_beams", [0, 9])
def test_summary_request_bounds_beams(num_beams: int) -> None:
    with pytest.raises(ValidationError):
        SummaryRequest(text="valid", num_beams=num_beams)


def test_text_asset_title_is_bounded() -> None:
    with pytest.raises(ValidationError):
        TextAssetRequest(text="valid", title="x" * 201)


@pytest.mark.parametrize("limit", [0, 51])
def test_search_request_bounds_limit(limit: int) -> None:
    with pytest.raises(ValidationError):
        SearchRequest(text="valid", limit=limit)


def test_text_embedding_response_converts_typed_output() -> None:
    response = EmbeddingResponse.from_text(TextEmbedding(vector=(0.6, 0.8)))
    assert response.vector == [0.6, 0.8]
    assert response.dimensions == 2
    assert response.model_key == "text_embedding"


def test_image_embedding_response_converts_typed_output() -> None:
    response = EmbeddingResponse.from_image(ImageEmbedding(vector=(0.0, 1.0, 0.0)))
    assert response.dimensions == 3
    assert response.model_key == "image_embedding"


def test_sentiment_response_converts_typed_output() -> None:
    response = SentimentResponse.from_result(Sentiment(label="POSITIVE", score=0.9))
    assert response.label == "POSITIVE"
    assert response.score == 0.9


def test_entity_response_converts_nested_collection() -> None:
    response = EntitiesResponse.from_result(
        EntityCollection(entities=(NamedEntity("Paris", "LOC", 0.99, start=0, end=5),))
    )
    assert response.entities[0].text == "Paris"
    assert response.entities[0].start == 0


def test_summary_response_converts_typed_output() -> None:
    assert SummaryResponse.from_result(Summary("short")).text == "short"


def test_image_classification_response_keeps_labels_aligned() -> None:
    response = ImageClassificationResponse.from_result(
        ImageClassification(
            predictions=(
                LabelPrediction("first", 0.8),
                LabelPrediction("second", 0.1),
            )
        )
    )
    assert response.labels == ["first", "second"]
    assert [prediction.label for prediction in response.predictions] == response.labels


def test_transcription_response_converts_typed_output() -> None:
    response = TranscriptionResponse.from_result(Transcription(text="hello", duration_seconds=1.0))
    assert response.text == "hello"
    assert response.sample_rate == 16_000


def test_asset_response_converts_domain_record() -> None:
    asset = StoredAsset(
        id=uuid4(),
        kind=AssetKind.TEXT,
        title="Title",
        content_text="content",
        content_sha256="a" * 64,
        metadata={"characters": 7},
        analysis={"summary": {"text": "content"}},
        created_at=datetime.now(UTC),
    )
    response = AssetResponse.from_asset(asset)
    assert response.id == asset.id
    assert response.kind == "text"
    assert response.analysis == asset.analysis


def test_search_hit_response_converts_domain_record() -> None:
    hit = SearchHit(
        id=uuid4(),
        kind=AssetKind.IMAGE,
        title="Scene",
        content_text="colors",
        score=1.2,
    )
    response = SearchHitResponse.from_hit(hit)
    assert response.id == hit.id
    assert response.kind == "image"
    assert response.score == 1.2
