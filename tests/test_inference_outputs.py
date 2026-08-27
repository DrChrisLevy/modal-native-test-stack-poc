from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from modal_native_test_stack_poc.inference import (
    EntityCollection,
    ImageAnalysis,
    ImageClassification,
    ImageEmbedding,
    LabelPrediction,
    NamedEntity,
    Sentiment,
    Summary,
    TextAnalysis,
    TextEmbedding,
    Transcription,
)


def _text_analysis() -> TextAnalysis:
    return TextAnalysis(
        embedding=TextEmbedding(vector=(0.6, 0.8)),
        sentiment=Sentiment(label="POSITIVE", score=0.98),
        entities=EntityCollection(
            entities=(NamedEntity(text="Modal", label="ORG", score=0.99, start=0, end=5),)
        ),
        summary=Summary(text="Modal runs Python in the cloud."),
    )


def _image_analysis() -> ImageAnalysis:
    return ImageAnalysis(
        classification=ImageClassification(
            predictions=(
                LabelPrediction(label="monitor", score=0.7),
                LabelPrediction(label="screen", score=0.2),
            )
        ),
        embedding=ImageEmbedding(vector=(0.0, 1.0, 0.0)),
    )


def test_text_embedding_reports_dimensions() -> None:
    assert TextEmbedding(vector=(0.1, 0.2, 0.3)).dimensions == 3


def test_text_embedding_serializes_tuple_as_json_list() -> None:
    assert TextEmbedding(vector=(0.1, 0.2)).to_dict()["vector"] == [0.1, 0.2]


def test_text_embedding_round_trips() -> None:
    value = TextEmbedding(vector=(0.6, 0.8))
    assert TextEmbedding.from_dict(value.to_dict()) == value


def test_text_embedding_defaults_to_normalized() -> None:
    assert TextEmbedding(vector=(1.0,)).normalized is True


def test_sentiment_round_trips() -> None:
    value = Sentiment(label="NEGATIVE", score=0.97)
    assert Sentiment.from_dict(value.to_dict()) == value


def test_sentiment_casts_numeric_score() -> None:
    assert Sentiment.from_dict({"label": "POSITIVE", "score": "0.75"}).score == 0.75


@pytest.mark.parametrize(("start", "end"), [(None, None), (0, 5), (12, 18)])
def test_named_entity_round_trips_optional_spans(start: int | None, end: int | None) -> None:
    value = NamedEntity(text="Modal", label="ORG", score=0.9, start=start, end=end)
    assert NamedEntity.from_dict(value.to_dict()) == value


def test_entity_collection_is_iterable() -> None:
    entity = NamedEntity(text="Paris", label="LOC", score=0.9)
    assert list(EntityCollection(entities=(entity,))) == [entity]


def test_entity_collection_reports_length() -> None:
    entities = EntityCollection(
        entities=(
            NamedEntity(text="Paris", label="LOC", score=0.9),
            NamedEntity(text="France", label="LOC", score=0.8),
        )
    )
    assert len(entities) == 2


def test_entity_collection_round_trips() -> None:
    value = _text_analysis().entities
    assert EntityCollection.from_dict(value.to_dict()) == value


def test_summary_round_trips() -> None:
    value = Summary(text="A short summary.")
    assert Summary.from_dict(value.to_dict()) == value


def test_label_prediction_round_trips() -> None:
    value = LabelPrediction(label="tabby", score=0.85)
    assert LabelPrediction.from_dict(value.to_dict()) == value


def test_image_classification_exposes_labels_in_rank_order() -> None:
    value = _image_analysis().classification
    assert value.labels == ("monitor", "screen")


def test_image_classification_round_trips() -> None:
    value = _image_analysis().classification
    assert ImageClassification.from_dict(value.to_dict()) == value


def test_image_embedding_reports_dimensions() -> None:
    assert ImageEmbedding(vector=(0.0, 1.0, 0.0)).dimensions == 3


def test_image_embedding_round_trips() -> None:
    value = ImageEmbedding(vector=(0.0, 1.0, 0.0))
    assert ImageEmbedding.from_dict(value.to_dict()) == value


def test_transcription_round_trips() -> None:
    value = Transcription(text="hello", duration_seconds=1.25)
    assert Transcription.from_dict(value.to_dict()) == value


def test_transcription_defaults_to_whisper_sample_rate() -> None:
    assert Transcription(text="", duration_seconds=1.0).sample_rate == 16_000


def test_text_analysis_round_trips_nested_contract() -> None:
    value = _text_analysis()
    assert TextAnalysis.from_dict(value.to_dict()) == value


def test_image_analysis_round_trips_nested_contract() -> None:
    value = _image_analysis()
    assert ImageAnalysis.from_dict(value.to_dict()) == value


@pytest.mark.parametrize(
    "value",
    [
        TextEmbedding(vector=(1.0,)),
        Sentiment(label="POSITIVE", score=1.0),
        Summary(text="summary"),
        ImageEmbedding(vector=(1.0,)),
        Transcription(text="words", duration_seconds=0.5),
    ],
)
def test_public_inference_results_are_immutable(value) -> None:
    with pytest.raises((FrozenInstanceError, AttributeError)):
        value.model_key = "changed"
