from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pytest

from tests.conftest import value_of

pytestmark = pytest.mark.xdist_group(name="text")


def _vector(output: Any) -> list[float]:
    candidate = (
        output
        if isinstance(output, Sequence) and not isinstance(output, str)
        else value_of(output, "vector", "embedding")
    )
    return [float(item) for item in candidate]


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm)


def _label(output: Any) -> str:
    return str(value_of(output, "label")).upper()


def _score(output: Any) -> float:
    return float(value_of(output, "score", "confidence"))


def _entities(output: Any) -> list[Any]:
    if isinstance(output, list):
        return output
    return list(value_of(output, "entities", "predictions"))


@pytest.mark.model
def test_text_embedding_has_minilm_dimension(registry, positive_text: str) -> None:
    assert len(_vector(registry.embed_text(positive_text))) == 384


@pytest.mark.model
def test_text_embedding_contains_only_finite_numbers(registry, positive_text: str) -> None:
    assert all(math.isfinite(value) for value in _vector(registry.embed_text(positive_text)))


@pytest.mark.model
def test_text_embedding_is_unit_normalized(registry, positive_text: str) -> None:
    vector = _vector(registry.embed_text(positive_text))
    norm = math.sqrt(sum(value * value for value in vector))
    assert norm == pytest.approx(1.0, abs=1e-4)


@pytest.mark.model
def test_text_embedding_is_deterministic(registry, positive_text: str) -> None:
    first = _vector(registry.embed_text(positive_text))
    second = _vector(registry.embed_text(positive_text))
    assert first == pytest.approx(second, abs=1e-7)


@pytest.mark.model
def test_related_text_is_closer_than_unrelated_text(registry) -> None:
    anchor = _vector(registry.embed_text("A dog is running through a grassy park."))
    related = _vector(registry.embed_text("A puppy runs outside on the grass."))
    unrelated = _vector(registry.embed_text("A database transaction was rolled back."))
    assert _cosine(anchor, related) > _cosine(anchor, unrelated) + 0.15


@pytest.mark.model
def test_sentiment_recognizes_clearly_positive_text(registry, positive_text: str) -> None:
    output = registry.sentiment(positive_text)
    assert "POSITIVE" in _label(output)
    assert 0.8 <= _score(output) <= 1.0


@pytest.mark.model
def test_sentiment_recognizes_clearly_negative_text(registry, negative_text: str) -> None:
    output = registry.sentiment(negative_text)
    assert "NEGATIVE" in _label(output)
    assert 0.8 <= _score(output) <= 1.0


@pytest.mark.model
def test_sentiment_score_is_a_probability(registry, positive_text: str) -> None:
    assert 0.0 <= _score(registry.sentiment(positive_text)) <= 1.0


@pytest.mark.model
def test_named_entity_model_finds_people_and_places(registry, entity_text: str) -> None:
    entities = _entities(registry.named_entities(entity_text))
    labels = {
        str(value_of(entity, "label", "entity_group", "entity")).upper() for entity in entities
    }
    assert any("PER" in label for label in labels)
    assert any("LOC" in label for label in labels)


@pytest.mark.model
def test_named_entity_model_returns_nonempty_source_spans(registry, entity_text: str) -> None:
    entities = _entities(registry.named_entities(entity_text))
    texts = [str(value_of(entity, "text", "word")) for entity in entities]
    assert texts
    assert all(text.strip() for text in texts)
    assert any("Obama" in text or "Barack" in text for text in texts)


@pytest.mark.model
def test_named_entity_scores_are_probabilities(registry, entity_text: str) -> None:
    entities = _entities(registry.named_entities(entity_text))
    scores = [float(value_of(entity, "score", "confidence")) for entity in entities]
    assert scores
    assert all(0.0 <= score <= 1.0 for score in scores)


@pytest.mark.model
def test_summary_is_nonempty(registry, summary_text: str) -> None:
    output = registry.summarize(summary_text)
    assert str(value_of(output, "text", "summary")).strip()


@pytest.mark.model
def test_summary_is_shorter_than_source(registry, summary_text: str) -> None:
    output = registry.summarize(summary_text)
    summary = str(value_of(output, "text", "summary"))
    assert len(summary.split()) < len(summary_text.split())


@pytest.mark.model
def test_summary_mentions_the_subject(registry, summary_text: str) -> None:
    output = registry.summarize(summary_text)
    summary = str(value_of(output, "text", "summary")).lower()
    assert any(term in summary for term in ("modal", "cloud", "python", "model"))


@pytest.mark.model
def test_summary_generation_is_deterministic(registry, summary_text: str) -> None:
    first = str(value_of(registry.summarize(summary_text), "text", "summary"))
    second = str(value_of(registry.summarize(summary_text), "text", "summary"))
    assert first == second
