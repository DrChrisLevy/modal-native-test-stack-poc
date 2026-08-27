from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import value_of


def _vector(output: Any) -> list[float]:
    candidate = (
        output
        if isinstance(output, Sequence) and not isinstance(output, str)
        else value_of(output, "vector", "embedding")
    )
    return [float(item) for item in candidate]


def _predictions(output: Any) -> list[Any]:
    if isinstance(output, list):
        return output
    return list(value_of(output, "predictions", "labels"))


def _prediction_label(prediction: Any) -> str:
    return str(value_of(prediction, "label", "class_name"))


def _prediction_score(prediction: Any) -> float:
    return float(value_of(prediction, "score", "confidence"))


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        math.sqrt(sum(value * value for value in left))
        * math.sqrt(sum(value * value for value in right))
    )


@pytest.mark.model
def test_clip_image_embedding_has_expected_dimension(registry, generated_image_path: Path) -> None:
    assert len(_vector(registry.embed_image(generated_image_path))) == 512


@pytest.mark.model
def test_clip_image_embedding_contains_only_finite_values(
    registry, generated_image_path: Path
) -> None:
    vector = _vector(registry.embed_image(generated_image_path))
    assert all(math.isfinite(value) for value in vector)


@pytest.mark.model
def test_clip_image_embedding_is_unit_normalized(registry, generated_image_path: Path) -> None:
    vector = _vector(registry.embed_image(generated_image_path))
    assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.model
def test_clip_image_embedding_is_deterministic(registry, generated_image_path: Path) -> None:
    first = _vector(registry.embed_image(generated_image_path))
    second = _vector(registry.embed_image(generated_image_path))
    assert first == pytest.approx(second, abs=1e-7)


@pytest.mark.model
def test_clip_distinguishes_generated_images(
    registry, generated_image_path: Path, alternate_image_path: Path
) -> None:
    first = _vector(registry.embed_image(generated_image_path))
    alternate = _vector(registry.embed_image(alternate_image_path))
    assert _cosine(first, alternate) < 0.999


@pytest.mark.model
def test_resnet_returns_ranked_predictions(registry, generated_image_path: Path) -> None:
    predictions = _predictions(registry.classify_image(generated_image_path))
    assert predictions
    scores = [_prediction_score(prediction) for prediction in predictions]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.model
def test_resnet_prediction_labels_are_nonempty(registry, generated_image_path: Path) -> None:
    predictions = _predictions(registry.classify_image(generated_image_path))
    assert all(_prediction_label(prediction).strip() for prediction in predictions)


@pytest.mark.model
def test_resnet_prediction_scores_are_probabilities(registry, generated_image_path: Path) -> None:
    predictions = _predictions(registry.classify_image(generated_image_path))
    assert all(0.0 <= _prediction_score(prediction) <= 1.0 for prediction in predictions)


@pytest.mark.model
def test_resnet_top_prediction_is_deterministic(registry, generated_image_path: Path) -> None:
    first = _predictions(registry.classify_image(generated_image_path))
    second = _predictions(registry.classify_image(generated_image_path))
    assert _prediction_label(first[0]) == _prediction_label(second[0])
    assert _prediction_score(first[0]) == pytest.approx(_prediction_score(second[0]), abs=1e-7)


@pytest.mark.model
def test_resnet_prediction_labels_are_unique(registry, alternate_image_path: Path) -> None:
    predictions = _predictions(registry.classify_image(alternate_image_path))
    labels = [_prediction_label(prediction) for prediction in predictions]
    assert len(labels) == len(set(labels))
