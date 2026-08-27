from __future__ import annotations

from pathlib import Path

import pytest

from modal_native_test_stack_poc.inference import ModelRegistry


def test_registry_constructs_without_loading_ml_frameworks(
    models_lock_path: Path, models_root: Path
) -> None:
    registry = ModelRegistry.from_lockfile(models_lock_path, models_root=models_root)
    assert registry.loaded_keys == frozenset()


def test_registry_exposes_all_seven_specs(registry: ModelRegistry) -> None:
    assert len(registry.specs) == 7


def test_registry_exposes_each_capability_key(registry: ModelRegistry) -> None:
    assert {spec.key for spec in registry.specs} == {
        "text_embedding",
        "sentiment",
        "named_entities",
        "summary",
        "image_classification",
        "image_embedding",
        "speech_to_text",
    }


@pytest.mark.parametrize(
    "key",
    [
        "text_embedding",
        "sentiment",
        "named_entities",
        "summary",
        "image_classification",
        "image_embedding",
        "speech_to_text",
    ],
)
def test_registry_get_spec_round_trips_key(registry: ModelRegistry, key: str) -> None:
    assert registry.get_spec(key).key == key


def test_registry_unknown_capability_is_actionable(registry: ModelRegistry) -> None:
    with pytest.raises(KeyError, match="unknown model capability"):
        registry.get_spec("not_a_model")


def test_registry_status_is_ordered_like_manifest(registry: ModelRegistry) -> None:
    assert [status.key for status in registry.status()] == [spec.key for spec in registry.specs]


def test_registry_status_starts_unloaded(models_lock_path: Path, models_root: Path) -> None:
    fresh = ModelRegistry.from_lockfile(models_lock_path, models_root=models_root)
    assert all(status.loaded is False for status in fresh.status())


@pytest.mark.parametrize("text", ["", " ", "\n\t"])
def test_text_inference_rejects_blank_input_without_loading_model(
    models_lock_path: Path, models_root: Path, text: str
) -> None:
    fresh = ModelRegistry.from_lockfile(models_lock_path, models_root=models_root)
    with pytest.raises(ValueError, match="must not be empty"):
        fresh.embed_text(text)
    assert fresh.loaded_keys == frozenset()


@pytest.mark.parametrize("max_new_tokens", [0, 257])
def test_summary_token_limit_is_validated_before_model_load(
    models_lock_path: Path, models_root: Path, max_new_tokens: int
) -> None:
    fresh = ModelRegistry.from_lockfile(models_lock_path, models_root=models_root)
    with pytest.raises(ValueError, match="max_new_tokens"):
        fresh.summarize("valid text", max_new_tokens=max_new_tokens)
    assert fresh.loaded_keys == frozenset()


@pytest.mark.parametrize("num_beams", [0, 9])
def test_summary_beam_count_is_validated_before_model_load(
    models_lock_path: Path, models_root: Path, num_beams: int
) -> None:
    fresh = ModelRegistry.from_lockfile(models_lock_path, models_root=models_root)
    with pytest.raises(ValueError, match="num_beams"):
        fresh.summarize("valid text", num_beams=num_beams)
    assert fresh.loaded_keys == frozenset()


@pytest.mark.parametrize("top_k", [0, 21])
def test_classifier_top_k_is_validated_before_model_load(
    models_lock_path: Path, models_root: Path, top_k: int
) -> None:
    fresh = ModelRegistry.from_lockfile(models_lock_path, models_root=models_root)
    with pytest.raises(ValueError, match="top_k"):
        fresh.classify_image(b"not-decoded", top_k=top_k)
    assert fresh.loaded_keys == frozenset()


@pytest.mark.model
@pytest.mark.xdist_group(name="text")
def test_registry_can_unload_one_real_model(registry: ModelRegistry, positive_text: str) -> None:
    registry.embed_text(positive_text)
    assert "text_embedding" in registry.loaded_keys
    registry.unload("text_embedding")
    assert "text_embedding" not in registry.loaded_keys
