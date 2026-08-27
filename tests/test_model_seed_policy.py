from __future__ import annotations

import pytest

from modal_native_test_stack_poc.remote.seed_worker import (
    COMMON_SNAPSHOT_IGNORE_PATTERNS,
    _snapshot_ignore_patterns,
)


def test_clip_seed_policy_retains_its_only_pytorch_weights() -> None:
    assert "pytorch_model.bin" not in _snapshot_ignore_patterns("image_embedding")


@pytest.mark.parametrize(
    "model_key",
    [
        "text_embedding",
        "sentiment",
        "named_entities",
        "summary",
        "image_classification",
        "speech_to_text",
    ],
)
def test_non_clip_seed_policy_excludes_duplicate_pytorch_weights(model_key: str) -> None:
    assert "pytorch_model.bin" in _snapshot_ignore_patterns(model_key)


@pytest.mark.parametrize("model_key", ["image_embedding", "text_embedding", "speech_to_text"])
def test_every_seed_policy_excludes_nonruntime_artifacts(model_key: str) -> None:
    patterns = _snapshot_ignore_patterns(model_key)
    assert set(COMMON_SNAPSHOT_IGNORE_PATTERNS).issubset(patterns)
    assert "onnx/*" in patterns
    assert "openvino/*" in patterns


@pytest.mark.parametrize("model_key", ["image_embedding", "summary"])
def test_seed_policy_never_excludes_safetensors(model_key: str) -> None:
    patterns = _snapshot_ignore_patterns(model_key)
    assert "*.safetensors" not in patterns
    assert "model.safetensors" not in patterns


def test_seed_policy_returns_a_fresh_mutable_list() -> None:
    first = _snapshot_ignore_patterns("summary")
    first.append("new-pattern")
    assert "new-pattern" not in _snapshot_ignore_patterns("summary")
