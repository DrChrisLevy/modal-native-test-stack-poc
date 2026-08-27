from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

EXPECTED_MODELS = {
    "text_embedding": (
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence_embedding",
    ),
    "sentiment": (
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "sentiment_analysis",
    ),
    "named_entities": ("dslim/bert-base-NER", "token_classification"),
    "summary": ("google/flan-t5-small", "text2text_generation"),
    "image_classification": ("microsoft/resnet-18", "image_classification"),
    "image_embedding": ("openai/clip-vit-base-patch32", "clip_embedding"),
    "speech_to_text": ("openai/whisper-tiny.en", "automatic_speech_recognition"),
}


@pytest.fixture(scope="module")
def manifest(project_root: Path) -> dict[str, object]:
    return json.loads((project_root / "models.lock.json").read_text())


def test_model_manifest_has_a_version(manifest: dict[str, object]) -> None:
    assert manifest["schema_version"] == 1


def test_model_manifest_has_exactly_the_public_capabilities(manifest: dict[str, object]) -> None:
    models = manifest["models"]
    assert isinstance(models, list)
    assert {model["key"] for model in models} == set(EXPECTED_MODELS)


@pytest.mark.parametrize(("key", "expected"), EXPECTED_MODELS.items())
def test_model_repository_and_task_are_locked(
    manifest: dict[str, object], key: str, expected: tuple[str, str]
) -> None:
    models = manifest["models"]
    assert isinstance(models, list)
    model = next(candidate for candidate in models if candidate["key"] == key)
    assert (model["repo_id"], model["task"]) == expected


@pytest.mark.parametrize("required_key", ["key", "repo_id", "revision", "task"])
def test_every_model_entry_has_required_fields(
    manifest: dict[str, object], required_key: str
) -> None:
    models = manifest["models"]
    assert isinstance(models, list)
    assert all(required_key in model for model in models)


@pytest.mark.parametrize("field", ["key", "repo_id", "revision", "task"])
def test_model_manifest_strings_are_nonempty(manifest: dict[str, object], field: str) -> None:
    models = manifest["models"]
    assert isinstance(models, list)
    assert all(isinstance(model[field], str) and model[field].strip() for model in models)


def test_model_keys_are_unique(manifest: dict[str, object]) -> None:
    models = manifest["models"]
    assert isinstance(models, list)
    keys = [model["key"] for model in models]
    assert len(keys) == len(set(keys))


def test_model_repositories_are_unique(manifest: dict[str, object]) -> None:
    models = manifest["models"]
    assert isinstance(models, list)
    repositories = [model["repo_id"] for model in models]
    assert len(repositories) == len(set(repositories))


def test_every_model_revision_is_an_immutable_commit(manifest: dict[str, object]) -> None:
    models = manifest["models"]
    assert isinstance(models, list)
    assert all(re.fullmatch(r"[0-9a-f]{40}", model["revision"]) for model in models)


def test_project_does_not_define_docker_compose(project_root: Path) -> None:
    forbidden = {
        "compose.yml",
        "compose.yaml",
        "docker-compose.yml",
        "docker-compose.yaml",
    }
    assert forbidden.isdisjoint(path.name for path in project_root.iterdir())


def test_project_does_not_define_a_dockerfile(project_root: Path) -> None:
    assert not any(path.name.lower().startswith("dockerfile") for path in project_root.iterdir())


def test_project_does_not_define_github_actions(project_root: Path) -> None:
    assert not (project_root / ".github" / "workflows").exists()


def test_project_never_defines_mock_model_implementations(project_root: Path) -> None:
    python_files = [
        *(project_root / "src").rglob("*.py"),
        *(project_root / "tests").rglob("*.py"),
    ]
    forbidden_names = {"FakeModel", "MockModel", "fake_embedding", "mock_prediction"}
    offenders: list[tuple[Path, str]] = []
    for path in python_files:
        tree = ast.parse(path.read_text())
        offenders.extend(
            (path, node.name)
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name in forbidden_names
        )
    assert offenders == []
