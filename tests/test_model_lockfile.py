from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from modal_native_test_stack_poc.inference import (
    ModelManifest,
    ModelManifestError,
    ModelSnapshotMissingError,
    ModelSpec,
    SnapshotResolver,
)
from modal_native_test_stack_poc.inference.lockfile import EXPECTED_TASKS


@pytest.fixture
def manifest_payload(project_root: Path) -> dict[str, object]:
    return json.loads((project_root / "models.lock.json").read_text())


def _write_manifest(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "models.lock.json"
    path.write_text(json.dumps(payload))
    return path


def test_loads_public_model_manifest(models_lock_path: Path) -> None:
    manifest = ModelManifest.load(models_lock_path)
    assert manifest.schema_version == 1
    assert len(manifest.models) == 7


def test_manifest_indexes_models_by_capability(models_lock_path: Path) -> None:
    manifest = ModelManifest.load(models_lock_path)
    assert set(manifest.by_key) == set(EXPECTED_TASKS)


def test_missing_lockfile_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(ModelManifestError, match="does not exist"):
        ModelManifest.load(tmp_path / "missing.json")


def test_invalid_json_has_actionable_error(tmp_path: Path) -> None:
    path = tmp_path / "models.lock.json"
    path.write_text("{ definitely not json")
    with pytest.raises(ModelManifestError, match="invalid JSON"):
        ModelManifest.load(path)


def test_manifest_root_must_be_an_object(tmp_path: Path) -> None:
    with pytest.raises(ModelManifestError, match="root must be an object"):
        ModelManifest.load(_write_manifest(tmp_path, []))


@pytest.mark.parametrize("schema_version", [None, 0, 2, "1"])
def test_rejects_unsupported_schema_versions(
    tmp_path: Path, manifest_payload: dict[str, object], schema_version: object
) -> None:
    payload = copy.deepcopy(manifest_payload)
    payload["schema_version"] = schema_version
    with pytest.raises(ModelManifestError, match="unsupported model lock schema"):
        ModelManifest.load(_write_manifest(tmp_path, payload))


@pytest.mark.parametrize("models", [None, [], {}, "models"])
def test_models_must_be_a_nonempty_list(
    tmp_path: Path, manifest_payload: dict[str, object], models: object
) -> None:
    payload = copy.deepcopy(manifest_payload)
    payload["models"] = models
    with pytest.raises(ModelManifestError, match="non-empty models list"):
        ModelManifest.load(_write_manifest(tmp_path, payload))


def test_each_model_must_be_an_object(tmp_path: Path, manifest_payload: dict[str, object]) -> None:
    payload = copy.deepcopy(manifest_payload)
    payload["models"][0] = "not-an-object"  # type: ignore[index]
    with pytest.raises(ModelManifestError, match="each models entry must be an object"):
        ModelManifest.load(_write_manifest(tmp_path, payload))


@pytest.mark.parametrize("field", ["key", "repo_id", "revision", "task"])
def test_model_entries_require_all_fields(
    tmp_path: Path, manifest_payload: dict[str, object], field: str
) -> None:
    payload = copy.deepcopy(manifest_payload)
    del payload["models"][0][field]  # type: ignore[index]
    with pytest.raises(ModelManifestError, match="missing required fields"):
        ModelManifest.load(_write_manifest(tmp_path, payload))


@pytest.mark.parametrize("key", ["UPPER", "with-dash", "1starts_wrong", "has space", "../x"])
def test_model_keys_must_be_safe(key: str) -> None:
    with pytest.raises(ModelManifestError, match="unsafe model key"):
        ModelSpec.from_dict(
            {"key": key, "repo_id": "owner/model", "revision": "main", "task": "task"}
        )


@pytest.mark.parametrize("repo_id", ["model", "too/many/slashes", "/model", "owner/"])
def test_model_repository_must_be_namespaced(repo_id: str) -> None:
    with pytest.raises(ModelManifestError, match="invalid Hugging Face repo ID"):
        ModelSpec.from_dict(
            {"key": "model", "repo_id": repo_id, "revision": "main", "task": "task"}
        )


def test_model_revision_must_be_nonempty() -> None:
    with pytest.raises(ModelManifestError, match="empty revision"):
        ModelSpec.from_dict(
            {"key": "model", "repo_id": "owner/model", "revision": "", "task": "task"}
        )


def test_model_task_must_be_nonempty() -> None:
    with pytest.raises(ModelManifestError, match="empty task"):
        ModelSpec.from_dict(
            {"key": "model", "repo_id": "owner/model", "revision": "main", "task": ""}
        )


def test_duplicate_model_keys_are_rejected(
    tmp_path: Path, manifest_payload: dict[str, object]
) -> None:
    payload = copy.deepcopy(manifest_payload)
    payload["models"].append(copy.deepcopy(payload["models"][0]))  # type: ignore[union-attr,index]
    with pytest.raises(ModelManifestError, match="model keys must be unique"):
        ModelManifest.load(_write_manifest(tmp_path, payload))


def test_missing_required_capability_is_rejected(
    tmp_path: Path, manifest_payload: dict[str, object]
) -> None:
    payload = copy.deepcopy(manifest_payload)
    payload["models"] = [  # type: ignore[index]
        model
        for model in payload["models"]
        if model["key"] != "summary"  # type: ignore[index]
    ]
    with pytest.raises(ModelManifestError, match="missing required capabilities: summary"):
        ModelManifest.load(_write_manifest(tmp_path, payload))


@pytest.mark.parametrize(("key", "task"), EXPECTED_TASKS.items())
def test_wrong_task_for_capability_is_rejected(
    tmp_path: Path,
    manifest_payload: dict[str, object],
    key: str,
    task: str,
) -> None:
    payload = copy.deepcopy(manifest_payload)
    model = next(model for model in payload["models"] if model["key"] == key)  # type: ignore[index]
    model["task"] = f"not_{task}"
    with pytest.raises(ModelManifestError, match="expected"):
        ModelManifest.load(_write_manifest(tmp_path, payload))


def test_full_sha_is_recognized_as_commit_pin() -> None:
    spec = ModelSpec("model", "owner/model", "a" * 40, "task")
    assert spec.is_commit_pinned is True


@pytest.mark.parametrize("revision", ["main", "v1.0", "abc123", "a" * 39, "g" * 40])
def test_floating_revision_is_not_a_commit_pin(revision: str) -> None:
    spec = ModelSpec("model", "owner/model", revision, "task")
    assert spec.is_commit_pinned is False


def test_manifest_can_require_commit_pins(
    tmp_path: Path, manifest_payload: dict[str, object]
) -> None:
    payload = copy.deepcopy(manifest_payload)
    for index, model in enumerate(payload["models"]):  # type: ignore[index]
        model["revision"] = f"{index + 1:040x}"
    manifest = ModelManifest.load(_write_manifest(tmp_path, payload))
    manifest.require_commit_pins()
    assert manifest.all_commit_pinned is True


def test_public_manifest_is_fully_commit_pinned(models_lock_path: Path) -> None:
    manifest = ModelManifest.load(models_lock_path)
    manifest.require_commit_pins()
    assert manifest.all_commit_pinned is True


def test_snapshot_resolver_prefers_direct_seed_layout(tmp_path: Path) -> None:
    spec = ModelSpec("model", "owner/model", "a" * 40, "task")
    snapshot = tmp_path / "model"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    assert SnapshotResolver(tmp_path).resolve(spec) == snapshot.resolve()


@pytest.mark.parametrize("cache_root", ["", "hub", ".cache"])
def test_snapshot_resolver_supports_hugging_face_cache_layout(
    tmp_path: Path, cache_root: str
) -> None:
    spec = ModelSpec("model", "owner/model", "a" * 40, "task")
    root = tmp_path / cache_root if cache_root else tmp_path
    snapshot = root / "models--owner--model" / "snapshots" / spec.revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    assert SnapshotResolver(tmp_path).resolve(spec) == snapshot.resolve()


def test_snapshot_requires_model_config(tmp_path: Path) -> None:
    spec = ModelSpec("model", "owner/model", "a" * 40, "task")
    (tmp_path / "model").mkdir()
    with pytest.raises(ModelSnapshotMissingError, match="offline snapshot"):
        SnapshotResolver(tmp_path).resolve(spec)


def test_snapshot_availability_is_false_when_missing(tmp_path: Path) -> None:
    spec = ModelSpec("model", "owner/model", "a" * 40, "task")
    assert SnapshotResolver(tmp_path).is_available(spec) is False


def test_snapshot_candidates_never_point_at_a_remote_repo(tmp_path: Path) -> None:
    spec = ModelSpec("model", "owner/model", "a" * 40, "task")
    candidates = SnapshotResolver(tmp_path).candidates(spec)
    assert all(str(candidate).startswith(str(tmp_path)) for candidate in candidates)
