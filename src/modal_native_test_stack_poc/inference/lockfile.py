"""Model manifest parsing and offline snapshot resolution."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SAFE_KEY = re.compile(r"^[a-z][a-z0-9_]*$")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

EXPECTED_TASKS = {
    "text_embedding": "sentence_embedding",
    "sentiment": "sentiment_analysis",
    "named_entities": "token_classification",
    "summary": "text2text_generation",
    "image_classification": "image_classification",
    "image_embedding": "clip_embedding",
    "speech_to_text": "automatic_speech_recognition",
}


class ModelManifestError(ValueError):
    """Raised when ``models.lock.json`` violates its public contract."""


class ModelSnapshotMissingError(FileNotFoundError):
    """Raised when a model has not been seeded into the mounted Volume."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    repo_id: str
    revision: str
    task: str

    @property
    def is_commit_pinned(self) -> bool:
        return bool(_COMMIT_SHA.fullmatch(self.revision))

    @property
    def cache_component(self) -> str:
        return f"models--{self.repo_id.replace('/', '--')}"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelSpec:
        missing = {"key", "repo_id", "revision", "task"}.difference(raw)
        if missing:
            fields = ", ".join(sorted(missing))
            raise ModelManifestError(f"model entry is missing required fields: {fields}")

        spec = cls(
            key=str(raw["key"]),
            repo_id=str(raw["repo_id"]),
            revision=str(raw["revision"]),
            task=str(raw["task"]),
        )
        if not _SAFE_KEY.fullmatch(spec.key):
            raise ModelManifestError(f"unsafe model key: {spec.key!r}")
        if not _REPO_ID.fullmatch(spec.repo_id):
            raise ModelManifestError(f"invalid Hugging Face repo ID: {spec.repo_id!r}")
        if not spec.revision:
            raise ModelManifestError(f"model {spec.key!r} has an empty revision")
        if not spec.task:
            raise ModelManifestError(f"model {spec.key!r} has an empty task")
        return spec


@dataclass(frozen=True, slots=True)
class ModelManifest:
    schema_version: int
    models: tuple[ModelSpec, ...]

    @classmethod
    def load(cls, path: str | Path) -> ModelManifest:
        lock_path = Path(path)
        try:
            raw = json.loads(lock_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ModelManifestError(f"model lockfile does not exist: {lock_path}") from exc
        except json.JSONDecodeError as exc:
            raise ModelManifestError(f"invalid JSON in {lock_path}: {exc}") from exc

        if not isinstance(raw, dict):
            raise ModelManifestError("model lockfile root must be an object")
        if raw.get("schema_version") != 1:
            raise ModelManifestError(
                f"unsupported model lock schema: {raw.get('schema_version')!r}"
            )
        raw_models = raw.get("models")
        if not isinstance(raw_models, list) or not raw_models:
            raise ModelManifestError("model lockfile must contain a non-empty models list")

        specs: list[ModelSpec] = []
        for raw_model in raw_models:
            if not isinstance(raw_model, dict):
                raise ModelManifestError("each models entry must be an object")
            specs.append(ModelSpec.from_dict(raw_model))

        keys = [spec.key for spec in specs]
        if len(keys) != len(set(keys)):
            raise ModelManifestError("model keys must be unique")

        by_key = {spec.key: spec for spec in specs}
        missing = set(EXPECTED_TASKS).difference(by_key)
        if missing:
            raise ModelManifestError(
                f"model lockfile is missing required capabilities: {', '.join(sorted(missing))}"
            )
        for key, expected_task in EXPECTED_TASKS.items():
            actual_task = by_key[key].task
            if actual_task != expected_task:
                raise ModelManifestError(
                    f"model {key!r} has task {actual_task!r}; expected {expected_task!r}"
                )

        return cls(schema_version=1, models=tuple(specs))

    @property
    def by_key(self) -> dict[str, ModelSpec]:
        return {spec.key: spec for spec in self.models}

    @property
    def all_commit_pinned(self) -> bool:
        return all(spec.is_commit_pinned for spec in self.models)

    def require_commit_pins(self) -> None:
        floating = [spec.key for spec in self.models if not spec.is_commit_pinned]
        if floating:
            raise ModelManifestError(
                "model revisions must be full 40-character commit SHAs; floating entries: "
                + ", ".join(floating)
            )


class SnapshotResolver:
    """Resolve only pre-fetched local snapshots; never resolve a remote repo ID.

    The public Modal seeder materializes each snapshot at ``/models/{key}``. Two
    standard Hugging Face cache layouts are also understood to make the registry
    convenient in diagnostics and contract tests.
    """

    def __init__(self, models_root: str | Path) -> None:
        self.models_root = Path(models_root)

    def candidates(self, spec: ModelSpec) -> tuple[Path, ...]:
        direct = self.models_root / spec.key
        cache_roots = (self.models_root, self.models_root / "hub", self.models_root / ".cache")
        cached = tuple(
            root / spec.cache_component / "snapshots" / spec.revision for root in cache_roots
        )
        return (direct, *cached)

    def resolve(self, spec: ModelSpec) -> Path:
        for candidate in self.candidates(spec):
            if candidate.is_dir() and (candidate / "config.json").is_file():
                return candidate.resolve()
        checked = "\n  - ".join(str(path) for path in self.candidates(spec))
        raise ModelSnapshotMissingError(
            f"offline snapshot for {spec.key!r} ({spec.repo_id}@{spec.revision}) is missing; "
            f"checked:\n  - {checked}"
        )

    def is_available(self, spec: ModelSpec) -> bool:
        try:
            self.resolve(spec)
        except ModelSnapshotMissingError:
            return False
        return True
