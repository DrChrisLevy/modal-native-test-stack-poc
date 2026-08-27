from __future__ import annotations

import json
import math
import os
import struct
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from modal_native_test_stack_poc.inference import ModelRegistry


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def models_lock_path(project_root: Path) -> Path:
    configured = os.getenv("MODAL_ML_MODELS_LOCK_PATH") or os.getenv("MODAL_ML_MODELS_LOCK")
    return Path(configured) if configured else project_root / "models.lock.json"


@pytest.fixture(scope="session")
def model_specs(models_lock_path: Path) -> dict[str, dict[str, str]]:
    payload = json.loads(models_lock_path.read_text())
    return {entry["key"]: entry for entry in payload["models"]}


@pytest.fixture(scope="session")
def models_root() -> Path:
    return Path(os.getenv("MODAL_ML_MODELS_ROOT", "/models"))


@pytest.fixture(scope="session")
def registry(models_lock_path: Path, models_root: Path) -> ModelRegistry:
    from modal_native_test_stack_poc.inference import ModelRegistry

    return ModelRegistry.from_lockfile(
        models_lock_path,
        models_root=models_root,
        require_commit_pins=True,
    )


@pytest.fixture(scope="session")
def positive_text() -> str:
    return "I absolutely loved this thoughtful, beautiful, and wonderfully useful project."


@pytest.fixture(scope="session")
def negative_text() -> str:
    return "This was a terrible, frustrating experience and I strongly disliked it."


@pytest.fixture(scope="session")
def entity_text() -> str:
    return "Barack Obama met Angela Merkel in Berlin before visiting Paris."


@pytest.fixture(scope="session")
def summary_text() -> str:
    return (
        "Modal is a cloud platform for running Python workloads. "
        "It builds reusable container images, mounts persistent volumes, and can start "
        "isolated compute on demand. Developers can invoke remote functions from local "
        "Python while source changes are synchronized to the cloud environment. "
        "This project uses those capabilities to execute real machine-learning models "
        "and integration tests without maintaining a local container stack."
    )


def _make_rgb_fixture(path: Path, *, reverse: bool = False) -> Path:
    from PIL import Image, ImageDraw

    size = 224
    image = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    if reverse:
        draw.rectangle((0, 0, size // 2, size), fill=(15, 40, 200))
        draw.ellipse((70, 45, 180, 155), fill=(235, 180, 25), outline=(20, 20, 20), width=5)
        draw.line((0, size - 1, size - 1, 0), fill=(220, 30, 60), width=9)
    else:
        draw.rectangle((0, 0, size // 2, size), fill=(220, 35, 45))
        draw.ellipse((45, 55, 165, 175), fill=(30, 180, 75), outline=(10, 10, 10), width=5)
        draw.line((0, 0, size - 1, size - 1), fill=(30, 70, 220), width=9)
    image.save(path, format="PNG")
    return path


@pytest.fixture(scope="session")
def generated_image_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _make_rgb_fixture(tmp_path_factory.mktemp("media") / "geometric-scene.png")


@pytest.fixture(scope="session")
def alternate_image_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _make_rgb_fixture(
        tmp_path_factory.mktemp("alternate-media") / "alternate-scene.png",
        reverse=True,
    )


def _write_wave(path: Path, samples: list[float], sample_rate: int = 16_000) -> Path:
    pcm = b"".join(
        struct.pack("<h", max(-32768, min(32767, round(sample * 32767)))) for sample in samples
    )
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return path


@pytest.fixture(scope="session")
def generated_silence_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    sample_rate = 16_000
    return _write_wave(
        tmp_path_factory.mktemp("audio") / "silence.wav",
        [0.0] * sample_rate,
        sample_rate,
    )


@pytest.fixture(scope="session")
def generated_tone_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    sample_rate = 16_000
    duration_seconds = 1.25
    samples = [
        0.15 * math.sin(2 * math.pi * 440 * index / sample_rate)
        for index in range(round(sample_rate * duration_seconds))
    ]
    return _write_wave(
        tmp_path_factory.mktemp("tone-audio") / "tone.wav",
        samples,
        sample_rate,
    )


@pytest.fixture(scope="session")
def api_client(
    models_lock_path: Path,
    models_root: Path,
    registry: ModelRegistry,
    testrun_uid: str,
    worker_id: str,
) -> Iterator[TestClient]:
    from fastapi.testclient import TestClient

    from modal_native_test_stack_poc.application import (
        ApplicationSettings,
        build_service,
        create_app,
    )

    settings = ApplicationSettings(
        models_lock_path=models_lock_path,
        models_root=models_root,
        require_commit_pins=True,
        opensearch_index=f"modal-ml-api-tests-{uuid4().hex}",
        cache_namespace=f"modal-native-test-stack-poc:v1:{testrun_uid}:{worker_id}:api",
    )
    app = create_app(settings=settings, service=build_service(settings, registry=registry))
    with TestClient(app) as client:
        yield client


def value_of(value: Any, *names: str) -> Any:
    """Read a public output field from either a typed object or JSON-compatible mapping."""
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    raise AssertionError(f"None of {names!r} are present on {type(value).__name__}")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Keep every real-model capability in one xdist worker process."""

    for item in items:
        groups = list(item.iter_markers("xdist_group"))
        if len(groups) > 1:
            raise pytest.UsageError(f"{item.nodeid} belongs to multiple xdist groups")
        if item.get_closest_marker("model") is not None and not groups:
            raise pytest.UsageError(f"real-model test lacks an xdist affinity group: {item.nodeid}")
        if not groups:
            item.add_marker(pytest.mark.xdist_group(name="core"))
