from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio

from modal_native_test_stack_poc.application import (
    ApplicationSettings,
    MultimodalService,
    build_service,
)
from modal_native_test_stack_poc.inference import ModelRegistry

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def live_service(
    models_lock_path: Path,
    models_root: Path,
    registry: ModelRegistry,
    testrun_uid: str,
    worker_id: str,
):
    settings = ApplicationSettings(
        models_lock_path=models_lock_path,
        models_root=models_root,
        require_commit_pins=True,
        opensearch_index=f"modal-ml-e2e-{uuid4().hex}",
        cache_namespace=f"modal-native-test-stack-poc:v1:{testrun_uid}:{worker_id}:e2e",
        cache_ttl_seconds=300,
    )
    service = build_service(settings, registry=registry)
    await service.startup()
    try:
        yield service
    finally:
        client = getattr(service.search_index, "_client", None)
        if client is not None:
            with suppress(Exception):
                await client.indices.delete(index=settings.opensearch_index, ignore=[404])
        await service.close()


@pytest.mark.services
async def test_full_stack_semantic_readiness(live_service: MultimodalService) -> None:
    checks = await live_service.readiness()
    assert checks["postgres"] is True
    assert checks["redis"] is True
    assert checks["opensearch"] is True
    assert checks["model_snapshots"] is True
    assert isinstance(checks["model_revisions_pinned"], bool)


@pytest.mark.e2e
@pytest.mark.model
@pytest.mark.services
@pytest.mark.slow
@pytest.mark.xdist_group(name="text")
async def test_real_text_asset_crosses_models_cache_postgres_and_opensearch(
    live_service: MultimodalService,
) -> None:
    text = (
        "Modal provides fast cloud compute for Python developers in New York. "
        "The platform can run machine-learning tests in isolated environments."
    )
    asset = await live_service.create_text_asset(text, title="Modal test environment")

    assert asset.kind.value == "text"
    assert asset.analysis["embedding"]["dimensions"] == 384
    assert asset.analysis["sentiment"]["label"]
    assert asset.analysis["summary"]["text"]
    assert asset.analysis["entities"]["entities"]

    stored = await live_service.get_asset(asset.id)
    assert stored == asset

    hits = await live_service.search("cloud machine learning", limit=5)
    assert any(hit.id == asset.id for hit in hits)


@pytest.mark.e2e
@pytest.mark.model
@pytest.mark.services
@pytest.mark.xdist_group(name="text")
async def test_real_text_analysis_round_trips_through_redis_cache(
    live_service: MultimodalService,
) -> None:
    text = "Angela Merkel discussed European technology policy in Berlin."
    first = await live_service.analyze_text(text)
    second = await live_service.analyze_text(text)
    assert second == first

    cache_client = getattr(live_service.cache, "_client", None)
    assert cache_client is not None
    keys = [
        key
        async for key in cache_client.scan_iter(f"{live_service.settings.cache_namespace}:text:*")
    ]
    assert keys


@pytest.mark.e2e
@pytest.mark.model
@pytest.mark.services
@pytest.mark.xdist_group(name="image")
async def test_real_image_asset_crosses_models_cache_postgres_and_opensearch(
    live_service: MultimodalService, generated_image_path: Path
) -> None:
    image = generated_image_path.read_bytes()
    asset = await live_service.create_image_asset(
        image,
        title="Generated geometric scene",
        content_type="image/png",
        top_k=4,
    )
    assert asset.kind.value == "image"
    assert asset.metadata["bytes"] == len(image)
    assert len(asset.analysis["classification"]["predictions"]) == 4
    assert asset.analysis["embedding"]["dimensions"] == 512
    assert await live_service.get_asset(asset.id) == asset


@pytest.mark.e2e
@pytest.mark.model
@pytest.mark.services
@pytest.mark.xdist_group(name="image")
async def test_real_image_analysis_round_trips_through_redis_cache(
    live_service: MultimodalService, alternate_image_path: Path
) -> None:
    image = alternate_image_path.read_bytes()
    first = await live_service.analyze_image(image, top_k=3)
    second = await live_service.analyze_image(image, top_k=3)
    assert second == first
    assert len(first.classification.predictions) == 3


@pytest.mark.e2e
@pytest.mark.model
@pytest.mark.services
@pytest.mark.xdist_group(name="audio")
async def test_real_audio_asset_crosses_whisper_cache_postgres_and_opensearch(
    live_service: MultimodalService, generated_silence_wav: Path
) -> None:
    audio = generated_silence_wav.read_bytes()
    asset = await live_service.create_audio_asset(
        audio,
        title="Generated silence",
        content_type="audio/wav",
    )
    assert asset.kind.value == "audio"
    assert asset.metadata["bytes"] == len(audio)
    assert asset.analysis["transcription"]["sample_rate"] == 16_000
    assert asset.analysis["transcription"]["duration_seconds"] == pytest.approx(1.0, abs=0.02)
    assert await live_service.get_asset(asset.id) == asset


@pytest.mark.e2e
@pytest.mark.model
@pytest.mark.services
@pytest.mark.xdist_group(name="audio")
async def test_real_whisper_result_round_trips_through_redis_cache(
    live_service: MultimodalService, generated_tone_wav: Path
) -> None:
    audio = generated_tone_wav.read_bytes()
    first = await live_service.cached_transcription(audio)
    second = await live_service.cached_transcription(audio)
    assert second == first
    assert second.sample_rate == 16_000
