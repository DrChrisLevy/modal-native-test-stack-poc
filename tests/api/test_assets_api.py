from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.e2e
@pytest.mark.model
@pytest.mark.services
@pytest.mark.slow
@pytest.mark.xdist_group(name="text")
def test_text_asset_http_round_trip_uses_real_models_and_services(api_client) -> None:
    response = api_client.post(
        "/v1/assets/text",
        json={
            "title": "  Remote ML  ",
            "text": "Modal runs Python machine-learning workloads in isolated cloud compute.",
        },
    )
    assert response.status_code == 200, response.text
    created = response.json()
    assert created["kind"] == "text"
    assert created["title"] == "Remote ML"
    assert created["analysis"]["embedding"]["dimensions"] == 384

    fetched = api_client.get(f"/v1/assets/{created['id']}")
    assert fetched.status_code == 200, fetched.text
    assert fetched.json() == created


@pytest.mark.e2e
@pytest.mark.model
@pytest.mark.services
@pytest.mark.xdist_group(name="image")
def test_image_asset_http_round_trip_uses_real_models_and_services(
    api_client, generated_image_path: Path
) -> None:
    response = api_client.post(
        "/v1/assets/image?top_k=3",
        data={"title": "Generated scene"},
        files={"file": ("scene.png", generated_image_path.read_bytes(), "image/png")},
    )
    assert response.status_code == 200, response.text
    created = response.json()
    assert created["kind"] == "image"
    assert created["metadata"]["content_type"] == "image/png"
    assert len(created["analysis"]["classification"]["predictions"]) == 3
    assert created["analysis"]["embedding"]["dimensions"] == 512


@pytest.mark.e2e
@pytest.mark.model
@pytest.mark.services
@pytest.mark.xdist_group(name="audio")
def test_audio_asset_http_round_trip_uses_real_whisper_and_services(
    api_client, generated_silence_wav: Path
) -> None:
    response = api_client.post(
        "/v1/assets/audio",
        data={"title": "Generated silence"},
        files={"file": ("silence.wav", generated_silence_wav.read_bytes(), "audio/wav")},
    )
    assert response.status_code == 200, response.text
    created = response.json()
    assert created["kind"] == "audio"
    assert created["metadata"]["content_type"] == "audio/wav"
    assert created["analysis"]["transcription"]["sample_rate"] == 16_000


@pytest.mark.e2e
@pytest.mark.model
@pytest.mark.services
@pytest.mark.slow
@pytest.mark.xdist_group(name="text")
def test_search_http_endpoint_finds_real_model_indexed_asset(api_client) -> None:
    create = api_client.post(
        "/v1/assets/text",
        json={
            "title": "Cloud inference",
            "text": "A hummingbird inference service runs on ephemeral cloud compute.",
        },
    )
    assert create.status_code == 200, create.text
    asset_id = create.json()["id"]

    response = api_client.post("/v1/search", json={"text": "cloud model inference", "limit": 10})
    assert response.status_code == 200, response.text
    assert any(hit["id"] == asset_id for hit in response.json()["hits"])
