from __future__ import annotations

import math
from pathlib import Path

import pytest


@pytest.mark.services
def test_liveness_endpoint_is_independent_of_model_loading(api_client) -> None:
    response = api_client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "checks": {}}


@pytest.mark.services
def test_readiness_reports_every_external_dependency(api_client) -> None:
    response = api_client.get("/health/ready")
    assert response.status_code in {200, 503}
    checks = response.json()["checks"]
    assert checks["postgres"] is True
    assert checks["redis"] is True
    assert checks["opensearch"] is True
    assert checks["model_snapshots"] is True
    assert isinstance(checks["model_revisions_pinned"], bool)


@pytest.mark.services
def test_models_endpoint_lists_all_seven_capabilities(api_client) -> None:
    response = api_client.get("/v1/models")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 7
    assert {model["key"] for model in payload} == {
        "text_embedding",
        "sentiment",
        "named_entities",
        "summary",
        "image_classification",
        "image_embedding",
        "speech_to_text",
    }


@pytest.mark.model
@pytest.mark.services
@pytest.mark.xdist_group(name="text")
def test_text_embedding_endpoint_runs_real_minilm(api_client, positive_text: str) -> None:
    response = api_client.post("/v1/text/embed", json={"text": positive_text})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model_key"] == "text_embedding"
    assert payload["dimensions"] == len(payload["vector"]) == 384
    assert all(math.isfinite(value) for value in payload["vector"])


@pytest.mark.model
@pytest.mark.services
@pytest.mark.xdist_group(name="text")
def test_sentiment_endpoint_runs_real_positive_classifier(api_client, positive_text: str) -> None:
    response = api_client.post("/v1/text/sentiment", json={"text": positive_text})
    assert response.status_code == 200, response.text
    assert "POSITIVE" in response.json()["label"].upper()


@pytest.mark.model
@pytest.mark.services
@pytest.mark.xdist_group(name="text")
def test_sentiment_endpoint_runs_real_negative_classifier(api_client, negative_text: str) -> None:
    response = api_client.post("/v1/text/sentiment", json={"text": negative_text})
    assert response.status_code == 200, response.text
    assert "NEGATIVE" in response.json()["label"].upper()


@pytest.mark.model
@pytest.mark.services
@pytest.mark.xdist_group(name="text")
def test_entities_endpoint_runs_real_ner_model(api_client, entity_text: str) -> None:
    response = api_client.post("/v1/text/entities", json={"text": entity_text})
    assert response.status_code == 200, response.text
    entities = response.json()["entities"]
    assert entities
    assert any("PER" in entity["label"].upper() for entity in entities)
    assert any("LOC" in entity["label"].upper() for entity in entities)


@pytest.mark.model
@pytest.mark.services
@pytest.mark.xdist_group(name="text")
def test_summary_endpoint_runs_real_flan_t5(api_client, summary_text: str) -> None:
    response = api_client.post(
        "/v1/text/summarize",
        json={"text": summary_text, "max_new_tokens": 48, "num_beams": 2},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model_key"] == "summary"
    assert payload["text"].strip()


@pytest.mark.model
@pytest.mark.services
@pytest.mark.xdist_group(name="image")
def test_image_classification_endpoint_runs_real_resnet(
    api_client, generated_image_path: Path
) -> None:
    response = api_client.post(
        "/v1/images/classify?top_k=3",
        files={"file": ("scene.png", generated_image_path.read_bytes(), "image/png")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model_key"] == "image_classification"
    assert len(payload["predictions"]) == len(payload["labels"]) == 3
    assert payload["labels"] == [item["label"] for item in payload["predictions"]]


@pytest.mark.model
@pytest.mark.services
@pytest.mark.xdist_group(name="image")
def test_image_embedding_endpoint_runs_real_clip(api_client, generated_image_path: Path) -> None:
    response = api_client.post(
        "/v1/images/embed",
        files={"file": ("scene.png", generated_image_path.read_bytes(), "image/png")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model_key"] == "image_embedding"
    assert payload["dimensions"] == len(payload["vector"]) == 512


@pytest.mark.model
@pytest.mark.services
@pytest.mark.xdist_group(name="audio")
def test_audio_endpoint_runs_real_whisper(api_client, generated_silence_wav: Path) -> None:
    response = api_client.post(
        "/v1/audio/transcribe",
        files={"file": ("silence.wav", generated_silence_wav.read_bytes(), "audio/wav")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model_key"] == "speech_to_text"
    assert payload["sample_rate"] == 16_000
    assert payload["duration_seconds"] == pytest.approx(1.0, abs=0.02)


@pytest.mark.services
@pytest.mark.parametrize(
    "path",
    [
        "/v1/text/embed",
        "/v1/text/sentiment",
        "/v1/text/entities",
        "/v1/text/summarize",
        "/v1/assets/text",
        "/v1/search",
    ],
)
def test_text_endpoints_reject_empty_strings_without_model_inference(api_client, path: str) -> None:
    response = api_client.post(path, json={"text": ""})
    assert response.status_code == 422


@pytest.mark.services
def test_text_contract_forbids_unknown_fields(api_client) -> None:
    response = api_client.post("/v1/text/embed", json={"text": "valid", "unexpected": "field"})
    assert response.status_code == 422


@pytest.mark.services
@pytest.mark.parametrize("top_k", [0, 21])
def test_image_classifier_validates_top_k_before_inference(
    api_client, generated_image_path: Path, top_k: int
) -> None:
    response = api_client.post(
        f"/v1/images/classify?top_k={top_k}",
        files={"file": ("scene.png", generated_image_path.read_bytes(), "image/png")},
    )
    assert response.status_code == 422


@pytest.mark.services
@pytest.mark.parametrize(
    "path",
    ["/v1/images/classify", "/v1/images/embed", "/v1/audio/transcribe"],
)
def test_upload_endpoints_reject_empty_files(api_client, path: str) -> None:
    response = api_client.post(path, files={"file": ("empty.bin", b"", "application/octet-stream")})
    assert response.status_code == 422


@pytest.mark.services
def test_unknown_asset_returns_not_found(api_client) -> None:
    response = api_client.get("/v1/assets/00000000-0000-0000-0000-000000000001")
    assert response.status_code == 404
    assert response.json()["detail"] == "asset not found"
