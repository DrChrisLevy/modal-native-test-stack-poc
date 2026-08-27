from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest


def _opensearch_url() -> str:
    url = os.getenv("MULTIMODAL_OPENSEARCH_URL") or os.getenv("OPENSEARCH_URL")
    if not url:
        pytest.skip("OpenSearch URL is not configured")
    return url.rstrip("/")


@pytest.fixture
async def opensearch_client() -> AsyncIterator[Any]:
    httpx = pytest.importorskip("httpx")
    username = os.getenv("OPENSEARCH_USERNAME")
    password = os.getenv("OPENSEARCH_PASSWORD", "")
    auth = (username, password) if username else None
    async with httpx.AsyncClient(
        base_url=_opensearch_url(),
        auth=auth,
        verify=False,
        timeout=30,
    ) as client:
        yield client


@pytest.fixture
async def search_index(opensearch_client) -> AsyncIterator[str]:
    name = f"contract-{uuid.uuid4().hex}"
    response = await opensearch_client.put(
        f"/{name}",
        json={
            "settings": {"index": {"knn": True, "number_of_shards": 1}},
            "mappings": {
                "properties": {
                    "external_id": {"type": "keyword"},
                    "text": {"type": "text"},
                    "embedding": {"type": "knn_vector", "dimension": 3},
                }
            },
        },
    )
    assert response.status_code in {200, 201}, response.text
    try:
        yield name
    finally:
        response = await opensearch_client.delete(f"/{name}")
        assert response.status_code in {200, 404}, response.text


async def _index_document(
    client: Any,
    index: str,
    document_id: str,
    *,
    text: str,
    embedding: list[float],
) -> None:
    response = await client.put(
        f"/{index}/_doc/{document_id}?refresh=true",
        json={"external_id": document_id, "text": text, "embedding": embedding},
    )
    assert response.status_code in {200, 201}, response.text


@pytest.mark.services
async def test_opensearch_cluster_is_healthy(opensearch_client) -> None:
    response = await opensearch_client.get("/_cluster/health")
    assert response.status_code == 200, response.text
    assert response.json()["status"] in {"green", "yellow"}


@pytest.mark.services
async def test_opensearch_reports_a_version(opensearch_client) -> None:
    response = await opensearch_client.get("/")
    assert response.status_code == 200, response.text
    assert response.json()["version"]["number"]


@pytest.mark.services
async def test_opensearch_creates_dimensioned_vector_mapping(
    opensearch_client, search_index: str
) -> None:
    response = await opensearch_client.get(f"/{search_index}/_mapping")
    assert response.status_code == 200, response.text
    embedding = response.json()[search_index]["mappings"]["properties"]["embedding"]
    assert embedding["type"] == "knn_vector"
    assert embedding["dimension"] == 3


@pytest.mark.services
async def test_opensearch_rejects_wrong_vector_dimension(
    opensearch_client, search_index: str
) -> None:
    response = await opensearch_client.put(
        f"/{search_index}/_doc/wrong",
        json={"external_id": "wrong", "text": "bad vector", "embedding": [1.0, 0.0]},
    )
    assert response.status_code == 400


@pytest.mark.services
async def test_opensearch_lexical_search_returns_matching_document(
    opensearch_client, search_index: str
) -> None:
    await _index_document(
        opensearch_client,
        search_index,
        "modal",
        text="Modal starts isolated Python compute in the cloud",
        embedding=[1.0, 0.0, 0.0],
    )
    await _index_document(
        opensearch_client,
        search_index,
        "database",
        text="PostgreSQL provides transactional storage",
        embedding=[0.0, 1.0, 0.0],
    )
    response = await opensearch_client.post(
        f"/{search_index}/_search", json={"query": {"match": {"text": "Modal cloud"}}}
    )
    assert response.status_code == 200, response.text
    hits = response.json()["hits"]["hits"]
    assert hits[0]["_id"] == "modal"


@pytest.mark.services
async def test_opensearch_vector_search_returns_nearest_document(
    opensearch_client, search_index: str
) -> None:
    await _index_document(
        opensearch_client,
        search_index,
        "north",
        text="north",
        embedding=[1.0, 0.0, 0.0],
    )
    await _index_document(
        opensearch_client,
        search_index,
        "east",
        text="east",
        embedding=[0.0, 1.0, 0.0],
    )
    response = await opensearch_client.post(
        f"/{search_index}/_search",
        json={"size": 1, "query": {"knn": {"embedding": {"vector": [0.99, 0.01, 0.0], "k": 1}}}},
    )
    assert response.status_code == 200, response.text
    assert response.json()["hits"]["hits"][0]["_id"] == "north"


@pytest.mark.services
async def test_opensearch_delete_removes_document(opensearch_client, search_index: str) -> None:
    await _index_document(
        opensearch_client,
        search_index,
        "temporary",
        text="temporary",
        embedding=[0.0, 0.0, 1.0],
    )
    response = await opensearch_client.delete(f"/{search_index}/_doc/temporary?refresh=true")
    assert response.status_code == 200, response.text
    response = await opensearch_client.get(f"/{search_index}/_doc/temporary")
    assert response.status_code == 404
