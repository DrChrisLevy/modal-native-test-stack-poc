from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from modal_native_test_stack_poc.application.ports import (
    AssetKind,
    IndexedAsset,
    SearchHit,
    StoredAsset,
)


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        (AssetKind.TEXT, "text"),
        (AssetKind.IMAGE, "image"),
        (AssetKind.AUDIO, "audio"),
    ],
)
def test_asset_kind_has_stable_wire_value(kind: AssetKind, value: str) -> None:
    assert kind.value == value
    assert str(kind) == value


def test_stored_asset_preserves_authoritative_fields() -> None:
    asset_id = uuid4()
    created_at = datetime.now(UTC)
    asset = StoredAsset(
        id=asset_id,
        kind=AssetKind.TEXT,
        title="A document",
        content_text="real source text",
        content_sha256="a" * 64,
        metadata={"characters": 16},
        analysis={"sentiment": {"label": "POSITIVE", "score": 0.9}},
        created_at=created_at,
    )
    assert asset.id == asset_id
    assert asset.kind is AssetKind.TEXT
    assert asset.created_at == created_at


def test_indexed_asset_does_not_require_both_embedding_modalities() -> None:
    asset = IndexedAsset(
        id=uuid4(),
        kind=AssetKind.TEXT,
        title=None,
        content_text="document",
        text_embedding=(0.6, 0.8),
    )
    assert asset.text_embedding == (0.6, 0.8)
    assert asset.image_embedding is None


def test_search_hit_uses_uuid_identity() -> None:
    asset_id = uuid4()
    hit = SearchHit(
        id=asset_id,
        kind=AssetKind.IMAGE,
        title="Generated image",
        content_text="shape, color",
        score=1.5,
    )
    assert isinstance(hit.id, UUID)
    assert hit.id == asset_id


@pytest.mark.parametrize(
    "value",
    [
        StoredAsset(
            id=uuid4(),
            kind=AssetKind.TEXT,
            title=None,
            content_text="x",
            content_sha256="a" * 64,
            metadata={},
            analysis={},
            created_at=datetime.now(UTC),
        ),
        IndexedAsset(
            id=uuid4(),
            kind=AssetKind.TEXT,
            title=None,
            content_text="x",
        ),
        SearchHit(
            id=uuid4(),
            kind=AssetKind.TEXT,
            title=None,
            content_text="x",
            score=1.0,
        ),
    ],
)
def test_application_records_are_immutable(value) -> None:
    with pytest.raises((FrozenInstanceError, AttributeError)):
        value.title = "changed"
