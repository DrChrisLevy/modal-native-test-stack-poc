"""FastAPI application and production-shaped service adapters."""

from .adapters import OpenSearchAssetIndex, PostgresAssetRepository, RedisJsonCache
from .api import create_app
from .ports import AssetKind, IndexedAsset, SearchHit, StoredAsset
from .service import MultimodalService, build_service
from .settings import ApplicationSettings, Settings

__all__ = [
    "ApplicationSettings",
    "AssetKind",
    "IndexedAsset",
    "MultimodalService",
    "OpenSearchAssetIndex",
    "PostgresAssetRepository",
    "RedisJsonCache",
    "SearchHit",
    "Settings",
    "StoredAsset",
    "build_service",
    "create_app",
]
