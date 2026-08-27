"""Application configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApplicationSettings(BaseSettings):
    """Configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="MULTIMODAL_",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Multimodal Asset API"
    app_version: str = "0.1.0"
    models_lock_path: Path = Path("/workspace/models.lock.json")
    models_root: Path = Path("/models")
    model_device: str = "cpu"
    require_commit_pins: bool = True

    postgres_url: str = "postgresql://postgres:postgres@postgres:5432/multimodal"
    redis_url: str = "redis://redis:6379/0"
    opensearch_url: str = "http://opensearch:9200"
    opensearch_index: str = "multimodal-assets-v1"
    opensearch_verify_certs: bool = False

    cache_namespace: str = "multimodal-assets:v1"
    cache_ttl_seconds: int = Field(default=3_600, ge=1, le=86_400)
    text_embedding_dimensions: int = Field(default=384, ge=1)
    image_embedding_dimensions: int = Field(default=512, ge=1)
    maximum_text_characters: int = Field(default=20_000, ge=1)
    maximum_upload_bytes: int = Field(default=25 * 1024 * 1024, ge=1)


Settings = ApplicationSettings
