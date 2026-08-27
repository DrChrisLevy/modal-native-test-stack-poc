"""Runtime configuration for the remote-only application environment."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApplicationSettings(BaseSettings):
    """Configuration supplied by the Modal Sandbox and its Sidecars."""

    model_config = SettingsConfigDict(
        env_prefix="MODAL_ML_",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Modal-Native Test Stack POC"
    app_version: str = "0.1.0"
    models_lock_path: Path = Path("/workspace/models.lock.json")
    models_root: Path = Path("/models")
    model_device: str = "cpu"
    require_commit_pins: bool = True

    postgres_url: str = "postgresql://modal_lab:modal_lab@postgres:5432/modal_lab"
    redis_url: str = "redis://redis:6379/0"
    opensearch_url: str = "http://opensearch:9200"
    opensearch_index: str = "modal-ml-assets-v1"
    opensearch_verify_certs: bool = False

    cache_ttl_seconds: int = Field(default=3_600, ge=1, le=86_400)
    text_embedding_dimensions: int = Field(default=384, ge=1)
    image_embedding_dimensions: int = Field(default=512, ge=1)
    maximum_text_characters: int = Field(default=20_000, ge=1)
    maximum_upload_bytes: int = Field(default=25 * 1024 * 1024, ge=1)


# Short alias for callers that prefer the conventional name.
Settings = ApplicationSettings
