"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from modal_native_test_stack_poc.inference import ModelSnapshotMissingError

from .schemas import (
    AssetResponse,
    EmbeddingResponse,
    EntitiesResponse,
    HealthResponse,
    ImageClassificationResponse,
    ModelStatusResponse,
    SearchHitResponse,
    SearchRequest,
    SearchResponse,
    SentimentResponse,
    SummaryRequest,
    SummaryResponse,
    TextAssetRequest,
    TextRequest,
    TranscriptionResponse,
)
from .service import MultimodalService, build_service
from .settings import ApplicationSettings


def create_app(
    settings: ApplicationSettings | None = None,
    service: MultimodalService | None = None,
) -> FastAPI:
    """Create the API without eagerly loading any model weights."""

    resolved_settings = settings or (
        service.settings if service is not None else ApplicationSettings()
    )
    resolved_service = service or build_service(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await resolved_service.startup()
        app.state.service = resolved_service
        try:
            yield
        finally:
            await resolved_service.close()

    app = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        description="A multimodal API backed by PostgreSQL, Redis, and OpenSearch.",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.service = resolved_service

    @app.exception_handler(ModelSnapshotMissingError)
    async def missing_snapshot_handler(
        _request: Request, exc: ModelSnapshotMissingError
    ) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health/live", response_model=HealthResponse, tags=["health"])
    async def live() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/health/ready", response_model=HealthResponse, tags=["health"])
    async def ready(response: Response) -> HealthResponse:
        checks = await resolved_service.readiness()
        healthy = all(checks.values())
        if not healthy:
            response.status_code = 503
        return HealthResponse(status="ok" if healthy else "not_ready", checks=checks)

    @app.get("/v1/models", response_model=list[ModelStatusResponse], tags=["models"])
    async def models() -> list[ModelStatusResponse]:
        return [ModelStatusResponse.from_status(item) for item in resolved_service.model_status()]

    @app.post("/v1/text/embed", response_model=EmbeddingResponse, tags=["inference"])
    async def embed_text(request: TextRequest) -> EmbeddingResponse:
        return EmbeddingResponse.from_text(await resolved_service.embed_text(request.text))

    @app.post("/v1/text/sentiment", response_model=SentimentResponse, tags=["inference"])
    async def sentiment(request: TextRequest) -> SentimentResponse:
        result = await resolved_service.predict_sentiment(request.text)
        return SentimentResponse.from_result(result)

    @app.post("/v1/text/entities", response_model=EntitiesResponse, tags=["inference"])
    async def entities(request: TextRequest) -> EntitiesResponse:
        result = await resolved_service.extract_entities(request.text)
        return EntitiesResponse.from_result(result)

    @app.post("/v1/text/summarize", response_model=SummaryResponse, tags=["inference"])
    async def summarize(request: SummaryRequest) -> SummaryResponse:
        result = await resolved_service.summarize(
            request.text,
            max_new_tokens=request.max_new_tokens,
            num_beams=request.num_beams,
        )
        return SummaryResponse.from_result(result)

    @app.post("/v1/images/classify", response_model=ImageClassificationResponse, tags=["inference"])
    async def classify_image(
        file: Annotated[UploadFile, File()],
        top_k: Annotated[int, Query(ge=1, le=20)] = 5,
    ) -> ImageClassificationResponse:
        content = await _read_upload(file, resolved_settings.maximum_upload_bytes)
        result = await resolved_service.classify_image(content, top_k=top_k)
        return ImageClassificationResponse.from_result(result)

    @app.post("/v1/images/embed", response_model=EmbeddingResponse, tags=["inference"])
    async def embed_image(file: Annotated[UploadFile, File()]) -> EmbeddingResponse:
        content = await _read_upload(file, resolved_settings.maximum_upload_bytes)
        return EmbeddingResponse.from_image(await resolved_service.embed_image(content))

    @app.post("/v1/audio/transcribe", response_model=TranscriptionResponse, tags=["inference"])
    async def transcribe(file: Annotated[UploadFile, File()]) -> TranscriptionResponse:
        content = await _read_upload(file, resolved_settings.maximum_upload_bytes)
        return TranscriptionResponse.from_result(await resolved_service.transcribe(content))

    @app.post("/v1/assets/text", response_model=AssetResponse, tags=["assets"])
    async def create_text_asset(request: TextAssetRequest) -> AssetResponse:
        asset = await resolved_service.create_text_asset(request.text, title=request.title)
        return AssetResponse.from_asset(asset)

    @app.post("/v1/assets/image", response_model=AssetResponse, tags=["assets"])
    async def create_image_asset(
        file: Annotated[UploadFile, File()],
        title: Annotated[str | None, Form(max_length=200)] = None,
        top_k: Annotated[int, Query(ge=1, le=20)] = 5,
    ) -> AssetResponse:
        content = await _read_upload(file, resolved_settings.maximum_upload_bytes)
        asset = await resolved_service.create_image_asset(
            content,
            title=title,
            content_type=file.content_type,
            top_k=top_k,
        )
        return AssetResponse.from_asset(asset)

    @app.post("/v1/assets/audio", response_model=AssetResponse, tags=["assets"])
    async def create_audio_asset(
        file: Annotated[UploadFile, File()],
        title: Annotated[str | None, Form(max_length=200)] = None,
    ) -> AssetResponse:
        content = await _read_upload(file, resolved_settings.maximum_upload_bytes)
        asset = await resolved_service.create_audio_asset(
            content,
            title=title,
            content_type=file.content_type,
        )
        return AssetResponse.from_asset(asset)

    @app.get("/v1/assets/{asset_id}", response_model=AssetResponse, tags=["assets"])
    async def get_asset(asset_id: UUID) -> AssetResponse:
        asset = await resolved_service.get_asset(asset_id)
        if asset is None:
            raise HTTPException(status_code=404, detail="asset not found")
        return AssetResponse.from_asset(asset)

    @app.post("/v1/search", response_model=SearchResponse, tags=["search"])
    async def search(request: SearchRequest) -> SearchResponse:
        hits = await resolved_service.search(request.text, limit=request.limit)
        return SearchResponse(hits=[SearchHitResponse.from_hit(hit) for hit in hits])

    return app


async def _read_upload(upload: UploadFile, maximum_bytes: int) -> bytes:
    content = await upload.read(maximum_bytes + 1)
    if not content:
        raise ValueError("upload must not be empty")
    if len(content) > maximum_bytes:
        raise ValueError(f"upload exceeds {maximum_bytes} bytes")
    return content
