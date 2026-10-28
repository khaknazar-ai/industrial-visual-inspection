"""FastAPI service with bounded image decoding and a singleton model runtime."""

from __future__ import annotations

import base64
import binascii
import errno
import logging
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from industrial_inspection import __version__
from industrial_inspection.core.config import AppConfig, load_config
from industrial_inspection.core.runtime_resources import runtime_resource_path
from industrial_inspection.inference.evidence import EvidenceStore
from industrial_inspection.inference.pipeline import (
    InspectionPipeline,
    decode_image_bytes,
    image_dimensions,
    quality_policy_from_config,
)
from industrial_inspection.inference.schemas import (
    BatchInspectionError,
    BatchInspectionFailure,
    BatchInspectRequest,
    BatchInspectResponse,
    InspectionResponse,
)
from industrial_inspection.models.registry import model_capabilities, runtime_model_name

_SAFE_CATEGORY = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp", "application/octet-stream"}
_LOGGER = logging.getLogger(__name__)


class _ImageDecodeError(ValueError):
    """Separate client pixel errors from errors raised during model execution."""


@dataclass(slots=True)
class RuntimeState:
    config: AppConfig
    pipeline: InspectionPipeline | None = None
    load_error: str | None = None


async def _bounded_body(request: Request, limit: int) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) < 0:
                raise ValueError("Negative Content-Length")
            if int(content_length) > limit:
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Image payload is too large")
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid Content-Length") from exc
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Image payload is too large")
        chunks.append(chunk)
    return b"".join(chunks)


def create_app(
    pipeline: InspectionPipeline | None = None,
    config: AppConfig | None = None,
) -> FastAPI:
    settings = config or load_config()
    runtime = RuntimeState(settings, pipeline)
    quality_policy = quality_policy_from_config(settings)
    inference_limiter = threading.BoundedSemaphore(settings.runtime.max_concurrent_inference)
    storage = settings.storage_paths()
    report_store = EvidenceStore(storage.artifact_root)
    web_root = runtime_resource_path("web")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Probe imports once off the event loop; /models only reads the cached result.
        await run_in_threadpool(model_capabilities)
        if runtime.pipeline is None:
            model_path = storage.model_root / settings.model.model_id
            if model_path.is_dir() or model_path.is_file():
                try:
                    runtime.pipeline = InspectionPipeline.from_config(settings)
                except (OSError, ValueError, RuntimeError) as exc:
                    runtime.load_error = str(exc)
            else:
                runtime.load_error = (
                    f"Model state not found at {model_path}. Run the train command first."
                )
        yield

    application = FastAPI(
        title="Open-Set Industrial Visual Inspection",
        version=__version__,
        description="Auditable PASS/REVIEW/FAIL inspection service.",
        lifespan=lifespan,
    )
    if runtime.pipeline is not None:
        runtime.pipeline.quality_policy = quality_policy
    application.state.inspection_runtime = runtime
    application.state.inspection_limiter = inference_limiter

    if web_root.is_dir():
        application.mount("/static", StaticFiles(directory=web_root), name="static")

        @application.get("/", include_in_schema=False)
        def web_home() -> FileResponse:
            return FileResponse(web_root / "index.html")

    @application.get("/evidence/{relative_path:path}", include_in_schema=False)
    def evidence(relative_path: str) -> FileResponse:
        """Serve only files below the configured artifact root for the operator UI."""
        if (any(ord(char) < 32 or ord(char) == 127 for char in relative_path)
                or Path(relative_path).is_absolute() or PureWindowsPath(relative_path).drive):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid evidence path")
        root = report_store.root
        try:
            candidate = (root / relative_path).resolve()
            if candidate != root and root not in candidate.parents:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid evidence path")
            if not candidate.is_file():
                raise HTTPException(status.HTTP_404_NOT_FOUND, "Evidence file not found")
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid evidence path") from exc
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, errno.ENAMETOOLONG, errno.ELOOP} and getattr(exc, "winerror", None) not in {123, 206}:
                raise
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid evidence path") from exc
        return FileResponse(candidate)

    def require_pipeline(model_id: str | None = None) -> InspectionPipeline:
        loaded = runtime.pipeline
        if loaded is None:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                runtime.load_error or "Inspection model is not ready",
            )
        if model_id is not None and model_id != loaded.model.model_id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Model is not loaded: {model_id}")
        return loaded

    def validate_category(category: str | None, pipeline: InspectionPipeline) -> str:
        selected = category or pipeline.category
        if not _SAFE_CATEGORY.fullmatch(selected):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Invalid category")
        try:
            return pipeline.validate_category(selected)
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    def acquire_inference_slot() -> None:
        if not inference_limiter.acquire(blocking=False):
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "Inference queue is full",
            )

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @application.get("/ready")
    async def ready() -> dict[str, Any]:
        return {
            "ready": runtime.pipeline is not None,
            "model_id": runtime.pipeline.model.model_id if runtime.pipeline else None,
            "detail": runtime.load_error,
        }

    @application.get("/models")
    async def models() -> dict[str, Any]:
        loaded_id = runtime.pipeline.model.model_id if runtime.pipeline else None
        loaded_name = runtime_model_name(runtime.pipeline.model) if runtime.pipeline else None
        return {
            "loaded_model_id": loaded_id,
            "profile": {
                "category": runtime.pipeline.category if runtime.pipeline else settings.data.category,
                "model_name": loaded_name or settings.model.name,
                "score_space": runtime.pipeline.score_space if runtime.pipeline else "raw_distance",
                "calibration_status": (
                    runtime.pipeline.calibration_status if runtime.pipeline else "unloaded"
                ),
                "calibration_id": runtime.pipeline.calibration_id if runtime.pipeline else None,
                "thresholds": runtime.pipeline.thresholds.as_dict() if runtime.pipeline else None,
                "upload_limits": {
                    "max_image_bytes": settings.runtime.max_image_bytes,
                    "max_image_pixels": settings.runtime.max_image_pixels,
                    "max_batch_bytes": settings.runtime.max_batch_bytes,
                    "max_batch_size": settings.runtime.max_batch_size,
                    "max_batch_pixels": settings.runtime.max_batch_pixels,
                    "max_batch_decoded_bytes": settings.runtime.max_batch_decoded_bytes,
                    "max_concurrent_inference": settings.runtime.max_concurrent_inference,
                    "allowed_content_types": ["image/png", "image/jpeg", "image/webp"],
                },
                "quality_policy": quality_policy.as_dict(),
            },
            "capabilities": [
                capability.as_dict()
                for capability in model_capabilities(
                    loaded_name=loaded_name,
                    loaded_validated=bool(
                        runtime.pipeline and runtime.pipeline.calibration_status == "validated"
                    ),
                )
            ],
        }

    def inspect_payload(pipeline, payload, category):
        started = time.perf_counter()
        try:
            image = decode_image_bytes(payload, max_pixels=settings.runtime.max_image_pixels)
        except ValueError as exc:
            raise _ImageDecodeError(str(exc)) from exc
        decode_ms = (time.perf_counter() - started) * 1000
        response = pipeline.inspect(image, category=category, original_bytes=payload)
        response.timing_ms.decode = decode_ms
        response.timing_ms.total += decode_ms
        return response

    @application.post("/inspect", response_model=InspectionResponse)
    async def inspect(
        request: Request,
        category: str | None = Query(default=None),
        model_id: str | None = Query(default=None),
    ) -> InspectionResponse:
        pipeline = require_pipeline(model_id)
        selected_category = validate_category(category, pipeline)
        content_type = request.headers.get("content-type", "application/octet-stream").split(";", 1)[0]
        if content_type not in _ALLOWED_CONTENT_TYPES:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Unsupported image media type")
        acquire_inference_slot()
        try:
            payload = await _bounded_body(request, settings.runtime.max_image_bytes)
            return await run_in_threadpool(inspect_payload, pipeline, payload, selected_category)
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Could not persist inspection evidence") from exc
        finally:
            inference_limiter.release()

    def inspect_batch_payload(pipeline, payload):
        try:
            body = BatchInspectRequest.model_validate_json(payload)
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Invalid batch JSON or fields") from exc
        category = validate_category(body.category, pipeline)
        if len(body.images_base64) > settings.runtime.max_batch_size:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"Batch size exceeds {settings.runtime.max_batch_size}")
        payloads = []
        compressed_bytes = total_pixels = decoded_bytes = 0
        for encoded in body.images_base64:
            if (len(encoded) * 3) // 4 > settings.runtime.max_image_bytes + 2:
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Image payload is too large")
            try:
                image_payload = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Invalid base64 image") from exc
            compressed_bytes += len(image_payload)
            if len(image_payload) > settings.runtime.max_image_bytes or compressed_bytes > settings.runtime.max_batch_bytes:
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Batch image payload is too large")
            width, height = image_dimensions(image_payload, max_pixels=settings.runtime.max_image_pixels)
            total_pixels += width * height
            decoded_bytes += width * height * 3
            if total_pixels > settings.runtime.max_batch_pixels or decoded_bytes > settings.runtime.max_batch_decoded_bytes:
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Batch decoded image budget exceeded")
            payloads.append(image_payload)
        # All headers and aggregate budgets are checked before the first pixel decode.
        items: list[InspectionResponse | BatchInspectionFailure] = []
        for index, image_payload in enumerate(payloads):
            try:
                items.append(inspect_payload(pipeline, image_payload, category))
                continue
            except _ImageDecodeError as exc:
                error = BatchInspectionError(code="invalid_image", status_code=422, detail=str(exc))
            except Exception:
                # Isolate ordinary item failures so completed report IDs remain reachable.
                # Process shutdown/cancellation (BaseException) must still propagate.
                _LOGGER.exception("Inspection failed for batch item %s", index)
                error = BatchInspectionError(
                    code="inspection_failed", status_code=500, detail="Inspection failed",
                )
            items.append(BatchInspectionFailure(index=index, error=error))
        return BatchInspectResponse(items=items)

    @application.post("/inspect/batch", response_model=BatchInspectResponse)
    async def inspect_batch(request: Request, model_id: str | None = Query(default=None)) -> BatchInspectResponse:
        """Return ordered success/error items after preflight (HTTP 200, even if all fail).

        JSON, base64, image-header and aggregate-budget errors reject the whole batch
        before inference. Successful items keep the single-inspection response schema.
        Retrying executes a new inspection; callers should retry only failed items.
        """
        pipeline = require_pipeline(model_id)
        content_type = request.headers.get("content-type", "").split(";", 1)[0]
        if content_type != "application/json":
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Batch body must be application/json")
        acquire_inference_slot()
        try:
            # Bound actual transport bytes before JSON parsing, including chunked bodies.
            limit = 4 * ((settings.runtime.max_batch_bytes + 2) // 3) + 4096 + 4 * settings.runtime.max_batch_size
            payload = await _bounded_body(request, limit)
            return await run_in_threadpool(inspect_batch_payload, pipeline, payload)
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Could not persist inspection evidence") from exc
        finally:
            inference_limiter.release()

    @application.get("/reports/{request_id}")
    def report(request_id: str) -> dict[str, Any]:
        try:
            normalized = str(uuid.UUID(request_id))
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Invalid request ID") from exc
        data = report_store.read_report(normalized)
        if data is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Report not found")
        return data

    return application


def __getattr__(name: str) -> FastAPI:
    # Preserve `uvicorn ...:app` without reading an unrelated profile during imports.
    if name != "app":
        raise AttributeError(name)
    application = create_app()
    globals()["app"] = application
    return application
