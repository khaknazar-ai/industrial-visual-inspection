"""Stable API and evidence response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RegionSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bbox_xyxy: tuple[int, int, int, int]
    area: int
    region_score: float


class EvidenceSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    report_id: str
    original_path: str
    heatmap_path: str | None = None
    mask_path: str | None = None
    overlay_path: str | None = None
    metadata_path: str


class TimingSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decode: float | None = None
    preprocess: float | None = None
    model: float | None = None
    postprocess: float | None = None
    decision: float | None = None
    evidence: float | None = None
    total: float


class InspectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str
    model_id: str
    category: str
    anomaly_score: float | None = Field(allow_inf_nan=False)
    decision: str
    thresholds: dict[str, float]
    regions: list[RegionSchema] = Field(default_factory=list)
    evidence: EvidenceSchema
    timing_ms: TimingSchema
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchInspectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    images_base64: list[str] = Field(min_length=1)
    category: str | None = None


class BatchInspectionError(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: Literal["invalid_image", "inspection_failed"]
    status_code: Literal[422, 500]
    detail: str


class BatchInspectionFailure(BaseModel):
    """An item that did not complete, with a safe error and no success request ID."""

    model_config = ConfigDict(extra="forbid")
    index: int = Field(ge=0, description="Zero-based position in images_base64")
    error: BatchInspectionError


class BatchInspectResponse(BaseModel):
    """One result per input, in input order; successful response fields are unchanged."""

    model_config = ConfigDict(extra="forbid")
    items: list[InspectionResponse | BatchInspectionFailure]
